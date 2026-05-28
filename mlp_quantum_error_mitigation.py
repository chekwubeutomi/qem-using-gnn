"""
MLP-Based Quantum Error Mitigation
=====================================
This script is a direct MLP counterpart to gnn_quantum_error_mitigation.py.
Both models consume the SAME dataset.json file, use the SAME train/val/test
split, and report the SAME metrics so results are directly comparable.

What the GNN does                 What this MLP does
─────────────────────────────────────────────────────
Input  : circuit DAG graph        Input  : flat feature vector extracted
         + noisy exp value                  from the same circuit DAG
Output : 1 mitigated exp value    Output : 1 mitigated exp value
         per observable                     per observable
Metrics: MSE, MAE, R²,            Metrics: MSE, MAE, R²,
         mitigation ratio                   mitigation ratio  ← identical

Feature encoding (mirrors what the GNN encodes in its node/edge features):
  - 8  device-level noise stats   (mean gate errors, readout error, T1, T2)
  - 6  gate counts                (rz, sx, x, cx, measure, barrier)
  - 40 rotation angle histogram   (binned RZ angles, bin_size = 0.1π)
  - 1  noisy expectation value     (the observable being mitigated)
  ────────────────────────────────
  55  total input features

This matches the information the GNN has access to:
  - Node features carry per-gate type, T1, T2, gate error, gate duration
  - Edge features carry T1, T2, readout error per wire
  - The noisy expectation value is appended to every node

Usage
-----
# 1. Generate data (same command as GNN):
    python generate_quantum_dataset.py --num_circuits 5000 --output dataset.json

# 2. Train MLP:
    python mlp_quantum_error_mitigation.py --dataset dataset.json --epochs 100

# 3. Evaluate with saved checkpoint:
    python mlp_quantum_error_mitigation.py --dataset dataset.json --mode eval --checkpoint best_mlp.pt

# 4. Compare with GNN:
    Both scripts print identical metric blocks — paste them side by side.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# ---------------------------------------------------------------------------
# 1.  Feature extraction from dataset.json
#     Mirrors exactly the information the GNN receives via node/edge features
# ---------------------------------------------------------------------------

def count_gates_by_rotation_angle(circuit_graph: dict, bin_size: float = 0.1 * np.pi) -> list:
    """
    Build a histogram of RZ rotation angles from the DAG node features.
    The GNN encodes rotation_angle at node feature index 0 for RZ gates
    (is_rz flag at index 1). We reproduce that information as a histogram.
    """
    bin_edges  = np.arange(-2 * np.pi, 2 * np.pi + bin_size, bin_size)
    num_bins   = len(bin_edges) - 1           # 40 bins
    counts     = np.zeros(num_bins, dtype=np.float32)

    op_feats = circuit_graph["nodes"]["DAGOpNode"]
    for feat in op_feats:
        is_rz = feat[1] if len(feat) > 1 else 0.0
        angle = feat[0] if len(feat) > 0 else 0.0
        if is_rz > 0.5:
            bin_idx = np.searchsorted(bin_edges, angle, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
            counts[bin_idx] += 1

    return counts.tolist()


def extract_device_noise_stats(circuit_graph: dict) -> list:
    """
    Extract 8 device-level noise statistics from node features.
    These aggregate the same information the GNN gets per-node:
      - mean T1  (node feat indices 11, 12)
      - mean T2  (node feat indices 14, 15)
      - mean gate error  (node feat index 20)
      - mean readout proxy (node feat index 21)
      - mean gate duration  (node feat index 17)
      - fraction of CX gates  (node feat index 4)

    Scaled × 100 to be in the same order of magnitude as expectation values.
    """
    op_feats = circuit_graph["nodes"]["DAGOpNode"]
    if not op_feats:
        return [0.0] * 8

    t1_vals, t2_vals, err_vals, ro_vals, dur_vals, cx_flags = [], [], [], [], [], []

    for feat in op_feats:
        if len(feat) < 22:
            feat = feat + [0.0] * (22 - len(feat))
        t1_vals  .append((feat[11] + feat[12]) / 2 if feat[12] > 0 else feat[11])
        t2_vals  .append((feat[14] + feat[15]) / 2 if feat[15] > 0 else feat[14])
        err_vals .append(feat[20])
        ro_vals  .append(feat[21])
        dur_vals .append(feat[17])
        cx_flags .append(feat[4])

    n = len(op_feats)
    stats = [
        np.mean(t1_vals)   * 100,   # 0  mean T1
        np.mean(t2_vals)   * 100,   # 1  mean T2
        np.mean(err_vals)  * 100,   # 2  mean gate error
        np.mean(ro_vals)   * 100,   # 3  mean readout proxy
        np.mean(dur_vals)  * 100,   # 4  mean gate duration
        np.sum(cx_flags)   / n,     # 5  fraction CX gates
        np.std(err_vals)   * 100,   # 6  std of gate errors (noise heterogeneity)
        np.std(t1_vals)    * 100,   # 7  std of T1 (qubit quality spread)
    ]
    return [float(s) for s in stats]


def extract_gate_counts(circuit_graph: dict) -> list:
    """
    Count gate types from node features.
    Gate type flags in node features:
      index 1 = is_rz, 2 = is_sx, 3 = is_x, 4 = is_cx, 8 = is_measure
    Returns [n_rz, n_sx, n_x, n_cx, n_measure, n_total] × 0.01
    """
    op_feats = circuit_graph["nodes"]["DAGOpNode"]
    counts = [0] * 6  # rz, sx, x, cx, measure, total

    for feat in op_feats:
        if len(feat) < 9:
            feat = feat + [0.0] * (9 - len(feat))
        counts[0] += int(feat[1] > 0.5)   # is_rz
        counts[1] += int(feat[2] > 0.5)   # is_sx
        counts[2] += int(feat[3] > 0.5)   # is_x
        counts[3] += int(feat[4] > 0.5)   # is_cx
        counts[4] += int(feat[8] > 0.5)   # is_measure
        counts[5] += 1                     # total

    return [c * 0.01 for c in counts]     # scale to order of expectation values


def encode_sample(sample: dict, obs_idx: int) -> tuple[np.ndarray, float, float] | None:
    """
    Encode one (circuit, observable) pair into a flat feature vector.

    Feature vector layout (55 elements):
      [0:8]   device noise statistics        (8 features)
      [8:14]  gate type counts               (6 features)
      [14:54] rotation angle histogram       (40 features)
      [54]    noisy expectation value        (1 feature)

    Returns (feature_vector, ideal_val, noisy_val) or None if degenerate.
    """
    graph    = sample["circuit_graph"]
    op_feats = graph["nodes"]["DAGOpNode"]

    if not op_feats:
        return None

    # Noisy expectation value for this observable
    noisy_trials = sample["noisy_exp_values"]   # [n_trials, n_obs]
    noisy_val    = float(np.mean([t[obs_idx] for t in noisy_trials]))

    # Ideal expectation value (label)
    ideal_val = float(sample["ideal_exp_value"][obs_idx])

    # Build feature vector
    device_stats  = extract_device_noise_stats(graph)    # 8
    gate_counts   = extract_gate_counts(graph)           # 6
    angle_hist    = count_gates_by_rotation_angle(graph) # 40
    # Noisy exp val appended last — mirrors GNN appending it to each node
    feat_vec = device_stats + gate_counts + angle_hist + [noisy_val]  # 55

    return np.array(feat_vec, dtype=np.float32), ideal_val, noisy_val


# ---------------------------------------------------------------------------
# 2.  Dataset class  (mirrors QuantumCircuitDataset in GNN script)
# ---------------------------------------------------------------------------

class MLPQuantumDataset(Dataset):
    """
    Flat-feature dataset built from the same dataset.json as the GNN.
    Each circuit × observable pair becomes one sample, exactly as in the GNN.
    """

    def __init__(self, samples: list[dict]):
        self.X         = []   # feature vectors
        self.y         = []   # ideal expectation values  (labels)
        self.noisy_vals= []   # noisy expectation values  (for mitigation ratio)

        for sample in samples:
            n_obs = len(sample["ideal_exp_value"])
            for obs_idx in range(n_obs):
                result = encode_sample(sample, obs_idx)
                if result is None:
                    continue
                feat_vec, ideal_val, noisy_val = result
                self.X         .append(feat_vec)
                self.y         .append(ideal_val)
                self.noisy_vals.append(noisy_val)

        self.X          = torch.tensor(np.array(self.X),          dtype=torch.float32)
        self.y          = torch.tensor(self.y,                    dtype=torch.float32).unsqueeze(1)
        self.noisy_vals = torch.tensor(self.noisy_vals,           dtype=torch.float32).unsqueeze(1)

        print(f"  Loaded {len(self.X)} samples from {len(samples)} circuits.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.noisy_vals[idx]


# ---------------------------------------------------------------------------
# 3.  Feature normalisation  (mirrors GNN's feature_normalize / normalize_dataset)
# ---------------------------------------------------------------------------

def feature_normalize(dataset: MLPQuantumDataset) -> tuple[np.ndarray, np.ndarray]:
    X_np = dataset.X.numpy()
    mean = X_np.mean(axis=0)
    std  = X_np.std(axis=0) + 1e-8
    return mean, std


def normalize_dataset(dataset: MLPQuantumDataset,
                      mean: np.ndarray,
                      std:  np.ndarray) -> None:
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t  = torch.tensor(std,  dtype=torch.float32)
    dataset.X = (dataset.X - mean_t) / std_t


# ---------------------------------------------------------------------------
# 4.  MLP Model
# ---------------------------------------------------------------------------

class QuantumErrorMitigationMLP(nn.Module):
    """
    Multi-layer perceptron for quantum error mitigation.

    Architecture:
        Linear(55 → hidden) → ReLU → Dropout
        Linear(hidden → hidden) → ReLU → Dropout
        Linear(hidden → 1)

    This is the direct MLP analogue of the GNN:
      - Both take the same information as input
      - Both output 1 mitigated expectation value per observable
      - The GNN processes this via graph message passing;
        the MLP processes it as a flat vector
    """

    def __init__(
        self,
        input_dim   : int   = 55,
        hidden_size : int   = 128,
        dropout     : float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,   hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),           # single output per observable
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)                        # [B, 1]


# ---------------------------------------------------------------------------
# 5.  Training utilities  (identical signatures to GNN script)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
) -> float:
    model.train()
    total_loss, total_samples = 0.0, 0

    for X_batch, y_batch, _ in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = F.mse_loss(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * len(X_batch)
        total_samples += len(X_batch)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model  : nn.Module,
    loader : DataLoader,
    device : torch.device,
) -> dict:
    """
    Compute all metrics in the same format as the GNN evaluate() function
    so the output blocks are directly comparable.
    """
    model.eval()
    preds, targets, noisy_vals = [], [], []

    for X_batch, y_batch, noisy_batch in loader:
        X_batch = X_batch.to(device)
        pred    = model(X_batch).cpu()
        preds     .append(pred)
        targets   .append(y_batch)
        noisy_vals.append(noisy_batch)

    preds      = torch.cat(preds)     .numpy().flatten()
    targets    = torch.cat(targets)   .numpy().flatten()
    noisy_vals = torch.cat(noisy_vals).numpy().flatten()

    mse_mitigated = float(np.mean((preds      - targets) ** 2))
    mse_noisy     = float(np.mean((noisy_vals - targets) ** 2))
    mae_mitigated = float(np.mean(np.abs(preds      - targets)))
    mae_noisy     = float(np.mean(np.abs(noisy_vals - targets)))

    ss_res = np.sum((preds - targets) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2) + 1e-12
    r2     = float(1.0 - ss_res / ss_tot)

    mitigation_ratio = float(1.0 - mse_mitigated / (mse_noisy + 1e-12))

    return {
        "mse"             : mse_mitigated,
        "mae"             : mae_mitigated,
        "mse_noisy"       : mse_noisy,
        "mae_noisy"       : mae_noisy,
        "r2"              : r2,
        "mitigation_ratio": mitigation_ratio,
        # Store arrays for plotting
        "_preds"          : preds,
        "_targets"        : targets,
        "_noisy"          : noisy_vals,
    }


# ---------------------------------------------------------------------------
# 6.  Visualisation  (mirrors GNN plots for direct visual comparison)
# ---------------------------------------------------------------------------

def plot_predictions(metrics: dict, split_name: str, prefix: str = "mlp"):
    try:
        import matplotlib.pyplot as plt
        preds   = metrics["_preds"]
        targets = metrics["_targets"]
        noisys  = metrics["_noisy"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"MLP — Predicted vs Actual ({split_name})", fontsize=13)

        # Mitigated
        ax = axes[0]
        ax.scatter(targets, preds, alpha=0.4, s=12, color='steelblue', label='Mitigated')
        lims = [targets.min(), targets.max()]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted (MLP)")
        ax.set_title("Mitigated"); ax.legend(fontsize=8)

        # Unmitigated
        ax = axes[1]
        ax.scatter(targets, noisys, alpha=0.4, s=12, color='tomato', label='Unmitigated')
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel("Actual"); ax.set_ylabel("Noisy")
        ax.set_title("Unmitigated"); ax.legend(fontsize=8)

        plt.tight_layout()
        fname = f"{prefix}_predictions_{split_name.lower()}.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved → {fname}")
    except ImportError:
        pass


def plot_loss_curve(train_losses: list, val_losses: list, prefix: str = "mlp"):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses,   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.title("MLP Training Curve")
        plt.legend(); plt.tight_layout()
        fname = f"{prefix}_loss_curve.png"
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"  Saved → {fname}")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 7.  Main training pipeline  (mirrors run_training() in GNN script)
# ---------------------------------------------------------------------------

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  MLP Quantum Error Mitigation — Training")
    print(f"{'='*60}")
    print(f"  Device  : {device}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  LR      : {args.lr}")
    print(f"{'='*60}\n")

    # ── Load dataset.json  (same file the GNN uses) ───────────────────────
    print("Loading dataset ...")
    with open(args.dataset) as f:
        raw = json.load(f)

    # ── Same split as GNN: 70 / 15 / 15 at circuit level ─────────────────
    train_raw, test_raw = train_test_split(raw, test_size=0.15, random_state=42)
    train_raw, val_raw  = train_test_split(train_raw, test_size=0.15, random_state=42)
    print(f"  Circuits — train: {len(train_raw)}  val: {len(val_raw)}  test: {len(test_raw)}")

    train_ds = MLPQuantumDataset(train_raw)
    val_ds   = MLPQuantumDataset(val_raw)
    test_ds  = MLPQuantumDataset(test_raw)

    # ── Feature normalisation  (same as GNN) ─────────────────────────────
    print("Normalising features ...")
    mean, std = feature_normalize(train_ds)
    normalize_dataset(train_ds, mean, std)
    normalize_dataset(val_ds,   mean, std)
    normalize_dataset(test_ds,  mean, std)

    norm_path = Path(args.checkpoint).with_suffix(".norm.npz")
    np.savez(norm_path, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Build MLP ─────────────────────────────────────────────────────────
    input_dim = train_ds.X.shape[1]   # should be 55
    model = QuantumErrorMitigationMLP(
        input_dim   = input_dim,
        hidden_size = args.hidden_size,
        dropout     = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model architecture:\n{model}")
    print(f"\n  Input features : {input_dim}")
    print(f"  Parameters     : {n_params:,}\n")

    # ── Optimiser & scheduler  (identical to GNN) ─────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    best_val_mse = math.inf
    history = {"train_mse": [], "val_mse": [], "val_r2": [], "val_mitigation_ratio": []}

    # ── Training loop ─────────────────────────────────────────────────────
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        t0        = time.time()
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        val_m     = evaluate(model, val_loader, device)

        scheduler.step(val_m["mse"])

        history["train_mse"]            .append(train_mse)
        history["val_mse"]              .append(val_m["mse"])
        history["val_r2"]               .append(val_m["r2"])
        history["val_mitigation_ratio"] .append(val_m["mitigation_ratio"])
        train_losses.append(train_mse)
        val_losses  .append(val_m["mse"])

        if val_m["mse"] < best_val_mse:
            best_val_mse = val_m["mse"]
            torch.save({"model_state": model.state_dict(), "args": vars(args)},
                       args.checkpoint)
            flag = " ← best"
        else:
            flag = ""

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>4}/{args.epochs}"
            f"  train_MSE={train_mse:.5f}"
            f"  val_MSE={val_m['mse']:.5f}"
            f"  val_R²={val_m['r2']:.4f}"
            f"  mitigation={val_m['mitigation_ratio']*100:.1f}%"
            f"  ({elapsed:.1f}s){flag}"
        )

    # ── Final test evaluation  (same print block as GNN) ──────────────────
    print(f"\n{'='*60}")
    print("  Final Test Evaluation (best checkpoint)")
    print(f"{'='*60}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader, device)

    print(f"  MSE  (mitigated) : {test_m['mse']:.6f}")
    print(f"  MSE  (noisy)     : {test_m['mse_noisy']:.6f}")
    print(f"  MAE  (mitigated) : {test_m['mae']:.6f}")
    print(f"  MAE  (noisy)     : {test_m['mae_noisy']:.6f}")
    print(f"  R²               : {test_m['r2']:.4f}")
    print(f"  Mitigation ratio : {test_m['mitigation_ratio']*100:.2f}%")

    # ── Save history & plots ──────────────────────────────────────────────
    history_path = Path(args.checkpoint).with_suffix(".history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_loss_curve(train_losses, val_losses)
    plot_predictions(test_m, "Test")

    print(f"\n  History saved  → {history_path}")
    print(f"  Best model     → {args.checkpoint}")
    print(f"  Norm stats     → {norm_path}")


def run_inference(args):
    """Load saved MLP and evaluate — mirrors GNN run_inference()."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("  MLP Quantum Error Mitigation — Inference")
    print(f"{'='*60}")

    with open(args.dataset) as f:
        raw = json.load(f)

    ds = MLPQuantumDataset(raw)

    norm_path = Path(args.checkpoint).with_suffix(".norm.npz")
    if norm_path.exists():
        npz  = np.load(norm_path)
        normalize_dataset(ds, npz["mean"], npz["std"])
        print(f"  Applied normalisation from {norm_path}")
    else:
        print("  Warning: no normalisation file found — features not normalised")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt  = torch.load(args.checkpoint, map_location=device)
    saved = ckpt.get("args", {})

    input_dim = ds.X.shape[1]
    model = QuantumErrorMitigationMLP(
        input_dim   = input_dim,
        hidden_size = saved.get("hidden_size", 128),
        dropout     = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    m = evaluate(model, loader, device)
    print(f"\n  MSE  (mitigated) : {m['mse']:.6f}")
    print(f"  MSE  (noisy)     : {m['mse_noisy']:.6f}")
    print(f"  MAE  (mitigated) : {m['mae']:.6f}")
    print(f"  MAE  (noisy)     : {m['mae_noisy']:.6f}")
    print(f"  R²               : {m['r2']:.4f}")
    print(f"  Mitigation ratio : {m['mitigation_ratio']*100:.2f}%")


# ---------------------------------------------------------------------------
# 8.  CLI  (identical arguments to GNN script)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="MLP Quantum Error Mitigation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",     type=str,   default="quantum_dataset.json",
                   help="Path to JSON dataset produced by generate_quantum_dataset.py")
    p.add_argument("--mode",        choices=["train", "eval"], default="train",
                   help="'train' to fit; 'eval' to run inference with saved checkpoint")
    p.add_argument("--checkpoint",  type=str,   default="best_mlp.pt",
                   help="Path to save (train) or load (eval) the model checkpoint")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden_size", type=int,   default=128)
    p.add_argument("--dropout",     type=float, default=0.2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    else:
        run_inference(args)
