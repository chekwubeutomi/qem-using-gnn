# Quantum Error Mitigation with Graph Neural Networks (QEM-GNN)

A machine learning framework that uses Graph Neural Networks (GNNs) to mitigate noise in quantum circuit expectation values. Quantum circuits are encoded as graphs capturing both gate structure and hardware noise characteristics, which are then used to train a GNN that learns to correct noisy expectation values toward their ideal (noiseless) counterparts.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Generation](#dataset-generation)
- [GNN Training](#gnn-training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Graph Encoding](#graph-encoding)
- [Output Files](#output-files)
- [Understanding the Results](#understanding-the-results)
- [Tips for Better Performance](#tips-for-better-performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

Real quantum hardware is noisy. Gates have errors, qubits decohere, and measurements are imperfect. Quantum Error Mitigation (QEM) aims to recover near-ideal expectation values from noisy measurements without the heavy overhead of full quantum error correction.

This project implements a GNN-based QEM approach where:

1. Each quantum circuit is transpiled into hardware-native gates and converted to a Directed Acyclic Graph (DAG)
2. The DAG nodes (gates) are assigned feature vectors encoding gate type, gate errors, coherence times (T1, T2), and readout errors
3. The DAG edges (qubit wires) carry hardware calibration features
4. A GNN processes this graph alongside the noisy expectation value to predict the ideal (mitigated) expectation value

---

## Project Structure

```
qem_gnn/
├── generate_quantum_dataset.py      # Dataset generator
├── gnn_quantum_error_mitigation.py  # GNN model and training
├── requirements.txt                 # Python dependencies
├── dataset.json                     # Generated dataset (after Step 1)
├── best_model.pt                    # Saved model checkpoint (after training)
├── best_model.norm.npz              # Feature normalisation statistics
└── best_model.history.json          # Training history (MSE, R², mitigation ratio per epoch)
```

---

## How It Works

```
Random Quantum Circuit
        │
        ▼
 Transpile to IBM native gates
 {RZ, SX, X, CX}
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
 Statevector Simulator                    Noisy AerSimulator
 (ideal expectation values)         (noisy expectation values)
        │                                          │
        └──────────────────┬───────────────────────┘
                           ▼
              Convert circuit DAG → Graph
              • Nodes = gates (22-dim feature vectors)
              • Edges = qubit wires (3-dim: T1, T2, readout error)
                           │
                           ▼
                  Save to dataset.json
                           │
                           ▼
              GNN processes circuit graph
              + noisy expectation value
                           │
                           ▼
              Predicts mitigated expectation value
              (trained to match ideal values)
```

---

## Installation

### Requirements

- Windows, Linux, or macOS
- Python 3.11.x
- 4 GB RAM minimum (8 GB recommended)
- GPU optional (CPU training is supported)

### Step-by-Step Setup

**1. Verify Python version**
```bash
python --version
# Must show Python 3.11.x
```

**2. Create and activate a virtual environment**
```bash
# Create
python -m venv vqem

# Activate (Windows)
vqem\Scripts\activate

# Activate (Linux/macOS)
source vqem/bin/activate
```

**3. Upgrade pip**
```bash
python -m pip install --upgrade pip
```

**4. Install PyTorch (CPU)**
```bash
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> For GPU (CUDA 12.1), replace `cpu` with `cu121` in the URL above.

**5. Install PyTorch Geometric**
```bash
pip install torch_geometric
```

**6. Install remaining dependencies**
```bash
# Scientific stack
pip install numpy scipy scikit-learn

# Qiskit
pip install qiskit==1.1.0
pip install qiskit-aer==0.14.0
```

**7. Verify the installation**
```bash
python -c "
import torch, torch_geometric, qiskit, qiskit_aer, sklearn, numpy as np
print('PyTorch        :', torch.__version__)
print('PyG            :', torch_geometric.__version__)
print('Qiskit         :', qiskit.__version__)
print('Qiskit Aer     :', qiskit_aer.__version__)
print('scikit-learn   :', sklearn.__version__)
print('NumPy          :', np.__version__)
print('All good!')
"
```

> **Important:** NumPy must be version 1.x. If you see NumPy 2.x, run:
> ```bash
> pip install "numpy<2"
> ```

---

## Quick Start

```bash
# Step 1: Generate the dataset
python generate_quantum_dataset.py --num_circuits 200 --output dataset.json

# Step 2: Train the GNN
python gnn_quantum_error_mitigation.py --dataset dataset.json --epochs 100

# Step 3: Run inference with the saved model
python gnn_quantum_error_mitigation.py --dataset dataset.json --mode eval --checkpoint best_model.pt
```

---

## Dataset Generation

`generate_quantum_dataset.py` creates a supervised learning dataset of random quantum circuits and their expectation values.

### What it generates per circuit

| Field | Description |
|---|---|
| `circuit_graph` | DAG graph with node and edge feature vectors |
| `circuit` | OpenQASM 2.0 string of the circuit |
| `observable` | List of Pauli strings measured (e.g. `["ZZIII", "XIIZI"]`) |
| `ideal_exp_value` | Noiseless expectation values from statevector simulation |
| `noisy_exp_values` | Noisy expectation values from AerSimulator with noise model |
| `circuit_depth` | Number of gate layers in the circuit |

### Command line options

```bash
python generate_quantum_dataset.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--num_circuits` | 50 | Number of circuits to generate |
| `--n_qubits` | 5 | Number of qubits per circuit |
| `--min_depth` | 3 | Minimum circuit depth (gate layers) |
| `--max_depth` | 8 | Maximum circuit depth (gate layers) |
| `--n_observables` | 4 | Number of Pauli observables per circuit |
| `--shots` | 4096 | Measurement shots for noisy simulation |
| `--n_noisy_trials` | 1 | Number of noisy simulation repeats per circuit |
| `--seed` | 42 | Global random seed for reproducibility |
| `--output` | `quantum_dataset.json` | Output JSON file path |

### Examples

```bash
# Minimal dataset (fast, for testing)
python generate_quantum_dataset.py --num_circuits 50 --output dataset_small.json

# Recommended dataset for good GNN performance
python generate_quantum_dataset.py --num_circuits 500 --output dataset.json

# Larger circuits with more observables
python generate_quantum_dataset.py --num_circuits 200 --n_qubits 7 --max_depth 12 --n_observables 6 --output dataset_large.json
```

### Hardware noise model

The generator simulates realistic IBM device noise including:

- **Thermal relaxation**: T1 (energy relaxation) and T2 (dephasing) sampled in the range 50–150 µs, typical of Eagle/Falcon processors
- **Depolarizing errors**: Per-gate error rates (single-qubit: 1×10⁻⁴ to 8×10⁻⁴, two-qubit CX: 3×10⁻³ to 1.5×10⁻²)
- **Readout errors**: Measurement bit-flip probabilities (0.5% to 5%)
- **Gate set**: {RZ, SX, X, CX} — the native IBM device gate set

---

## GNN Training

`gnn_quantum_error_mitigation.py` loads the generated dataset, builds the GNN, and trains it to predict ideal expectation values from noisy ones.

### Command line options

```bash
python gnn_quantum_error_mitigation.py [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--dataset` | `quantum_dataset.json` | Path to the JSON dataset |
| `--mode` | `train` | `train` to fit the model, `eval` for inference |
| `--checkpoint` | `best_model.pt` | Path to save/load the model |
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Initial learning rate |
| `--hidden_size` | 128 | Hidden layer size throughout the GNN |
| `--num_heads` | 4 | Number of attention heads in TransformerConv layers |
| `--dropout` | 0.2 | Dropout rate |

### Training examples

```bash
# Standard training run
python gnn_quantum_error_mitigation.py --dataset dataset.json --epochs 100

# Longer training with larger model
python gnn_quantum_error_mitigation.py --dataset dataset.json --epochs 300 --hidden_size 256

# Custom checkpoint path
python gnn_quantum_error_mitigation.py --dataset dataset.json --epochs 100 --checkpoint models/run1.pt
```

### Data split

The dataset is split at the **circuit level** to prevent data leakage across observables:

- **70%** — Training set
- **15%** — Validation set (used for early stopping and LR scheduling)
- **15%** — Test set (final evaluation only)

Each circuit × observable pair becomes one graph sample, so 200 circuits × 4 observables = 800 graph samples total.

---

## Inference

Run a trained model on new or existing data:

```bash
python gnn_quantum_error_mitigation.py \
    --dataset new_data.json \
    --mode eval \
    --checkpoint best_model.pt
```

The normalisation statistics saved in `best_model.norm.npz` are automatically loaded and applied to the new data, ensuring consistent feature scaling.

---

## Model Architecture

```
Input: Circuit DAG graph (nodes = gates, edges = qubit wires)
       + noisy expectation value appended to each node feature

       [N × 23]  node features (22 gate features + 1 noisy value)
       [E × 3]   edge features (T1, T2, readout_error)
            │
            ▼
    Linear Projection
    [N × 23] → [N × 128]
            │
            ▼
    TransformerConv Layer 1
    Multi-head attention (4 heads), edge features used
    [N × 128] → [N × 128]
    BatchNorm1d + ELU + Dropout
            │
            ▼
    SAGPooling Layer 1  (keeps 80% of nodes)
    Self-attention graph pooling — selects most informative gate nodes
            │
            ▼
    TransformerConv Layer 2
    [N' × 128] → [N' × 128]
    BatchNorm1d + ELU + Dropout
            │
            ▼
    Global Mean Pooling  +  Global Max Pooling
    [B × 128]  +  [B × 128]  +  [B × 1 noisy_val]
    = [B × 257]  graph-level representation
            │
            ▼
    Dense Layer 1:  [257] → [128]  ELU + Dropout
    Dense Layer 2:  [128] → [128]  ELU + Dropout
    Output:         [128] → [1]    mitigated expectation value
```

**Key design choices:**

- The noisy expectation value is concatenated to every node feature so the network learns the correction Δ = ideal − noisy, conditioned on the full circuit graph structure
- TransformerConv uses multi-head attention with edge features, enabling each gate node to attend to its neighbouring gates weighted by the qubit wire properties
- SAGPooling selects the most informative gate nodes based on a learned attention score, reducing graph size and focusing computation on the most noise-sensitive gates
- Global mean + max pooling is combined for a richer graph-level representation before the dense prediction layers

---

## Graph Encoding

Each quantum circuit is encoded as a graph with the following structure:

### Node features (22 dimensions per gate)

| Index | Feature | Description |
|---|---|---|
| 0 | `rotation_angle` | RZ rotation parameter θ (0 for other gates) |
| 1 | `is_rz` | Gate type flag |
| 2 | `is_sx` | Gate type flag |
| 3 | `is_x` | Gate type flag |
| 4 | `is_cx` | Gate type flag (2-qubit) |
| 5 | reserved | — |
| 6 | `is_single_qubit` | 1 if single-qubit gate |
| 7 | reserved | — |
| 8 | `is_measure` | 1 if measurement gate |
| 9 | `is_control_qubit` | 1 if CX control qubit |
| 10 | `is_target_qubit` | 1 if CX target qubit |
| 11 | `T1_qubit0` | Relaxation time of first qubit (seconds) |
| 12 | `T1_qubit1` | Relaxation time of second qubit (0 if single-qubit) |
| 13 | reserved | — |
| 14 | `T2_qubit0` | Dephasing time of first qubit (seconds) |
| 15 | `T2_qubit1` | Dephasing time of second qubit (0 if single-qubit) |
| 16 | reserved | — |
| 17 | `gate_duration` | Gate duration in seconds |
| 18 | `gate_duration_q1` | Gate duration for second qubit (or 0) |
| 19 | reserved | — |
| 20 | `gate_error` | Gate error rate |
| 21 | `readout_freq` | Readout error × 10⁴ (proxy for qubit quality) |

### Edge features (3 dimensions per wire)

Edges represent qubit wires between consecutive gate operations:

| Index | Feature | Description |
|---|---|---|
| 0 | `T1` | Relaxation time of the qubit on this wire |
| 1 | `T2` | Dephasing time of the qubit on this wire |
| 2 | `readout_error` | Readout error probability of the qubit |

The circuit DAG edges are made **undirected** so message passing can flow both forward and backward through the circuit, enabling the GNN to capture both causal and anti-causal noise correlations.

---

## Output Files

After training, three files are saved:

| File | Description |
|---|---|
| `best_model.pt` | PyTorch model checkpoint (best validation MSE) |
| `best_model.norm.npz` | Feature normalisation mean and std arrays — required for inference |
| `best_model.history.json` | Per-epoch training history: `train_mse`, `val_mse`, `val_r2`, `val_mitigation_ratio` |

### Loading the history for plotting

```python
import json
import matplotlib.pyplot as plt

with open("best_model.history.json") as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history["train_mse"], label="Train")
plt.plot(history["val_mse"],   label="Val")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Loss")

plt.subplot(1, 3, 2)
plt.plot(history["val_r2"])
plt.xlabel("Epoch"); plt.ylabel("R²"); plt.title("R² Score")

plt.subplot(1, 3, 3)
plt.plot([v * 100 for v in history["val_mitigation_ratio"]])
plt.xlabel("Epoch"); plt.ylabel("Mitigation %"); plt.title("Mitigation Ratio")

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
```

---

## Understanding the Results

The final test evaluation prints five metrics:

| Metric | What it measures |
|---|---|
| **MSE (mitigated)** | Mean squared error of GNN predictions vs ideal values |
| **MSE (noisy)** | Mean squared error of raw noisy values vs ideal values |
| **MAE (mitigated)** | Mean absolute error of GNN predictions |
| **MAE (noisy)** | Mean absolute error of raw noisy values |
| **R²** | Coefficient of determination — how well predictions track ideal values (1.0 = perfect) |
| **Mitigation ratio** | Fraction of noise-induced error corrected by the GNN |

### Reference result (200 circuits, 100 epochs)

```
MSE  (mitigated) : 0.000379
MSE  (noisy)     : 0.000596
MAE  (mitigated) : 0.009031
MAE  (noisy)     : 0.016635
R²               : 0.9831
Mitigation ratio : 36.38%
```

**Interpretation:** An R² of 0.98 indicates the model tracks ideal values very closely. The 36% mitigation ratio means the GNN corrects over one third of all noise-induced error. The remaining error is dominated by irreducible stochastic shot noise which cannot be predicted from circuit structure alone.

---

## Tips for Better Performance

| Goal | Action |
|---|---|
| Higher mitigation ratio | Increase `--num_circuits` (500–1000 recommended) |
| Better convergence | Increase `--epochs` to 200–300 |
| More model capacity | Set `--hidden_size 256` with larger datasets |
| Faster training | Use a GPU machine (set `CUDA_VISIBLE_DEVICES=0`) |
| Reduce overfitting | Increase `--dropout` to 0.3 |
| More noise diversity | Vary `--seed` and generate multiple datasets |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: Numpy is not available` | NumPy 2.x incompatible with PyTorch 2.3 | `pip install "numpy<2"` |
| `TypeError: ReduceLROnPlateau got unexpected argument 'verbose'` | PyTorch 2.4+ removed this arg | Already fixed in current code |
| `TypeError: ASAPooling.forward() got unexpected argument 'edge_attr'` | ASAPooling API changed | Already fixed — uses SAGPooling |
| `ModuleNotFoundError: No module named 'torch'` when installing PyG wheels | Building from source without torch | Use `pip install torch_geometric` without sparse wheel URLs |
| `Could not load library: torch_scatter/_version_cpu.pyd` | Sparse kernels built for wrong PyTorch version | Uninstall sparse kernels; PyG works without them on CPU |
| `qiskit-aer` install fails | Missing C++ build tools on Windows | Install from `aka.ms/buildtools` or use `pip install qiskit-aer --prefer-binary` |
| Dataset loads 0 samples | JSON file empty or wrong format | Re-run `generate_quantum_dataset.py` and check output file size |
