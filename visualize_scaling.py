"""
Data Scaling Study Visualization
==================================
Plots mitigation ratio vs number of training circuits
for GNN and MLP models on 4-qubit, depth 4-10 circuits.

Usage:
    python visualize_scaling.py

All plots saved to ./plots/
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import warnings

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

matplotlib.rcParams.update({
    "font.family"       : "DejaVu Serif",
    "mathtext.fontset"  : "dejavuserif",
    "font.size"         : 12,
    "axes.titlesize"    : 14,
    "axes.labelsize"    : 13,
    "xtick.labelsize"   : 11,
    "ytick.labelsize"   : 11,
    "legend.fontsize"   : 11,
    "legend.framealpha" : 0.92,
    "figure.dpi"        : 140,
    "savefig.dpi"       : 220,
    "savefig.bbox"      : "tight",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.30,
    "grid.linewidth"    : 0.7,
    "grid.linestyle"    : "--",
})

# ── Colours ───────────────────────────────────────────────────────────────────
GNN_COL   = "#1A6FAE"   # deep blue
MLP_COL   = "#C00000"   # deep red
GNN_LIGHT = "#AED6F1"   # light blue fill
MLP_LIGHT = "#F5B7B1"   # light red fill
GREY      = "#777777"
DARK      = "#1A1A1A"
BG        = "#FAFAFA"

# ── Experimental data ─────────────────────────────────────────────────────────
# 4-qubit, depth 4-10, best run per dataset size (200-epoch where available)
data = {
    "n_circuits": [1000,   2000,   3000,   4000,   5000  ],

    # Mitigation ratios (%)
    "gnn_mr":     [81.57,  84.50,  82.88,  83.70,  83.29 ],
    "mlp_mr":     [76.48,  79.17,  75.19,  80.08,  78.52 ],

    # R² scores
    "gnn_r2":     [0.9949, 0.9968, 0.9966, 0.9959, 0.9968],
    "mlp_r2":     [0.9941, 0.9957, 0.9950, 0.9949, 0.9959],

    # MSE mitigated
    "gnn_mse":    [0.000295, 0.000318, 0.000372, 0.000247, 0.000305],
    "mlp_mse":    [0.000337, 0.000427, 0.000539, 0.000302, 0.000392],

    # MAE mitigated
    "gnn_mae":    [0.008104, 0.009135, 0.010473, 0.008801, 0.008402],
    "mlp_mae":    [0.005591, 0.007623, 0.008230, 0.005484, 0.006729],

    # MSE noisy baseline (varies per dataset due to different noise seeds)
    "noisy_mse":  [0.001516, 0.002050, 0.002170, 0.001516, 0.001826],

    # Improvement factors (GNN only — from output)
    "gnn_impr":   [5.42,   6.45,   5.84,   6.14,   5.99  ],
}

x     = np.array(data["n_circuits"])
x_pos = np.arange(len(x))  # for bar charts


def save(fig, name):
    path = f"plots/{name}.png"
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved  →  {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Mitigation Ratio vs Number of Training Circuits (PRIMARY PLOT)
# ══════════════════════════════════════════════════════════════════════════════
def plot_mitigation_ratio():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    gnn_mr = np.array(data["gnn_mr"])
    mlp_mr = np.array(data["mlp_mr"])

    # ── Main lines ────────────────────────────────────────────────────────
    ax.plot(x, gnn_mr, "o-",
            color=GNN_COL, lw=2.8, ms=10, zorder=5,
            label="GNN", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(x, mlp_mr, "s--",
            color=MLP_COL, lw=2.8, ms=10, zorder=4,
            label="MLP", markeredgecolor="white", markeredgewidth=1.5)

    # ── Shaded gap between models ─────────────────────────────────────────
    ax.fill_between(x, gnn_mr, mlp_mr,
                    where=[g > m for g, m in zip(gnn_mr, mlp_mr)],
                    alpha=0.12, color=GNN_COL, label="GNN advantage")

    # ── Data labels ───────────────────────────────────────────────────────
    for xi, g, m in zip(x, gnn_mr, mlp_mr):
        ax.text(xi, g + 0.6, f"{g:.2f}%",
                ha="center", va="bottom", fontsize=9.5,
                color=GNN_COL, fontweight="bold")
        ax.text(xi, m - 1.2, f"{m:.2f}%",
                ha="center", va="top", fontsize=9.5,
                color=MLP_COL, fontweight="bold")

    # ── GNN-MLP gap annotations ───────────────────────────────────────────
    for xi, g, m in zip(x, gnn_mr, mlp_mr):
        gap = g - m
        mid = (g + m) / 2
        ax.annotate(f"+{gap:.2f} pp",
                    xy=(xi, mid),
                    fontsize=8, color=GREY, ha="left",
                    xytext=(12, 0), textcoords="offset points")

    # ── Reference lines ───────────────────────────────────────────────────
    for ref, lbl in [(75, "75%"), (80, "80%"), (85, "85%")]:
        ax.axhline(ref, color=GREY, lw=0.9, ls=":", alpha=0.55)
        ax.text(x[0] - 120, ref + 0.2, lbl, fontsize=8.5, color=GREY)

    # ── Trend lines (polynomial fit) ──────────────────────────────────────
    x_smooth = np.linspace(x[0], x[-1], 300)
    for vals, col in [(gnn_mr, GNN_COL), (mlp_mr, MLP_COL)]:
        z = np.polyfit(x, vals, 2)
        p = np.poly1d(z)
        ax.plot(x_smooth, p(x_smooth),
                color=col, lw=1.0, alpha=0.30, ls="-")

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xlabel("Number of Training Circuits", fontsize=13, labelpad=8)
    ax.set_ylabel("Mitigation Ratio (%)", fontsize=13, labelpad=8)
    ax.set_title(
        "Data Scaling Study: Mitigation Ratio vs Training Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=14, fontweight="bold", color=DARK, pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in x])
    ax.set_ylim(68, 92)
    ax.set_xlim(x[0] - 250, x[-1] + 250)
    ax.legend(loc="lower right", fontsize=11)

    # ── Secondary annotation box ──────────────────────────────────────────
    avg_gap = np.mean(np.array(gnn_mr) - np.array(mlp_mr))
    textstr = (
        f"GNN avg: {np.mean(gnn_mr):.2f}%\n"
        f"MLP avg: {np.mean(mlp_mr):.2f}%\n"
        f"Avg gap: +{avg_gap:.2f} pp (GNN)"
    )
    ax.text(0.02, 0.97, textstr,
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GREY, alpha=0.85))

    plt.tight_layout()
    save(fig, "01_mitigation_ratio_vs_dataset_size")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — R² vs Number of Training Circuits
# ══════════════════════════════════════════════════════════════════════════════
def plot_r2():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    gnn_r2 = np.array(data["gnn_r2"])
    mlp_r2 = np.array(data["mlp_r2"])

    ax.plot(x, gnn_r2, "o-",
            color=GNN_COL, lw=2.8, ms=10, zorder=5,
            label="GNN", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(x, mlp_r2, "s--",
            color=MLP_COL, lw=2.8, ms=10, zorder=4,
            label="MLP", markeredgecolor="white", markeredgewidth=1.5)

    ax.fill_between(x, gnn_r2, mlp_r2,
                    where=[g > m for g, m in zip(gnn_r2, mlp_r2)],
                    alpha=0.10, color=GNN_COL)

    for xi, g, m in zip(x, gnn_r2, mlp_r2):
        ax.text(xi, g + 0.0003, f"{g:.4f}",
                ha="center", va="bottom", fontsize=9,
                color=GNN_COL, fontweight="bold")
        ax.text(xi, m - 0.0003, f"{m:.4f}",
                ha="center", va="top", fontsize=9,
                color=MLP_COL, fontweight="bold")

    for ref in [0.994, 0.996, 0.998]:
        ax.axhline(ref, color=GREY, lw=0.9, ls=":", alpha=0.5)
        ax.text(x[0] - 120, ref + 0.0001, f"{ref}", fontsize=8, color=GREY)

    ax.set_xlabel("Number of Training Circuits", fontsize=13, labelpad=8)
    ax.set_ylabel("R² Score", fontsize=13, labelpad=8)
    ax.set_title(
        "Data Scaling Study: R² Score vs Training Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=14, fontweight="bold", color=DARK, pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in x])
    ax.set_ylim(0.992, 0.999)
    ax.set_xlim(x[0] - 250, x[-1] + 250)
    ax.legend(loc="lower right")

    avg_gap = np.mean(gnn_r2 - mlp_r2)
    textstr = (
        f"GNN avg R²: {np.mean(gnn_r2):.4f}\n"
        f"MLP avg R²: {np.mean(mlp_r2):.4f}\n"
        f"Avg gap: +{avg_gap:.4f} (GNN)"
    )
    ax.text(0.02, 0.20, textstr,
            transform=ax.transAxes, fontsize=9.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GREY, alpha=0.85))

    plt.tight_layout()
    save(fig, "02_r2_vs_dataset_size")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — MSE vs Number of Training Circuits
# ══════════════════════════════════════════════════════════════════════════════
def plot_mse():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    gnn_mse   = np.array(data["gnn_mse"])
    mlp_mse   = np.array(data["mlp_mse"])
    noisy_mse = np.array(data["noisy_mse"])

    ax.plot(x, noisy_mse, "^:",
            color="#E07B39", lw=2.0, ms=9, zorder=3,
            label="Noisy baseline", markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(x, gnn_mse, "o-",
            color=GNN_COL, lw=2.8, ms=10, zorder=5,
            label="GNN", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(x, mlp_mse, "s--",
            color=MLP_COL, lw=2.8, ms=10, zorder=4,
            label="MLP", markeredgecolor="white", markeredgewidth=1.5)

    ax.fill_between(x, gnn_mse, mlp_mse,
                    where=[g < m for g, m in zip(gnn_mse, mlp_mse)],
                    alpha=0.10, color=GNN_COL, label="GNN advantage")

    for xi, g, m in zip(x, gnn_mse, mlp_mse):
        ax.text(xi, g - 0.000015, f"{g:.5f}",
                ha="center", va="top", fontsize=8.5,
                color=GNN_COL, fontweight="bold")
        ax.text(xi, m + 0.000015, f"{m:.5f}",
                ha="center", va="bottom", fontsize=8.5,
                color=MLP_COL, fontweight="bold")

    ax.set_xlabel("Number of Training Circuits", fontsize=13, labelpad=8)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=13, labelpad=8)
    ax.set_title(
        "Data Scaling Study: MSE vs Training Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=14, fontweight="bold", color=DARK, pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in x])
    ax.set_xlim(x[0] - 250, x[-1] + 250)
    ax.legend(loc="upper right")

    plt.tight_layout()
    save(fig, "03_mse_vs_dataset_size")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — MAE vs Number of Training Circuits
# ══════════════════════════════════════════════════════════════════════════════
def plot_mae():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    gnn_mae = np.array(data["gnn_mae"])
    mlp_mae = np.array(data["mlp_mae"])

    ax.plot(x, gnn_mae, "o-",
            color=GNN_COL, lw=2.8, ms=10, zorder=5,
            label="GNN", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(x, mlp_mae, "s--",
            color=MLP_COL, lw=2.8, ms=10, zorder=4,
            label="MLP", markeredgecolor="white", markeredgewidth=1.5)

    ax.fill_between(x, gnn_mae, mlp_mae,
                    where=[m < g for g, m in zip(gnn_mae, mlp_mae)],
                    alpha=0.10, color=MLP_COL, label="MLP advantage")

    for xi, g, m in zip(x, gnn_mae, mlp_mae):
        ax.text(xi, g + 0.00015, f"{g:.5f}",
                ha="center", va="bottom", fontsize=8.5,
                color=GNN_COL, fontweight="bold")
        ax.text(xi, m - 0.00015, f"{m:.5f}",
                ha="center", va="top", fontsize=8.5,
                color=MLP_COL, fontweight="bold")

    ax.set_xlabel("Number of Training Circuits", fontsize=13, labelpad=8)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=13, labelpad=8)
    ax.set_title(
        "Data Scaling Study: MAE vs Training Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=14, fontweight="bold", color=DARK, pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in x])
    ax.set_xlim(x[0] - 250, x[-1] + 250)
    ax.legend(loc="upper left")

    note = "Note: MLP achieves lower MAE at all dataset sizes\n(more conservative correction strategy)"
    ax.text(0.98, 0.97, note,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=MLP_COL, alpha=0.85))

    plt.tight_layout()
    save(fig, "04_mae_vs_dataset_size")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Improvement Factor vs Number of Training Circuits (GNN only)
# ══════════════════════════════════════════════════════════════════════════════
def plot_improvement():
    fig, ax = plt.subplots(figsize=(11, 6), facecolor=BG)
    ax.set_facecolor(BG)

    gnn_impr = np.array(data["gnn_impr"])
    # Compute MLP improvement from MSE values
    mlp_impr = np.array(data["noisy_mse"]) / np.array(data["mlp_mse"])

    ax.plot(x, gnn_impr, "o-",
            color=GNN_COL, lw=2.8, ms=10, zorder=5,
            label="GNN", markeredgecolor="white", markeredgewidth=1.5)
    ax.plot(x, mlp_impr, "s--",
            color=MLP_COL, lw=2.8, ms=10, zorder=4,
            label="MLP", markeredgecolor="white", markeredgewidth=1.5)

    ax.fill_between(x, gnn_impr, mlp_impr,
                    where=[g > m for g, m in zip(gnn_impr, mlp_impr)],
                    alpha=0.10, color=GNN_COL)

    for xi, g, m in zip(x, gnn_impr, mlp_impr):
        ax.text(xi, g + 0.07, f"{g:.2f}×",
                ha="center", va="bottom", fontsize=9.5,
                color=GNN_COL, fontweight="bold")
        ax.text(xi, m - 0.07, f"{m:.2f}×",
                ha="center", va="top", fontsize=9.5,
                color=MLP_COL, fontweight="bold")

    ax.axhline(1.0, color=DARK, lw=1.0, ls="--", alpha=0.4,
               label="No improvement (1×)")

    ax.set_xlabel("Number of Training Circuits", fontsize=13, labelpad=8)
    ax.set_ylabel("Improvement Factor (MSE_noisy / MSE_mitigated)", fontsize=13, labelpad=8)
    ax.set_title(
        "Data Scaling Study: Improvement Factor vs Training Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=14, fontweight="bold", color=DARK, pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in x])
    ax.set_xlim(x[0] - 250, x[-1] + 250)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")

    plt.tight_layout()
    save(fig, "05_improvement_factor_vs_dataset_size")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — All Four Metrics: 2×2 Panel
# ══════════════════════════════════════════════════════════════════════════════
def plot_all_metrics_panel():
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor=BG)
    fig.suptitle(
        "Data Scaling Study — All Metrics vs Training Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=15, fontweight="bold", color=DARK, y=1.01
    )

    gnn_mr   = np.array(data["gnn_mr"])
    mlp_mr   = np.array(data["mlp_mr"])
    gnn_r2   = np.array(data["gnn_r2"])
    mlp_r2   = np.array(data["mlp_r2"])
    gnn_mse  = np.array(data["gnn_mse"])
    mlp_mse  = np.array(data["mlp_mse"])
    gnn_mae  = np.array(data["gnn_mae"])
    mlp_mae  = np.array(data["mlp_mae"])

    panel_data = [
        (axes[0,0], gnn_mr,  mlp_mr,  "Mitigation Ratio (%)",
         "Mitigation Ratio",   True,  68, 92,  True,  False),
        (axes[0,1], gnn_r2,  mlp_r2,  "R² Score",
         "R²",                 True,  0.992, 0.9995, True, False),
        (axes[1,0], gnn_mse, mlp_mse, "MSE (mitigated)",
         "MSE",                False, None, None,    False, True),
        (axes[1,1], gnn_mae, mlp_mae, "MAE (mitigated)",
         "MAE",                False, None, None,    False, True),
    ]

    for ax, gnn_v, mlp_v, ylabel, title, gnn_wins, ymin, ymax, shade_gnn, shade_mlp in panel_data:
        ax.set_facecolor(BG)
        ax.plot(x, gnn_v, "o-", color=GNN_COL, lw=2.5, ms=9,
                label="GNN", markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(x, mlp_v, "s--", color=MLP_COL, lw=2.5, ms=9,
                label="MLP", markeredgecolor="white", markeredgewidth=1.2)

        if shade_gnn:
            ax.fill_between(x, gnn_v, mlp_v,
                            where=[g > m for g, m in zip(gnn_v, mlp_v)],
                            alpha=0.10, color=GNN_COL)
        if shade_mlp:
            ax.fill_between(x, gnn_v, mlp_v,
                            where=[m < g for g, m in zip(gnn_v, mlp_v)],
                            alpha=0.10, color=MLP_COL)

        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlabel("Training Circuits", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n:,}" for n in x], rotation=15, ha="right")
        ax.legend(fontsize=9)
        if ymin is not None:
            ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    save(fig, "06_all_metrics_panel")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Grouped Bar Chart: GNN vs MLP at Each Dataset Size
# ══════════════════════════════════════════════════════════════════════════════
def plot_grouped_bars():
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
    ax.set_facecolor(BG)

    gnn_mr = np.array(data["gnn_mr"])
    mlp_mr = np.array(data["mlp_mr"])

    bar_width = 0.35
    xi = np.arange(len(x))

    bars_gnn = ax.bar(xi - bar_width/2, gnn_mr,
                      bar_width, label="GNN",
                      color=GNN_COL, alpha=0.87,
                      edgecolor="white", linewidth=1.5)
    bars_mlp = ax.bar(xi + bar_width/2, mlp_mr,
                      bar_width, label="MLP",
                      color=MLP_COL, alpha=0.87,
                      edgecolor="white", linewidth=1.5)

    # Value labels
    for bar in bars_gnn:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=GNN_COL)
    for bar in bars_mlp:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=MLP_COL)

    # Gap annotations above each group
    for i, (g, m) in enumerate(zip(gnn_mr, mlp_mr)):
        gap = g - m
        ax.text(i, max(g, m) + 1.5,
                f"Δ={gap:+.2f} pp",
                ha="center", fontsize=9, color=DARK,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="lightyellow",
                          edgecolor=GREY, alpha=0.8))

    # Reference lines
    for ref in [75, 80, 85]:
        ax.axhline(ref, color=GREY, lw=0.8, ls=":", alpha=0.5)
        ax.text(-0.5, ref + 0.2, f"{ref}%", fontsize=8, color=GREY)

    ax.set_xticks(xi)
    ax.set_xticklabels([f"{n:,}\ncircuits" for n in x], fontsize=11)
    ax.set_xlabel("Training Dataset Size", fontsize=13, labelpad=8)
    ax.set_ylabel("Mitigation Ratio (%)", fontsize=13, labelpad=8)
    ax.set_title(
        "Mitigation Ratio: GNN vs MLP at Each Training Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=14, fontweight="bold", color=DARK, pad=12
    )
    ax.set_ylim(65, 93)
    ax.legend(fontsize=12, loc="lower right")

    # Summary box
    avg_gap = np.mean(gnn_mr - mlp_mr)
    textstr = (
        f"GNN wins at all 5 dataset sizes\n"
        f"Average GNN advantage: {avg_gap:.2f} pp"
    )
    ax.text(0.98, 0.02, textstr,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=GNN_COL, alpha=0.90))

    plt.tight_layout()
    save(fig, "07_grouped_bar_chart")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8 — GNN-MLP Gap vs Dataset Size
# ══════════════════════════════════════════════════════════════════════════════
def plot_gap():
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    ax.set_facecolor(BG)

    gnn_mr = np.array(data["gnn_mr"])
    mlp_mr = np.array(data["mlp_mr"])
    gap    = gnn_mr - mlp_mr

    ax.bar(x, gap, width=350,
           color=[GNN_COL if g > 0 else MLP_COL for g in gap],
           alpha=0.80, edgecolor="white", linewidth=1.5)

    for xi, g in zip(x, gap):
        ax.text(xi, g + 0.08, f"{g:+.2f} pp",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=GNN_COL if g > 0 else MLP_COL)

    ax.axhline(0, color=DARK, lw=1.2, ls="--", alpha=0.5)
    ax.fill_between([x[0]-300, x[-1]+300], 0, 10,
                    alpha=0.05, color=GNN_COL)
    ax.text(x[-1]+100, 5, "GNN leads\n(positive gap)",
            fontsize=8.5, color=GNN_COL, va="center")

    ax.set_xlabel("Number of Training Circuits", fontsize=13, labelpad=8)
    ax.set_ylabel("Mitigation Ratio Gap  (GNN − MLP)  [pp]", fontsize=12)
    ax.set_title(
        "GNN Advantage Over MLP: Mitigation Ratio Gap vs Dataset Size\n"
        "4-Qubit Circuits, Depth 4–10",
        fontsize=14, fontweight="bold", color=DARK, pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in x])
    ax.set_xlim(x[0]-400, x[-1]+500)
    avg_gap = np.mean(gap)
    ax.axhline(avg_gap, color=GREY, lw=1.5, ls="-.",
               alpha=0.7, label=f"Mean gap = {avg_gap:.2f} pp")
    ax.legend(fontsize=10)

    plt.tight_layout()
    save(fig, "08_gnn_mlp_gap")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{'='*55}")
    print("  Data Scaling Study — Visualization")
    print(f"{'='*55}\n")

    print("Generating plots...")
    plot_mitigation_ratio()   # PRIMARY PLOT
    plot_r2()
    plot_mse()
    plot_mae()
    plot_improvement()
    plot_all_metrics_panel()
    plot_grouped_bars()
    plot_gap()

    print(f"\n{'='*55}")
    print("  All 8 plots saved to  ./plots/")
    print(f"{'='*55}")
    print("""
  Plot index:
    01  Mitigation Ratio vs Dataset Size       ← PRIMARY
    02  R² Score vs Dataset Size
    03  MSE vs Dataset Size
    04  MAE vs Dataset Size
    05  Improvement Factor vs Dataset Size
    06  All Metrics — 2×2 Panel
    07  Grouped Bar Chart (GNN vs MLP)
    08  GNN–MLP Gap vs Dataset Size
    """)

    # ── Print summary table ───────────────────────────────────────────────
    print("\n  Summary Table:")
    print(f"  {'Circuits':>10}  {'GNN MR':>8}  {'MLP MR':>8}  "
          f"{'Gap (pp)':>9}  {'GNN R²':>8}  {'MLP R²':>8}")
    print("  " + "-"*60)
    for i, n in enumerate(data["n_circuits"]):
        g  = data["gnn_mr"][i]
        m  = data["mlp_mr"][i]
        gr = data["gnn_r2"][i]
        mr = data["mlp_r2"][i]
        print(f"  {n:>10,}  {g:>7.2f}%  {m:>7.2f}%  "
              f"  {g-m:>+7.2f}  {gr:>8.4f}  {mr:>8.4f}")
    print()


if __name__ == "__main__":
    main()
