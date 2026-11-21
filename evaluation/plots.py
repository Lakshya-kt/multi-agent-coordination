"""
evaluation/plots.py — Publication-ready visualizations for experimental analysis.

Authoritative reference: Group_9_FAI_Project_Idea.pdf Section 4 & 5.

Generates:
  1. Success Rate vs Observation Radius (RQ1, H1) with 95% CIs.
  2. Completion Time vs Observation Radius.
  3. Recovery Ratio rho_X(r) comparison (RQ2, H2).
  4. Memory-Communication Interaction Delta_int(r) (RQ3).
  5. Mutual Information & Empirical Joint Distribution Heatmap (RQ4, H3).
  6. Training Learning Curves with 95% shaded confidence bands.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Clean academic color palette
COLORS = {
    "C1": "#1f77b4",  # Blue (Full observation reference)
    "C2": "#d62728",  # Red (Local-only baseline)
    "C3": "#2ca02c",  # Green (Recurrent memory)
    "C4": "#ff7f0e",  # Orange (Learned communication)
    "C5": "#9467bd",  # Purple (Memory + Communication)
    "C6": "#8c564b",  # Brown (Random message control)
    "heuristic": "#7f7f7f",  # Gray (Heuristic rule-based baseline)
}

MARKERS = {
    "C1": "o",
    "C2": "s",
    "C3": "^",
    "C4": "D",
    "C5": "P",
    "C6": "X",
    "heuristic": "v",
}


def plot_metric_vs_radius(
    summary_df: pd.DataFrame,
    metric: str = "success_rate",
    output_path: Optional[str] = "results/plots/success_vs_radius.png",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot performance vs observation radius r in {1, 2, 3, 5, 'full'} across conditions.
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Radii ordering on x-axis
    radius_order = ["1", "2", "3", "5", "full"]
    x_positions = {r: i for i, r in enumerate(radius_order)}

    conditions = summary_df["condition"].unique()
    for cond in sorted(conditions):
        sub = summary_df[summary_df["condition"] == cond].copy()
        sub["r_str"] = sub["radius"].astype(str)
        sub = sub[sub["r_str"].isin(x_positions)]
        sub["x"] = sub["r_str"].map(x_positions)
        sub = sub.sort_values("x")

        if sub.empty:
            continue

        mean_col = f"{metric}_mean" if f"{metric}_mean" in sub.columns else metric
        lower_col = f"{metric}_ci_lower"
        upper_col = f"{metric}_ci_upper"

        x_vals = sub["x"].values
        y_vals = sub[mean_col].values

        color = COLORS.get(cond, "#333333")
        marker = MARKERS.get(cond, "o")

        if lower_col in sub.columns and upper_col in sub.columns:
            yerr_lower = np.maximum(0, y_vals - sub[lower_col].values)
            yerr_upper = np.maximum(0, sub[upper_col].values - y_vals)
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=[yerr_lower, yerr_upper],
                label=cond,
                color=color,
                marker=marker,
                capsize=4,
                linewidth=1.8,
                markersize=6,
            )
        else:
            ax.plot(
                x_vals,
                y_vals,
                label=cond,
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=6,
            )

    ax.set_xticks(range(len(radius_order)))
    ax.set_xticklabels(["r = 1", "r = 2", "r = 3", "r = 5", "r = full"], fontsize=11)
    ax.set_xlabel("Observation Radius (r)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs. Observation Radius", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="lower right")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_recovery_ratios(
    recovery_dict: Dict[str, Dict[str, float]],
    output_path: Optional[str] = "results/plots/recovery_ratio.png",
) -> plt.Figure:
    """
    Plot Recovery Ratio rho_X(r) for X in {C3, C4, C5} across observation radii.

    recovery_dict structure:
        { "C3": {"1": val, "2": val, ...}, "C4": {...}, "C5": {...} }
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    radii = ["1", "2", "3", "5"]
    x = np.arange(len(radii))
    width = 0.25

    conds = ["C3", "C4", "C5"]
    offsets = [-width, 0, width]

    for i, cond in enumerate(conds):
        if cond not in recovery_dict:
            continue
        vals = [recovery_dict[cond].get(r, 0.0) for r in radii]
        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            label=f"{cond} ({'Memory' if cond=='C3' else 'Comm' if cond=='C4' else 'Both'})",
            color=COLORS.get(cond, "#333"),
            alpha=0.85,
            edgecolor="black",
        )

    ax.axhline(1.0, color="black", linestyle="--", alpha=0.7, label="Full Recovery (rho=1.0)")
    ax.axhline(0.0, color="gray", linestyle="-", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"r = {r}" for r in radii], fontsize=11)
    ax.set_xlabel("Observation Radius (r)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recovery Ratio (rho)", fontsize=12, fontweight="bold")
    ax.set_title("Gap Recovery Ratio rho_X(r) Relative to C1 and C2", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_interaction_analysis(
    interaction_dict: Dict[str, float],
    output_path: Optional[str] = "results/plots/interaction_delta.png",
) -> plt.Figure:
    """
    Plot Memory-Communication Interaction Delta_int(r) across radii.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    radii = sorted(interaction_dict.keys(), key=lambda x: int(x) if x.isdigit() else 99)
    vals = [interaction_dict[r] for r in radii]
    x_pos = np.arange(len(radii))

    bar_colors = ["#2ca02c" if v > 0.02 else "#d62728" if v < -0.02 else "#1f77b4" for v in vals]
    bars = ax.bar(x_pos, vals, color=bar_colors, width=0.45, edgecolor="black", alpha=0.85)

    ax.axhline(0.0, color="black", linestyle="-", linewidth=1.2)
    ax.axhspan(0.02, 1.0, color="#2ca02c", alpha=0.08, label="Complementarity (Delta > 0)")
    ax.axhspan(-1.0, -0.02, color="#d62728", alpha=0.08, label="Substitution (Delta < 0)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"r = {r}" for r in radii], fontsize=11)
    ax.set_xlabel("Observation Radius (r)", fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Interaction $\Delta_{int}(r)$", fontsize=12, fontweight="bold")
    ax.set_title(r"Memory-Communication Interaction: $\Delta_{int} = (S_{C5}-S_{C4}) - (S_{C3}-S_{C2})$", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(-0.5, 0.5)
    ax.legend(frameon=True, loc="upper right", fontsize=10)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_message_direction_confusion(
    messages: Sequence[int],
    directions: Sequence[int],
    num_messages: int = 4,
    output_path: Optional[str] = "results/plots/message_direction_heatmap.png",
    condition_name: str = "C4",
) -> plt.Figure:
    """
    Plot joint probability distribution p(m, d) between message symbols and target directions.
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    dir_labels = ["Up (0)", "Down (1)", "Left (2)", "Right (3)", "None (4)"]
    msg_labels = [f"Sym {m}" for m in range(num_messages)]

    counts = np.zeros((num_messages, 5), dtype=np.float64)
    for m, d in zip(messages, directions):
        if 0 <= m < num_messages and 0 <= d < 5:
            counts[m, d] += 1.0

    p_md = counts / max(len(messages), 1)

    cax = ax.imshow(p_md, cmap="YlGnBu", aspect="auto")
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Joint Probability p(M, D)", fontsize=11)

    ax.set_xticks(range(5))
    ax.set_xticklabels(dir_labels, fontsize=10, rotation=25)
    ax.set_yticks(range(num_messages))
    ax.set_yticklabels(msg_labels, fontsize=10)

    # Annotate probabilities in cells
    for i in range(num_messages):
        for j in range(5):
            val = p_md[i, j]
            color = "white" if val > (p_md.max() * 0.6) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Nearest Visible Target Direction (D)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Message Symbol (M)", fontsize=11, fontweight="bold")
    ax.set_title(f"Joint Distribution p(M, D) — {condition_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return fig


def plot_learning_curves_from_csv(
    csv_paths: Sequence[str],
    output_path: Optional[str] = "results/plots/learning_curves.png",
    window: int = 50,
) -> plt.Figure:
    """
    Plot smoothed training learning curves from CSV logs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    for path in csv_paths:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df.empty or "episode" not in df.columns:
            continue

        label = os.path.basename(path).replace("_train.csv", "")

        # Rolling smooth
        ep = df["episode"]
        ret_smooth = df["return"].rolling(window=window, min_periods=1).mean()
        suc_smooth = df["success"].rolling(window=window, min_periods=1).mean()

        ax1.plot(ep, ret_smooth, label=label, linewidth=1.5)
        ax2.plot(ep, suc_smooth, label=label, linewidth=1.5)

    ax1.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Return (Rolling Mean)", fontsize=11, fontweight="bold")
    ax1.set_title("Training Return Curves", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Episode", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Success Rate (Rolling Mean)", fontsize=11, fontweight="bold")
    ax2.set_title("Training Success Rate Curves", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
    return fig
