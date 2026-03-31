
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE Transactions formatting defaults
IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (3.5, 2.5),   # single-column IEEE
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Colorblind-safe palette
COLORS = {
    "PSAI": "#1f77b4",
    "Stake-Prop": "#ff7f0e",
    "QoS-Only": "#2ca02c",
    "Fixed-Slash": "#d62728",
    "Heuristic-β": "#9467bd",
}

def _apply_ieee():
    plt.rcParams.update(IEEE_RC)

def save_line(df: pd.DataFrame, x: str, y: str, out_png: str, out_pdf: str,
              title: str = "", ylabel: str = ""):
    """Single-series line plot with IEEE formatting."""
    _apply_ieee()
    fig, ax = plt.subplots()
    ax.plot(df[x].values, df[y].values, color=COLORS["PSAI"], linewidth=1.2)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel or y.replace("_", " ").title())
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

def save_hist(values, out_png: str, out_pdf: str, xlabel: str, title: str = ""):
    """Histogram with IEEE formatting."""
    _apply_ieee()
    fig, ax = plt.subplots()
    ax.hist(values, bins=30, color=COLORS["PSAI"], edgecolor="white", alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

def save_comparison_line(data_dict: dict, x_key: str, out_png: str, out_pdf: str,
                          title: str = "", xlabel: str = "Epoch", ylabel: str = "",
                          ci_dict: dict = None):
    """Multi-series comparison line plot with optional CI shading.

    Args:
        data_dict: {label: pd.DataFrame} with columns [x_key, "mean", and optionally "std"]
        x_key: column name for x-axis
        out_png, out_pdf: output paths
        ci_dict: {label: {"lower": array, "upper": array}} for 95% CI bands
    """
    _apply_ieee()
    fig, ax = plt.subplots()

    for label, df in data_dict.items():
        color = COLORS.get(label, None)
        x_vals = df[x_key].values if x_key in df.columns else np.arange(len(df))
        y_vals = df["mean"].values

        ax.plot(x_vals, y_vals, label=label, color=color, linewidth=1.2)

        # CI shading from ci_dict or from std column
        if ci_dict and label in ci_dict:
            lower = ci_dict[label]["lower"]
            upper = ci_dict[label]["upper"]
            ax.fill_between(x_vals, lower, upper, alpha=0.15, color=color)
        elif "std" in df.columns:
            ax.fill_between(x_vals, y_vals - 1.96 * df["std"].values,
                           y_vals + 1.96 * df["std"].values, alpha=0.15, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

def save_bar_comparison(labels: list, values_dict: dict, out_png: str, out_pdf: str,
                         title: str = "", ylabel: str = "", errors_dict: dict = None,
                         log_y: bool = False):
    """Save separate bar charts for comparing metrics."""
    _apply_ieee()
    
    n_bars = len(values_dict)
    
    base_png = out_png.replace(".png", "")
    base_pdf = out_pdf.replace(".pdf", "")

    for j, metric in enumerate(labels):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        
        names = []
        vals_list = []
        errs_list = []
        colors = []
        
        for name, vals in values_dict.items():
            names.append(name)
            vals_list.append(vals[j])
            errs = errors_dict.get(name) if errors_dict else None
            errs_list.append(errs[j] if errs else 0.0)
            colors.append(COLORS.get(name, "gray"))
            
        x = np.arange(n_bars)
        bars = ax.bar(x, vals_list, color=colors, alpha=0.85, yerr=errs_list, capsize=3)
        
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel(metric)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        if metric == "Total Penalty":
            for i, val in enumerate(vals_list):
                if val <= 1e-6:
                    ax.text(x[i], max(vals_list)*0.02, "0 (N/A)", ha='center', va='bottom', fontsize=8, color='black', alpha=0.8)

        # No separate legend needed since x-ticks label the bars directly
        fig.tight_layout()
        metric_slug = metric.lower().replace(" ", "_")
        fig.savefig(f"{base_png}_{metric_slug}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{base_pdf}_{metric_slug}.pdf", bbox_inches="tight")
        plt.close(fig)
