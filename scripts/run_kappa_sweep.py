"""
κ_s Pareto sweep: vary the action smoothing coefficient and plot
stability satisfaction vs. welfare to show the trade-off frontier.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psai.config import SimConfig, RLConfig, AblationConfig
from psai.plotting import IEEE_RC
from scripts.run_pipeline import run_single_seed

KAPPA_VALUES = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.50]
SEEDS = [7, 42, 99]
N_EPOCHS = 200


def main():
    results = []

    for kappa in KAPPA_VALUES:
        print(f"\n=== κ_s = {kappa} ===")
        seed_rows = []
        for seed in SEEDS:
            out_dir = os.path.join("results", "kappa_sweep", f"kappa_{kappa}", f"seed_{seed}")
            cfg = SimConfig(seed=seed, epochs=N_EPOCHS)
            # Override kappa_smooth
            import psai.config as config_mod
            original_kappa = RLConfig.kappa_smooth
            RLConfig.kappa_smooth = kappa

            df = run_single_seed(cfg, seed=seed, output_dir=out_dir)
            RLConfig.kappa_smooth = original_kappa  # restore

            welfare = float(df["welfare"].mean())
            stab_10 = float((df["Delta_proxy"] <= 10).mean())
            stab_5 = float((df["Delta_proxy"] <= 5).mean())
            c2 = float(df["c2"].mean())

            seed_rows.append({
                "welfare": welfare,
                "stability_10": stab_10,
                "stability_5": stab_5,
                "c2": c2,
            })

        agg = pd.DataFrame(seed_rows)
        results.append({
            "kappa": kappa,
            "welfare_mean": agg["welfare"].mean(),
            "welfare_std": agg["welfare"].std(),
            "stab10_mean": agg["stability_10"].mean(),
            "stab10_std": agg["stability_10"].std(),
            "stab5_mean": agg["stability_5"].mean(),
            "stab5_std": agg["stability_5"].std(),
            "c2_mean": agg["c2"].mean(),
        })

    rdf = pd.DataFrame(results)
    os.makedirs("results/kappa_sweep", exist_ok=True)
    rdf.to_csv("results/kappa_sweep/pareto_data.csv", index=False)

    # Generate Pareto plot
    plt.rcParams.update(IEEE_RC)
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))

    color1 = "#1f77b4"
    color2 = "#d62728"

    ax1.errorbar(rdf["kappa"], rdf["stab10_mean"], yerr=rdf["stab10_std"],
                 color=color1, marker="o", markersize=5, linewidth=1.2,
                 label=r"Stability@$\epsilon{=}10$", capsize=3)
    ax1.set_xlabel(r"Smoothing Coefficient $\kappa_s$")
    ax1.set_ylabel("Stability Satisfaction Rate", color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 1.05])

    ax2 = ax1.twinx()
    ax2.errorbar(rdf["kappa"], rdf["welfare_mean"], yerr=rdf["welfare_std"],
                 color=color2, marker="s", markersize=5, linewidth=1.2,
                 label="Welfare", capsize=3)
    ax2.set_ylabel("Welfare", color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Annotate current choice
    idx_cur = next(i for i, k in enumerate(rdf["kappa"]) if abs(k - 0.50) < 0.01)
    ax1.annotate(r"$\kappa_s{=}0.50$" + "\n(selected)",
                 xy=(rdf["kappa"].iloc[idx_cur], rdf["stab10_mean"].iloc[idx_cur]),
                 xytext=(0.3, 0.5), fontsize=7,
                 arrowprops=dict(arrowstyle="->", color="gray"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", fontsize=7)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/kappa_pareto.png", dpi=300)
    fig.savefig("figures/kappa_pareto.pdf")
    plt.close(fig)
    print("\nPareto plot saved to figures/kappa_pareto.pdf")


if __name__ == "__main__":
    main()
