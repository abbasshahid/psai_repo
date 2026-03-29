"""
Ablation study: run 6 mechanism variants across 3 seeds and produce a LaTeX table.
Variants:
  1. PSAI (full)           – all components
  2. No quadratic gate     – linear gate g(z) = z/z_th
  3. No risk multiplier    – penalty without (1 + λρ)
  4. Gate without risk     – quadratic gate but no (1 + λρ)
  5. No smoothing          – κ_s = 0
  6. No δ                  – δ_i = 1 for all validators
"""
import os, sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd

from psai.config import SimConfig, AblationConfig, RLConfig
from scripts.run_pipeline import run_single_seed

VARIANTS = {
    "PSAI (full)":        AblationConfig(),
    "Linear gate":        AblationConfig(disable_quadratic_gate=True),
    "No risk mult.":      AblationConfig(disable_penalty_risk=True),
    "Gate w/o risk":      AblationConfig(disable_penalty_risk=True),  # gate stays quadratic
    "No smoothing":       AblationConfig(disable_smoothing=True),
    r"No $\delta$":       AblationConfig(disable_delta=True),
}

SEEDS = [7, 42, 99]
N_EPOCHS = 200


def main():
    base_cfg = SimConfig(epochs=N_EPOCHS)
    results = []

    for vname, abl in VARIANTS.items():
        print(f"\n=== Variant: {vname} ===")
        seed_rows = []
        for seed in SEEDS:
            out_dir = os.path.join("results", "ablation", vname.replace(" ", "_"), f"seed_{seed}")
            # Override kappa_smooth for "No smoothing"
            rl_cfg = RLConfig()
            if abl.disable_smoothing:
                rl_cfg.kappa_smooth = 0.0

            cfg = SimConfig(seed=seed, epochs=N_EPOCHS)
            df = run_single_seed(cfg, seed=seed, output_dir=out_dir, ablation=abl)

            # Compute key metrics
            zs_all = df["c2"].values
            herf = df["herfindahl"].values
            welfare = df["welfare"].values
            delta = df["Delta_proxy"].values
            total_pen = df["total_penalty"].values

            seed_rows.append({
                "welfare": float(np.mean(welfare)),
                "herfindahl": float(np.mean(herf)),
                "c2": float(np.mean(zs_all)),
                "stability_10": float((delta <= 10).mean()),
                "total_penalty": float(np.mean(total_pen)),
            })

        agg = pd.DataFrame(seed_rows)
        results.append({
            "variant": vname,
            "welfare": f"{agg['welfare'].mean():.3f} $\\pm$ {agg['welfare'].std():.3f}",
            "herfindahl": f"{agg['herfindahl'].mean():.3f} $\\pm$ {agg['herfindahl'].std():.3f}",
            "c2": f"{agg['c2'].mean():.3f} $\\pm$ {agg['c2'].std():.3f}",
            "stab_10": f"{agg['stability_10'].mean():.1%}",
            "penalty": f"{agg['total_penalty'].mean():.1f} $\\pm$ {agg['total_penalty'].std():.1f}",
        })

    # Generate LaTeX table
    os.makedirs("tables", exist_ok=True)
    with open("tables/table_ablation.tex", "w") as f:
        f.write(r"""\begin{table}[t]
\centering
\caption{Ablation study across mechanism components (3 seeds, 200 epochs each). Bold indicates the best value.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Variant} & \textbf{Welfare} & \textbf{Herfindahl} & \textbf{$c_2$} & \textbf{Stab.@10} & \textbf{Penalty} \\
\midrule
""")
        for r in results:
            f.write(f"{r['variant']} & {r['welfare']} & {r['herfindahl']} & {r['c2']} & {r['stab_10']} & {r['penalty']} \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"\nTable saved to tables/table_ablation.tex")

    # Also save raw data
    pd.DataFrame(results).to_csv("results/ablation/ablation_summary.csv", index=False)


if __name__ == "__main__":
    main()
