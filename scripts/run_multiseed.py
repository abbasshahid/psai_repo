#!/usr/bin/env python3
"""Multi-seed runner for PSAI experiments.

Runs the PSAI pipeline across multiple seeds and produces
aggregated results with mean±std and 95% CI.
"""
import argparse, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from scipy import stats

from psai.config import SimConfig, AblationConfig
from psai.plotting import save_comparison_line, save_bar_comparison
from scripts.run_pipeline import run_single_seed


def aggregate_seeds(all_dfs: list, output_dir: str, cfg: SimConfig):
    """Aggregate per-seed DataFrames into summary tables and comparison plots."""
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    n_seeds = len(all_dfs)

    # Stack all seed data
    metrics_of_interest = [
        "welfare", "herfindahl", "total_penalty", "Delta_proxy",
        "c0", "c1", "c2", "sybil_ratio", "sybil_occurred", "collusion_occurred",
        "herfindahl_prop", "herfindahl_qos", "herfindahl_heur",
        "total_penalty_fixed", "total_penalty_heur",
    ]

    # Per-epoch aggregation (for time-series plots)
    max_epochs = max(len(df) for df in all_dfs)
    epoch_data = {}
    for col in metrics_of_interest:
        vals = np.full((n_seeds, max_epochs), np.nan)
        for i, df in enumerate(all_dfs):
            if col in df.columns:
                vals[i, :len(df)] = df[col].values
        epoch_data[col] = vals

    # Mean ± std per epoch
    epoch_summary = {"t": np.arange(max_epochs)}
    for col in metrics_of_interest:
        if col in epoch_data:
            epoch_summary[f"{col}_mean"] = np.nanmean(epoch_data[col], axis=0)
            epoch_summary[f"{col}_std"] = np.nanstd(epoch_data[col], axis=0)
    epoch_df = pd.DataFrame(epoch_summary)
    epoch_df.to_csv(os.path.join(output_dir, "tables", "epoch_aggregated.csv"), index=False)

    # Overall summary table
    summary_rows = []
    for col in ["welfare", "herfindahl", "total_penalty", "Delta_proxy", "c1", "c2"]:
        per_seed_means = [df[col].mean() for df in all_dfs if col in df.columns]
        if per_seed_means:
            arr = np.array(per_seed_means)
            ci = stats.t.interval(0.95, len(arr)-1, loc=arr.mean(), scale=stats.sem(arr)) if len(arr) > 1 else (arr[0], arr[0])
            summary_rows.append({
                "metric": col,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "n_seeds": len(arr),
            })
    pd.DataFrame(summary_rows).to_csv(os.path.join(output_dir, "tables", "summary_stats.csv"), index=False)

    # Stability rate table at multiple epsilon values
    eps_values = [1.0, 2.0, 5.0, 10.0]
    stab_rows = []
    for eps in eps_values:
        rates = [float((df["Delta_proxy"] <= eps).mean()) for df in all_dfs]
        arr = np.array(rates)
        ci = stats.t.interval(0.95, len(arr)-1, loc=arr.mean(), scale=stats.sem(arr)) if len(arr) > 1 else (arr[0], arr[0])
        stab_rows.append({
            "epsilon": eps,
            "stability_rate_mean": float(arr.mean()),
            "stability_rate_std": float(arr.std()),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
        })
    pd.DataFrame(stab_rows).to_csv(os.path.join(output_dir, "tables", "stability_rates.csv"), index=False)

    # Constraint violation rates
    viol_rows = []
    for cname, threshold in [("c1", 0.40), ("c2", 0.10)]:
        rates = [float((df[cname] > threshold).mean()) for df in all_dfs]
        arr = np.array(rates)
        viol_rows.append({
            "constraint": cname,
            "threshold": threshold,
            "violation_rate_mean": float(arr.mean()),
            "violation_rate_std": float(arr.std()),
        })
    pd.DataFrame(viol_rows).to_csv(os.path.join(output_dir, "tables", "constraint_violations.csv"), index=False)

    # Stress test coverage
    sybil_counts = [int((df["sybil_occurred"]>0).sum()) for df in all_dfs]
    collusion_counts = [int((df["collusion_occurred"]>0).sum()) for df in all_dfs]
    stress_df = pd.DataFrame({
        "metric": ["sybil_events", "collusion_events"],
        "mean": [np.mean(sybil_counts), np.mean(collusion_counts)],
        "min": [np.min(sybil_counts), np.min(collusion_counts)],
        "max": [np.max(sybil_counts), np.max(collusion_counts)],
    })
    stress_df.to_csv(os.path.join(output_dir, "tables", "stress_test_coverage.csv"), index=False)

    # --- Comparison plots ---
    # 1) Welfare vs epochs: PSAI vs baselines approximation
    #    PSAI welfare is directly logged; baselines share same obs so welfare is same
    #    but we compare herfindahl and penalties which differ
    welfare_comp = {
        "PSAI": pd.DataFrame({"t": epoch_df["t"], "mean": epoch_df["welfare_mean"], "std": epoch_df["welfare_std"]}),
    }
    save_comparison_line(welfare_comp, "t",
                         os.path.join(output_dir, "plots", "welfare_comparison.png"),
                         os.path.join(output_dir, "plots", "welfare_comparison.pdf"),
                         title="Social Welfare Over Epochs", ylabel="W(t)")

    # 2) Herfindahl comparison
    herf_comp = {}
    for label, col in [("PSAI", "herfindahl_mean"), ("Stake-Prop", "herfindahl_prop_mean"),
                        ("QoS-Only", "herfindahl_qos_mean"), ("Heuristic-β", "herfindahl_heur_mean")]:
        if col in epoch_df.columns:
            std_col = col.replace("_mean", "_std")
            herf_comp[label] = pd.DataFrame({
                "t": epoch_df["t"], "mean": epoch_df[col],
                "std": epoch_df.get(std_col, np.zeros(len(epoch_df)))
            })
    save_comparison_line(herf_comp, "t",
                         os.path.join(output_dir, "plots", "herfindahl_comparison.png"),
                         os.path.join(output_dir, "plots", "herfindahl_comparison.pdf"),
                         title="Herfindahl Index (Decentralization)", ylabel="H(t)")

    # 3) Penalty comparison
    pen_comp = {}
    for label, col in [("PSAI", "total_penalty_mean"), ("Fixed-Slash", "total_penalty_fixed_mean"),
                        ("Heuristic-β", "total_penalty_heur_mean")]:
        if col in epoch_df.columns:
            std_col = col.replace("_mean", "_std")
            pen_comp[label] = pd.DataFrame({
                "t": epoch_df["t"], "mean": epoch_df[col],
                "std": epoch_df.get(std_col, np.zeros(len(epoch_df)))
            })
    save_comparison_line(pen_comp, "t",
                         os.path.join(output_dir, "plots", "penalty_comparison.png"),
                         os.path.join(output_dir, "plots", "penalty_comparison.pdf"),
                         title="Total Penalty Over Epochs", ylabel="Penalty")

    # 4) Bar chart: summary comparison
    bar_labels = ["Welfare", "Herfindahl", "Total Penalty"]
    psai_vals = [epoch_df["welfare_mean"].mean(), epoch_df["herfindahl_mean"].mean(), epoch_df["total_penalty_mean"].mean()]
    bar_data = {"PSAI": psai_vals}
    bar_err = {"PSAI": [epoch_df["welfare_std"].mean(), epoch_df["herfindahl_std"].mean(), epoch_df["total_penalty_std"].mean()]}

    if "herfindahl_prop_mean" in epoch_df.columns:
        bar_data["Stake-Prop"] = [epoch_df["welfare_mean"].mean(), epoch_df["herfindahl_prop_mean"].mean(), 0.0]
        bar_err["Stake-Prop"] = [epoch_df["welfare_std"].mean(), epoch_df["herfindahl_prop_std"].mean(), 0.0]
    if "herfindahl_qos_mean" in epoch_df.columns:
        bar_data["QoS-Only"] = [epoch_df["welfare_mean"].mean(), epoch_df["herfindahl_qos_mean"].mean(), 0.0]
        bar_err["QoS-Only"] = [epoch_df["welfare_std"].mean(), epoch_df["herfindahl_qos_std"].mean(), 0.0]

    save_bar_comparison(bar_labels, bar_data,
                        os.path.join(output_dir, "plots", "summary_bars.png"),
                        os.path.join(output_dir, "plots", "summary_bars.pdf"),
                        title="PSAI vs Baselines (Aggregated)", ylabel="Value",
                        errors_dict=bar_err)

    print(f"\nAggregated results saved to {output_dir}/")
    print(f"  Tables: {os.path.join(output_dir, 'tables')}")
    print(f"  Plots:  {os.path.join(output_dir, 'plots')}")


def main():
    ap = argparse.ArgumentParser(description="Run PSAI across multiple seeds")
    ap.add_argument("--seeds", type=int, default=10, help="Number of seeds to run")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--validators", type=int, default=30)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--output_dir", type=str, default="results")
    args = ap.parse_args()

    cfg = SimConfig(epochs=args.epochs, validators=args.validators, users=args.users, K=args.K)

    all_dfs = []
    for seed in range(args.seeds):
        print(f"\n{'='*60}")
        print(f"Running seed {seed}/{args.seeds-1}")
        print(f"{'='*60}")
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        df = run_single_seed(cfg, seed=seed, output_dir=seed_dir)
        all_dfs.append(df)

    # Aggregate
    agg_dir = os.path.join(args.output_dir, "aggregated")
    aggregate_seeds(all_dfs, agg_dir, cfg)

    print(f"\nAll {args.seeds} seeds complete. Aggregated output in {agg_dir}/")


if __name__ == "__main__":
    main()
