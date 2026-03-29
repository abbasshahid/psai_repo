
import argparse, json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from tqdm import trange

from psai.config import SimConfig, Bounds, RLConfig, ConstraintConfig, AblationConfig
from psai.simulation import PSAISimEnv
from psai.predictors import LogisticPredictor, train_predictor
from psai.contract_mirror import PSAIContractMirror
from psai.orchestrator import PSAIOrchestrator
from psai.encoding import Action
from psai.rl_engine import PrimalDualPPO, Rollout
from psai.baselines import baseline_proportional, baseline_qos, baseline_fixed_slashing, baseline_heuristic_beta
from psai.metrics import herfindahl_from_dict, welfare, deviation_gain, sybil_unprofitability_check
from psai.plotting import save_line, save_hist


def run_single_seed(cfg: SimConfig, seed: int, output_dir: str,
                    ablation: AblationConfig = None) -> pd.DataFrame:
    """Run PSAI pipeline for a single seed. Returns the metrics DataFrame."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    rng = np.random.default_rng(seed)
    ablation = ablation or AblationConfig()

    # 1) Initialize simulation with forced stress tests
    env = PSAISimEnv(K=cfg.K, N=cfg.validators, users=cfg.users, rng=rng,
                    sybil_prob=cfg.sybil_prob, collusion_prob=cfg.collusion_prob,
                    forced_sybil_epochs=cfg.forced_sybil_epochs,
                    forced_collusion_epochs=cfg.forced_collusion_epochs,
                    adversary_intensity=cfg.adversary_intensity)

    # 2) Initialize predictors — train on warmup data
    warm_X, warm_yq, warm_yr = [], [], []
    for t in range(60):
        v_onchain, obs0, _, x_dict = env.step(t, action_params={})
        for vid, v in v_onchain.items():
            warm_X.append(v.x)
            warm_yq.append(1.0 if (v.m > 0.7 and v.z < 0.3) else 0.0)
            warm_yr.append(1.0 if (v.z > 0.5 or obs0["colluding"] > 0.5) else 0.0)

    warm_X = np.stack(warm_X, axis=0)
    warm_yq = np.array(warm_yq, dtype=float)
    warm_yr = np.array(warm_yr, dtype=float)

    pred = LogisticPredictor(cfg.K)
    wq, wr, pred_metrics = train_predictor(pred, warm_X, warm_yq, warm_yr, epochs=100, lr=2e-2)

    # Save predictor metrics
    with open(os.path.join(output_dir, "tables", "predictor_metrics.json"), "w") as f:
        json.dump(pred_metrics, f, indent=2)

    # 3) Initialize contract mirror and orchestrator
    bounds = Bounds()
    contract = PSAIContractMirror(wq=wq, wr=wr, bounds={
        "alpha_min": bounds.alpha_min, "alpha_max": bounds.alpha_max,
        "beta_min": bounds.beta_min, "beta_max": bounds.beta_max,
        "lambda_max": bounds.lambda_max, "eta_max_abs": bounds.eta_max_abs
    }, ablation=ablation)
    orch = PSAIOrchestrator(contract)

    # 4) RL Engine (CMDP)
    obs_dim = 10
    rl = RLConfig()
    cc = ConstraintConfig()
    total_updates = cfg.epochs // rl.steps_per_update
    agent = PrimalDualPPO(obs_dim=obs_dim, K=cfg.K, bounds=bounds, rl=rl, cc=cc,
                          total_updates=total_updates)

    # 5) Main loop
    logs = []
    prev_herf = 1.0 / cfg.validators
    rollout_data = []
    prev_a_raw = None
    heuristic_ema_z = 0.0  # for heuristic-β baseline

    for t in trange(cfg.epochs, desc=f"Seed {seed}", leave=False):
        v_onchain, obs, _, _ = env.step(t, action_params={})
        obs["reward_pool"] = cfg.reward_pool

        o = np.array([obs["T"], obs["L"], obs["F"], obs["A"], obs["demand"], obs["colluding"],
                      obs["feature_attack"], obs["num_validators"], obs["reward_pool"], prev_herf], dtype=float)

        action, a_raw, logp, val = agent.act(o)

        payouts, penalties, aux = orch.run_epoch(t, v_onchain, obs, action)

        # Baselines
        base_prop = baseline_proportional(v_onchain, cfg.reward_pool)
        base_qos = baseline_qos(v_onchain, cfg.reward_pool)
        base_fixed_pen = baseline_fixed_slashing(v_onchain, cfg.reward_pool)
        base_heur_pay, base_heur_pen, heuristic_ema_z = baseline_heuristic_beta(
            v_onchain, cfg.reward_pool, heuristic_ema_z)

        herf = herfindahl_from_dict(payouts)
        herf_prop = herfindahl_from_dict(base_prop)
        herf_qos = herfindahl_from_dict(base_qos)
        herf_heur = herfindahl_from_dict(base_heur_pay)
        prev_herf = herf

        W = welfare(obs["T"], obs["L"], obs["F"], obs["A"])

        # Action smoothing penalty (Issue #1)
        smooth_penalty = 0.0
        if prev_a_raw is not None and not ablation.disable_smoothing:
            smooth_penalty = float(np.sum((a_raw - prev_a_raw) ** 2))
        prev_a_raw = a_raw.copy()

        r_t = W - rl.kappa_smooth * smooth_penalty

        # CMDP costs
        paid = sum(payouts.values())
        c0 = abs(paid - cfg.reward_pool) / (cfg.reward_pool + 1e-12)
        c1 = herf
        zs = np.array([v_onchain[i].z for i in v_onchain.keys()], dtype=float)
        ls = np.array([penalties[i] for i in penalties.keys()], dtype=float)
        mask = (zs < 0.2).astype(float)
        c2 = float((ls * mask).sum() / (ls.sum() + 1e-12)) if ls.sum() > 0 else 0.0

        Delta = deviation_gain(base_prop, payouts)

        # Sybil check
        w_sum_sybil = sum(a["w"] for vid, a in aux.items() if "_S" in vid)
        w_agg = 0.0
        for vid, a in aux.items():
            if "_S" not in vid and any(vid in s for s in aux.keys() if "_S" in s):
                w_agg = max(w_agg, a["w"])
        sybil_ratio = sybil_unprofitability_check(w_agg, w_sum_sybil) if w_sum_sybil > 0 and w_agg > 0 else 0.0

        top5 = sorted(payouts.values(), reverse=True)[:5]
        coalition_u = float(sum(top5))

        logs.append({
            "t": t,
            "alpha": action.alpha, "beta": action.beta, "lambda": action.lam, "kappa": action.kappa,
            "reward_paid": paid,
            "total_penalty": sum(penalties.values()),
            "total_penalty_fixed": sum(base_fixed_pen.values()),
            "total_penalty_heur": sum(base_heur_pen.values()),
            "herfindahl": herf,
            "herfindahl_prop": herf_prop,
            "herfindahl_qos": herf_qos,
            "herfindahl_heur": herf_heur,
            "welfare": W,
            "smooth_penalty": smooth_penalty,
            "Delta_proxy": Delta,
            "sybil_ratio": sybil_ratio,
            "coalition_u_top5": coalition_u,
            "mu0": agent.mu.detach().cpu().numpy()[0],
            "mu1": agent.mu.detach().cpu().numpy()[1],
            "mu2": agent.mu.detach().cpu().numpy()[2],
            "c0": c0, "c1": c1, "c2": c2,
            "T": obs["T"], "L": obs["L"], "F": obs["F"], "A": obs["A"],
            "sybil_occurred": obs.get("sybil_occurred", 0.0),
            "collusion_occurred": obs.get("collusion_occurred", 0.0),
        })

        rollout_data.append((o, a_raw, logp, r_t, val, np.array([c0, c1, c2], dtype=float)))

        if (t+1) % rl.steps_per_update == 0:
            obs_arr = np.stack([x[0] for x in rollout_data], axis=0)
            act_arr = np.stack([x[1] for x in rollout_data], axis=0)
            logp_arr = np.array([x[2] for x in rollout_data], dtype=float)
            rew_arr = np.array([x[3] for x in rollout_data], dtype=float)
            val_arr = np.array([x[4] for x in rollout_data], dtype=float)
            cost_arr = np.stack([x[5] for x in rollout_data], axis=0)

            roll = Rollout(obs=obs_arr, act=act_arr, logp=logp_arr, rew=rew_arr, val=val_arr, costs=cost_arr)
            info = agent.update(roll)
            rollout_data = []

            logs[-1].update({
                "policy_loss": info["policy_loss"],
                "value_loss": info["value_loss"],
                "entropy": info["entropy"],
                "lr": info["lr"],
            })

    df = pd.DataFrame(logs)
    df.to_csv(os.path.join(output_dir, "tables", "epoch_metrics.csv"), index=False)
    orch.save(os.path.join(output_dir, "summary_records.json"))

    # Validation tables
    budget = pd.DataFrame({
        "mean_abs_budget_error": [float(np.mean(np.abs(df["reward_paid"] - cfg.reward_pool)))],
        "max_abs_budget_error": [float(np.max(np.abs(df["reward_paid"] - cfg.reward_pool)))],
    })
    budget.to_csv(os.path.join(output_dir, "tables", "budget_adherence.csv"), index=False)

    cons = pd.DataFrame({
        "mean_c0": [float(df["c0"].mean())],
        "mean_c1_herf": [float(df["c1"].mean())],
        "mean_c2_false_slash": [float(df["c2"].mean())],
        "viol_c1_rate": [float((df["c1"] > 0.40).mean())],
        "viol_c2_rate": [float((df["c2"] > 0.10).mean())],
    })
    cons.to_csv(os.path.join(output_dir, "tables", "constraint_satisfaction.csv"), index=False)

    eps_values = [1.0, 2.0, 5.0, 10.0]
    stab_data = {"epsilon": eps_values}
    for eps in eps_values:
        stab_data[f"rate_Delta_le_{eps}"] = [float((df["Delta_proxy"] <= eps).mean())]
    stab_data["mean_Delta"] = [float(df["Delta_proxy"].mean())]
    stab_data["max_Delta"] = [float(df["Delta_proxy"].max())]
    stab = pd.DataFrame(stab_data) if len(eps_values) == 1 else pd.DataFrame({
        k: v if isinstance(v, list) and len(v) > 1 else v * len(eps_values) if isinstance(v, list) else [v]*len(eps_values)
        for k, v in stab_data.items()
    })
    # Simpler: just write one-row summary
    stab_row = {"mean_Delta": float(df["Delta_proxy"].mean()), "max_Delta": float(df["Delta_proxy"].max())}
    for eps in eps_values:
        stab_row[f"rate_le_{eps}"] = float((df["Delta_proxy"] <= eps).mean())
    pd.DataFrame([stab_row]).to_csv(os.path.join(output_dir, "tables", "stability_check.csv"), index=False)

    syb = pd.DataFrame({
        "mean_sybil_ratio": [float(df["sybil_ratio"][df["sybil_ratio"]>0].mean() if (df["sybil_ratio"]>0).any() else 0.0)],
        "max_sybil_ratio": [float(df["sybil_ratio"].max())],
        "count_sybil_events": [int((df["sybil_occurred"]>0).sum())],
        "count_collusion_events": [int((df["collusion_occurred"]>0).sum())],
    })
    syb.to_csv(os.path.join(output_dir, "tables", "sybil_unprofitability.csv"), index=False)

    # Plots
    save_line(df, "t", "welfare", os.path.join(output_dir, "plots", "welfare.png"),
              os.path.join(output_dir, "plots", "welfare.pdf"), "Social Welfare Proxy", "W(t)")
    save_line(df, "t", "herfindahl", os.path.join(output_dir, "plots", "herfindahl.png"),
              os.path.join(output_dir, "plots", "herfindahl.pdf"), "Herfindahl Index", "H(t)")
    save_line(df, "t", "total_penalty", os.path.join(output_dir, "plots", "penalties.png"),
              os.path.join(output_dir, "plots", "penalties.pdf"), "Total Penalty", "P(t)")
    save_line(df, "t", "alpha", os.path.join(output_dir, "plots", "alpha.png"),
              os.path.join(output_dir, "plots", "alpha.pdf"), r"Learned $\alpha(t)$", r"$\alpha$")
    save_line(df, "t", "beta", os.path.join(output_dir, "plots", "beta.png"),
              os.path.join(output_dir, "plots", "beta.pdf"), r"Learned $\beta(t)$", r"$\beta$")
    save_line(df, "t", "lambda", os.path.join(output_dir, "plots", "lambda.png"),
              os.path.join(output_dir, "plots", "lambda.pdf"), r"Learned $\lambda(t)$", r"$\lambda$")
    save_hist(df["Delta_proxy"].values, os.path.join(output_dir, "plots", "delta_hist.png"),
              os.path.join(output_dir, "plots", "delta_hist.pdf"), r"$\Delta$ proxy", "Stability Proxy Distribution")

    # Summary JSON
    summary = {
        "seed": seed,
        "predictor_metrics": pred_metrics,
        "budget_balance": {"max_abs_budget_error": float(budget["max_abs_budget_error"][0])},
        "stability": stab_row,
        "sybil_events": int((df["sybil_occurred"]>0).sum()),
        "collusion_events": int((df["collusion_occurred"]>0).sum()),
        "final_multipliers": {"mu0": float(df["mu0"].iloc[-1]), "mu1": float(df["mu1"].iloc[-1]), "mu2": float(df["mu2"].iloc[-1])},
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--validators", type=int, default=30)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sybil_prob", type=float, default=0.15)
    ap.add_argument("--collusion_prob", type=float, default=0.10)
    ap.add_argument("--output_dir", type=str, default="results")
    args = ap.parse_args()

    cfg = SimConfig(seed=args.seed, epochs=args.epochs, validators=args.validators,
                    users=args.users, K=args.K,
                    sybil_prob=args.sybil_prob, collusion_prob=args.collusion_prob)

    df = run_single_seed(cfg, seed=args.seed, output_dir=args.output_dir)
    print(f"Done. Outputs saved under {args.output_dir}/")


if __name__ == "__main__":
    main()