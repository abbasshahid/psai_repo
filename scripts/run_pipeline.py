
import argparse, json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from tqdm import trange

from psai.config import SimConfig, Bounds, RLConfig, ConstraintConfig
from psai.simulation import PSAISimEnv
from psai.predictors import LogisticPredictor, train_predictor
from psai.contract_mirror import PSAIContractMirror
from psai.orchestrator import PSAIOrchestrator
from psai.encoding import Action
from psai.rl_engine import PrimalDualPPO, Rollout
from psai.baselines import baseline_proportional, baseline_qos
from psai.metrics import herfindahl_from_dict, welfare, deviation_gain, sybil_unprofitability_check
from psai.plotting import save_line, save_hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--validators", type=int, default=30)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--sybil_prob", type=float, default=0.15)
    ap.add_argument("--collusion_prob", type=float, default=0.10)
    args = ap.parse_args()

    cfg = SimConfig(epochs=args.epochs, validators=args.validators, users=args.users, K=args.K,
                    sybil_prob=args.sybil_prob, collusion_prob=args.collusion_prob)

    rng = np.random.default_rng(cfg.seed)

    # -------------------------
    # 1) Initialize simulation
    # -------------------------
    env = PSAISimEnv(K=cfg.K, N=cfg.validators, users=cfg.users, rng=rng,
                    sybil_prob=cfg.sybil_prob, collusion_prob=cfg.collusion_prob)

    # -------------------------
    # 2) Initialize predictors
    #    Train on synthetic warmup data
    # -------------------------
    warm_X, warm_yq, warm_yr = [], [], []
    for t in range(60):
        v_onchain, obs0, _, x_dict = env.step(t, action_params={})
        # labels (conservative): reliability = 1 if m>0.7 and z<0.3; risk label = 1 if z>0.5 or collusion flagged
        for vid, v in v_onchain.items():
            warm_X.append(v.x)
            warm_yq.append(1.0 if (v.m > 0.7 and v.z < 0.3) else 0.0)
            warm_yr.append(1.0 if (v.z > 0.5 or obs0["colluding"] > 0.5) else 0.0)

    warm_X = np.stack(warm_X, axis=0)
    warm_yq = np.array(warm_yq, dtype=float)
    warm_yr = np.array(warm_yr, dtype=float)

    pred = LogisticPredictor(cfg.K)
    wq, wr = train_predictor(pred, warm_X, warm_yq, warm_yr, epochs=100, lr=2e-2)

    # -------------------------
    # 3) Initialize contract mirror and orchestrator
    # -------------------------
    bounds = Bounds()
    contract = PSAIContractMirror(wq=wq, wr=wr, bounds={
        "alpha_min": bounds.alpha_min, "alpha_max": bounds.alpha_max,
        "beta_min": bounds.beta_min, "beta_max": bounds.beta_max,
        "lambda_max": bounds.lambda_max, "eta_max_abs": bounds.eta_max_abs
    })
    orch = PSAIOrchestrator(contract)

    # -------------------------
    # 4) RL Engine (CMDP)
    # -------------------------
    # Observation vector o_t = [T,L,F,A,demand,colluding,feature_attack,num_validators, reward_pool, prevHerf]
    obs_dim = 10
    rl = RLConfig()
    cc = ConstraintConfig()
    agent = PrimalDualPPO(obs_dim=obs_dim, K=cfg.K, bounds=bounds, rl=rl, cc=cc)

    # -------------------------
    # 5) Main loop: Observation -> AI -> Commit -> Reveal -> Settlement
    # -------------------------
    logs = []
    prev_herf = 1.0 / cfg.validators
    rollout_data = []

    for t in trange(cfg.epochs, desc="Running PSAI pipeline"):
        v_onchain, obs, _, _ = env.step(t, action_params={})
        obs["reward_pool"] = cfg.reward_pool

        o = np.array([obs["T"], obs["L"], obs["F"], obs["A"], obs["demand"], obs["colluding"],
                      obs["feature_attack"], obs["num_validators"], obs["reward_pool"], prev_herf], dtype=float)

        action, a_raw, logp, val = agent.act(o)

        payouts, penalties, aux = orch.run_epoch(t, v_onchain, obs, action)

        # Baselines for comparison + deviation proxy
        base_p = baseline_proportional(v_onchain, cfg.reward_pool)
        base_q = baseline_qos(v_onchain, cfg.reward_pool)

        herf = herfindahl_from_dict(payouts)
        prev_herf = herf

        # welfare (Eq. 29 proxy) and RL reward (Eq. 10)
        W = welfare(obs["T"], obs["L"], obs["F"], obs["A"])
        r_t = W

        # CMDP costs (proxies):
        # c0 inflation proxy = |paid - reward_pool| (should be ~0 due to budget balance)
        paid = sum(payouts.values())
        c0 = abs(paid - cfg.reward_pool) / (cfg.reward_pool + 1e-12)
        # c1 centralization = herfindahl
        c1 = herf
        # c2 false slashing proxy: penalty applied when z is low (avg over validators)
        zs = np.array([v_onchain[i].z for i in v_onchain.keys()], dtype=float)
        ls = np.array([penalties[i] for i in penalties.keys()], dtype=float)
        # define "low misbehavior" as z<0.2; any penalty there counts as false-slash mass
        mask = (zs < 0.2).astype(float)
        c2 = float((ls * mask).sum() / (ls.sum() + 1e-12)) if ls.sum() > 0 else 0.0

        # stability proxy Delta_i(t) and Sybil-unprofitability check
        Delta = deviation_gain(base_p, payouts)

        # Sybil check: compare sum of weights if any sybils exist (ids containing "_S")
        w_sum_sybil = sum(a["w"] for vid, a in aux.items() if "_S" in vid)
        # approximate aggregate weight as the max weight among sybil cluster base id (if present)
        w_agg = 0.0
        for vid, a in aux.items():
            if "_S" not in vid and any(vid in s for s in aux.keys() if "_S" in s):
                w_agg = max(w_agg, a["w"])
        sybil_ratio = sybil_unprofitability_check(w_agg, w_sum_sybil) if w_sum_sybil > 0 and w_agg > 0 else 0.0

        # coalition utility proxy: total payout to top-5 (if collusion likely, should be damped)
        top5 = sorted(payouts.values(), reverse=True)[:5]
        coalition_u = float(sum(top5))

        logs.append({
            "t": t,
            "alpha": action.alpha, "beta": action.beta, "lambda": action.lam, "kappa": action.kappa,
            "reward_paid": paid,
            "total_penalty": sum(penalties.values()),
            "herfindahl": herf,
            "welfare": W,
            "Delta_proxy": Delta,
            "sybil_ratio": sybil_ratio,
            "coalition_u_top5": coalition_u,
            "mu0": agent.mu.detach().cpu().numpy()[0],
            "mu1": agent.mu.detach().cpu().numpy()[1],
            "mu2": agent.mu.detach().cpu().numpy()[2],
            "c0": c0, "c1": c1, "c2": c2,
            "T": obs["T"], "L": obs["L"], "F": obs["F"], "A": obs["A"],
        })

        rollout_data.append((o, a_raw, logp, r_t, val, np.array([c0, c1, c2], dtype=float)))

        # update RL periodically
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

            # log update info
            logs[-1].update({
                "policy_loss": info["policy_loss"],
                "value_loss": info["value_loss"],
                "entropy": info["entropy"],
                "avg_cost0": info["avg_cost"][0],
                "avg_cost1": info["avg_cost"][1],
                "avg_cost2": info["avg_cost"][2],
            })

    df = pd.DataFrame(logs)
    df.to_csv("results/tables/epoch_metrics.csv", index=False)

    # Save raw orchestrator records (commit/reveal/payouts/aux)
    orch.save("results/summary_records.json")

    # Validation tables
    # 1) Budget adherence
    budget = pd.DataFrame({
        "mean_abs_budget_error": [float(np.mean(np.abs(df["reward_paid"] - 1000.0)))],
        "max_abs_budget_error": [float(np.max(np.abs(df["reward_paid"] - 1000.0)))],
    })
    budget.to_csv("results/tables/budget_adherence.csv", index=False)

    # 2) Constraint satisfaction
    cons = pd.DataFrame({
        "mean_c0": [float(df["c0"].mean())],
        "mean_c1_herf": [float(df["c1"].mean())],
        "mean_c2_false_slash": [float(df["c2"].mean())],
        "viol_c1_rate": [float((df["c1"] > 0.40).mean())],
        "viol_c2_rate": [float((df["c2"] > 0.10).mean())],
    })
    cons.to_csv("results/tables/constraint_satisfaction.csv", index=False)

    # 3) Stability condition Delta <= epsilon
    eps = 5.0
    stab = pd.DataFrame({
        "epsilon": [eps],
        "rate_Delta_le_eps": [float((df["Delta_proxy"] <= eps).mean())],
        "mean_Delta": [float(df["Delta_proxy"].mean())],
        "max_Delta": [float(df["Delta_proxy"].max())],
    })
    stab.to_csv("results/tables/stability_check.csv", index=False)

    # 4) Sybil-unprofitability (ratio <= 1 ideally, here proxy)
    syb = pd.DataFrame({
        "mean_sybil_ratio": [float(df["sybil_ratio"][df["sybil_ratio"]>0].mean() if (df["sybil_ratio"]>0).any() else 0.0)],
        "max_sybil_ratio": [float(df["sybil_ratio"].max())],
        "count_sybil_events": [int((df["sybil_ratio"]>0).sum())],
    })
    syb.to_csv("results/tables/sybil_unprofitability.csv", index=False)

    # 5) Collusion deterrence proxy: top5 payout mass vs herfindahl
    coll = pd.DataFrame({
        "mean_top5_payout": [float(df["coalition_u_top5"].mean())],
        "mean_herfindahl": [float(df["herfindahl"].mean())],
    })
    coll.to_csv("results/tables/collusion_proxy.csv", index=False)

    # Plots
    os.makedirs("results/plots", exist_ok=True)
    save_line(df, "t", "welfare", "results/plots/welfare.png", "results/plots/welfare.pdf", "Welfare proxy (Eq. 29)")
    save_line(df, "t", "herfindahl", "results/plots/herfindahl.png", "results/plots/herfindahl.pdf", "Centralization (Herfindahl)")
    save_line(df, "t", "total_penalty", "results/plots/penalties.png", "results/plots/penalties.pdf", "Penalty total (Eq. 27)")
    save_line(df, "t", "alpha", "results/plots/alpha.png", "results/plots/alpha.pdf", "Learned alpha(t)")
    save_line(df, "t", "beta", "results/plots/beta.png", "results/plots/beta.pdf", "Learned beta(t)")
    save_line(df, "t", "lambda", "results/plots/lambda.png", "results/plots/lambda.pdf", "Learned lambda(t)")
    save_hist(df["Delta_proxy"].values, "results/plots/delta_hist.png", "results/plots/delta_hist.pdf", "Delta proxy", "Predictive stability proxy")

    # Summary JSON with claim validation pointers
    summary = {
        "budget_balance": {
            "claim": "Eq. (28) holds: sum_i p_i(t) = R_t",
            "evidence_table": "results/tables/budget_adherence.csv",
            "metric": {"max_abs_budget_error": float(budget["max_abs_budget_error"][0])}
        },
        "predictive_stability": {
            "claim": "Delta_i(t) <= epsilon (Eq. 31 / eps-equilibrium target) via proxy",
            "evidence_table": "results/tables/stability_check.csv",
            "metric": {"rate_Delta_le_eps": float(stab["rate_Delta_le_eps"][0])}
        },
        "sybil_unprofitability": {
            "claim": "Sybil splitting does not increase total weight (Eq. 40) via proxy ratio<=1",
            "evidence_table": "results/tables/sybil_unprofitability.csv",
            "metric": {"max_sybil_ratio": float(syb["max_sybil_ratio"][0])}
        },
        "collusion_deterrence": {
            "claim": "Coalition benefit is damped by risk amplification (Eqs. 41–42) proxied by top5 mass and herfindahl",
            "evidence_table": "results/tables/collusion_proxy.csv",
            "metric": {"mean_herfindahl": float(coll["mean_herfindahl"][0])}
        },
        "training": {
            "claim": "CMDP primal–dual PPO reduces constraint violations while improving welfare proxy",
            "evidence_plot": "results/plots/welfare.png",
            "multipliers": df[["mu0","mu1","mu2"]].tail(1).to_dict(orient="records")[0]
        }
    }
    with open("results/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Outputs saved under results/")

if __name__ == "__main__":
    main()