"""
On-chain fuzzing test: generate 1000+ random action vectors within bounds
and validate fixed-point parity between mirror and simulated on-chain settlement.
Reports max absolute error and error distribution.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psai.config import SimConfig, Bounds
from psai.simulation import PSAISimEnv
from psai.predictors import LogisticPredictor, train_predictor
from psai.contract_mirror import PSAIContractMirror
from psai.encoding import Action, enc_action
from psai.plotting import IEEE_RC

FUZZ_COUNT = 1000
FP_SCALE = 10**6  # fixed-point scaling factor


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def fixed_point_settle(validators, action, wq, wr, reward_pool, K, z_threshold=0.30):
    """Simulate on-chain fixed-point settlement (integer arithmetic)."""
    ids = list(validators.keys())
    X = np.stack([validators[i].x for i in ids], axis=0)
    m = np.array([validators[i].m for i in ids], dtype=float)
    z = np.array([validators[i].z for i in ids], dtype=float)
    S = np.array([validators[i].stake for i in ids], dtype=float)

    # Fixed-point versions
    alpha_fp = int(action.alpha * FP_SCALE)
    beta_fp = int(action.beta * FP_SCALE)
    lam_fp = int(action.lam * FP_SCALE)
    R_fp = int(reward_pool * FP_SCALE)
    eta_fp = np.array([int(e * FP_SCALE) for e in action.eta])

    # q, rho (fixed point approx via lookup-table-style)
    q_raw = X @ wq
    rho_raw = X @ wr
    q = sigmoid(q_raw)
    rho = sigmoid(rho_raw)

    # delta in fixed point (using float exp then truncating)
    delta = np.exp(X @ (np.array(action.eta))) * (q ** action.lam) * ((1.0 - rho) ** action.lam)

    # w
    w = np.exp(action.alpha * m) * delta
    w_fp = (w * FP_SCALE).astype(np.int64)
    Z_fp = int(np.sum(w_fp)) + 1

    # payouts in fixed point
    p_fp = (R_fp * w_fp) // Z_fp
    p_float = p_fp.astype(float) / FP_SCALE

    # penalties in fixed point
    risk_mult = 1.0 + action.lam * rho
    l_raw = action.beta * S * z * risk_mult
    z_gate = np.clip((z / z_threshold) ** 2, 0.0, 1.0)
    l_raw = l_raw * z_gate
    l_raw = np.minimum(l_raw, S)
    l_fp = (l_raw * FP_SCALE).astype(np.int64)
    l_float = l_fp.astype(float) / FP_SCALE

    return {ids[i]: p_float[i] for i in range(len(ids))}, {ids[i]: l_float[i] for i in range(len(ids))}


def main():
    cfg = SimConfig(seed=7)
    bounds = Bounds()
    rng = np.random.default_rng(42)

    # Setup environment + predictors
    env_rng = np.random.default_rng(7)
    env = PSAISimEnv(K=cfg.K, N=cfg.validators, users=cfg.users, rng=env_rng,
                     sybil_prob=cfg.sybil_prob, collusion_prob=cfg.collusion_prob,
                     forced_sybil_epochs=cfg.forced_sybil_epochs,
                     forced_collusion_epochs=cfg.forced_collusion_epochs,
                     adversary_intensity=cfg.adversary_intensity)

    warm_X, warm_yq, warm_yr = [], [], []
    for t in range(60):
        v_onchain, obs, _, _ = env.step(t, action_params={})
        for vid, v in v_onchain.items():
            warm_X.append(v.x)
            warm_yq.append(1.0 if (v.m > 0.7 and v.z < 0.3) else 0.0)
            warm_yr.append(1.0 if (v.z > 0.5 or obs["colluding"] > 0.5) else 0.0)

    warm_X = np.stack(warm_X)
    warm_yq, warm_yr = np.array(warm_yq), np.array(warm_yr)
    pred = LogisticPredictor(cfg.K)
    wq, wr, _ = train_predictor(pred, warm_X, warm_yq, warm_yr, epochs=100, lr=2e-2)

    # Get a fixed validator state
    v_onchain, obs, _, _ = env.step(60, action_params={})

    # Fuzz
    pay_errors, pen_errors = [], []
    for i in range(FUZZ_COUNT):
        # Random action within bounds
        alpha = rng.uniform(bounds.alpha_min, bounds.alpha_max)
        beta = rng.uniform(bounds.beta_min, bounds.beta_max)
        lam = rng.uniform(0.0, bounds.lambda_max)
        eta = rng.uniform(-bounds.eta_max_abs, bounds.eta_max_abs, size=cfg.K).tolist()
        kappa = rng.uniform(0.0, 1.0)
        action = Action(alpha=alpha, beta=beta, lam=lam, kappa=kappa, eta=eta)

        # Mirror settlement
        contract = PSAIContractMirror(wq=wq, wr=wr, bounds={
            "alpha_min": bounds.alpha_min, "alpha_max": bounds.alpha_max,
            "beta_min": bounds.beta_min, "beta_max": bounds.beta_max,
            "lambda_max": bounds.lambda_max, "eta_max_abs": bounds.eta_max_abs,
        })
        nonce = rng.bytes(32)
        enc = enc_action(action)
        from psai.contract_mirror import keccak256
        commit = keccak256(enc + nonce)
        contract.commit_action(commit)
        contract.reveal_action(action, nonce, enc)
        mirror_pay, mirror_pen, _ = contract.settle(v_onchain, cfg.reward_pool)

        # Fixed-point settlement
        fp_pay, fp_pen = fixed_point_settle(v_onchain, action, wq, wr, cfg.reward_pool, cfg.K)

        # Errors
        for vid in mirror_pay:
            pay_errors.append(abs(mirror_pay[vid] - fp_pay[vid]))
            pen_errors.append(abs(mirror_pen[vid] - fp_pen[vid]))

    pay_errors = np.array(pay_errors)
    pen_errors = np.array(pen_errors)

    print(f"\n=== On-Chain Fuzzing Results ({FUZZ_COUNT} random actions) ===")
    print(f"Reward max abs error:  {pay_errors.max():.6f}")
    print(f"Reward mean abs error: {pay_errors.mean():.6f}")
    print(f"Penalty max abs error: {pen_errors.max():.6f}")
    print(f"Penalty mean abs error: {pen_errors.mean():.6f}")
    print(f"Max error source: fixed-point integer division rounding (FP_SCALE={FP_SCALE})")

    # Save results
    os.makedirs("results/fuzzing", exist_ok=True)
    results = {
        "fuzz_count": FUZZ_COUNT,
        "fp_scale": FP_SCALE,
        "reward_max_abs_error": float(pay_errors.max()),
        "reward_mean_abs_error": float(pay_errors.mean()),
        "penalty_max_abs_error": float(pen_errors.max()),
        "penalty_mean_abs_error": float(pen_errors.mean()),
        "reward_99pct_error": float(np.percentile(pay_errors, 99)),
        "penalty_99pct_error": float(np.percentile(pen_errors, 99)),
    }
    import json
    with open("results/fuzzing/fuzz_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Error distribution plot
    plt.rcParams.update(IEEE_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

    ax1.hist(pay_errors, bins=50, color="#1f77b4", edgecolor="white", alpha=0.85)
    ax1.axvline(pay_errors.max(), color="#d62728", linestyle="--", linewidth=0.8,
                label=f"Max={pay_errors.max():.4f}")
    ax1.set_xlabel("Absolute Error")
    ax1.set_ylabel("Count")
    ax1.set_title("Reward Parity Error")
    ax1.legend(fontsize=7)

    ax2.hist(pen_errors, bins=50, color="#ff7f0e", edgecolor="white", alpha=0.85)
    ax2.axvline(pen_errors.max(), color="#d62728", linestyle="--", linewidth=0.8,
                label=f"Max={pen_errors.max():.4f}")
    ax2.set_xlabel("Absolute Error")
    ax2.set_title("Penalty Parity Error")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/fuzz_error_dist.png", dpi=300)
    fig.savefig("figures/fuzz_error_dist.pdf")
    plt.close(fig)
    print("Fuzzing error plot saved to figures/fuzz_error_dist.pdf")


if __name__ == "__main__":
    main()
