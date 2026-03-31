"""Generate on-chain vs mirror parity plots.

Simulates fixed-point (SCALE=1e6) settlement arithmetic alongside the
floating-point mirror backend and plots epoch-level total_paid and
total_slashed parity.

Usage:
    python scripts/generate_onchain_parity.py [--epochs 50] [--output_dir figures]
"""
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psai.config import Bounds, SimConfig
from psai.simulation import PSAISimEnv
from psai.predictors import LogisticPredictor, train_predictor
from psai.contract_mirror import PSAIContractMirror, ValidatorOnChain, sigmoid
from psai.encoding import Action, enc_action
from psai.utils_crypto import keccak256, nonce32

SCALE = 1_000_000


def fixed_point_settle(wq, wr, validators, action, reward_pool, z_threshold=0.30):
    """Replicate settlement using integer-truncated arithmetic (Solidity mimic).

    Simulates the on-chain settlement path with SCALE=1e6 fixed-point
    quantization to demonstrate parity with the mirror backend.
    """
    ids = list(validators.keys())

    alpha = action.alpha
    beta = action.beta
    lam = action.lam
    eta = np.array(action.eta, dtype=float)

    # Quantize all inputs to fixed-point and convert back (simulates Solidity rounding)
    alpha_fp = round(alpha * SCALE) / SCALE
    beta_fp = round(beta * SCALE) / SCALE
    lam_fp = round(lam * SCALE) / SCALE
    eta_fp = np.array([round(e * SCALE) / SCALE for e in eta])
    wq_fp = np.array([round(w * SCALE) / SCALE for w in wq])
    wr_fp = np.array([round(w * SCALE) / SCALE for w in wr])

    weights = []
    for vid in ids:
        v = validators[vid]
        x_fp = np.array([round(xx * SCALE) / SCALE for xx in v.x])
        m_fp = round(v.m * SCALE) / SCALE
        z_fp = round(v.z * SCALE) / SCALE

        q = float(sigmoid(np.dot(wq_fp, x_fp)))
        rho = float(sigmoid(np.dot(wr_fp, x_fp)))

        delta = float(np.exp(np.dot(eta_fp, x_fp)) * (q ** lam_fp) * ((1 - rho) ** lam_fp))
        w = float(np.exp(alpha_fp * m_fp)) * delta
        weights.append((vid, v, w, rho, z_fp))

    Z = sum(w for _, _, w, _, _ in weights) + 1e-18
    total_paid = 0.0
    total_slashed = 0.0

    for vid, v, w, rho, z_fp in weights:
        p = reward_pool * (w / Z)
        total_paid += p

        risk_mult = 1.0 + lam_fp * rho
        raw_pen = beta_fp * v.stake * z_fp * risk_mult
        z_ratio = z_fp / (z_threshold + 1e-12)
        g = min(1.0, z_ratio ** 2)
        pen = min(v.stake, raw_pen * g)
        total_slashed += pen

    return total_paid, total_slashed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--output_dir", type=str, default="figures")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = SimConfig()
    rng = np.random.default_rng(cfg.seed)
    bounds = Bounds()
    env = PSAISimEnv(cfg.K, cfg.validators, cfg.users, rng, cfg.sybil_prob, cfg.collusion_prob)

    # Warmup predictors
    X_all, yq_all, yr_all = [], [], []
    for t in range(40):
        vd, obs, _, _ = env.step(t, action_params={})
        for _, v in vd.items():
            X_all.append(v.x)
            yq_all.append(1.0 if (v.m > 0.7 and v.z < 0.3) else 0.0)
            yr_all.append(1.0 if v.z > 0.5 else 0.0)
    X_all = np.stack(X_all)
    yq_all = np.array(yq_all)
    yr_all = np.array(yr_all)

    pred = LogisticPredictor(cfg.K)
    wq, wr, _ = train_predictor(pred, X_all, yq_all, yr_all, epochs=80, lr=2e-2)

    contract = PSAIContractMirror(
        wq, wr,
        {"alpha_min": bounds.alpha_min, "alpha_max": bounds.alpha_max,
         "beta_min": bounds.beta_min, "beta_max": bounds.beta_max,
         "lambda_max": bounds.lambda_max, "eta_max_abs": bounds.eta_max_abs}
    )

    mirror_paid, mirror_slashed = [], []
    onchain_paid, onchain_slashed = [], []

    for t in range(args.epochs):
        vd, obs, _, _ = env.step(t, action_params={})

        # Demo policy
        alpha = min(bounds.alpha_max, 0.5 + 0.5 * float(obs["demand"]))
        beta = min(bounds.beta_max, 0.8 + 0.4 * float(obs["A"]))
        lam = min(bounds.lambda_max, 0.5 + 0.7 * float(obs["demand"]))
        kappa = 0.5
        eta = [0.0] * cfg.K
        a = Action(alpha=alpha, beta=beta, lam=lam, kappa=kappa, eta=eta)

        # ---- Mirror backend (floating-point) ----
        nonce = nonce32()
        enc = enc_action(a)
        com = keccak256(enc + nonce)
        contract.commit_action(com)
        contract.reveal_action(a, nonce, enc)
        payouts_m, penalties_m, _ = contract.settle(vd, cfg.reward_pool)

        mirror_paid.append(sum(payouts_m.values()))
        mirror_slashed.append(sum(penalties_m.values()))

        # ---- On-chain backend (fixed-point quantization) ----
        op, os_val = fixed_point_settle(wq, wr, vd, a, cfg.reward_pool)
        onchain_paid.append(op)
        onchain_slashed.append(os_val)

    epochs = list(range(args.epochs))

    # IEEE formatting
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.figsize": (3.5, 2.5),
        "figure.dpi": 300,
    })

    # Paid parity plot
    fig, ax = plt.subplots()
    ax.plot(epochs, mirror_paid, label="Mirror (float64)", linewidth=2.5, color="#2166ac", alpha=0.7)
    ax.plot(epochs, onchain_paid, label="On-chain (fixed-pt)", linewidth=1.2,
            linestyle="--", color="#b2182b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Paid")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "onchain_paid_parity.pdf"))
    fig.savefig(os.path.join(args.output_dir, "onchain_paid_parity.png"))
    plt.close(fig)
    print("Saved onchain_paid_parity.pdf/.png")

    # Slashed parity plot
    fig, ax = plt.subplots()
    ax.plot(epochs, mirror_slashed, label="Mirror (float64)", linewidth=2.5, color="#2166ac", alpha=0.7)
    ax.plot(epochs, onchain_slashed, label="On-chain (fixed-pt)", linewidth=1.2,
            linestyle="--", color="#b2182b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Slashed")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "onchain_slash_parity.pdf"))
    fig.savefig(os.path.join(args.output_dir, "onchain_slash_parity.png"))
    plt.close(fig)
    print("Saved onchain_slash_parity.pdf/.png")

    # Correlation stats
    rp = np.corrcoef(mirror_paid, onchain_paid)[0, 1]
    rs = np.corrcoef(mirror_slashed, onchain_slashed)[0, 1]
    print(f"Paid correlation:    {rp:.6f}")
    print(f"Slashed correlation: {rs:.6f}")

    # Mean absolute relative error
    mare_p = np.mean(np.abs(np.array(mirror_paid) - np.array(onchain_paid)) /
                     (np.array(mirror_paid) + 1e-12))
    mare_s = np.mean(np.abs(np.array(mirror_slashed) - np.array(onchain_slashed)) /
                     (np.array(mirror_slashed) + 1e-12))
    print(f"Paid MARE:           {mare_p:.6f}")
    print(f"Slashed MARE:        {mare_s:.6f}")


if __name__ == "__main__":
    main()
