"""
Generate reliability diagrams (calibration plots) for the q and ρ predictors.
Uses the same warmup data as the pipeline to train, then evaluates calibration.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psai.config import SimConfig
from psai.simulation import PSAISimEnv
from psai.predictors import LogisticPredictor, train_predictor
from psai.plotting import IEEE_RC


def reliability_diagram(y_true, y_prob, n_bins=10):
    """Compute bin-wise mean predicted probability and true fraction."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    true_fractions = []
    counts = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if mask.sum() == 0:
            continue
        bin_centers.append(y_prob[mask].mean())
        true_fractions.append(y_true[mask].mean())
        counts.append(mask.sum())
    return np.array(bin_centers), np.array(true_fractions), np.array(counts)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def main():
    cfg = SimConfig(seed=7, epochs=200)
    rng = np.random.default_rng(7)

    env = PSAISimEnv(K=cfg.K, N=cfg.validators, users=cfg.users, rng=rng,
                     sybil_prob=cfg.sybil_prob, collusion_prob=cfg.collusion_prob,
                     forced_sybil_epochs=cfg.forced_sybil_epochs,
                     forced_collusion_epochs=cfg.forced_collusion_epochs,
                     adversary_intensity=cfg.adversary_intensity)

    # Collect warmup data (same as pipeline)
    warm_X, warm_yq, warm_yr = [], [], []
    for t in range(60):
        v_onchain, obs0, _, _ = env.step(t, action_params={})
        for vid, v in v_onchain.items():
            warm_X.append(v.x)
            warm_yq.append(1.0 if (v.m > 0.7 and v.z < 0.3) else 0.0)
            warm_yr.append(1.0 if (v.z > 0.5 or obs0["colluding"] > 0.5) else 0.0)

    warm_X = np.stack(warm_X, axis=0)
    warm_yq = np.array(warm_yq, dtype=float)
    warm_yr = np.array(warm_yr, dtype=float)

    pred = LogisticPredictor(cfg.K)
    wq, wr, metrics = train_predictor(pred, warm_X, warm_yq, warm_yr, epochs=100, lr=2e-2)

    # Get calibrated probabilities
    q_prob = sigmoid(warm_X @ wq) * metrics.get("tau_q", 1.0)
    r_prob = sigmoid(warm_X @ wr) * metrics.get("tau_rho", 1.0)
    # After temperature scaling the probabilities need re-sigmoid
    q_prob = sigmoid(warm_X @ wq / max(metrics.get("tau_q", 1.0), 0.01))
    r_prob = sigmoid(warm_X @ wr / max(metrics.get("tau_rho", 1.0), 0.01))

    # Reliability diagrams
    plt.rcParams.update(IEEE_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    for ax, y_true, y_prob, label, tau in [
        (ax1, warm_yq, q_prob, r"$q$ predictor", metrics.get("tau_q", 1.0)),
        (ax2, warm_yr, r_prob, r"$\rho$ predictor", metrics.get("tau_rho", 1.0)),
    ]:
        bc, tf, counts = reliability_diagram(y_true, y_prob, n_bins=10)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
        ax.bar(bc, tf, width=0.08, alpha=0.6, color="#1f77b4", label="Observed fraction")
        ax.plot(bc, tf, "o-", color="#d62728", markersize=4, linewidth=1.0, label="Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("True Fraction")
        ax.set_title(f"{label}" + r" ($\tau$=" + f"{tau:.3f})")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.legend(fontsize=6, loc="upper left")

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/calibration_diagram.png", dpi=300)
    fig.savefig("figures/calibration_diagram.pdf")
    plt.close(fig)
    print("Calibration diagram saved to figures/calibration_diagram.pdf")
    print(f"Predictor metrics: {metrics}")


if __name__ == "__main__":
    main()
