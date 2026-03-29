
import numpy as np
from typing import Dict
from .contract_mirror import ValidatorOnChain

def baseline_proportional(validators: Dict[str, ValidatorOnChain], reward_pool: float) -> Dict[str, float]:
    """Stake-proportional payout baseline (no QoS, no penalty)."""
    ids = list(validators.keys())
    S = np.array([validators[i].stake for i in ids], dtype=float)
    S = S / (S.sum() + 1e-12)
    p = reward_pool * S
    return {ids[i]: float(p[i]) for i in range(len(ids))}

def baseline_qos(validators: Dict[str, ValidatorOnChain], reward_pool: float) -> Dict[str, float]:
    """QoS-weighted payout: proportional to stake * m_i."""
    ids = list(validators.keys())
    S = np.array([validators[i].stake for i in ids], dtype=float)
    m = np.array([validators[i].m for i in ids], dtype=float)
    w = S * (0.1 + m)
    w = w / (w.sum() + 1e-12)
    p = reward_pool * w
    return {ids[i]: float(p[i]) for i in range(len(ids))}

def baseline_fixed_slashing(validators: Dict[str, ValidatorOnChain], reward_pool: float,
                             beta_fixed: float = 1.0) -> Dict[str, float]:
    """Fixed slashing baseline: penalty = beta_fixed * stake * z.

    No RL adaptation, no risk multiplier. Returns penalties (not payouts).
    Payout is QoS-weighted minus penalty.
    """
    ids = list(validators.keys())
    S = np.array([validators[i].stake for i in ids], dtype=float)
    z = np.array([validators[i].z for i in ids], dtype=float)
    penalties = beta_fixed * S * z
    penalties = np.minimum(penalties, S)
    return {ids[i]: float(penalties[i]) for i in range(len(ids))}

def baseline_heuristic_beta(validators: Dict[str, ValidatorOnChain], reward_pool: float,
                             ema_z: float, beta_base: float = 0.5, alpha_ema: float = 0.1
                             ) -> tuple:
    """Heuristic adaptive-β controller.

    β(t) = beta_base * (1 + EMA(z_avg(t)))
    Adapts penalty severity based on exponential moving average of avg misbehavior.

    Returns: (payouts, penalties, new_ema_z)
    """
    ids = list(validators.keys())
    S = np.array([validators[i].stake for i in ids], dtype=float)
    m = np.array([validators[i].m for i in ids], dtype=float)
    z = np.array([validators[i].z for i in ids], dtype=float)

    # Update EMA of average z
    z_avg = float(np.mean(z))
    new_ema_z = alpha_ema * z_avg + (1.0 - alpha_ema) * ema_z

    # Adaptive β
    beta_t = beta_base * (1.0 + new_ema_z)

    # QoS-weighted payouts
    w = S * (0.1 + m)
    w = w / (w.sum() + 1e-12)
    p = reward_pool * w

    # Penalties
    penalties = beta_t * S * z
    penalties = np.minimum(penalties, S)

    payouts = {ids[i]: float(p[i]) for i in range(len(ids))}
    pen_dict = {ids[i]: float(penalties[i]) for i in range(len(ids))}
    return payouts, pen_dict, float(new_ema_z)
