
import numpy as np
from typing import Dict, Tuple

def herfindahl_from_dict(p: Dict[str, float]) -> float:
    vals = np.array(list(p.values()), dtype=float)
    s = vals.sum() + 1e-12
    sh = vals / s
    return float(np.sum(sh**2))

def welfare(T: float, L: float, F: float, A: float, omega_T=1e-3, omega_L=1.0, omega_F=1.0, omega_A=1.0):
    # Eq. (10)-aligned shape used in RL: ω_T log(1+T) - ω_L L + ω_F F - ω_A A
    return float(omega_T*np.log(1.0+T) - omega_L*L + omega_F*F - omega_A*A)

def deviation_gain(p_base: Dict[str,float], p_psai: Dict[str,float]) -> float:
    """Relative deviation proxy for Delta_i(t).

    Computes max_i |p_psai(i) - p_base(i)| / p_base(i),
    i.e. the maximum relative payout deviation from the baseline.
    This is comparable to the epsilon threshold in Eq. (31).
    """
    ids = set(p_psai.keys()).intersection(p_base.keys())
    if not ids:
        return 0.0
    diffs = [abs(p_psai[i] - p_base[i]) / (abs(p_base[i]) + 1e-8) for i in ids]
    return float(max(diffs))

def sybil_unprofitability_check(weights_before: float, weights_after: float) -> float:
    # Eq. (40): sum w_sybil <= w_agg; return ratio
    return float(weights_after / (weights_before + 1e-12))
