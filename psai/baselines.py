
import numpy as np
from typing import Dict
from .contract_mirror import ValidatorOnChain

def baseline_proportional(validators: Dict[str, ValidatorOnChain], reward_pool: float) -> Dict[str, float]:
    # simple stake-proportional payout
    ids = list(validators.keys())
    S = np.array([validators[i].stake for i in ids], dtype=float)
    S = S / (S.sum() + 1e-12)
    p = reward_pool * S
    return {ids[i]: float(p[i]) for i in range(len(ids))}

def baseline_qos(validators: Dict[str, ValidatorOnChain], reward_pool: float) -> Dict[str, float]:
    # proportional to stake * m_i
    ids = list(validators.keys())
    S = np.array([validators[i].stake for i in ids], dtype=float)
    m = np.array([validators[i].m for i in ids], dtype=float)
    w = S * (0.1 + m)
    w = w / (w.sum() + 1e-12)
    p = reward_pool * w
    return {ids[i]: float(p[i]) for i in range(len(ids))}
