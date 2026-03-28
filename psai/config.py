
from dataclasses import dataclass

@dataclass
class Bounds:
    alpha_min: float = 0.0
    alpha_max: float = 5.0
    beta_min: float = 0.0
    beta_max: float = 5.0
    lambda_max: float = 5.0
    eta_max_abs: float = 2.0

@dataclass
class SimConfig:
    seed: int = 7
    K: int = 10
    validators: int = 30
    users: int = 200
    epochs: int = 200
    reward_pool: float = 1000.0
    epsilon_eq: float = 5.0  # epsilon threshold for Delta_i(t)
    sybil_prob: float = 0.15
    collusion_prob: float = 0.10
    manipulation_budget: float = 1.0
    drift: float = 0.01

@dataclass
class RLConfig:
    device: str = "cpu"
    gamma: float = 0.97
    lr: float = 3e-4
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    steps_per_update: int = 64
    minibatch: int = 32
    epochs_per_update: int = 5

@dataclass
class ConstraintConfig:
    # CMDP constraints (Eq. 15): inflation/centralization/false slashing proxies
    Jc: int = 3
    C0: float = 0.0   # inflation proxy target (budget is exact in PSAI)
    C1: float = 0.40  # centralization metric upper bound (Herfindahl)
    C2: float = 0.10  # false slashing proxy bound
