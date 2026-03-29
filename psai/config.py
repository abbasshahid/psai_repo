
from dataclasses import dataclass, field
from typing import List

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
    # Deterministic stress-test epochs (Issue #3)
    forced_sybil_epochs: List[int] = field(default_factory=lambda: [20, 50, 80, 120, 160])
    forced_collusion_epochs: List[int] = field(default_factory=lambda: [30, 60, 100, 140, 180])
    adversary_intensity: float = 0.5  # scales Sybil splits & coalition size

@dataclass
class RLConfig:
    device: str = "cpu"
    gamma: float = 0.97
    lr: float = 3e-4
    clip: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    steps_per_update: int = 64
    minibatch: int = 32
    epochs_per_update: int = 5
    # Stability improvements (Issue #1)
    gae_lambda: float = 0.95
    kappa_smooth: float = 0.5       # action smoothing penalty weight
    lr_end_factor: float = 0.1      # cosine anneal to lr * this
    grad_clip: float = 1.0

@dataclass
class ConstraintConfig:
    # CMDP constraints (Eq. 15): inflation/centralization/false slashing proxies
    Jc: int = 3
    C0: float = 0.0   # inflation proxy target (budget is exact in PSAI)
    C1: float = 0.40  # centralization metric upper bound (Herfindahl)
    C2: float = 0.10  # false slashing proxy bound

@dataclass
class AblationConfig:
    """Toggle ablation controls for mechanism components."""
    disable_rho: bool = False          # set ρ_i(t) = 0
    disable_delta: bool = False        # set δ_i(t) = 1
    disable_penalty_risk: bool = False # remove (1 + λ·ρ) from penalty
    disable_smoothing: bool = False    # disable action smoothing penalty
    disable_quadratic_gate: bool = False  # use linear gate instead of quadratic
