
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from .encoding import Action
from .utils_crypto import keccak256
from .config import AblationConfig

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

@dataclass
class ValidatorOnChain:
    stake: float
    x: np.ndarray
    m: float
    z: float

class PSAIContractMirror:
    """Deterministic Python mirror of PSAISettlement.sol.

    Implements:
      - Commit–reveal verification (Eq. 33)
      - Predictors (Eqs. 7–8), delta (Eq. 9)
      - Reward kernel (Eq. 26), budget balance (Eq. 28)
      - Penalty function (Eq. 27) with stake bound
      - Soft penalty gating: penalty *= min(1, z / z_threshold) to reduce false slashing
      - Ablation controls for ρ, δ, penalty-risk multiplier
    """

    def __init__(self, wq: np.ndarray, wr: np.ndarray, bounds: Dict[str, float],
                 ablation: Optional[AblationConfig] = None,
                 z_threshold: float = 0.30):
        self.wq = wq.astype(float).copy()
        self.wr = wr.astype(float).copy()
        self.bounds = bounds
        self.ablation = ablation or AblationConfig()
        self.z_threshold = z_threshold
        self.epoch = 0
        self._commit: bytes | None = None
        self._revealed_action: Action | None = None

    def commit_action(self, commit_hash: bytes):
        if self._commit is not None:
            raise RuntimeError("commit already set")
        self._commit = commit_hash

    def reveal_action(self, action: Action, nonce: bytes, enc: bytes):
        if self._commit is None:
            raise RuntimeError("no commit")
        com = keccak256(enc + nonce)
        if com != self._commit:
            raise RuntimeError("bad reveal / commitment mismatch")

        a = Action(
            alpha=float(np.clip(action.alpha, self.bounds["alpha_min"], self.bounds["alpha_max"])),
            beta=float(np.clip(action.beta, self.bounds["beta_min"], self.bounds["beta_max"])),
            lam=float(np.clip(action.lam, 0.0, self.bounds["lambda_max"])),
            kappa=float(action.kappa),
            eta=[float(np.clip(e, -self.bounds["eta_max_abs"], self.bounds["eta_max_abs"])) for e in action.eta],
        )
        self._revealed_action = a
        self._commit = None

    def settle(self, validators: Dict[str, ValidatorOnChain], reward_pool: float):
        if self._revealed_action is None:
            raise RuntimeError("no revealed action")
        a = self._revealed_action
        K = self.wq.shape[0]
        eta = np.array(a.eta, dtype=float)
        if eta.shape[0] != K:
            raise ValueError("eta length mismatch")

        ids = list(validators.keys())
        X = np.stack([validators[i].x for i in ids], axis=0)
        m = np.array([validators[i].m for i in ids], dtype=float)
        z = np.array([validators[i].z for i in ids], dtype=float)
        S = np.array([validators[i].stake for i in ids], dtype=float)

        # Eqs. 7–8
        q = sigmoid(X @ self.wq)
        rho = sigmoid(X @ self.wr)

        # Ablation: disable rho
        if self.ablation.disable_rho:
            rho = np.zeros_like(rho)

        # Eq. 9
        if self.ablation.disable_delta:
            delta = np.ones(len(ids), dtype=float)
        else:
            delta = np.exp(X @ eta) * (q ** a.lam) * ((1.0 - rho) ** a.lam)

        # Eq. 39
        w = np.exp(a.alpha * m) * delta
        Z = float(np.sum(w)) + 1e-12

        # Eq. 26
        p = reward_pool * (w / Z)

        # Eq. 27 — penalty with soft z-gating for false-slash reduction
        if self.ablation.disable_penalty_risk:
            risk_mult = np.ones_like(rho)
        else:
            risk_mult = 1.0 + a.lam * rho

        l = a.beta * S * z * risk_mult

        # Penalty gate: quadratic by default, linear if ablation flag set
        if self.ablation.disable_quadratic_gate:
            z_gate = np.clip(z / self.z_threshold, 0.0, 1.0)  # linear gate
        else:
            z_gate = np.clip((z / self.z_threshold) ** 2, 0.0, 1.0)  # quadratic gate
        l = l * z_gate

        l = np.minimum(l, S)

        payouts = {ids[i]: float(p[i]) for i in range(len(ids))}
        penalties = {ids[i]: float(l[i]) for i in range(len(ids))}
        aux = {ids[i]: {"q": float(q[i]), "rho": float(rho[i]), "delta": float(delta[i]), "w": float(w[i])} for i in range(len(ids))}

        self.epoch += 1
        return payouts, penalties, aux
