
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from .contract_mirror import ValidatorOnChain

@dataclass
class SimState:
    t: int
    rng: np.random.Generator

class ValidatorAgent:
    """Validator agent with stake, behavior, and latent type.

    Behavior affects:
      - QoS m_i(t) (Eq. 20)
      - Misbehavior z_i(t)
      - Feature vectors x_i(t) used in predictors (Eqs. 7–9)

    Adversaries may:
      - perturb features (Eq. 36)
      - collude (Eqs. 41–42)
      - trigger provable faults (z bursts)
    """

    def __init__(self, vid: str, K: int, stake: float, kind: str, rng: np.random.Generator):
        self.vid = vid
        self.K = K
        self.stake = float(stake)
        self.kind = kind  # "honest", "strategic", "adversary"
        self.rng = rng
        self.effort = float(rng.uniform(0.4, 0.9))
        self.riskiness = float(rng.uniform(0.0, 0.7)) if kind != "honest" else float(rng.uniform(0.0, 0.2))
        self.history = []

    def step(self, t: int, global_demand: float, network_noise: float, colluding: bool, feature_attack: bool) -> Tuple[float, float, np.ndarray]:
        miss = np.clip(0.35*(1.0-self.effort) + 0.25*global_demand + network_noise, 0.0, 1.0)
        delay = np.clip(0.25*(1.0-self.effort) + 0.35*global_demand + 0.5*network_noise, 0.0, 1.0)
        fairdev = np.clip(0.15*self.riskiness + (0.25 if colluding else 0.0) + 0.2*global_demand, 0.0, 1.0)

        # Eq. (20): m_i(t) = 1 - (nu1 Miss + nu2 Delay + nu3 FairDev)
        nu1, nu2, nu3 = 0.45, 0.35, 0.20
        m = 1.0 - (nu1*miss + nu2*delay + nu3*fairdev)
        m = float(np.clip(m, 0.0, 1.0))

        base_z = 0.05 + 0.25*self.riskiness + (0.25 if colluding else 0.0)
        if self.kind == "adversary":
            base_z += 0.15
        burst = 1.0 if self.rng.random() < (0.03 + 0.10*self.riskiness) else 0.0
        z = float(np.clip(base_z + 0.5*burst, 0.0, 1.0))

        x = np.zeros(self.K, dtype=float)
        x[0] = miss
        x[1] = delay
        x[2] = fairdev
        x[3] = z
        x[4] = global_demand
        x[5] = 1.0 if colluding else 0.0
        x[6] = 1.0 if self.kind == "adversary" else 0.0
        x[7] = self.riskiness
        x[8] = self.effort
        if self.K > 9:
            x[9] = network_noise

        # Eq. (36): x = x_ben + Δx
        if feature_attack and self.kind == "adversary":
            dx = np.zeros_like(x)
            dx[3] -= 0.15
            dx[2] -= 0.10
            x = np.clip(x + dx, 0.0, 1.0)

        self.history.append({"t": t, "m": m, "z": z, "x": x.copy()})
        return m, z, x

class UserPopulation:
    def __init__(self, users: int, rng: np.random.Generator):
        self.users = users
        self.rng = rng

    def step(self, demand_level: float) -> Tuple[float, float, float]:
        base = self.users * (0.5 + demand_level)
        congestion = demand_level
        T = float(max(base * (1.0 - 0.3*congestion), 0.0))
        L = float(np.clip(0.5*congestion + 0.1*self.rng.normal(), 0.0, 2.0))
        F = float(np.clip(1.0 - 0.4*congestion + 0.05*self.rng.normal(), 0.0, 1.0))
        return T, L, F

class PSAISimEnv:
    """Agent-based simulation environment consistent with the PSAI model."""

    def __init__(self, K: int, N: int, users: int, rng: np.random.Generator, sybil_prob: float, collusion_prob: float):
        self.K, self.N = K, N
        self.rng = rng
        self.users = UserPopulation(users, rng)
        self.sybil_prob = sybil_prob
        self.collusion_prob = collusion_prob
        self.validators: Dict[str, ValidatorAgent] = {}
        self._init_validators()

    def _init_validators(self):
        for i in range(self.N):
            r = self.rng.random()
            kind = "honest"
            if r < 0.25:
                kind = "strategic"
            if r < 0.10:
                kind = "adversary"
            stake = float(self.rng.lognormal(mean=3.0, sigma=0.6))
            self.validators[f"V{i:03d}"] = ValidatorAgent(f"V{i:03d}", self.K, stake, kind, self.rng)

    def step(self, t: int, action_params: Dict[str, float]) -> Tuple[Dict[str, ValidatorOnChain], Dict[str, float], Dict[str, float], Dict[str, np.ndarray]]:
        demand = float(np.clip(0.5 + 0.25*np.sin(2*np.pi*t/50.0) + 0.15*self.rng.normal(), 0.0, 1.5))
        network_noise = float(0.10*self.rng.normal())

        colluding_flag = self.rng.random() < self.collusion_prob
        coalition = set()
        if colluding_flag:
            ids = list(self.validators.keys())
            self.rng.shuffle(ids)
            coalition = set(ids[: int(self.rng.integers(3, 7))])

        feature_attack = self.rng.random() < 0.25

        v_onchain: Dict[str, ValidatorOnChain] = {}
        m_dict, x_dict = {}, {}
        z_dict = {}
        for vid, v in self.validators.items():
            m, z, x = v.step(t, demand, network_noise, colluding=(vid in coalition), feature_attack=feature_attack)
            v_onchain[vid] = ValidatorOnChain(stake=v.stake, x=x, m=m, z=z)
            m_dict[vid] = m
            z_dict[vid] = z
            x_dict[vid] = x

        # Eq. (38): Sybil splitting
        if self.rng.random() < self.sybil_prob:
            adv_ids = [vid for vid, v in self.validators.items() if v.kind == "adversary"]
            if adv_ids:
                target = self.rng.choice(adv_ids)
                n = int(self.rng.integers(2, 5))
                total = self.validators[target].stake
                parts = (self.rng.dirichlet(np.ones(n)) * total).astype(float)
                base_x = x_dict[target].copy()
                base_m = m_dict[target]
                base_z = z_dict[target]
                if target in v_onchain:
                    del v_onchain[target]
                for k in range(n):
                    sid = f"{target}_S{k}"
                    v_onchain[sid] = ValidatorOnChain(stake=float(parts[k]), x=base_x, m=base_m, z=base_z)
                    m_dict[sid] = base_m
                    z_dict[sid] = base_z
                    x_dict[sid] = base_x

        T, L, F = self.users.step(demand)
        obs = {
            "T": T,
            "L": L,
            "F": F,
            "A": float(np.mean(list(z_dict.values()))),
            "demand": demand,
            "colluding": float(colluding_flag),
            "feature_attack": float(feature_attack),
            "num_validators": float(len(v_onchain)),
        }
        return v_onchain, obs, m_dict, x_dict
