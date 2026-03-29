
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Tuple
from .encoding import Action
from .config import Bounds, RLConfig, ConstraintConfig

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, K: int):
        super().__init__()
        hid = 128
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
        )
        self.mean = nn.Linear(hid, 4 + K)
        self.logstd = nn.Parameter(torch.zeros(4 + K))
        self.value = nn.Linear(hid, 1)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.mean(h)
        std = torch.exp(self.logstd).clamp(1e-3, 2.0)
        v = self.value(h)
        return mean, std, v

@dataclass
class Rollout:
    obs: np.ndarray
    act: np.ndarray
    logp: np.ndarray
    rew: np.ndarray
    val: np.ndarray
    costs: np.ndarray  # (T, Jc)

class PrimalDualPPO:
    r"""Primal–dual PPO-style learner for CMDP incentives (Eqs. 13–17, 34).

    Improvements over baseline:
      - GAE (Generalized Advantage Estimation) for lower-variance advantages
      - Value function clipping (PPO best practice)
      - Cosine LR schedule
      - Gradient clipping (configurable)

    Lagrangian shaping:
      A^L_t = A_t - \sum_j \mu_j \hat{C}^{(j)}_t
    with multiplier update:
      \mu_j \leftarrow \max\{0, \mu_j + \xi(\bar{c}_j - C_j)\}.
    """

    def __init__(self, obs_dim: int, K: int, bounds: Bounds, rl: RLConfig, cc: ConstraintConfig, total_updates: int = 50):
        self.obs_dim, self.K = obs_dim, K
        self.bounds = bounds
        self.rl = rl
        self.cc = cc
        self.device = torch.device(rl.device)
        self.model = PolicyNet(obs_dim, K).to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=rl.lr)

        # Cosine LR schedule: anneal from lr to lr * lr_end_factor
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(total_updates, 1), eta_min=rl.lr * rl.lr_end_factor
        )

        self.mu = torch.zeros(cc.Jc, dtype=torch.float32, device=self.device)
        self.mu_lr = 5e-3

    def _squash_to_action(self, raw: torch.Tensor) -> Action:
        r = raw.detach().cpu().numpy()
        sig = lambda u: 1.0 / (1.0 + np.exp(-u))
        alpha = self.bounds.alpha_min + (self.bounds.alpha_max - self.bounds.alpha_min) * sig(r[0])
        beta  = self.bounds.beta_min  + (self.bounds.beta_max  - self.bounds.beta_min ) * sig(r[1])
        lam   = self.bounds.lambda_max * sig(r[2])
        kappa = 1.0 * sig(r[3])
        eta = (self.bounds.eta_max_abs * np.tanh(r[4:])).tolist()
        return Action(alpha=float(alpha), beta=float(beta), lam=float(lam), kappa=float(kappa), eta=eta)

    def act(self, obs: np.ndarray) -> Tuple[Action, np.ndarray, float, float]:
        o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std, v = self.model(o)
        dist = torch.distributions.Normal(mean, std)
        a_raw = dist.sample()
        logp = dist.log_prob(a_raw).sum(dim=-1)
        action = self._squash_to_action(a_raw.squeeze(0))
        return action, a_raw.squeeze(0).detach().cpu().numpy(), float(logp.item()), float(v.item())

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns.

        GAE: A_t = sum_{l=0}^{T-t-1} (gamma*lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        T = len(rewards)
        gamma = self.rl.gamma
        lam = self.rl.gae_lambda
        advantages = np.zeros(T, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else 0.0
            delta = rewards[t] + gamma * next_val - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    def update(self, roll: Rollout):
        T = roll.rew.shape[0]
        gamma = self.rl.gamma

        # GAE advantages (replaces simple discounted returns)
        adv, rets = self._compute_gae(roll.rew, roll.val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        Jc = self.cc.Jc
        cret = np.zeros((T, Jc), dtype=float)
        lastc = np.zeros(Jc, dtype=float)
        for t in reversed(range(T)):
            lastc = roll.costs[t] + gamma * lastc
            cret[t] = lastc

        obs = torch.tensor(roll.obs, dtype=torch.float32, device=self.device)
        act = torch.tensor(roll.act, dtype=torch.float32, device=self.device)
        old_logp = torch.tensor(roll.logp, dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=self.device)
        rets_t = torch.tensor(rets, dtype=torch.float32, device=self.device)
        cret_t = torch.tensor(cret, dtype=torch.float32, device=self.device)
        old_vals = torch.tensor(roll.val, dtype=torch.float32, device=self.device)

        last_pol_loss = last_val_loss = last_ent = 0.0

        for _ in range(self.rl.epochs_per_update):
            idx = torch.randperm(T, device=self.device)
            for start in range(0, T, self.rl.minibatch):
                j = idx[start:start+self.rl.minibatch]
                mean, std, v = self.model(obs[j])
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act[j]).sum(dim=-1)
                ratio = torch.exp(logp - old_logp[j])

                lag_adv = adv_t[j] - (self.mu.unsqueeze(0) * cret_t[j]).sum(dim=1)

                surr1 = ratio * lag_adv
                surr2 = torch.clamp(ratio, 1-self.rl.clip, 1+self.rl.clip) * lag_adv
                pol_loss = -torch.min(surr1, surr2).mean()

                # Value function clipping (PPO best practice)
                v_sq = v.squeeze(-1)
                v_clipped = old_vals[j] + torch.clamp(
                    v_sq - old_vals[j], -self.rl.clip, self.rl.clip
                )
                val_loss1 = (v_sq - rets_t[j]) ** 2
                val_loss2 = (v_clipped - rets_t[j]) ** 2
                val_loss = torch.max(val_loss1, val_loss2).mean()

                ent = dist.entropy().sum(dim=-1).mean()

                loss = pol_loss + self.rl.value_coef * val_loss - self.rl.entropy_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.rl.grad_clip)
                self.opt.step()

                last_pol_loss = float(pol_loss.item())
                last_val_loss = float(val_loss.item())
                last_ent = float(ent.item())

        # Step LR schedule
        self.scheduler.step()

        C = torch.tensor([self.cc.C0, self.cc.C1, self.cc.C2], dtype=torch.float32, device=self.device)
        avg_cost = torch.tensor(roll.costs.mean(axis=0), dtype=torch.float32, device=self.device)
        self.mu = torch.clamp(self.mu + self.mu_lr*(avg_cost - C), min=0.0)

        return {
            "policy_loss": last_pol_loss,
            "value_loss": last_val_loss,
            "entropy": last_ent,
            "mu": self.mu.detach().cpu().numpy().tolist(),
            "avg_cost": avg_cost.detach().cpu().numpy().tolist(),
            "lr": self.opt.param_groups[0]["lr"],
        }
