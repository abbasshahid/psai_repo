
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, brier_score_loss

class LogisticPredictor(nn.Module):
    """Predictive model for q_i(t) and rho_i(t).

    Eqs. (7–8):
      q_i(t)   = sigma(w_q^T x_i(t) / T_q)
      rho_i(t) = sigma(w_rho^T x_i(t) / T_rho)

    Temperature parameters T_q, T_rho are learned via Platt scaling
    on a held-out validation set for improved calibration.
    """

    def __init__(self, K: int):
        super().__init__()
        self.q = nn.Linear(K, 1, bias=False)
        self.r = nn.Linear(K, 1, bias=False)
        # Platt scaling temperatures (initialized to 1 = no scaling)
        self.temp_q = nn.Parameter(torch.ones(1))
        self.temp_r = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor):
        q = torch.sigmoid(self.q(x) / self.temp_q.clamp(min=0.1))
        r = torch.sigmoid(self.r(x) / self.temp_r.clamp(min=0.1))
        return q, r

    def forward_uncalibrated(self, x: torch.Tensor):
        """Forward pass without temperature scaling (for weight extraction)."""
        q = torch.sigmoid(self.q(x))
        r = torch.sigmoid(self.r(x))
        return q, r


def train_predictor(
    model: LogisticPredictor,
    X: np.ndarray,
    yq: np.ndarray,
    yr: np.ndarray,
    l2: float = 1e-3,
    lr: float = 1e-2,
    epochs: int = 50,
    device: str = "cpu",
    val_frac: float = 0.2,
    patience: int = 10,
):
    r"""Train predictor via cross-entropy + L2, matching Eq. (32).

    Improvements:
      - 80/20 train/validation split with early stopping
      - Platt/temperature scaling on validation set post-training
      - Logs AUC-ROC and Brier score for both predictors

    Eq. (32) (LaTeX):
    \[
      J_{\mathrm{pred}}(w_q,w_\rho) =
      \sum_i \mathrm{CE}(q_i, y_i^{(q)}) + \mathrm{CE}(\rho_i, y_i^{(\rho)}) + \zeta(\lVert w_q\rVert_2^2 + \lVert w_\rho\rVert_2^2).
    \]
    """
    model.to(device)

    # Train/val split
    N = X.shape[0]
    n_val = max(int(N * val_frac), 1)
    n_train = N - n_val
    idx = np.random.permutation(N)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    Xt = torch.tensor(X[train_idx], dtype=torch.float32, device=device)
    yqt = torch.tensor(yq[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    yrt = torch.tensor(yr[train_idx].reshape(-1, 1), dtype=torch.float32, device=device)

    Xv = torch.tensor(X[val_idx], dtype=torch.float32, device=device)
    yqv = torch.tensor(yq[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)
    yrv = torch.tensor(yr[val_idx].reshape(-1, 1), dtype=torch.float32, device=device)

    # Phase 1: Train weights (freeze temperatures at 1.0)
    model.temp_q.requires_grad_(False)
    model.temp_r.requires_grad_(False)
    opt = optim.Adam([model.q.weight, model.r.weight], lr=lr, weight_decay=l2)
    bce = nn.BCELoss()

    best_val_loss = float('inf')
    wait = 0
    best_state = None

    for ep in range(epochs):
        # Train step
        opt.zero_grad()
        q, r = model.forward_uncalibrated(Xt)
        loss = bce(q, yqt) + bce(r, yrt)
        loss.backward()
        opt.step()

        # Validation
        with torch.no_grad():
            qv, rv = model.forward_uncalibrated(Xv)
            val_loss = float(bce(qv, yqv) + bce(rv, yrv))

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Phase 2: Platt/temperature scaling on validation set (freeze weights, learn temps)
    model.temp_q.requires_grad_(True)
    model.temp_r.requires_grad_(True)
    for p in [model.q.weight, model.r.weight]:
        p.requires_grad_(False)

    temp_opt = optim.LBFGS([model.temp_q, model.temp_r], lr=0.01, max_iter=50)

    def temp_closure():
        temp_opt.zero_grad()
        qv, rv = model(Xv)
        loss = bce(qv, yqv) + bce(rv, yrv)
        loss.backward()
        return loss

    temp_opt.step(temp_closure)

    # Re-enable all gradients
    for p in model.parameters():
        p.requires_grad_(True)

    # Compute calibration metrics
    metrics = {}
    with torch.no_grad():
        Xa = torch.tensor(X, dtype=torch.float32, device=device)
        qa, ra = model(Xa)
        q_pred = qa.cpu().numpy().ravel()
        r_pred = ra.cpu().numpy().ravel()

        # AUC-ROC (handle edge case of single-class labels)
        try:
            metrics["auc_q"] = float(roc_auc_score(yq, q_pred))
        except ValueError:
            metrics["auc_q"] = float('nan')
        try:
            metrics["auc_rho"] = float(roc_auc_score(yr, r_pred))
        except ValueError:
            metrics["auc_rho"] = float('nan')

        metrics["brier_q"] = float(brier_score_loss(yq, q_pred))
        metrics["brier_rho"] = float(brier_score_loss(yr, r_pred))
        metrics["temp_q"] = float(model.temp_q.item())
        metrics["temp_rho"] = float(model.temp_r.item())

    with torch.no_grad():
        wq = model.q.weight.detach().cpu().numpy().reshape(-1)
        wr = model.r.weight.detach().cpu().numpy().reshape(-1)
    return wq, wr, metrics
