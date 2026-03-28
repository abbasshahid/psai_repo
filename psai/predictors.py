
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LogisticPredictor(nn.Module):
    """Predictive model for q_i(t) and rho_i(t).

    Eqs. (7–8):
      q_i(t)   = sigma(w_q^T x_i(t))
      rho_i(t) = sigma(w_rho^T x_i(t))
    """

    def __init__(self, K: int):
        super().__init__()
        self.q = nn.Linear(K, 1, bias=False)
        self.r = nn.Linear(K, 1, bias=False)

    def forward(self, x: torch.Tensor):
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
):
    """Train predictor via cross-entropy + L2, matching Eq. (32).

    Eq. (32) (LaTeX):
    \[
      J_{\mathrm{pred}}(w_q,w_\rho) =
      \sum_i \mathrm{CE}(q_i, y_i^{(q)}) + \mathrm{CE}(\rho_i, y_i^{(\rho)}) + \zeta(\lVert w_q\rVert_2^2 + \lVert w_\rho\rVert_2^2).
    \]
    """
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    bce = nn.BCELoss()

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yqt = torch.tensor(yq.reshape(-1, 1), dtype=torch.float32, device=device)
    yrt = torch.tensor(yr.reshape(-1, 1), dtype=torch.float32, device=device)

    for _ in range(epochs):
        opt.zero_grad()
        q, r = model(Xt)
        loss = bce(q, yqt) + bce(r, yrt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        wq = model.q.weight.detach().cpu().numpy().reshape(-1)
        wr = model.r.weight.detach().cpu().numpy().reshape(-1)
    return wq, wr
