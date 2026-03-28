
import struct
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Action:
    # a_t = (alpha, beta, lambda, kappa, eta_k) (Eq. 6 / Eq. (action_def))
    alpha: float
    beta: float
    lam: float
    kappa: float
    eta: List[float]

def enc_action(a: Action) -> bytes:
    """Deterministic encoding enc(a_t) used for commit–reveal (Eq. 33).

    We encode as:
      [alpha,beta,lam,kappa] as float64 + len(eta) uint32 + eta entries float64.
    The encoding must be identical across commit and reveal.
    """
    b = bytearray()
    b += struct.pack("<4d", a.alpha, a.beta, a.lam, a.kappa)
    b += struct.pack("<I", len(a.eta))
    for x in a.eta:
        b += struct.pack("<d", float(x))
    return bytes(b)
