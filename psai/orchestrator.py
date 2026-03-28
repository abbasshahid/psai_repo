
import json
from dataclasses import dataclass
from typing import Dict, List
from .encoding import Action, enc_action
from .utils_crypto import keccak256, nonce32
from .contract_mirror import PSAIContractMirror, ValidatorOnChain

@dataclass
class EpochRecord:
    t: int
    action: Action
    commit: str
    nonce: str
    obs: Dict[str, float]
    payouts: Dict[str, float]
    penalties: Dict[str, float]
    aux: Dict[str, Dict[str, float]]

class PSAIOrchestrator:
    """Off-chain orchestrator implementing:
      Observation -> AI inference -> Commit -> Reveal -> Settlement
    """

    def __init__(self, contract: PSAIContractMirror):
        self.contract = contract
        self.records: List[EpochRecord] = []

    def run_epoch(self, t: int, validators: Dict[str, ValidatorOnChain], obs: Dict[str, float], ai_action: Action):
        enc = enc_action(ai_action)
        nonce = nonce32()
        com = keccak256(enc + nonce)

        self.contract.commit_action(com)
        self.contract.reveal_action(ai_action, nonce, enc)
        payouts, penalties, aux = self.contract.settle(validators, reward_pool=obs["reward_pool"])

        self.records.append(EpochRecord(
            t=t, action=ai_action, commit=com.hex(), nonce=nonce.hex(),
            obs=obs, payouts=payouts, penalties=penalties, aux=aux
        ))
        return payouts, penalties, aux

    def save(self, path: str):
        out = []
        for r in self.records:
            out.append({
                "t": r.t,
                "action": {"alpha": r.action.alpha, "beta": r.action.beta, "lam": r.action.lam, "kappa": r.action.kappa, "eta": r.action.eta},
                "commit": r.commit,
                "nonce": r.nonce,
                "obs": r.obs,
                "payouts": r.payouts,
                "penalties": r.penalties,
                "aux": r.aux
            })
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
