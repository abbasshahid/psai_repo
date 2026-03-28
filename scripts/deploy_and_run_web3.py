import argparse
import os
import sys
from pathlib import Path
import csv
from datetime import datetime, timezone

import numpy as np
from web3 import Web3
from solcx import compile_standard, install_solc
from eth_account import Account
from eth_utils import to_checksum_address

# Ensure imports from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from psai.config import Bounds, SimConfig
from psai.simulation import PSAISimEnv
from psai.predictors import LogisticPredictor, train_predictor
from psai.encoding import Action  # keep Action for demo policy
from psai.utils_crypto import nonce32  # keep nonce generator

SCALE = 10**6


def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_i(x: float) -> int:
    """Scale float to int for Solidity fixed-point (1e6)."""
    return int(round(float(x) * SCALE))


def pseudo_addr(i: int) -> str:
    """Deterministic unique address per validator index (demo only)."""
    return to_checksum_address("0x" + f"{i+1:040x}")


def compile_contract(sol_path: Path):
    """Compile PSAISettlement with viaIR to avoid stack-too-deep."""
    install_solc("0.8.20")
    source = sol_path.read_text(encoding="utf-8")

    compiled = compile_standard(
        {
            "language": "Solidity",
            "sources": {sol_path.name: {"content": source}},
            "settings": {
                "viaIR": True,
                "optimizer": {"enabled": True, "runs": 200},
                "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}},
            },
        },
        solc_version="0.8.20",
    )

    contract = compiled["contracts"][sol_path.name]["PSAISettlement"]
    abi = contract["abi"]
    bytecode = contract["evm"]["bytecode"]["object"]
    return abi, bytecode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc", type=str, default="http://127.0.0.1:8545")
    ap.add_argument("--private_key", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--validators", type=int, default=20)
    ap.add_argument("--users", type=int, default=120)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--sybil_prob", type=float, default=0.15)
    ap.add_argument("--collusion_prob", type=float, default=0.10)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    sol_path = repo / "contracts" / "PSAISettlement.sol"

    # 0) Connect
    w3 = Web3(Web3.HTTPProvider(args.rpc))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to RPC at {args.rpc}")

    acct = Account.from_key(args.private_key)
    owner = Web3.to_checksum_address(acct.address)

    print("Connected:", w3.client_version)
    print("Owner:", owner)

    # -------------------------
    # TX helpers
    # -------------------------
    def _sign_and_send(tx: dict):
        signed = acct.sign_transaction(tx)
        raw = getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction")
        tx_hash = w3.eth.send_raw_transaction(raw)
        rcpt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if getattr(rcpt, "status", 1) != 1:
            raise RuntimeError(f"TX reverted. tx={rcpt.transactionHash.hex()} gasUsed={rcpt.gasUsed}")
        return rcpt

    def _send_fn(fn, preflight: bool = True):
        """Preflight with call() to catch revert reason, then send tx."""
        if preflight:
            try:
                fn.call({"from": owner})
            except Exception as e:
                raise RuntimeError(f"Preflight revert for {fn.fn_name}: {repr(e)}")

        n = w3.eth.get_transaction_count(owner, "pending")
        tx = fn.build_transaction(
            {
                "from": owner,
                "nonce": n,
                "gas": 6_000_000,
                "gasPrice": w3.eth.gas_price,
            }
        )
        return _sign_and_send(tx)

    # 1) Build simulation env
    cfg = SimConfig(
        epochs=args.epochs,
        validators=args.validators,
        users=args.users,
        K=args.K,
        sybil_prob=args.sybil_prob,
        collusion_prob=args.collusion_prob,
    )
    rng = np.random.default_rng(cfg.seed)
    env = PSAISimEnv(cfg.K, cfg.validators, cfg.users, rng, cfg.sybil_prob, cfg.collusion_prob)

    # 2) Warmup predictor weights (wq, wr)
    warm_X, warm_yq, warm_yr = [], [], []
    for t in range(40):
        v_tmp, obs0, _, _ = env.step(t, action_params={})
        for _, v in v_tmp.items():
            warm_X.append(v.x)
            warm_yq.append(1.0 if (v.m > 0.7 and v.z < 0.3) else 0.0)
            warm_yr.append(1.0 if (v.z > 0.5 or obs0["colluding"] > 0.5) else 0.0)

    warm_X = np.stack(warm_X, axis=0)
    warm_yq = np.array(warm_yq, dtype=float)
    warm_yr = np.array(warm_yr, dtype=float)

    pred = LogisticPredictor(cfg.K)
    wq, wr = train_predictor(pred, warm_X, warm_yq, warm_yr, epochs=80, lr=2e-2)

    wq_i = [int(round(x * SCALE)) for x in wq.tolist()]
    wr_i = [int(round(x * SCALE)) for x in wr.tolist()]

    # 3) Compile + deploy
    abi, bytecode = compile_contract(sol_path)
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

    deploy_nonce = w3.eth.get_transaction_count(owner, "pending")
    deploy_tx = Contract.constructor(wq_i, wr_i).build_transaction(
        {
            "from": owner,
            "nonce": deploy_nonce,
            "gas": 6_000_000,
            "gasPrice": w3.eth.gas_price,
        }
    )
    deploy_rcpt = _sign_and_send(deploy_tx)
    addr = deploy_rcpt.contractAddress
    print("Deployed PSAISettlement at:", addr)

    c = w3.eth.contract(address=addr, abi=abi)
    bounds = Bounds()

    # -------------------------
    # CSV logging setup
    # -------------------------
    out_dir = repo / "results_onchain"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "onchain_epoch_metrics.csv"

    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "timestamp_utc",
                    "contract",
                    "epoch_before",
                    "epoch_after",
                    "tx_commit",
                    "tx_reveal",
                    "tx_settle",
                    "rewardPool",
                    "totalPaid",
                    "totalSlashed",
                    "blockNumber",
                ]
            )

    def extract_settled_args(rcpt_settle):
        # Try decoding via dict(receipt)
        try:
            logs = c.events.Settled().process_receipt(dict(rcpt_settle))
            if logs:
                return logs[0]["args"]
        except Exception:
            pass
        # Fallback: decode per-log
        for lg in rcpt_settle.logs:
            try:
                d = c.events.Settled().process_log(lg)
                return d["args"]
            except Exception:
                pass
        return None

    # 4) Run epochs
    for t in range(args.epochs):
        v_onchain, obs, _, _ = env.step(t, action_params={})
        reward_pool = int(round(cfg.reward_pool))

        epoch_before = c.functions.epoch().call()

        # Clear + upsert validators
        _send_fn(c.functions.clearValidators(), preflight=True)

        for idx, (_, v) in enumerate(v_onchain.items()):
            vid_addr = pseudo_addr(idx)
            x_i = [int(round(float(xx) * SCALE)) for xx in v.x.tolist()]  # int256[]
            m_i = int(round(float(v.m) * SCALE))  # uint256
            z_i = int(round(float(v.z) * SCALE))  # uint256
            stake_i = int(round(float(v.stake)))  # uint256

            _send_fn(c.functions.upsertValidator(vid_addr, owner, stake_i, x_i, m_i, z_i), preflight=True)

        # Demo policy action
        alpha = min(bounds.alpha_max, 0.5 + 0.5 * float(obs["demand"]))
        beta = min(bounds.beta_max, 0.8 + 0.4 * float(obs["A"]))
        lam = min(bounds.lambda_max, 0.5 + 0.7 * float(obs["demand"]))
        kappa = 0.5
        eta = [0.0] * cfg.K
        a = Action(alpha=alpha, beta=beta, lam=lam, kappa=kappa, eta=eta)

        # Solidity Action tuple (scaled)
        a_sol = (
            to_i(a.alpha),
            to_i(a.beta),
            to_i(a.lam),
            to_i(a.kappa),
            [to_i(x) for x in a.eta],
        )

        # -------------------------
        # CRITICAL FIX:
        # commitment = keccak256( encodeAction(a) || nonce )
        # compute encodeAction(a) using Solidity itself to match exact encoding
        # -------------------------
        nonce_bytes = nonce32()  # 32 bytes
        if not isinstance(nonce_bytes, (bytes, bytearray)) or len(nonce_bytes) != 32:
            raise RuntimeError("nonce32() must return 32 bytes")

        action_bytes = c.functions.encodeAction(a_sol).call()  # bytes
        com = Web3.keccak(action_bytes + nonce_bytes)          # bytes32
        com32 = Web3.to_bytes(com)

        # Commit / Reveal / Settle
        rcpt_commit = _send_fn(c.functions.commitAction(com32), preflight=True)

        nonce32b = Web3.to_bytes(nonce_bytes)  # bytes32 for reveal
        rcpt_reveal = _send_fn(c.functions.revealAction(a_sol, nonce32b), preflight=True)

        # settle should no longer revert "no revealed action"
        rcpt_settle = _send_fn(c.functions.settle(reward_pool), preflight=True)

        ev = extract_settled_args(rcpt_settle)
        if ev is None:
            raise RuntimeError("Could not decode Settled event (ABI/log issue).")

        epoch_after = c.functions.epoch().call()

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    utc_ts(),
                    addr,
                    int(epoch_before),
                    int(epoch_after),
                    rcpt_commit.transactionHash.hex(),
                    rcpt_reveal.transactionHash.hex(),
                    rcpt_settle.transactionHash.hex(),
                    int(ev["rewardPool"]),
                    int(ev["totalPaid"]),
                    int(ev["totalSlashed"]),
                    int(rcpt_settle.blockNumber),
                ]
            )

        print(f"[epoch {t}] committed/revealed/settled on-chain")

    print("Done.")
    print(f"On-chain results saved to: {csv_path}")


if __name__ == "__main__":
    main()
