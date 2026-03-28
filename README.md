# PSAI End-to-End Reference Implementation (Paper Companion)

This repository provides a **complete, reproducible** end-to-end implementation of the **Predictive Stability-Aware Incentives (PSAI)** architecture as defined in the PSAI system model for:

**“AI-driven Incentive Mechanisms on Blockchain Platforms.”**

It includes:
- On-chain Solidity contract (`contracts/PSAISettlement.sol`)
- Off-chain orchestrator (commit→reveal→settle)
- Predictive trust/risk model and training
- CMDP-based RL training loop (primal–dual PPO-style)
- Full agent-based simulation (validators, users, adversaries, Sybils, collusion)
- Integrated pipeline and automatic results (tables + plots)

> Note: The notebook environment used to generate the included example results does not run a live Ethereum node.
> Therefore, the default pipeline executes the **same settlement logic** with a deterministic Python mirror (`psai/contract_mirror.py`),
> while still providing **complete Web3 scripts** for deployment to Hardhat/Anvil/Ganache.

## Repository Structure
```
psai_repo/
  contracts/
    PSAISettlement.sol
  psai/
    __init__.py
    config.py
    utils_crypto.py
    encoding.py
    contract_mirror.py
    simulation.py
    predictors.py
    rl_engine.py
    orchestrator.py
    baselines.py
    metrics.py
    plotting.py
  scripts/
    run_pipeline.py
    deploy_and_run_web3.py
  results/
    tables/
    plots/
  requirements.txt
```

## Quickstart (simulation + deterministic settlement)
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --epochs 200 --validators 30 --sybil_prob 0.15 --collusion_prob 0.1
```

Outputs:
- `results/tables/*.csv`
- `results/plots/*.png` and `.pdf`
- `results/summary.json`

## Optional: Run on a local chain (Hardhat/Anvil)
1) Start a local node (example with Anvil):
```bash
anvil
```
2) Deploy + run:
```bash
python scripts/deploy_and_run_web3.py --rpc http://127.0.0.1:8545
```

## Reproducibility
All random sources are seeded. Action encoding `enc(a_t)` is deterministic. Commitments are computed as:
\[
\mathsf{com}_t = \mathrm{keccak256}(\mathrm{enc}(a_t)\ \Vert\ \mathsf{nonce}_t)
\]
matching Eq. (33) in the system model.

