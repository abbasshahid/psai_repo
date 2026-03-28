
import os, hashlib
from dataclasses import dataclass

def keccak256(data: bytes) -> bytes:
    # Python doesn't ship keccak by default; we use sha3_256 as a stand-in.
    # For exact Ethereum keccak, install `pycryptodome` or `eth-hash`.
    # In this reference pipeline we keep consistency between commit and verify.
    return hashlib.sha3_256(data).digest()

def nonce32() -> bytes:
    return os.urandom(32)
