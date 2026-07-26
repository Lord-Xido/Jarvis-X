"""Deterministic hash-chained execution ledger."""

import hashlib
import json
from typing import Dict, Mapping


class OmegaLedger:
    """Append canonical JSON records to a deterministic SHA-256 chain."""

    def __init__(self):
        self.chain = []

    def log(self, state: Mapping[str, int], opcode: int) -> Dict[str, object]:
        payload = {
            "sequence": len(self.chain),
            "opcode": int(opcode),
            "state": dict(state),
        }
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        previous = self.chain[-1]["hash"].encode("ascii") if self.chain else b""
        record = {
            "hash": hashlib.sha256(previous + canonical).hexdigest(),
            "payload": payload,
        }
        self.chain.append(record)
        return record
