"""Canonical hash-chained audit ledger."""

import hashlib
import json
import time


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class OmegaLedger:
    def __init__(self, entries=None):
        self.chain = list(entries or [])
        if self.chain and not self.verify():
            raise ValueError("ledger hash chain is invalid")

    def log(self, state, opcode, metadata=None, timestamp_ns=None):
        sequence = len(self.chain)
        previous_hash = self.chain[-1]["hash"] if self.chain else ""
        record = {
            "sequence": sequence,
            "timestamp_ns": int(time.time_ns() if timestamp_ns is None else timestamp_ns),
            "opcode": int(opcode),
            "state": state,
            "metadata": metadata or {},
            "previous_hash": previous_hash,
        }
        digest = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
        entry = dict(record, hash=digest)
        self.chain.append(entry)
        return entry

    def verify(self):
        previous_hash = ""
        for expected_sequence, entry in enumerate(self.chain):
            if entry.get("sequence") != expected_sequence:
                return False
            if entry.get("previous_hash") != previous_hash:
                return False
            record = {key: value for key, value in entry.items() if key != "hash"}
            digest = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
            if entry.get("hash") != digest:
                return False
            previous_hash = digest
        return True
