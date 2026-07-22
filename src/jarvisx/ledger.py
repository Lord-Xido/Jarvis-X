import hashlib
import json
import time


class OmegaLedger:
    """Hash-chained, JSON-serializable execution ledger."""

    def __init__(self):
        self.chain = []

    @staticmethod
    def _canonical_payload(payload):
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")

    def log(self, state, opcode):
        previous_hash = self.chain[-1]["hash"] if self.chain else ""
        payload = {
            "timestamp_ns": time.time_ns(),
            "state": state,
            "opcode": str(opcode),
        }
        digest = hashlib.sha256(
            previous_hash.encode("ascii") + self._canonical_payload(payload)
        ).hexdigest()
        entry = {
            "hash": digest,
            "previous_hash": previous_hash,
            "payload": payload,
        }
        self.chain.append(entry)
        return entry

    def verify(self):
        previous_hash = ""
        for entry in self.chain:
            payload = entry.get("payload")
            expected = hashlib.sha256(
                previous_hash.encode("ascii") + self._canonical_payload(payload)
            ).hexdigest()
            if entry.get("previous_hash", "") != previous_hash:
                return False
            if entry.get("hash") != expected:
                return False
            previous_hash = expected
        return True
