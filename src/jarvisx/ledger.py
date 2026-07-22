import hashlib
import json


class OmegaLedger:
    """Deterministic hash-chained execution ledger."""

    def __init__(self):
        self.chain = []

    @staticmethod
    def _serialize_payload(payload):
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

    def log(self, state, opcode):
        payload = {
            "sequence": len(self.chain),
            "state": state,
            "opcode": int(opcode),
        }
        previous_hash = self.chain[-1]["hash"].encode("ascii") if self.chain else b""
        digest = hashlib.sha256(previous_hash + self._serialize_payload(payload)).hexdigest()
        entry = {"hash": digest, "payload": payload}
        self.chain.append(entry)
        return entry

    def verify(self):
        previous_hash = b""
        for sequence, entry in enumerate(self.chain):
            payload = entry.get("payload")
            if not isinstance(payload, dict) or payload.get("sequence") != sequence:
                return False
            expected = hashlib.sha256(
                previous_hash + self._serialize_payload(payload)
            ).hexdigest()
            if entry.get("hash") != expected:
                return False
            previous_hash = expected.encode("ascii")
        return True
