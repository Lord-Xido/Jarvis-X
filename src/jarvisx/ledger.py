import hashlib
import json


class OmegaLedger:
    def __init__(self, entries=None):
        self.chain = list(entries or [])
        self.logical_time = max(
            (int(entry.get("logical_time", 0)) for entry in self.chain),
            default=0,
        )

    @staticmethod
    def _canonical_json(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @classmethod
    def _digest(cls, previous_hash, payload):
        material = "{}|{}".format(previous_hash or "", cls._canonical_json(payload))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def checkpoint(self):
        return len(self.chain), self.logical_time

    def restore(self, checkpoint):
        length, logical_time = checkpoint
        del self.chain[int(length) :]
        self.logical_time = int(logical_time)

    def log(self, state, opcode, metadata=None):
        next_time = self.logical_time + 1
        payload = {
            "logical_time": next_time,
            "opcode": int(opcode),
            "state": {name: int(value) for name, value in sorted(state.items())},
            "metadata": dict(metadata or {}),
        }
        previous_hash = self.chain[-1]["hash"] if self.chain else None
        entry = dict(payload)
        entry["previous_hash"] = previous_hash
        entry["hash"] = self._digest(previous_hash, payload)
        self.chain.append(entry)
        self.logical_time = next_time
        return entry

    def verify(self):
        previous_hash = None
        expected_time = 1
        for entry in self.chain:
            if int(entry.get("logical_time", -1)) != expected_time:
                return False
            if entry.get("previous_hash") != previous_hash:
                return False
            payload = {
                "logical_time": int(entry["logical_time"]),
                "opcode": int(entry["opcode"]),
                "state": {name: int(value) for name, value in sorted(entry["state"].items())},
                "metadata": dict(entry.get("metadata", {})),
            }
            if entry.get("hash") != self._digest(previous_hash, payload):
                return False
            previous_hash = entry["hash"]
            expected_time += 1
        return True
