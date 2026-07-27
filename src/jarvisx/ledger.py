import hashlib
import time


class OmegaLedger:
    def __init__(self):
        self.chain = []

    def log(self, state, opcode):
        """Append a JSON-serializable hash-linked ledger entry."""

        payload = "{}|{}|{}".format(time.time(), state, opcode)
        previous_hash = self.chain[-1]["hash"] if self.chain else ""
        digest_input = (previous_hash + payload).encode("utf-8")
        entry_hash = hashlib.sha256(digest_input).hexdigest()
        self.chain.append({"hash": entry_hash, "payload": payload})
