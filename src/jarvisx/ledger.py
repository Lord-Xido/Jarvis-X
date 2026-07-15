import hashlib
import time


class OmegaLedger:
    def __init__(self):
        self.chain = []

    def log(self, state, opcode):
        """Append a JSON-safe, hash-linked audit record.

        The hash is computed from the exact UTF-8 payload bytes, while the
        persisted payload is retained as text. This preserves the original
        audit semantics without placing raw ``bytes`` objects in the ledger
        chain, which would make the chain impossible to serialize as JSON.
        """
        payload = f"{time.time()}|{state}|{opcode}"
        payload_bytes = payload.encode("utf-8")
        prev = self.chain[-1]["hash"].encode("ascii") if self.chain else b""
        digest = hashlib.sha256(prev + payload_bytes).hexdigest()
        self.chain.append(
            {
                "hash": digest,
                "payload": payload,
                "payload_encoding": "utf-8",
            }
        )
