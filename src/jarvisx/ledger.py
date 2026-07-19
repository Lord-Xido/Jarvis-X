import hashlib
import time


class OmegaLedger:
    def __init__(self):
        self.chain = []

    def log(self, state, opcode):
        payload_text = f"{time.time()}|{state}|{opcode}"
        payload_bytes = payload_text.encode()
        prev = self.chain[-1]["hash"].encode() if self.chain else b""
        entry_hash = hashlib.sha256(prev + payload_bytes).hexdigest()
        self.chain.append({"hash": entry_hash, "payload": payload_text})
