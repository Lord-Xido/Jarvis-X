import hashlib
import time


class OmegaLedger:
    def __init__(self):
        self.chain = []

    def log(self, state, opcode):
        payload = f"{time.time()}|{state}|{opcode}"
        previous = self.chain[-1]["hash"].encode() if self.chain else b""
        digest = hashlib.sha256(previous + payload.encode()).hexdigest()
        self.chain.append({"hash": digest, "payload": payload})
