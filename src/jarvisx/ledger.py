import hashlib
import time

class OmegaLedger:
    def __init__(self):
        self.chain = []

    def log(self, state, opcode):
        payload = f"{time.time()}|{state}|{opcode}".encode()
        prev = self.chain[-1]["hash"].encode() if self.chain else b""
        h = hashlib.sha256(prev + payload).hexdigest()
        self.chain.append({"hash": h, "payload": payload})
