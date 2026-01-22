import json
import os
from .ledger import OmegaLedger

class PersistentLedger(OmegaLedger):
    def __init__(self, path="omega_ledger.json"):
        super().__init__()
        self.path = path
        if os.path.exists(path):
            with open(path) as f:
                self.chain = json.load(f)

    def log(self, state, opcode):
        super().log(state, opcode)
        with open(self.path, "w") as f:
            json.dump(self.chain, f, indent=2)
