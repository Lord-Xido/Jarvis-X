import json
import os
from typing import Optional

from .ledger import OmegaLedger


class PersistentLedger(OmegaLedger):
    def __init__(self, path: Optional[str] = "omega_ledger.json"):
        super().__init__()
        self.path = path
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as ledger_file:
                self.chain = json.load(ledger_file)

    def log(self, state, opcode):
        super().log(state, opcode)
        if self.path:
            with open(self.path, "w", encoding="utf-8") as ledger_file:
                json.dump(self.chain, ledger_file, indent=2)
