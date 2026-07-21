"""Persistent storage for the deterministic Omega ledger."""

import json
import os
from pathlib import Path
from typing import Mapping, Union

from .ledger import OmegaLedger


class PersistentLedger(OmegaLedger):
    def __init__(self, path: Union[str, Path] = "omega_ledger.json"):
        super().__init__()
        self.path = Path(path)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, list):
                raise ValueError("ledger file must contain a JSON list")
            self.chain = loaded

    def log(self, state: Mapping[str, int], opcode: int):
        record = super().log(state, opcode)
        temporary = self.path.with_name(self.path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(self.chain, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        return record
