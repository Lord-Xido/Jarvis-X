"""Atomic optional persistence for the Omega audit ledger."""

import json
import os
import tempfile
import threading
from pathlib import Path

from .ledger import OmegaLedger


class PersistentLedger(OmegaLedger):
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        entries = []
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                entries = json.load(handle)
        super().__init__(entries)

    def log(self, state, opcode, metadata=None, timestamp_ns=None):
        with self._lock:
            entry = super().log(
                state, opcode, metadata=metadata, timestamp_ns=timestamp_ns
            )
            if self.path:
                self._persist()
            return entry

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.chain, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
