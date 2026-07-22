import json
import os
import tempfile

from .ledger import OmegaLedger


class PersistentLedger(OmegaLedger):
    """Durable Omega ledger with atomic JSON replacement."""

    def __init__(self, path="omega_ledger.json"):
        super().__init__()
        self.path = path
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                self.chain = json.load(handle)
            if not self.verify():
                raise RuntimeError(f"Ledger integrity verification failed: {path}")

    def log(self, state, opcode):
        entry = super().log(state, opcode)
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".omega-ledger-", suffix=".json", dir=directory, text=True
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.chain, handle, indent=2, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
            raise
        return entry
