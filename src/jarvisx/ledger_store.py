import json
import os
import tempfile

from .ledger import OmegaLedger


class PersistentLedger(OmegaLedger):
    def __init__(self, path=None):
        self.path = path
        entries = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            if not isinstance(loaded, list):
                raise ValueError("Ledger file must contain a JSON array")
            entries = loaded
        super().__init__(entries)
        if not self.verify():
            raise ValueError("Ledger verification failed")

    def _persist(self):
        if not self.path:
            return
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".omega-ledger-", suffix=".json", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    self.chain,
                    handle,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def log(self, state, opcode, metadata=None):
        checkpoint = self.checkpoint()
        entry = super().log(state, opcode, metadata=metadata)
        try:
            self._persist()
        except Exception:
            super().restore(checkpoint)
            raise
        return entry

    def restore(self, checkpoint):
        super().restore(checkpoint)
        self._persist()
