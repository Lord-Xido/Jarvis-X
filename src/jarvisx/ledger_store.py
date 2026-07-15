import json
import os

from .ledger import OmegaLedger


class PersistentLedger(OmegaLedger):
    """Omega ledger persisted with an atomic replace operation."""

    def __init__(self, path="omega_ledger.json"):
        super().__init__()
        self.path = path
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as source:
                    chain = json.load(source)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("Unable to load persistent Omega ledger") from exc
            if not isinstance(chain, list):
                raise RuntimeError("Persistent Omega ledger must contain a list")
            self.chain = chain
            if not self.verify():
                raise RuntimeError("Persistent Omega ledger hash chain is invalid")

    def log(self, state, opcode):
        previous_length = len(self.chain)
        entry = super().log(state, opcode)
        temporary_path = "{}.tmp".format(self.path)

        try:
            with open(temporary_path, "w", encoding="utf-8") as destination:
                json.dump(self.chain, destination, ensure_ascii=False, indent=2)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary_path, self.path)
        except Exception:
            del self.chain[previous_length:]
            try:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
            finally:
                raise

        return entry
