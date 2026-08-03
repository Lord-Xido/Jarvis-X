from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from .ledger import OmegaLedger


class PersistentLedger(OmegaLedger):
    """Omega ledger with atomic JSON persistence and load-time verification."""

    def __init__(
        self,
        path: str | os.PathLike[str] = "omega_ledger.json",
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(clock_ns=clock_ns)
        self.path = Path(path)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as source:
                loaded: Any = json.load(source)
            if not isinstance(loaded, list):
                raise ValueError("ledger file must contain a JSON array")
            self.chain = loaded
            if not self.verify():
                raise ValueError("ledger integrity verification failed")

    def log(self, state: Mapping[str, Any], opcode: int) -> dict[str, Any]:
        checkpoint = self.checkpoint()
        entry = cast(dict[str, Any], OmegaLedger.log(self, state, opcode))
        try:
            self._persist()
        except Exception:
            OmegaLedger.restore(self, checkpoint)
            raise
        return entry

    def restore(self, checkpoint: int) -> None:
        if checkpoint == len(self.chain):
            return
        OmegaLedger.restore(self, checkpoint)
        self._persist()

    def reset(self) -> None:
        self.restore(0)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(self.chain, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.path)
