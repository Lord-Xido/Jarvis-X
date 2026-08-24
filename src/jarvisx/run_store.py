"""Durable run-artifact storage for bounded Jarvis-X executions."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .ledger_store import PersistentLedger

_RUN_ID = re.compile(r"^[0-9a-f]{32}$")


class RunArtifactStore:
    """Persist run summaries and Omega journals under opaque validated IDs.

    The store uses atomic file replacement for summaries. Journal durability and
    integrity checking are delegated to :class:`PersistentLedger`.
    """

    def __init__(self, root: str | os.PathLike[str] = "data/runs") -> None:
        self.root = Path(root)

    def new_run_id(self) -> str:
        return uuid.uuid4().hex

    def ledger(self, run_id: str) -> PersistentLedger:
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        return PersistentLedger(directory / "omega.json")

    def write_summary(self, run_id: str, summary: Mapping[str, Any]) -> Path:
        directory = self._directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "summary.json"
        temporary = directory / ".summary.json.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(dict(summary), output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        return target

    def read_summary(self, run_id: str) -> dict[str, Any]:
        target = self._directory(run_id) / "summary.json"
        with target.open(encoding="utf-8") as source:
            value: Any = json.load(source)
        if not isinstance(value, dict):
            raise ValueError("stored run summary must be a JSON object")
        return value

    def verify(self, run_id: str) -> dict[str, Any]:
        summary = self.read_summary(run_id)
        ledger = PersistentLedger(self._directory(run_id) / "omega.json")
        head = ledger.chain[-1]["hash"] if ledger.chain else None
        expected_head = summary.get("journal_head_hash")
        journal_verified = ledger.verify()
        head_matches_summary = head == expected_head
        return {
            "run_id": run_id,
            "verified": journal_verified and head_matches_summary,
            "journal_verified": journal_verified,
            "head_matches_summary": head_matches_summary,
            "journal_entries": len(ledger.chain),
            "journal_head_hash": head,
            "final_state_digest": summary.get("final_state_digest"),
            "latent_digest": summary.get("latent_digest"),
        }

    def _directory(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a 32-character lowercase hexadecimal identifier")
        return self.root / run_id
