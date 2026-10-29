"""Read-only durable coordinator decision attestation for local shadow RPC.

Participant services check the coordinator's previously fsynced SQLite state
rather than treating possession of a shared transport HMAC key as approval.
This is *local trust in one SQLite coordinator database*, not independent
consensus, multi-host signed certificates or production authorization.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from urllib.parse import quote

from reference.canonical_shadow_tx.bridge import DOMAINS, DomainFrame, _state_hash
from reference.canonical_shadow_tx.durable import _frames
from reference.canonical_shadow_tx.multiprocess import _id

MAX_MANIFEST = 1_048_576


class DecisionAuthority:
    """Fail-closed participant proof binding txid, epoch, fence, bytes and state."""

    def __init__(self, root: Path, domain: str) -> None:
        if domain not in DOMAINS:
            raise ValueError("unknown participant domain")
        self.domain = domain
        self.path = root / "coordinator.sqlite3"

    def _record(self, txid: str) -> tuple[int, int, DomainFrame, str]:
        txid = _id(txid)
        if not self.path.is_file() or self.path.is_symlink():
            raise PermissionError("coordinator decision journal unavailable")
        uri = "file:" + quote(str(self.path.resolve()), safe="/") + "?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=2) as connection:
                row = connection.execute(
                    "SELECT epoch,fence,manifest,candidate_hash,status "
                    "FROM tx WHERE txid=?", (txid,)
                ).fetchone()
        except (sqlite3.DatabaseError, OSError) as exc:
            raise PermissionError("coordinator decision journal unreadable") from exc
        if row is None:
            raise PermissionError("no coordinator transaction record")
        epoch, token, raw, candidate_hash, status = row
        if (type(epoch) is not int or epoch < 0 or type(token) is not int
                or token <= 0 or type(status) is not str or
                type(candidate_hash) is not str or len(candidate_hash) != 64 or
                not isinstance(raw, bytes) or len(raw) > MAX_MANIFEST):
            raise PermissionError("invalid durable coordinator record")
        try:
            frames = _frames(raw)
            if _state_hash(frames) != candidate_hash:
                raise ValueError("coordinator candidate hash mismatch")
            frame = next(frame for frame in frames if frame.domain == self.domain)
        except (ValueError, TypeError, StopIteration, KeyError) as exc:
            raise PermissionError("coordinator manifest failed verification") from exc
        return epoch, token, frame, status

    def check_prepare(self, txid: str, epoch: int, token: int,
                      frame: DomainFrame) -> None:
        actual_epoch, actual_token, expected, status = self._record(txid)
        if (status not in ("PREPARING", "PREPARED")
                or type(epoch) is not int or type(token) is not int
                or (epoch, token, frame) != (actual_epoch, actual_token, expected)):
            raise PermissionError("prepare lacks matching durable coordinator intent")

    def check_commit(self, txid: str, token: int) -> None:
        actual_epoch, actual_token, expected, status = self._record(txid)
        if (status not in ("COMMIT_DECIDED", "COMPLETE")
                or type(token) is not int or token != actual_token):
            raise PermissionError("commit lacks matching durable coordinator decision")

    def check_abort(self, txid: str) -> None:
        _, _, _, status = self._record(txid)
        if status not in ("PREPARING", "PREPARED", "ABORTED"):
            raise PermissionError("cannot abort a durably decided transaction")
