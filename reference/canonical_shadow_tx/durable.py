"""Durable *single-SQLite-store* shadow transaction reference for ADR-016.

The four stored domain snapshots are owned by this SQLite database alone.
They are NOT live VM, OS, ANN or external-service authorities. SQLite provides
atomic, crash-recoverable transactions for this one file; not distributed ACID.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Callable, Iterable

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, DomainFrame, _checked, _state_hash, encoded, sha,
)

MAX_PENDING = 32
TXID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
ZERO_HEAD = "0" * 64
FAULT = Callable[[str], None]


def _valid_id(txid: str) -> str:
    if type(txid) is not str or TXID.fullmatch(txid) is None:
        raise ValueError("invalid transaction identifier")
    return txid


def _manifest(frames: tuple[DomainFrame, ...]) -> bytes:
    return encoded([
        {"domain": frame.domain, "profile": frame.profile,
         "payload_hex": frame.payload.hex()}
        for frame in frames
    ])


def _frames(manifest: bytes) -> tuple[DomainFrame, ...]:
    try:
        obj = json.loads(manifest)
        if not isinstance(obj, list) or len(obj) != len(DOMAINS):
            raise ValueError("wrong number of domains")
        result = []
        for part in obj:
            if not isinstance(part, dict) or set(part) != {"domain", "profile", "payload_hex"}:
                raise ValueError("invalid frame manifest")
            result.append(DomainFrame(
                part["domain"], part["profile"], bytes.fromhex(part["payload_hex"])
            ))
        if _manifest(tuple(result)) != manifest:
            raise ValueError("noncanonical manifest")
        return _checked(result)
    except (UnicodeDecodeError, ValueError, TypeError, KeyError) as exc:
        raise ValueError("prepared manifest malformed or unverified") from exc


def _chain(previous: str, epoch: int, txid: str, before: str, after: str) -> str:
    record = encoded({
        "epoch": epoch, "txid": txid, "before": before,
        "after": after, "decision": "COMMITTED",
    })
    return sha(bytes.fromhex(previous) + record)


@dataclass(frozen=True)
class DurableReceipt:
    transaction_id: str
    epoch: int
    before_hash: str
    after_hash: str
    journal_parent: str
    journal_head: str
    decision: str
    domains: tuple[str, ...]
    external_evidence: str = "NOT_APPLICABLE"

    def as_dict(self) -> dict:
        return {
            "schema": "CanonicalShadowTx/DurableSQLite1",
            "transaction_id": self.transaction_id, "epoch": self.epoch,
            "before_hash": self.before_hash, "after_hash": self.after_hash,
            "journal_parent": self.journal_parent, "journal_head": self.journal_head,
            "decision": self.decision, "domains": list(self.domains),
            "external_evidence": self.external_evidence,
            "authority": "one-local-SQLite-database",
        }


class DurableShadowStore:
    """Persistent prepared candidates and atomic promotion of owned snapshots.

    BEGIN IMMEDIATE serializes writers, epoch/hash checks reject stale prepares,
    WAL+FULL synchronizes transaction commit. An unfinished SQLite transaction
    rolls back on restart. PREPARED is never automatically promoted on recovery.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) == ":memory:" or self.path.is_dir():
            raise ValueError("a durable on-disk SQLite file is required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=5, isolation_level=None)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA foreign_keys=ON")
        if self.db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
            raise RuntimeError("WAL mode unavailable")
        self.db.execute("PRAGMA synchronous=FULL")
        try:
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    epoch INTEGER NOT NULL, state_hash TEXT NOT NULL,
                    journal_head TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS current (
                    domain TEXT PRIMARY KEY, profile TEXT NOT NULL, payload BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS proposals (
                    txid TEXT PRIMARY KEY, base_epoch INTEGER NOT NULL,
                    base_hash TEXT NOT NULL, candidate_hash TEXT NOT NULL,
                    manifest BLOB NOT NULL, status TEXT NOT NULL
                    CHECK(status IN ('PREPARED', 'COMMITTED', 'ABORTED'))
                );
                CREATE TABLE IF NOT EXISTS commits (
                    epoch INTEGER PRIMARY KEY, txid TEXT NOT NULL UNIQUE,
                    before_hash TEXT NOT NULL, after_hash TEXT NOT NULL,
                    previous_head TEXT NOT NULL, journal_head TEXT NOT NULL,
                    receipt BLOB NOT NULL,
                    FOREIGN KEY(txid) REFERENCES proposals(txid)
                );
            """)
            self.db.execute(
                "INSERT OR IGNORE INTO meta(id, epoch, state_hash, journal_head) VALUES(1, 0, ?, ?)",
                (_state_hash(()), ZERO_HEAD),
            )
            if not self.verify():
                raise RuntimeError("durable shadow state/journal integrity verification failed")
        except BaseException:
            self.db.close()
            raise

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "DurableShadowStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _meta(self) -> tuple[int, str, str]:
        row = self.db.execute(
            "SELECT epoch,state_hash,journal_head FROM meta WHERE id=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("missing metadata record")
        return int(row[0]), row[1], row[2]

    def snapshot(self) -> tuple[DomainFrame, ...]:
        rows = self.db.execute(
            "SELECT domain,profile,payload FROM current ORDER BY domain"
        ).fetchall()
        return tuple(DomainFrame(domain, profile, bytes(payload))
                     for domain, profile, payload in rows)

    def epoch(self) -> int:
        return self._meta()[0]

    def pending(self) -> tuple[str, ...]:
        return tuple(row[0] for row in self.db.execute(
            "SELECT txid FROM proposals WHERE status='PREPARED' ORDER BY txid"
        ))

    def _begin(self) -> None:
        self.db.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        if self.db.in_transaction:
            self.db.execute("ROLLBACK")

    def prepare(self, txid: str, frames: Iterable[DomainFrame],
                fault: FAULT | None = None) -> str:
        txid = _valid_id(txid)
        checked = _checked(frames)
        manifest = _manifest(checked)
        candidate = _state_hash(checked)
        try:
            self._begin()
            previous = self.db.execute(
                "SELECT base_hash,candidate_hash,manifest,status FROM proposals WHERE txid=?",
                (txid,),
            ).fetchone()
            if previous is not None:
                if (previous[1], bytes(previous[2]), previous[3]) != (
                        candidate, manifest, "PREPARED"):
                    raise ValueError("transaction identifier collision or completed")
                self.db.execute("COMMIT")
                return candidate
            count = self.db.execute(
                "SELECT COUNT(*) FROM proposals WHERE status='PREPARED'"
            ).fetchone()[0]
            if count >= MAX_PENDING:
                raise RuntimeError("prepared candidate resource budget exceeded")
            epoch, base_hash, _ = self._meta()
            self.db.execute(
                "INSERT INTO proposals(txid,base_epoch,base_hash,candidate_hash,manifest,status)"
                " VALUES(?,?,?,?,?,'PREPARED')",
                (txid, epoch, base_hash, candidate, manifest),
            )
            if fault:
                fault("before_prepare_commit")
            self.db.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise
        if fault:
            fault("after_prepare_durable")
        return candidate

    def abort(self, txid: str) -> None:
        txid = _valid_id(txid)
        try:
            self._begin()
            row = self.db.execute(
                "SELECT status FROM proposals WHERE txid=?", (txid,)
            ).fetchone()
            if row is None:
                raise ValueError("unknown transaction")
            if row[0] == "COMMITTED":
                raise ValueError("a committed transaction cannot be aborted")
            if row[0] == "PREPARED":
                self.db.execute(
                    "UPDATE proposals SET status='ABORTED' WHERE txid=?", (txid,)
                )
            self.db.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise

    def commit(self, txid: str, *, approve: bool = False,
               fault: FAULT | None = None) -> DurableReceipt:
        txid = _valid_id(txid)
        if approve is not True:
            raise PermissionError("explicit local approval required")
        try:
            self._begin()
            prepared = self.db.execute(
                "SELECT base_epoch,base_hash,candidate_hash,manifest,status"
                " FROM proposals WHERE txid=?", (txid,)
            ).fetchone()
            if prepared is None:
                raise ValueError("unknown transaction")
            base_epoch, before, target, manifest, status = prepared
            if status == "COMMITTED":
                row = self.db.execute(
                    "SELECT receipt FROM commits WHERE txid=?", (txid,)
                ).fetchone()
                if row is None:
                    raise RuntimeError("missing committed receipt")
                receipt = DurableReceipt(**json.loads(row[0]))
                self.db.execute("COMMIT")
                return receipt
            if status != "PREPARED":
                raise ValueError("aborted transaction cannot commit")
            current_epoch, current_hash, current_head = self._meta()
            if base_epoch != current_epoch or before != current_hash:
                raise ValueError("stale candidate/epoch conflict")
            frames = _frames(bytes(manifest))
            if _state_hash(frames) != target:
                raise ValueError("candidate hash mismatch")
            for frame in frames:
                self.db.execute(
                    "INSERT INTO current(domain,profile,payload) VALUES(?,?,?) "
                    "ON CONFLICT(domain) DO UPDATE SET profile=excluded.profile,"
                    "payload=excluded.payload",
                    (frame.domain, frame.profile, frame.payload),
                )
                if fault:
                    fault("after_domain_" + frame.domain)
            new_epoch = current_epoch + 1
            new_head = _chain(current_head, new_epoch, txid, before, target)
            receipt = DurableReceipt(
                txid, new_epoch, before, target, current_head,
                new_head, "COMMITTED", DOMAINS,
            )
            self.db.execute(
                "INSERT INTO commits(epoch,txid,before_hash,after_hash,"
                "previous_head,journal_head,receipt) VALUES(?,?,?,?,?,?,?)",
                (new_epoch, txid, before, target, current_head,
                 new_head, encoded(receipt.__dict__)),
            )
            self.db.execute(
                "UPDATE proposals SET status='COMMITTED' WHERE txid=?", (txid,)
            )
            self.db.execute(
                "UPDATE meta SET epoch=?,state_hash=?,journal_head=? WHERE id=1",
                (new_epoch, target, new_head),
            )
            if fault:
                fault("before_sql_commit")
            self.db.execute("COMMIT")
        except BaseException:
            self._rollback()
            raise
        if fault:
            fault("after_sql_commit")
        return receipt

    def verify(self) -> bool:
        """Fail closed on truncated/corrupt state, broken lineage or metadata."""
        try:
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                return False
            epoch, current_hash, head = self._meta()
            frames = self.snapshot()
            if frames:
                _checked(frames)
            if current_hash != _state_hash(frames):
                return False
            journal = self.db.execute(
                "SELECT epoch,txid,before_hash,after_hash,previous_head,journal_head,receipt"
                " FROM commits ORDER BY epoch"
            ).fetchall()
            before = _state_hash(())
            previous_head = ZERO_HEAD
            if len(journal) != epoch:
                return False
            for index, (n, txid, stored_before, after, prev, recorded_head, raw_receipt) in enumerate(journal, 1):
                if (n != index or stored_before != before or prev != previous_head
                        or recorded_head != _chain(previous_head, n, txid, before, after)):
                    return False
                d = json.loads(raw_receipt)
                if d != DurableReceipt(
                        txid, n, before, after, previous_head,
                        recorded_head, "COMMITTED", DOMAINS,
                ).__dict__:
                    # JSON turns tuples into lists; compare canonical representation.
                    if encoded(d) != encoded(DurableReceipt(
                            txid, n, before, after, previous_head,
                            recorded_head, "COMMITTED", DOMAINS,
                    ).__dict__):
                        return False
                proposal = self.db.execute(
                    "SELECT candidate_hash,status FROM proposals WHERE txid=?",
                    (txid,),
                ).fetchone()
                if proposal != (after, "COMMITTED"):
                    return False
                before, previous_head = after, recorded_head
            return before == current_hash and previous_head == head
        except (ValueError, TypeError, KeyError, sqlite3.DatabaseError, RuntimeError):
            return False


def recovery_receipt(path: Path, txid: str) -> dict:
    with DurableShadowStore(path) as store:
        epoch, state_hash, head = store._meta()
        return {
            "schema": "CanonicalShadowTx/DurableSQLite1",
            "db_authority": "single SQLite store, not external services",
            "epoch": epoch,
            "state_sha256": state_hash,
            "journal_head": head,
            "pending": list(store.pending()),
            "verified": store.verify(),
            "transaction": txid,
            "scope": "crash-atomic promotion of four coordinator-owned snapshots only",
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--transaction-id", default="demo-001")
    args = parser.parse_args()
    print(json.dumps(recovery_receipt(args.database, args.transaction_id),
                     indent=2, sort_keys=True))
