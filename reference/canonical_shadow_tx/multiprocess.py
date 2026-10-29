"""ADR-016: independent-file participant recovery laboratory, not distributed ACID.

Four *separately persisted* SQLite participants own shadow snapshots, with a
fifth SQLite database holding the coordinator's durable COMMIT decision.
After COMMIT_DECIDED, deterministic recovery eventually applies every shadow
participant. Direct participant reads CAN see partially applied commits.
No live VM/OS/ANN, network actions, or production authority is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Callable, Iterable

from reference.canonical_shadow_tx.bridge import (
    DOMAINS, DomainFrame, _checked, _state_hash, encoded, sha,
)
from reference.canonical_shadow_tx.durable import _frames, _manifest

Fault = Callable[[str], None]
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
ACTIVE = ("PREPARING", "PREPARED", "COMMIT_DECIDED")


def _id(value: str) -> str:
    if type(value) is not str or not ID.fullmatch(value):
        raise ValueError("invalid transaction ID")
    return value


def _connect(path: Path, *, create: bool = True) -> sqlite3.Connection:
    if not create and not path.is_file():
        raise RuntimeError("participant database missing during recovery")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), isolation_level=None, timeout=5)
    db.execute("PRAGMA busy_timeout=5000")
    db.execute("PRAGMA foreign_keys=ON")
    if db.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() != "wal":
        db.close()
        raise RuntimeError("SQLite WAL unavailable")
    db.execute("PRAGMA synchronous=FULL")
    return db


def _begin(db: sqlite3.Connection) -> None:
    db.execute("BEGIN IMMEDIATE")


def _rollback(db: sqlite3.Connection) -> None:
    if db.in_transaction:
        db.execute("ROLLBACK")


def _digest(frame: DomainFrame | None) -> str:
    return _state_hash(()) if frame is None else sha(
        encoded((frame.domain, frame.profile, sha(frame.payload)))
    )


class Participant:
    """One independent DB, local prepare/commit/abort, monotonic fencing token."""

    def __init__(self, root: Path, domain: str, *, create: bool = True):
        if domain not in DOMAINS:
            raise ValueError("unrecognized participant")
        self.domain = domain
        self.db = _connect(root / (domain + ".sqlite3"), create=create)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(
                id INTEGER PRIMARY KEY CHECK(id=1),
                version INTEGER NOT NULL, fence INTEGER NOT NULL,
                profile TEXT NOT NULL, payload BLOB NOT NULL, digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS prepared(
                txid TEXT PRIMARY KEY, base_version INTEGER NOT NULL,
                fence INTEGER NOT NULL, profile TEXT NOT NULL,
                payload BLOB NOT NULL, digest TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('PREPARED','COMMITTED','ABORTED'))
            );
        """)
        self.db.execute(
            "INSERT OR IGNORE INTO meta VALUES(1,0,0,'',?,?)",
            (b"", _digest(None)),
        )
        if not self.verify():
            self.close()
            raise RuntimeError("participant integrity verification failed")

    def __enter__(self) -> "Participant":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def state(self) -> tuple[int, int, DomainFrame | None]:
        row = self.db.execute(
            "SELECT version,fence,profile,payload FROM meta WHERE id=1"
        ).fetchone()
        if row is None:
            raise RuntimeError("participant metadata missing")
        version, fence, profile, payload = row
        frame = DomainFrame(self.domain, profile, bytes(payload)) if profile else None
        return version, fence, frame

    def verify(self) -> bool:
        try:
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                return False
            version, fence, frame = self.state()
            row = self.db.execute("SELECT digest FROM meta WHERE id=1").fetchone()
            if row is None or row[0] != _digest(frame):
                return False
            if (type(version) is not int or version < 0 or type(fence) is not int
                    or fence < 0 or (version == 0) != (frame is None)):
                return False
            if frame is not None:
                from reference.canonical_shadow_tx.bridge import _verify_frame
                _verify_frame(frame)
            return True
        except (RuntimeError, ValueError, TypeError, KeyError, sqlite3.DatabaseError):
            return False

    def status(self, txid: str) -> str | None:
        row = self.db.execute(
            "SELECT status FROM prepared WHERE txid=?", (_id(txid),)
        ).fetchone()
        return None if row is None else row[0]

    def matches_prepared(self, txid: str, epoch: int, token: int,
                         frame: DomainFrame) -> bool:
        row = self.db.execute(
            "SELECT base_version,fence,profile,payload,digest,status "
            "FROM prepared WHERE txid=?", (_id(txid),)
        ).fetchone()
        version, fence, _ = self.state()
        return (row == (epoch, token, frame.profile, frame.payload,
                        _digest(frame), "PREPARED")
                and version == epoch and fence == token)

    def prepare(self, txid: str, epoch: int, token: int, frame: DomainFrame) -> None:
        from reference.canonical_shadow_tx.bridge import _verify_frame
        txid = _id(txid)
        if frame.domain != self.domain:
            raise ValueError("participant/frame mismatch")
        _verify_frame(frame)
        try:
            _begin(self.db)
            existing = self.db.execute(
                "SELECT base_version,fence,profile,payload,digest,status "
                "FROM prepared WHERE txid=?", (txid,)
            ).fetchone()
            if existing is not None:
                if existing != (epoch, token, frame.profile, frame.payload,
                                _digest(frame), "PREPARED"):
                    raise ValueError("participant transaction ID collision")
                self.db.execute("COMMIT")
                return
            version, current_fence, _ = self.state()
            if version != epoch or token <= current_fence:
                raise ValueError("stale participant version/fence")
            if self.db.execute(
                "SELECT COUNT(*) FROM prepared WHERE status='PREPARED'"
            ).fetchone()[0]:
                raise RuntimeError("participant already has a prepared candidate")
            self.db.execute(
                "INSERT INTO prepared VALUES(?,?,?,?,?,?,'PREPARED')",
                (txid, epoch, token, frame.profile, frame.payload, _digest(frame)),
            )
            self.db.execute("UPDATE meta SET fence=? WHERE id=1", (token,))
            self.db.execute("COMMIT")
        except BaseException:
            _rollback(self.db)
            raise

    def abort(self, txid: str) -> None:
        txid = _id(txid)
        try:
            _begin(self.db)
            status = self.status(txid)
            if status == "COMMITTED":
                raise RuntimeError("cannot abort a committed participant")
            if status == "PREPARED":
                self.db.execute(
                    "UPDATE prepared SET status='ABORTED' WHERE txid=?", (txid,)
                )
            self.db.execute("COMMIT")
        except BaseException:
            _rollback(self.db)
            raise

    def commit(self, txid: str, token: int) -> None:
        txid = _id(txid)
        try:
            _begin(self.db)
            row = self.db.execute(
                "SELECT base_version,fence,profile,payload,digest,status "
                "FROM prepared WHERE txid=?", (txid,)
            ).fetchone()
            if row is None:
                raise RuntimeError("participant lacks durable prepare")
            epoch, stored_token, profile, payload, digest, status = row
            version, current_fence, current = self.state()
            if status == "COMMITTED":
                if version != epoch + 1 or current_fence != token or (
                    current != DomainFrame(self.domain, profile, bytes(payload))
                ):
                    raise RuntimeError("committed participant state diverged")
                self.db.execute("COMMIT")
                return
            if (status != "PREPARED" or token != stored_token
                    or current_fence != token or epoch != version):
                raise RuntimeError("participant fence/version or status rejected")
            frame = DomainFrame(self.domain, profile, bytes(payload))
            from reference.canonical_shadow_tx.bridge import _verify_frame
            _verify_frame(frame)
            if _digest(frame) != digest:
                raise RuntimeError("prepared payload integrity failure")
            self.db.execute(
                "UPDATE meta SET version=?,profile=?,payload=?,digest=? WHERE id=1",
                (epoch + 1, profile, payload, digest),
            )
            self.db.execute(
                "UPDATE prepared SET status='COMMITTED' WHERE txid=?", (txid,)
            )
            self.db.execute("COMMIT")
        except BaseException:
            _rollback(self.db)
            raise


@dataclass(frozen=True)
class DecisionReceipt:
    txid: str
    epoch: int
    token: int
    state_hash: str
    status: str
    domains: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "schema": "CanonicalShadowTx/SeparateDB-Recovery1",
            "txid": self.txid, "epoch": self.epoch, "fence": self.token,
            "state_sha256": self.state_hash, "status": self.status,
            "domains": list(self.domains),
            "authority": "four independent shadow SQLite files + decision SQLite",
            "external_evidence": "NOT_APPLICABLE",
            "visibility": "coordinator-gated only; raw participant reads can be partial",
        }


class Coordinator:
    """Recoverable COMMIT decision, participant application, and replay."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.db = _connect(self.root / "coordinator.sqlite3")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(
                id INTEGER PRIMARY KEY CHECK(id=1),
                epoch INTEGER NOT NULL, fence INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tx(
                txid TEXT PRIMARY KEY, epoch INTEGER NOT NULL,
                fence INTEGER NOT NULL, manifest BLOB NOT NULL,
                candidate_hash TEXT NOT NULL, status TEXT NOT NULL
                CHECK(status IN ('PREPARING','PREPARED','COMMIT_DECIDED','COMPLETE','ABORTED')),
                receipt BLOB
            );
            CREATE TABLE IF NOT EXISTS acks(
                txid TEXT NOT NULL, domain TEXT NOT NULL,
                PRIMARY KEY(txid,domain),
                FOREIGN KEY(txid) REFERENCES tx(txid)
            );
        """)
        self.db.execute("INSERT OR IGNORE INTO meta VALUES(1,0,0)")
        self.db.execute("PRAGMA foreign_keys=ON")
        if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            self.close()
            raise RuntimeError("coordinator storage corrupt")

    def __enter__(self) -> "Coordinator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def _meta(self) -> tuple[int, int]:
        row = self.db.execute("SELECT epoch,fence FROM meta WHERE id=1").fetchone()
        if row is None:
            raise RuntimeError("missing coordinator metadata")
        return row

    def _record(self, txid: str):
        return self.db.execute(
            "SELECT epoch,fence,manifest,candidate_hash,status,receipt "
            "FROM tx WHERE txid=?", (_id(txid),)
        ).fetchone()

    def _set_status(self, txid: str, previous: str, updated: str) -> None:
        try:
            _begin(self.db)
            cursor = self.db.execute(
                "UPDATE tx SET status=? WHERE txid=? AND status=?",
                (updated, txid, previous),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("coordinator decision race or stale status")
            self.db.execute("COMMIT")
        except BaseException:
            _rollback(self.db)
            raise

    def _participant(self, domain: str, *, create: bool = False) -> Participant:
        return Participant(self.root, domain, create=create)

    def prepare(self, txid: str, frames: Iterable[DomainFrame],
                fault: Fault | None = None) -> str:
        txid = _id(txid)
        checked = _checked(frames)
        packed = _manifest(checked)
        candidate = _state_hash(checked)
        try:
            _begin(self.db)
            if self._record(txid) is not None:
                raise ValueError("transaction ID already used")
            if self.db.execute(
                "SELECT COUNT(*) FROM tx WHERE status IN ('PREPARING','PREPARED','COMMIT_DECIDED')"
            ).fetchone()[0]:
                raise RuntimeError("coordinator has an unresolved transaction")
            epoch, last_fence = self._meta()
            token = last_fence + 1
            self.db.execute("UPDATE meta SET fence=? WHERE id=1", (token,))
            self.db.execute(
                "INSERT INTO tx VALUES(?,?,?,?,?,'PREPARING',NULL)",
                (txid, epoch, token, packed, candidate),
            )
            self.db.execute("COMMIT")
        except BaseException:
            _rollback(self.db)
            raise
        if fault:
            fault("after_intent")
        for frame in checked:
            with self._participant(frame.domain, create=True) as participant:
                participant.prepare(txid, epoch, token, frame)
            if fault:
                fault("after_prepare_" + frame.domain)
        self._set_status(txid, "PREPARING", "PREPARED")
        if fault:
            fault("after_all_prepared")
        return candidate

    def approve(self, txid: str, *, approve: bool = False,
                fault: Fault | None = None) -> DecisionReceipt:
        txid = _id(txid)
        if approve is not True:
            raise PermissionError("explicit local approval required")
        row = self._record(txid)
        if row is None:
            raise ValueError("unknown transaction")
        epoch, token, manifest, digest, status, receipt = row
        if status == "COMPLETE":
            return self._receipt(receipt)
        if status == "COMMIT_DECIDED":
            return self._finish(txid, fault=fault)
        if status != "PREPARED":
            raise RuntimeError("not all participants durably prepared")
        checked = _frames(bytes(manifest))
        if _state_hash(checked) != digest or self._meta()[0] != epoch:
            raise RuntimeError("prepared manifest or coordinator epoch stale")
        for frame in checked:
            with self._participant(frame.domain) as participant:
                if not participant.matches_prepared(txid, epoch, token, frame):
                    raise RuntimeError("participant did not acknowledge valid prepare")
        if fault:
            fault("before_decision")
        self._set_status(txid, "PREPARED", "COMMIT_DECIDED")
        if fault:
            fault("after_decision")
        return self._finish(txid, fault=fault)

    @staticmethod
    def _receipt(raw: bytes | str) -> DecisionReceipt:
        d = json.loads(raw)
        if not isinstance(d, dict) or set(d) != {
            "txid", "epoch", "token", "state_hash", "status", "domains"
        }:
            raise RuntimeError("stored receipt shape invalid")
        d["domains"] = tuple(d["domains"])
        if d["domains"] != DOMAINS or d["status"] != "COMPLETE":
            raise RuntimeError("stored receipt identity invalid")
        return DecisionReceipt(**d)

    def _finish(self, txid: str, fault: Fault | None = None) -> DecisionReceipt:
        epoch, token, manifest, digest, status, receipt = self._record(txid)
        if status == "COMPLETE":
            return self._receipt(receipt)
        if status != "COMMIT_DECIDED":
            raise RuntimeError("no durable coordinator commit decision")
        checked = _frames(bytes(manifest))
        if _state_hash(checked) != digest:
            raise RuntimeError("coordinator decision manifest integrity failure")
        for frame in checked:
            with self._participant(frame.domain) as participant:
                participant.commit(txid, token)
            if fault:
                fault("after_apply_" + frame.domain)
            self.db.execute(
                "INSERT OR IGNORE INTO acks(txid,domain) VALUES(?,?)",
                (txid, frame.domain),
            )
        acked = self.db.execute(
            "SELECT domain FROM acks WHERE txid=? ORDER BY domain", (txid,)
        ).fetchall()
        if tuple(x[0] for x in acked) != DOMAINS:
            raise RuntimeError("incomplete participant application")
        result = DecisionReceipt(txid, epoch + 1, token, digest, "COMPLETE", DOMAINS)
        try:
            _begin(self.db)
            if self._record(txid)[4] != "COMMIT_DECIDED" or self._meta()[0] != epoch:
                raise RuntimeError("coordinator completion conflict")
            self.db.execute(
                "UPDATE tx SET status='COMPLETE',receipt=? WHERE txid=?",
                (encoded(result.__dict__), txid),
            )
            self.db.execute("UPDATE meta SET epoch=? WHERE id=1", (epoch + 1,))
            self.db.execute("COMMIT")
        except BaseException:
            _rollback(self.db)
            raise
        if fault:
            fault("after_complete")
        return result

    def recover(self) -> dict:
        """Abort undecided prepares; replay only an already durable decision."""
        changes: dict[str, str] = {}
        rows = self.db.execute(
            "SELECT txid,status FROM tx WHERE status IN "
            "('PREPARING','PREPARED','COMMIT_DECIDED','ABORTED') ORDER BY rowid"
        ).fetchall()
        for txid, status in rows:
            if status == "COMMIT_DECIDED":
                self._finish(txid)
                changes[txid] = "COMPLETE"
            else:
                # ABORTED is a durable *abort decision*, not proof that every
                # participant has acknowledged it. Persist the decision FIRST.
                # On a crash mid-abort, replay all ABORTED rows on next start.
                if status != "ABORTED":
                    self._set_status(txid, status, "ABORTED")
                for domain in DOMAINS:
                    path = self.root / (domain + ".sqlite3")
                    if path.exists():
                        with self._participant(domain) as participant:
                            participant.abort(txid)
                changes[txid] = "ABORTED"
        return changes

    def coordinated_read(self, txid: str) -> tuple[DomainFrame, ...]:
        """Opt-in read gate; raw participant files are NOT transaction-isolated."""
        row = self._record(txid)
        if row is None or row[4] != "COMPLETE":
            raise RuntimeError("coordinator has not completed all participants")
        checked = _frames(bytes(row[2]))
        observed: list[DomainFrame] = []
        for expected in checked:
            with self._participant(expected.domain) as participant:
                version, _, frame = participant.state()
                if version != row[0] + 1 or frame != expected or not participant.verify():
                    raise RuntimeError("participant state mismatched coordinator")
                observed.append(frame)
        if _state_hash(tuple(observed)) != row[3]:
            raise RuntimeError("coordinated read digest mismatch")
        return tuple(observed)

    def decision_status(self, txid: str) -> str | None:
        row = self._record(txid)
        return row[4] if row is not None else None

    def receipt(self, txid: str) -> dict:
        row = self._record(txid)
        if row is None:
            raise ValueError("unknown transaction")
        return {
            "schema": "CanonicalShadowTx/SeparateDB-Recovery1",
            "txid": txid, "status": row[4],
            "coordinator_epoch": self._meta()[0],
            "fence": row[1], "candidate_sha256": row[3],
            "acked_domains": [
                r[0] for r in self.db.execute(
                    "SELECT domain FROM acks WHERE txid=? ORDER BY domain", (txid,)
                )
            ],
            "complete": None if row[5] is None else self._receipt(row[5]).as_dict(),
            "external_evidence": "NOT_APPLICABLE",
            "warning": "raw participant reads may observe partially applied commits",
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("command", choices=("recover", "status"))
    parser.add_argument("--txid", default="example-001")
    args = parser.parse_args()
    with Coordinator(args.root) as coordinator:
        value = coordinator.recover() if args.command == "recover" else coordinator.receipt(args.txid)
    print(json.dumps(value, indent=2, sort_keys=True))
