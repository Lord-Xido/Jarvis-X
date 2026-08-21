"""Durable state for the operational HyperCloud reference deployment.

SQLite/WAL remains the zero-dependency single-node state backend. The schema
also models worker registration, topology coordinates and expiring job leases
so the same service contract can be moved to a distributed database later.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from .multimodal import MediaEnvelope, MediaKind
from .routing import ShardCoordinate
from .topology import WorkerDescriptor, manhattan_distance


@dataclass(frozen=True)
class JobRecord:
    id: str
    namespace: str
    operation: str
    status: str
    input: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    created_at: float
    updated_at: float
    target: ShardCoordinate | None = None
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    attempts: int = 0
    max_attempts: int = 3

    def as_dict(self) -> dict[str, Any]:
        target = None
        if self.target is not None:
            target = {"x": self.target.x, "y": self.target.y, "z": self.target.z}
        return {
            "id": self.id,
            "namespace": self.namespace,
            "operation": self.operation,
            "status": self.status,
            "input": self.input,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "target": target,
            "lease_owner": self.lease_owner,
            "lease_expires_at": self.lease_expires_at,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
        }


class SQLiteStateStore:
    """Transactional parameters, media, workers and leased jobs."""

    def __init__(self, database: str, object_root: str) -> None:
        self.database = database
        self.object_root = Path(object_root)
        self.object_root.mkdir(parents=True, exist_ok=True)
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._connection = sqlite3.connect(database, check_same_thread=False, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._initialize()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS parameters (
                namespace TEXT NOT NULL,
                address TEXT NOT NULL,
                value REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(namespace, address)
            );

            CREATE TABLE IF NOT EXISTS media (
                namespace TEXT NOT NULL,
                digest TEXT NOT NULL,
                kind TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                object_path TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY(namespace, digest)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                input_json TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS workers (
                worker_id TEXT PRIMARY KEY,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                z INTEGER NOT NULL,
                capabilities_json TEXT NOT NULL,
                backend TEXT NOT NULL,
                load REAL NOT NULL,
                last_seen REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS jobs_status_created_idx
                ON jobs(status, created_at);
            CREATE INDEX IF NOT EXISTS media_digest_idx
                ON media(digest);
            CREATE INDEX IF NOT EXISTS workers_last_seen_idx
                ON workers(last_seen);
            """
        )
        # Forward-compatible migration for databases created by HyperCloud 0.2.
        self._ensure_column("jobs", "target_x", "INTEGER")
        self._ensure_column("jobs", "target_y", "INTEGER")
        self._ensure_column("jobs", "target_z", "INTEGER")
        self._ensure_column("jobs", "lease_owner", "TEXT")
        self._ensure_column("jobs", "lease_expires_at", "REAL")
        self._ensure_column("jobs", "attempts", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("jobs", "max_attempts", "INTEGER NOT NULL DEFAULT 3")
        self._connection.commit()

    def _ensure_column(self, table: str, name: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if name not in columns:
            self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    def ping(self) -> bool:
        with self._lock:
            row = self._connection.execute("SELECT 1").fetchone()
        return row is not None and int(row[0]) == 1

    # ---- sparse parameters -------------------------------------------------
    def set_parameter(self, namespace: str, address: str, value: float) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO parameters(namespace, address, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, address) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (namespace, address, float(value), now),
            )
            self._connection.commit()

    def get_parameter(self, namespace: str, address: str) -> float | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM parameters WHERE namespace=? AND address=?",
                (namespace, address),
            ).fetchone()
        return None if row is None else float(row["value"])

    def parameter_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM parameters").fetchone()[0])

    # ---- content-addressed multimedia -------------------------------------
    def put_media(self, namespace: str, media: MediaEnvelope) -> dict[str, str | int | float]:
        digest = media.digest
        target = self.object_root / digest[:2] / digest[2:4] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
            temporary.write_bytes(media.payload)
            temporary.replace(target)

        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO media(namespace, digest, kind, content_type, size_bytes, object_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, digest) DO UPDATE SET
                    kind = excluded.kind,
                    content_type = excluded.content_type,
                    size_bytes = excluded.size_bytes,
                    object_path = excluded.object_path
                """,
                (
                    namespace,
                    digest,
                    media.kind.value,
                    media.content_type,
                    media.size_bytes,
                    str(target),
                    now,
                ),
            )
            self._connection.commit()
        return self.media_record(namespace, digest) or {}

    def media_record(self, namespace: str, digest: str) -> dict[str, str | int | float] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT digest, namespace, kind, content_type, size_bytes, created_at
                FROM media WHERE namespace=? AND digest=?
                """,
                (namespace, digest),
            ).fetchone()
        if row is None:
            return None
        return {
            "sha256": str(row["digest"]),
            "namespace": str(row["namespace"]),
            "kind": str(row["kind"]),
            "content_type": str(row["content_type"]),
            "size_bytes": int(row["size_bytes"]),
            "created_at": float(row["created_at"]),
        }

    def get_media(self, namespace: str, digest: str) -> MediaEnvelope | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT kind, content_type, object_path
                FROM media WHERE namespace=? AND digest=?
                """,
                (namespace, digest),
            ).fetchone()
        if row is None:
            return None
        path = Path(str(row["object_path"]))
        if not path.exists():
            raise FileNotFoundError(f"media object missing from object store: {digest}")
        payload = path.read_bytes()
        media = MediaEnvelope(
            kind=MediaKind(str(row["kind"])),
            payload=payload,
            content_type=str(row["content_type"]),
        )
        if media.digest != digest:
            raise RuntimeError(f"media integrity check failed for {digest}")
        return media

    def media_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM media").fetchone()[0])

    # ---- worker topology ---------------------------------------------------
    def register_worker(
        self,
        *,
        worker_id: str,
        coordinate: ShardCoordinate,
        capabilities: Iterable[str],
        backend: str,
        load: float = 0.0,
    ) -> WorkerDescriptor:
        if not worker_id:
            raise ValueError("worker_id must not be empty")
        normalized = tuple(sorted(set(str(item) for item in capabilities if str(item))))
        if not normalized:
            raise ValueError("worker requires at least one capability")
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO workers(worker_id, x, y, z, capabilities_json, backend, load, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    x=excluded.x, y=excluded.y, z=excluded.z,
                    capabilities_json=excluded.capabilities_json,
                    backend=excluded.backend, load=excluded.load, last_seen=excluded.last_seen
                """,
                (
                    worker_id,
                    coordinate.x,
                    coordinate.y,
                    coordinate.z,
                    json.dumps(normalized),
                    backend,
                    min(1.0, max(0.0, float(load))),
                    now,
                ),
            )
            self._connection.commit()
        return WorkerDescriptor(worker_id, coordinate, normalized, backend, float(load), now)

    def heartbeat_worker(self, worker_id: str, *, load: float) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                "UPDATE workers SET load=?, last_seen=? WHERE worker_id=?",
                (min(1.0, max(0.0, float(load))), time.time(), worker_id),
            )
            self._connection.commit()
            return cursor.rowcount == 1

    def active_workers(self, *, ttl_seconds: float = 30.0) -> list[WorkerDescriptor]:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        cutoff = time.time() - ttl_seconds
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM workers WHERE last_seen>=? ORDER BY worker_id", (cutoff,)
            ).fetchall()
        workers: list[WorkerDescriptor] = []
        for row in rows:
            capabilities = tuple(str(item) for item in json.loads(str(row["capabilities_json"])))
            workers.append(
                WorkerDescriptor(
                    worker_id=str(row["worker_id"]),
                    coordinate=ShardCoordinate(int(row["x"]), int(row["y"]), int(row["z"])),
                    capabilities=capabilities,
                    backend=str(row["backend"]),
                    load=float(row["load"]),
                    last_seen=float(row["last_seen"]),
                )
            )
        return workers

    # ---- durable leased jobs ----------------------------------------------
    def create_job(
        self,
        namespace: str,
        operation: str,
        payload: dict[str, Any],
        *,
        target: ShardCoordinate | None = None,
        max_attempts: int = 3,
    ) -> JobRecord:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        job_id = uuid.uuid4().hex
        now = time.time()
        target_values = (None, None, None)
        if target is not None:
            target_values = (target.x, target.y, target.z)
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO jobs(
                    id, namespace, operation, status, input_json,
                    created_at, updated_at, target_x, target_y, target_z,
                    attempts, max_attempts
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    job_id,
                    namespace,
                    operation,
                    json.dumps(payload, sort_keys=True),
                    now,
                    now,
                    *target_values,
                    max_attempts,
                ),
            )
            self._connection.commit()
        record = self.get_job(job_id)
        assert record is not None
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return None if row is None else self._job_from_row(row)

    def requeue_expired_leases(self) -> int:
        now = time.time()
        with self._lock:
            requeued = self._requeue_expired_locked(now)
            self._connection.commit()
        return requeued

    def _requeue_expired_locked(self, now: float) -> int:
        failed = self._connection.execute(
            """
            UPDATE jobs
            SET status='failed', error='worker lease expired after maximum attempts',
                lease_owner=NULL, lease_expires_at=NULL, updated_at=?
            WHERE status='running' AND lease_expires_at IS NOT NULL
              AND lease_expires_at<? AND attempts>=max_attempts
            """,
            (now, now),
        ).rowcount
        requeued = self._connection.execute(
            """
            UPDATE jobs
            SET status='queued', lease_owner=NULL, lease_expires_at=NULL, updated_at=?
            WHERE status='running' AND lease_expires_at IS NOT NULL
              AND lease_expires_at<? AND attempts<max_attempts
            """,
            (now, now),
        ).rowcount
        return max(0, failed) + max(0, requeued)

    def claim_next_job(
        self,
        *,
        worker_id: str,
        coordinate: ShardCoordinate,
        operations: Iterable[str],
        lease_seconds: float = 30.0,
    ) -> JobRecord | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        allowed = tuple(sorted(set(str(item) for item in operations if str(item))))
        if not allowed:
            return None
        placeholders = ",".join("?" for _ in allowed)
        now = time.time()
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                self._requeue_expired_locked(now)
                rows = cursor.execute(
                    f"""
                    SELECT * FROM jobs
                    WHERE status='queued' AND operation IN ({placeholders})
                    ORDER BY created_at LIMIT 64
                    """,
                    allowed,
                ).fetchall()
                if not rows:
                    self._connection.commit()
                    return None

                def rank(row: sqlite3.Row) -> tuple[int, float, str]:
                    target = self._target_from_row(row)
                    distance = 0 if target is None else manhattan_distance(coordinate, target)
                    return distance, float(row["created_at"]), str(row["id"])

                selected = min(rows, key=rank)
                lease_expires = now + lease_seconds
                cursor.execute(
                    """
                    UPDATE jobs
                    SET status='running', lease_owner=?, lease_expires_at=?,
                        attempts=attempts+1, updated_at=?
                    WHERE id=? AND status='queued'
                    """,
                    (worker_id, lease_expires, now, selected["id"]),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return None
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_job(str(selected["id"]))

    def complete_job(self, job_id: str, result: dict[str, Any], *, worker_id: str | None = None) -> bool:
        query = (
            "UPDATE jobs SET status='succeeded', result_json=?, error=NULL, "
            "lease_owner=NULL, lease_expires_at=NULL, updated_at=? WHERE id=? AND status='running'"
        )
        params: tuple[object, ...] = (json.dumps(result, sort_keys=True), time.time(), job_id)
        if worker_id is not None:
            query += " AND lease_owner=?"
            params = (*params, worker_id)
        with self._lock:
            cursor = self._connection.execute(query, params)
            self._connection.commit()
            return cursor.rowcount == 1

    def fail_job(self, job_id: str, error: str, *, worker_id: str | None = None) -> bool:
        query = (
            "UPDATE jobs SET status='failed', error=?, lease_owner=NULL, "
            "lease_expires_at=NULL, updated_at=? WHERE id=? AND status='running'"
        )
        params: tuple[object, ...] = (error[:4000], time.time(), job_id)
        if worker_id is not None:
            query += " AND lease_owner=?"
            params = (*params, worker_id)
        with self._lock:
            cursor = self._connection.execute(query, params)
            self._connection.commit()
            return cursor.rowcount == 1

    def job_counts(self) -> dict[str, int]:
        self.requeue_expired_leases()
        counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        return counts

    @staticmethod
    def _target_from_row(row: sqlite3.Row) -> ShardCoordinate | None:
        if row["target_x"] is None or row["target_y"] is None or row["target_z"] is None:
            return None
        return ShardCoordinate(int(row["target_x"]), int(row["target_y"]), int(row["target_z"]))

    @classmethod
    def _job_from_row(cls, row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=str(row["id"]),
            namespace=str(row["namespace"]),
            operation=str(row["operation"]),
            status=str(row["status"]),
            input=json.loads(str(row["input_json"])),
            result=None if row["result_json"] is None else json.loads(str(row["result_json"])),
            error=None if row["error"] is None else str(row["error"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            target=cls._target_from_row(row),
            lease_owner=None if row["lease_owner"] is None else str(row["lease_owner"]),
            lease_expires_at=(
                None if row["lease_expires_at"] is None else float(row["lease_expires_at"])
            ),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
        )
