"""Durable local state for the operational HyperCloud reference deployment.

SQLite in WAL mode gives the single-host/container deployment transactional
persistence without introducing a service dependency. Horizontal production
scale should replace this adapter with a distributed database/object store;
the public service contract intentionally does not depend on SQLite details.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from .multimodal import MediaEnvelope, MediaKind


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

    def as_dict(self) -> dict[str, Any]:
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
        }


class SQLiteStateStore:
    """Transactional parameter, media metadata and job persistence."""

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
                digest TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                kind TEXT NOT NULL,
                content_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                object_path TEXT NOT NULL,
                created_at REAL NOT NULL
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

            CREATE INDEX IF NOT EXISTS jobs_status_created_idx
                ON jobs(status, created_at);
            """
        )
        self._connection.commit()

    def ping(self) -> bool:
        with self._lock:
            return self._connection.execute("SELECT 1").fetchone()[0] == 1

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
                INSERT INTO media(digest, namespace, kind, content_type, size_bytes, object_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(digest) DO UPDATE SET
                    namespace = excluded.namespace,
                    kind = excluded.kind,
                    content_type = excluded.content_type,
                    size_bytes = excluded.size_bytes
                """,
                (
                    digest,
                    namespace,
                    media.kind.value,
                    media.content_type,
                    media.size_bytes,
                    str(target),
                    now,
                ),
            )
            self._connection.commit()
        return self.media_record(digest) or {}

    def media_record(self, digest: str) -> dict[str, str | int | float] | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT digest, namespace, kind, content_type, size_bytes, created_at
                FROM media WHERE digest=?
                """,
                (digest,),
            ).fetchone()
        if row is None:
            return None
        return {
            "sha256": row["digest"],
            "namespace": row["namespace"],
            "kind": row["kind"],
            "content_type": row["content_type"],
            "size_bytes": int(row["size_bytes"]),
            "created_at": float(row["created_at"]),
        }

    def get_media(self, digest: str) -> MediaEnvelope | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT kind, content_type, object_path FROM media WHERE digest=?",
                (digest,),
            ).fetchone()
        if row is None:
            return None
        path = Path(row["object_path"])
        if not path.exists():
            raise FileNotFoundError(f"media object missing from object store: {digest}")
        payload = path.read_bytes()
        media = MediaEnvelope(
            kind=MediaKind(row["kind"]),
            payload=payload,
            content_type=row["content_type"],
        )
        if media.digest != digest:
            raise RuntimeError(f"media integrity check failed for {digest}")
        return media

    def media_count(self) -> int:
        with self._lock:
            return int(self._connection.execute("SELECT COUNT(*) FROM media").fetchone()[0])

    # ---- durable jobs ------------------------------------------------------
    def create_job(self, namespace: str, operation: str, payload: dict[str, Any]) -> JobRecord:
        job_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO jobs(id, namespace, operation, status, input_json, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?, ?)
                """,
                (job_id, namespace, operation, json.dumps(payload, sort_keys=True), now, now),
            )
            self._connection.commit()
        record = self.get_job(job_id)
        assert record is not None
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return None if row is None else self._job_from_row(row)

    def claim_next_job(self) -> JobRecord | None:
        """Atomically claim the oldest queued job for one worker."""
        with self._lock:
            cursor = self._connection.cursor()
            try:
                cursor.execute("BEGIN IMMEDIATE")
                row = cursor.execute(
                    "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                now = time.time()
                cursor.execute(
                    "UPDATE jobs SET status='running', updated_at=? WHERE id=? AND status='queued'",
                    (now, row["id"]),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return None
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise
        return self.get_job(str(row["id"]))

    def complete_job(self, job_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE jobs
                SET status='succeeded', result_json=?, error=NULL, updated_at=?
                WHERE id=?
                """,
                (json.dumps(result, sort_keys=True), time.time(), job_id),
            )
            self._connection.commit()

    def fail_job(self, job_id: str, error: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE jobs
                SET status='failed', error=?, updated_at=?
                WHERE id=?
                """,
                (error[:4000], time.time(), job_id),
            )
            self._connection.commit()

    def job_counts(self) -> dict[str, int]:
        counts = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
        with self._lock:
            rows = self._connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        return counts

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=str(row["id"]),
            namespace=str(row["namespace"]),
            operation=str(row["operation"]),
            status=str(row["status"]),
            input=json.loads(row["input_json"]),
            result=None if row["result_json"] is None else json.loads(row["result_json"]),
            error=None if row["error"] is None else str(row["error"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )
