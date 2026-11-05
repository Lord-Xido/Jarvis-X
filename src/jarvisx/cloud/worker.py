"""Durable leased HyperCloud worker."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
import uuid
from hashlib import blake2b
from threading import Event, Thread
from typing import Any

from .codec import LosslessMultimodalCodec
from .inference import ChatBackend, backend_from_environment
from .persistence import JobRecord, SQLiteStateStore
from .routing import ShardCoordinate

LOGGER = logging.getLogger("jarvisx.hypercloud.worker")


class HyperCloudWorker:
    """A registered worker that claims and renews expiring ownership leases."""

    def __init__(
        self,
        state: SQLiteStateStore,
        *,
        backend: ChatBackend | None = None,
        codec: LosslessMultimodalCodec | None = None,
        worker_id: str | None = None,
        coordinate: ShardCoordinate | None = None,
        capabilities: tuple[str, ...] = ("chat", "codec"),
        lease_seconds: float = 300.0,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.state = state
        self.backend = backend or backend_from_environment()
        self.codec = codec or LosslessMultimodalCodec()
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:12]}"
        self.coordinate = coordinate or ShardCoordinate(0, 0, 0)
        self.capabilities = tuple(sorted(set(capabilities)))
        self.lease_seconds = float(lease_seconds)
        self.state.register_worker(
            worker_id=self.worker_id,
            coordinate=self.coordinate,
            capabilities=self.capabilities,
            backend=self.backend.name,
            load=0.0,
        )

    @property
    def operations(self) -> tuple[str, ...]:
        operations: list[str] = []
        if "chat" in self.capabilities:
            operations.append("chat")
        if "codec" in self.capabilities:
            operations.append("codec_roundtrip")
        return tuple(operations)

    def execute(self, job: JobRecord) -> dict[str, Any]:
        if job.operation == "codec_roundtrip":
            digest = str(job.input["media_sha256"])
            media = self.state.get_media(job.namespace, digest)
            if media is None:
                raise KeyError(f"media object not found in namespace {job.namespace}: {digest}")
            codec_result = self.codec.roundtrip(media)
            return {**codec_result, "operation": job.operation}

        if job.operation == "chat":
            chat_result = self.backend.generate(
                prompt=str(job.input.get("prompt", "")),
                system=(
                    None
                    if job.input.get("system") is None
                    else str(job.input.get("system"))
                ),
            )
            return {**chat_result, "operation": job.operation}

        raise ValueError(f"unsupported job operation: {job.operation}")

    def _renew_lease_until_stopped(self, job_id: str, stop: Event) -> None:
        interval = max(0.005, min(5.0, self.lease_seconds / 3.0))
        while not stop.wait(interval):
            renewed = self.state.renew_job_lease(
                job_id,
                worker_id=self.worker_id,
                lease_seconds=self.lease_seconds,
            )
            self.state.heartbeat_worker(self.worker_id, load=1.0)
            if not renewed:
                LOGGER.warning("lease renewal lost for job %s worker=%s", job_id, self.worker_id)
                return

    def run_once(self) -> bool:
        self.state.heartbeat_worker(self.worker_id, load=0.0)
        job = self.state.claim_next_job(
            worker_id=self.worker_id,
            coordinate=self.coordinate,
            operations=self.operations,
            lease_seconds=self.lease_seconds,
        )
        if job is None:
            return False
        self.state.heartbeat_worker(self.worker_id, load=1.0)
        LOGGER.info(
            "claimed job %s operation=%s worker=%s cell=(%s,%s,%s) attempt=%s/%s",
            job.id,
            job.operation,
            self.worker_id,
            self.coordinate.x,
            self.coordinate.y,
            self.coordinate.z,
            job.attempts,
            job.max_attempts,
        )

        stop = Event()
        renewer = Thread(
            target=self._renew_lease_until_stopped,
            args=(job.id, stop),
            name=f"lease-{self.worker_id}-{job.id[:8]}",
            daemon=True,
        )
        renewer.start()
        result: dict[str, Any] | None = None
        execution_error: Exception | None = None
        try:
            result = self.execute(job)
        except Exception as exc:  # worker boundary must persist all failures
            execution_error = exc
            LOGGER.exception("job %s failed", job.id)
        finally:
            stop.set()
            renewer.join(timeout=max(0.05, min(1.0, self.lease_seconds)))

        if execution_error is not None:
            self.state.fail_job(
                job.id,
                f"{type(execution_error).__name__}: {execution_error}",
                worker_id=self.worker_id,
            )
        else:
            assert result is not None
            committed = self.state.complete_job(job.id, result, worker_id=self.worker_id)
            if not committed:
                LOGGER.warning("job %s result rejected because lease ownership changed", job.id)
            else:
                LOGGER.info("completed job %s", job.id)
        self.state.heartbeat_worker(self.worker_id, load=0.0)
        return True

    def run_forever(self, poll_seconds: float = 0.25) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        LOGGER.info(
            "worker started id=%s backend=%s cell=(%s,%s,%s) capabilities=%s",
            self.worker_id,
            self.backend.name,
            self.coordinate.x,
            self.coordinate.y,
            self.coordinate.z,
            ",".join(self.capabilities),
        )
        while True:
            processed = self.run_once()
            if not processed:
                time.sleep(poll_seconds)


def _state_from_environment() -> SQLiteStateStore:
    return SQLiteStateStore(
        os.getenv("JARVISX_STATE_DB", "/tmp/jarvisx-hypercloud/state.db"),
        os.getenv("JARVISX_OBJECT_ROOT", "/tmp/jarvisx-hypercloud/objects"),
    )


def _shape_from_environment() -> tuple[int, int, int]:
    parts = tuple(
        int(part.strip())
        for part in os.getenv("JARVISX_LATTICE_SHAPE", "16,16,16").split(",")
    )
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise RuntimeError("JARVISX_LATTICE_SHAPE must contain three positive integers")
    return parts[0], parts[1], parts[2]


def _worker_identity() -> str:
    configured = os.getenv("JARVISX_WORKER_ID", "").strip()
    if configured:
        return configured
    host = socket.gethostname().strip() or "host"
    return f"{host}-{os.getpid()}"


def _worker_coordinate(worker_id: str, shape: tuple[int, int, int]) -> ShardCoordinate:
    explicit = os.getenv("JARVISX_WORKER_COORD", "").strip()
    if explicit:
        parts = tuple(int(part.strip()) for part in explicit.split(","))
        if len(parts) != 3:
            raise RuntimeError("JARVISX_WORKER_COORD must contain x,y,z")
        coordinate = ShardCoordinate(parts[0], parts[1], parts[2])
        if not (
            0 <= coordinate.x < shape[0]
            and 0 <= coordinate.y < shape[1]
            and 0 <= coordinate.z < shape[2]
        ):
            raise RuntimeError("JARVISX_WORKER_COORD is outside JARVISX_LATTICE_SHAPE")
        return coordinate

    digest = blake2b(worker_id.encode("utf-8"), digest_size=12, person=b"jx-worker").digest()
    return ShardCoordinate(
        int.from_bytes(digest[0:4], "big") % shape[0],
        int.from_bytes(digest[4:8], "big") % shape[1],
        int.from_bytes(digest[8:12], "big") % shape[2],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jarvis-X HyperCloud worker")
    parser.add_argument("--once", action="store_true", help="process at most one queued job")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.getenv("JARVISX_WORKER_POLL_SECONDS", "0.25")),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.getenv("JARVISX_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    shape = _shape_from_environment()
    worker_id = _worker_identity()
    coordinate = _worker_coordinate(worker_id, shape)
    capabilities = tuple(
        item.strip()
        for item in os.getenv("JARVISX_WORKER_CAPABILITIES", "chat,codec").split(",")
        if item.strip()
    )
    state = _state_from_environment()
    worker = HyperCloudWorker(
        state,
        worker_id=worker_id,
        coordinate=coordinate,
        capabilities=capabilities,
        lease_seconds=float(os.getenv("JARVISX_JOB_LEASE_SECONDS", "300")),
    )
    if args.once:
        worker.run_once()
        return 0
    worker.run_forever(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
