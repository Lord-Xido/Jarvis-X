"""Durable HyperCloud job worker."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

from .codec import LosslessMultimodalCodec
from .inference import ChatBackend, backend_from_environment
from .persistence import JobRecord, SQLiteStateStore

LOGGER = logging.getLogger("jarvisx.hypercloud.worker")


class HyperCloudWorker:
    def __init__(
        self,
        state: SQLiteStateStore,
        *,
        backend: ChatBackend | None = None,
        codec: LosslessMultimodalCodec | None = None,
    ) -> None:
        self.state = state
        self.backend = backend or backend_from_environment()
        self.codec = codec or LosslessMultimodalCodec()

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

    def run_once(self) -> bool:
        job = self.state.claim_next_job()
        if job is None:
            return False
        LOGGER.info("claimed job %s operation=%s", job.id, job.operation)
        try:
            result = self.execute(job)
        except Exception as exc:  # worker boundary must record all failures
            LOGGER.exception("job %s failed", job.id)
            self.state.fail_job(job.id, f"{type(exc).__name__}: {exc}")
        else:
            self.state.complete_job(job.id, result)
            LOGGER.info("completed job %s", job.id)
        return True

    def run_forever(self, poll_seconds: float = 0.25) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        LOGGER.info("worker started backend=%s", self.backend.name)
        while True:
            processed = self.run_once()
            if not processed:
                time.sleep(poll_seconds)


def _state_from_environment() -> SQLiteStateStore:
    return SQLiteStateStore(
        os.getenv("JARVISX_STATE_DB", "/tmp/jarvisx-hypercloud/state.db"),
        os.getenv("JARVISX_OBJECT_ROOT", "/tmp/jarvisx-hypercloud/objects"),
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
    state = _state_from_environment()
    worker = HyperCloudWorker(state)
    if args.once:
        worker.run_once()
        return 0
    worker.run_forever(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
