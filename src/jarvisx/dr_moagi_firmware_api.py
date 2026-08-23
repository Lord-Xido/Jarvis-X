"""FastAPI control plane for verified Dr Moagi firmware images."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dr_moagi_firmware import FirmwareBootSession, FirmwareImage

app = FastAPI(
    title="Dr Moagi DMLAMBDA Firmware Service",
    version="1.0.0",
    description="Verified boot and bounded autonomic execution for 1 GiB firmware containers.",
)


class RunRequest(BaseModel):
    cycles: int = Field(default=1, ge=1, le=10_000)


class FirmwareService:
    def __init__(
        self,
        image_path: Path,
        *,
        public_key_path: Path | None = None,
        encryption_key_path: Path | None = None,
    ) -> None:
        self.image_path = image_path
        self.public_key_path = public_key_path
        self.encryption_key_path = encryption_key_path
        self.session: FirmwareBootSession | None = None

    def _keys(self) -> tuple[bytes | None, bytes | None]:
        public = self.public_key_path.read_bytes() if self.public_key_path is not None else None
        encryption = (
            self.encryption_key_path.read_bytes() if self.encryption_key_path is not None else None
        )
        return public, encryption

    def image(self) -> FirmwareImage:
        return FirmwareImage(self.image_path)

    def verify(self) -> dict[str, object]:
        public, encryption = self._keys()
        return self.image().verify(public_key=public, encryption_key=encryption).as_dict()

    def boot(self) -> dict[str, object]:
        public, encryption = self._keys()
        self.session = self.image().boot(public_key=public, encryption_key=encryption)
        return self.status()

    def run(self, cycles: int) -> dict[str, object]:
        if self.session is None:
            self.boot()
        assert self.session is not None
        report = self.session.run(cycles)
        return {"report": report.as_dict(), "status": self.status()}

    def status(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "image": str(self.image_path),
            "image_exists": self.image_path.exists(),
            "booted": self.session is not None,
        }
        if self.session is not None:
            payload.update(
                {
                    "trace_head": self.session.trace_head,
                    "metric_cells": len(self.session.metric),
                    "system": self.session.architecture.status(),
                }
            )
        return payload


def _service_from_env() -> FirmwareService:
    image = os.getenv("JARVISX_FIRMWARE_IMAGE")
    if not image:
        raise RuntimeError("JARVISX_FIRMWARE_IMAGE is not configured")
    public = os.getenv("JARVISX_FIRMWARE_PUBLIC_KEY")
    encryption = os.getenv("JARVISX_FIRMWARE_ENCRYPTION_KEY")
    return FirmwareService(
        Path(image),
        public_key_path=Path(public) if public else None,
        encryption_key_path=Path(encryption) if encryption else None,
    )


_service: FirmwareService | None = None


def _get_service() -> FirmwareService:
    global _service
    if _service is None:
        try:
            _service = _service_from_env()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _service


@app.get("/healthz")
def healthz() -> dict[str, object]:
    try:
        service = _get_service()
    except HTTPException:
        return {"status": "unconfigured", "booted": False}
    return {"status": "ok", **service.status()}


@app.get("/v1/firmware/status")
def status() -> dict[str, object]:
    return _get_service().status()


@app.get("/v1/firmware/manifest")
def manifest() -> dict[str, object]:
    return _get_service().image().manifest


@app.post("/v1/firmware/verify")
def verify() -> dict[str, object]:
    try:
        return _get_service().verify()
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/firmware/boot")
def boot() -> dict[str, object]:
    try:
        return _get_service().boot()
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/firmware/run")
def run(request: RunRequest) -> dict[str, object]:
    try:
        return _get_service().run(request.cycles)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
