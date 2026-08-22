"""FastAPI control plane for the Dr Moagi 3D OS kernel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn, cast

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .dr_moagi_field_runtime import SparseField
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field
from .dr_moagi_os_ui import DR_MOAGI_OS_HTML


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _config_from_env() -> DrMoagiOSConfig:
    state_dir_raw = os.getenv("JARVISX_STATE_DIR", "state/dr-moagi-os")
    state_dir = None if state_dir_raw.lower() == "none" else Path(state_dir_raw)
    return DrMoagiOSConfig(
        side=_int_env("JARVISX_OS_SIDE", 64),
        max_active_cells=_int_env("JARVISX_OS_MAX_ACTIVE_CELLS", 50_000),
        activation_threshold=_float_env("JARVISX_OS_THRESHOLD", 0.5),
        contraction=_float_env("JARVISX_OS_CONTRACTION", 0.08),
        attenuation=_float_env("JARVISX_OS_ATTENUATION", 0.10),
        prune_epsilon=_float_env("JARVISX_OS_PRUNE_EPSILON", 0.0),
        block_size=_int_env("JARVISX_OS_BLOCK_SIZE", 2),
        quantization=_float_env("JARVISX_OS_QUANTIZATION", 0.01),
        fixed_point_passes=_int_env("JARVISX_OS_FIXED_POINT_PASSES", 1),
        autorun_interval_seconds=_float_env("JARVISX_OS_AUTORUN_INTERVAL", 0.5),
        state_dir=state_dir,
    )


app = FastAPI(
    title="Dr Moagi 3D OS",
    version="1.0.0",
    description=(
        "Bounded sparse 3D auto-encoding/decoding operating control plane with "
        "Uint64 bit-plane execution, inward folding, verification and auto-run scheduling."
    ),
)

_kernel = DrMoagiOSKernel(_config_from_env())


class FieldCell(BaseModel):
    x: int
    y: int
    z: int
    value: float


class LoadRequest(BaseModel):
    field: list[FieldCell] = Field(min_length=1)


class RunRequest(BaseModel):
    cycles: int = Field(default=8, ge=1, le=4_096)


class AutorunRequest(BaseModel):
    interval_seconds: float = Field(default=0.5, ge=0.05, le=60.0)


def _raise_http(exc: Exception) -> NoReturn:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=409, detail=str(exc)) from exc


def _field_from_request(request: LoadRequest) -> SparseField:
    field: SparseField = {}
    for cell in request.field:
        coordinate = (cell.x, cell.y, cell.z)
        field[coordinate] = float(cell.value)
    return field


def _numeric(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _integer(value: object, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DR_MOAGI_OS_HTML)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    status = _kernel.status()
    return {
        "status": "ok",
        "system": status["system"],
        "lifecycle": status["lifecycle"],
        "loaded": status["loaded"],
        "journal_valid": status["journal_valid"],
    }


@app.post("/v1/os/boot")
def boot() -> dict[str, object]:
    try:
        return _kernel.boot(restore=True)
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/shutdown")
def shutdown() -> dict[str, object]:
    try:
        return _kernel.shutdown()
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/status")
def status() -> dict[str, object]:
    return _kernel.status()


@app.post("/v1/os/load")
def load(request: LoadRequest) -> dict[str, object]:
    try:
        if _kernel.lifecycle.value == "offline":
            _kernel.boot(restore=False)
        return _kernel.load(_field_from_request(request))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/demo")
def load_demo() -> dict[str, object]:
    try:
        if _kernel.lifecycle.value == "offline":
            _kernel.boot(restore=False)
        return _kernel.load(demo_field(_kernel.config.side))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/step")
def step() -> dict[str, object]:
    try:
        return _kernel.step().as_dict()
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/run")
def run(request: RunRequest) -> dict[str, object]:
    try:
        reports = _kernel.run(request.cycles)
        return {
            "reports": [report.as_dict() for report in reports],
            "status": _kernel.status(),
        }
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/autorun/start")
def autorun_start(request: AutorunRequest) -> dict[str, object]:
    try:
        return _kernel.start_autorun(request.interval_seconds)
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/autorun/stop")
def autorun_stop() -> dict[str, object]:
    try:
        return _kernel.stop_autorun()
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/halt/reset")
def halt_reset() -> dict[str, object]:
    try:
        return _kernel.reset_halt()
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/snapshot")
def snapshot(limit: int = Query(default=2_048, ge=1, le=20_000)) -> dict[str, object]:
    try:
        return _kernel.snapshot(limit)
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/bitplane")
def bitplane(limit: int = Query(default=256, ge=1, le=4_096)) -> dict[str, object]:
    try:
        return _kernel.bitplane(limit)
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    state = _kernel.status()
    plane_raw = state.get("bitplane")
    plane = cast(dict[str, object], plane_raw) if isinstance(plane_raw, dict) else {}
    body = "\n".join(
        [
            "# HELP jarvisx_dr_moagi_os_cycles Authoritative committed OS cycles.",
            "# TYPE jarvisx_dr_moagi_os_cycles gauge",
            f"jarvisx_dr_moagi_os_cycles {_integer(state.get('cycle'))}",
            "# HELP jarvisx_dr_moagi_os_active_cells Current sparse active cells.",
            "# TYPE jarvisx_dr_moagi_os_active_cells gauge",
            f"jarvisx_dr_moagi_os_active_cells {_integer(state.get('active_cells'))}",
            "# HELP jarvisx_dr_moagi_os_bit_density Current binary voxel density.",
            "# TYPE jarvisx_dr_moagi_os_bit_density gauge",
            f"jarvisx_dr_moagi_os_bit_density {_numeric(plane.get('density'))}",
            "# HELP jarvisx_dr_moagi_os_spatial_entropy Binary occupancy entropy.",
            "# TYPE jarvisx_dr_moagi_os_spatial_entropy gauge",
            f"jarvisx_dr_moagi_os_spatial_entropy {_numeric(plane.get('entropy'))}",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
