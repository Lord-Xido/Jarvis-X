"""FastAPI control plane for the end-to-end Dr Moagi 3D OS kernel."""

from __future__ import annotations

import base64
import binascii
import os
from pathlib import Path
from typing import NoReturn, cast

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .dr_moagi_field_runtime import SparseField
from .dr_moagi_meta_optimizer import SelfOptimizing3DSystem
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field
from .dr_moagi_os_ui import DR_MOAGI_OS_HTML


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean environment value")


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
        deep_distiller_enabled=_bool_env("JARVISX_OS_DEEP_DISTILLER", True),
        deep_distiller_passes=_int_env("JARVISX_OS_DEEP_DISTILLER_PASSES", 1),
        deep_distiller_max_latent_cells=_int_env(
            "JARVISX_OS_DEEP_DISTILLER_MAX_LATENT", 25_000
        ),
        deep_distiller_learning_rate=_float_env(
            "JARVISX_OS_DEEP_DISTILLER_LR", 0.05
        ),
        fixed_point_passes=_int_env("JARVISX_OS_FIXED_POINT_PASSES", 1),
        autorun_interval_seconds=_float_env("JARVISX_OS_AUTORUN_INTERVAL", 0.5),
        state_dir=state_dir,
    )


app = FastAPI(
    title="Dr Moagi 3D OS",
    version="2.1.0",
    description=(
        "End-to-end bounded sparse 3D operating control plane with Uint64 bit-plane "
        "execution, inward folding, Deep Distiller adaptation, fixed-point verification, "
        "exact Morton transport, checkpoint recovery, bounded 3D meta-optimization "
        "and auto-run scheduling."
    ),
)

_system = SelfOptimizing3DSystem(DrMoagiOSKernel(_config_from_env()))


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


class ImportPacketRequest(BaseModel):
    packet_base64: str = Field(min_length=1)
    checksum_sha256: str | None = Field(default=None, min_length=64, max_length=64)


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


def _payload(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    return HTMLResponse(DR_MOAGI_OS_HTML)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    status = _system.status()
    return {
        "status": "ok",
        "system": status["system"],
        "lifecycle": status["lifecycle"],
        "loaded": status["loaded"],
        "journal_valid": status["journal_valid"],
        "meta_journal_valid": _payload(status["meta_optimizer"])["journal_valid"],
    }


@app.get("/v1/os/capabilities")
def capabilities() -> dict[str, object]:
    capabilities = dict(_system.kernel.capabilities())
    capabilities["meta_optimizer"] = {
        "search_space": "bounded 3D policy lattice",
        "axes": {
            "x": "compression geometry",
            "y": "adaptive dynamics",
            "z": "spatial/fixed-point dynamics",
        },
        "promotion": "deterministic replay + transactional meta gate",
        "external_sota_verified": False,
    }
    return capabilities


@app.post("/v1/os/boot")
def boot() -> dict[str, object]:
    try:
        return _payload(_system.kernel.boot(restore=True))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/shutdown")
def shutdown() -> dict[str, object]:
    try:
        return _payload(_system.kernel.shutdown())
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/status")
def status() -> dict[str, object]:
    return _payload(_system.status())


@app.post("/v1/os/load")
def load(request: LoadRequest) -> dict[str, object]:
    try:
        if _system.kernel.lifecycle.value == "offline":
            _system.kernel.boot(restore=False)
        return _payload(_system.kernel.load(_field_from_request(request)))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/demo")
def load_demo() -> dict[str, object]:
    try:
        if _system.kernel.lifecycle.value == "offline":
            _system.kernel.boot(restore=False)
        return _payload(_system.kernel.load(demo_field(_system.kernel.config.side)))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/step")
def step() -> dict[str, object]:
    try:
        return _payload(_system.step().as_dict())
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/run")
def run(request: RunRequest) -> dict[str, object]:
    try:
        reports = _system.run(request.cycles)
        return {
            "reports": [report.as_dict() for report in reports],
            "status": _system.status(),
        }
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/autorun/start")
def autorun_start(request: AutorunRequest) -> dict[str, object]:
    try:
        return _payload(_system.kernel.start_autorun(request.interval_seconds))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/autorun/stop")
def autorun_stop() -> dict[str, object]:
    try:
        return _payload(_system.kernel.stop_autorun())
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/halt/reset")
def halt_reset() -> dict[str, object]:
    try:
        return _payload(_system.kernel.reset_halt())
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/snapshot")
def snapshot(limit: int = Query(default=2_048, ge=1, le=20_000)) -> dict[str, object]:
    try:
        return _payload(_system.kernel.snapshot(limit))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/bitplane")
def bitplane(limit: int = Query(default=256, ge=1, le=4_096)) -> dict[str, object]:
    try:
        return _payload(_system.kernel.bitplane(limit))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/export")
def export_packet() -> dict[str, object]:
    try:
        packet = _system.kernel.export_state_packet()
        return {
            **packet.as_dict(),
            "packet_base64": base64.b64encode(packet.payload).decode("ascii"),
            "state_hash": _system.status()["state_hash"],
        }
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/os/import")
def import_packet(request: ImportPacketRequest) -> dict[str, object]:
    try:
        if _system.kernel.lifecycle.value == "offline":
            _system.kernel.boot(restore=False)
        try:
            payload = base64.b64decode(request.packet_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("packet_base64 is not valid base64") from exc
        return _payload(
            _system.kernel.import_state_packet(
                payload,
                expected_checksum=request.checksum_sha256,
            )
        )
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/os/meta/status")
def meta_status() -> dict[str, object]:
    return _payload(_system.status()["meta_optimizer"])


@app.get("/v1/os/meta/lattice")
def meta_lattice() -> dict[str, object]:
    return _payload(_system.meta_lattice())


@app.post("/v1/os/meta/optimize")
def meta_optimize() -> dict[str, object]:
    try:
        report = _system.turn_inward()
        return {
            "report": report.as_dict(),
            "status": _system.status(),
            "lattice": _system.meta_lattice(),
        }
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    state = _system.status()
    plane_raw = state.get("bitplane")
    plane = cast(dict[str, object], plane_raw) if isinstance(plane_raw, dict) else {}
    distiller_raw = state.get("distiller")
    distiller = cast(dict[str, object], distiller_raw) if isinstance(distiller_raw, dict) else {}
    transport_raw = state.get("transport")
    transport = cast(dict[str, object], transport_raw) if isinstance(transport_raw, dict) else {}
    last_raw = state.get("last_report")
    last = cast(dict[str, object], last_raw) if isinstance(last_raw, dict) else {}
    meta_raw = state.get("meta_optimizer")
    meta = cast(dict[str, object], meta_raw) if isinstance(meta_raw, dict) else {}
    meta_report_raw = meta.get("last_report")
    meta_report = (
        cast(dict[str, object], meta_report_raw) if isinstance(meta_report_raw, dict) else {}
    )
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
            "# HELP jarvisx_dr_moagi_os_distiller_iteration Committed DM-DD adaptive iterations.",
            "# TYPE jarvisx_dr_moagi_os_distiller_iteration gauge",
            f"jarvisx_dr_moagi_os_distiller_iteration {_integer(distiller.get('iteration'))}",
            "# HELP jarvisx_dr_moagi_os_distiller_residual_rms Latest DM-DD residual RMS.",
            "# TYPE jarvisx_dr_moagi_os_distiller_residual_rms gauge",
            f"jarvisx_dr_moagi_os_distiller_residual_rms {_numeric(last.get('distiller_residual_rms'))}",
            "# HELP jarvisx_dr_moagi_os_transport_bytes Exact sparse transport packet bytes.",
            "# TYPE jarvisx_dr_moagi_os_transport_bytes gauge",
            f"jarvisx_dr_moagi_os_transport_bytes {_integer(transport.get('encoded_bytes'))}",
            "# HELP jarvisx_dr_moagi_os_meta_epoch Completed inward meta-optimization epochs.",
            "# TYPE jarvisx_dr_moagi_os_meta_epoch gauge",
            f"jarvisx_dr_moagi_os_meta_epoch {_integer(meta.get('epoch'))}",
            "# HELP jarvisx_dr_moagi_os_meta_relative_improvement Latest relative incumbent improvement.",
            "# TYPE jarvisx_dr_moagi_os_meta_relative_improvement gauge",
            f"jarvisx_dr_moagi_os_meta_relative_improvement {_numeric(meta_report.get('relative_improvement'))}",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
