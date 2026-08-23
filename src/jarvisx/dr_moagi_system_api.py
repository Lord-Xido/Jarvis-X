"""FastAPI control plane for four-scale Dr Moagi system auto-evolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dr_moagi_field_runtime import SparseField
from .dr_moagi_meta_optimizer import MetaSearchConfig, SelfOptimizing3DSystem
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field
from .dr_moagi_system_evolution import ArchitecturePolicy, SelfEvolving3DArchitecture


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _config_from_env() -> DrMoagiOSConfig:
    state_dir_raw = os.getenv("JARVISX_STATE_DIR", "state/dr-moagi-os")
    state_dir = None if state_dir_raw.lower() == "none" else Path(state_dir_raw)
    return DrMoagiOSConfig(
        side=_int_env("JARVISX_OS_SIDE", 64),
        max_active_cells=_int_env("JARVISX_OS_MAX_ACTIVE_CELLS", 50_000),
        deep_distiller_max_latent_cells=_int_env(
            "JARVISX_OS_DEEP_DISTILLER_MAX_LATENT", 25_000
        ),
        state_dir=state_dir,
    )


def _policy_from_env() -> ArchitecturePolicy:
    meta_search = MetaSearchConfig(
        max_candidates=_int_env("JARVISX_META_MAX_CANDIDATES", 13),
        probe_cycles=_int_env("JARVISX_META_PROBE_CYCLES", 1),
        confirm_cycles=_int_env("JARVISX_META_CONFIRM_CYCLES", 3),
        max_eval_cells=_int_env("JARVISX_META_MAX_EVAL_CELLS", 2_048),
        survivors=_int_env("JARVISX_META_SURVIVORS", 4),
    )
    return ArchitecturePolicy(
        state_cycles_per_meta=_int_env("JARVISX_ARCH_STATE_CYCLES_PER_META", 8),
        meta_epochs_per_architecture_review=_int_env(
            "JARVISX_ARCH_META_EPOCHS_PER_REVIEW", 3
        ),
        max_architecture_candidates=_int_env("JARVISX_ARCH_MAX_CANDIDATES", 7),
        max_architecture_eval_cells=_int_env("JARVISX_ARCH_MAX_EVAL_CELLS", 512),
        max_eval_state_cycles=_int_env("JARVISX_ARCH_MAX_EVAL_STATE_CYCLES", 4),
        min_architecture_improvement=_float_env("JARVISX_ARCH_MIN_IMPROVEMENT", 0.01),
        meta_search=meta_search,
    )


app = FastAPI(
    title="Dr Moagi Self-Evolving 3D System",
    version="3.0.0",
    description=(
        "Four-scale bounded adaptive control plane: sparse state, DM-DD model, "
        "runtime configuration and architecture orchestration policy."
    ),
)

_policy = _policy_from_env()
_meta = SelfOptimizing3DSystem(DrMoagiOSKernel(_config_from_env()), search=_policy.meta_search)
_architecture: SelfEvolving3DArchitecture = SelfEvolving3DArchitecture(_meta, policy=_policy)


class FieldCell(BaseModel):
    x: int
    y: int
    z: int
    value: float


class LoadRequest(BaseModel):
    field: list[FieldCell] = Field(min_length=1)


class RunRequest(BaseModel):
    cycles: int = Field(default=8, ge=1, le=4_096)


class AutonomicRunRequest(BaseModel):
    cycles: int = Field(default=16, ge=1, le=256)


def _payload(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def _raise_http(exc: Exception) -> NoReturn:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise HTTPException(status_code=409, detail=str(exc)) from exc


def _field_from_request(request: LoadRequest) -> SparseField:
    return {(item.x, item.y, item.z): float(item.value) for item in request.field}


@app.get("/healthz")
def healthz() -> dict[str, object]:
    status = _architecture.status()
    architecture = _payload(status["architecture_evolution"])
    meta = _payload(status["meta_optimizer"])
    return {
        "status": "ok",
        "lifecycle": status["lifecycle"],
        "loaded": status["loaded"],
        "state_journal_valid": status["journal_valid"],
        "meta_journal_valid": meta["journal_valid"],
        "architecture_journal_valid": architecture["journal_valid"],
    }


@app.get("/v1/system/capabilities")
def capabilities() -> dict[str, object]:
    return cast(dict[str, object], _architecture.capabilities())


@app.get("/v1/system/status")
def status() -> dict[str, object]:
    return cast(dict[str, object], _architecture.status())


@app.post("/v1/system/boot")
def boot() -> dict[str, object]:
    try:
        return _payload(_architecture.kernel.boot(restore=True))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/system/demo")
def load_demo() -> dict[str, object]:
    try:
        if _architecture.kernel.lifecycle.value == "offline":
            _architecture.kernel.boot(restore=False)
        return _payload(_architecture.kernel.load(demo_field(_architecture.kernel.config.side)))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/system/load")
def load(request: LoadRequest) -> dict[str, object]:
    try:
        if _architecture.kernel.lifecycle.value == "offline":
            _architecture.kernel.boot(restore=False)
        return _payload(_architecture.kernel.load(_field_from_request(request)))
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/system/step")
def step() -> dict[str, object]:
    try:
        return _payload(_architecture.step().as_dict())
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/system/run")
def run(request: RunRequest) -> dict[str, object]:
    try:
        reports = _architecture.run(request.cycles)
        return {
            "reports": [item.as_dict() for item in reports],
            "status": _architecture.status(),
        }
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/system/meta/optimize")
def meta_optimize() -> dict[str, object]:
    try:
        report = _architecture.turn_inward()
        return {"report": report.as_dict(), "status": _architecture.status()}
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.get("/v1/system/architecture/lattice")
def architecture_lattice() -> dict[str, object]:
    return cast(dict[str, object], _architecture.architecture_lattice())


@app.post("/v1/system/architecture/evolve")
def architecture_evolve() -> dict[str, object]:
    try:
        report = _architecture.evolve_architecture()
        return {
            "report": report.as_dict(),
            "status": _architecture.status(),
            "lattice": _architecture.architecture_lattice(),
        }
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)


@app.post("/v1/system/autonomic/run")
def autonomic_run(request: AutonomicRunRequest) -> dict[str, object]:
    try:
        report = _architecture.run_autonomic(request.cycles)
        return {"report": report.as_dict(), "status": _architecture.status()}
    except (ValueError, RuntimeError) as exc:
        _raise_http(exc)
