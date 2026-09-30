from __future__ import annotations

from base64 import b64decode, b64encode
from binascii import Error as BinasciiError
from hashlib import sha256
from threading import Lock

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .kinetic3d import Kinetic3DRuntime, MAX_KINETIC_VOXELS, Shape3D
from .kinetic3d_backend import available_backends
from .kinetic3d_capsule import (
    CapsuleError,
    build_capsule,
    decode_capsule,
    parse_capsule,
    plan_rate_distortion,
)

app = FastAPI(
    title="Jarvis-X Kinetic 3D Runtime",
    version="2.0.0",
    description=(
        "Predictive sparse 3D execution with native backend routing, deterministic "
        "rate-distortion planning, reversible JXK2 capsules, and verify-before-commit state."
    ),
)

_sessions: dict[str, Kinetic3DRuntime] = {}
_session_lock = Lock()
_metrics_lock = Lock()
_executions = 0
_adaptive_executions = 0
_failures = 0
_commits = 0
_total_active_cells = 0
_total_cells = 0
_total_capsule_bytes = 0


class KineticExecuteRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    shape: Shape3D
    values: list[float]
    previous: list[float] | None = None
    active_threshold: float = Field(default=0.0, ge=0.0)
    coarse_factor: int = Field(default=2, ge=1, le=32)
    refine_threshold: float = Field(default=0.0, ge=0.0)
    tolerance: float = Field(default=0.0, ge=0.0)
    backend: str = Field(default="auto", min_length=1, max_length=32)


class AdaptiveExecuteRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    shape: Shape3D
    values: list[float]
    previous: list[float] | None = None
    tolerance: float = Field(default=0.0, ge=0.0)
    backend: str = Field(default="auto", min_length=1, max_length=32)


class CapsuleDecodeRequest(BaseModel):
    capsule_base64: str = Field(min_length=1, max_length=32_000_000)
    prediction: list[float]


def _runtime_for(session_id: str) -> Kinetic3DRuntime:
    with _session_lock:
        runtime = _sessions.get(session_id)
        if runtime is None:
            runtime = Kinetic3DRuntime()
            _sessions[session_id] = runtime
        return runtime


def _prediction_for(
    runtime: Kinetic3DRuntime,
    previous: list[float] | None,
    shape: Shape3D,
) -> list[float]:
    if len(shape) != 3 or any(dimension < 1 for dimension in shape):
        raise ValueError("shape must contain exactly three positive dimensions")
    count = shape[0] * shape[1] * shape[2]
    if count > MAX_KINETIC_VOXELS:
        raise ValueError(f"voxel count {count} exceeds runtime limit {MAX_KINETIC_VOXELS}")
    if previous is not None:
        return list(previous)
    if runtime.committed_shape == shape and runtime.committed_world is not None:
        return list(runtime.committed_world)
    return [0.0] * count


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "runtime": "kinetic3d",
        "api_version": "2.0.0",
        "backend_policy": "auto",
        "available_backends": list(available_backends()),
        "max_voxels": MAX_KINETIC_VOXELS,
        "execution_model": "predict-residual-active-encode-refine-decode-verify-commit",
        "adaptive_planner": "deterministic-rate-distortion",
        "capsule_format": "JXK2",
    }


@app.post("/v1/kinetic3d/execute")
def execute_kinetic3d(request: KineticExecuteRequest) -> dict[str, object]:
    global _executions, _failures, _commits, _total_active_cells, _total_cells

    runtime = _runtime_for(request.session_id)
    try:
        with _session_lock:
            result = runtime.execute(
                request.values,
                request.shape,
                previous=request.previous,
                active_threshold=request.active_threshold,
                coarse_factor=request.coarse_factor,
                refine_threshold=request.refine_threshold,
                tolerance=request.tolerance,
                backend=request.backend,
            )
    except ValueError as exc:
        with _metrics_lock:
            _failures += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _metrics_lock:
        _executions += 1
        _commits += int(result.committed)
        _total_active_cells += result.telemetry.active_cells
        _total_cells += result.telemetry.total_cells

    payload: dict[str, object] = result.as_payload()
    payload["session_id"] = request.session_id
    return payload


@app.post("/v2/kinetic3d/execute-adaptive")
def execute_adaptive_kinetic3d(request: AdaptiveExecuteRequest) -> dict[str, object]:
    global _executions, _adaptive_executions, _failures, _commits
    global _total_active_cells, _total_cells, _total_capsule_bytes

    runtime = _runtime_for(request.session_id)
    try:
        with _session_lock:
            prediction = _prediction_for(runtime, request.previous, request.shape)
            plan = plan_rate_distortion(
                request.values,
                prediction,
                request.shape,
                tolerance=request.tolerance,
            )
            selected = plan.selected
            result = runtime.execute(
                request.values,
                request.shape,
                previous=prediction,
                active_threshold=selected.active_threshold,
                coarse_factor=selected.coarse_factor,
                refine_threshold=selected.refine_threshold,
                tolerance=request.tolerance,
                backend=request.backend,
            )
            if not result.committed:
                raise RuntimeError("selected backend diverged from the verified rate-distortion plan")

            capsule_bytes = build_capsule(
                shape=result.shape,
                prediction=result.prediction,
                active_threshold=selected.active_threshold,
                coarse_factor=selected.coarse_factor,
                refine_threshold=selected.refine_threshold,
                tolerance=request.tolerance,
                active_indices=result.active_indices,
                coarse_values=tuple(
                    (item.block, item.residual) for item in result.coarse_latent
                ),
                fine_corrections=tuple(
                    (item.index, item.correction) for item in result.fine_corrections
                ),
            )
            independently_decoded = decode_capsule(capsule_bytes, result.prediction)
            if independently_decoded != result.reconstructed:
                raise RuntimeError("JXK2 capsule decode does not reproduce committed reconstruction")
            capsule = parse_capsule(capsule_bytes)
    except ValueError as exc:
        with _metrics_lock:
            _failures += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (CapsuleError, RuntimeError) as exc:
        with _metrics_lock:
            _failures += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with _metrics_lock:
        _executions += 1
        _adaptive_executions += 1
        _commits += 1
        _total_active_cells += result.telemetry.active_cells
        _total_cells += result.telemetry.total_cells
        _total_capsule_bytes += len(capsule_bytes)

    payload: dict[str, object] = result.as_payload()
    payload["session_id"] = request.session_id
    payload["adaptive_plan"] = plan.as_payload()
    payload["capsule"] = {
        **capsule.as_payload(),
        "bytes": len(capsule_bytes),
        "transport_sha256": sha256(capsule_bytes).hexdigest(),
        "wire_compression_ratio": (len(request.values) * 8) / len(capsule_bytes),
        "base64": b64encode(capsule_bytes).decode("ascii"),
    }
    return payload


@app.post("/v2/kinetic3d/decode-capsule")
def decode_kinetic_capsule(request: CapsuleDecodeRequest) -> dict[str, object]:
    try:
        data = b64decode(request.capsule_base64, validate=True)
        capsule = parse_capsule(data)
        reconstructed = capsule.decode(request.prediction)
    except (BinasciiError, ValueError, CapsuleError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "capsule": {**capsule.as_payload(), "bytes": len(data)},
        "reconstructed": list(reconstructed),
    }


@app.get("/v1/kinetic3d/state/{session_id}")
def session_state(session_id: str) -> dict[str, object]:
    with _session_lock:
        runtime = _sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="unknown session")
        world = runtime.committed_world
        return {
            "session_id": session_id,
            "epoch": runtime.epoch,
            "shape": list(runtime.committed_shape) if runtime.committed_shape else None,
            "world": list(world) if world is not None else None,
        }


@app.delete("/v1/kinetic3d/state/{session_id}")
def reset_session(session_id: str) -> dict[str, object]:
    with _session_lock:
        runtime = _sessions.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="unknown session")
        runtime.reset()
    return {"session_id": session_id, "status": "reset"}


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    with _metrics_lock:
        executions = _executions
        adaptive_executions = _adaptive_executions
        failures = _failures
        commits = _commits
        active = _total_active_cells
        cells = _total_cells
        capsule_bytes = _total_capsule_bytes

    active_fraction = active / cells if cells else 0.0
    body = "\n".join(
        [
            "# HELP jarvisx_kinetic3d_executions_total Successful API executions.",
            "# TYPE jarvisx_kinetic3d_executions_total counter",
            f"jarvisx_kinetic3d_executions_total {executions}",
            "# HELP jarvisx_kinetic3d_adaptive_executions_total Adaptive planned executions.",
            "# TYPE jarvisx_kinetic3d_adaptive_executions_total counter",
            f"jarvisx_kinetic3d_adaptive_executions_total {adaptive_executions}",
            "# HELP jarvisx_kinetic3d_failures_total Rejected API executions.",
            "# TYPE jarvisx_kinetic3d_failures_total counter",
            f"jarvisx_kinetic3d_failures_total {failures}",
            "# HELP jarvisx_kinetic3d_commits_total Verified committed transitions.",
            "# TYPE jarvisx_kinetic3d_commits_total counter",
            f"jarvisx_kinetic3d_commits_total {commits}",
            "# HELP jarvisx_kinetic3d_active_fraction Cumulative active cell fraction.",
            "# TYPE jarvisx_kinetic3d_active_fraction gauge",
            f"jarvisx_kinetic3d_active_fraction {active_fraction}",
            "# HELP jarvisx_kinetic3d_capsule_bytes_total JXK2 transport bytes emitted.",
            "# TYPE jarvisx_kinetic3d_capsule_bytes_total counter",
            f"jarvisx_kinetic3d_capsule_bytes_total {capsule_bytes}",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
