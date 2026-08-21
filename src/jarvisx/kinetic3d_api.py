from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .kinetic3d import Kinetic3DRuntime, MAX_KINETIC_VOXELS, Shape3D

app = FastAPI(
    title="Jarvis-X Kinetic 3D Runtime",
    version="1.0.0",
    description="Predictive sparse 3D execution with hierarchical residuals and verify-before-commit state.",
)

_sessions: dict[str, Kinetic3DRuntime] = {}
_session_lock = Lock()
_metrics_lock = Lock()
_executions = 0
_failures = 0
_commits = 0
_total_active_cells = 0
_total_cells = 0


class KineticExecuteRequest(BaseModel):
    session_id: str = Field(default="default", min_length=1, max_length=128)
    shape: Shape3D
    values: list[float]
    previous: list[float] | None = None
    active_threshold: float = Field(default=0.0, ge=0.0)
    coarse_factor: int = Field(default=2, ge=1, le=32)
    refine_threshold: float = Field(default=0.0, ge=0.0)
    tolerance: float = Field(default=0.0, ge=0.0)


def _runtime_for(session_id: str) -> Kinetic3DRuntime:
    with _session_lock:
        runtime = _sessions.get(session_id)
        if runtime is None:
            runtime = Kinetic3DRuntime()
            _sessions[session_id] = runtime
        return runtime


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "runtime": "kinetic3d",
        "backend": "cpu-reference",
        "max_voxels": MAX_KINETIC_VOXELS,
        "execution_model": "predict-residual-active-encode-refine-decode-verify-commit",
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
        failures = _failures
        commits = _commits
        active = _total_active_cells
        cells = _total_cells

    active_fraction = active / cells if cells else 0.0
    body = "\n".join(
        [
            "# HELP jarvisx_kinetic3d_executions_total Successful API executions.",
            "# TYPE jarvisx_kinetic3d_executions_total counter",
            f"jarvisx_kinetic3d_executions_total {executions}",
            "# HELP jarvisx_kinetic3d_failures_total Rejected API executions.",
            "# TYPE jarvisx_kinetic3d_failures_total counter",
            f"jarvisx_kinetic3d_failures_total {failures}",
            "# HELP jarvisx_kinetic3d_commits_total Verified committed transitions.",
            "# TYPE jarvisx_kinetic3d_commits_total counter",
            f"jarvisx_kinetic3d_commits_total {commits}",
            "# HELP jarvisx_kinetic3d_active_fraction Cumulative active cell fraction.",
            "# TYPE jarvisx_kinetic3d_active_fraction gauge",
            f"jarvisx_kinetic3d_active_fraction {active_fraction}",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
