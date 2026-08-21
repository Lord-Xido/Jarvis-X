from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .bitcode3d import BitCode3DRuntime, MAX_VOXELS, Shape3D

app = FastAPI(
    title="Jarvis-X 3D Bit Code Runtime",
    version="1.0.0",
    description="Bounded end-to-end Q16.16 3D encode/decode execution service.",
)

_runtime = BitCode3DRuntime()
_metrics_lock = Lock()
_executions = 0
_failures = 0
_total_cycles = 0
_total_active_cells = 0


class ExecuteRequest(BaseModel):
    shape: Shape3D
    values: list[float]
    pool: int = Field(default=2, ge=1, le=16)
    tolerance: float = Field(default=1.0, ge=0.0)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "runtime": "bitcode3d",
        "max_voxels": MAX_VOXELS,
        "isa": "32-bit compact lowering",
        "numeric_format": "Q16.16",
    }


@app.post("/v1/bitcode3d/execute")
def execute_bitcode3d(request: ExecuteRequest) -> dict[str, object]:
    global _executions, _failures, _total_cycles, _total_active_cells

    try:
        result = _runtime.execute(
            request.values,
            request.shape,
            pool=request.pool,
            tolerance=request.tolerance,
        )
    except ValueError as exc:
        with _metrics_lock:
            _failures += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _metrics_lock:
        _executions += 1
        _total_cycles += result.telemetry.cycles
        _total_active_cells += result.telemetry.active_cells
    payload: dict[str, object] = result.as_payload()
    return payload


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    with _metrics_lock:
        executions = _executions
        failures = _failures
        cycles = _total_cycles
        active_cells = _total_active_cells

    body = "\n".join(
        [
            "# HELP jarvisx_bitcode3d_executions_total Successful runtime executions.",
            "# TYPE jarvisx_bitcode3d_executions_total counter",
            f"jarvisx_bitcode3d_executions_total {executions}",
            "# HELP jarvisx_bitcode3d_failures_total Rejected runtime executions.",
            "# TYPE jarvisx_bitcode3d_failures_total counter",
            f"jarvisx_bitcode3d_failures_total {failures}",
            "# HELP jarvisx_bitcode3d_cycles_total Executed 32-bit instruction cycles.",
            "# TYPE jarvisx_bitcode3d_cycles_total counter",
            f"jarvisx_bitcode3d_cycles_total {cycles}",
            "# HELP jarvisx_bitcode3d_active_cells_total Input voxels processed.",
            "# TYPE jarvisx_bitcode3d_active_cells_total counter",
            f"jarvisx_bitcode3d_active_cells_total {active_cells}",
            "",
        ]
    )
    return Response(content=body, media_type="text/plain; version=0.0.4")
