from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .assembler import Assembler
from .auto_codec_loop import AutoCodecLoop, AutoCodecLoopConfig, UniformQuantizedFieldCodec
from .core import CodexVM
from .dashboard_3d import DASHBOARD_HTML
from .dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime
from .parser import Parser
from .run_store import RunArtifactStore
from .spatial_codec_3d import MortonQuantizedFieldCodec3D, SpatialAutoCodec3DSystem

app = FastAPI(
    title="Jarvis-X / Dr Moagi 3D Runtime API",
    version="0.1.0",
    description="Deterministic VM and bounded persistent operational 3D auto-codec runtime.",
)
RUN_STORE = RunArtifactStore(os.environ.get("JARVISX_RUN_STORE", "data/runs"))


class VMRunRequest(BaseModel):
    source: str = ""


class SparseCell(BaseModel):
    x: int
    y: int
    z: int
    value: float


class AutoCodecRequest(BaseModel):
    cells: list[SparseCell] = Field(default_factory=list)
    side: int = Field(default=64, gt=0, le=1000)
    quantization_step: float = Field(default=0.05, gt=0.0)
    alpha: float = Field(default=1.0, ge=0.0)
    lambda_residual: float = Field(default=0.0, ge=0.0)
    eta: float = 0.0
    dt: float = Field(default=0.05, gt=0.0)
    max_active_cells: int = Field(default=100_000, gt=0, le=100_000)
    expand_halo: bool = False
    prune_epsilon: float = Field(default=0.0, ge=0.0)
    max_cycles: int = Field(default=64, gt=0, le=512)
    min_cycles: int = Field(default=1, gt=0, le=512)
    reconstruction_mse_target: float = Field(default=1e-8, ge=0.0)
    max_consecutive_rejections: int = Field(default=3, gt=0, le=100)
    stop_on_fixed_point: bool = True


class AutoCodec3DRequest(AutoCodecRequest):
    frame_stride: int = Field(default=1, gt=0, le=512)
    max_render_points: int = Field(default=4096, gt=0, le=10_000)
    max_frames: int = Field(default=128, gt=0, le=256)
    persist: bool = True


def _field_from_cells(cells: list[SparseCell]) -> dict[tuple[int, int, int], float]:
    field: dict[tuple[int, int, int], float] = {}
    for cell in cells:
        coordinate = (cell.x, cell.y, cell.z)
        if coordinate in field:
            raise ValueError(f"duplicate sparse coordinate: {coordinate}")
        field[coordinate] = cell.value
    return field


def _field_config(payload: AutoCodecRequest) -> DrMoagiFieldConfig:
    return DrMoagiFieldConfig(
        side=payload.side,
        alpha=payload.alpha,
        lambda_residual=payload.lambda_residual,
        eta=payload.eta,
        dt=payload.dt,
        max_active_cells=payload.max_active_cells,
        expand_halo=payload.expand_halo,
        prune_epsilon=payload.prune_epsilon,
    )


def _loop_config(payload: AutoCodecRequest) -> AutoCodecLoopConfig:
    return AutoCodecLoopConfig(
        max_cycles=payload.max_cycles,
        min_cycles=payload.min_cycles,
        reconstruction_mse_target=payload.reconstruction_mse_target,
        max_consecutive_rejections=payload.max_consecutive_rejections,
        stop_on_fixed_point=payload.stop_on_fixed_point,
    )


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime": "jarvisx-auto-codec-3d",
        "spatial_codec": "morton-63bit-int32",
        "persistence": "enabled",
        "endpoints": [
            "/run",
            "/codec/run",
            "/codec/3d/run",
            "/codec/3d/runs/{run_id}",
            "/codec/3d/runs/{run_id}/verify",
        ],
    }


@app.post("/run")
def run_code(payload: VMRunRequest) -> dict[str, Any]:
    try:
        ast = Parser().parse(payload.source)
        bytecode = Assembler().assemble(ast)
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        return {"registers": vm.regs.snapshot(), "ledger_entries": len(vm.ledger.chain)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/codec/run")
def run_auto_codec(payload: AutoCodecRequest) -> dict[str, Any]:
    if payload.min_cycles > payload.max_cycles:
        raise HTTPException(status_code=422, detail="min_cycles cannot exceed max_cycles")
    try:
        field = _field_from_cells(payload.cells)
        codec = UniformQuantizedFieldCodec(step=payload.quantization_step)
        runtime = DrMoagiFieldRuntime(codec, _field_config(payload))
        loop = AutoCodecLoop(runtime, _loop_config(payload))
        loop.load(field)
        return loop.run().to_dict()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/codec/3d/run")
def run_auto_codec_3d(payload: AutoCodec3DRequest) -> dict[str, Any]:
    if payload.min_cycles > payload.max_cycles:
        raise HTTPException(status_code=422, detail="min_cycles cannot exceed max_cycles")
    try:
        field = _field_from_cells(payload.cells)
        codec = MortonQuantizedFieldCodec3D(
            step=payload.quantization_step,
            side=payload.side,
        )
        runtime = DrMoagiFieldRuntime(codec, _field_config(payload))

        run_id = RUN_STORE.new_run_id() if payload.persist else None
        ledger = RUN_STORE.ledger(run_id) if run_id is not None else None
        loop = AutoCodecLoop(runtime, _loop_config(payload), ledger=ledger)
        system = SpatialAutoCodec3DSystem(
            loop,
            codec,
            side=payload.side,
            frame_stride=payload.frame_stride,
            max_render_points=payload.max_render_points,
            max_frames=payload.max_frames,
        )
        result = system.run(field).to_dict()
        result["run_id"] = run_id
        result["persisted"] = run_id is not None
        if run_id is not None:
            RUN_STORE.write_summary(run_id, result)
        return result
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"run persistence failed: {exc}") from exc


@app.get("/codec/3d/runs/{run_id}")
def get_auto_codec_3d_run(run_id: str) -> dict[str, Any]:
    try:
        return RUN_STORE.read_summary(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"run retrieval failed: {exc}") from exc


@app.get("/codec/3d/runs/{run_id}/verify")
def verify_auto_codec_3d_run(run_id: str) -> dict[str, Any]:
    try:
        verification = RUN_STORE.verify(run_id)
        if not verification["verified"]:
            raise HTTPException(status_code=409, detail=verification)
        return verification
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=f"run verification failed: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"run verification failed: {exc}") from exc


def start_api() -> None:
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("jarvisx.api:app", host="0.0.0.0", port=port)
