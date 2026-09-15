"""FastAPI control surface for the Dr Moagi Cloud OS reference runtime."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Annotated, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from .cloud_os import DrMoagiCloudOS, Field3D
from .qvector3d import QVectorField3D, q16_from_float
from .qvector_cloud import DrMoagiQVectorCloudEngine3D
from .qvector_v2 import QBoundaryMode, QScalarKernel3D, QVectorFieldOps3D

SERVICE_NAME = "Jarvis-X Dr Moagi Cloud OS"


class FieldPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, int, int]
    values: list[float] = Field(min_length=1)


class QVectorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shape: tuple[int, int, int]
    vectors: list[tuple[float, float, float]] = Field(min_length=1)


class RoundTripRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=256)
    field: FieldPayload
    latent_shape: tuple[int, int, int]


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=256)
    field: FieldPayload
    complexity_weight: float = Field(default=0.01, ge=0.0)
    candidates: list[tuple[int, int, int]] | None = None


class QVectorRoundTripRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=256)
    field: QVectorPayload
    latent_shape: tuple[int, int, int]


class QVectorOptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=256)
    field: QVectorPayload
    complexity_weight: float = Field(default=0.01, ge=0.0)
    candidates: list[tuple[int, int, int]] | None = None


class QVectorFieldOpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: QVectorPayload
    operation: Literal["grad-x", "grad-y", "grad-z", "divergence", "curl", "laplacian"]
    spacing: float = Field(default=1.0, gt=0.0)
    boundary: Literal["clamp", "zero", "wrap"] = "clamp"


class QVectorConvolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: QVectorPayload
    kernel_shape: tuple[int, int, int]
    kernel_weights: list[float] = Field(min_length=1)
    boundary: Literal["clamp", "zero", "wrap"] = "clamp"


class NodeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=128)
    max_cells: int = Field(gt=0)
    max_concurrency: int = Field(default=1, gt=0)


def _runtime_from_env() -> DrMoagiCloudOS:
    ledger = Path(os.getenv("JARVISX_CLOUD_LEDGER", "state/cloud-os-ledger.jsonl"))
    runtime = DrMoagiCloudOS(ledger_path=ledger)
    max_cells = int(os.getenv("JARVISX_CLOUD_MAX_CELLS", "1000000"))
    max_concurrency = int(os.getenv("JARVISX_CLOUD_MAX_CONCURRENCY", "4"))
    runtime.register_node("local-reference-node", max_cells, max_concurrency)
    return runtime


def _auth_dependency():
    configured = os.getenv("JARVISX_CLOUD_API_KEY")

    def authorize(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if configured is None:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, configured):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid API key",
            )

    return authorize


def _qvector_field(payload: QVectorPayload) -> QVectorField3D:
    return QVectorField3D.from_vectors(payload.vectors, payload.shape)


def _boundary_mode(value: str) -> QBoundaryMode:
    return {
        "clamp": QBoundaryMode.CLAMP,
        "zero": QBoundaryMode.ZERO,
        "wrap": QBoundaryMode.WRAP,
    }[value]


def _field_op_payload(field: QVectorField3D, ops: QVectorFieldOps3D) -> dict[str, object]:
    return {
        **field.raw_payload(),
        "digest": field.digest,
        "numeric_status": {
            "saturated": ops.status.saturated,
            "accumulator_saturated": ops.status.accumulator_saturated,
            "inexact": ops.status.inexact,
            "divide_by_zero": ops.status.divide_by_zero,
        },
    }


def create_app(runtime: DrMoagiCloudOS | None = None) -> FastAPI:
    cloud = runtime or _runtime_from_env()
    qvector = DrMoagiQVectorCloudEngine3D(cloud=cloud)
    authorize = _auth_dependency()
    app = FastAPI(
        title=SERVICE_NAME,
        version="2.0.0",
        description=(
            "Bounded deterministic 3D auto-encoding and Q16.16x3 geometric-field "
            "cloud control plane. This is a user-space reference runtime, not a "
            "bare-metal kernel or hypervisor."
        ),
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "nodes": len(cloud.nodes),
            "jobs": len(cloud.jobs),
            "qvector_jobs": len(qvector.jobs),
            "ledger_valid": cloud.ledger.verify(),
        }

    @app.get("/v1/nodes", dependencies=[Depends(authorize)])
    def list_nodes() -> list[dict[str, object]]:
        return cast(list[dict[str, object]], cloud.node_snapshots())

    @app.post(
        "/v1/nodes",
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(authorize)],
    )
    def register_node(request: NodeRequest) -> dict[str, object]:
        try:
            node = cloud.register_node(
                request.node_id,
                request.max_cells,
                request.max_concurrency,
            )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "node_id": node.node_id,
            "max_cells": node.max_cells,
            "max_concurrency": node.max_concurrency,
            "healthy": node.healthy,
        }

    @app.post("/v1/roundtrip", dependencies=[Depends(authorize)])
    def round_trip(request: RoundTripRequest) -> dict[str, object]:
        try:
            field = Field3D.from_values(request.field.values, request.field.shape)
            job = cloud.round_trip(
                field,
                request.latent_shape,
                request_id=request.request_id,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return cast(dict[str, object], cloud.job_snapshot(job.job_id))

    @app.post("/v1/auto-optimize", dependencies=[Depends(authorize)])
    def auto_optimize(request: OptimizeRequest) -> dict[str, object]:
        try:
            field = Field3D.from_values(request.field.values, request.field.shape)
            job = cloud.auto_optimize(
                field,
                request_id=request.request_id,
                complexity_weight=request.complexity_weight,
                candidates=request.candidates,
            )
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return cast(dict[str, object], cloud.job_snapshot(job.job_id))

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_job(job_id: str) -> dict[str, object]:
        try:
            return cast(dict[str, object], cloud.job_snapshot(job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/v2/qvector/roundtrip", dependencies=[Depends(authorize)])
    def qvector_round_trip(request: QVectorRoundTripRequest) -> dict[str, object]:
        try:
            job = qvector.round_trip(
                _qvector_field(request.field),
                request.latent_shape,
                request_id=request.request_id,
            )
        except (ValueError, RuntimeError, OverflowError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return cast(dict[str, object], qvector.job_snapshot(job.job_id))

    @app.post("/v2/qvector/auto-optimize", dependencies=[Depends(authorize)])
    def qvector_auto_optimize(request: QVectorOptimizeRequest) -> dict[str, object]:
        try:
            job = qvector.auto_optimize(
                _qvector_field(request.field),
                request_id=request.request_id,
                complexity_weight=request.complexity_weight,
                candidates=request.candidates,
            )
        except (ValueError, RuntimeError, OverflowError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return cast(dict[str, object], qvector.job_snapshot(job.job_id))

    @app.get("/v2/qvector/jobs/{job_id}", dependencies=[Depends(authorize)])
    def get_qvector_job(job_id: str) -> dict[str, object]:
        try:
            return cast(dict[str, object], qvector.job_snapshot(job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/v2/qvector/field-op", dependencies=[Depends(authorize)])
    def qvector_field_op(request: QVectorFieldOpRequest) -> dict[str, object]:
        try:
            field = _qvector_field(request.field)
            ops = QVectorFieldOps3D(boundary=_boundary_mode(request.boundary))
            spacing_q16 = q16_from_float(request.spacing)
            if request.operation == "grad-x":
                result = ops.directional_derivative(field, 0, spacing_q16=spacing_q16)
            elif request.operation == "grad-y":
                result = ops.directional_derivative(field, 1, spacing_q16=spacing_q16)
            elif request.operation == "grad-z":
                result = ops.directional_derivative(field, 2, spacing_q16=spacing_q16)
            elif request.operation == "divergence":
                result = ops.divergence(field, spacing_q16=spacing_q16)
            elif request.operation == "curl":
                result = ops.curl(field, spacing_q16=spacing_q16)
            else:
                result = ops.laplacian(field, spacing_q16=spacing_q16)
        except (ValueError, RuntimeError, OverflowError, ZeroDivisionError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _field_op_payload(result, ops)

    @app.post("/v2/qvector/convolve", dependencies=[Depends(authorize)])
    def qvector_convolve(request: QVectorConvolutionRequest) -> dict[str, object]:
        try:
            field = _qvector_field(request.field)
            kernel = QScalarKernel3D.from_floats(request.kernel_weights, request.kernel_shape)
            ops = QVectorFieldOps3D(boundary=_boundary_mode(request.boundary))
            result = ops.convolve(field, kernel)
        except (ValueError, RuntimeError, OverflowError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _field_op_payload(result, ops)

    @app.get("/v1/ledger/verify", dependencies=[Depends(authorize)])
    def verify_ledger() -> dict[str, object]:
        return {
            "valid": cloud.ledger.verify(),
            "records": len(cloud.ledger.records),
            "head": cloud.ledger.records[-1]["digest"] if cloud.ledger.records else None,
        }

    app.state.cloud = cloud
    app.state.qvector = qvector
    return app


app = create_app()
