"""FastAPI control plane for the Jarvis-X 3D HyperCloud runtime."""

from __future__ import annotations

import base64
import binascii

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .extent import HierarchicalAddress
from .multimodal import MediaEnvelope, MediaKind
from .runtime import HyperCloudRuntime

app = FastAPI(
    title="Jarvis-X 3D HyperCloud",
    version="0.1.0",
    description=(
        "Sparse symbolic virtual-parameter control plane with deterministic 3D "
        "routing and typed multimodal ingestion."
    ),
)
runtime = HyperCloudRuntime()


class AddressRequest(BaseModel):
    namespace: str = Field(min_length=1, max_length=256)
    modality: str = Field(min_length=1, max_length=64)
    digits: list[int] = Field(min_length=1, max_length=256)


class ParameterWriteRequest(AddressRequest):
    value: float


class MediaIngestRequest(BaseModel):
    namespace: str = Field(min_length=1, max_length=256)
    kind: MediaKind
    content_type: str = Field(min_length=1, max_length=256)
    payload_base64: str = Field(min_length=1)


def _address(digits: list[int]) -> HierarchicalAddress:
    address = HierarchicalAddress(tuple(digits))
    try:
        address.validate(radix=runtime.extent.address_radix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return address


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/runtime")
def describe_runtime() -> dict[str, object]:
    return runtime.describe()


@app.post("/v1/route")
def route_parameter(request: AddressRequest) -> dict[str, int]:
    coordinate = runtime.route(
        namespace=request.namespace,
        modality=request.modality,
        address=_address(request.digits),
    )
    return {"x": coordinate.x, "y": coordinate.y, "z": coordinate.z}


@app.put("/v1/parameters")
def write_parameter(request: ParameterWriteRequest) -> dict[str, object]:
    address = _address(request.digits)
    coordinate = runtime.route(
        namespace=request.namespace,
        modality=request.modality,
        address=address,
    )
    runtime.store.set(request.namespace, address, request.value)
    return {
        "stored": True,
        "address": address.canonical(),
        "shard": {"x": coordinate.x, "y": coordinate.y, "z": coordinate.z},
        "materialized_parameters": runtime.store.materialized_parameters,
    }


@app.post("/v1/parameters/read")
def read_parameter(request: AddressRequest) -> dict[str, object]:
    address = _address(request.digits)
    return {
        "address": address.canonical(),
        "value": runtime.store.get(request.namespace, address),
    }


@app.post("/v1/media")
def ingest_media(request: MediaIngestRequest) -> dict[str, str | int]:
    try:
        payload = base64.b64decode(request.payload_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="payload_base64 is invalid") from exc

    try:
        envelope = MediaEnvelope(
            kind=request.kind,
            payload=payload,
            content_type=request.content_type,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return runtime.ingest(namespace=request.namespace, media=envelope)
