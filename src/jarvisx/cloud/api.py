"""Operational FastAPI control plane for the Jarvis-X 3D HyperCloud runtime."""

from __future__ import annotations

import base64
import binascii
import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field

from .extent import HierarchicalAddress
from .multimodal import MediaEnvelope, MediaKind
from .operational import OperationalHyperCloud

app = FastAPI(
    title="Jarvis-X 3D HyperCloud",
    version="0.2.0",
    description=(
        "Operational sparse virtual-parameter cloud with durable multimedia state, "
        "deterministic 3D routing and asynchronous model/codec execution."
    ),
)
runtime = OperationalHyperCloud.from_environment()


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
    payload_base64: str = Field(min_length=1, max_length=64_000_000)


class CodecJobRequest(BaseModel):
    namespace: str = Field(min_length=1, max_length=256)
    media_sha256: str = Field(min_length=64, max_length=64)


class ChatJobRequest(BaseModel):
    namespace: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=1_000_000)
    system: str | None = Field(default=None, max_length=100_000)


def _address(digits: list[int]) -> HierarchicalAddress:
    address = HierarchicalAddress(tuple(digits))
    try:
        address.validate(radix=runtime.extent.address_radix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return address


def _authorize(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    expected = os.getenv("JARVISX_API_KEY", "")
    if not expected:
        return
    supplied = x_api_key
    if supplied is None and authorization and authorization.startswith("Bearer "):
        supplied = authorization[7:]
    if supplied is None or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid API key")


def _backend_name() -> str:
    return (
        "openai-compatible"
        if os.getenv("JARVISX_MODEL_BASE_URL", "").strip()
        else "local-reference-non-llm"
    )


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readiness() -> dict[str, str]:
    assert runtime.state is not None
    if not runtime.state.ping():
        raise HTTPException(status_code=503, detail="state store unavailable")
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> Response:
    assert runtime.state is not None
    jobs = runtime.state.job_counts()
    lines = [
        "# HELP jarvisx_hypercloud_materialized_parameters Sparse materialized parameter count.",
        "# TYPE jarvisx_hypercloud_materialized_parameters gauge",
        f"jarvisx_hypercloud_materialized_parameters {runtime.state.parameter_count()}",
        "# HELP jarvisx_hypercloud_media_objects Content-addressed media object count.",
        "# TYPE jarvisx_hypercloud_media_objects gauge",
        f"jarvisx_hypercloud_media_objects {runtime.state.media_count()}",
        "# HELP jarvisx_hypercloud_jobs Jobs by durable state.",
        "# TYPE jarvisx_hypercloud_jobs gauge",
    ]
    for status, count in sorted(jobs.items()):
        lines.append(f'jarvisx_hypercloud_jobs{{status="{status}"}} {count}')
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/v1/runtime", dependencies=[Depends(_authorize)])
def describe_runtime() -> dict[str, object]:
    description = runtime.describe(backend_name=_backend_name())
    description["authentication"] = {
        "api_key_required": bool(os.getenv("JARVISX_API_KEY", ""))
    }
    return description


@app.post("/v1/route", dependencies=[Depends(_authorize)])
def route_parameter(request: AddressRequest) -> dict[str, int]:
    coordinate = runtime.route(
        namespace=request.namespace,
        modality=request.modality,
        address=_address(request.digits),
    )
    return {"x": coordinate.x, "y": coordinate.y, "z": coordinate.z}


@app.put("/v1/parameters", dependencies=[Depends(_authorize)])
def write_parameter(request: ParameterWriteRequest) -> dict[str, object]:
    address = _address(request.digits)
    coordinate = runtime.route(
        namespace=request.namespace,
        modality=request.modality,
        address=address,
    )
    runtime.set_parameter(request.namespace, address, request.value)
    assert runtime.state is not None
    return {
        "stored": True,
        "address": address.canonical(),
        "shard": {"x": coordinate.x, "y": coordinate.y, "z": coordinate.z},
        "materialized_parameters": runtime.state.parameter_count(),
    }


@app.post("/v1/parameters/read", dependencies=[Depends(_authorize)])
def read_parameter(request: AddressRequest) -> dict[str, object]:
    address = _address(request.digits)
    return {
        "address": address.canonical(),
        "value": runtime.get_parameter(request.namespace, address),
    }


@app.post("/v1/media", dependencies=[Depends(_authorize)])
def ingest_media(request: MediaIngestRequest) -> dict[str, str | int | float]:
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

    return runtime.ingest(request.namespace, envelope)


@app.get("/v1/media/{digest}", dependencies=[Depends(_authorize)])
def media_metadata(digest: str) -> dict[str, str | int | float]:
    assert runtime.state is not None
    record = runtime.state.media_record(digest)
    if record is None:
        raise HTTPException(status_code=404, detail="media object not found")
    return record


@app.post("/v1/jobs/codec", dependencies=[Depends(_authorize)])
def enqueue_codec(request: CodecJobRequest) -> dict[str, object]:
    try:
        job = runtime.enqueue_codec(request.namespace, request.media_sha256)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="media object not found") from exc
    return job.as_dict()


@app.post("/v1/chat", dependencies=[Depends(_authorize)])
def enqueue_chat(request: ChatJobRequest) -> dict[str, object]:
    try:
        job = runtime.enqueue_chat(request.namespace, request.prompt, request.system)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return job.as_dict()


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(_authorize)])
def get_job(job_id: str) -> dict[str, object]:
    job = runtime.job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.as_dict()
