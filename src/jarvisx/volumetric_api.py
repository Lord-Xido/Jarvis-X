from __future__ import annotations

import base64
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

from .volumetric_ae import ArtifactError, Universal3DAutoEncoder

app = FastAPI(
    title="Jarvis-X 3D Volumetric Auto-Encoding/Decoding API",
    version="1.1.0",
)
engine = Universal3DAutoEncoder()
_DASHBOARD_PATH = Path(__file__).with_name("volumetric_dashboard.html")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    return HTMLResponse(_DASHBOARD_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "engine": "3d-volumetric-aead",
        "capacity_gib": engine.spec.capacity_gib,
        "dashboard": "/",
    }


@app.get("/v1/volumetric/metrics")
def metrics() -> dict[str, object]:
    return engine.spec.metrics()


@app.post("/v1/volumetric/encode")
async def encode(request: Request) -> dict[str, object]:
    payload = await request.body()
    try:
        artifact, receipt = engine.encode(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "receipt": receipt.to_dict(),
        "artifact_base64": base64.b64encode(artifact).decode("ascii"),
    }


@app.post("/v1/volumetric/decode")
async def decode(request: Request) -> Response:
    artifact = await request.body()
    try:
        payload, receipt = engine.decode(artifact)
    except ArtifactError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-JarvisX-SHA256": receipt.payload_sha256,
            "X-JarvisX-Verified": "true",
        },
    )


@app.post("/v1/volumetric/cycle")
async def cycle(request: Request) -> dict[str, object]:
    """Execute encode -> artifact -> decode -> verification in one request."""

    payload = await request.body()
    try:
        artifact, encoded = engine.encode(payload)
        restored, decoded = engine.decode(artifact)
    except (TypeError, ValueError, ArtifactError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "verified": restored == payload and decoded.verified,
        "encode": encoded.to_dict(),
        "decode": decoded.to_dict(),
    }


def main() -> None:
    uvicorn.run("jarvisx.volumetric_api:app", host="0.0.0.0", port=8090)


if __name__ == "__main__":
    main()
