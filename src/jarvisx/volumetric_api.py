from __future__ import annotations

import base64

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response

from .volumetric_ae import ArtifactError, Universal3DAutoEncoder

app = FastAPI(
    title="Jarvis-X 3D Volumetric Auto-Encoding/Decoding API",
    version="1.0.0",
)
engine = Universal3DAutoEncoder()


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "engine": "3d-volumetric-aead", "capacity_gib": 6400}


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


def main() -> None:
    uvicorn.run("jarvisx.volumetric_api:app", host="0.0.0.0", port=8090)


if __name__ == "__main__":
    main()
