from __future__ import annotations

import hashlib
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .dr_moagi_codec_3d import CodecConfig, DrMoagiCodec3D, Volume3D
from .operational import capability_manifest, execute_source

_MAX_SOURCE_CHARS = 1_000_000
_MAX_API_VOXELS = 262_144

_DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Jarvis-X</title></head>
<body>
<h1>Jarvis-X Operational Console</h1>
<p>Deterministic VM + bounded Dr Moagi 3D codec reference.</p>
<textarea id="source" rows="10" cols="70">SET Ψ 10\nSET Φ 20\nADD A Ψ Φ\nHALT</textarea><br>
<button id="run">Run VM</button>
<pre id="result"></pre>
<script>
document.getElementById('run').onclick = async () => {
  const source = document.getElementById('source').value;
  const response = await fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source})
  });
  document.getElementById('result').textContent = JSON.stringify(await response.json(), null, 2);
};
</script>
</body>
</html>
"""

app = FastAPI(
    title="Jarvis-X API",
    version="0.1.0",
    description="Deterministic VM and bounded Dr Moagi 3D codec reference API.",
)


class RunRequest(BaseModel):
    source: str


class CodecRoundTripRequest(BaseModel):
    shape: tuple[int, int, int]
    values: list[float]
    anchor_values: list[float] | None = None
    quant_step: float = 0.25
    max_anchor_mse: float | None = None
    max_rate_bpv: float | None = None
    virtual_depth: int = 1


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _DASHBOARD_HTML


@app.get("/health")
def health() -> dict[str, object]:
    manifest = capability_manifest()
    invariants = manifest["invariants"]
    return {
        "status": "ok",
        "system": manifest["system"],
        "schema": manifest["schema"],
        "authority": manifest["authority"],
        "capabilities": invariants,
    }


@app.post("/run")
def run_code(payload: RunRequest) -> dict[str, object]:
    if len(payload.source) > _MAX_SOURCE_CHARS:
        raise HTTPException(status_code=413, detail="source exceeds API size limit")

    try:
        receipt = execute_source(payload.source)
    except (TypeError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return receipt.to_dict()


@app.post("/codec/roundtrip")
def codec_roundtrip(payload: CodecRoundTripRequest) -> dict[str, object]:
    try:
        source = Volume3D(payload.shape, tuple(payload.values))
        anchor = (
            Volume3D(payload.shape, tuple(payload.anchor_values))
            if payload.anchor_values is not None
            else None
        )
        config = CodecConfig(
            quant_step=payload.quant_step,
            max_voxels=_MAX_API_VOXELS,
            max_anchor_mse=payload.max_anchor_mse,
            max_rate_bpv=payload.max_rate_bpv,
            virtual_depth=payload.virtual_depth,
        )
        runtime = DrMoagiCodec3D(config)
        result = runtime.process(source, anchor=anchor)
    except (OverflowError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "committed": result.committed,
        "rejection_reason": result.rejection_reason,
        "metadata": asdict(result.metadata),
        "metrics": asdict(result.metrics),
        "memory": asdict(result.memory_after),
        "bitstream_digest_sha256": hashlib.sha256(result.bitstream).hexdigest(),
        "virtual_depth": result.virtual_depth,
        "measured_microsteps_executed": result.measured_microsteps_executed,
        "wall_clock_seconds": result.wall_clock_seconds,
        "measured_throughput_voxels_per_second": (
            result.measured_throughput_voxels_per_second
        ),
        "reconstructed_values": list(result.reconstructed.values),
    }


def start_api(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Start the canonical FastAPI service for local development."""

    import uvicorn

    uvicorn.run("jarvisx.api:app", host=host, port=port)
