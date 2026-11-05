"""FastAPI control plane for the Jarvis-X MMVM kernel."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

from .mmvm import MMVMKernel, decode_base64


DB_PATH = os.getenv("MMVM_DB", "/data/mmvm.sqlite3")
FRONTEND_PATH = Path(
    os.getenv("MMVM_FRONTEND", "/app/apps/mmvm-fullstack/index.html")
)
WORKER_INTERVAL = max(0.01, float(os.getenv("MMVM_WORKER_INTERVAL", "0.05")))

kernel = MMVMKernel(DB_PATH)
_worker_task: asyncio.Task[None] | None = None


async def worker_loop() -> None:
    while True:
        metrics = await asyncio.to_thread(kernel.run_next)
        if metrics is None:
            await asyncio.sleep(WORKER_INTERVAL)
        else:
            await asyncio.sleep(0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _worker_task
    _worker_task = asyncio.create_task(worker_loop(), name="mmvm-scheduler")
    try:
        yield
    finally:
        if _worker_task is not None:
            _worker_task.cancel()
            try:
                await _worker_task
            except asyncio.CancelledError:
                pass
        kernel.memory.close()


app = FastAPI(
    title="Jarvis-X MMVM",
    version="0.1.0",
    description="Auto-encoding/decoding multimodal virtual-computer control plane",
    lifespan=lifespan,
)


class SubmitRequest(BaseModel):
    text: str | None = None
    payload_base64: str | None = None
    modality: str = Field(default="text", min_length=1, max_length=32)
    generate: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if FRONTEND_PATH.exists():
        return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>Jarvis-X MMVM</h1><p>Kernel online. Frontend asset not mounted.</p>"
    )


@app.get("/health")
def health() -> dict[str, Any]:
    status = kernel.status()
    return {
        "ok": True,
        "system": status["system"],
        "cycle": status["cycle"],
        "queue_depth": status["queue_depth"],
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    return cast(dict[str, Any], kernel.status())


@app.post("/api/submit", status_code=202)
def submit(request: SubmitRequest) -> dict[str, Any]:
    try:
        if (request.text is None) == (request.payload_base64 is None):
            raise ValueError("provide exactly one of text or payload_base64")
        payload = (
            request.text.encode("utf-8")
            if request.text is not None
            else decode_base64(request.payload_base64 or "")
        )
        task = kernel.submit(
            payload,
            modality=request.modality,
            target=request.generate,
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return cast(dict[str, Any], task.public())


@app.post("/api/cycle")
async def cycle() -> dict[str, Any]:
    metrics = await asyncio.to_thread(kernel.run_next)
    return {"metrics": metrics.public() if metrics else None, "status": kernel.status()}


@app.get("/api/tasks")
def tasks(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], kernel.tasks(limit))


@app.get("/api/tasks/{task_id}")
def task(task_id: str) -> dict[str, Any]:
    result = kernel.task(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="task not found")
    return cast(dict[str, Any], result)


@app.get("/api/memory")
def memory() -> dict[str, Any]:
    return cast(dict[str, Any], kernel.memory.stats())


@app.get("/api/objects/{object_id}")
def object_metadata(object_id: str) -> dict[str, Any]:
    result = kernel.memory.fetch_object(object_id)
    if result is None:
        raise HTTPException(status_code=404, detail="object not found")
    result = dict(result)
    result.pop("encoded", None)
    latent = result.pop("latent", ())
    result["latent_dimensions"] = len(latent)
    result["latent_preview"] = [round(float(value), 6) for value in latent[:12]]
    return result


@app.get("/api/artifacts/{artifact_id}")
def artifact(artifact_id: str) -> Response:
    result = kernel.memory.fetch_artifact(artifact_id)
    if result is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return Response(
        content=result.payload,
        media_type=result.media_type,
        headers={
            "Content-Disposition": f'inline; filename="{result.filename}"',
            "X-Content-SHA256": result.sha256,
        },
    )


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(kernel.status())
            await asyncio.sleep(0.10)
    except WebSocketDisconnect:
        return
