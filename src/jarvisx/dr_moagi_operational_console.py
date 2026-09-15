"""Live FastAPI/WebSocket console for the sparse Dr Moagi 3D runtime.

The Python engine is authoritative. The browser only renders telemetry and active-tile
state emitted by this module; it does not fabricate MSE, latency, or voxel positions.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from importlib import resources
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from .dr_moagi_3d_1000x import DrMoagi3D1000xEngine, EngineConfig

DEFAULT_CADENCE_MS = 100
MAX_RENDERED_TILES = 1200


def _tile_xyz(tile_id: int, tiles_per_axis: int) -> tuple[int, int, int]:
    z = tile_id // (tiles_per_axis * tiles_per_axis)
    remainder = tile_id % (tiles_per_axis * tiles_per_axis)
    y = remainder // tiles_per_axis
    x = remainder % tiles_per_axis
    return int(x), int(y), int(z)


class OperationalRuntime:
    """Thread-safe authoritative engine state shared by API and WebSocket clients."""

    def __init__(self, *, max_active_tiles: int = 1000) -> None:
        self.engine = DrMoagi3D1000xEngine(
            EngineConfig(
                active_fraction=0.001,
                tile_edge=10,
                latent_dim=4,
                max_active_tiles=max_active_tiles,
            )
        )
        self.engine.ingest_bytes(
            b"Dr Moagi 3D multimodal multimedia programming engine "
            b"World Encode Omega Decode Residual Correct Schedule Recur"
        )
        self._lock = threading.RLock()
        self._latest = self._step_locked(MAX_RENDERED_TILES)

    def _step_locked(self, max_tiles: int) -> dict[str, Any]:
        report = self.engine.cycle()
        ids = self.engine.active_tile_ids[:max_tiles]
        residuals = self.engine.residual_score[:max_tiles]
        max_residual = float(residuals.max()) if len(residuals) else 1.0
        if max_residual <= 0.0:
            max_residual = 1.0

        tiles: list[dict[str, Any]] = []
        for tile_id, residual in zip(ids.tolist(), residuals.tolist()):
            x, y, z = _tile_xyz(int(tile_id), self.engine.tiles_per_axis)
            tiles.append(
                {
                    "id": int(tile_id),
                    "x": x,
                    "y": y,
                    "z": z,
                    "residual": float(residual),
                    "residual_norm": float(residual) / max_residual,
                }
            )

        control_words = self.engine.packed_control_words()[:8].tolist()
        return {
            "telemetry": asdict(report),
            "tiles": tiles,
            "control_words_sample": [int(value) for value in control_words],
            "contract": {
                "authoritative_state": "python-engine",
                "visualization_role": "render-only",
                "error_determines_computation": True,
                "wall_clock_1000x_guaranteed": False,
            },
        }

    def step(self, max_tiles: int = MAX_RENDERED_TILES) -> dict[str, Any]:
        with self._lock:
            self._latest = self._step_locked(max_tiles)
            return self._latest

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._latest


class ConsoleHub:
    """Single global scheduler that broadcasts one authoritative cycle stream."""

    def __init__(self, runtime: OperationalRuntime) -> None:
        self.runtime = runtime
        self.cadence_ms = DEFAULT_CADENCE_MS
        self.paused = False
        self.clients: set[WebSocket] = set()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            if not self.paused:
                state = await asyncio.to_thread(self.runtime.step)
                dead: list[WebSocket] = []
                for client in tuple(self.clients):
                    try:
                        await client.send_json(state)
                    except Exception:
                        dead.append(client)
                for client in dead:
                    self.clients.discard(client)
            await asyncio.sleep(self.cadence_ms / 1000.0)

    def set_cadence(self, milliseconds: int) -> None:
        self.cadence_ms = max(16, min(1000, int(milliseconds)))


runtime = OperationalRuntime()
hub = ConsoleHub(runtime)
app = FastAPI(title="Dr Moagi 3D Operational Console", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    await hub.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await hub.stop()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (
        resources.files("jarvisx")
        .joinpath("static/dr_moagi_operational_console.html")
        .read_text(encoding="utf-8")
    )
    return HTMLResponse(html)


@app.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(runtime.snapshot())


@app.websocket("/ws")
async def websocket_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    hub.clients.add(websocket)
    await websocket.send_json(runtime.snapshot())
    try:
        while True:
            message = await websocket.receive_json()
            command = str(message.get("type", ""))
            if command == "pause":
                hub.paused = True
            elif command == "resume":
                hub.paused = False
            elif command == "cadence":
                hub.set_cadence(int(message.get("ms", hub.cadence_ms)))
            elif command == "step":
                await websocket.send_json(await asyncio.to_thread(runtime.step))
    except WebSocketDisconnect:
        hub.clients.discard(websocket)


def main() -> int:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
