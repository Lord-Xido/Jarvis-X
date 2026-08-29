"""FastAPI application for the operational Dr Moagi ANN IDE."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dr_moagi_ide import (
    ANNRegistry,
    EventJournal,
    ProjectStore,
    execute_program,
    refactor_program,
)
from .dr_moagi_os_api import app as os_control_plane


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


STATIC_DIR = Path(
    os.getenv("JARVISX_IDE_STATIC_DIR", str(_repo_root() / "apps" / "dr-moagi-ide" / "static"))
)
DB_PATH = Path(
    os.getenv("JARVISX_IDE_DB", str(_repo_root() / "state" / "dr-moagi-ide" / "ide.sqlite3"))
)

app = FastAPI(
    title="Dr Moagi ANN IDE",
    version="1.0.0",
    description=(
        "Bounded end-to-end engineering surface for the Jarvis-X VM, deterministic "
        "code refactoring, inward-4D ANN runtime and Dr Moagi 3D OS control plane."
    ),
)

projects = ProjectStore(DB_PATH)
events = EventJournal(max_events=500)
ann = ANNRegistry(max_sessions=16)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.mount("/os", os_control_plane)


class VMRunRequest(BaseModel):
    source: str = Field(min_length=1, max_length=262_144)
    max_cycles: int = Field(default=10_000, ge=1, le=100_000)
    enable_reflex: bool = False


class RefactorRequest(BaseModel):
    source: str = Field(min_length=1, max_length=262_144)
    seed: int = Field(default=41, ge=0, le=2_147_483_647)
    max_cycles: int = Field(default=1_000, ge=1, le=10_000)
    max_mutations: int = Field(default=10, ge=1, le=100)


class ProjectSaveRequest(BaseModel):
    project_id: str | None = Field(default=None, min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    source: str = Field(default="", max_length=262_144)


class ANNCreateRequest(BaseModel):
    side: int = Field(default=6, ge=3, le=10)
    fold_factor: float = Field(default=1.0, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.005, gt=0.0, le=0.25)
    prune_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    seed: int = Field(default=41, ge=0, le=2_147_483_647)


class ANNValuesRequest(BaseModel):
    values: list[float] = Field(min_length=1, max_length=1_000)


class ANNOptimizeRequest(ANNValuesRequest):
    max_epochs: int = Field(default=20, ge=1, le=50)
    tolerance: float | None = Field(default=None, gt=0.0, le=1.0)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="resource not found")
    if isinstance(exc, (TypeError, ValueError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=409, detail=str(exc))


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse | HTMLResponse:
    page = STATIC_DIR / "index.html"
    if page.is_file():
        return FileResponse(page)
    return HTMLResponse(
        "<h1>Dr Moagi ANN IDE</h1><p>Static bundle unavailable; API is operational.</p>"
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "dr-moagi-ann-ide",
        "version": app.version,
        "time": time.time(),
        "project_store": str(DB_PATH),
        "static_bundle": STATIC_DIR.is_dir(),
        "telemetry_sequence": events.sequence,
        "os_control_plane": "/os",
    }


@app.get("/v1/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "vm": {
            "engine": "CodexVM",
            "opcodes": ["SET", "ADD", "SUB", "HALT"],
            "transactional": True,
            "cycle_bounded": True,
            "arbitrary_shell": False,
        },
        "refactorer": {
            "deterministic": True,
            "transforms": ["dead_code_elimination", "const_propagation"],
            "unsafe_mutation": False,
        },
        "ann": {
            "engine": "Inward4DANN",
            "side_range": [3, 10],
            "max_epochs_per_request": 50,
            "max_sessions": 16,
        },
        "persistence": {"engine": "sqlite", "projects": True},
        "telemetry": {"http": "/v1/telemetry", "websocket": "/ws/telemetry"},
        "dr_moagi_os": {"mounted": True, "base_path": "/os"},
    }


@app.post("/v1/vm/run")
def vm_run(request: VMRunRequest) -> dict[str, Any]:
    try:
        started = time.perf_counter()
        result = execute_program(
            request.source,
            max_cycles=request.max_cycles,
            enable_reflex=request.enable_reflex,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["elapsed_ms"] = elapsed_ms
        events.emit(
            "vm.run",
            {
                "cycles": result["cycles"],
                "elapsed_ms": elapsed_ms,
                "bytecode_words": len(result["bytecode"]),
            },
        )
        return result
    except Exception as exc:
        events.emit("vm.error", {"error": str(exc)})
        raise _http_error(exc) from exc


@app.post("/v1/refactor")
def refactor(request: RefactorRequest) -> dict[str, Any]:
    try:
        result = refactor_program(
            request.source,
            seed=request.seed,
            max_cycles=request.max_cycles,
            max_mutations=request.max_mutations,
        )
        events.emit(
            "refactor.complete",
            {
                "proposed": result["mutations_proposed"],
                "applied": result["mutations_applied"],
                "rejected": result["mutations_rejected"],
            },
        )
        return result
    except Exception as exc:
        events.emit("refactor.error", {"error": str(exc)})
        raise _http_error(exc) from exc


@app.get("/v1/projects")
def project_list(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    return {"projects": projects.list(limit)}


@app.get("/v1/projects/{project_id}")
def project_get(project_id: str) -> dict[str, Any]:
    try:
        return projects.get(project_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.put("/v1/projects")
def project_save(request: ProjectSaveRequest) -> dict[str, Any]:
    try:
        project = projects.save(
            project_id=request.project_id,
            name=request.name,
            source=request.source,
        )
        events.emit("project.saved", {"project_id": project["id"], "name": project["name"]})
        return project
    except Exception as exc:
        raise _http_error(exc) from exc


@app.delete("/v1/projects/{project_id}")
def project_delete(project_id: str) -> dict[str, Any]:
    deleted = projects.delete(project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="resource not found")
    events.emit("project.deleted", {"project_id": project_id})
    return {"deleted": True, "project_id": project_id}


@app.post("/v1/ann/sessions")
def ann_create(request: ANNCreateRequest) -> dict[str, Any]:
    try:
        session = ann.create(
            side=request.side,
            fold_factor=request.fold_factor,
            learning_rate=request.learning_rate,
            prune_threshold=request.prune_threshold,
            seed=request.seed,
        )
        events.emit("ann.created", session)
        return session
    except Exception as exc:
        raise _http_error(exc) from exc


@app.get("/v1/ann/sessions/{session_id}")
def ann_status(session_id: str) -> dict[str, Any]:
    try:
        return ann.status(session_id)
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/v1/ann/sessions/{session_id}/evaluate")
def ann_evaluate(session_id: str, request: ANNValuesRequest) -> dict[str, Any]:
    try:
        result = ann.evaluate(session_id, request.values)
        events.emit(
            "ann.evaluate",
            {
                "session_id": session_id,
                "epoch": result["epoch"],
                "loss": result["metrics"]["loss"]["total"],
            },
        )
        return result
    except Exception as exc:
        events.emit("ann.error", {"session_id": session_id, "error": str(exc)})
        raise _http_error(exc) from exc


@app.post("/v1/ann/sessions/{session_id}/optimize")
def ann_optimize(session_id: str, request: ANNOptimizeRequest) -> dict[str, Any]:
    try:
        started = time.perf_counter()
        result = ann.optimize(
            session_id,
            request.values,
            max_epochs=request.max_epochs,
            tolerance=request.tolerance,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result["elapsed_ms"] = elapsed_ms
        report = result["report"]
        events.emit(
            "ann.optimize",
            {
                "session_id": session_id,
                "epoch": result["epoch"],
                "committed_epochs": report["committed_epochs"],
                "loss_final": report["final"]["loss"]["total"],
                "elapsed_ms": elapsed_ms,
            },
        )
        return result
    except Exception as exc:
        events.emit("ann.error", {"session_id": session_id, "error": str(exc)})
        raise _http_error(exc) from exc


@app.delete("/v1/ann/sessions/{session_id}")
def ann_delete(session_id: str) -> dict[str, Any]:
    if not ann.delete(session_id):
        raise HTTPException(status_code=404, detail="resource not found")
    events.emit("ann.deleted", {"session_id": session_id})
    return {"deleted": True, "session_id": session_id}


@app.get("/v1/telemetry")
def telemetry(since: int = Query(default=0, ge=0)) -> dict[str, Any]:
    return {"sequence": events.sequence, "events": events.since(since)}


@app.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    last_sequence = 0
    try:
        while True:
            batch = events.since(last_sequence)
            if batch:
                last_sequence = batch[-1]["sequence"]
            await websocket.send_json(
                {
                    "type": "telemetry",
                    "sequence": events.sequence,
                    "events": batch,
                    "heartbeat": time.time(),
                }
            )
            await asyncio.sleep(0.75)
    except WebSocketDisconnect:
        return
