"""FastAPI application for the operational Dr Moagi ANN IDE."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dr_moagi_ide import ANNRegistry, EventJournal, ProjectStore, execute_program, refactor_program
from .dr_moagi_os_api import app as os_control_plane

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(os.getenv("JARVISX_IDE_STATIC_DIR", ROOT / "apps/dr-moagi-ide/static"))
DB_PATH = Path(os.getenv("JARVISX_IDE_DB", ROOT / "state/dr-moagi-ide/ide.sqlite3"))

app = FastAPI(title="Dr Moagi ANN IDE", version="1.0.0", description="Bounded Jarvis-X VM, ANN and 3D OS engineering surface.")
projects, events, ann = ProjectStore(DB_PATH), EventJournal(500), ANNRegistry(16)
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/os", os_control_plane)

class VMRunRequest(BaseModel):
    source: str = Field(min_length=1, max_length=262_144)
    max_cycles: int = Field(10_000, ge=1, le=100_000)
    enable_reflex: bool = False

class RefactorRequest(BaseModel):
    source: str = Field(min_length=1, max_length=262_144)
    seed: int = Field(41, ge=0, le=2_147_483_647)
    max_cycles: int = Field(1_000, ge=1, le=10_000)
    max_mutations: int = Field(10, ge=1, le=100)

class ProjectSaveRequest(BaseModel):
    project_id: str | None = Field(None, min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    source: str = Field("", max_length=262_144)

class ANNCreateRequest(BaseModel):
    side: int = Field(6, ge=3, le=10)
    fold_factor: float = Field(1.0, ge=0.0, le=1.0)
    learning_rate: float = Field(0.005, gt=0.0, le=0.25)
    prune_threshold: float = Field(0.15, ge=0.0, le=1.0)
    seed: int = Field(41, ge=0, le=2_147_483_647)

class ANNValuesRequest(BaseModel):
    values: list[float] = Field(min_length=1, max_length=1_000)

class ANNOptimizeRequest(ANNValuesRequest):
    max_epochs: int = Field(20, ge=1, le=50)
    tolerance: float | None = Field(None, gt=0.0, le=1.0)

def fail(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError): return HTTPException(404, "resource not found")
    if isinstance(exc, (TypeError, ValueError)): return HTTPException(422, str(exc))
    return HTTPException(409, str(exc))

@app.get("/", response_class=HTMLResponse)
def index() -> Response:
    page = STATIC_DIR / "index.html"
    return FileResponse(page) if page.is_file() else HTMLResponse("<h1>Dr Moagi ANN IDE</h1><p>API operational.</p>")

@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {"status":"ok","service":"dr-moagi-ann-ide","version":app.version,"time":time.time(),"static_bundle":STATIC_DIR.is_dir(),"telemetry_sequence":events.sequence,"os_control_plane":"/os"}

@app.get("/v1/capabilities")
def capabilities() -> dict[str, Any]:
    return {"vm":{"engine":"CodexVM","opcodes":["SET","ADD","SUB","HALT"],"transactional":True,"cycle_bounded":True,"arbitrary_shell":False},"refactorer":{"deterministic":True,"unsafe_mutation":False},"ann":{"engine":"Inward4DANN","side_range":[3,10],"max_epochs_per_request":50,"max_sessions":16},"persistence":{"engine":"sqlite","projects":True},"telemetry":{"http":"/v1/telemetry","websocket":"/ws/telemetry"},"dr_moagi_os":{"mounted":True,"base_path":"/os"}}

@app.post("/v1/vm/run")
def vm_run(req: VMRunRequest) -> dict[str, Any]:
    try:
        start=time.perf_counter(); out=execute_program(req.source,max_cycles=req.max_cycles,enable_reflex=req.enable_reflex); out["elapsed_ms"]=(time.perf_counter()-start)*1000
        events.emit("vm.run",{"cycles":out["cycles"],"elapsed_ms":out["elapsed_ms"],"bytecode_words":len(out["bytecode"])}); return out
    except Exception as exc:
        events.emit("vm.error",{"error":str(exc)}); raise fail(exc) from exc

@app.post("/v1/refactor")
def refactor(req: RefactorRequest) -> dict[str, Any]:
    try:
        out=refactor_program(req.source,seed=req.seed,max_cycles=req.max_cycles,max_mutations=req.max_mutations); events.emit("refactor.complete",{"proposed":out["mutations_proposed"],"applied":out["mutations_applied"],"rejected":out["mutations_rejected"]}); return out
    except Exception as exc:
        events.emit("refactor.error",{"error":str(exc)}); raise fail(exc) from exc

@app.get("/v1/projects")
def project_list(limit:int=Query(100,ge=1,le=500))->dict[str,Any]: return {"projects":projects.list(limit)}

@app.get("/v1/projects/{project_id}")
def project_get(project_id:str)->dict[str,Any]:
    try: return projects.get(project_id)
    except Exception as exc: raise fail(exc) from exc

@app.put("/v1/projects")
def project_save(req:ProjectSaveRequest)->dict[str,Any]:
    try:
        out=projects.save(project_id=req.project_id,name=req.name,source=req.source); events.emit("project.saved",{"project_id":out["id"],"name":out["name"]}); return out
    except Exception as exc: raise fail(exc) from exc

@app.delete("/v1/projects/{project_id}")
def project_delete(project_id:str)->dict[str,Any]:
    if not projects.delete(project_id): raise HTTPException(404,"resource not found")
    events.emit("project.deleted",{"project_id":project_id}); return {"deleted":True,"project_id":project_id}

@app.post("/v1/ann/sessions")
def ann_create(req:ANNCreateRequest)->dict[str,Any]:
    try:
        out=ann.create(side=req.side,fold_factor=req.fold_factor,learning_rate=req.learning_rate,prune_threshold=req.prune_threshold,seed=req.seed); events.emit("ann.created",out); return out
    except Exception as exc: raise fail(exc) from exc

@app.get("/v1/ann/sessions/{session_id}")
def ann_status(session_id:str)->dict[str,Any]:
    try: return ann.status(session_id)
    except Exception as exc: raise fail(exc) from exc

@app.post("/v1/ann/sessions/{session_id}/evaluate")
def ann_evaluate(session_id:str,req:ANNValuesRequest)->dict[str,Any]:
    try:
        out=ann.evaluate(session_id,req.values); events.emit("ann.evaluate",{"session_id":session_id,"epoch":out["epoch"],"loss":out["metrics"]["loss"]["total"]}); return out
    except Exception as exc:
        events.emit("ann.error",{"session_id":session_id,"error":str(exc)}); raise fail(exc) from exc

@app.post("/v1/ann/sessions/{session_id}/optimize")
def ann_optimize(session_id:str,req:ANNOptimizeRequest)->dict[str,Any]:
    try:
        start=time.perf_counter(); out=ann.optimize(session_id,req.values,max_epochs=req.max_epochs,tolerance=req.tolerance); out["elapsed_ms"]=(time.perf_counter()-start)*1000; report=out["report"]
        events.emit("ann.optimize",{"session_id":session_id,"epoch":out["epoch"],"committed_epochs":report["committed_epochs"],"loss_final":report["final"]["loss"]["total"],"elapsed_ms":out["elapsed_ms"]}); return out
    except Exception as exc:
        events.emit("ann.error",{"session_id":session_id,"error":str(exc)}); raise fail(exc) from exc

@app.delete("/v1/ann/sessions/{session_id}")
def ann_delete(session_id:str)->dict[str,Any]:
    if not ann.delete(session_id): raise HTTPException(404,"resource not found")
    events.emit("ann.deleted",{"session_id":session_id}); return {"deleted":True,"session_id":session_id}

@app.get("/v1/telemetry")
def telemetry(since:int=Query(0,ge=0))->dict[str,Any]: return {"sequence":events.sequence,"events":events.since(since)}

@app.websocket("/ws/telemetry")
async def telemetry_socket(ws:WebSocket)->None:
    await ws.accept(); last=0
    try:
        while True:
            batch=events.since(last)
            if batch: last=batch[-1]["sequence"]
            await ws.send_json({"type":"telemetry","sequence":events.sequence,"events":batch,"heartbeat":time.time()}); await asyncio.sleep(.75)
    except WebSocketDisconnect: return
