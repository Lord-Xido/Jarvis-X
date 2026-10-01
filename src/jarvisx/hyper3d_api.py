from __future__ import annotations

import base64, binascii
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .hyper3d_runtime import DEFAULT_NODES, DEFAULT_PROGRAM, MAX_NODES, MAX_PAYLOAD, Hyper3DRuntime, Modality, capabilities

INDEX = Path(__file__).resolve().parent / "_static" / "hyper3d" / "index.html"

class Attachment(BaseModel):
    modality: Modality
    name: str = Field(default="payload", min_length=1, max_length=160)
    data_base64: str = Field(min_length=1)

class ExecuteRequest(BaseModel):
    program: str = Field(default=DEFAULT_PROGRAM, min_length=1, max_length=32000)
    text: Optional[str] = Field(default=None, max_length=262144)
    attachments: list[Attachment] = Field(default_factory=list, max_length=16)
    max_active_nodes: int = Field(default=DEFAULT_NODES, ge=1, le=MAX_NODES)

app = FastAPI(title="Jarvis-X Hyper3D", version="0.1.0")

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.get("/api/hyper3d/capabilities")
def get_capabilities():
    return capabilities()

def _decode(a):
    try: raw=base64.b64decode(a.data_base64, validate=True)
    except (binascii.Error,ValueError) as exc: raise HTTPException(400,"invalid base64 for "+a.name) from exc
    if len(raw)>MAX_PAYLOAD: raise HTTPException(413,a.name+" exceeds runtime payload limit")
    return raw

@app.post("/api/hyper3d/execute")
def execute(req:ExecuteRequest):
    r=Hyper3DRuntime(req.max_active_nodes)
    try:
        if req.text: r.ingest(Modality.TEXT,req.text,"prompt.txt")
        for a in req.attachments: r.ingest(a.modality,_decode(a),a.name)
        if not r.nodes: r.ingest(Modality.TEXT,"Jarvis-X Hyper3D empty-input seed","seed.txt")
        return r.execute(req.program)
    except HTTPException: raise
    except (ValueError,RuntimeError) as exc: raise HTTPException(400,str(exc)) from exc

@app.get("/")
def index():
    if not INDEX.is_file(): raise HTTPException(404,"Hyper3D interface is not installed")
    return FileResponse(INDEX,media_type="text/html")
