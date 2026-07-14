"""FastAPI service for isolated Jarvis-X executions."""

import os
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser

app = FastAPI(title="Jarvis-X", version="0.2.0")


class AssemblyRequest(BaseModel):
    source: str = Field(min_length=1, max_length=100000)
    ann_input: Optional[List[float]] = None
    ann_target: float = 0.0


class ANN30Request(BaseModel):
    input: List[float]
    target: float = 0.0


def _authorize(authorization: Optional[str] = Header(default=None)):
    expected = os.getenv("JARVISX_API_TOKEN")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid bearer token")


@app.get("/health")
def health():
    return {"status": "ok", "service": "jarvisx"}


@app.post("/v1/run/assembly", dependencies=[Depends(_authorize)])
def run_assembly(request: AssemblyRequest):
    try:
        if request.ann_input is not None and len(request.ann_input) > 4096:
            raise ValueError("ANN input exceeds 4096 values")
        bytecode = Assembler().assemble(Parser().parse(request.source))
        vm = CodexVM()
        vm.load(bytecode, ann_input=request.ann_input, ann_target=request.ann_target)
        return vm.run()
    except (ValueError, RuntimeError, FloatingPointError, MemoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/run/ann30d", dependencies=[Depends(_authorize)])
def run_ann30d(request: ANN30Request):
    source = """LOAD30
ENCODE30
PLACE30
FIELD30
PREDICT30
COMPARE30
UPDATE_MEMORY30
PROJECT30
DECODE30
HALT30"""
    try:
        if not request.input or len(request.input) > 4096:
            raise ValueError("ANN input length must be inside [1, 4096]")
        bytecode = Assembler().assemble(Parser().parse(source))
        vm = CodexVM()
        vm.load(bytecode, ann_input=request.input, ann_target=request.target)
        return vm.run()
    except (ValueError, RuntimeError, FloatingPointError, MemoryError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def start_api(host="127.0.0.1", port=8080):
    import uvicorn

    uvicorn.run("jarvisx.api:app", host=host, port=port)
