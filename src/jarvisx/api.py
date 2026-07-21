"""FastAPI service for the Jarvis-X VM and sparse 3-D automaton."""

from __future__ import annotations

from threading import RLock
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .assembler import Assembler
from .automaton import Coordinate3D, Sparse3DAutomaton
from .core import CodexVM
from .parser import Parser

app = FastAPI(
    title="Jarvis-X",
    version="0.2.0",
    description="Deterministic VM plus transactional sparse 3-D autoencoder automaton.",
)
_automaton = Sparse3DAutomaton()
_automaton_lock = RLock()


class RunRequest(BaseModel):
    source: str = ""


class CellInjection(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    z: int = Field(ge=0)
    value: float


class AutomatonStepRequest(BaseModel):
    injections: List[CellInjection] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "jarvisx",
        "automaton_cycle": _automaton.cycle,
    }


@app.post("/run")
def run_code(payload: RunRequest) -> dict:
    try:
        ast = Parser().parse(payload.source)
        bytecode = Assembler().assemble(ast)
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        return {
            "registers": vm.regs.snapshot(),
            "ledger_entries": len(vm.ledger.chain),
            "cycles": vm.cycles,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/automaton")
def automaton_state() -> dict:
    with _automaton_lock:
        return _automaton.snapshot()


@app.post("/automaton/step")
def automaton_step(payload: AutomatonStepRequest) -> dict:
    injections = {}
    try:
        for item in payload.injections:
            coordinate = Coordinate3D(item.x, item.y, item.z)
            injections[coordinate] = injections.get(coordinate, 0.0) + item.value
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _automaton_lock:
        metrics = _automaton.step(injections)
        response = _automaton.snapshot()
        response["transaction"] = metrics.to_dict()
        if not metrics.committed:
            raise HTTPException(status_code=409, detail=response)
        return response


def start_api() -> None:
    uvicorn.run("jarvisx.api:app", host="0.0.0.0", port=8080, reload=False)
