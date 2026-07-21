"""FastAPI service for the Jarvis-X VM and tetration field automaton."""
from __future__ import annotations

from threading import RLock
from typing import Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser
from .tetration_field import (
    BRICK_SIZE,
    TetrationAddress,
    TetrationFieldAutomaton,
    TetrationUniverse,
)

app = FastAPI(
    title="Jarvis-X",
    version="0.3.0",
    description="Deterministic VM plus transactional sparse tetration brick field.",
)
_field = TetrationFieldAutomaton()
_field_lock = RLock()


class RunRequest(BaseModel):
    source: str = ""


class BrickInjection(BaseModel):
    chart: str = "origin"
    x: int = 0
    y: int = 0
    z: int = 0
    value: Optional[float] = None
    values: Optional[List[float]] = None


class FieldStepRequest(BaseModel):
    injections: List[BrickInjection] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "jarvisx", "field_cycle": _field.cycle}


@app.get("/universe")
def universe(tower_height: int = 2) -> dict:
    try:
        return TetrationUniverse(height=tower_height).descriptor()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/run")
def run_code(payload: RunRequest) -> dict:
    try:
        bytecode = Assembler().assemble(Parser().parse(payload.source))
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
@app.get("/field")
def field_state() -> dict:
    with _field_lock:
        return _field.snapshot()


def _merge_observation(
    current: Optional[Union[float, List[float]]],
    incoming: Union[float, List[float]],
) -> Union[float, List[float]]:
    if current is None:
        return incoming
    if isinstance(current, list) or isinstance(incoming, list):
        left = current if isinstance(current, list) else [current] * BRICK_SIZE
        right = incoming if isinstance(incoming, list) else [incoming] * BRICK_SIZE
        return [a + b for a, b in zip(left, right)]
    return current + incoming


@app.post("/automaton/step")
@app.post("/field/step")
def field_step(payload: FieldStepRequest) -> dict:
    injections: Dict[TetrationAddress, Union[float, List[float]]] = {}
    try:
        for item in payload.injections:
            if item.values is not None:
                if len(item.values) != BRICK_SIZE:
                    raise ValueError(f"values must contain {BRICK_SIZE} numbers")
                observation: Union[float, List[float]] = item.values
            elif item.value is not None:
                observation = item.value
            else:
                raise ValueError("each injection requires value or values")
            address = TetrationAddress(_field.universe.height, item.chart, item.x, item.y, item.z)
            injections[address] = _merge_observation(injections.get(address), observation)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _field_lock:
        metrics = _field.step(injections)
        response = _field.snapshot()
        response["transaction"] = metrics.to_dict()
        if not metrics.committed:
            raise HTTPException(status_code=409, detail=response)
        return response


def start_api() -> None:
    uvicorn.run("jarvisx.api:app", host="0.0.0.0", port=8080, reload=False)
