from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser

app = FastAPI(title="Jarvis-X", version="0.1.0")


class RunRequest(BaseModel):
    source: str = Field(min_length=1, max_length=65536)


class VisualMemoryRequest(BaseModel):
    size: int = Field(default=12, ge=4, le=64)
    auto_optimize: bool = True


def _execute_source(source: str) -> Dict[str, Any]:
    try:
        ast = Parser().parse(source)
        bytecode = Assembler().assemble(ast)
        if not bytecode:
            raise ValueError("source assembled to an empty program")
        vm = CodexVM(ledger_path=None)
        vm.load(bytecode)
        vm.run()
        return {
            "registers": vm.regs.snapshot(),
            "cycles": vm.cycles,
            "ledger_entries": len(vm.ledger.chain),
            "trace_entries": len(vm.tracer.log),
        }
    except (KeyError, ValueError, RuntimeError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/run")
def run_code(request: RunRequest) -> Dict[str, Any]:
    return _execute_source(request.source)


@app.post("/visual-memory")
def visual_memory(request: VisualMemoryRequest) -> Dict[str, Any]:
    try:
        vm = CodexVM(ledger_path=None)
        result = vm.run_visual_memory(
            size=request.size,
            auto_optimize=request.auto_optimize,
        )
        return result.summary()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def start_api(host: str = "127.0.0.1", port: int = 8080) -> None:
    uvicorn.run(app, host=host, port=port)
