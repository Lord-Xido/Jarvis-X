from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser


class RunRequest(BaseModel):
    source: str


def execute_source(source):
    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM()
    vm.load(bytecode)
    vm.run()
    return vm.state_snapshot()


app = FastAPI(title="Jarvis-X API", version="0.2.0")


@app.get("/health")
def health():
    return {"status": "ok", "runtime": "jarvis-x"}


@app.post("/run")
def run_code(request: RunRequest):
    try:
        return execute_source(request.source)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def start_api():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)
