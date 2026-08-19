from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .assembler import Assembler
from .auto_codec_loop import (
    AutoCodecLoop,
    AutoCodecLoopConfig,
    UniformQuantizedFieldCodec,
)
from .core import CodexVM
from .dr_moagi_field_runtime import DrMoagiFieldConfig, DrMoagiFieldRuntime
from .parser import Parser

app = FastAPI(
    title="Jarvis-X / Dr Moagi Runtime API",
    version="0.1.0",
    description="Deterministic VM and bounded sparse auto-codec runtime.",
)

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Dr Moagi Auto-Codec Runtime</title>
<style>
:root { color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
body { margin:0; background:#080b12; color:#e8eefc; }
main { max-width:1100px; margin:auto; padding:28px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }
.card { background:#111725; border:1px solid #26324a; border-radius:12px; padding:16px; }
label { display:block; font-size:12px; color:#91a0bc; margin-bottom:6px; }
input, textarea, button { width:100%; box-sizing:border-box; border-radius:8px; border:1px solid #33415f; background:#0a1020; color:#eef4ff; padding:10px; }
textarea { min-height:180px; resize:vertical; }
button { cursor:pointer; background:#162b52; font-weight:700; }
button:hover { background:#1c386b; }
.metric { font-size:22px; font-weight:700; }
.muted { color:#91a0bc; font-size:12px; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:420px; overflow:auto; }
.status { display:inline-block; padding:4px 8px; border-radius:999px; border:1px solid #33415f; }
</style>
</head>
<body>
<main>
  <h1>Dr Moagi Auto-Encoding / Decoding Runtime</h1>
  <p class="muted">Executable loop: ingest → encode → decode → residual → field update → verify → journal → repeat.</p>
  <div class="grid">
    <div class="card"><label>Quantization step</label><input id="step" type="number" step="0.01" value="0.10"></div>
    <div class="card"><label>Alpha (closure)</label><input id="alpha" type="number" step="0.1" value="1.0"></div>
    <div class="card"><label>dt</label><input id="dt" type="number" step="0.01" value="0.10"></div>
    <div class="card"><label>Max cycles</label><input id="cycles" type="number" value="64"></div>
    <div class="card"><label>MSE target</label><input id="target" type="number" step="0.0001" value="0.001"></div>
  </div>
  <div class="card" style="margin-top:12px">
    <label>Sparse input cells (JSON array)</label>
    <textarea id="cells">[
  {"x": 2, "y": 2, "z": 2, "value": 0.26},
  {"x": 3, "y": 2, "z": 2, "value": -0.37}
]</textarea>
    <button id="run" style="margin-top:10px">Run closed loop</button>
  </div>
  <div class="grid" style="margin-top:12px">
    <div class="card"><div class="muted">Stop reason</div><div id="stop" class="metric">—</div></div>
    <div class="card"><div class="muted">Cycles</div><div id="cycleCount" class="metric">0</div></div>
    <div class="card"><div class="muted">Final MSE</div><div id="mse" class="metric">—</div></div>
    <div class="card"><div class="muted">Journal</div><div id="journal" class="status">not run</div></div>
  </div>
  <div class="card" style="margin-top:12px"><div class="muted">Runtime receipt</div><pre id="result">Ready.</pre></div>
</main>
<script>
const $ = (id) => document.getElementById(id);
$('run').addEventListener('click', async () => {
  $('run').disabled = true;
  $('result').textContent = 'Executing…';
  try {
    const body = {
      cells: JSON.parse($('cells').value),
      side: 64,
      quantization_step: Number($('step').value),
      alpha: Number($('alpha').value),
      lambda_residual: 0,
      eta: 0,
      dt: Number($('dt').value),
      expand_halo: false,
      max_cycles: Number($('cycles').value),
      reconstruction_mse_target: Number($('target').value),
      stop_on_fixed_point: true
    };
    const response = await fetch('/codec/run', {
      method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
    $('stop').textContent = data.stop_reason;
    $('cycleCount').textContent = data.cycles_executed;
    $('mse').textContent = data.final_reconstruction_mse === null ? '—' : Number(data.final_reconstruction_mse).toExponential(3);
    $('journal').textContent = data.journal_verified ? `verified · ${data.journal_entries} receipts` : 'verification failed';
    $('result').textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    $('result').textContent = String(error);
  } finally {
    $('run').disabled = false;
  }
});
</script>
</body>
</html>"""


class VMRunRequest(BaseModel):
    source: str = ""


class SparseCell(BaseModel):
    x: int
    y: int
    z: int
    value: float


class AutoCodecRequest(BaseModel):
    cells: list[SparseCell] = Field(default_factory=list)
    side: int = Field(default=64, gt=0, le=1000)
    quantization_step: float = Field(default=0.05, gt=0.0)
    alpha: float = Field(default=1.0, ge=0.0)
    lambda_residual: float = Field(default=0.0, ge=0.0)
    eta: float = 0.0
    dt: float = Field(default=0.05, gt=0.0)
    max_active_cells: int = Field(default=100_000, gt=0, le=100_000)
    expand_halo: bool = False
    prune_epsilon: float = Field(default=0.0, ge=0.0)
    max_cycles: int = Field(default=64, gt=0, le=512)
    min_cycles: int = Field(default=1, gt=0, le=512)
    reconstruction_mse_target: float = Field(default=1e-8, ge=0.0)
    max_consecutive_rejections: int = Field(default=3, gt=0, le=100)
    stop_on_fixed_point: bool = True


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "runtime": "jarvisx-auto-codec"}


@app.post("/run")
def run_code(payload: VMRunRequest) -> dict[str, Any]:
    try:
        ast = Parser().parse(payload.source)
        bytecode = Assembler().assemble(ast)
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        return {
            "registers": vm.regs.snapshot(),
            "ledger_entries": len(vm.ledger.chain),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/codec/run")
def run_auto_codec(payload: AutoCodecRequest) -> dict[str, Any]:
    if payload.min_cycles > payload.max_cycles:
        raise HTTPException(status_code=422, detail="min_cycles cannot exceed max_cycles")

    field = {}
    for cell in payload.cells:
        coordinate = (cell.x, cell.y, cell.z)
        if coordinate in field:
            raise HTTPException(
                status_code=422,
                detail=f"duplicate sparse coordinate: {coordinate}",
            )
        field[coordinate] = cell.value

    try:
        codec = UniformQuantizedFieldCodec(step=payload.quantization_step)
        runtime = DrMoagiFieldRuntime(
            codec,
            DrMoagiFieldConfig(
                side=payload.side,
                alpha=payload.alpha,
                lambda_residual=payload.lambda_residual,
                eta=payload.eta,
                dt=payload.dt,
                max_active_cells=payload.max_active_cells,
                expand_halo=payload.expand_halo,
                prune_epsilon=payload.prune_epsilon,
            ),
        )
        loop = AutoCodecLoop(
            runtime,
            AutoCodecLoopConfig(
                max_cycles=payload.max_cycles,
                min_cycles=payload.min_cycles,
                reconstruction_mse_target=payload.reconstruction_mse_target,
                max_consecutive_rejections=payload.max_consecutive_rejections,
                stop_on_fixed_point=payload.stop_on_fixed_point,
            ),
        )
        loop.load(field)
        return loop.run().to_dict()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def start_api() -> None:
    uvicorn.run("jarvisx.api:app", host="0.0.0.0", port=8080)
