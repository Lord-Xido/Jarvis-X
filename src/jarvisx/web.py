import uvicorn
from fastapi.responses import HTMLResponse

from .api import app

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Jarvis-X Control Panel</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 900px; }
    textarea { width: 100%; min-height: 12rem; font-family: monospace; }
    button { margin: .5rem .5rem .5rem 0; padding: .65rem 1rem; }
    pre { background: #111; color: #eee; padding: 1rem; overflow: auto; }
  </style>
</head>
<body>
  <h1>Jarvis-X Control Panel</h1>
  <p>Local development interface. Network exposure requires an external
     authentication and TLS boundary.</p>
  <textarea id="source">SET Ψ 10
SET Φ 20
ADD A Ψ Φ
HALT</textarea>
  <div>
    <button onclick="runVm()">Run VM</button>
    <button onclick="runVisualMemory()">Run 3D Visual Memory</button>
  </div>
  <pre id="output">Ready.</pre>
  <script>
    async function post(path, payload) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(data));
      return data;
    }
    async function runVm() {
      try {
        output.textContent = JSON.stringify(
          await post("/run", {source: source.value}), null, 2
        );
      } catch (error) {
        output.textContent = String(error);
      }
    }
    async function runVisualMemory() {
      try {
        output.textContent = JSON.stringify(
          await post("/visual-memory", {size: 12, auto_optimize: true}),
          null,
          2
        );
      } catch (error) {
        output.textContent = String(error);
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(HTML)


def start_web(host: str = "127.0.0.1", port: int = 5000) -> None:
    uvicorn.run(app, host=host, port=port)
