from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .api import RunRequest, execute_source


app = FastAPI(title="Jarvis-X Control Panel", version="0.2.0")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Jarvis-X Control Panel</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }
    textarea { width: 100%; min-height: 18rem; font-family: ui-monospace, monospace; }
    button { margin: 1rem 0; padding: .7rem 1.1rem; }
    pre { white-space: pre-wrap; background: #f3f4f6; padding: 1rem; overflow: auto; }
  </style>
</head>
<body>
  <h1>Jarvis-X Control Panel</h1>
  <textarea id="source">SET A 3
SET B 1
SET C 0
loop:
ADD C C A
SUB A A B
CMP A Ξ
JNZ loop
STORE C 0
LOAD D 0
HALT</textarea>
  <button id="run">Run transaction</button>
  <pre id="result">Ready.</pre>
  <script>
    document.getElementById('run').addEventListener('click', async () => {
      const result = document.getElementById('result');
      result.textContent = 'Running…';
      const response = await fetch('/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({source: document.getElementById('source').value})
      });
      const payload = await response.json();
      result.textContent = JSON.stringify(payload, null, 2);
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/run")
def run_code(request: RunRequest):
    try:
        return execute_source(request.source)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def start_web():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5000)
