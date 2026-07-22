"""Minimal FastAPI dashboard with no undeclared Flask dependency."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="Jarvis-X Dashboard")

HTML = """
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Jarvis-X</title></head>
<body>
  <h1>Jarvis-X 30D Virtual ANN Processor</h1>
  <p>The operational execution API is available through <code>jarvisx api</code>.</p>
  <ul>
    <li><code>GET /health</code></li>
    <li><code>POST /v1/run/assembly</code></li>
    <li><code>POST /v1/run/ann30d</code></li>
  </ul>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


def start_web(host="127.0.0.1", port=5000):
    import uvicorn

    uvicorn.run("jarvisx.web:app", host=host, port=port)
