"""JARVIS-X 3D WEB runtime. Python 3.10+, standard library only."""
from __future__ import annotations

import base64
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from codec import CodecError, decode, encode, evolve, generate, hash_volume, reconstruction_error, checked_dim

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
ARTIFACTS: OrderedDict[str, bytes] = OrderedDict()
ARTIFACT_LOCK = threading.Lock()
MAX_POST = 1_500_000


def record_artifact(packet: bytes) -> str:
    identifier = secrets.token_urlsafe(16)
    with ARTIFACT_LOCK:
        ARTIFACTS[identifier] = packet
        while len(ARTIFACTS) > 48:
            ARTIFACTS.popitem(last=False)
    return identifier


def response_for(volume: bytes, dim: int, packet: bytes, source: bytes | None = None, rounds=None):
    original = source if source is not None else volume
    mae, mse = reconstruction_error(original, volume)
    return {
        "dimension": dim, "codec": "JX3D v1", "mode": decode(packet).mode,
        "checksum_verified": True, "sha256": hash_volume(volume),
        "raw_bytes": dim**3, "encoded_bytes": len(packet),
        "compression_ratio": round(dim**3/len(packet), 3),
        "mae": round(mae, 6), "mse": round(mse, 6),
        "encoded_id": record_artifact(packet),
        "volume_b64": base64.b64encode(volume).decode("ascii"),
        "rounds": rounds or [],
    }


def simulate(opts: dict):
    dim = checked_dim(opts.get("dimension", 32))
    geometry = opts.get("geometry", "mandelbulb")
    mode = opts.get("mode", "lossless")
    seed = opts.get("seed", 7)
    cycles = opts.get("cycles", 3)
    alpha = opts.get("alpha", 0.2)
    if type(seed) is not int or not (0 <= seed <= 1_000_000):
        raise CodecError("Seed must be an integer from 0 to 1,000,000")
    if type(cycles) is not int or not (1 <= cycles <= 8):
        raise CodecError("Cycles must be an integer from 1 to 8")
    if type(alpha) not in (int,float) or not (0 <= alpha <= 0.5):
        raise CodecError("Alpha must be between 0 and 0.5")
    source = generate(dim, geometry, seed)
    history = []
    for i in range(cycles):
        start = time.perf_counter()
        packet = encode(source, dim, mode)
        out = decode(packet)
        mae, mse = reconstruction_error(source, out.volume)
        history.append({
            "cycle": i+1,
            "bytes": len(packet),
            "ratio": round(len(source)/len(packet), 3),
            "mae": round(mae, 6), "mse": round(mse, 6),
            "sha256": out.sha256,
            "verified": out.checksum_verified and (mode != "lossless" or out.volume == source),
            "duration_ms": round((time.perf_counter()-start)*1000, 2)
        })
        if i < cycles-1:
            source = evolve(out.volume, dim, float(alpha))
    return response_for(out.volume, dim, packet, source, history)


class Handler(BaseHTTPRequestHandler):
    server_version = "JX3D-Web/1.0"

    def _send(self, status: int, payload: bytes, mime: str, attachment: bool=False):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="jarvisx-state.jx3d"')
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, status: int, value):
        self._send(status, json.dumps(value, separators=(",", ":")).encode("utf-8"), "application/json")

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/api/health":
            return self._json(200, {"status":"ok", "engine":"JARVIS-X 3D WEB", "codec":"JX3D v1", "api_version": 1})
        if route.startswith("/api/download/"):
            key = route[len("/api/download/"):]
            with ARTIFACT_LOCK:
                content = ARTIFACTS.get(key)
            if content is None:
                return self._json(404, {"error":"Artifact not found or expired"})
            return self._send(200, content, "application/octet-stream", True)
        names = {"/":"index.html", "/index.html":"index.html", "/app.js":"app.js", "/styles.css":"styles.css"}
        name = names.get(route)
        if name is None:
            return self._json(404, {"error":"Not found"})
        p = STATIC/name
        mime = {".html":"text/html; charset=utf-8", ".js":"application/javascript; charset=utf-8", ".css":"text/css; charset=utf-8"}[p.suffix]
        return self._send(200, p.read_bytes(), mime)

    def do_POST(self):
        route = urlsplit(self.path).path
        try:
            n = int(self.headers.get("Content-Length", "-1"))
            if n < 0 or n > MAX_POST:
                return self._json(413, {"error":"Payload too large or missing length"})
            body = self.rfile.read(n)
            if len(body) != n:
                return self._json(400, {"error":"Incomplete request body"})
            if route == "/api/simulate":
                if n > 16000:
                    raise CodecError("Configuration payload too large")
                opts = json.loads(body)
                if not isinstance(opts, dict):
                    raise CodecError("JSON object expected")
                return self._json(200, simulate(opts))
            if route == "/api/encode":
                dim = checked_dim(int(self.headers.get("X-JX3D-Dimension", "32")))
                mode = self.headers.get("X-JX3D-Mode", "lossless")
                packet = encode(body, dim, mode)
                result = decode(packet)
                return self._json(200, response_for(result.volume, dim, packet, body))
            if route == "/api/decode":
                result = decode(body)
                return self._json(200, response_for(result.volume, result.dim, body))
            return self._json(404, {"error":"Route not found"})
        except (CodecError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
            return self._json(400, {"error":str(exc)})
        except Exception:
            self.log_error("Unhandled runtime error")
            return self._json(500, {"error":"Internal runtime error"})


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"JARVIS-X 3D web runtime listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

if __name__ == "__main__":
    main()
