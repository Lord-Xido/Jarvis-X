"""Bounded HTTP bridge for the JARVIS X Phi3D native VM.

Cloud authentication is delegated to Cloud Run IAM. Bind to localhost when used
without a private Cloud Run endpoint; a local optional control token is supported.
"""
import hmac
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent
BIN = ROOT / 'jarvisx_vm'
MAX_BODY_BYTES = 131072
MAX_PARALLEL_RUNS = 4
SLOTS = threading.BoundedSemaphore(MAX_PARALLEL_RUNS)
LIMITS = {
    'side': (int, 4, 32, 12),
    'steps': (int, 1, 100, 8),
    'seed': (int, 0, 2147483647, 42),
    'horizon': (int, 1, 32, 8),
    'alpha': (float, 0.0, 1.0 / 6.0, 0.09),
    'beta': (float, 0.0, 0.5, 0.12),
    'chaos': (float, 0.0, 0.2, 0.03),
    'epsilon': (float, 0.0, 1.0, 0.0001),
    'decay': (float, 0.0, 1.0, 0.94),
}


def validate(body):
    """Return (bounded native argv, optional serialized volume)."""
    extras = {'values', 'emit_field', 'trace'}
    if not isinstance(body, dict) or set(body) - (set(LIMITS) | extras):
        raise ValueError('Expected an object containing only supported VM parameters')
    args = []
    for key, (typ, low, high, default) in LIMITS.items():
        value = body.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{key} must be numeric')
        if typ is int and not isinstance(value, int):
            raise ValueError(f'{key} must be an integer')
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} outside [{low}, {high}]')
        args.extend([f'--{key}', str(value)])
    if 6 * float(body.get('alpha', 0.09)) + float(body.get('beta', 0.12)) > 1 + 1e-12:
        raise ValueError('Spatial CFL bound violated: 6*alpha + beta <= 1 required')

    side = body.get('side', 12)
    voxels = None
    if 'values' in body:
        values = body['values']
        if side > 16 or not isinstance(values, list) or len(values) != side ** 3:
            raise ValueError('Explicit voxels require side <=16 and exactly side^3 values')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or
               not math.isfinite(v) or abs(v) > 1 for v in values):
            raise ValueError('Each explicit voxel must be a finite number in [-1,1]')
        # There is no shell, no filesystem path and no code evaluation in this input.
        voxels = ' '.join(str(v) for v in values) + '\n'
        args += ['--input-stdin', '1']

    for key in ('emit_field', 'trace'):
        value = body.get(key, False if key == 'emit_field' else True)
        if not isinstance(value, bool):
            raise ValueError(f'{key} must be boolean')
        if key == 'emit_field' and value and side > 16:
            raise ValueError('Full 3D field export requires side <= 16')
        args += [f'--{key.replace("_", "-")}', '1' if value else '0']
    return args, voxels


class Handler(BaseHTTPRequestHandler):
    server_version = 'JarvisX-Phi3D/2.0'

    def send_json(self, status, payload):
        data = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        secret = os.environ.get('JX_VM_CONTROL_TOKEN')
        if secret and not hmac.compare_digest(self.headers.get('X-JX-Control-Token', ''), secret):
            self.send_json(401, {'error': 'authentication required'})
            return False
        return True

    def do_GET(self):
        if self.path == '/healthz':
            return self.send_json(200 if BIN.is_file() else 503, {'ok': BIN.is_file()})
        if not self.authorized():
            return
        if self.path == '/api/status':
            return self.send_json(200, {
                'ok': BIN.is_file(), 'engine': 'jarvisx-phi3d-native-cpu-v2',
                'mode': 'native-container', 'cloud_deployed': bool(os.environ.get('K_SERVICE')),
                'hardware_vms': None, 'max_parallel_runs': MAX_PARALLEL_RUNS,
                'trained_model': False,
            })
        static = {'/': ('index.html', 'text/html; charset=utf-8'),
                  '/index.html': ('index.html', 'text/html; charset=utf-8'),
                  '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if self.path not in static:
            return self.send_json(404, {'error': 'not found'})
        filename, media = static[self.path]
        raw = (ROOT / 'static' / filename).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', media)
        self.send_header('Content-Security-Policy',
                         "default-src 'none'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        self.send_header('X-Content-Length-Options', 'nosniff')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path != '/api/execute':
            return self.send_json(404, {'error': 'not found'})
        if not self.authorized():
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            return self.send_json(415, {'error': 'Content-Type must be application/json'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 2 <= length <= MAX_BODY_BYTES:
                return self.send_json(413, {'error': 'request body size out of bounds'})
            data = json.loads(self.rfile.read(length))
            args, input_voxels = validate(data)
            if not SLOTS.acquire(blocking=False):
                return self.send_json(429, {'error': 'max concurrent VM executions reached'})
            try:
                t0 = time.monotonic()
                result = subprocess.run([str(BIN), *args], input=input_voxels,
                                        capture_output=True, text=True, timeout=8,
                                        check=False, cwd=str(ROOT))
            finally:
                SLOTS.release()
            if result.returncode != 0:
                return self.send_json(422, {'error': 'native VM rejected request'})
            payload = json.loads(result.stdout)
            payload['gateway_ms'] = round((time.monotonic() - t0) * 1000, 3)
            self.send_json(200, payload)
        except (ValueError, UnicodeDecodeError) as exc:
            self.send_json(400, {'error': str(exc)[:160]})
        except subprocess.TimeoutExpired:
            self.send_json(504, {'error': 'bounded execution timeout'})
        except (OSError, KeyError):
            self.send_json(503, {'error': 'native VM unavailable'})


def main():
    port = int(os.environ.get('PORT', '8080'))
    host = os.environ.get('HOST', '127.0.0.1')
    # Cloud Run requires listening on 0.0.0.0:$PORT; local defaults to loopback.
    if os.environ.get('K_SERVICE'):
        host = '0.0.0.0'
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print(f'JARVIS X Phi3D VM gateway on {host}:{port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
