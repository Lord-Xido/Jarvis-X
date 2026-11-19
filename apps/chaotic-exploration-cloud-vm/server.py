"""Small strict HTTP gateway for a native, bounded JARVIS X 3D bytecode VM."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
BIN = ROOT / 'jarvisx_vm'
LIMITS = {
    'side': (int, 4, 32, 12),
    'steps': (int, 1, 100, 8),
    'seed': (int, 0, 2147483647, 42),
    'horizon': (int, 1, 32, 8),
    'alpha': (float, 0, .166666666666, 0.09),
    'beta': (float, 0, .5, .12),
    'chaos': (float, 0, .2, .03),
    'epsilon': (float, 0, 1, .0001),
}


def validate(body):
    if not isinstance(body, dict) or set(body) - set(LIMITS):
        raise ValueError('Expected an object containing only supported VM parameters')
    args = []
    for key, (typ, low, high, default) in LIMITS.items():
        value = body.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{key} must be numeric')
        if typ is int and (not isinstance(value, int)):
            raise ValueError(f'{key} must be an integer')
        if not low <= value <= high:
            raise ValueError(f'{key} outside [{low}, {high}]')
        args.extend([f'--{key}', str(value)])
    return args


class Handler(BaseHTTPRequestHandler):
    server_version = 'JarvisX-CloudVM/1.0'

    def send_json(self, status, payload):
        data = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/healthz', '/api/status'):
            self.send_json(200, {'ok': BIN.is_file(), 'mode': 'native-container',
                                 'hardware_vms': None, 'cloud_deployed': bool(os.environ.get('K_SERVICE'))})
        elif self.path in ('/', '/index.html'):
            raw = (ROOT / 'static' / 'index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Security-Policy', "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path in ('/app.js', '/style.css'):
            raw = (ROOT / 'static' / self.path.lstrip('/')).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', ('text/css' if self.path == '/style.css' else 'application/javascript') + '; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self.send_json(404, {'error': 'not found'})

    def do_POST(self):
        if self.path != '/api/execute':
            return self.send_json(404, {'error': 'not found'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 2 <= length <= 4096:
                return self.send_json(413, {'error': 'request body size out of bounds'})
            data = json.loads(self.rfile.read(length))
            args = validate(data)
            t0 = time.monotonic()
            result = subprocess.run([str(BIN), *args], capture_output=True, text=True,
                                    timeout=8, check=False, cwd=str(ROOT))
            if result.returncode != 0:
                return self.send_json(422, {'error': 'native VM rejected the request'})
            payload = json.loads(result.stdout)
            payload['gateway_ms'] = round((time.monotonic() - t0) * 1000, 3)
            self.send_json(200, payload)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_json(400, {'error': str(exc)[:160]})
        except subprocess.TimeoutExpired:
            self.send_json(504, {'error': 'bounded execution timeout'})
        except (OSError, KeyError):
            self.send_json(503, {'error': 'native VM unavailable'})


def main():
    port = int(os.environ.get('PORT', '8080'))
    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
    print(f'JARVIS X native VM gateway on {port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
