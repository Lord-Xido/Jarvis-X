"""Public demonstration server: stateless API, aggregate-only data contract.
No accounts, persistent customer storage, order execution or payment connections.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from engine import compute, InputError
from demo import sample_payload

ROOT=Path(__file__).resolve().parent
MAX_BODY=192*1024
HTML=(ROOT/"index.html").read_bytes()

class Handler(BaseHTTPRequestHandler):
    server_version="JarvisXCommerceDemo/1.0"
    sys_version=""
    def _headers(self, code, content_type, size):
        self.send_response(code)
        for k,v in {
            "Content-Type":content_type,
            "Content-Length":str(size),
            "Cache-Control":"no-store",
            "X-Content-Type-Options":"nosniff",
            "X-Frame-Options":"DENY",
            "Referrer-Policy":"no-referrer",
            "Content-Security-Policy":"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
        }.items(): self.send_header(k,v)
        self.end_headers()
    def _reply(self, code, data):
        body=json.dumps(data,allow_nan=False,separators=(",",":")).encode("utf-8")
        self._headers(code,"application/json; charset=utf-8",len(body))
        self.wfile.write(body)
    def do_GET(self):
        path=urlsplit(self.path).path
        if path in ("/","/index.html"):
            self._headers(200,"text/html; charset=utf-8",len(HTML))
            return self.wfile.write(HTML)
        if path=="/healthz":
            return self._reply(200,{"ok":True,"version":"1.0.0","mode":"stateless_demo"})
        if path=="/api/demo":
            sample=sample_payload()
            return self._reply(200,{"input":sample,"plan":compute(sample)})
        return self._reply(404,{"error":"not_found"})
    def do_POST(self):
        if urlsplit(self.path).path!="/api/plan":
            return self._reply(404,{"error":"not_found"})
        try:
            if not self.headers.get("Content-Type","").split(";")[0].strip()=="application/json":
                raise InputError("Content-Type must be application/json")
            length=self.headers.get("Content-Length")
            if length is None or not length.isdecimal() or int(length)>MAX_BODY or int(length)==0:
                raise InputError("Request body must be 1..192 KiB")
            document=json.loads(self.rfile.read(int(length)),parse_constant=lambda _: (_ for _ in ()).throw(InputError("Nonfinite JSON number")))
            result=compute(document)
            return self._reply(200,result)
        except (InputError,ValueError,UnicodeDecodeError,json.JSONDecodeError) as e:
            return self._reply(400,{"error":"invalid_request","message":str(e)[:200]})
    def log_message(self, fmt, *args):
        # Do not log full URLs, request bodies, or raw client-provided SKU data.
        print("request handled",flush=True)

if __name__=="__main__":
    port=int(os.getenv("PORT","8080"))
    if port<1 or port>65535: raise SystemExit("Invalid PORT")
    server=ThreadingHTTPServer(("0.0.0.0",port),Handler)
    server.daemon_threads=True
    print(f"Jarvis X Commerce demo listening on port {port}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
