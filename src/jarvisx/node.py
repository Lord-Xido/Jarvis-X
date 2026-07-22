import json
import socket

from .api import execute_source


class CodexNode:
    def __init__(
        self,
        host="127.0.0.1",
        port=9000,
        max_payload_bytes=65536,
        socket_timeout_s=10.0,
    ):
        self.host = host
        self.port = int(port)
        self.max_payload_bytes = int(max_payload_bytes)
        self.socket_timeout_s = float(socket_timeout_s)

    def _response(self, payload):
        return (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )

    def start(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(5)
            print("CodexNode listening on {}:{}".format(self.host, self.port))

            while True:
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(self.socket_timeout_s)
                    try:
                        data = conn.recv(self.max_payload_bytes + 1)
                        if len(data) > self.max_payload_bytes:
                            raise RuntimeError("Request exceeds node payload limit")
                        source = data.decode("utf-8").strip()
                        if not source:
                            raise RuntimeError("Empty Jarvis-X program")
                        response = {"ok": True, "result": execute_source(source)}
                    except Exception as exc:
                        response = {"ok": False, "error": str(exc)}
                    conn.sendall(self._response(response))
