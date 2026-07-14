"""Length-bounded TCP node for trusted development environments."""

import socket

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser


class CodexNode:
    def __init__(self, host="127.0.0.1", port=9000, max_request_bytes=65536):
        self.host = host
        self.port = int(port)
        self.max_request_bytes = int(max_request_bytes)

    def start(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(5)
            while True:
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(5.0)
                    data = conn.recv(self.max_request_bytes + 1)
                    if len(data) > self.max_request_bytes:
                        conn.sendall(b'{"error":"request too large"}')
                        continue
                    try:
                        source = data.decode("utf-8")
                        bytecode = Assembler().assemble(Parser().parse(source))
                        vm = CodexVM()
                        vm.load(bytecode)
                        response = str(vm.run()).encode("utf-8")
                    except Exception as exc:
                        response = str({"error": str(exc)}).encode("utf-8")
                    conn.sendall(response)
