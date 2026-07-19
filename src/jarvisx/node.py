import socket

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser


class CodexNode:
    def __init__(
        self,
        host="127.0.0.1",
        port=9000,
        max_request_bytes=65536,
        timeout_seconds=10.0,
    ):
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.timeout_seconds = timeout_seconds

    def _execute(self, source):
        vm = CodexVM(ledger_path=None)
        ast = Parser().parse(source)
        bytecode = Assembler().assemble(ast)
        if not bytecode:
            raise ValueError("source assembled to an empty program")
        vm.load(bytecode)
        vm.run()
        return vm.regs.snapshot()

    def start(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(5)
            print(f"CodexNode listening on {self.host}:{self.port}")

            while True:
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(self.timeout_seconds)
                    try:
                        data = conn.recv(self.max_request_bytes + 1)
                        if len(data) > self.max_request_bytes:
                            raise ValueError("request exceeds configured limit")
                        source = data.decode("utf-8")
                        result = self._execute(source)
                        conn.sendall(str(result).encode("utf-8"))
                    except Exception as exc:
                        conn.sendall(f"ERROR: {exc}".encode("utf-8"))
