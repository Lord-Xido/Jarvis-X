import socket
from .core import CodexVM
from .parser import Parser
from .assembler import Assembler

class CodexNode:
    def __init__(self, host="0.0.0.0", port=9000):
        self.host = host
        self.port = port
        self.vm = CodexVM()

    def start(self):
        s = socket.socket()
        s.bind((self.host, self.port))
        s.listen(5)
        print(f"CodexNode listening on {self.port}")

        while True:
            conn, _ = s.accept()
            data = conn.recv(4096).decode()
            ast = Parser().parse(data)
            bytecode = Assembler().assemble(ast)
            self.vm.load(bytecode)
            self.vm.run()
            conn.send(str(self.vm.regs.snapshot()).encode())
            conn.close()
