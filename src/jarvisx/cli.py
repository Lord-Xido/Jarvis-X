import sys
from .parser import Parser
from .assembler import Assembler
from .core import CodexVM

def main():
    if len(sys.argv) < 3:
        print("Usage: jarvisx run file.codex")
        return

    cmd, file = sys.argv[1], sys.argv[2]

    with open(file) as f:
        source = f.read()

    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)

    if cmd == "run":
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        print("Registers:", vm.regs.snapshot())
        print("Ledger entries:", len(vm.ledger.chain))
