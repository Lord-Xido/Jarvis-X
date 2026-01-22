import sys
from .parser import Parser
from .assembler import Assembler
from .core import CodexVM
from .api import start_api
from .web import start_web
from .node import CodexNode

def main():
    if len(sys.argv) < 2:
        print("Usage: jarvisx [run|api|web|node] <file>")
        return

    cmd = sys.argv[1]

    if cmd == "run":
        file = sys.argv[2]
        with open(file) as f:
            source = f.read()
        ast = Parser().parse(source)
        bytecode = Assembler().assemble(ast)
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        print("Registers:", vm.regs.snapshot())

    elif cmd == "api":
        start_api()

    elif cmd == "web":
        start_web()

    elif cmd == "node":
        node = CodexNode()
        node.start()

if __name__ == "__main__":
    main()
