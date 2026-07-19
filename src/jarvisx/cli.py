import json
import sys

from .api import start_api
from .assembler import Assembler
from .core import CodexVM
from .node import CodexNode
from .parser import Parser
from .web import start_web


def _usage():
    print("Usage: jarvisx [run <file>|api|web|node|visual-memory [size]]")


def main():
    if len(sys.argv) < 2:
        _usage()
        return

    cmd = sys.argv[1]

    if cmd == "run":
        if len(sys.argv) < 3:
            _usage()
            return
        file = sys.argv[2]
        with open(file) as source_file:
            source = source_file.read()
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

    elif cmd == "visual-memory":
        from .geometric_memory import VisualMemoryANN, make_demo_volume

        size = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        volume = make_demo_volume(size)
        result = VisualMemoryANN().permeate(volume, auto_optimize=True)
        print(json.dumps(result.summary(), indent=2, sort_keys=True))

    else:
        _usage()


if __name__ == "__main__":
    main()
