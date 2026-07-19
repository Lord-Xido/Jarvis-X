import json
import sys

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser


def _usage() -> None:
    print("Usage: jarvisx [run <file>|api|web|node|visual-memory [size]]")


def main() -> None:
    if len(sys.argv) < 2:
        _usage()
        return

    cmd = sys.argv[1]

    if cmd == "run":
        if len(sys.argv) < 3:
            _usage()
            return
        with open(sys.argv[2], encoding="utf-8") as source_file:
            source = source_file.read()
        ast = Parser().parse(source)
        bytecode = Assembler().assemble(ast)
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        print("Registers:", vm.regs.snapshot())

    elif cmd == "api":
        from .api import start_api

        start_api()

    elif cmd == "web":
        from .web import start_web

        start_web()

    elif cmd == "node":
        from .node import CodexNode

        CodexNode().start()

    elif cmd == "visual-memory":
        size = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        vm = CodexVM(ledger_path=None)
        result = vm.run_visual_memory(size=size, auto_optimize=True)
        print(json.dumps(result.summary(), indent=2, sort_keys=True))

    else:
        _usage()


if __name__ == "__main__":
    main()
