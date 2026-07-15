import json
import sys

from .parser import Parser
from .assembler import Assembler
from .core import CodexVM
from .api import start_api
from .web import start_web
from .node import CodexNode


def main():
    if len(sys.argv) < 2:
        print("Usage: jarvisx [run|cognitive|api|web|node] <arguments>")
        return

    cmd = sys.argv[1]

    if cmd == "run":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: jarvisx run <assembly-file>")
        file = sys.argv[2]
        with open(file) as source_file:
            source = source_file.read()
        ast = Parser().parse(source)
        bytecode = Assembler().assemble(ast)
        vm = CodexVM()
        vm.load(bytecode)
        vm.run()
        print("Registers:", vm.regs.snapshot())

    elif cmd == "cognitive":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: jarvisx cognitive <number> [number ...]")
        try:
            values = [float(value) for value in sys.argv[2:]]
        except ValueError as exc:
            raise SystemExit("cognitive inputs must be finite numbers") from exc
        vm = CodexVM()
        result = vm.cognitive_cycle(values)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    elif cmd == "api":
        start_api()

    elif cmd == "web":
        start_web()

    elif cmd == "node":
        node = CodexNode()
        node.start()

    else:
        raise SystemExit("Unknown command: {}".format(cmd))


if __name__ == "__main__":
    main()
