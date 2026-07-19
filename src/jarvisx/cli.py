import json
import sys

from .parser import Parser
from .assembler import Assembler
from .core import CodexVM
from .api import start_api
from .web import start_web
from .node import CodexNode
from .self_evolving_rom import run_self_evolving_rom
from .swarm800 import run_swarm


def main():
    if len(sys.argv) < 2:
        print("Usage: jarvisx [run|api|web|node|swarm|ser] <file|cycles|epochs>")
        return

    cmd = sys.argv[1]

    if cmd == "run":
        if len(sys.argv) < 3:
            raise SystemExit("jarvisx run requires an input file")
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

    elif cmd == "swarm":
        cycles = int(sys.argv[2]) if len(sys.argv) >= 3 and not sys.argv[2].startswith("--") else 1
        mutate = "--no-mutate" not in sys.argv[2:]
        report = run_swarm(cycles=cycles, mutate=mutate)
        print(json.dumps(report, indent=2, sort_keys=True))

    elif cmd == "ser":
        epochs = int(sys.argv[2]) if len(sys.argv) >= 3 else 8
        report = run_self_evolving_rom(max_epochs=epochs)
        print(json.dumps(report, indent=2, sort_keys=True))

    else:
        raise SystemExit("unknown command: %s" % cmd)


if __name__ == "__main__":
    main()
