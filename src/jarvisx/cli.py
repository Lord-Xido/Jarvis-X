import json
import math
import sys

from .assembler import Assembler
from .core import CodexVM
from .parser import Parser
from .serialization import json_safe


def _parse_geometry_arguments(arguments):
    cycles = 4
    values = list(arguments)
    if values.count("--cycles") > 1:
        raise SystemExit("--cycles may be specified only once")
    if "--cycles" in values:
        index = values.index("--cycles")
        try:
            cycles = int(values[index + 1])
        except (IndexError, ValueError) as exc:
            raise SystemExit("--cycles requires a positive integer") from exc
        del values[index : index + 2]
    if cycles < 1:
        raise SystemExit("--cycles requires a positive integer")
    if not values:
        raise SystemExit("Usage: jarvisx geometry3d [--cycles N] <number> [number ...]")
    try:
        parsed = [float(value) for value in values]
    except (OverflowError, ValueError) as exc:
        raise SystemExit("geometry3d inputs must be finite numbers") from exc
    if any(not math.isfinite(value) for value in parsed):
        raise SystemExit("geometry3d inputs must be finite numbers")
    return cycles, parsed


def main():
    if len(sys.argv) < 2:
        print("Usage: jarvisx [run|cognitive|geometry3d|api|web|node] <arguments>")
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
            if any(not math.isfinite(value) for value in values):
                raise ValueError("non-finite cognitive input")
            vm = CodexVM()
            result = vm.cognitive_cycle(values)
        except (OverflowError, ValueError) as exc:
            raise SystemExit("cognitive inputs must be finite numbers") from exc
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    elif cmd == "geometry3d":
        cycles, values = _parse_geometry_arguments(sys.argv[2:])
        try:
            vm = CodexVM()
            results = vm.geometric_feedback(values, cycles)
        except (OverflowError, ValueError) as exc:
            raise SystemExit("geometry3d inputs must be finite and fit the lattice") from exc
        payload = {
            "cycles": [result.to_dict() for result in results],
            "final_state": vm.geometric.snapshot(),
            "registers": vm.regs.snapshot(),
        }
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))

    elif cmd == "api":
        from .api import start_api

        start_api()

    elif cmd == "web":
        from .web import start_web

        start_web()

    elif cmd == "node":
        from .node import CodexNode

        node = CodexNode()
        node.start()

    else:
        raise SystemExit("Unknown command: {}".format(cmd))


if __name__ == "__main__":
    main()
