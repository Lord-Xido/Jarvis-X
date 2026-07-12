from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .ann import TinyAutoencoder
from .compiler import Assembler
from .demo import DEMO_SOURCE, build_demo_vm, demo_input
from .vm import VANNVirtualMachine


def _demo(_: argparse.Namespace) -> int:
    vm = build_demo_vm(output_sink=lambda text: print(f"Decoded output:\n{text}"))
    vm.set_input(demo_input())
    result = vm.run()
    print(json.dumps({
        "halted": result.halted,
        "cycles": result.cycles,
        "metrics": result.metrics,
        "policy": result.policy,
        "rom": vm.rom.stats(),
    }, indent=2))
    return 0


def _assemble(args: argparse.Namespace) -> int:
    source = Path(args.source).read_text(encoding="utf-8")
    program = Assembler().assemble(source)
    data = b"".join(instruction.encode() for instruction in program.instructions)
    Path(args.output).write_bytes(data)
    print(f"assembled {len(program.instructions)} instructions -> {args.output}")
    return 0


def _run(args: argparse.Namespace) -> int:
    source = Path(args.source).read_text(encoding="utf-8") if args.source else DEMO_SOURCE
    program = Assembler().assemble(source)
    input_values = np.asarray(json.loads(Path(args.input).read_text()), dtype=np.float32)
    if input_values.ndim == 1:
        input_values = input_values[None, :]
    latent_dim = args.latent_dim or max(1, input_values.shape[1] // 3)
    model = TinyAutoencoder(input_values.shape[1], latent_dim, seed=args.seed)
    vm = VANNVirtualMachine(model, output_sink=print)
    vm.load_program(program.instructions)
    vm.set_input(input_values)
    print(json.dumps(vm.run().__dict__, indent=2))
    return 0


def _ide(_: argparse.Namespace) -> int:
    from .ide import main as ide_main

    ide_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vann-rom", description="VANN-ROM Ω³ virtual ANN SDK")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the built-in autoencoder VM demo")
    demo.set_defaults(func=_demo)

    assemble = sub.add_parser("assemble", help="assemble .vann source into 128-bit bytecode")
    assemble.add_argument("source")
    assemble.add_argument("-o", "--output", default="program.vbc")
    assemble.set_defaults(func=_assemble)

    run = sub.add_parser("run", help="run a VANN source program with JSON tensor input")
    run.add_argument("--source")
    run.add_argument("--input", required=True)
    run.add_argument("--latent-dim", type=int)
    run.add_argument("--seed", type=int, default=7)
    run.set_defaults(func=_run)

    ide = sub.add_parser("ide", help="launch the Tkinter virtual IDE")
    ide.set_defaults(func=_ide)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
