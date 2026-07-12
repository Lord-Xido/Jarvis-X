from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .aether import (
    AetherConfig,
    AetherEngine,
    AetherInput,
    AetherPolicy,
    GraphTensor,
    synthetic_aether_input,
)
from .ann import TinyAutoencoder
from .compiler import Assembler
from .demo import DEMO_SOURCE, build_demo_vm, demo_input
from .isa import Instruction
from .vm import VANNVirtualMachine


def _load_input(path: str) -> np.ndarray:
    input_values = np.asarray(json.loads(Path(path).read_text(encoding="utf-8")), dtype=np.float32)
    if input_values.ndim == 1:
        input_values = input_values[None, :]
    if input_values.ndim != 2 or input_values.shape[1] < 2:
        raise ValueError("input must be a vector or batch with at least two features")
    return input_values


def _load_aether_input(path: str) -> AetherInput:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        raise ValueError("Aether input requires graph.node_features and graph.adjacency")
    return AetherInput(
        video=np.asarray(payload["video"], dtype=np.float32),
        audio=np.asarray(payload["audio"], dtype=np.float32),
        graph=GraphTensor(
            np.asarray(graph["node_features"], dtype=np.float32),
            np.asarray(graph["adjacency"], dtype=np.float32),
        ),
        context=np.asarray(payload["context"], dtype=np.float32),
    )


def _resolve_latent_dim(input_dim: int, requested: int | None) -> int:
    latent_dim = requested if requested is not None else max(1, input_dim // 3)
    if not 1 <= latent_dim < input_dim:
        raise ValueError(f"latent dimension must be in [1, {input_dim - 1}]")
    return latent_dim


def _execute(
    instructions: list[Instruction],
    input_values: np.ndarray,
    *,
    latent_dim: int | None,
    seed: int,
) -> dict[str, object]:
    model = TinyAutoencoder(
        input_values.shape[1],
        _resolve_latent_dim(input_values.shape[1], latent_dim),
        seed=seed,
    )
    vm = VANNVirtualMachine(model, output_sink=print)
    vm.load_program(instructions)
    vm.set_input(input_values)
    result = vm.run()
    report = result.to_dict()
    report["rom"] = vm.rom.stats()
    return report


def _execute_aether(data: AetherInput, args: argparse.Namespace) -> dict[str, object]:
    engine = AetherEngine(
        AetherConfig(
            hidden_dim=args.hidden_dim,
            latent_dim=args.aether_latent_dim,
            max_tokens=args.max_tokens,
            learning_rate=args.learning_rate,
            max_update_norm=args.max_update_norm,
            semantic_tolerance=args.semantic_tolerance,
            seed=args.seed,
        ),
        policy=AetherPolicy(
            evolution=args.evolution,
            recurrent_steps=args.steps,
            cross_modal_gain=args.cross_modal_gain,
        ),
    )
    result = engine.run(data, adapt=args.adapt, optimize=args.optimize)
    return result.to_dict(include_arrays=args.include_arrays)


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


def _aether_demo(args: argparse.Namespace) -> int:
    report = _execute_aether(synthetic_aether_input(args.seed), args)
    print(json.dumps(report, indent=2))
    return 0


def _aether_run(args: argparse.Namespace) -> int:
    report = _execute_aether(_load_aether_input(args.input), args)
    print(json.dumps(report, indent=2))
    return 0


def _assemble(args: argparse.Namespace) -> int:
    source = Path(args.source).read_text(encoding="utf-8")
    program = Assembler().assemble(source)
    data = Instruction.encode_stream(program.instructions)
    Path(args.output).write_bytes(data)
    print(f"assembled {len(program.instructions)} instructions -> {args.output}")
    return 0


def _run(args: argparse.Namespace) -> int:
    source = Path(args.source).read_text(encoding="utf-8") if args.source else DEMO_SOURCE
    program = Assembler().assemble(source)
    report = _execute(
        program.instructions,
        _load_input(args.input),
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


def _run_bytecode(args: argparse.Namespace) -> int:
    instructions = Instruction.decode_stream(Path(args.bytecode).read_bytes())
    report = _execute(
        instructions,
        _load_input(args.input),
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    instructions = Instruction.decode_stream(Path(args.bytecode).read_bytes())
    for index, instruction in enumerate(instructions):
        print(
            f"{index:04d} {instruction.opcode.name:<18} "
            f"phase={instruction.phase.name:<9} format={instruction.numeric_format.name:<8} "
            f"lambda=0x{instruction.lambda_mask:04x} geo=0x{instruction.geo:06x} "
            f"imm={instruction.immediate} bytes={instruction.encode().hex()}"
        )
    return 0


def _ide(_: argparse.Namespace) -> int:
    from .ide import main as ide_main

    ide_main()
    return 0


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--seed", type=int, default=7)


def _add_aether_arguments(parser: argparse.ArgumentParser, *, require_input: bool) -> None:
    if require_input:
        parser.add_argument("--input", required=True, help="JSON file with video, audio, graph and context")
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--aether-latent-dim", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-update-norm", type=float, default=0.25)
    parser.add_argument("--semantic-tolerance", type=float, default=0.75)
    parser.add_argument("--evolution", choices=("ssm", "euler"), default="ssm")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--cross-modal-gain", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adapt", action="store_true", help="shadow-test and commit a bounded Ω overlay")
    parser.add_argument("--optimize", action="store_true", help="search the bounded latent policy space")
    parser.add_argument("--include-arrays", action="store_true", help="include reconstructed arrays in JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vann-rom", description="VANN-ROM Ω³ virtual ANN SDK")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the built-in autoencoder VM demo")
    demo.set_defaults(func=_demo)

    aether_demo = sub.add_parser("aether-demo", help="run the sparse 4D multimodal Aether Engine")
    _add_aether_arguments(aether_demo, require_input=False)
    aether_demo.set_defaults(func=_aether_demo)

    aether_run = sub.add_parser("aether-run", help="run Aether Engine on normalized JSON modalities")
    _add_aether_arguments(aether_run, require_input=True)
    aether_run.set_defaults(func=_aether_run)

    assemble = sub.add_parser("assemble", help="assemble .vann source into 128-bit bytecode")
    assemble.add_argument("source")
    assemble.add_argument("-o", "--output", default="program.vbc")
    assemble.set_defaults(func=_assemble)

    run = sub.add_parser("run", help="run a VANN source program with JSON tensor input")
    run.add_argument("--source")
    _add_execution_arguments(run)
    run.set_defaults(func=_run)

    run_bytecode = sub.add_parser("run-bytecode", help="execute a CRC-validated .vbc image")
    run_bytecode.add_argument("--bytecode", required=True)
    _add_execution_arguments(run_bytecode)
    run_bytecode.set_defaults(func=_run_bytecode)

    inspect = sub.add_parser("inspect", help="disassemble and inspect a .vbc image")
    inspect.add_argument("bytecode")
    inspect.set_defaults(func=_inspect)

    ide = sub.add_parser("ide", help="launch the Tkinter virtual IDE")
    ide.set_defaults(func=_ide)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
