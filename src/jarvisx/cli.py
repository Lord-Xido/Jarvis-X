"""Command-line interface for the Jarvis-X VM and sparse 3-D automaton."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx",
        description="Jarvis-X deterministic VM and sparse 3-D processing automaton",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="assemble and execute a Jarvis-X source file")
    run_parser.add_argument("file", type=Path)

    subparsers.add_parser("api", help="start the FastAPI service")
    subparsers.add_parser("web", help="start the existing Jarvis-X web interface")
    subparsers.add_parser("node", help="start a Jarvis-X node")

    automaton_parser = subparsers.add_parser(
        "automaton",
        aliases=["simulate"],
        help="run the sparse 10^9000-cell virtual 3-D automaton",
    )
    automaton_parser.add_argument("--steps", type=int, default=12)
    automaton_parser.add_argument("--side", type=int, default=3)
    automaton_parser.add_argument("--amplitude", type=float, default=1.0)
    automaton_parser.add_argument("--seed", type=int, default=1337)
    automaton_parser.add_argument("--latent-dim", type=int, default=8)
    automaton_parser.add_argument("--max-active", type=int, default=10000)
    automaton_parser.add_argument("--auto-optimize", action="store_true")
    automaton_parser.add_argument("--json", action="store_true", dest="as_json")

    subparsers.add_parser(
        "universe",
        help="print the exact virtual-universe descriptor without allocating it",
    )
    return parser


def _run_source(path: Path) -> int:
    from .assembler import Assembler
    from .core import CodexVM
    from .parser import Parser

    source = path.read_text(encoding="utf-8")
    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM()
    vm.load(bytecode)
    vm.run()
    print("Registers:", vm.regs.snapshot())
    return 0


def _run_automaton(args: argparse.Namespace) -> int:
    from .automaton import (
        BoundedMechanicsOptimizer,
        Coordinate3D,
        Mechanics,
        Sparse3DAutomaton,
        make_echo_injections,
    )

    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.max_active < 7:
        raise SystemExit("--max-active must be at least 7")

    mechanics = Mechanics(max_active_cells=args.max_active)
    engine = Sparse3DAutomaton(
        seed=args.seed,
        latent_dim=args.latent_dim,
        mechanics=mechanics,
    )
    pulse = make_echo_injections(
        side=args.side,
        amplitude=args.amplitude,
        origin=Coordinate3D(0, 0, 0),
    )

    optimization = None
    if args.auto_optimize:
        optimizer = BoundedMechanicsOptimizer()
        optimization = optimizer.optimize(engine, [pulse, {}, {}])

    metrics = []
    for step_index in range(args.steps):
        step_inputs = pulse if step_index == 0 else None
        result = engine.step(step_inputs)
        metrics.append(result.to_dict())
        if not result.committed:
            break

    payload = engine.snapshot()
    payload["metrics"] = metrics
    if optimization is not None:
        payload["optimization"] = {
            "adopted": optimization.adopted,
            "baseline_score": optimization.baseline_score,
            "candidate_score": optimization.candidate_score,
            "previous_mechanics": asdict(optimization.previous_mechanics),
            "selected_mechanics": asdict(optimization.selected_mechanics),
        }

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        universe = payload["universe"]
        print("Jarvis-X sparse 3-D auto-encoding/decoding automaton")
        print("Virtual axis:", universe["axis_size"], "positions")
        print("Virtual cells:", universe["virtual_cells"])
        if optimization is not None:
            print(
                "Bounded mechanics optimization:",
                "adopted" if optimization.adopted else "baseline retained",
            )
        for item in metrics:
            status = "COMMIT" if item["committed"] else "ROLLBACK"
            print(
                "cycle={cycle:>3} status={status:<8} active={active_cells:>5} "
                "materialised={materialised_cells:>5} frontier={frontier_cells:>5} "
                "mse={reconstruction_mse:.6f} energy={energy:.6f}".format(
                    status=status,
                    **item
                )
            )
        print("Journal:", payload["journal_hash"])
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        return _run_source(args.file)
    if args.command == "api":
        from .api import start_api

        start_api()
        return 0
    if args.command == "web":
        from .web import start_web

        start_web()
        return 0
    if args.command == "node":
        from .node import CodexNode

        node = CodexNode()
        node.start()
        return 0
    if args.command in ("automaton", "simulate"):
        return _run_automaton(args)
    if args.command == "universe":
        from .automaton import Sparse3DAutomaton

        print(json.dumps(Sparse3DAutomaton.universe_descriptor(), indent=2, sort_keys=True))
        return 0

    parser.error("unknown command: {}".format(args.command))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
