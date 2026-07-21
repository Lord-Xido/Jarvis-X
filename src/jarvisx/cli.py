"""Command-line interface for the Jarvis-X VM and tetration field automaton."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx",
        description="Jarvis-X deterministic VM and sparse tetration field automaton",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="assemble and execute a Jarvis-X source file")
    run_parser.add_argument("file", type=Path)
    subparsers.add_parser("api", help="start the FastAPI service")
    subparsers.add_parser("web", help="start the existing Jarvis-X web interface")
    subparsers.add_parser("node", help="start a Jarvis-X node")

    automaton = subparsers.add_parser(
        "automaton",
        aliases=["simulate"],
        help="run the sparse brick field over a symbolic tetration address manifold",
    )
    automaton.add_argument("--steps", type=int, default=12)
    automaton.add_argument("--tower-height", type=int, default=2)
    automaton.add_argument("--chart", default="origin")
    automaton.add_argument("--amplitude", type=float, default=48.0)
    automaton.add_argument("--seed", type=int, default=1337)
    automaton.add_argument("--latent-dim", type=int, default=16)
    automaton.add_argument("--experts", type=int, default=4)
    automaton.add_argument("--max-active", type=int, default=128)
    automaton.add_argument("--buckets", type=int, default=257)
    automaton.add_argument("--json", action="store_true", dest="as_json")

    universe = subparsers.add_parser(
        "universe", help="print the symbolic tetration-universe descriptor"
    )
    universe.add_argument("--tower-height", type=int, default=2)
    return parser


def _run_source(path: Path) -> int:
    from .assembler import Assembler
    from .core import CodexVM
    from .parser import Parser

    source = path.read_text(encoding="utf-8")
    bytecode = Assembler().assemble(Parser().parse(source))
    vm = CodexVM()
    vm.load(bytecode)
    vm.run()
    print("Registers:", vm.regs.snapshot())
    return 0


def _run_automaton(args: argparse.Namespace) -> int:
    from .tetration_field import (
        FieldMechanics,
        TetrationAddress,
        TetrationFieldAutomaton,
        TetrationUniverse,
        make_brick_pulse,
    )

    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    if args.tower_height < 1:
        raise SystemExit("--tower-height must be positive")
    if args.max_active < 7:
        raise SystemExit("--max-active must be at least 7")
    if args.buckets < 1:
        raise SystemExit("--buckets must be positive")

    universe = TetrationUniverse(height=args.tower_height)
    mechanics = FieldMechanics(max_active_bricks=args.max_active)
    engine = TetrationFieldAutomaton(
        universe=universe,
        mechanics=mechanics,
        latent_dim=args.latent_dim,
        expert_count=args.experts,
        seed=args.seed,
        bucket_count=args.buckets,
    )
    origin = TetrationAddress(args.tower_height, args.chart, 0, 0, 0)
    pulse = {origin: make_brick_pulse(args.amplitude)}

    metrics = []
    for step_index in range(args.steps):
        result = engine.step(pulse if step_index == 0 else None)
        metrics.append(result.to_dict())
        if not result.committed:
            break

    payload = engine.snapshot()
    payload["metrics"] = metrics
    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    descriptor = payload["universe"]
    print("Jarvis-X sparse tetration field automaton")
    print("Virtual axis:", descriptor["axis_size"])
    print("Virtual cells:", descriptor["virtual_cells"])
    print("Coordinate naming:", descriptor["coordinate_bits"], "bits")
    print("Brick:", payload["brick_shape"], "router:", payload["router"])
    for item in metrics:
        status = "COMMIT" if item["committed"] else "ROLLBACK"
        print(
            "cycle={cycle:>3} status={status:<8} active={active_bricks:>4} "
            "materialised={materialised_bricks:>4} frontier={frontier_bricks:>4} "
            "mse={reconstruction_mse:.6f} collisions={collisions}".format(
                status=status, **item
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

        CodexNode().start()
        return 0
    if args.command in ("automaton", "simulate"):
        return _run_automaton(args)
    if args.command == "universe":
        from .tetration_field import TetrationUniverse

        print(
            json.dumps(
                TetrationUniverse(height=args.tower_height).descriptor(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    parser.error("unknown command: {}".format(args.command))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
