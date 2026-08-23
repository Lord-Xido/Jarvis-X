"""CLI for four-scale Dr Moagi system auto-evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dr_moagi_meta_optimizer import MetaSearchConfig, SelfOptimizing3DSystem
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field
from .dr_moagi_system_evolution import ArchitecturePolicy, SelfEvolving3DArchitecture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-dr-moagi-system",
        description=(
            "Run the nested state/model/configuration/architecture auto-evolution framework."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run bounded autonomic evolution on the demo field")
    _add_common(demo)

    file_cmd = sub.add_parser("file", help="Run bounded autonomic evolution on a sparse field")
    file_cmd.add_argument("path", type=Path)
    _add_common(file_cmd)
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--state-cycles-per-meta", type=int, default=4)
    parser.add_argument("--meta-epochs-per-architecture", type=int, default=2)
    parser.add_argument("--meta-candidates", type=int, default=5)
    parser.add_argument("--architecture-candidates", type=int, default=3)
    parser.add_argument("--max-eval-cells", type=int, default=128)
    parser.add_argument("--pretty", action="store_true")


def _load_field(path: Path) -> dict[tuple[int, int, int], float]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("field") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("JSON must contain a non-empty list or {'field': [...]} object")
    result: dict[tuple[int, int, int], float] = {}
    for row in rows:
        if isinstance(row, dict):
            coordinate = int(row["x"]), int(row["y"]), int(row["z"])
            value = float(row["value"])
        elif isinstance(row, list) and len(row) == 4:
            coordinate = int(row[0]), int(row[1]), int(row[2])
            value = float(row[3])
        else:
            raise ValueError("each field row must be {x,y,z,value} or [x,y,z,value]")
        result[coordinate] = value
    return result


def _run(args: argparse.Namespace) -> dict[str, object]:
    config = DrMoagiOSConfig(
        side=args.side,
        max_active_cells=50_000,
        deep_distiller_max_latent_cells=25_000,
        state_dir=None,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    source = demo_field(args.side) if args.command == "demo" else _load_field(args.path)
    kernel.load(source)

    search = MetaSearchConfig(
        max_candidates=args.meta_candidates,
        probe_cycles=1,
        confirm_cycles=1,
        max_eval_cells=args.max_eval_cells,
        survivors=min(2, args.meta_candidates),
    )
    system = SelfOptimizing3DSystem(kernel, search=search)
    policy = ArchitecturePolicy(
        state_cycles_per_meta=args.state_cycles_per_meta,
        meta_epochs_per_architecture_review=args.meta_epochs_per_architecture,
        max_architecture_candidates=args.architecture_candidates,
        max_architecture_eval_cells=args.max_eval_cells,
        max_eval_state_cycles=2,
        meta_search=search,
    )
    architecture = SelfEvolving3DArchitecture(system, policy=policy)
    report = architecture.run_autonomic(args.cycles)
    return {
        "autonomic_report": report.as_dict(),
        "status": architecture.status(),
        "capabilities": architecture.capabilities(),
    }


def main() -> int:
    args = _parser().parse_args()
    payload = _run(args)
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
