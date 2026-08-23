"""CLI for bounded inward 3D meta-optimization of the Dr Moagi runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dr_moagi_meta_optimizer import MetaSearchConfig, SelfOptimizing3DSystem
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jarvisx.dr_moagi_meta_cli",
        description="Benchmark and promote bounded 3D Dr Moagi runtime configurations.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Optimize the deterministic sparse demo field")
    demo.add_argument("--side", type=int, default=16)
    demo.add_argument("--max-candidates", type=int, default=13)
    demo.add_argument("--probe-cycles", type=int, default=1)
    demo.add_argument("--confirm-cycles", type=int, default=3)
    demo.add_argument("--max-eval-cells", type=int, default=2048)
    demo.add_argument("--pretty", action="store_true")

    file_cmd = sub.add_parser("file", help="Optimize a sparse JSON field")
    file_cmd.add_argument("path", type=Path)
    file_cmd.add_argument("--side", type=int, default=64)
    file_cmd.add_argument("--max-candidates", type=int, default=13)
    file_cmd.add_argument("--probe-cycles", type=int, default=1)
    file_cmd.add_argument("--confirm-cycles", type=int, default=3)
    file_cmd.add_argument("--max-eval-cells", type=int, default=2048)
    file_cmd.add_argument("--pretty", action="store_true")
    return parser


def _load_field(path: Path) -> dict[tuple[int, int, int], float]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("field") if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not rows:
        raise ValueError("JSON must contain a non-empty list or {'field': [...]} object")
    field: dict[tuple[int, int, int], float] = {}
    for row in rows:
        if isinstance(row, dict):
            coordinate = int(row["x"]), int(row["y"]), int(row["z"])
            value = float(row["value"])
        elif isinstance(row, list) and len(row) == 4:
            coordinate = int(row[0]), int(row[1]), int(row[2])
            value = float(row[3])
        else:
            raise ValueError("each field row must be {x,y,z,value} or [x,y,z,value]")
        field[coordinate] = value
    return field


def _run(args: argparse.Namespace) -> dict[str, object]:
    config = DrMoagiOSConfig(side=args.side, max_active_cells=50_000, state_dir=None)
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    source = demo_field(args.side) if args.command == "demo" else _load_field(args.path)
    kernel.load(source)
    search = MetaSearchConfig(
        max_candidates=args.max_candidates,
        probe_cycles=args.probe_cycles,
        confirm_cycles=args.confirm_cycles,
        max_eval_cells=args.max_eval_cells,
        survivors=min(4, args.max_candidates),
    )
    system = SelfOptimizing3DSystem(kernel, search=search)
    report = system.turn_inward()
    return {
        "meta_report": report.as_dict(),
        "runtime_status": system.status(),
    }


def main() -> int:
    args = _parser().parse_args()
    payload = _run(args)
    print(json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
