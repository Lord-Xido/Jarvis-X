"""CLI for the bounded Dr Moagi 3D auto-execution engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .dr_moagi_autoexec import (
    AutoExecPolicy,
    DrMoagiAutoExecConfig,
    DrMoagiAutoExecutionEngine,
)
from .dr_moagi_field_runtime import DrMoagiFieldConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-dr-moagi",
        description="Run the sparse 3D parse/encode/decode/execute/verify loop.",
    )
    parser.add_argument("--input", type=Path, help="JSON field file")
    parser.add_argument("--side", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--max-active-cells", type=int, default=50_000)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--quantization", type=float, default=0.01)
    parser.add_argument("--prune-epsilon", type=float, default=0.0)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--no-auto-optimize", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def _demo_field(side: int) -> dict[tuple[int, int, int], float]:
    center = side // 2
    points: dict[tuple[int, int, int], float] = {(center, center, center): 1.0}
    for delta, value in ((1, 0.75), (2, 0.50)):
        for dx, dy, dz in (
            (delta, 0, 0),
            (-delta, 0, 0),
            (0, delta, 0),
            (0, -delta, 0),
            (0, 0, delta),
            (0, 0, -delta),
        ):
            coordinate = center + dx, center + dy, center + dz
            if all(0 <= axis < side for axis in coordinate):
                points[coordinate] = value
    return points


def _load_json(path: Path) -> list[tuple[tuple[int, int, int], float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("field") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("input JSON must be a list or an object containing a 'field' list")

    field = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"field[{index}] must be an object")
        try:
            coordinate = int(item["x"]), int(item["y"]), int(item["z"])
            value = float(item["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"field[{index}] must contain integer x/y/z and numeric value"
            ) from exc
        field.append((coordinate, value))
    return field


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.side <= 0:
        raise SystemExit("--side must be positive")
    if args.cycles <= 0:
        raise SystemExit("--cycles must be positive")
    if args.max_active_cells <= 0:
        raise SystemExit("--max-active-cells must be positive")

    field_config = DrMoagiFieldConfig(
        side=args.side,
        alpha=1.0,
        lambda_residual=0.25,
        eta=0.05,
        dt=0.02,
        max_active_cells=args.max_active_cells,
        expand_halo=True,
        prune_epsilon=args.prune_epsilon,
    )
    config = DrMoagiAutoExecConfig(
        field_config=field_config,
        policy=AutoExecPolicy(
            block_size=args.block_size,
            quantization=args.quantization,
            prune_epsilon=args.prune_epsilon,
        ),
        cycles=args.cycles,
        auto_optimize=not args.no_auto_optimize,
    )
    engine = DrMoagiAutoExecutionEngine(config, journal_path=args.journal)
    source = _load_json(args.input) if args.input else _demo_field(args.side)
    engine.load(source)
    reports = engine.run()

    result = {
        "engine": "Dr Moagi 3D AutoExec",
        "reports": [report.as_dict() for report in reports],
        "status": engine.status(),
    }
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
