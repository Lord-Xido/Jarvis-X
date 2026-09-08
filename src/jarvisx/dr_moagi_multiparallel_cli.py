"""CLI for the sparse 1000x1000 multiparallel Dr Moagi 3D runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .dr_moagi_multiparallel import DrMoagiMultiparallel3D, MultiparallelConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-dr-moagi-multiparallel",
        description=(
            "Run the sparse 1000x1000 logical AE/AD lattice with inward depth, "
            "kinetic permeation, global core coupling and transactional verification."
        ),
    )
    parser.add_argument("--input", type=Path, help="JSON loop-field input")
    parser.add_argument("--side", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--max-active-loops", type=int, default=100_000)
    parser.add_argument("--diffusion", type=float, default=0.08)
    parser.add_argument("--global-coupling", type=float, default=0.05)
    parser.add_argument("--quantization", type=float, default=1.0e-6)
    parser.add_argument("--prune-epsilon", type=float, default=1.0e-12)
    parser.add_argument("--max-reconstruction-mse", type=float, default=0.25)
    parser.add_argument("--no-halo", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def _demo_field(side: int) -> dict[tuple[int, int], tuple[float, ...]]:
    center = side // 2
    vectors = {
        (0, 0): (1.0, 0.20, 0.40, 0.30, 0.80, 0.10),
        (1, 0): (0.70, 0.10, 0.35, 0.45, 0.55, 0.20),
        (-1, 0): (0.60, 0.15, 0.20, 0.25, 0.50, 0.15),
        (0, 1): (0.50, 0.30, 0.30, 0.50, 0.40, 0.25),
        (0, -1): (0.55, 0.25, 0.25, 0.40, 0.45, 0.30),
    }
    return {
        (center + di, center + dj): value
        for (di, dj), value in vectors.items()
        if 0 <= center + di < side and 0 <= center + dj < side
    }


def _load_json(path: Path) -> dict[tuple[int, int], tuple[float, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("loops") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("input JSON must be a list or an object containing a 'loops' list")

    field: dict[tuple[int, int], tuple[float, ...]] = {}
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"loops[{index}] must be an object")
        try:
            coordinate = int(item["x"]), int(item["y"])
            raw_values = item["values"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"loops[{index}] must contain integer x/y and values") from exc
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"loops[{index}].values must be a non-empty numeric list")
        try:
            vector = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"loops[{index}].values must be numeric") from exc
        field[coordinate] = vector
    return field


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = MultiparallelConfig(
        side=args.side,
        depth=args.depth,
        max_active_loops=args.max_active_loops,
        diffusion=args.diffusion,
        global_coupling=args.global_coupling,
        quantization=args.quantization,
        prune_epsilon=args.prune_epsilon,
        expand_halo=not args.no_halo,
        max_reconstruction_mse=args.max_reconstruction_mse,
    )
    engine = DrMoagiMultiparallel3D(config)
    source = _load_json(args.input) if args.input else _demo_field(args.side)
    engine.load(source)
    reports = engine.run(args.cycles)

    result = {
        "engine": "Dr Moagi 1000x1000 Multiparallel 3D",
        "equation": (
            "surface -> kinetic permeation -> inward fold -> global core -> "
            "outward fold -> reconstruction/error -> memory -> commit"
        ),
        "reports": [asdict(report) for report in reports],
        "status": engine.status(),
    }
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
