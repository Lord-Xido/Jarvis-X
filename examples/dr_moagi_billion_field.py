"""Run a bounded sparse Dr Moagi billion-field demonstration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from jarvisx.dr_moagi_billion_field import BillionFieldConfig, SparseBillionField


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute a deterministic sparse simulation over the virtual 1000^3 Dr Moagi field."
        )
    )
    parser.add_argument("--cycles", type=int, default=4, help="number of synchronous cycles")
    parser.add_argument(
        "--reasoning-steps",
        type=int,
        default=5,
        help="bounded internal refinement steps per cycle",
    )
    parser.add_argument(
        "--halo-depth",
        type=int,
        default=1,
        help="six-neighbour support rings added per cycle",
    )
    parser.add_argument(
        "--max-active-cells",
        type=int,
        default=10_000,
        help="hard sparse-support budget",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional JSON checkpoint output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BillionFieldConfig(
        reasoning_steps=args.reasoning_steps,
        halo_depth=args.halo_depth,
        max_active_cells=args.max_active_cells,
        prune_epsilon=1.0e-12,
    )
    field = SparseBillionField(config)
    metrics = field.run(
        cycles=args.cycles,
        observations={
            (500, 500, 500): 1.0,
            (501, 500, 500): 0.5,
            (500, 501, 500): -0.25,
        },
        controls={(500, 500, 500): -0.02},
    )

    output = {
        "config": asdict(config),
        "metrics": asdict(metrics),
        "active_coordinates_preview": [
            list(coordinate) for coordinate in field.active_coordinates()[:16]
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(
            json.dumps(field.checkpoint(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
