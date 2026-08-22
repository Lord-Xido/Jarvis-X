"""Command-line interface for the Dr Moagi Deep Distiller runtime."""

from __future__ import annotations

import argparse
import json

from .dr_moagi_deep_distiller import DeepDistiller, DeepDistillerConfig


def _demo_field(side: int) -> dict[tuple[int, int, int], float]:
    points = {
        (0, 0, 0): 1.0,
        (1 % side, 0, 0): 0.75,
        (0, 1 % side, 0): -0.50,
        (0, 0, 1 % side): 0.25,
    }
    return {coordinate: value for coordinate, value in points.items() if all(axis < side for axis in coordinate)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the IP-locked Dr Moagi Deep Distiller")
    parser.add_argument("--side", type=int, default=16, help="logical lattice side")
    parser.add_argument("--steps", type=int, default=8, help="maximum distillation ticks")
    parser.add_argument("--tolerance", type=float, default=1.0e-6, help="residual stop tolerance")
    parser.add_argument("--max-active", type=int, default=4096, help="maximum committed active cells")
    parser.add_argument("--max-latent", type=int, default=2048, help="maximum latent active cells")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = DeepDistillerConfig(
        logical_side=args.side,
        max_active_cells=args.max_active,
        max_latent_cells=min(args.max_latent, args.max_active),
        residual_tolerance=args.tolerance,
        max_iterations=args.steps,
    )
    engine = DeepDistiller(config)
    engine.load(_demo_field(args.side))
    reports = engine.run(args.steps)
    output = {
        "status": engine.status(),
        "reports": [report.as_dict() for report in reports],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if all(report.committed for report in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
