"""CLI for the Dr Moagi frontier research runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dr_moagi_autoexec import AutoExecPolicy
from .dr_moagi_frontier import DrMoagiFrontierRuntime, FrontierConfig
from .dr_moagi_os import demo_field


def _runtime(args: argparse.Namespace) -> DrMoagiFrontierRuntime:
    policy = AutoExecPolicy(
        block_size=args.block_size,
        quantization=args.quantization,
        prune_epsilon=args.prune_epsilon,
    )
    config = FrontierConfig(
        side=args.side,
        max_active_cells=args.max_active_cells,
        policy=policy,
        contraction=args.contraction,
        attenuation=args.attenuation,
        equilibrium_tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        anderson_depth=args.anderson_depth,
        anderson_damping=args.anderson_damping,
        fixed_point_gain=args.fixed_point_gain,
        prune_epsilon=args.prune_epsilon,
    )
    return DrMoagiFrontierRuntime(config, journal_path=args.journal)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--max-active-cells", type=int, default=20_000)
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--quantization", type=float, default=0.01)
    parser.add_argument("--prune-epsilon", type=float, default=0.0)
    parser.add_argument("--contraction", type=float, default=0.08)
    parser.add_argument("--attenuation", type=float, default=0.10)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--max-iterations", type=int, default=32)
    parser.add_argument("--anderson-depth", type=int, default=4)
    parser.add_argument("--anderson-damping", type=float, default=1.0)
    parser.add_argument("--fixed-point-gain", type=float, default=0.5)
    parser.add_argument("--journal", type=Path, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-dr-moagi-frontier",
        description=(
            "Run the empirical frontier candidate: hierarchical sparse geometry, "
            "entropy packets and Anderson-accelerated 3D fixed points."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a deterministic sparse 3D demo")
    _add_common(demo)
    demo.add_argument("--cycles", type=int, default=3)

    status = subparsers.add_parser("status", help="load demo state and print frontier status")
    _add_common(status)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    runtime = _runtime(args)
    runtime.load(demo_field(args.side))

    if args.command == "demo":
        reports = runtime.run(args.cycles)
        payload = {
            "reports": [report.as_dict() for report in reports],
            "status": runtime.status(),
        }
    else:
        payload = runtime.status()

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
