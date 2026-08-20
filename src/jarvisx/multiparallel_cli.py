"""Command-line surface for the bounded 3D multiparallel reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .multiparallel import (
    EvolutionConfig,
    FramedArtifact,
    JarvisX3DEngine,
    RuntimeLimits,
    default_topology,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-multiparallel",
        description="Bounded deterministic Jarvis-X 3D multiparallel reference",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run", help="process one UTF-8 text file")
    run.add_argument("file", type=Path)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--backend", choices=("sequential", "process"), default="sequential")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--compression-level", type=int, default=6)
    run.add_argument("--output", type=Path)

    spatial = subcommands.add_parser("map-code", help="map Python source to read-only 3D points")
    spatial.add_argument("file", type=Path)
    spatial.add_argument("--axis-order", default="xyz")
    spatial.add_argument("--scale", type=float, default=1.0)

    evolve = subcommands.add_parser("evolve", help="run seeded bounded topology search")
    evolve.add_argument("file", type=Path)
    evolve.add_argument("--generations", type=int, default=5)
    evolve.add_argument("--population", type=int, default=10)
    evolve.add_argument("--seed", type=int, default=0)
    evolve.add_argument("--workers", type=int, default=4)
    evolve.add_argument("--batch-size", type=int, default=64)
    return parser


def _read_text(path: Path, limit: int) -> str:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot inspect input file: {exc}") from exc
    if size > limit:
        raise ValueError("input file exceeds the configured byte limit")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read input as UTF-8 text: {exc}") from exc


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    limits = RuntimeLimits()
    try:
        source = _read_text(args.file, limits.max_input_bytes)
        if args.command == "run":
            topology = default_topology(
                parallelism=args.workers,
                batch_size=args.batch_size,
                compression_level=args.compression_level,
            )
            engine = JarvisX3DEngine(code=source, topology=topology, limits=limits)
            run = engine.process_parallel(
                num_workers=args.workers,
                backend=args.backend,
            )
            if not run.success or run.output is None:
                _emit(
                    {
                        "success": False,
                        "run_id": run.run_id,
                        "errors": [
                            {
                                "package_id": receipt.package_id,
                                "type": receipt.error_type,
                                "message": receipt.error_message,
                            }
                            for receipt in run.receipts
                            if not receipt.success
                        ],
                    }
                )
                return 2
            artifact = run.output
            if not isinstance(artifact, FramedArtifact):
                raise RuntimeError("default pipeline did not produce a framed artifact")
            if args.output is not None:
                args.output.write_bytes(artifact.to_bytes())
            _emit(
                {
                    "success": True,
                    "run_id": run.run_id,
                    "topology_digest": run.topology_digest,
                    "artifact_digest": artifact.digest_sha256,
                    "artifact_path": str(args.output) if args.output is not None else None,
                    "packages": run.stats.package_count,
                    "workers": run.stats.worker_count,
                    "backend": run.stats.backend,
                    "codec_runtime_version": run.stats.codec_runtime_version,
                    "input_bytes": run.stats.input_bytes,
                    "output_bytes": run.stats.output_bytes,
                    "compression_ratio": run.stats.compression_ratio,
                    "elapsed_ns_telemetry": run.stats.elapsed_ns,
                    "throughput_packages_per_second_telemetry": (
                        run.stats.throughput_packages_per_second
                    ),
                }
            )
            return 0

        if args.command == "map-code":
            engine = JarvisX3DEngine(code=source, limits=limits)
            geometry = engine.spatial_process(
                axis_order=args.axis_order,
                scale=args.scale,
            )
            _emit(
                {
                    "source_sha256": geometry.source_sha256,
                    "source_line_count": geometry.source_line_count,
                    "axis_order": geometry.axis_order,
                    "scale": geometry.scale,
                    "points": [
                        {
                            "line": point.line_number,
                            "x": point.coordinate.x,
                            "y": point.coordinate.y,
                            "z": point.coordinate.z,
                            "line_sha256": point.line_sha256,
                        }
                        for point in geometry.points
                    ],
                    "source_rewritten": False,
                }
            )
            return 0

        if args.command == "evolve":
            topology = default_topology(
                parallelism=args.workers,
                batch_size=args.batch_size,
            )
            engine = JarvisX3DEngine(code=source, topology=topology, limits=limits)
            result = engine.auto_evolve(
                source,
                EvolutionConfig(
                    generations=args.generations,
                    population_size=args.population,
                    seed=args.seed,
                ),
            )
            _emit(
                {
                    "success": True,
                    "promoted": result.promoted,
                    "initial_topology_digest": result.initial_topology_digest,
                    "selected_topology_digest": result.selected_topology.digest_sha256,
                    "fitness": result.best_score.fitness,
                    "compression_ratio": result.best_score.compression_ratio,
                    "estimated_work_units": result.best_score.estimated_work_units,
                    "observed_elapsed_ns_telemetry": result.best_score.observed_elapsed_ns,
                    "history": [
                        {
                            "generation": report.generation,
                            "population": report.population_size,
                            "best_fitness": report.best_fitness,
                            "best_topology_digest": report.best_topology_digest,
                        }
                        for report in result.history
                    ],
                }
            )
            return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"jarvisx-multiparallel: {exc}", file=sys.stderr)
        return 2
    raise RuntimeError("unreachable command state")


if __name__ == "__main__":
    raise SystemExit(main())
