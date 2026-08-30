"""Command-line interface for the bounded Dr Moagi PDF bytecode runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dr_moagi_pdf_bytecode import (
    ProgramLimits,
    build_pdf_package,
    canonical_autoencoder_program,
    load_pdf_package,
    make_seed_volume,
    run_pdf_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dr Moagi PDF-carried DM3D bytecode runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a verified PDF bytecode package")
    build.add_argument("output", type=Path)
    build.add_argument("--pool", type=int, default=2)
    build.add_argument("--passes", type=int, default=4)

    inspect = subparsers.add_parser("inspect", help="verify and inspect a PDF bytecode package")
    inspect.add_argument("pdf", type=Path)

    run = subparsers.add_parser("run", help="explicitly execute a verified PDF bytecode package")
    run.add_argument("pdf", type=Path)
    run.add_argument("--size", type=int, default=16)
    run.add_argument("--max-physical-steps", type=int, default=50_000_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        program = canonical_autoencoder_program(pool=args.pool, refinement_passes=args.passes)
        manifest = build_pdf_package(args.output, program)
        print(json.dumps(manifest.__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "inspect":
        manifest, payload = load_pdf_package(args.pdf)
        report = {**manifest.__dict__, "verified_payload_bytes": len(payload)}
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        limits = ProgramLimits(max_physical_steps=args.max_physical_steps)
        result = run_pdf_package(args.pdf, make_seed_volume(args.size), limits)
        print(json.dumps(result.report(), indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
