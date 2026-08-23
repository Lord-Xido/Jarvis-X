"""Command-line entry point for the end-to-end Dr Moagi 3D OS control plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import uvicorn

from .dr_moagi_field_runtime import SparseField
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field
from .dr_moagi_os_store import SparseStateCodec3D


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvisx-dr-moagi-os")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the FastAPI + browser control plane")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=10000)

    demo = sub.add_parser("demo", help="run the built-in sparse 3D field locally")
    demo.add_argument("--side", type=int, default=64)
    demo.add_argument("--cycles", type=int, default=8)
    demo.add_argument("--state-dir", type=Path)
    demo.add_argument("--pretty", action="store_true")

    run_file = sub.add_parser("run-file", help="run a JSON sparse field locally")
    run_file.add_argument("input", type=Path)
    run_file.add_argument("--side", type=int, default=64)
    run_file.add_argument("--cycles", type=int, default=8)
    run_file.add_argument("--state-dir", type=Path)
    run_file.add_argument("--pretty", action="store_true")

    pack = sub.add_parser("pack", help="encode JSON sparse field to exact .dmos packet")
    pack.add_argument("input", type=Path)
    pack.add_argument("output", type=Path)
    pack.add_argument("--side", type=int, default=64)
    pack.add_argument("--pretty", action="store_true")

    run_packet = sub.add_parser("run-packet", help="run an exact .dmos sparse packet")
    run_packet.add_argument("input", type=Path)
    run_packet.add_argument("--cycles", type=int, default=8)
    run_packet.add_argument("--state-dir", type=Path)
    run_packet.add_argument("--pretty", action="store_true")

    inspect_packet = sub.add_parser("inspect-packet", help="validate and inspect .dmos packet")
    inspect_packet.add_argument("input", type=Path)
    inspect_packet.add_argument("--pretty", action="store_true")
    return parser


def _load_field(path: Path) -> SparseField:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("field") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError("input JSON must be a list or object containing a 'field' list")
    field: SparseField = {}
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
        field[coordinate] = value
    return field


def _run_local(
    *,
    side: int,
    cycles: int,
    state_dir: Path | None,
    field: SparseField,
    pretty: bool,
) -> int:
    config = DrMoagiOSConfig(side=side, state_dir=state_dir)
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    kernel.load(field)
    reports = kernel.run(cycles)
    result = {
        "reports": [report.as_dict() for report in reports],
        "status": kernel.status(),
        "capabilities": kernel.capabilities(),
    }
    print(json.dumps(result, indent=2 if pretty else None, sort_keys=True))
    return 0


def _validate_cycles(cycles: int) -> None:
    if cycles <= 0:
        raise SystemExit("--cycles must be positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        if args.port <= 0 or args.port > 65535:
            raise SystemExit("--port must be in [1, 65535]")
        uvicorn.run(
            "jarvisx.dr_moagi_os_api:app",
            host=str(args.host),
            port=int(args.port),
            log_level="info",
        )
        return 0

    if args.command == "pack":
        if args.side <= 0:
            raise SystemExit("--side must be positive")
        codec = SparseStateCodec3D()
        packet = codec.encode(_load_field(args.input), side=args.side)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(packet.payload)
        print(json.dumps(packet.as_dict(), indent=2 if args.pretty else None, sort_keys=True))
        return 0

    if args.command == "inspect-packet":
        codec = SparseStateCodec3D()
        packet, field = codec.decode_payload(args.input.read_bytes())
        result = {**packet.as_dict(), "decoded_active_cells": len(field)}
        print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
        return 0

    if args.command == "run-packet":
        _validate_cycles(args.cycles)
        codec = SparseStateCodec3D()
        packet, field = codec.decode_payload(args.input.read_bytes())
        return _run_local(
            side=packet.side,
            cycles=args.cycles,
            state_dir=args.state_dir,
            field=field,
            pretty=bool(args.pretty),
        )

    if args.side <= 0:
        raise SystemExit("--side must be positive")
    _validate_cycles(args.cycles)

    if args.command == "demo":
        return _run_local(
            side=args.side,
            cycles=args.cycles,
            state_dir=args.state_dir,
            field=demo_field(args.side),
            pretty=bool(args.pretty),
        )
    if args.command == "run-file":
        return _run_local(
            side=args.side,
            cycles=args.cycles,
            state_dir=args.state_dir,
            field=_load_field(args.input),
            pretty=bool(args.pretty),
        )
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
