from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .api import start_api
from .assembler import Assembler
from .core import CodexVM
from .dr_moagi_codec_3d import CodecConfig, DrMoagiCodec3D, Volume3D
from .node import CodexNode
from .parser import Parser
from .web import start_web


def _load_volume(path: Path) -> Volume3D:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("volume JSON must be an object")
    shape = payload.get("shape")
    values = payload.get("values")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("volume JSON shape must contain three dimensions")
    if not isinstance(values, list):
        raise ValueError("volume JSON values must be a list")
    return Volume3D(
        tuple(int(value) for value in shape),
        tuple(float(value) for value in values),
    )


def _write_volume(path: Path, volume: Volume3D) -> None:
    path.write_text(
        json.dumps(
            {"shape": list(volume.shape), "values": list(volume.values)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_source(path: Path, max_cycles: int) -> int:
    source = path.read_text(encoding="utf-8")
    ast = Parser().parse(source)
    bytecode = Assembler().assemble(ast)
    vm = CodexVM(max_cycles=max_cycles)
    vm.load(bytecode)
    registers = vm.run()
    print(
        json.dumps(
            {
                "registers": registers,
                "cycles": vm.cycles,
                "ledger_entries": len(vm.ledger.chain),
                "ledger_valid": vm.ledger.verify(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _codec_roundtrip(args: argparse.Namespace) -> int:
    source = _load_volume(args.input)
    anchor = _load_volume(args.anchor) if args.anchor is not None else None
    config = CodecConfig(
        quant_step=args.quant_step,
        max_voxels=args.max_voxels,
        max_anchor_mse=args.max_anchor_mse,
        max_rate_bpv=args.max_rate_bpv,
        virtual_depth=args.virtual_depth,
    )
    runtime = DrMoagiCodec3D(config)
    result = runtime.process(source, anchor=anchor)

    if args.bitstream is not None:
        args.bitstream.write_bytes(result.bitstream)
    if args.reconstructed is not None:
        _write_volume(args.reconstructed, result.reconstructed)

    receipt = {
        "committed": result.committed,
        "rejection_reason": result.rejection_reason,
        "metadata": asdict(result.metadata),
        "metrics": asdict(result.metrics),
        "memory": asdict(result.memory_after),
        "virtual_depth": result.virtual_depth,
        "measured_microsteps_executed": result.measured_microsteps_executed,
        "wall_clock_seconds": result.wall_clock_seconds,
        "measured_throughput_voxels_per_second": (
            result.measured_throughput_voxels_per_second
        ),
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.committed else 2


def _codec_decode(args: argparse.Namespace) -> int:
    runtime = DrMoagiCodec3D(CodecConfig(max_voxels=args.max_voxels))
    volume = runtime.decode(args.input.read_bytes())
    _write_volume(args.output, volume)
    print(
        json.dumps(
            {"shape": list(volume.shape), "voxels": volume.voxel_count},
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvisx", description="Jarvis-X operational CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="execute Jarvis-X assembly")
    run_parser.add_argument("file", type=Path)
    run_parser.add_argument("--max-cycles", type=int, default=10_000)

    api_parser = subparsers.add_parser("api", help="serve the FastAPI runtime")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8080)

    web_parser = subparsers.add_parser("web", help="serve the unified dashboard/API")
    web_parser.add_argument("--host", default="0.0.0.0")
    web_parser.add_argument("--port", type=int, default=8080)

    node_parser = subparsers.add_parser("node", help="start the legacy CodexNode reference")
    node_parser.add_argument("--host", default="127.0.0.1")
    node_parser.add_argument("--port", type=int, default=9000)

    codec_parser = subparsers.add_parser("codec", help="run one bounded 3D codec transaction")
    codec_parser.add_argument("input", type=Path, help="JSON file with shape and flat values")
    codec_parser.add_argument("--anchor", type=Path)
    codec_parser.add_argument("--bitstream", type=Path)
    codec_parser.add_argument("--reconstructed", type=Path)
    codec_parser.add_argument("--quant-step", type=float, default=0.25)
    codec_parser.add_argument("--max-voxels", type=int, default=1_000_000)
    codec_parser.add_argument("--max-anchor-mse", type=float)
    codec_parser.add_argument("--max-rate-bpv", type=float)
    codec_parser.add_argument("--virtual-depth", type=int, default=1)

    decode_parser = subparsers.add_parser("codec-decode", help="decode a JX3D bitstream")
    decode_parser.add_argument("input", type=Path)
    decode_parser.add_argument("output", type=Path)
    decode_parser.add_argument("--max-voxels", type=int, default=1_000_000)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "run":
            return _run_source(args.file, args.max_cycles)
        if args.command == "api":
            start_api(host=args.host, port=args.port)
            return 0
        if args.command == "web":
            start_web(host=args.host, port=args.port)
            return 0
        if args.command == "node":
            CodexNode(host=args.host, port=args.port).start()
            return 0
        if args.command == "codec":
            return _codec_roundtrip(args)
        if args.command == "codec-decode":
            return _codec_decode(args)
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"jarvisx: {error}", file=sys.stderr)
        return 1

    parser = _build_parser()
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
