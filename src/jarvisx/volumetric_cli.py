from __future__ import annotations

import argparse
import json
from pathlib import Path

from .volumetric_ae import Universal3DAutoEncoder, VirtualVolumeSpec


def _engine(args: argparse.Namespace) -> Universal3DAutoEncoder:
    return Universal3DAutoEncoder(
        VirtualVolumeSpec(
            capacity_gib=args.capacity_gib,
            cell_bits=args.cell_bits,
            chunk_bytes=args.chunk_bytes,
        ),
        compression_level=args.compression_level,
    )


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jarvis-X 3D volumetric AE/AD runtime")
    parser.add_argument("--capacity-gib", type=int, default=6400)
    parser.add_argument("--cell-bits", type=int, default=32, help="32 = Q16.16 storage cell")
    parser.add_argument("--chunk-bytes", type=int, default=1 << 20)
    parser.add_argument("--compression-level", type=int, default=9)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("metrics", help="show logical substrate metrics")
    sub.add_parser("selftest", help="run encode/decode verification")

    enc = sub.add_parser("encode", help="encode a file into a .jx3d latent artifact")
    enc.add_argument("input", type=Path)
    enc.add_argument("output", type=Path)

    dec = sub.add_parser("decode", help="decode and verify a .jx3d latent artifact")
    dec.add_argument("input", type=Path)
    dec.add_argument("output", type=Path)

    args = parser.parse_args(argv)
    engine = _engine(args)

    if args.command == "metrics":
        _print_json(engine.spec.metrics())
        return 0
    if args.command == "selftest":
        report = engine.self_test()
        _print_json(report)
        return 0 if report["ok"] else 1
    if args.command == "encode":
        artifact, receipt = engine.encode(args.input.read_bytes())
        args.output.write_bytes(artifact)
        _print_json(receipt.to_dict())
        return 0
    if args.command == "decode":
        payload, receipt = engine.decode(args.input.read_bytes())
        args.output.write_bytes(payload)
        _print_json(receipt.to_dict())
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
