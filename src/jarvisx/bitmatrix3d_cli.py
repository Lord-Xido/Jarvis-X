"""CLI for the deterministic Bit Matrix 3D codec-runtime."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from .bitmatrix3d import Dimensions3D, decode_text, encode_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-bitmatrix3d",
        description="Encode UTF-8 text into a deterministic 3D bit lattice and verify round-trip decoding.",
    )
    parser.add_argument("text", help="UTF-8 text to encode")
    parser.add_argument("--x", type=int, default=16, help="X lattice dimension")
    parser.add_argument("--y", type=int, default=16, help="Y lattice dimension")
    parser.add_argument("--z", type=int, default=16, help="Z lattice dimension")
    parser.add_argument(
        "--coordinates",
        action="store_true",
        help="Include active voxel coordinates in JSON output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dimensions = Dimensions3D(args.x, args.y, args.z)
    matrix = encode_text(args.text, dimensions)
    decoded = decode_text(matrix)

    result: dict[str, object] = {
        "dimensions": {"x": dimensions.x, "y": dimensions.y, "z": dimensions.z},
        "cells": dimensions.cells,
        "payload_capacity_bytes": dimensions.payload_capacity_bytes,
        "payload_bytes": len(args.text.encode("utf-8")),
        "active_voxels": matrix.active_count,
        "round_trip_ok": decoded == args.text,
        "decoded": decoded,
    }
    if args.coordinates:
        result["active_coordinates"] = [
            {"x": coordinate.x, "y": coordinate.y, "z": coordinate.z}
            for coordinate in matrix.active_coordinates()
        ]

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if decoded == args.text else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
