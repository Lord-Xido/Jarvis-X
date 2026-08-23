"""Command-line surface for the Dr Moagi worldwide 3D world fabric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dr_moagi_world_fabric import ConsistencyClass, ProvenanceRecord, WorldFabric


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jarvisx-dr-moagi-world",
        description="Operate the bounded reference Dr Moagi sparse worldwide 3D fabric.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="ingest one local file into a reference fabric")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--source", default="local-file")
    ingest.add_argument("--source-uri")
    ingest.add_argument("--media-type")
    ingest.add_argument(
        "--consistency",
        choices=[item.value for item in ConsistencyClass],
        default=ConsistencyClass.IMMUTABLE.value,
    )

    demo = subparsers.add_parser("demo", help="run a deterministic in-process world-fabric demo")
    demo.add_argument("--objects", type=int, default=16)
    return parser


def _ingest(args: argparse.Namespace) -> int:
    payload = args.path.read_bytes()
    provenance = ProvenanceRecord.now(
        args.source,
        source_uri=args.source_uri or args.path.resolve().as_uri(),
        media_type=args.media_type,
    )
    with WorldFabric() as fabric:
        cell = fabric.ingest(
            payload,
            provenance=provenance,
            consistency=ConsistencyClass(args.consistency),
        )
        result = {
            "address": cell.address.to_text(),
            "cid": cell.cid,
            "region": cell.region_id,
            "bytes": cell.byte_length,
            "latent_dimensions": len(cell.latent),
            "integrity_verified": fabric.verify(),
            "stats": fabric.stats(),
            "capability_boundary": "in-process reference; no geographic persistence or consensus",
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _demo(args: argparse.Namespace) -> int:
    if args.objects <= 0:
        raise SystemExit("--objects must be positive")
    provenance = ProvenanceRecord.now("deterministic-demo")
    with WorldFabric() as fabric:
        cells = [
            fabric.ingest(
                f"world-fabric-object-{index:06d}".encode("utf-8"),
                provenance=provenance,
            )
            for index in range(args.objects)
        ]
        for left, right in zip(cells, cells[1:]):
            fabric.record_interaction(left.address, right.address, weight=2.0)
        moves = fabric.fold_placement(max_moves=max(1, args.objects // 4))
        query = fabric.nearest(b"world-fabric-object-000000", k=min(3, args.objects))
        result = {
            "objects": args.objects,
            "placement_moves": moves,
            "nearest": [
                {
                    "address": cell.address.to_text(),
                    "cid": cell.cid,
                    "score": score,
                }
                for cell, score in query
            ],
            "integrity_verified": fabric.verify(),
            "stats": fabric.stats(),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "demo":
        return _demo(args)
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
