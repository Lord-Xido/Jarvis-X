#!/usr/bin/env python3
"""Deterministic microbenchmark for the Dr Moagi worldwide 3D world fabric.

This benchmark measures only the in-process reference implementation.  It does
not compare against external systems unless the operator performs a separate,
workload-matched benchmark and records that provenance independently.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from jarvisx.dr_moagi_world_fabric import ProvenanceRecord, RegionSpec, WorldFabric


def _payload(index: int, payload_bytes: int) -> bytes:
    prefix = f"dm-world:{index:08d}:".encode("ascii")
    if len(prefix) >= payload_bytes:
        return prefix[:payload_bytes]
    repeated = (f"object-{index % 97:02d}|".encode("ascii") * (payload_bytes + 32))
    return (prefix + repeated)[:payload_bytes]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def run(objects: int, payload_bytes: int, reads: int) -> dict[str, object]:
    regions = (
        RegionSpec("africa", 0.0, 0.0, 0.0, capacity_cells=max(10, objects)),
        RegionSpec("europe", 1.0, 0.0, 0.0, capacity_cells=max(10, objects)),
        RegionSpec("americas", 0.0, 1.0, 0.0, capacity_cells=max(10, objects)),
        RegionSpec("asia", 1.0, 1.0, 0.0, capacity_cells=max(10, objects)),
    )
    provenance = ProvenanceRecord.now("world-fabric-benchmark")

    with WorldFabric(regions=regions, workers=4, max_in_flight=64) as fabric:
        start = time.perf_counter()
        cells = [
            fabric.ingest(_payload(index, payload_bytes), provenance=provenance)
            for index in range(objects)
        ]
        ingest_seconds = time.perf_counter() - start

        for left, right in zip(cells, cells[1:]):
            fabric.record_interaction(left.address, right.address, weight=1.0)
        fold_start = time.perf_counter()
        moves = fabric.fold_placement(max_moves=max(1, objects // 16))
        fold_seconds = time.perf_counter() - fold_start

        read_latencies: list[float] = []
        read_count = min(reads, len(cells))
        for index in range(read_count):
            begin = time.perf_counter_ns()
            data = fabric.retrieve(cells[index].address)
            elapsed_us = (time.perf_counter_ns() - begin) / 1000.0
            if data != _payload(index, payload_bytes):
                raise RuntimeError("retrieval mismatch")
            read_latencies.append(elapsed_us)

        query_start = time.perf_counter()
        nearest = fabric.nearest(_payload(0, payload_bytes), k=min(10, objects))
        query_seconds = time.perf_counter() - query_start

        verified = fabric.verify()
        stats = fabric.stats()

    return {
        "benchmark": "dr-moagi-world-fabric-in-process-v1",
        "claim_status": "reference-measurement-only",
        "objects": objects,
        "payload_bytes": payload_bytes,
        "logical_input_bytes": objects * payload_bytes,
        "ingest_seconds": ingest_seconds,
        "ingest_objects_per_second": objects / ingest_seconds if ingest_seconds else 0.0,
        "fold_seconds": fold_seconds,
        "placement_moves": len(moves),
        "read_count": read_count,
        "read_latency_us_mean": statistics.fmean(read_latencies) if read_latencies else 0.0,
        "read_latency_us_p50": _percentile(read_latencies, 0.50),
        "read_latency_us_p95": _percentile(read_latencies, 0.95),
        "nearest_query_seconds": query_seconds,
        "nearest_results": len(nearest),
        "integrity_verified": verified,
        "stats": stats,
        "boundary": (
            "Numbers are local Python reference results only; no external SOTA comparison, "
            "network durability, geographic consensus, or production scalability is implied."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=int, default=2_000)
    parser.add_argument("--payload-bytes", type=int, default=512)
    parser.add_argument("--reads", type=int, default=500)
    args = parser.parse_args()
    if args.objects <= 0 or args.payload_bytes <= 0 or args.reads < 0:
        parser.error("objects and payload-bytes must be positive; reads must be non-negative")
    print(json.dumps(run(args.objects, args.payload_bytes, args.reads), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
