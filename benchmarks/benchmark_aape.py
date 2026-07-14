"""Compare scalar and packed majority-of-seven projection.

Run after installing the package:
    python benchmarks/benchmark_aape.py --side 16 --repeats 7
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from jarvisx.aape import AAPEConfig, BitLattice, JXAAPEEngine


def scalar_step(bits: int, side: int) -> int:
    plane = side * side
    result = 0
    for z in range(side):
        for y in range(side):
            for x in range(side):
                coordinates = (
                    (x, y, z),
                    ((x + 1) % side, y, z),
                    ((x - 1) % side, y, z),
                    (x, (y + 1) % side, z),
                    (x, (y - 1) % side, z),
                    (x, y, (z + 1) % side),
                    (x, y, (z - 1) % side),
                )
                count = 0
                for nx, ny, nz in coordinates:
                    index = nx + side * ny + plane * nz
                    count += (bits >> index) & 1
                if count >= 4:
                    result |= 1 << (x + side * y + plane * z)
    return result


def timed(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples) / 1_000_000_000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.side ** 3 % 64:
        raise SystemExit("side^3 must be divisible by 64")
    if args.repeats < 1:
        raise SystemExit("repeats must be positive")

    rng = random.Random(args.seed)
    voxel_count = args.side ** 3
    bits = rng.getrandbits(voxel_count)
    engine = JXAAPEEngine(AAPEConfig(side=args.side, max_ca_steps=1))
    lattice = BitLattice(args.side, bits)

    expected = scalar_step(bits, args.side)
    actual = engine.project_once(lattice).bits
    if actual != expected:
        raise SystemExit("packed and scalar projections disagree")

    scalar_seconds = timed(lambda: scalar_step(bits, args.side), args.repeats)
    packed_seconds = timed(lambda: engine.project_once(lattice), args.repeats)
    speedup = scalar_seconds / packed_seconds

    print(f"side={args.side} voxels={voxel_count:,}")
    print(f"scalar median={scalar_seconds:.9f}s")
    print(f"packed median={packed_seconds:.9f}s")
    print(f"measured speedup={speedup:.3f}x")
    print(f"8x target met={'yes' if speedup >= 8.0 else 'no'}")


if __name__ == "__main__":
    main()
