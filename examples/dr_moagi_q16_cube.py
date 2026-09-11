"""Run the Dr Moagi normalized codec on a bounded part of a virtual 10^18-cell cube."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import product
from time import perf_counter

from jarvisx.dr_moagi_q16_field import (
    Q_SCALE,
    DrMoagiQ16Config,
    DrMoagiQ16FieldRuntime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.side < 4 or not 1 <= args.steps <= 32:
        parser.error("side must be >= 4 and steps must be in [1, 32]")

    runtime = DrMoagiQ16FieldRuntime(DrMoagiQ16Config(side=args.side))
    origin = args.side // 2 - 2
    source = {
        (origin + x, origin + y, origin + z): (1 + (x + y + z) % 4) * (Q_SCALE // 8)
        for x, y, z in product(range(4), repeat=3)
    }
    runtime.load(source)
    kernel = {offset: Q_SCALE for offset in product(range(-1, 3), repeat=3)}
    cycles = []
    started = perf_counter()
    for _ in range(args.steps):
        report = runtime.step_codec(phi_kernel=kernel, theta_weights=[Q_SCALE // 4])
        record = asdict(report)
        record["output"] = [[*coord, raw] for coord, raw in sorted(report.output.items())]
        # Retain the immutable original target when assessing recursive distortion.
        record["source_squared_error_raw"] = sum(
            (report.output.get(coord, 0) - source.get(coord, 0)) ** 2
            for coord in set(source) | set(report.output)
        )
        cycles.append(record)
    print(
        json.dumps(
            {
                "layout": runtime.logical_layout(),
                "initial_cells": len(source),
                "measured_seconds": perf_counter() - started,
                "cycles": cycles,
                "ledger": runtime.ledger.entries,
                "ledger_head": runtime.ledger.head.hex(),
                "ledger_valid": runtime.ledger.verify(),
                "scope": "Sparse fixed-weight codec; measured work is processed_cells per cycle.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
