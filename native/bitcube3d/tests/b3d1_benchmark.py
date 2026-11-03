#!/usr/bin/env python3
"""Observational B3D1 Python/native CLI comparison; never a hardware speedup claim."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from jarvisx.bitcube3d_b3d1 import deserialize, from_points  # noqa: E402


def benchmark(cli: Path, repetitions: int = 3) -> dict[str, object]:
    if repetitions < 1 or repetitions > 20:
        raise ValueError("repetitions must be between 1 and 20")

    # 8 source bricks, repeatable spatial occupancy, never a synthetic 'full 10^27' run.
    points = sorted({(i % 32, (i * 7) % 32, (i * 11) % 32) for i in range(512)})
    rounds = 2
    expected = sorted(points)
    python_times: list[float] = []
    native_times: list[float] = []
    payload: bytes = b""

    with tempfile.TemporaryDirectory(prefix="b3d1-observed-bench-") as name:
        folder = Path(name)
        binary = folder / "output.b3d"
        script = folder / "bench.b3ds"
        commands = ["RESET"] + [f"SET {x} {y} {z} 1" for x, y, z in points]
        commands.extend([f"ENCODE {rounds}", f"SAVE {binary}"])
        script.write_text("\n".join(commands) + "\n", encoding="utf-8")

        for trial in range(repetitions + 1):
            start = time.perf_counter()
            py_state = from_points(points, rounds=rounds)
            py_payload = py_state.serialize()
            py_elapsed = time.perf_counter() - start
            start = time.perf_counter()
            result = subprocess.run(
                [str(cli), "run", str(script)],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            native_elapsed = time.perf_counter() - start
            if result.returncode:
                raise RuntimeError(f"native B3D1 benchmark failed: {result.stdout} {result.stderr}")

            native_payload = binary.read_bytes()
            if py_payload != native_payload:
                raise AssertionError("native and Python output bytes differ")
            parsed = deserialize(native_payload)
            if parsed.active_points(inverse_rounds=True) != expected:
                raise AssertionError("decoded spatial fidelity is not exact")
            payload = native_payload
            if trial:
                python_times.append(py_elapsed)
                native_times.append(native_elapsed)

    py_median = statistics.median(python_times)
    native_median = statistics.median(native_times)
    bits_count = len(expected)
    # Work-unit rate = original active bits divided by elapsed wall time;
    # not logical address extent nor dense bits/s or CPU instruction throughput.
    return {
        "schema": "jarvisx.b3d1.observational-benchmark.v1",
        "project": "Jarvis-X",
        "claim_boundary": (
            "Native time includes subprocess launch, DSL interpretation and file I/O; "
            "Python time includes reference encoding and in-memory serialization. "
            "Neither is an isolated SIMD/GPU kernel benchmark and ratios are not "
            "portable apples-to-apples speedups."
        ),
        "repetitions_after_warmup": repetitions,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "source_active_bits": bits_count,
        "virtual_bit_positions": 10**27,
        "format": "B3D1",
        "rounds": rounds,
        "on_disk_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "exact_bytes_cpp_python": True,
        "exact_inverse_source_bits": True,
        "python_reference_median_wall_seconds": py_median,
        "cpp_cli_median_wall_seconds_including_process_start_and_io": native_median,
        "python_active_source_bits_per_second_observed": bits_count / py_median,
        "cpp_cli_active_source_bits_per_second_observed": bits_count / native_median,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cli", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.cli.resolve(), repetitions=args.repetitions)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
