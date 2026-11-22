#!/usr/bin/env python3
"""Validate deterministic output structure from the bounded JARVIS-X PDF graphics VM."""
import csv
import pathlib
import sys


def audit(ppm: pathlib.Path, trace: pathlib.Path) -> None:
    data = ppm.read_bytes()
    assert data.startswith(b"P6\n128 128\n255\n"), "incorrect PPM header"
    payload = data[len(b"P6\n128 128\n255\n") :]
    assert len(payload) == 128 * 128 * 3, "incorrect PPM payload"
    assert any(channel < 255 for channel in payload), "empty frame"
    # Verified pixel in red rectangle: PDF y=70 -> PPM y=57.
    pixel = ((127 - 70) * 128 + 80) * 3
    assert payload[pixel : pixel + 3] == b"\xff\x00\x00", "red rectangle pixel failed"
    with trace.open(newline="") as f:
        rows = list(csv.DictReader(f))
    instructions = [r for r in rows if r["kind"] == "instruction"]
    primitives = [r for r in rows if r["kind"] == "primitive"]
    assert [r["op"] for r in instructions] == ["q", "cm", "BT", "Tf", "Tm", "Tj", "ET", "rg", "re", "f", "Q"]
    assert len(primitives) > 50
    for r in primitives:
        assert all(0 <= int(r[col]) < 1_000_000 for col in ("tile_x", "tile_y", "tile_z"))
    print(f"AUDIT_PASS pixels={len(payload)//3} operators={len(instructions)} polygons={len(primitives)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: audit.py IMAGE.ppm TRACE.csv")
    audit(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
