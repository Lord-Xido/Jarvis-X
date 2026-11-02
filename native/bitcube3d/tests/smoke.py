#!/usr/bin/env python3
"""CLI persistence and malformed-input smoke tests (stdlib only)."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

BIN = Path(sys.argv[1]).resolve()
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "3d_program.b3ds"

with tempfile.TemporaryDirectory(prefix="bitcube3d-") as dirname:
    work = Path(dirname)
    run = subprocess.run([str(BIN), "run", str(EXAMPLE)], cwd=work,
                         capture_output=True, text=True, check=True)
    assert "VERIFY PASS" in run.stdout, run.stdout
    plain = (work / "cube_plain.b3d").read_bytes()
    encoded = (work / "cube_encoded.b3d").read_bytes()
    restored = (work / "cube_roundtrip.b3d").read_bytes()
    assert plain == restored, "B3D1 round-trip is not byte-identical"
    assert plain[:4] == encoded[:4] == b"B3D1"
    side, rounds, bricks = struct.unpack_from("<IIQ", encoded, 4)
    assert (side, rounds) == (1_000_000_000, 100)
    assert bricks > 0
    assert len(encoded) == 20 + bricks * 524

    def must_fail(source: str):
        path = work / "bad.b3ds"
        path.write_text(source, encoding="utf-8")
        outcome = subprocess.run([str(BIN), "run", str(path)], cwd=work,
                                 capture_output=True, text=True)
        assert outcome.returncode != 0, f"accepted invalid script {source!r}"

    must_fail("SET 1000000000 0 0 1\n")
    must_fail("ENCODE 10001\n")
    must_fail("SET 0 0 0 1\nENCODE 1\nFLIP 0 0 0\n")
    (work / "corrupt.b3d").write_bytes(encoded[:-1])
    must_fail("LOAD corrupt.b3d\n")
    generated = work / "generated.cpp"
    subprocess.run([str(BIN), "codegen", str(EXAMPLE), str(generated)],
                   check=True, capture_output=True, text=True)
    assert "vm.run_line" in generated.read_text(encoding="utf-8")

print("BitCube3D CLI/B3D1 smoke: PASS")
