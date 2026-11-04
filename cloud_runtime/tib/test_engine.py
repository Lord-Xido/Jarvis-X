#!/usr/bin/env python3
"""Deterministic bounded-size JX3D 1 TiB streaming regression suite."""
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SIM = HERE / "jx3d_1tib"
MEDIA = HERE / "jx3d_media"

def run(*args, expect=0):
    p = subprocess.run([str(args[0]), *map(str, args[1:])], capture_output=True, text=True)
    if (p.returncode == 0) != (expect == 0):
        raise AssertionError(f"expected {'success' if expect == 0 else 'failure'}, got {p.returncode}: {p.args}\n{p.stdout}\n{p.stderr}")
    return p.stdout + p.stderr

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(65536), b""):
            h.update(part)
    return h.hexdigest()

def main():
    checks = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        def passed(message):
            checks.append(message)
            print("PASS", message, flush=True)
        synth = tmp / "synth.jxtb"
        result = run(SIM, "--start", 0, "--tiles", 32, "--out", synth, "--corrupt-test")
        assert "verified_reconstruction=PASS" in result and "corruption_test=PASS" in result
        passed("32-tile roundtrip + injected corruption gate")
        run(SIM, "--verify", synth)
        passed("disk-backed stream verification")
        resumed = tmp / "resumed.jxtb"
        run(SIM, "--start", 64, "--tiles", 4, "--out", resumed)
        run(SIM, "--start", 64, "--tiles", 12, "--out", resumed, "--resume")
        reference = tmp / "reference.jxtb"
        run(SIM, "--start", 64, "--tiles", 12, "--out", reference)
        assert sha(resumed) == sha(reference)
        passed("resumed stream identical to uninterrupted output")
        high = tmp / "high.jxtb"
        run(SIM, "--start", 1048574, "--tiles", 2, "--out", high)
        run(SIM, "--verify", high)
        passed("2^40 high address boundary")
        run(SIM, "--start", 1048575, "--tiles", 2, expect=1)
        passed("out-of-range tile rejected")
        damaged = bytearray(synth.read_bytes())
        damaged[len(damaged)//2] ^= 1
        bad = tmp / "bad.jxtb"; bad.write_bytes(damaged)
        run(SIM, "--verify", bad, expect=1)
        passed("stored stream corruption rejected")
        truncated = tmp / "truncated.jxtb"; truncated.write_bytes(synth.read_bytes()[:-1])
        run(SIM, "--verify", truncated, expect=1)
        passed("truncated record rejected")
        rng = random.Random(1701)
        original = tmp / "media.bin"
        original.write_bytes(bytes(1048576) + bytes([37])*1048576 + rng.randbytes(1048576+37))
        packed = tmp / "media.jxtb"
        run(MEDIA, "encode", original, packed, 4096)
        run(MEDIA, "verify", packed, original)
        restored = tmp / "restored.bin"
        run(MEDIA, "decode", packed, restored)
        assert sha(original) == sha(restored)
        passed("binary multi-tile roundtrip with final partial tile")
        broken = bytearray(packed.read_bytes())
        broken[len(broken)//2] ^= 1
        corrupted = tmp / "media_bad.jxtb"; corrupted.write_bytes(broken)
        run(MEDIA, "verify", corrupted, original, expect=1)
        passed("corrupted arbitrary file rejected")
        tiny = tmp / "tiny.bin"; tiny.write_bytes(b"JARVIS-X\x00\xff")
        tiny_stream = tmp / "tiny.jxtb"
        run(MEDIA, "encode", tiny, tiny_stream)
        out = tmp / "tiny.out"
        run(MEDIA, "decode", tiny_stream, out)
        assert out.read_bytes() == tiny.read_bytes()
        assert tiny_stream.stat().st_size == len(tiny.read_bytes()) + 32
        passed("DIRECT mode preserves small incompressible files")
        run(MEDIA, "encode", tiny, tiny_stream, expect=1)
        passed("existing output protection")
    print(f"ALL {len(checks)} REGRESSION TESTS PASSED")
    if "--large" in sys.argv:
        print("1 GiB RUN:\n" + run(SIM, "--start", 0, "--tiles", 1024, "--corrupt-test"))

if __name__ == "__main__":
    main()
