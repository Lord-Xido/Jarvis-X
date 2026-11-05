#!/usr/bin/env python3
"""End-to-end finite DM3D spectral/swarm/codec/ledger checks, standard library only."""
import csv
import pathlib
import subprocess
import sys
import tempfile


def invoke(executable, *args, success=True):
    p = subprocess.run([str(executable), *map(str, args)], text=True,
                       capture_output=True, check=False, timeout=60)
    if (p.returncode == 0) != success:
        raise AssertionError(f"Unexpected exit={p.returncode}: {' '.join(map(str, args))}\n"
                             f"stdout={p.stdout}\nstderr={p.stderr}")
    return p.stdout + p.stderr


def check_trace(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows and all(float(r["objective"]) >= 0 for r in rows)
    values = [float(r["objective"]) for r in rows]
    assert all(b <= a + 1e-11 for a, b in zip(values, values[1:])), values
    assert values[-1] < values[0], values
    return values[-1]


def main():
    exe = pathlib.Path(sys.argv[1]).resolve()
    assert exe.is_file()
    invoke(exe, "--self-test")
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        input_file = root / "input.bin"
        encoded = root / "encoded.dm3d"
        decoded = root / "recovered.bin"
        # Non-text test includes all 256 byte values, embedded NUL and uneven length.
        data = bytes(range(256)) * 7 + b"\0DM3D\xff"
        input_file.write_bytes(data)
        invoke(exe, "--encode-file", input_file, encoded)
        invoke(exe, "--decode-file", encoded, decoded)
        assert decoded.read_bytes() == data, "Codec must be byte-exact"
        corrupt = bytearray(encoded.read_bytes())
        assert len(corrupt) > 32
        corrupt[-1] ^= 1
        tampered = root / "tampered.dm3d"
        tampered.write_bytes(corrupt)
        invoke(exe, "--decode-file", tampered, root / "tampered_out.bin", success=False)
        assert not (root / "tampered_out.bin").exists(), "Tampered data must not publish"
        fft_trace = root / "fft.csv"
        saved = root / "state.dm3ds"
        ledger = root / "ledger.csv"
        output = invoke(exe, "--side", 16, "--workers", 8, "--threads", 2,
                        "--steps", 4, "--solver", "fft", "--query", "22/7",
                        "--trace", fft_trace, "--ledger", ledger,
                        "--save-state", saved)
        assert "solver=spectral-3d" in output
        assert "CTR status: PASS" in output
        assert any(row["numerator"] == "22" and row["denominator"] == "7" and row["verified"] == "1"
                   for row in csv.DictReader(ledger.open(newline="")))
        fft_objective = check_trace(fft_trace)
        restored_trace = root / "restored.csv"
        invoke(exe, "--side", 16, "--steps", 0, "--load-state", saved,
               "--trace", restored_trace)
        assert abs(float(list(csv.DictReader(restored_trace.open()))[0]["objective"])
                   - fft_objective) < 1e-11
        fallback = root / "fallback.csv"
        output = invoke(exe, "--side", 10, "--workers", 8, "--threads", 2,
                        "--steps", 3, "--solver", "auto", "--trace", fallback)
        assert "solver=verified-swarm" in output
        check_trace(fallback)
        invoke(exe, "--side", 10, "--steps", 1, "--solver", "fft", success=False)
    print("INTEGRATION PASS: exact codec, tamper rejection, RAG, FFT, restart, swarm fallback")


if __name__ == "__main__":
    main()
