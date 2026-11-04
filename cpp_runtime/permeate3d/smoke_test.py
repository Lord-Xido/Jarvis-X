#!/usr/bin/env python3
"""End-to-end CLI conformance for the independently reconstructed JXA codec.

Runs from a temporary directory and doesn't commit training artifacts.
Requires only Python 3 and the compiled C++20 executable.
"""
import pathlib
import re
import struct
import subprocess
import sys
import tempfile


def run(*args, success=True):
    completed = subprocess.run(
        [str(arg) for arg in args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if (completed.returncode == 0) != success:
        raise AssertionError(
            f"Unexpected exit {completed.returncode}: {args!r}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout + completed.stderr


def header(path):
    data = path.read_bytes()
    if len(data) < 44:
        raise AssertionError("too short for JXA header")
    fields = struct.unpack_from("<8sIIIIQQI", data)
    magic, version, dim, side, original_length, fingerprint, count = (None,)*7
    magic, version, dim, side, mode, original_length, fingerprint, count = fields
    assert magic == b"JXAE3D03" and version == 1
    assert dim == 1000 and side == 10 and original_length == 1_000_000_000
    assert mode in (1, 2) and 0 < fingerprint < 2**64
    return mode, count, len(data)


def require_stats(out, visited, records, elided, macs):
    for key, n in (
        ("visited", visited), ("records", records),
        ("zero-elided", elided), ("analytical_MACs", macs)
    ):
        assert re.search(rf"\b{re.escape(key)}={n}\b", out), out


def main(binary):
    binary = pathlib.Path(binary).resolve()
    with tempfile.TemporaryDirectory(prefix="jarvisx-permeate-ci-") as td:
        p = pathlib.Path(td)
        model = p / "model.jxm"
        self_stream = p / "self.jxa"
        lossless = p / "lossless.jxa"
        lossy = p / "lossy.jxa"
        terminal = p / "corner.jxa"
        assert "SELFTEST PASS" in run(binary, "selftest")
        run(binary, "train", model, 500)
        assert model.stat().st_size == 908
        assert "PERMEATE SELFTEST PASS" in run(binary, "permeate-selftest", model, self_stream)
        assert header(self_stream)[:2] == (1, 18)
        assert "VERIFY PASS" in run(binary, "permeate-verify", model, self_stream)

        for mode, path, mode_id in (("lossless", lossless, 1), ("lossy", lossy, 2)):
            output = run(binary, "permeate", model, path, 50, 50, 50, 2, mode, 7)
            require_stats(output, 125, 108, 17, 24_786_000)
            assert header(path)[:2] == (mode_id, 108)
            assert "VERIFY PASS" in run(binary, "permeate-verify", model, path)

        # Absolute terminal corner in global voxel coordinates.
        terminal_result = run(binary, "permeate", model, terminal, 99, 99, 99, 1, "lossless")
        require_stats(terminal_result, 8, 8, 0, 8 * 229_500)
        assert header(terminal)[:2] == (1, 8)
        assert "voxel[999,999,999]=" in run(binary, "probe", model, terminal, 999, 999, 999)
        assert "VERIFY PASS" in run(binary, "permeate-verify", model, terminal)

        corrupted = p / "corrupted.jxa"
        content = bytearray(lossless.read_bytes())
        content[-1] ^= 0x01  # mutate last stored residual, not the header
        corrupted.write_bytes(content)
        assert "CRC mismatch" in run(binary, "permeate-verify", model, corrupted, success=False)
        assert "already exists" in run(binary, "permeate", model, lossless, 50, 50, 50, 1, "lossless", success=False)
        assert "outside permitted range" in run(binary, "permeate", model, p / "invalid.jxa", 101, 0, 0, 0, "lossless", success=False)

    print("PERMEATE C++20 conformance PASS: neural codec, 1M tile shell enumeration, "
          "sparse lossless/lossy streams, CRC tampering and boundary handling")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_test.py /path/to/jarvisx-permeate3d")
    main(sys.argv[1])
