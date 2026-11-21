#!/usr/bin/env python3
"""Independent process-level conformance for isolated DM3D-GEOM1 C++ runtime."""
import argparse
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--exe", type=Path, required=True)
KNOWN, _ = PARSER.parse_known_args()
EXE = KNOWN.exe.resolve()


def execute(*args, success=True):
    p = subprocess.run([str(EXE), *map(str, args)], capture_output=True, text=True, timeout=30)
    if success and p.returncode != 0:
        raise AssertionError(f"command failed ({p.returncode}): {p.stderr}")
    if not success and p.returncode == 0:
        raise AssertionError("invalid command was accepted")
    return p


class GeometricLoopProcessTests(unittest.TestCase):
    def test_compiled_invariants(self):
        self.assertIn("SELF_TEST_OK checks=13", execute("--self-test").stdout)

    def test_external_raw_u8_bitwise_roundtrip(self):
        rng = random.Random(20261029)
        for side, brick in [(4, 4), (8, 4), (8, 2), (16, 8)]:
            with self.subTest(side=side, brick=brick), tempfile.TemporaryDirectory() as d:
                d = Path(d)
                raw = bytes(rng.randrange(256) for _ in range(side**3))
                (d / "input.u8").write_bytes(raw)
                p = json.loads(execute("--side", side, "--brick", brick,
                                       "--input", d / "input.u8", "--passes", 4,
                                       "--out-prefix", d / "codec").stdout)
                self.assertTrue(p["byte_identical_recovery"])
                self.assertEqual((d / "codec.recovered.u8").read_bytes(), raw)
                self.assertEqual((d / "codec.dm3dtri").stat().st_size, p["wire_bytes"])
                self.assertEqual(p["residual_bytes"], side**3)
                self.assertTrue(p["passes"][1]["fixed_point"])
                self.assertEqual(p["passes"][1]["mse_to_previous"], 0)
                self.assertFalse(any(x["period_two"] for x in p["passes"]))

    def test_lossy_is_explicit_and_bounded(self):
        data = json.loads(execute("--lossy", "--side", 8, "--brick", 4, "--passes", 4).stdout)
        self.assertFalse(data["lossless"])
        self.assertFalse(data["byte_identical_recovery"])
        self.assertEqual(data["latent_bytes"], 64)
        self.assertEqual(data["residual_bytes"], 0)
        self.assertEqual(data["wire_bytes"], 26 + 64)
        self.assertGreater(data["first_pass_mse"], 0)

    def test_fail_closed_invalid_geometry(self):
        for args in [("--side", 0), ("--side", 65), ("--brick", 0),
                     ("--side", 9, "--brick", 4), ("--passes", 0),
                     ("--passes", 101), ("--nonexistent",), ("--side", "abc")]:
            with self.subTest(args=args):
                self.assertIn("ERROR:", execute(*args, success=False).stderr)

    def test_input_size_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "short.u8"
            src.write_bytes(b"bad")
            self.assertIn("exactly side^3", execute("--input", src, success=False).stderr)

    def test_coordinate_orientation_is_actually_3d(self):
        side = 8
        raw = bytes((30*x+3*y+z) % 256 for z in range(side)
                    for y in range(side) for x in range(side))
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "spatial.u8"
            src.write_bytes(raw)
            result = json.loads(execute("--input", src, "--side", 8, "--brick", 4).stdout)
            samples = {tuple(row[:3]): row for row in result["sampled_voxels"]}
            self.assertEqual(samples[(1, 0, 0)][3], 30)
            self.assertEqual(samples[(0, 1, 0)][3], 3)
            self.assertEqual(samples[(0, 0, 1)][3], 1)
            self.assertEqual(result["first_pass_mse"], 0)


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)