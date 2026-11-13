"""Tests for the bounded DM-vOmegaXi+ 3D quine."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAM = HERE / "dm_quine.py"
sys.path.insert(0, str(HERE))
import dm_quine as q


def run(*args):
    return subprocess.run([sys.executable, str(PROGRAM), *map(str, args)],
                          capture_output=True, check=False)


class QuineTests(unittest.TestCase):
    def test_exact_byte_quine(self):
        result = run("--quine")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stdout, PROGRAM.read_bytes())
        self.assertEqual(q.source_of(q.GENOME).encode(), PROGRAM.read_bytes())

    def test_selftest(self):
        result = run("--selftest")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b"PASS:", result.stdout)

    def test_geometry(self):
        self.assertEqual(len(q.VOLUME), 8**3)
        self.assertEqual(len(q.LATENT), 4**3)
        self.assertGreater(q.mse([0, 0, 0]), 0)
        self.assertLessEqual(q.mse([16, 16, 16]), q.mse([0, 0, 0]))

    def test_deterministic_fitness_selection(self):
        a = q.evolve([0, 0, 0], 200, 707)
        b = q.evolve([0, 0, 0], 200, 707)
        self.assertEqual(a, b)
        self.assertLessEqual(a[1], q.mse([0, 0, 0]))
        self.assertTrue(all(a[3][i]["loss_mse"] <= a[3][i-1]["loss_mse"] + 1e-12
                            for i in range(1, len(a[3]))))
        self.assertTrue(all(row["committed_sha256"] == q.digest(q.source_of(row["genome"]))
                            for row in a[3]))

    def test_emit_child_is_itself_a_quine(self):
        with tempfile.TemporaryDirectory() as td:
            child = Path(td) / "child.py"
            trace = Path(td) / "trace.jsonl"
            result = run("--steps", 80, "--seed", 707, "--streams", 1000000,
                         "--emit-child", child, "--trace-out", trace)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            summary = json.loads(result.stdout)
            self.assertEqual(summary["logical_streams"], 1000000)
            self.assertEqual(summary["copies_materialized"], 1)
            self.assertFalse(summary["speedup_1000x_proven"])
            self.assertLessEqual(summary["final_mse"], summary["initial_mse"])
            reproduced = subprocess.run([sys.executable, str(child), "--quine"],
                                        capture_output=True, check=False)
            self.assertEqual(reproduced.returncode, 0, reproduced.stderr.decode())
            self.assertEqual(reproduced.stdout, child.read_bytes())
            check = subprocess.run([sys.executable, str(child), "--selftest"],
                                   capture_output=True, check=False)
            self.assertEqual(check.returncode, 0, check.stderr.decode())
            rows = [json.loads(x) for x in trace.read_text().splitlines()]
            self.assertEqual(len(rows), 80)
            self.assertEqual(rows[-1]["committed_sha256"], summary["descendant_sha256"])

    def test_resource_bounds(self):
        self.assertNotEqual(run("--steps", 1001).returncode, 0)
        self.assertNotEqual(run("--streams", 1000001).returncode, 0)
        self.assertNotEqual(run("--steps", -1).returncode, 0)

    def test_fail_closed_exclusive_write(self):
        with tempfile.TemporaryDirectory() as td:
            child = Path(td) / "child.py"
            child.write_text("DO NOT OVERWRITE\n")
            result = run("--steps", 1, "--emit-child", child)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(child.read_text(), "DO NOT OVERWRITE\n")

    def test_tamper_detection(self):
        with tempfile.TemporaryDirectory() as td:
            modified = Path(td) / "modified.py"
            modified.write_bytes(PROGRAM.read_bytes() + b"# unauthorized edit\n")
            result = subprocess.run([sys.executable, str(modified), "--selftest"],
                                    capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"quine byte mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
