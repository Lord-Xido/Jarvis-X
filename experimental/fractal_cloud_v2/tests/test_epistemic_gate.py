#!/usr/bin/env python3
"""Black-box epistemic safety checks using actual JX3D binary and fake processes."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from epistemic_gate import Verdict, supervise, verify_evidence


class GateTests(unittest.TestCase):
    binary = None
    golden = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='jx3d-ignorance-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_true_verified_run(self):
        result = supervise(self.binary, self.root / 'true', 2, 24, 2, 2, 30, self.golden)
        self.assertEqual(result['verdict'], 'TRUE')
        self.assertTrue((self.root / 'true' / 'verified' / 'checkpoint.bin').is_file())
        self.assertEqual(json.loads((self.root / 'true' / 'epistemic_decision.json').read_text())['status'], 'COMMITTED')

    def test_false_injected_arithmetic_rejection(self):
        result = supervise(self.binary, self.root / 'false', 2, 24, 2, 3, 30, fault_step=2)
        self.assertEqual(result['verdict'], 'FALSE')
        self.assertEqual(len(result['attempts']), 1)
        self.assertFalse((self.root / 'false' / 'verified').exists())
        checkpoint = (self.root / 'false' / 'attempt_01' / 'checkpoint.bin').read_bytes()
        self.assertEqual(struct.unpack_from('<I', checkpoint, 12)[0], 1)

    def test_unknown_timeout_bounded_and_abstained(self):
        script = self.root / 'stall.py'
        script.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n')
        script.chmod(0o755)
        result = supervise(script, self.root / 'unknown', 2, 24, 1, 2, 0.18)
        self.assertEqual(result['status'], 'SAFE_HALT_UNKNOWN')
        self.assertEqual(len(result['attempts']), 2)
        self.assertFalse((self.root / 'unknown' / 'verified').exists())
        self.assertTrue(all(x['verdict'] == 'UNKNOWN' for x in result['attempts']))

    def test_unknown_missing_evidence_never_commits(self):
        script = self.root / 'empty.py'
        script.write_text('#!/usr/bin/env python3\n')
        script.chmod(0o755)
        result = supervise(script, self.root / 'missing', 2, 24, 1, 1, 5)
        self.assertEqual(result['verdict'], 'UNKNOWN')
        self.assertFalse((self.root / 'missing' / 'verified').exists())

    def test_tampered_checkpoint_rejected(self):
        supervise(self.binary, self.root / 'good', 2, 24, 1, 1, 30, self.golden)
        directory = self.root / 'good' / 'verified'
        checkpoint = directory / 'checkpoint.bin'
        data = bytearray(checkpoint.read_bytes())
        data[-1] ^= 1
        checkpoint.write_bytes(data)
        verdict, reason, _ = verify_evidence(directory, 2, 24, self.golden)
        self.assertEqual(verdict, Verdict.FALSE, reason)

    def test_golden_trace_drift_rejected(self):
        supervise(self.binary, self.root / 'good', 2, 24, 1, 1, 30, self.golden)
        directory = self.root / 'good' / 'verified'
        with (directory / 'trace.csv').open('a') as f:
            f.write('trailing corrupted row\n')
        verdict, reason, _ = verify_evidence(directory, 2, 24, self.golden)
        self.assertEqual(verdict, Verdict.FALSE, reason)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('Usage: python test_epistemic_gate.py /path/jx3d_cloud /path/golden.csv')
    GateTests.binary = Path(sys.argv[1]).resolve()
    GateTests.golden = Path(sys.argv[2]).resolve()
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
