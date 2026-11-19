import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server import validate


def run(*args):
    return subprocess.run([str(ROOT / 'jarvisx_vm'), *args], capture_output=True, text=True)


class NativeVmTests(unittest.TestCase):
    def test_verified_bounded_run(self):
        result = run('--side', '8', '--steps', '5')
        self.assertEqual(result.returncode, 0, result.stderr)
        item = json.loads(result.stdout)
        self.assertEqual(item['voxels'], 512)
        self.assertTrue(item['verified_finite'])
        self.assertTrue(1 <= item['iterations'] <= 5)
        self.assertEqual(len(item['opcodes']), 7)
        self.assertTrue(item['residual_rms'] >= 0)
        self.assertTrue(0 <= item['path_id'] <= 3)

    def test_determinism(self):
        a = json.loads(run('--seed', '19').stdout)
        b = json.loads(run('--seed', '19').stdout)
        self.assertEqual(a['fingerprint64'], b['fingerprint64'])
        self.assertEqual(a['path_id'], b['path_id'])
        self.assertAlmostEqual(a['residual_rms'], b['residual_rms'])

    def test_bad_arguments_fail_closed(self):
        for args in [('--steps', '-1'), ('--side', '200'), ('--alpha', 'nan'),
                     ('--beta', '0.9'), ('--no-such-op', '1')]:
            self.assertNotEqual(run(*args).returncode, 0)

    def test_gateway_rejects_invalid(self):
        for payload in [{'steps': -1}, {'side': 1000000}, {'seed': True},
                        {'steps': '4'}, {'command': 'rm -rf /'}]:
            with self.assertRaises(ValueError):
                validate(payload)
        self.assertEqual(len(validate({})), 16)


if __name__ == '__main__':
    unittest.main()
