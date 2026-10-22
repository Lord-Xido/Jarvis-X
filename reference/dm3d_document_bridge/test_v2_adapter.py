"""Synthetic fixtures only: no private source or DOCX packaged in git."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from v2_adapter import InvalidPackage, inspect_document, run_approved_runtime


def fixture(path, *, mutate=None):
    source = b'# synthetic code (never executed from Word)\n'
    parts = {
        'dm3d/runtime.py': source,
        'dm3d/runtime.sha256': hashlib.sha256(source).hexdigest().encode(),
        'dm3d/programs.json': b'{"demo":[{"op":"HALT"}]}',
        'dm3d/model.json': b'{"architecture":[26,12,1]}',
        'dm3d/volumes.json': b'{"ram":{},"rom":{},"rom_sealed":true}',
        'dm3d/assets.json': b'{"objects":{}}',
        '[Content_Types].xml': b'<Types/>',
        'word/document.xml': b'<document/>',
    }
    if mutate:
        mutate(parts)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.docx = self.root / 'demo.docx'
        fixture(self.docx)

    def test_valid_inspect(self):
        report = inspect_document(self.docx)
        self.assertEqual(report['programs'], ['demo'])
        self.assertEqual(report['ann_architecture'], [26, 12, 1])
        self.assertTrue(report['rom_sealed'])

    def test_tampered_runtime_rejected(self):
        fixture(self.docx, mutate=lambda p: p.__setitem__('dm3d/runtime.py', b'TAMPER'))
        with self.assertRaisesRegex(InvalidPackage, 'digest'):
            inspect_document(self.docx)

    def test_missing_required_member_rejected(self):
        fixture(self.docx, mutate=lambda p: p.pop('dm3d/model.json'))
        with self.assertRaisesRegex(InvalidPackage, 'missing'):
            inspect_document(self.docx)

    def test_duplicate_zip_name_rejected(self):
        with zipfile.ZipFile(self.docx, 'a') as archive:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                archive.writestr('dm3d/runtime.py', b'TAMPER')
        with self.assertRaisesRegex(InvalidPackage, 'duplicate'):
            inspect_document(self.docx)

    def test_unsafe_path_rejected(self):
        with zipfile.ZipFile(self.docx, 'a') as archive:
            archive.writestr('../escape.txt', b'xx')
        with self.assertRaisesRegex(InvalidPackage, 'invalid ZIP'):
            inspect_document(self.docx)

    def test_unauthorized_run_rejected(self):
        with self.assertRaises(PermissionError):
            run_approved_runtime(self.docx, '/nonexistent.py', 'demo', self.root)

    def test_unknown_program_rejected(self):
        runner = self.root / 'runner.py'
        runner.write_text('print("hello")\n')
        with self.assertRaises(ValueError):
            run_approved_runtime(self.docx, runner, 'not-present', self.root, approved=True)

    def test_injection_rejected(self):
        runner = self.root / 'runner.py'
        runner.write_text('print("hello")\n')
        with self.assertRaises(ValueError):
            run_approved_runtime(self.docx, runner, '--unsafe', self.root, approved=True)

    def test_explicit_local_runner(self):
        runner = self.root / 'runner.py'
        runner.write_text('import sys\nprint("ARGS:"+"|".join(sys.argv[1:]))\n')
        proc = run_approved_runtime(self.docx, runner, 'demo', self.root/'out', approved=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('run|', proc.stdout)
        self.assertIn('|demo|--out|', proc.stdout)

    def test_cli(self):
        cli = Path(__file__).with_name('v2_adapter.py')
        done = subprocess.run([sys.executable, str(cli), 'inspect', str(self.docx)],
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)['format'], 'dm3d-docx-v2')


if __name__ == '__main__':
    unittest.main()
