"""Structural safeguards for the user-supplied DM3D visual demonstration."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

APP = Path(__file__).resolve().parent / 'index.html'


class ObservatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = APP.read_text(encoding='utf-8')
        cls.module = re.search(r'<script type="module">([\s\S]*?)</script>', cls.html)

    def test_document_structure(self):
        self.assertTrue(self.html.startswith('<!DOCTYPE html>'))
        self.assertIn('<div id="root"></div>', self.html)
        self.assertIsNotNone(self.module)
        self.assertEqual(self.html.count('<script type="module">'), 1)
        self.assertNotIn('ecto:artifact', self.html)
        self.assertNotIn('data-clippy-inline-media', self.html)

    def test_disclosure(self):
        self.assertIn('id="dm3d-evidence-banner"', self.html)
        self.assertIn('SIMULATED VISUALISATION', self.html)
        self.assertIn('symbolic address-space', self.html)
        self.assertIn('AUTO-EVOLVE DEMO', self.module.group(1))
        self.assertIn('SIMULATED', self.module.group(1))

    def test_real_rendering_but_synthetic_telemetry(self):
        module = self.module.group(1)
        self.assertIn('requestAnimationFrame', module)
        self.assertIn('getContext("2d")', module)
        self.assertIn('setMatrixAt', module)
        self.assertIn('Math.random()', module)
        self.assertIn('1GB³ BIT-MAPPED', module)
        self.assertIn('jJ=32', module)
        self.assertIn('F0=512', module)

    def test_no_remote_script_dependencies(self):
        self.assertNotRegex(self.html, r'<script[^>]+src=')
        self.assertNotIn('src="/fonts/', self.html)

    def test_node_module_syntax(self):
        if not shutil.which('node'):
            self.skipTest('Node.js not installed')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle.mjs'
            path.write_text(self.module.group(1), encoding='utf-8')
            result = subprocess.run(['node', '--check', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[:2000])


if __name__ == '__main__':
    unittest.main()
