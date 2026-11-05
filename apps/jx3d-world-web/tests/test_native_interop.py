"""Python <-> native C++20 packet interoperability. Set JX3D_NATIVE to executable."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from codec import encode, decode, generate

class NativeInterop(unittest.TestCase):
    def test_both_directions(self):
        executable = os.environ.get('JX3D_NATIVE')
        if not executable:
            self.skipTest('JX3D_NATIVE was not configured')
        native = str(Path(executable).resolve())
        for mode in ('lossless', 'preview'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                p = Path(td)
                raw = generate(16, 'mandelbulb')
                (p/'input.raw').write_bytes(raw)
                (p/'python.jx3d').write_bytes(encode(raw, 16, mode))
                subprocess.run([native,'decode',str(p/'python.jx3d'),str(p/'native.raw')],check=True,capture_output=True)
                self.assertEqual((p/'native.raw').read_bytes(),decode((p/'python.jx3d').read_bytes()).volume)
                subprocess.run([native,'encode',str(p/'input.raw'),'16',mode,str(p/'native.jx3d')],check=True,capture_output=True)
                self.assertEqual(decode((p/'native.jx3d').read_bytes()).volume,(p/'native.raw').read_bytes())
if __name__=='__main__': unittest.main()
