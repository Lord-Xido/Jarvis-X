"""Public synthetic tests: no private SDK, DOCX, or intellectual property."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from xml.sax.saxutils import escape
import zipfile

from v3_iterative_adapter import digest, geometry, inspect_docx, run

WORD_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def synthetic_docx(path: Path, lines: list[str], *, extra: str = '') -> None:
    code = ['<<<DM3D-PYTHON-BEGIN>>>', *lines, '<<<DM3D-PYTHON-END>>>']
    ps = ''.join(f'<w:p><w:r><w:t>{escape(line)}</w:t></w:r></w:p>' for line in code)
    xml = f'<w:document xmlns:w="{WORD_NS}"><w:body>{ps}{extra}</w:body></w:document>'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('word/document.xml', xml)


class V3PublicBridgeTests(unittest.TestCase):
    def test_inspect_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'simple.docx'
            synthetic_docx(path, ['x = 2 + 3', 'print(x)'])
            proof = inspect_docx(path)
            self.assertEqual(proof['source_sha256'], digest(b'x = 2 + 3\nprint(x)\n'))
            self.assertEqual(proof['source_lines'], 2)

    def test_reject_missing_and_duplicated_markers(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'dup.docx'
            synthetic_docx(path, ['<<<DM3D-PYTHON-BEGIN>>>', 'print(1)'])
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                inspect_docx(path)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('word/document.xml', '<root/>')
            with self.assertRaises(ValueError):
                inspect_docx(path)

    def test_reject_xml_entities(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'entity.docx'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('word/document.xml', b'<!DOCTYPE x [<!ENTITY a "bad">]><x/>')
            with self.assertRaisesRegex(ValueError, 'entities'):
                inspect_docx(path)

    def test_reject_duplicate_zip_member(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'duplicate.docx'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('word/document.xml', '<root/>')
                archive.writestr('word/document.xml', '<root/>')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                inspect_docx(path)

    def test_read_outer_geometry_and_morton_check(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'sample.dm3d'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('manifest.json', json.dumps({'grid_side': 2, 'blocks': [
                    {'index': 0, 'xyz': [0, 0, 0]}, {'index': 1, 'xyz': [1, 0, 0]},
                    {'index': 2, 'xyz': [0, 1, 0]}, {'index': 3, 'xyz': [1, 1, 0]}]}))
            self.assertEqual(geometry(path)['logical_blocks'], 4)
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('manifest.json', json.dumps({'grid_side': 2, 'blocks': [
                    {'index': 0, 'xyz': [1, 0, 0]}]}))
            with self.assertRaisesRegex(ValueError, 'Morton'):
                geometry(path)

    def test_runtime_permission_and_pin_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            runtime = root / 'sdk.py'
            runtime.write_text('raise AssertionError("must not execute")\n')
            docx = root / 'example.docx'
            synthetic_docx(docx, ['print(1)'])
            with self.assertRaises(PermissionError):
                run(docx, runtime, root / 'out', '0'*64, rounds=2, layers=1, approved=False)
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                run(docx, runtime, root / 'out', '0'*64, rounds=2, layers=1, approved=True)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
