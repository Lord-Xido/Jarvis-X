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



    def test_approved_pinned_synthetic_runtime_end_to_end(self):
        """Exercise the subprocess + source + SHA + independent-decode contract.

        This synthetic reversible ZIP fixture is deliberately NOT the private 3D codec.
        """
        from textwrap import dedent
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            docx = root / 'synthetic.docx'
            synthetic_docx(docx, ['energy = 7', 'print(energy)'])
            runtime = root / 'fixture_runtime.py'
            runtime.write_text(dedent(r'''
                import hashlib
                import json
                from pathlib import Path
                import sys
                import xml.etree.ElementTree as ET
                import zipfile

                def sha(data):
                    return hashlib.sha256(data).hexdigest()

                command, *args = sys.argv[1:]
                if command == 'iterate-docx':
                    source_docx = Path(args[0])
                    output = Path(args[args.index('--out') + 1])
                    layers = int(args[args.index('--layers') + 1])
                    with zipfile.ZipFile(source_docx) as source_zip:
                        root = ET.fromstring(source_zip.read('word/document.xml'))
                    namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                    paragraphs = [''.join(t.text or '' for t in p.iter(namespace + 't'))
                                  for p in root.iter(namespace + 'p')]
                    start = paragraphs.index('<<<DM3D-PYTHON-BEGIN>>>')
                    end = paragraphs.index('<<<DM3D-PYTHON-END>>>')
                    data = ('\n'.join(paragraphs[start + 1:end]) + '\n').encode('utf-8')
                    output.mkdir(parents=True, exist_ok=True)
                    for name in ('original.py', 'best.py', 'decoded_verified.py'):
                        (output / name).write_bytes(data)
                    with zipfile.ZipFile(output / 'best.dm3d', 'w') as package:
                        package.writestr('manifest.json', json.dumps({
                            'grid_side': 1,
                            'blocks': [{'index': 0, 'xyz': [0, 0, 0]}]
                        }))
                        package.writestr('source.bin', data)
                    report = {
                        'engine': 'DM3D-CLOSED-LOOP-2', 'layers': layers,
                        'final_sha256': sha(data), 'accepted_iterations': 0,
                        'halt_reason': 'synthetic_fixed_point'
                    }
                    (output / 'iteration_report.json').write_text(json.dumps(report))
                    print(json.dumps(report))
                elif command == 'nested-decode':
                    package, target = Path(args[0]), Path(args[1])
                    with zipfile.ZipFile(package) as archive:
                        data = archive.read('source.bin')
                    wanted = args[args.index('--sha256') + 1]
                    if sha(data) != wanted:
                        raise ValueError('SHA mismatch')
                    target.write_bytes(data)
                    print(json.dumps({'sha256': sha(data)}))
                else:
                    raise ValueError('unknown CLI command')
            ''').lstrip(), encoding='utf-8')
            proof = run(docx, runtime, root / 'output', digest(runtime.read_bytes()),
                        rounds=4, layers=2, approved=True)
            self.assertTrue(proof['verified'])
            self.assertEqual(proof['accepted_iterations'], 0)
            self.assertEqual(proof['geometry']['block_xyz'], [[0, 0, 0]])
            self.assertEqual((root / 'output' / 'best.py').read_bytes(),
                             b'energy = 7\nprint(energy)\n')
            self.assertTrue((root / 'output' / 'public_bridge_proof.json').is_file())


if __name__ == '__main__':
    unittest.main()
