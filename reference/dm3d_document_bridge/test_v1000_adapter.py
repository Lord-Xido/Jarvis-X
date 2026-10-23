"""Synthetic, non-proprietary contract tests for the public DM3D 1000 adapter."""
import base64
import binascii
import hashlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

import v1000_adapter as a


def fixture(*, fault=None):
    """Make a disposable mock package; never include private VM instructions."""
    ns = a.NS
    root = ET.Element(f'{{{ns}}}machine', version='2.0')
    fields = ET.SubElement(root, f'{{{ns}}}manifest')
    code = bytearray()
    for pc in range(1000):
        header = struct.pack('>HBBH', pc, 0xAA, 0, 0)
        code.extend(header + struct.pack('>H', binascii.crc_hqx(header, 0xffff)))
    program = bytes(code)
    if fault == 'crc':
        program = program[:5] + bytes([program[5] ^ 1]) + program[6:]
    payload = b'mock-payload-not-private-source'
    metadata = {
        'format': 'DM3D-1000-VM', 'version': '2.0', 'instructions': '1000',
        'bytecode_bytes': '8000', 'tile_bytes': '512', 'length': '100',
        'tiles': '1', 'grid_side': '1', 'input_size': str(len(payload)),
        'source_sha256': hashlib.sha256(b'mock').hexdigest(),
        'input_sha256': hashlib.sha256(payload).hexdigest(),
        'program_sha256': hashlib.sha256(program).hexdigest(),
    }
    if fault == 'bad-hash':
        metadata['input_sha256'] = '0' * 64
    if fault == 'invalid-geometry':
        metadata['grid_side'] = '3'
    for name, value in metadata.items():
        ET.SubElement(fields, f'{{{ns}}}field', name=name, value=value)
    ET.SubElement(root, f'{{{ns}}}program', encoding='hex',
        record='8-byte-pc-op-count-index-crc16').text = program.hex()
    ET.SubElement(root, f'{{{ns}}}payload', encoding='base64+zlib').text = base64.b64encode(payload).decode()
    if fault == 'duplicate-program':
        ET.SubElement(root, f'{{{ns}}}program', encoding='hex').text = program.hex()
    xml = ET.tostring(root)
    if fault == 'doctype':
        xml = b'<!DOCTYPE dm:machine [<!ENTITY ignored "not-used">]>' + xml
    with io.BytesIO() as stream:
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('[Content_Types].xml', '<Types/>')
            z.writestr('_rels/.rels', '<Relationships/>')
            if fault != 'missing-word':
                z.writestr('word/document.xml', '<w:document xmlns:w="urn:test"/>')
            z.writestr(a.PART, xml)
            if fault == 'traversal':
                z.writestr('../escape.txt', 'unexpected')
            if fault == 'duplicate-path':
                z.writestr(a.PART, xml)
        return stream.getvalue()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.path = self.base / 'synthetic.docx'

    def inspect(self, fault=None):
        self.path.write_bytes(fixture(fault=fault))
        return a.inspect_document(self.path)

    def test_synthetic_valid(self):
        summary = self.inspect()
        self.assertEqual(summary['instructions'], 1000)
        self.assertEqual(summary['bytecode_bytes'], 8000)
        self.assertEqual(summary['source_bytes'], 100)

    def test_bad_program_crc(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'CRC'):
            self.inspect('crc')

    def test_bad_payload_digest(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'digest'):
            self.inspect('bad-hash')

    def test_invalid_geometry(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'geometry'):
            self.inspect('invalid-geometry')

    def test_duplicate_archive_path(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'duplicate'):
            self.inspect('duplicate-path')

    def test_missing_word_part_rejected(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'Word OOXML'):
            self.inspect('missing-word')

    def test_xml_doctype_rejected(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'document type'):
            self.inspect('doctype')

    def test_duplicate_program_node_rejected(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'exactly one'):
            self.inspect('duplicate-program')

    def test_zip_path_traversal_rejected(self):
        with self.assertRaisesRegex(a.InvalidPackage, 'invalid'):
            self.inspect('traversal')

    def test_execution_denied_without_approval(self):
        self.inspect()
        with self.assertRaises(PermissionError):
            a.run_approved_runtime(self.path, self.base / 'runtime.py', self.base / 'output.bin')

    def test_explicit_external_runtime_and_pin(self):
        self.inspect()
        runtime = self.base / 'trusted-runtime.py'
        runtime.write_text('pass\n')
        digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
        with patch.object(a.subprocess, 'run') as run:
            run.return_value.returncode = 0
            result = a.run_approved_runtime(self.path, runtime, self.base / 'out.bin',
                                            approved=True, runtime_sha256=digest)
            self.assertEqual(result.returncode, 0)
            argv = run.call_args.args[0]
            self.assertEqual(argv[2], 'run')
            self.assertEqual(argv[4], '--output')
            run.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            a.run_approved_runtime(self.path, runtime, self.base / 'out.bin',
                                  approved=True, runtime_sha256='0'*64)


if __name__ == '__main__':
    unittest.main()
