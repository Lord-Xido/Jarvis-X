"""Deterministic bounded regression suite for the public reference bridge."""
import hashlib
import io
import os
import unittest
import zipfile
from pathlib import Path

import codec


class PublicReferenceTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(codec.decode(codec.encode(b"")), b"")

    def test_arithmetic_ast_string(self):
        source = b'{"S":"10^24","N":"S^(S^S)"}'
        self.assertEqual(codec.decode(codec.encode(source)), source)

    def test_binary_zeros(self):
        source = bytes(range(256)) * 16 + b"\0" * 150
        self.assertEqual(codec.decode(codec.encode(source)), source)

    def test_incompressible_bytes(self):
        source = bytes((i * 73 + i * i * 19) % 256 for i in range(4000))
        self.assertEqual(codec.decode(codec.encode(source)), source)

    def test_self_source(self):
        source = Path(codec.__file__).read_bytes()
        self.assertEqual(codec.decode(codec.encode(source)), source)
        self.assertEqual(hashlib.sha256(codec.decode(codec.encode(source))).digest(), hashlib.sha256(source).digest())

    def test_volume_transform_16_and_32(self):
        for side in codec.SIDES:
            with self.subTest(side=side):
                data = bytes((i * 37) % 256 for i in range(side ** 3))
                self.assertEqual(codec._decode_3d(codec._encode_3d(data, side), side), data)

    def test_reject_over_limit(self):
        with self.assertRaises(ValueError):
            codec.encode(b"x" * (codec.MAX_INPUT + 1))

    def test_reject_missing_embedded_image(self):
        with self.assertRaises(ValueError):
            codec.decode(b"not-a-docx")

    def test_reject_png_pixel_tamper(self):
        archive = codec.encode(b"3d" * 999)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            content = {name: z.read(name) for name in z.namelist()}
        png = bytearray(content[codec.MEDIA])
        png[-12] ^= 1
        content[codec.MEDIA] = bytes(png)
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, data in content.items():
                z.writestr(name, data)
        with self.assertRaises(ValueError):
            codec.decode(out.getvalue())

    def test_word_opens_embedded_image(self):
        from docx import Document  # optional local validation; CI installs python-docx
        doc = Document(io.BytesIO(codec.encode(b"example")))
        self.assertEqual(len(doc.inline_shapes), 1)
        self.assertIn("DM3D Public Reference", doc.paragraphs[0].text)


if __name__ == "__main__":
    unittest.main()
