import io
import json
import pathlib
import sys
import unittest
import zlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from codec import encode, decode, generate, evolve, CodecError, reconstruction_error, HEADER
from server import simulate

class TestJX3D(unittest.TestCase):
    def test_exact_roundtrip_every_geometry(self):
        for geom in ("mandelbulb", "torus", "sphere", "waves"):
            with self.subTest(geometry=geom):
                data = generate(16, geom)
                packet = encode(data, 16)
                decoded = decode(packet)
                self.assertEqual(decoded.volume, data)
                self.assertTrue(decoded.checksum_verified)
                self.assertEqual(decoded.mode, "lossless")

    def test_preview_is_approximate(self):
        v = generate(16, "sphere")
        p = encode(v, 16, "preview")
        d = decode(p)
        self.assertEqual(d.mode, "preview")
        self.assertGreater(reconstruction_error(v,d.volume)[1], 0)

    def test_tamper_is_rejected(self):
        p = bytearray(encode(generate(16,"waves"),16))
        p[HEADER.size - 1] ^= 1
        with self.assertRaises(CodecError): decode(bytes(p))
        p[0]=ord("X")
        with self.assertRaises(CodecError): decode(bytes(p))

    def test_bad_dimension_and_lengths(self):
        with self.assertRaises(CodecError): encode(b"", 17)
        with self.assertRaises(CodecError): encode(b"", 16)
        with self.assertRaises(CodecError): decode(b"bad")

    def test_recursive_lossless_verifies_every_pass(self):
        r = simulate({"dimension": 16, "cycles": 4, "geometry":"torus", "alpha":0.25})
        self.assertEqual(len(r["rounds"]), 4)
        self.assertTrue(all(p["verified"] and p["mse"]==0 for p in r["rounds"]))

    def test_heat_relaxation_constant_field(self):
        v = bytes([75])*16**3
        self.assertEqual(evolve(v,16,0.5),v)

    def test_payload_bomb_rejected(self):
        packet = encode(bytes([0])*16**3, 16)
        self.assertLess(len(packet),len(bytes([0])*16**3))
        self.assertEqual(decode(packet).volume,bytes([0])*16**3)

if __name__ == "__main__":
    unittest.main()
