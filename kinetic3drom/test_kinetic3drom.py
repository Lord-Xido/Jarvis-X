import unittest
from kinetic3drom.kinetic3drom import Kinetic3DROM

class TestKineticROM(unittest.TestCase):
    def test_roundtrip(self):
        e = Kinetic3DROM(64)
        for p in (0, 1, 12345, 54320, 54321, 64**3-1):
            self.assertEqual(e.decode(e.encode(p)), p)

    def test_nearby_retrieval(self):
        e = Kinetic3DROM()
        e.etch_rom([12345, 54321, 99999])
        result = e.run(54320)
        self.assertIsNotNone(result)
        self.assertEqual(result["packet"], 54321)

    def test_virtual_scale(self):
        e = Kinetic3DROM(100000)
        p = 10**15-1
        self.assertEqual(e.decode(e.encode(p)), p)

    def test_empty_rom(self):
        self.assertIsNone(Kinetic3DROM().run(100))

if __name__ == "__main__":
    unittest.main()
