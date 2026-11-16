"""Focused CPU unit tests; run: python -m unittest discover -s . -p 'test_*.py'"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ann3d_memory import Autoencoder3D, decode, encode, make_volume


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.archive = Path(cls.tmp.name)/'sample.npz'
        cls.volume = make_volume(width=32, height=24, depth=8)
        cls.prediction = encode(cls.volume, cls.archive, steps=4)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_reconstruction(self):
        self.assertTrue(np.array_equal(decode(self.archive), self.volume))

    def test_approximate_shape(self):
        self.assertEqual(decode(self.archive, exact=False).shape, self.volume.shape)
        self.assertEqual(self.prediction.shape, self.volume.shape)

    def test_latent_is_smaller_than_raw(self):
        expected_latent = 12*(8//4)*(24//4)*(32//4)
        self.assertLess(expected_latent, self.volume.nbytes)
        self.assertGreater(self.archive.stat().st_size, 0)

    def test_modified_residual_is_rejected(self):
        with np.load(self.archive, allow_pickle=False) as f:
            data = {k: f[k].copy() for k in f.files}
        data['residual'][0, 0, 0, 0] ^= 1
        altered = Path(self.tmp.name)/'tampered.npz'
        np.savez_compressed(altered, **data)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            decode(altered)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            make_volume(width=33, height=24, depth=8)


if __name__ == '__main__':
    unittest.main()
