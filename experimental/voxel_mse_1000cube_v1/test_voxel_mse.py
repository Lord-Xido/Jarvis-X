"""Unittest suite; no additional test framework beyond PyTorch/NumPy."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from voxel_mse import (SpatialAutoencoder, UInt8Cube, guarded_step, iter_origins,
                       model_digest, tiled_loss, validate_config)


class VoxelMSETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_boundary_accounting_and_manual_objective(self):
        cube = UInt8Cube(5)
        model = SpatialAutoencoder()
        actual, count = tiled_loss(model, cube, 4)
        manual_sse, manual_count = 0., 0
        with torch.no_grad():
            for xyz in iter_origins(5, 4):
                x, mask, n = cube.read(xyz, 4)
                manual_sse += ((model(x) - x).square() * mask).sum().item()
                manual_count += n
        self.assertEqual((count, manual_count), (125, 125))
        self.assertAlmostEqual(actual, manual_sse / 125, places=7)

    def test_gradients_equal_explicit_full_tile_reference(self):
        cube = UInt8Cube(4)
        model = SpatialAutoencoder()
        model.zero_grad(set_to_none=True)
        loss, n = tiled_loss(model, cube, 4, backward=True)
        grads = [p.grad.detach().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        x, mask, count = cube.read((0, 0, 0), 4)
        direct = ((model(x) - x).square() * mask).sum() / 64
        direct.backward()
        self.assertEqual((n, count), (64, 64))
        self.assertAlmostEqual(loss, direct.item(), places=7)
        for a, p in zip(grads, model.parameters()):
            torch.testing.assert_close(a, p.grad, rtol=1e-6, atol=1e-7)

    def test_guard_commits_only_improvement_and_replay(self):
        cube = UInt8Cube(8)
        a = SpatialAutoencoder()
        b = SpatialAutoencoder()
        b.load_state_dict(a.state_dict())
        before, n = tiled_loss(a, cube, 4)
        trace_a = guarded_step(a, cube, 4, 0.25)
        trace_b = guarded_step(b, cube, 4, 0.25)
        after, n_after = tiled_loss(a, cube, 4)
        self.assertEqual((n, n_after), (512, 512))
        self.assertLessEqual(after, before)
        self.assertAlmostEqual(after, trace_a["new_mse"], places=6)
        self.assertEqual(trace_a, trace_b)
        self.assertEqual(model_digest(a), model_digest(b))

    def test_raw_input_exact_length_and_coordinates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v.raw"
            a = np.arange(5**3, dtype=np.uint8).reshape(5, 5, 5)
            path.write_bytes(a.tobytes())
            cube = UInt8Cube(5, str(path))
            x, mask, n = cube.read((4, 4, 4), 4)
            self.assertEqual(n, 1)
            self.assertAlmostEqual(x[0, 0, 0, 0, 0].item(), a[4, 4, 4] / 255, places=6)
            self.assertEqual(int(mask.sum().item()), 1)
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                UInt8Cube(5, str(path))

    def test_dimensions_and_1000cube_count(self):
        validate_config(1000, 64)
        self.assertEqual(1000**3, 1_000_000_000)
        self.assertEqual(len(range(0, 1000, 64)) ** 3, 4096)
        for edge, tile in ((0, 8), (5, 3), (8, 0), (8, 258)):
            with self.assertRaises(ValueError):
                validate_config(edge, tile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
