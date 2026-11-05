import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dr_moagi_vnext import (
    Config, Geometry, H8, Swarm, candidate_search, decode_packet,
    demo_data, encode_packet, inverse_pyramid, join_level,
    pyramid, run, split_level, tile_address,
)


class GeometryTests(unittest.TestCase):
    def test_orthogonal_haar(self):
        np.testing.assert_allclose(H8.T @ H8, np.eye(8), atol=1e-14)

    def test_level_round_trip(self):
        x = np.random.default_rng(8).normal(size=(2, 8, 8, 8, 4))
        c, d = split_level(x)
        np.testing.assert_allclose(join_level(c, d), x, atol=1e-14)

    def test_multilevel_roundtrip(self):
        x = np.random.default_rng(8).normal(size=(2, 16, 16, 16, 4))
        c, ds = pyramid(x, 3)
        np.testing.assert_allclose(inverse_pyramid(c, ds), x, atol=1e-14)

    def test_parseval(self):
        x = np.random.default_rng(5).normal(size=(2, 8, 8, 8, 4))
        c, ds = pyramid(x, 2)
        self.assertAlmostEqual(
            float((x * x).sum()),
            float((c * c).sum() + sum((d * d).sum() for d in ds)),
            places=11,
        )

    def test_adjoints(self):
        g = Geometry(4)
        rng = np.random.default_rng(99)
        x = rng.normal(size=(2, 4, 4, 4, 3))
        v = rng.normal(size=x.shape)
        for j in range(9):
            self.assertAlmostEqual(
                float((g.apply(j, x) * v).sum()),
                float((x * g.adjoint(j, v)).sum()),
                places=10,
            )

    def test_symbolic_tile(self):
        self.assertEqual(tile_address(13)["free_decimal_digits_per_axis"], 5_999_994)

    def test_virtual_index_cap(self):
        for i in (-1, 1_000_000):
            with self.assertRaises(ValueError):
                tile_address(i)


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(edge=8, tiles=4, levels=2, hidden=6, train_steps=5)
        self.x = demo_data(self.cfg)
        self.model = Swarm(self.cfg)

    def test_exact_packet(self):
        y, _ = decode_packet(encode_packet(self.x, self.model, 1.0, True))
        self.assertLess(float(np.max(np.abs(y - self.x))), 1e-13)

    def test_packet_independent_decode(self):
        packet = encode_packet(self.x, self.model, 0.25)
        del self.model
        decoded, _ = decode_packet(packet)
        self.assertEqual(decoded.shape, self.x.shape)
        self.assertTrue(np.isfinite(decoded).all())

    def test_progressive_error_nonincrease(self):
        mse = []
        for fraction in (0, 0.125, 0.25, 0.5, 0.75, 1):
            y, _ = decode_packet(encode_packet(self.x, self.model, fraction))
            mse.append(float(np.mean((y - self.x) ** 2)))
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(mse, mse[1:])), mse)

    def test_inward_correction_improves_coarse(self):
        c, _ = pyramid(self.x, self.cfg.levels)
        prior, _ = self.model.forward(c)
        refined, _ = self.model.encode_coarse(c)
        posterior = refined @ self.model.Wd + self.model.bd
        self.assertLessEqual(
            float(np.mean((posterior - c) ** 2)),
            float(np.mean((prior - c) ** 2)) + 1e-12,
        )

    def test_gradient_finite_difference(self):
        c, _ = pyramid(self.x, self.cfg.levels)
        grads = self.model.gradients(c)
        for name, idx in (("We", (0, 2)), ("Wd", (3, 1)), ("q", (8,))):
            p = getattr(self.model, name)
            original = p[idx]
            eps = 1e-5
            p[idx] = original + eps
            plus = self.model.mse(c)
            p[idx] = original - eps
            minus = self.model.mse(c)
            p[idx] = original
            self.assertAlmostEqual(
                (plus - minus) / (2 * eps),
                float(grads[name][idx]),
                delta=1e-7,
            )

    def test_ctr_validation_monotonic(self):
        c, _ = pyramid(self.x, self.cfg.levels)
        outcome = self.model.train(c[:3], c[3:])
        self.assertLessEqual(outcome["validation_final"], outcome["validation_initial"] + 1e-10)
        self.assertLessEqual(outcome["train_final"], outcome["train_initial"] + 1e-10)

    def test_search_real_decoding(self):
        chosen, rows = candidate_search(self.model, self.x[-1:], 0.04)
        self.assertEqual(len(rows), 12)
        self.assertIsInstance(chosen["bytes"], int)
        self.assertLessEqual(chosen["mse"], 0.04)

    def test_reject_bad_retention(self):
        with self.assertRaises(ValueError):
            encode_packet(self.x, self.model, -0.01)

    def test_digest_integrity(self):
        blob = encode_packet(self.x, self.model, 0.5)
        with np.load(io.BytesIO(blob), allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in f.files}
        arrays["Z"].flat[0] += 1
        out = io.BytesIO()
        np.savez_compressed(out, **arrays)
        with self.assertRaisesRegex(ValueError, "integrity"):
            decode_packet(out.getvalue())

    def test_memory_limit(self):
        packet = encode_packet(self.x, self.model, 0.5)
        with self.assertRaisesRegex(ValueError, "oversized"):
            decode_packet(packet, max_voxels=10)

    def test_reproducibility(self):
        m2 = Swarm(self.cfg)
        self.assertEqual(
            encode_packet(self.x, self.model, 0.25),
            encode_packet(self.x, m2, 0.25),
        )

    def test_invalid_cfg_rejected(self):
        with self.assertRaises(ValueError):
            Config(edge=8, tiles=16, levels=3).validate()
        with self.assertRaises(ValueError):
            Config(edge=64, tiles=128, levels=2).validate()

    def test_full_system(self):
        with TemporaryDirectory() as d:
            result = run(self.cfg, Path(d), target_mse=0.02)
            packet_path = Path(d) / "encoded.dm3d.npz"
            self.assertTrue(packet_path.exists())
            self.assertIsNotNone(result["independent_test"])
            y, _ = decode_packet(packet_path.read_bytes())
            self.assertEqual(y.shape, self.x.shape)
            saved = json.loads((Path(d) / "summary.json").read_text())
            self.assertAlmostEqual(saved["full_mse"], np.mean((y - self.x) ** 2))


class PersistenceTests(unittest.TestCase):
    def test_checkpoint_round_trip(self):
        cfg = Config(edge=8, tiles=4, levels=2, hidden=6, train_steps=3)
        x = demo_data(cfg)
        c, _ = pyramid(x, cfg.levels)
        model = Swarm(cfg)
        model.train(c[:3], c[-1:])
        clone = Swarm.from_checkpoint(model.checkpoint_bytes())
        self.assertEqual(model.accepted, clone.accepted)
        for k, v in model.params().items():
            np.testing.assert_array_equal(v, getattr(clone, k))
        np.testing.assert_array_equal(model.forward(c)[0], clone.forward(c)[0])

    def test_reject_malformed_checkpoint(self):
        model = Swarm(Config(edge=8, tiles=4, levels=2, hidden=6, train_steps=0))
        with np.load(io.BytesIO(model.checkpoint_bytes()), allow_pickle=False) as f:
            arrays = {k: f[k].copy() for k in f.files}
        arrays["Wd"] = arrays["Wd"][:2]
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        with self.assertRaises(ValueError):
            Swarm.from_checkpoint(stream.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
