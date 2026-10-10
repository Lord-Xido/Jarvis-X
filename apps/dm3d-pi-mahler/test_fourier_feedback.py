"""Regression coverage of independent Fourier correction and adaptive order."""
import math
import unittest
import numpy as np
from fourier_feedback import emulate, features, modes, fit_on_pilot
from pi_mahler import test_polynomials, smyth_reference


class FourierTests(unittest.TestCase):
    def test_features_constant_integral(self):
        n = 40
        u = (np.arange(n) + .5) * math.pi / n
        samples = np.stack(np.meshgrid(u, u, indexing="ij"), -1).reshape(-1, 2)
        X = features(samples, 3)
        np.testing.assert_allclose(X[:, 0].mean(), 1, atol=1e-12)
        np.testing.assert_allclose(X[:, 1:].mean(axis=0), 0, atol=2e-14)

    def test_adaptive_order_and_holdout(self):
        result = emulate(test_polynomials()["smyth"], 42)
        self.assertEqual(result["Fourier_order"], 3)
        self.assertTrue(result["independent_holdout"])
        self.assertEqual(result["evaluations_plain"], result["evaluations_feedback"])
        self.assertLess(result["se_feedback"], result["se_plain"])
        self.assertLess(abs(result["estimate_feedback"] - smyth_reference()),
                        4 * result["se_feedback"])

    def test_invalid_sample_budget(self):
        with self.assertRaises(ValueError):
            emulate(test_polynomials()["smyth"], pilot_n=4)

    def test_invalid_order(self):
        with self.assertRaises(ValueError):
            modes(-2)

    def test_deterministic_replay(self):
        a = emulate(test_polynomials()["smyth"], seed=14)
        b = emulate(test_polynomials()["smyth"], seed=14)
        self.assertEqual(a["estimate_feedback"], b["estimate_feedback"])

    def test_pilot_validation_separation(self):
        rng = np.random.default_rng(11)
        pilot = rng.random((256, 2)) * math.pi
        val = rng.random((128, 2)) * math.pi
        order, coef, record = fit_on_pilot(test_polynomials()["smyth"], pilot, val=val)
        self.assertIn(order, [0, 1, 2, 3])
        self.assertEqual(coef.size, 1 + 2 * len(modes(order)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
