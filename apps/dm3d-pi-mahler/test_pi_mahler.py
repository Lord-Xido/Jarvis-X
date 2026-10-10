"""Correctness regressions for DM3D pi torus Mahler reduction."""
import math
import unittest
import numpy as np
from pi_mahler import (SIDE, VIRTUAL_VOXELS, Laurent3, Term, test_polynomials,
                       jensen_2d, direct_3d, resolution_feedback,
                       pi_phase_integer, smyth_reference)


class PiMahlerTests(unittest.TestCase):
    def test_septillion_virtual_shape(self):
        self.assertEqual(SIDE, 10**24)
        self.assertEqual(VIRTUAL_VOXELS, 10**72)

    def test_bigint_phase_far_corner(self):
        q = pi_phase_integer(SIDE - 1, 55)
        self.assertEqual(q["coordinate"], str(SIDE - 1))
        self.assertTrue(q["fraction_of_pi"].startswith("0.999999999999999999999999"))
        self.assertTrue(q["phase_imag"].startswith("-6.283185307179586476925286"))

    def test_phase_bounds(self):
        for k in [-1, SIDE]:
            with self.assertRaises(ValueError):
                pi_phase_integer(k)

    def test_smyth_formula(self):
        self.assertAlmostEqual(smyth_reference(), .4262783988175058, places=13)

    def test_2d_jensen_matches_3d_in_limit(self):
        for p in test_polynomials().values():
            reduced = jensen_2d(p, 64)["value"]
            dense = direct_3d(p, 64)["value"]
            self.assertLess(abs(dense - reduced), .002)

    def test_analytic_dominant_coefficient(self):
        p = test_polynomials()["dominant"]
        self.assertLess(abs(jensen_2d(p, 128)["value"] - math.log(1.85)), 2e-9)

    def test_smyth_reduction_improves_equal_resolution(self):
        p = test_polynomials()["smyth"]
        exact = smyth_reference()
        direct = direct_3d(p, 32)
        reduced = jensen_2d(p, 32)
        self.assertEqual(direct["evaluations"], 32768)
        self.assertEqual(reduced["evaluations"], 1024)
        self.assertLess(abs(reduced["value"] - exact), abs(direct["value"] - exact))

    def test_linear_z_reduction_with_cross_terms(self):
        p = test_polynomials()["cross_terms"]
        self.assertLess(abs(direct_3d(p, 64)["value"] - jensen_2d(p, 64)["value"]), .002)

    def test_non_unit_torus_radii(self):
        p = test_polynomials()["radii"]
        self.assertEqual(p.radii, (1.1, .8, .9))
        self.assertLess(abs(direct_3d(p, 48)["value"] - jensen_2d(p, 48)["value"]), .003)

    def test_refinement_non_oracle(self):
        result = resolution_feedback(test_polynomials()["smyth"], 1e-6)
        self.assertTrue(result["converged_heuristically"])
        self.assertEqual(result["history"][-1]["n"], 64)
        self.assertLess(abs(result["estimate"] - smyth_reference()), 5e-8)

    def test_refinement_bad_tol(self):
        with self.assertRaises(ValueError):
            resolution_feedback(test_polynomials()["smyth"], -1)

    def test_malformed_non_linear_z_rejected(self):
        p = Laurent3((Term(1, (0, 0, 0)), Term(1, (0, 0, 2))))
        with self.assertRaises(ValueError):
            jensen_2d(p, 16)

    def test_zero_radius_rejected(self):
        with self.assertRaises(ValueError):
            Laurent3((Term(1, (0, 0, 0)),), (1, 1, 0))

    def test_observed_delta_not_certified(self):
        r = resolution_feedback(test_polynomials()["cross_terms"], 1e-12,
                                levels=(8, 16, 32))
        self.assertFalse(r["converged_heuristically"])

    def test_empty_levels_rejected(self):
        with self.assertRaises(ValueError):
            resolution_feedback(test_polynomials()["smyth"], levels=())

    def test_nonfinite_coefficient_rejected(self):
        with self.assertRaises(ValueError):
            Laurent3((Term(float("nan"), (0, 0, 0)),))


if __name__ == "__main__":
    unittest.main(verbosity=2)
