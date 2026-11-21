"""Deterministic isolated numerical regression checks; unittest discovery compatible."""
import unittest
import numpy as np
from engine import Parameters, laplacian, rotate, inverse_rotate, step, conserved_charge, direct_limit, finite_value_memory_fft, morton3


class TestPeriodicReference(unittest.TestCase):
    def setUp(self):
        self.p = Parameters(side=8)
        rng = np.random.default_rng(20261023)
        self.v = rng.random((8,8,8))
        self.m = rng.random((8,8,8))
        self.e = np.zeros((8,8,8))

    def test_laplacian_zero_sum(self):
        self.assertAlmostEqual(float(laplacian(self.v).sum()), 0.0, places=11)

    def test_constant_laplacian(self):
        self.assertTrue(np.allclose(laplacian(np.ones_like(self.v)), 0))

    def test_rotation_order_four(self):
        a = self.v.copy()
        for _ in range(4): a = rotate(a)
        np.testing.assert_array_equal(a, self.v)

    def test_rotation_coordinates(self):
        arr = np.zeros_like(self.v);arr[1,2,3]=1
        self.assertEqual(rotate(arr)[3,2,7], 1)

    def test_rotation_inverse(self):
        np.testing.assert_array_equal(inverse_rotate(rotate(self.v)), self.v)

    def test_rotation_commutes_with_laplacian(self):
        np.testing.assert_allclose(rotate(laplacian(self.v)), laplacian(rotate(self.v)), atol=1e-13)

    def test_charge_conserved(self):
        first = conserved_charge(self.v,self.m,self.p)
        v,m,e=self.v,self.m,self.e
        for _ in range(150): v,m,e=step(v,m,e,self.p)
        self.assertAlmostEqual(first,conserved_charge(v,m,self.p),places=10)

    def test_energy_mean_monotonic(self):
        v,m,e=self.v,self.m,self.e
        previous=0
        for _ in range(30):
            v,m,e=step(v,m,e,self.p)
            self.assertGreaterEqual(float(e.mean())+1e-13,previous)
            previous=float(e.mean())

    def test_direct_limit(self):
        lv,lm=direct_limit(self.v,self.m,self.p)
        self.assertAlmostEqual(float(lv.mean()),float(lm.mean()),places=14)
        self.assertAlmostEqual(conserved_charge(lv,lm,self.p),conserved_charge(self.v,self.m,self.p),places=10)

    def test_fft_matches_steps(self):
        for n in (0,1,2,3,4,9,35):
            v,m,e=self.v,self.m,self.e
            for _ in range(n): v,m,e=step(v,m,e,self.p)
            fv,fm=finite_value_memory_fft(self.v,self.m,n,self.p)
            np.testing.assert_allclose(v,fv,rtol=0,atol=4e-13)
            np.testing.assert_allclose(m,fm,rtol=0,atol=4e-13)

    def test_constant_stays_stationary(self):
        field=np.full_like(self.v,0.33)
        v,m,e=step(field,field,np.zeros_like(field),self.p)
        np.testing.assert_allclose(v,field,rtol=0,atol=1e-14)
        np.testing.assert_allclose(m,field,rtol=0,atol=1e-14)
        np.testing.assert_allclose(e,0,rtol=0,atol=1e-14)

    def test_morton_bijection(self):
        addresses={morton3(x,y,z,3) for x in range(8) for y in range(8) for z in range(8)}
        self.assertEqual(addresses,set(range(512)))

    def test_invalid_parameters(self):
        for p in (Parameters(side=7),Parameters(side=128),Parameters(diffusion=0.3)):
            with self.assertRaises(ValueError):p.validate()

    def test_reject_bad_inputs(self):
        with self.assertRaises(ValueError):step(np.full((8,8,8),np.nan),self.m,self.e,self.p)
        with self.assertRaises(ValueError):finite_value_memory_fft(self.v,self.m,-1,self.p)
        with self.assertRaises(ValueError):morton3(-1,0,0)


if __name__=='__main__': unittest.main()
