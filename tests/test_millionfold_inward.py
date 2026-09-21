import unittest

from jarvisx.millionfold_inward import (
    ActiveSetController,
    MillionFoldConfig,
    ResidualWork,
    RuntimeCost,
    RuntimePolicy,
    RuntimeWeights,
    accept_runtime_candidate,
    effective_work_speedup,
    region_converged,
    relative_state_delta,
    runtime_objective,
)


class MillionFoldHierarchyTests(unittest.TestCase):
    def test_canonical_spatial_reduction_is_one_million(self) -> None:
        config = MillionFoldConfig()
        self.assertEqual(config.full_sites, 1_000_000_000)
        self.assertEqual(config.mid_sites, 1_000_000)
        self.assertEqual(config.core_sites, 1_000)
        self.assertEqual(config.full_to_mid_reduction, 1_000.0)
        self.assertEqual(config.mid_to_core_reduction, 1_000.0)
        self.assertEqual(config.coarse_spatial_reduction, 1_000_000.0)

    def test_effective_speedup_accounts_for_residual_work(self) -> None:
        config = MillionFoldConfig()
        self.assertEqual(effective_work_speedup(config), 1_000_000.0)
        speedup = effective_work_speedup(
            config,
            ResidualWork(level0_sites=999_000, level1_sites=0),
        )
        self.assertEqual(speedup, 1_000.0)

    def test_active_set_uses_error_priority_and_cap(self) -> None:
        controller = ActiveSetController(MillionFoldConfig(error_threshold=0.1))
        selected = controller.select(
            {
                (0, 0, 0): 0.05,
                (1, 0, 0): 0.20,
                (2, 0, 0): 0.40,
                (3, 0, 0): 0.30,
            },
            max_active=2,
        )
        self.assertEqual(selected, ((2, 0, 0), (3, 0, 0)))

    def test_relative_delta_and_per_region_convergence(self) -> None:
        previous = (1.0, 1.0, 1.0)
        current = (1.0, 1.0, 1.000001)
        delta = relative_state_delta(previous, current)
        self.assertGreater(delta, 0.0)
        self.assertTrue(region_converged(previous, current, tolerance=1.0e-5))
        self.assertFalse(region_converged(previous, current, tolerance=1.0e-8))


class RuntimePolicyTests(unittest.TestCase):
    def test_runtime_candidate_requires_measured_improvement(self) -> None:
        baseline = RuntimeCost(
            latency_s=10.0,
            bandwidth_bytes=100.0,
            memory_bytes=100.0,
            energy_j=5.0,
            quality_loss=0.01,
        )
        candidate = RuntimeCost(
            latency_s=5.0,
            bandwidth_bytes=80.0,
            memory_bytes=90.0,
            energy_j=4.0,
            quality_loss=0.02,
        )
        weights = RuntimeWeights(latency=1.0, quality=10.0)
        self.assertLess(runtime_objective(candidate, weights), runtime_objective(baseline, weights))
        self.assertTrue(
            accept_runtime_candidate(
                baseline,
                candidate,
                max_quality_loss=0.025,
                weights=weights,
            )
        )
        self.assertFalse(
            accept_runtime_candidate(
                baseline,
                candidate,
                max_quality_loss=0.015,
                weights=weights,
            )
        )

    def test_runtime_policy_validates_precision(self) -> None:
        policy = RuntimePolicy(precision_bits=8, recursion_depth=4)
        self.assertEqual(policy.precision_bits, 8)
        with self.assertRaises(ValueError):
            RuntimePolicy(precision_bits=12)


if __name__ == "__main__":
    unittest.main()
