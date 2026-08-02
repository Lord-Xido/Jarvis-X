from __future__ import annotations

import math
import unittest

from reference_vm import Config, DMVOmegaXiVM, Status


class ReferenceVMTests(unittest.TestCase):
    def test_nominal_transaction_commits(self) -> None:
        vm = DMVOmegaXiVM()
        result = vm.process([0.10, -0.20, 0.30, -0.40])
        self.assertEqual(result.status, Status.COMMITTED)
        self.assertTrue(vm.omega_committed)
        self.assertEqual(result.receipt.outcome, "committed")
        self.assertLessEqual(
            result.receipt.reconstruction_distance,
            vm.config.reconstruction_tolerance,
        )

    def test_rejected_transaction_preserves_committed_state(self) -> None:
        vm = DMVOmegaXiVM()
        first = vm.process([0.05, 0.10])
        self.assertEqual(first.status, Status.COMMITTED)
        committed = list(vm.omega_committed)

        rejected = vm.process([math.nan])
        self.assertEqual(rejected.status, Status.REJECTED)
        self.assertEqual(vm.omega_committed, committed)
        self.assertEqual(rejected.receipt.outcome, "rejected")

    def test_budget_is_bounded(self) -> None:
        vm = DMVOmegaXiVM(Config(max_elements=2))
        result = vm.process([0.0, 0.1, 0.2])
        self.assertEqual(result.status, Status.REJECTED)
        self.assertEqual(result.receipt.status, int(Status.BUDGET_FAILED))

    def test_authorization_is_required(self) -> None:
        vm = DMVOmegaXiVM()
        result = vm.process([0.1], authorized=False)
        self.assertEqual(result.status, Status.REJECTED)
        self.assertEqual(result.receipt.status, int(Status.AUTH_FAILED))

    def test_retry_is_bounded(self) -> None:
        vm = DMVOmegaXiVM(Config(free_energy_limit=1e-12, max_retries=2))
        result = vm.process([0.9, -0.9])
        self.assertEqual(result.status, Status.REJECTED)
        self.assertEqual(result.receipt.retries, 2)
        self.assertEqual(result.receipt.status, int(Status.DRIFT))


if __name__ == "__main__":
    unittest.main()
