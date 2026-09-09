from __future__ import annotations

import unittest

import torch

from sparse_bridge import (
    SparseDenseBridgeConfig,
    SparseDenseTileBridge,
    TorchVolumetricTileBackend,
    build_demo_runtime,
)
from engine import VolumetricMultimodalInterpreter


class SparseDenseTorchBridgeTests(unittest.TestCase):
    def test_torch_backend_executes_candidate_first_sparse_dense_round_trip(self) -> None:
        torch.manual_seed(7)
        runtime = build_demo_runtime()
        before = runtime.snapshot_surface()
        model = VolumetricMultimodalInterpreter(base_res=16, device="cpu")
        backend = TorchVolumetricTileBackend(model, recursive_depth=1)
        bridge = SparseDenseTileBridge(
            SparseDenseBridgeConfig(
                tile_side=16,
                tile_depth=16,
                max_tiles=4,
                max_candidate_mse=0.25,
                output_blend=0.01,
                convergence_tol=1.0e-9,
            )
        )

        result = bridge.run(runtime, backend)

        self.assertTrue(result.receipt.committed)
        self.assertEqual(result.tile_count, 1)
        self.assertEqual(model.last_metrics["recursive_steps"], 1)
        self.assertNotEqual(runtime.snapshot_surface(), before)
        self.assertEqual(result.receipt.after_state_hash, result.receipt.candidate_state_hash)
        self.assertLessEqual(result.candidate_mse, 0.25)

    def test_authority_rejection_preserves_sparse_surface(self) -> None:
        torch.manual_seed(11)
        runtime = build_demo_runtime()
        before = runtime.snapshot_surface()
        model = VolumetricMultimodalInterpreter(base_res=16, device="cpu")
        backend = TorchVolumetricTileBackend(model, recursive_depth=1)
        bridge = SparseDenseTileBridge(
            SparseDenseBridgeConfig(
                tile_side=16,
                tile_depth=16,
                max_tiles=4,
                max_candidate_mse=1.0,
                output_blend=0.01,
            )
        )

        result = bridge.run(
            runtime,
            backend,
            authority_validator=lambda _candidate, _metrics: False,
        )

        self.assertFalse(result.receipt.committed)
        self.assertEqual(runtime.snapshot_surface(), before)
        self.assertEqual(result.receipt.after_state_hash, result.receipt.parent_state_hash)


if __name__ == "__main__":
    unittest.main()
