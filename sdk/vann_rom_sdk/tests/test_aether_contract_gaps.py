import unittest

import numpy as np

from vann_rom_sdk.aether import (
    AetherConfig,
    AetherEngine,
    AetherInput,
    AetherLossWeights,
    GraphTensor,
    synthetic_aether_input,
)


class AetherKnownMathematicalContractGaps(unittest.TestCase):
    """Executable statements of mathematical guarantees not yet enforced.

    These are expected failures, not silently skipped checks. If a future
    implementation fixes one of these contracts, unittest reports an
    unexpected success so the test can be promoted to a normal invariant.
    """

    @unittest.expectedFailure
    def test_loss_weights_reject_negative_objective_coefficients(self):
        with self.assertRaises(ValueError):
            AetherLossWeights(
                reconstruction=1.0,
                perceptual=-0.2,
                semantic=0.1,
                efficiency=0.1,
                novelty=0.0,
            ).normalized()

    @unittest.expectedFailure
    def test_negative_minimum_improvement_is_rejected(self):
        with self.assertRaises(ValueError):
            AetherConfig(
                hidden_dim=16,
                latent_dim=8,
                min_improvement=-1.0,
            )

    @unittest.expectedFailure
    def test_mutated_input_is_rejected_before_authoritative_state_changes(self):
        data = synthetic_aether_input()
        data.video[0, 0, 0, 0] = np.nan
        engine = AetherEngine(AetherConfig(hidden_dim=16, latent_dim=8))
        with self.assertRaises(ValueError):
            engine.run(data)
        self.assertEqual(engine.base_digest, "uninitialized")
        self.assertEqual(engine.journal, [])

    @unittest.expectedFailure
    def test_decoder_can_represent_directed_graph_adjacency(self):
        video = np.zeros((1, 2, 2, 1), dtype=np.float32)
        audio = np.zeros((1, 2), dtype=np.float32)
        nodes = np.asarray([[0.2, 0.8], [0.7, 0.1]], dtype=np.float32)
        directed = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        context = np.zeros((1, 2), dtype=np.float32)
        data = AetherInput(video, audio, GraphTensor(nodes, directed), context)
        result = AetherEngine(AetherConfig(hidden_dim=8, latent_dim=4)).run(data)
        self.assertFalse(
            np.allclose(
                result.output.graph.adjacency,
                result.output.graph.adjacency.T,
            )
        )

    @unittest.expectedFailure
    def test_decoder_can_represent_graph_self_loops(self):
        video = np.zeros((1, 2, 2, 1), dtype=np.float32)
        audio = np.zeros((1, 2), dtype=np.float32)
        nodes = np.asarray([[0.2, 0.8]], dtype=np.float32)
        self_loop = np.asarray([[1.0]], dtype=np.float32)
        context = np.zeros((1, 2), dtype=np.float32)
        data = AetherInput(video, audio, GraphTensor(nodes, self_loop), context)
        result = AetherEngine(AetherConfig(hidden_dim=8, latent_dim=4)).run(data)
        self.assertGreater(float(result.output.graph.adjacency[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
