import unittest

import numpy as np

from vann_rom_sdk.aether import (
    AetherConfig,
    AetherEngine,
    AetherPolicy,
    morton4d_decode,
    morton4d_encode,
    synthetic_aether_input,
)


class AetherEngineTests(unittest.TestCase):
    def test_morton4d_round_trip(self):
        point = (12, 345, 67, 3)
        self.assertEqual(morton4d_decode(morton4d_encode(*point)), point)

    def test_multimodal_round_trip_shapes(self):
        data = synthetic_aether_input()
        engine = AetherEngine(AetherConfig(hidden_dim=16, latent_dim=8, max_tokens=256))
        result = engine.run(data)

        self.assertEqual(result.output.video.shape, data.video.shape)
        self.assertEqual(result.output.audio.shape, data.audio.shape)
        self.assertEqual(result.output.graph.node_features.shape, data.graph.node_features.shape)
        self.assertEqual(result.output.graph.adjacency.shape, data.graph.adjacency.shape)
        self.assertEqual(result.output.context.shape, data.context.shape)
        self.assertGreater(result.field.features.shape[0], 0)
        self.assertTrue(np.isfinite(result.loss.total))

    def test_execution_is_deterministic(self):
        data = synthetic_aether_input()
        config = AetherConfig(hidden_dim=16, latent_dim=8)
        first = AetherEngine(config).run(data)
        second = AetherEngine(config).run(data)

        np.testing.assert_allclose(first.output.video, second.output.video)
        np.testing.assert_allclose(first.latent, second.latent)
        self.assertEqual(first.base_digest, second.base_digest)
        self.assertEqual(first.state_digest, second.state_digest)

    def test_adaptation_is_transactionally_nonworsening(self):
        data = synthetic_aether_input()
        engine = AetherEngine(
            AetherConfig(
                hidden_dim=16,
                latent_dim=8,
                learning_rate=0.1,
                semantic_tolerance=2.0,
            )
        )
        baseline = engine.run(data).loss.total
        result = engine.run(data, adapt=True)

        self.assertLessEqual(result.loss.total, baseline + 1e-7)
        self.assertIn(result.journal[-1]["event"], {"ADAPT_COMMIT", "ADAPT_ROLLBACK"})
        self.assertEqual(result.journal[-1]["previous_hash"], result.journal[-2]["hash"])

    def test_policy_search_is_bounded_and_nonworsening(self):
        data = synthetic_aether_input()
        engine = AetherEngine(
            AetherConfig(hidden_dim=16, latent_dim=8, semantic_tolerance=2.0),
            policy=AetherPolicy("ssm", 2, 0.25),
        )
        baseline = engine.run(data).loss.total
        result = engine.run(data, optimize=True)

        self.assertLessEqual(result.loss.total, baseline + 1e-7)
        self.assertIn(result.policy.evolution, {"ssm", "euler"})
        self.assertTrue(1 <= result.policy.recurrent_steps <= 4)
        self.assertTrue(0.0 <= result.policy.cross_modal_gain <= 1.0)

    def test_rejects_unbounded_input(self):
        data = synthetic_aether_input()
        data.video[0, 0, 0, 0] = 2.0
        with self.assertRaises(ValueError):
            type(data)(data.video, data.audio, data.graph, data.context)


if __name__ == "__main__":
    unittest.main()
