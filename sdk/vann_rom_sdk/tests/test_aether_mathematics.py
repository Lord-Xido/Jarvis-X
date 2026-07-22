import hashlib
import itertools
import json
import math
import unittest
from dataclasses import asdict

import numpy as np

from vann_rom_sdk.aether import (
    AetherConfig,
    AetherEngine,
    AetherInput,
    AetherLossWeights,
    AetherPolicy,
    GraphTensor,
    morton4d_decode,
    morton4d_encode,
    synthetic_aether_input,
)


class Morton4DMathematicsTests(unittest.TestCase):
    def test_axis_bit_placement_is_exact(self):
        for bit in range(16):
            value = 1 << bit
            self.assertEqual(morton4d_encode(value, 0, 0, 0), 1 << (4 * bit))
            self.assertEqual(morton4d_encode(0, value, 0, 0), 1 << (4 * bit + 1))
            self.assertEqual(morton4d_encode(0, 0, value, 0), 1 << (4 * bit + 2))
            self.assertEqual(morton4d_encode(0, 0, 0, value), 1 << (4 * bit + 3))

    def test_boundary_round_trips(self):
        values = (0, 1, 2, 255, 256, 32767, 32768, 65535)
        points = [
            (0, 0, 0, 0),
            (65535, 65535, 65535, 65535),
            (65535, 0, 65535, 0),
            (0, 65535, 0, 65535),
        ]
        points.extend((value, value, value, value) for value in values)
        for point in points:
            self.assertEqual(morton4d_decode(morton4d_encode(*point)), point)

    def test_randomized_bijection_over_5000_points(self):
        rng = np.random.default_rng(20260712)
        points = rng.integers(0, 2**16, size=(5000, 4), dtype=np.int64)
        for point_array in points:
            point = tuple(int(value) for value in point_array)
            self.assertEqual(morton4d_decode(morton4d_encode(*point)), point)

    def test_injective_on_finite_four_dimensional_lattice(self):
        codes = {
            morton4d_encode(t, x, y, z)
            for t, x, y, z in itertools.product(range(8), repeat=4)
        }
        self.assertEqual(len(codes), 8**4)

    def test_out_of_domain_values_are_rejected(self):
        for point in [(-1, 0, 0, 0), (0, 0, 0, 2**16)]:
            with self.assertRaises(ValueError):
                morton4d_encode(*point)
        for code in (-1, 2**64):
            with self.assertRaises(ValueError):
                morton4d_decode(code)


class AetherMathematicalInvariantTests(unittest.TestCase):
    @staticmethod
    def _small_input(seed: int, *, scale: int = 0) -> AetherInput:
        rng = np.random.default_rng(seed)
        time = 1 + (seed + scale) % 3
        height = 2 + (2 * seed + scale) % 5
        width = 2 + (3 * seed + scale) % 5
        video_channels = 1 + seed % 3
        audio_steps = 1 + (seed + 1) % 5
        audio_features = 2 + seed % 5
        node_count = 1 + (seed + 2) % 6
        node_features = 2 + (seed + 3) % 5
        context_steps = 1 + (seed + 4) % 4
        context_features = 2 + (seed + 5) % 6

        video = rng.random((time, height, width, video_channels), dtype=np.float32)
        audio = rng.random((audio_steps, audio_features), dtype=np.float32)
        nodes = rng.random((node_count, node_features), dtype=np.float32)
        adjacency = rng.random((node_count, node_count), dtype=np.float32)
        adjacency = 0.5 * (adjacency + adjacency.T)
        np.fill_diagonal(adjacency, 0.0)
        context = rng.random((context_steps, context_features), dtype=np.float32)
        return AetherInput(video, audio, GraphTensor(nodes, adjacency), context)

    @staticmethod
    def _assert_unit_interval(test: unittest.TestCase, array: np.ndarray) -> None:
        test.assertTrue(np.all(np.isfinite(array)))
        test.assertGreaterEqual(float(np.min(array)), 0.0)
        test.assertLessEqual(float(np.max(array)), 1.0)

    @staticmethod
    def _verify_journal_chain(journal: list[dict[str, object]]) -> None:
        previous_hash = ""
        for sequence, stored in enumerate(journal):
            assert stored["sequence"] == sequence
            assert stored["previous_hash"] == previous_hash
            record = {
                "sequence": stored["sequence"],
                "event": stored["event"],
                "payload": stored["payload"],
                "previous_hash": stored["previous_hash"],
            }
            encoded = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            expected = hashlib.sha256(encoded).hexdigest()
            assert stored["hash"] == expected
            previous_hash = expected

    def test_loss_weights_form_a_probability_simplex(self):
        weights = AetherLossWeights(6.0, 1.5, 1.5, 0.8, 0.2).normalized()
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=15)
        self.assertTrue(all(value >= 0.0 for value in weights.values()))

        scaled = AetherLossWeights(60.0, 15.0, 15.0, 8.0, 2.0).normalized()
        for name in weights:
            self.assertAlmostEqual(weights[name], scaled[name], places=15)

    def test_randomized_multimodal_shape_range_and_field_invariants(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                data = self._small_input(seed)
                engine = AetherEngine(
                    AetherConfig(
                        hidden_dim=12,
                        latent_dim=6,
                        max_tokens=96,
                        semantic_tolerance=2.0,
                        seed=seed,
                    )
                )
                result = engine.run(data)

                self.assertEqual(result.output.video.shape, data.video.shape)
                self.assertEqual(result.output.audio.shape, data.audio.shape)
                self.assertEqual(
                    result.output.graph.node_features.shape,
                    data.graph.node_features.shape,
                )
                self.assertEqual(
                    result.output.graph.adjacency.shape,
                    data.graph.adjacency.shape,
                )
                self.assertEqual(result.output.context.shape, data.context.shape)

                self._assert_unit_interval(self, result.output.video)
                self._assert_unit_interval(self, result.output.audio)
                self._assert_unit_interval(self, result.output.graph.node_features)
                self._assert_unit_interval(self, result.output.graph.adjacency)
                self._assert_unit_interval(self, result.output.context)

                self.assertTrue(np.all(np.isfinite(result.latent)))
                self.assertGreaterEqual(float(np.min(result.latent)), -1.0)
                self.assertLessEqual(float(np.max(result.latent)), 1.0)
                self.assertLessEqual(result.field.features.shape[0], 96)
                self.assertTrue(np.all(np.diff(result.field.morton_keys) >= 0))
                self.assertTrue(np.all(np.isfinite(result.field.features)))

                for coordinate, key, modality in zip(
                    result.field.coordinates,
                    result.field.morton_keys,
                    result.field.modalities,
                    strict=True,
                ):
                    decoded = morton4d_decode(int(key))
                    self.assertEqual(decoded, tuple(int(value) for value in coordinate))
                    self.assertEqual(decoded[3], int(modality))

                adjacency = result.output.graph.adjacency
                np.testing.assert_allclose(adjacency, adjacency.T, atol=1e-7)
                np.testing.assert_allclose(np.diag(adjacency), 0.0, atol=0.0)

    def test_loss_decomposition_is_exact(self):
        engine = AetherEngine(AetherConfig(hidden_dim=16, latent_dim=8))
        result = engine.run(synthetic_aether_input())
        weights = engine.loss_weights.normalized()
        recomposed = (
            weights["reconstruction"] * result.loss.reconstruction
            + weights["perceptual"] * result.loss.perceptual
            + weights["semantic"] * result.loss.semantic
            + weights["efficiency"] * result.loss.efficiency
            + weights["novelty"] * result.loss.novelty
        )
        self.assertAlmostEqual(result.loss.total, recomposed, places=12)
        self.assertGreaterEqual(result.loss.reconstruction, 0.0)
        self.assertLessEqual(result.loss.reconstruction, 5.0 + 1e-7)
        self.assertGreaterEqual(result.loss.perceptual, 0.0)
        self.assertGreaterEqual(result.loss.semantic, 0.0)
        self.assertLessEqual(result.loss.semantic, 2.0)
        self.assertGreaterEqual(result.loss.efficiency, 0.0)
        self.assertLessEqual(result.loss.efficiency, 1.0)
        self.assertGreaterEqual(result.loss.novelty, -1.0)
        self.assertTrue(math.isfinite(result.loss.total))

    def test_input_data_is_not_mutated_by_execution(self):
        data = synthetic_aether_input()
        snapshots = (
            data.video.copy(),
            data.audio.copy(),
            data.graph.node_features.copy(),
            data.graph.adjacency.copy(),
            data.context.copy(),
        )
        AetherEngine().run(data, adapt=True, optimize=True)
        np.testing.assert_array_equal(data.video, snapshots[0])
        np.testing.assert_array_equal(data.audio, snapshots[1])
        np.testing.assert_array_equal(data.graph.node_features, snapshots[2])
        np.testing.assert_array_equal(data.graph.adjacency, snapshots[3])
        np.testing.assert_array_equal(data.context, snapshots[4])

    def test_base_parameter_digest_is_immutable_across_adaptation(self):
        data = synthetic_aether_input()
        engine = AetherEngine(
            AetherConfig(
                hidden_dim=16,
                latent_dim=8,
                learning_rate=0.2,
                max_update_norm=0.05,
                semantic_tolerance=2.0,
            )
        )
        baseline = engine.run(data)
        base_digest = baseline.base_digest
        base_snapshot = {name: value.copy() for name, value in engine._base.items()}

        previous_loss = baseline.loss.total
        for _ in range(8):
            result = engine.run(data, adapt=True)
            self.assertEqual(result.base_digest, base_digest)
            self.assertLessEqual(result.loss.total, previous_loss + 1e-7)
            previous_loss = result.loss.total

        for name, expected in base_snapshot.items():
            np.testing.assert_array_equal(engine._base[name], expected)
        self._verify_journal_chain(engine.journal)

    def test_adaptation_commit_respects_norm_semantics_and_strict_improvement(self):
        data = synthetic_aether_input()
        config = AetherConfig(
            hidden_dim=16,
            latent_dim=8,
            learning_rate=1.0,
            max_update_norm=0.025,
            semantic_tolerance=2.0,
            min_improvement=1e-10,
        )
        engine = AetherEngine(config)
        baseline = engine.run(data)
        before_digest = baseline.state_digest
        result = engine.run(data, adapt=True)
        event = result.journal[-1]

        self.assertLessEqual(float(event["payload"]["update_norm"]), config.max_update_norm + 1e-8)
        if event["event"] == "ADAPT_COMMIT":
            self.assertTrue(result.adapted)
            self.assertLess(
                float(event["payload"]["candidate_loss"]),
                float(event["payload"]["baseline_loss"]) - config.min_improvement,
            )
            self.assertLessEqual(
                float(event["payload"]["semantic"]),
                config.semantic_tolerance,
            )
            self.assertNotEqual(result.state_digest, before_digest)
        else:
            self.assertFalse(result.adapted)
            self.assertEqual(result.state_digest, before_digest)

    def test_forced_adaptation_rollback_preserves_authoritative_state(self):
        data = synthetic_aether_input()
        engine = AetherEngine(
            AetherConfig(
                hidden_dim=16,
                latent_dim=8,
                semantic_tolerance=2.0,
                min_improvement=1e6,
            )
        )
        baseline = engine.run(data)
        result = engine.run(data, adapt=True)
        self.assertFalse(result.adapted)
        self.assertEqual(result.journal[-1]["event"], "ADAPT_ROLLBACK")
        self.assertEqual(result.base_digest, baseline.base_digest)
        self.assertEqual(result.state_digest, baseline.state_digest)
        self.assertAlmostEqual(result.loss.total, baseline.loss.total, places=12)

    def test_policy_search_is_monotone_and_reaches_a_fixed_point(self):
        data = synthetic_aether_input()
        engine = AetherEngine(
            AetherConfig(hidden_dim=16, latent_dim=8, semantic_tolerance=2.0),
            policy=AetherPolicy("ssm", 2, 0.25),
        )
        previous = engine.run(data).loss.total
        reached_fixed_point = False
        for _ in range(16):
            result = engine.run(data, optimize=True)
            self.assertLessEqual(result.loss.total, previous + 1e-7)
            self.assertIn(result.policy.evolution, {"ssm", "euler"})
            self.assertTrue(1 <= result.policy.recurrent_steps <= 4)
            self.assertTrue(0.0 <= result.policy.cross_modal_gain <= 1.0)
            previous = result.loss.total
            if not result.optimized:
                reached_fixed_point = True
                fixed_policy = result.policy
                fixed_loss = result.loss.total
                second = engine.run(data, optimize=True)
                self.assertFalse(second.optimized)
                self.assertEqual(second.policy, fixed_policy)
                self.assertAlmostEqual(second.loss.total, fixed_loss, places=12)
                break
        self.assertTrue(reached_fixed_point)
        self._verify_journal_chain(engine.journal)

    def test_same_seed_is_exactly_deterministic_and_different_seed_changes_base(self):
        data = synthetic_aether_input(19)
        config = AetherConfig(hidden_dim=16, latent_dim=8, seed=31)
        first = AetherEngine(config).run(data)
        second = AetherEngine(config).run(data)
        other = AetherEngine(AetherConfig(hidden_dim=16, latent_dim=8, seed=32)).run(data)

        np.testing.assert_array_equal(first.latent, second.latent)
        np.testing.assert_array_equal(first.output.video, second.output.video)
        self.assertEqual(first.base_digest, second.base_digest)
        self.assertEqual(first.state_digest, second.state_digest)
        self.assertNotEqual(first.base_digest, other.base_digest)

    def test_token_budget_is_a_hard_upper_bound(self):
        data = synthetic_aether_input()
        engine = AetherEngine(AetherConfig(hidden_dim=12, latent_dim=6, max_tokens=7))
        result = engine.run(data)
        self.assertEqual(result.field.features.shape[0], 7)
        self.assertEqual(len(result.field.coordinates), 7)
        self.assertEqual(len(result.field.morton_keys), 7)
        self.assertEqual(len(result.field.modalities), 7)

    def test_zero_and_one_extrema_remain_finite_and_bounded(self):
        for fill in (0.0, 1.0):
            with self.subTest(fill=fill):
                video = np.full((1, 2, 2, 1), fill, dtype=np.float32)
                audio = np.full((1, 2), fill, dtype=np.float32)
                nodes = np.full((1, 2), fill, dtype=np.float32)
                adjacency = np.zeros((1, 1), dtype=np.float32)
                context = np.full((1, 2), fill, dtype=np.float32)
                data = AetherInput(video, audio, GraphTensor(nodes, adjacency), context)
                result = AetherEngine(AetherConfig(hidden_dim=8, latent_dim=4)).run(data)
                self.assertTrue(math.isfinite(result.loss.total))
                self._assert_unit_interval(self, result.output.video)
                self._assert_unit_interval(self, result.output.audio)
                self._assert_unit_interval(self, result.output.graph.node_features)
                self._assert_unit_interval(self, result.output.context)

    def test_feature_signature_is_locked_after_base_sealing(self):
        engine = AetherEngine(AetherConfig(hidden_dim=12, latent_dim=6))
        engine.run(self._small_input(1))
        with self.assertRaises(ValueError):
            engine.run(self._small_input(2))

    def test_policy_and_configuration_domains_are_enforced(self):
        invalid_policies = [
            ("invalid", 2, 0.25),
            ("ssm", 0, 0.25),
            ("ssm", 5, 0.25),
            ("ssm", 2, -0.01),
            ("ssm", 2, 1.01),
        ]
        for values in invalid_policies:
            with self.subTest(values=values), self.assertRaises(ValueError):
                AetherPolicy(*values)

        invalid_configs = [
            {"hidden_dim": 3, "latent_dim": 2},
            {"hidden_dim": 8, "latent_dim": 8},
            {"hidden_dim": 8, "latent_dim": 4, "max_tokens": 3},
            {"hidden_dim": 8, "latent_dim": 4, "learning_rate": 0.0},
            {"hidden_dim": 8, "latent_dim": 4, "max_update_norm": 0.0},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AetherConfig(**kwargs)

    def test_constructor_rejects_nonfinite_and_out_of_range_tensors(self):
        base = synthetic_aether_input()
        invalid_video = base.video.copy()
        invalid_video[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            AetherInput(invalid_video, base.audio, base.graph, base.context)

        invalid_audio = base.audio.copy()
        invalid_audio[0, 0] = -1e-3
        with self.assertRaises(ValueError):
            AetherInput(base.video, invalid_audio, base.graph, base.context)

        invalid_context = base.context.copy()
        invalid_context[0, 0] = 1.001
        with self.assertRaises(ValueError):
            AetherInput(base.video, base.audio, base.graph, invalid_context)

        with self.assertRaises(ValueError):
            GraphTensor(base.graph.node_features, np.zeros((2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
