from __future__ import annotations

import numpy as np
import pytest

from jarvisx.neural_feedback_runtime import (
    HashingFeatureEncoder,
    JsonWebFeed,
    NeuralFeedbackRuntime,
    RuntimeConfig,
    SyntheticWebFeed,
    WebObservation,
)


def test_hashing_feature_encoder_is_deterministic() -> None:
    observation = WebObservation(
        claim="claim",
        content="same content",
        source="https://example.org/a",
        source_group="example",
        authority=0.9,
    )
    encoder = HashingFeatureEncoder(32)
    first = encoder.encode(observation)
    second = encoder.encode(observation)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)


def test_runtime_accepts_independent_evidence_and_promotes() -> None:
    config = RuntimeConfig(
        input_dim=32,
        latent_dim=8,
        batch_size=4,
        epochs_per_cycle=20,
        simulations_per_observation=1,
        verification_threshold=0.40,
        replay_capacity=64,
        seed=3,
    )
    runtime = NeuralFeedbackRuntime(config)
    feed = SyntheticWebFeed(seed=3)
    reports = [runtime.process(feed.observations(cycle)) for cycle in range(1, 5)]
    assert all(report.accepted_claims == report.claims for report in reports)
    assert reports[-1].replay_size >= config.batch_size
    assert runtime.production_generation >= 1


def test_high_counterevidence_is_rejected() -> None:
    config = RuntimeConfig(
        input_dim=32,
        latent_dim=8,
        batch_size=4,
        verification_threshold=0.70,
        simulations_per_observation=0,
    )
    runtime = NeuralFeedbackRuntime(config)
    observations = [
        WebObservation(
            claim="contested",
            content="shared statement",
            source=f"https://source-{index}.example/item",
            source_group=f"source-{index}",
            authority=0.8,
            counterevidence=0.95,
        )
        for index in range(3)
    ]
    report = runtime.process(observations)
    assert report.accepted_claims == 0
    assert report.replay_size == 0


def test_checkpoint_round_trip(tmp_path) -> None:
    runtime = NeuralFeedbackRuntime(RuntimeConfig(input_dim=32, latent_dim=8, batch_size=4, seed=5))
    runtime.process(SyntheticWebFeed(seed=5).observations(1))
    checkpoint = runtime.save_checkpoint(tmp_path / "runtime.npz")
    restored = NeuralFeedbackRuntime.load_checkpoint(checkpoint)
    assert restored.cycle == runtime.cycle
    assert restored.production_generation == runtime.production_generation
    assert np.allclose(restored.production.w_e, runtime.production.w_e)


def test_feed_url_blocks_local_targets() -> None:
    with pytest.raises(ValueError, match="non-public"):
        JsonWebFeed._validate_public_url("http://127.0.0.1/feed.json")
