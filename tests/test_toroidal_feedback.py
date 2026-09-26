from __future__ import annotations

import math

import numpy as np
import pytest

from jarvisx.toroidal_feedback import (
    TorusConfig,
    TorusState,
    ToroidalFeedbackEngine,
    make_state,
)


def test_default_embedding_matches_torus_equations() -> None:
    cfg = TorusConfig()
    state = TorusState(np.array([1.4]), np.array([4.8]), 1.0)
    engine = ToroidalFeedbackEngine(state, cfg)
    xyz = engine.embed()[0]

    r = cfg.minor_radius
    rho = cfg.major_radius + r * math.cos(4.8)
    expected = np.array(
        [
            rho * math.cos(1.4),
            r * math.sin(4.8),
            rho * math.sin(1.4),
        ]
    )
    np.testing.assert_allclose(xyz, expected)


def test_complex_angle_encoding_preserves_periodicity() -> None:
    u = np.array([0.0, 2.0 * math.pi])
    v = np.array([math.pi / 2.0, math.pi / 2.0 + 2.0 * math.pi])
    encoded = ToroidalFeedbackEngine.encode_angles(u, v)
    np.testing.assert_allclose(encoded[0], encoded[1], atol=1.0e-12)


def test_history_matrix_has_batch_by_four_window_shape() -> None:
    engine = ToroidalFeedbackEngine(make_state(batch=64), TorusConfig(window=32))
    assert engine.history_matrix().shape == (64, 128)


def test_spectral_energy_is_bounded_and_winding_is_positive() -> None:
    engine = ToroidalFeedbackEngine(make_state(batch=8), TorusConfig(window=16, top_k=3))
    model = engine.spectral_model()
    assert 0.0 <= model.energy <= 1.0
    assert 1 <= model.winding <= 8


def test_fft_detects_fourfold_temporal_winding() -> None:
    window = 32
    batch = 8
    engine = ToroidalFeedbackEngine(make_state(batch=batch), TorusConfig(window=window))
    t = np.arange(window)
    q = 4
    offsets = np.linspace(0.0, 1.0, batch)[:, np.newaxis]
    u_history = 2.0 * math.pi * q * t[np.newaxis, :] / window + offsets
    v_history = np.zeros_like(u_history)
    engine._history = np.stack(
        (
            np.cos(u_history),
            np.sin(u_history),
            np.cos(v_history),
            np.sin(v_history),
        ),
        axis=-1,
    )
    assert engine.spectral_model().winding == q


def test_sigma_permeation_is_exact_closed_form() -> None:
    cfg = TorusConfig(lambda_permeation=0.9997, dt=1.0)
    engine = ToroidalFeedbackEngine(make_state(batch=4), cfg)
    result = engine.run(20)
    assert result.final_state.sigma == pytest.approx(0.9997**20, rel=1.0e-13)


def test_core_distance_contracts_monotonically() -> None:
    engine = ToroidalFeedbackEngine(make_state(batch=4))
    result = engine.run(12)
    distances = [item.core_distance for item in result.telemetry]
    assert all(later < earlier for earlier, later in zip(distances, distances[1:]))


def test_trajectory_is_three_dimensional_and_finite() -> None:
    engine = ToroidalFeedbackEngine(make_state(batch=6))
    result = engine.run(5)
    assert result.trajectory.shape == (6, 6, 3)
    assert np.isfinite(result.trajectory).all()


def test_angles_remain_on_torus_domain_after_many_steps() -> None:
    engine = ToroidalFeedbackEngine(make_state(batch=8))
    result = engine.run(64)
    assert np.all((0.0 <= result.final_state.u) & (result.final_state.u < 2.0 * math.pi))
    assert np.all((0.0 <= result.final_state.v) & (result.final_state.v < 2.0 * math.pi))
