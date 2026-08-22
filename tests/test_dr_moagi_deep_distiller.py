from __future__ import annotations

import math

import pytest

from jarvisx.dm_vo_xi_operational import DeepDistiller as CompatibilityDeepDistiller
from jarvisx.dr_moagi_deep_distiller import (
    DeepDistiller,
    DeepDistillerConfig,
    DeepDistillerTheta,
)


def _field() -> dict[tuple[int, int, int], float]:
    return {
        (0, 0, 0): 1.0,
        (1, 0, 0): 0.5,
        (0, 1, 0): -0.25,
    }


def _config(**overrides: object) -> DeepDistillerConfig:
    values: dict[str, object] = {
        "logical_side": 8,
        "max_active_cells": 16,
        "max_latent_cells": 16,
        "max_iterations": 8,
        "learning_rate": 0.05,
        "omega_gain": 0.25,
        "rho": 0.5,
    }
    values.update(overrides)
    return DeepDistillerConfig(**values)  # type: ignore[arg-type]


def test_deep_distiller_commits_state_memory_and_theta_together() -> None:
    engine = DeepDistiller(_config(), theta=DeepDistillerTheta(0.8, 0.8))
    before = engine.load(_field())
    theta_before = engine.theta

    report = engine.step()

    assert report.committed is True
    assert report.iteration == 1
    assert report.residual_rms > 0.0
    assert engine.snapshot() != before
    assert engine.omega_snapshot()
    assert engine.theta != theta_before
    assert engine.theta.encoder_gain > theta_before.encoder_gain
    assert engine.theta.decoder_gain > theta_before.decoder_gain
    assert engine.status()["ip_locked"] is True
    assert engine.status()["journal_valid"] is True


def test_gate_rejection_is_atomic_for_state_omega_and_theta() -> None:
    def gate(candidate: object) -> bool:
        # Initial admission has zero latent cells; runtime proposals do not.
        return getattr(candidate, "latent_cells") == 0

    engine = DeepDistiller(_config(), theta=DeepDistillerTheta(0.8, 0.8), gate=gate)
    original_state = engine.load(_field())
    original_omega = engine.omega_snapshot()
    original_theta = engine.theta

    report = engine.step()

    assert report.committed is False
    assert report.gate_passed is False
    assert report.rejection_reason == "external Pi_Lambda policy rejected candidate"
    assert report.iteration == 0
    assert engine.snapshot() == original_state
    assert engine.omega_snapshot() == original_omega
    assert engine.theta == original_theta
    assert engine.status()["journal_valid"] is True


def test_exact_codec_fixed_point_stops_on_zero_residual() -> None:
    engine = DeepDistiller(
        _config(residual_tolerance=0.0),
        theta=DeepDistillerTheta(encoder_gain=1.0, decoder_gain=1.0),
    )
    original = engine.load(_field())

    reports = engine.run()

    assert len(reports) == 1
    assert reports[0].committed is True
    assert reports[0].converged is True
    assert reports[0].residual_rms == 0.0
    assert reports[0].grad_encoder == 0.0
    assert reports[0].grad_decoder == 0.0
    assert engine.snapshot() == original


def test_encoder_enforces_sparse_latent_budget_without_dense_expansion() -> None:
    engine = DeepDistiller(_config(max_latent_cells=1))
    engine.load(_field())

    latent = engine.encode(engine.snapshot(), engine.theta)
    report = engine.step()

    assert len(latent) == 1
    assert report.latent_cells == 1
    assert report.active_cells <= engine.config.max_active_cells
    assert engine.status()["materialization"] == "sparse-active-support-only"


def test_parameter_update_is_residual_gradient_and_bounded() -> None:
    engine = DeepDistiller(
        _config(learning_rate=100.0, theta_max_delta=0.01),
        theta=DeepDistillerTheta(0.8, 0.8),
    )
    engine.load(_field())

    report = engine.step()

    assert report.committed is True
    assert math.isclose(engine.theta.encoder_gain, 0.81, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(engine.theta.decoder_gain, 0.81, rel_tol=0.0, abs_tol=1.0e-12)


def test_load_rejects_initial_state_when_constitutional_gate_rejects() -> None:
    engine = DeepDistiller(_config(), gate=lambda candidate: False)

    with pytest.raises(ValueError, match="initial state rejected by Pi_Lambda"):
        engine.load(_field())


def test_run_stops_immediately_on_gate_reject() -> None:
    def gate(candidate: object) -> bool:
        return getattr(candidate, "latent_cells") == 0

    engine = DeepDistiller(_config(max_iterations=10), gate=gate)
    engine.load(_field())

    reports = engine.run()

    assert len(reports) == 1
    assert reports[0].committed is False


def test_configuration_prevents_latent_budget_exceeding_state_budget() -> None:
    with pytest.raises(ValueError, match="max_latent_cells cannot exceed max_active_cells"):
        DeepDistillerConfig(max_active_cells=2, max_latent_cells=3)


def test_compatibility_surface_resolves_to_product_runtime() -> None:
    assert CompatibilityDeepDistiller is DeepDistiller
