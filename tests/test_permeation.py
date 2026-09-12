from __future__ import annotations

import math

import pytest

from jarvisx.permeation import (
    PermeationConstitution,
    constitutional_gate,
    distance_to_core,
    divergence_rms,
    permeate,
    permeation_step,
    saturation_fraction,
    solenoidal_within,
)


def test_default_constitution_matches_locked_spectral_contract() -> None:
    constitution = PermeationConstitution()
    assert constitution.alpha == pytest.approx(0.85)
    assert constitution.target_spectral_radius == pytest.approx(0.85)
    assert constitution.semantic_hbar > 0.0
    assert constitution.alpha * constitution.dt <= constitution.target_spectral_radius / 6.0
    assert constitution.relaxation_gain == pytest.approx(1.5)


def test_unstable_explicit_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="unstable permeation step"):
        PermeationConstitution(alpha=0.85, dt=0.2, target_spectral_radius=0.85)


def test_constant_mode_contracts_exactly_by_target_radius() -> None:
    field = {(0, 0, 0): (1.0,)}
    updated = permeation_step(field, (0.0,))
    assert updated[(0, 0, 0)][0] == pytest.approx(0.85)


def test_diffusion_moves_signal_to_adjacent_site_without_regression() -> None:
    field = {
        (0, 0, 0): (1.0,),
        (1, 0, 0): (0.0,),
    }
    baseline = distance_to_core(field, (0.0,))
    candidate = permeation_step(field, (0.0,))
    assert candidate[(0, 0, 0)][0] == pytest.approx(0.765)
    assert candidate[(1, 0, 0)][0] == pytest.approx(0.085)
    assert distance_to_core(candidate, (0.0,)) < baseline


def test_permeation_accepts_monotone_steps() -> None:
    field = {
        (0, 0, 0): (1.0, -1.0),
        (1, 0, 0): (0.5, -0.5),
        (2, 0, 0): (0.25, -0.25),
    }
    result = permeate(field, (0.0, 0.0), steps=4)
    assert result.accepted_steps == 4
    assert result.rejected_steps == 0
    assert len(result.objective_history) == 5
    assert all(
        later <= earlier
        for earlier, later in zip(result.objective_history, result.objective_history[1:])
    )
    assert result.theoretical_spectral_ceiling == pytest.approx(0.85)
    assert result.semantic_hbar > 0.0


def test_constitutional_gate_rejects_regression_and_unstable_radius() -> None:
    constitution = PermeationConstitution()
    assert constitutional_gate(
        1.0,
        0.9,
        measured_spectral_radius=0.85,
        constitution=constitution,
    )
    assert not constitutional_gate(
        1.0,
        1.01,
        measured_spectral_radius=0.85,
        constitution=constitution,
    )
    assert not constitutional_gate(
        1.0,
        0.9,
        measured_spectral_radius=1.0,
        constitution=constitution,
    )


def test_semantic_uncertainty_must_remain_positive() -> None:
    with pytest.raises(ValueError, match="semantic_hbar"):
        PermeationConstitution(semantic_hbar=0.0)


def test_saturation_fraction_reports_sites_near_core() -> None:
    field = {
        (0, 0, 0): (0.01,),
        (1, 0, 0): (0.1,),
        (2, 0, 0): (1.0,),
    }
    assert saturation_fraction(field, (0.0,), tolerance=0.11) == pytest.approx(2.0 / 3.0)


def test_constant_vector_field_is_solenoidal_under_no_flux_boundary() -> None:
    field = {
        (0, 0, 0): (1.0, 2.0, 3.0),
        (1, 0, 0): (1.0, 2.0, 3.0),
        (0, 1, 0): (1.0, 2.0, 3.0),
        (0, 0, 1): (1.0, 2.0, 3.0),
    }
    assert divergence_rms(field) == pytest.approx(0.0)
    assert solenoidal_within(field, tolerance=1.0e-12)


def test_divergent_vector_field_is_detected() -> None:
    field = {
        (-1, 0, 0): (-1.0, 0.0, 0.0),
        (0, 0, 0): (0.0, 0.0, 0.0),
        (1, 0, 0): (1.0, 0.0, 0.0),
    }
    assert math.isfinite(divergence_rms(field))
    assert divergence_rms(field) > 0.0
    assert not solenoidal_within(field, tolerance=0.0)
