from __future__ import annotations

import math

import pytest

from jarvisx.cognitive_matrix_permeation import (
    CognitiveMatrixConfig,
    CognitiveMatrixError,
    CognitiveMatrixPermeation,
)


def test_virtual_septillion_domain_is_metadata_only() -> None:
    engine = CognitiveMatrixPermeation(
        CognitiveMatrixConfig(virtual_axis=10**24, particle_count=32)
    )
    assert engine.virtual_sites == 10**72
    assert engine.virtual_bits == 10**72
    assert len(engine.phase_positions(0.0)) == 32


def test_reference_encoder_decoder_round_trips_sampled_positions() -> None:
    engine = CognitiveMatrixPermeation(CognitiveMatrixConfig(particle_count=64))
    assert engine.reconstruction_mse() < 1e-24


def test_three_phase_geometry_is_bounded_and_distinct() -> None:
    engine = CognitiveMatrixPermeation(CognitiveMatrixConfig(particle_count=27))
    input_positions = engine.phase_positions(0.0)
    latent_positions = engine.phase_positions(1.0)
    output_positions = engine.phase_positions(2.0)
    assert input_positions != latent_positions
    assert latent_positions != output_positions
    assert output_positions != input_positions
    assert all(all(math.isfinite(value) for value in point) for point in output_positions)


def test_permeation_field_flux_entropy_and_receipt_are_finite() -> None:
    engine = CognitiveMatrixPermeation(CognitiveMatrixConfig(particle_count=64))
    receipt = None
    for _ in range(8):
        receipt = engine.step(0.75)
    assert receipt is not None
    assert receipt.finite
    assert receipt.accepted
    assert receipt.max_field_value > 0.0
    assert receipt.spatial_entropy_bits > 0.0
    assert math.isfinite(receipt.permeation_flux)
    assert 0.0 <= receipt.permeation_flux_percent <= 100.0
    assert receipt.claim_status == "bounded_stochastic_projection"


def test_seeded_execution_is_replayable() -> None:
    config = CognitiveMatrixConfig(particle_count=48, seed=123)
    left = CognitiveMatrixPermeation(config)
    right = CognitiveMatrixPermeation(config)
    for phase in (0.25, 0.75, 1.25, 1.75):
        left_receipt = left.step(phase)
        right_receipt = right.step(phase)
        assert left_receipt == right_receipt
        assert left.phase_positions(phase) == right.phase_positions(phase)


def test_entropy_is_zero_for_a_single_occupied_bin() -> None:
    engine = CognitiveMatrixPermeation(CognitiveMatrixConfig(particle_count=8))
    entropy = engine.spatial_entropy_bits([(0.0, 0.0, 0.0)] * 8)
    assert entropy == 0.0


def test_invalid_resource_and_numeric_inputs_fail_closed() -> None:
    with pytest.raises(CognitiveMatrixError):
        CognitiveMatrixConfig(particle_count=0)
    with pytest.raises(CognitiveMatrixError):
        CognitiveMatrixConfig(particle_count=10, max_particles=4)
    with pytest.raises(CognitiveMatrixError):
        CognitiveMatrixConfig(radial_cells=4)
    with pytest.raises(CognitiveMatrixError):
        CognitiveMatrixConfig(dt=0.0)
    engine = CognitiveMatrixPermeation(CognitiveMatrixConfig(particle_count=8))
    with pytest.raises(CognitiveMatrixError):
        engine.step(-0.1)
    with pytest.raises(CognitiveMatrixError):
        engine.step(2.1)
    with pytest.raises(CognitiveMatrixError):
        engine.step(0.5, dt=float("nan"))
