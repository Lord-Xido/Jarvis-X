import math

import pytest

from jarvisx.geometric_intelligence import (
    GeometricIntelligenceConfig,
    GeometricIntelligenceKernel,
    decode_field,
    encode_field,
    six_neighbor_laplacian,
)


def test_encode_decode_round_trip() -> None:
    values = (-2.0, -0.5, 0.0, 0.5, 2.0)
    decoded = decode_field(encode_field(values))
    assert decoded == pytest.approx(values, abs=1.0e-9)


def test_constant_field_has_zero_six_neighbor_laplacian() -> None:
    field = (3.25,) * (3**3)
    assert six_neighbor_laplacian(field, 3) == pytest.approx((0.0,) * (3**3))


def test_100_ps_propagation_ceiling_is_about_three_centimetres() -> None:
    config = GeometricIntelligenceConfig(local_period_ps=100.0)
    assert config.absolute_propagation_bound_m == pytest.approx(0.0299792458)


def test_subnanosecond_contract_rejects_one_nanosecond_or_more() -> None:
    with pytest.raises(ValueError, match="< 1000 ps"):
        GeometricIntelligenceConfig(local_period_ps=1_000.0)


def test_transition_is_deterministic_from_same_state_and_observation() -> None:
    config = GeometricIntelligenceConfig(side=2, local_period_ps=100.0)
    observation = tuple((index - 3.5) / 4.0 for index in range(config.node_count))
    first = GeometricIntelligenceKernel(config).step(observation)
    second = GeometricIntelligenceKernel(config).step(observation)
    assert first.current == second.current
    assert first.reconstruction == pytest.approx(second.reconstruction)
    assert first.residual == pytest.approx(second.residual)


def test_residual_memory_and_phase_advance() -> None:
    config = GeometricIntelligenceConfig(
        side=2,
        local_period_ps=100.0,
        major_frequency_hz=1.0e6,
        micro_frequency_hz=1.0e9,
    )
    kernel = GeometricIntelligenceKernel(config)
    observation = (0.5,) * config.node_count
    transition = kernel.step(observation)

    assert transition.current.step_index == 1
    assert any(abs(value) > 0.0 for value in transition.current.memory)
    assert transition.current.major_phase == pytest.approx(2.0 * math.pi * 1.0e6 * 100.0e-12)
    assert transition.current.micro_phase == pytest.approx(2.0 * math.pi * 1.0e9 * 100.0e-12)
    assert transition.reconstruction_mse >= 0.0
    assert transition.residual_rms >= 0.0


def test_uniform_observation_preserves_spatial_symmetry_after_one_step() -> None:
    config = GeometricIntelligenceConfig(side=3)
    kernel = GeometricIntelligenceKernel(config)
    transition = kernel.step((0.25,) * config.node_count)
    first = transition.current.latent[0]
    assert transition.current.latent == pytest.approx((first,) * config.node_count)
