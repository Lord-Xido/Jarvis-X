import math

import pytest

from jarvisx.silicon_embodiment import (
    SiliconLoopConfig,
    decode_level,
    encode_level,
    fixed_point_refine,
    materialize_geometry,
    run_cycle,
)


def test_materialize_geometry_maps_binary_mask() -> None:
    field = materialize_geometry([0, 1, 1, 0], off_conductivity=0.2, on_conductivity=3.0)
    assert field == (0.2, 3.0, 3.0, 0.2)


def test_materialize_geometry_rejects_non_binary_mask() -> None:
    with pytest.raises(ValueError, match="binary"):
        materialize_geometry([0, 2, 1])


def test_encode_decode_round_trip_preserves_2x2x2_volume() -> None:
    source = tuple(float(i) for i in range(8))
    encoded = encode_level(source)
    decoded = decode_level(encoded)

    assert encoded.source_side == 2
    assert encoded.latent_side == 1
    assert decoded == pytest.approx(source)


def test_encode_decode_round_trip_preserves_4x4x4_volume() -> None:
    source = tuple(math.sin(i / 5.0) for i in range(64))
    encoded = encode_level(source)
    decoded = decode_level(encoded)

    assert len(encoded.latent) == 8
    assert decoded == pytest.approx(source)


def test_fixed_point_refinement_is_contracting() -> None:
    latent = (0.0, 2.0, 4.0, 8.0)
    centre = sum(latent) / len(latent)
    before = max(abs(value - centre) for value in latent)

    result = fixed_point_refine(latent, contraction=0.5, tolerance=1e-12, max_steps=64)
    after = max(abs(value - centre) for value in result.latent)

    assert result.iterations <= 64
    assert after < before
    assert result.delta <= 1e-12


def test_fixed_point_memory_shifts_target() -> None:
    latent = (1.0, 1.0)
    omega = (2.0, -2.0)
    result = fixed_point_refine(
        latent,
        omega=omega,
        contraction=0.25,
        omega_gain=0.5,
        tolerance=1e-12,
        max_steps=64,
    )

    assert result.latent == pytest.approx((2.0, 0.0), abs=1e-9)


def test_run_cycle_is_finite_and_bounded() -> None:
    source = tuple(float(i % 11) for i in range(64))
    config = SiliconLoopConfig(
        contraction=0.5,
        omega_gain=0.1,
        omega_decay=0.8,
        tolerance=1e-10,
        max_steps=48,
    )

    result = run_cycle(source, config=config)

    assert len(result.encoded.latent) == 8
    assert len(result.reconstruction) == 64
    assert len(result.error) == 64
    assert len(result.omega_next) == 8
    assert result.iterations <= config.max_steps
    assert math.isfinite(result.fixed_point_delta)
    assert math.isfinite(result.mse)
    assert result.mse >= 0.0


def test_invalid_contraction_is_rejected() -> None:
    with pytest.raises(ValueError, match="contraction"):
        fixed_point_refine((1.0, 2.0), contraction=1.0)


def test_odd_cube_side_is_rejected() -> None:
    source = tuple(float(i) for i in range(27))
    with pytest.raises(ValueError, match="even"):
        encode_level(source)
