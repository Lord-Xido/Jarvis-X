from __future__ import annotations

import math

import pytest

from jarvisx.permeation import PermeationConfig, PermeationField


def field(**overrides: object) -> PermeationField:
    return PermeationField(PermeationConfig(**overrides))


def test_locked_core_and_exterior_one_over_r() -> None:
    runtime = field()

    assert runtime.potential_at_radius(0.0) == 1.0
    assert runtime.potential_at_radius(0.5) == 1.0
    assert runtime.potential_at_radius(1.0) == 1.0
    assert runtime.potential_at_radius(2.0) == pytest.approx(0.5)
    assert runtime.potential_at_radius(4.0) == pytest.approx(0.25)


def test_point_evaluation_is_radially_symmetric() -> None:
    runtime = field()

    assert runtime.potential((2.0, 0.0, 0.0)) == pytest.approx(0.5)
    assert runtime.potential((0.0, -2.0, 0.0)) == pytest.approx(0.5)
    assert runtime.potential((0.0, 0.0, 2.0)) == pytest.approx(0.5)


def test_exterior_gradient_is_nonzero_and_inverse_square() -> None:
    runtime = field()

    assert runtime.exterior_radial_derivative(2.0) == pytest.approx(-0.25)
    assert runtime.gradient((2.0, 0.0, 0.0)) == pytest.approx((-0.25, 0.0, 0.0))
    assert runtime.gradient((0.5, 0.0, 0.0)) == (0.0, 0.0, 0.0)


def test_shell_boundary_uses_distributional_source() -> None:
    runtime = field()

    with pytest.raises(ValueError, match="discontinuous"):
        runtime.gradient((1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="distributional"):
        runtime.radial_laplacian(1.0)

    assert runtime.radial_laplacian(0.5) == 0.0
    assert runtime.radial_laplacian(2.0) == 0.0


def test_normalized_shell_charge_matches_boundary_value() -> None:
    charge = PermeationField.normalized_shell_charge(1.0, 1.0)

    assert charge == pytest.approx(4.0 * math.pi)
    assert charge / (4.0 * math.pi * 2.0) == pytest.approx(0.5)


def test_helmholtz_excitation_reduces_to_static_at_zero_wavenumber() -> None:
    runtime = field()

    assert runtime.helmholtz_at_radius(3.0, 0.0) == complex(1.0 / 3.0, 0.0)

    excited = runtime.helmholtz_at_radius(3.0, 2.0)
    assert abs(excited) == pytest.approx(1.0 / 3.0)
    assert runtime.helmholtz_at_radius(1.0, 2.0) == complex(1.0, 0.0)


def test_threshold_radius_converts_infinite_support_to_finite_tolerance() -> None:
    runtime = field()

    assert runtime.threshold_radius(0.5) == pytest.approx(2.0)
    assert runtime.threshold_radius(0.01) == pytest.approx(100.0)
    assert runtime.threshold_radius(2.0) == pytest.approx(1.0)


def test_relaxation_projects_perturbation_toward_target() -> None:
    runtime = field()
    target = runtime.potential_at_radius(2.0)

    first = runtime.relax_value(2.0, current=0.0, gain=0.5)
    second = runtime.relax_value(2.0, current=first, gain=0.5)

    assert target == pytest.approx(0.5)
    assert first == pytest.approx(0.25)
    assert second == pytest.approx(0.375)
    assert abs(target - second) < abs(target - first)


def test_finite_sampling_and_metrics_are_bounded() -> None:
    runtime = field(max_radius=8.0, samples=9)

    profile = runtime.sample_profile()
    metrics = runtime.metrics()

    assert len(profile) == 9
    assert profile[0].radius == 0.0
    assert profile[-1].radius == 8.0
    assert profile[-1].potential == pytest.approx(0.125)
    assert metrics.outer_value == pytest.approx(0.125)
    assert metrics.outer_to_core_ratio == pytest.approx(0.125)
    assert metrics.sampled_points == 9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"core_radius": 0.0}, "core_radius"),
        ({"core_value": 0.0}, "core_value"),
        ({"max_radius": 0.5}, "max_radius"),
        ({"samples": 1}, "samples"),
    ],
)
def test_invalid_configuration_fails_closed(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        PermeationConfig(**kwargs)


def test_invalid_relaxation_gain_fails_closed() -> None:
    runtime = field()

    with pytest.raises(ValueError, match="gain"):
        runtime.relax_value(2.0, current=0.0, gain=0.0)
    with pytest.raises(ValueError, match="gain"):
        runtime.relax_value(2.0, current=0.0, gain=1.1)
