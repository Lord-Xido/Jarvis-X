from __future__ import annotations

import math

import pytest

from jarvisx.reasoning_geometry import (
    contract_n,
    diagonal_condition_number,
    locality_kernel,
    quadratic_value,
    verify_diagonal_preconditioning,
    verify_inward_locality,
    whitening_scale,
)


def test_ellipsoid_to_sphere_preserves_objective_and_improves_conditioning() -> None:
    curvature = (100.0, 1.0, 1.0)
    point = (0.3, -2.0, 1.5)

    result = verify_diagonal_preconditioning(curvature, point)

    assert diagonal_condition_number(curvature) == 100.0
    assert whitening_scale(curvature) == pytest.approx((0.1, 1.0, 1.0))
    assert result.original_condition == 100.0
    assert result.transformed_condition == 1.0
    assert result.objective_before == pytest.approx(result.objective_after)
    assert result.accepted is True


def test_isotropic_objective_is_preserved_without_claiming_extra_gain() -> None:
    result = verify_diagonal_preconditioning((1.0, 1.0, 1.0), (2.0, 3.0, 4.0))

    assert result.original_condition == 1.0
    assert result.transformed_condition == 1.0
    assert result.objective_preserved is True
    assert result.accepted is True


def test_inward_contraction_matches_closed_form_distance_scaling() -> None:
    point = (8.0, -4.0, 2.0)
    center = (0.0, 0.0, 0.0)
    lam = 0.5
    steps = 4

    contracted = contract_n(point, center, lam, steps)

    assert contracted == pytest.approx(tuple((lam**steps) * x for x in point))


def test_contraction_increases_locality_to_attractor_center() -> None:
    result = verify_inward_locality(
        point=(8.0, 0.0, 0.0),
        center=(0.0, 0.0, 0.0),
        lam=0.5,
        steps=3,
        sigma=2.0,
    )

    assert result.distance_after == pytest.approx(result.distance_before * (0.5**3))
    assert result.locality_after > result.locality_before
    assert result.accepted is True


def test_locality_kernel_is_bounded_and_maximal_at_zero_distance() -> None:
    same = locality_kernel((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), sigma=1.0)
    far = locality_kernel((1.0, 2.0, 3.0), (5.0, 2.0, 3.0), sigma=1.0)

    assert same == 1.0
    assert 0.0 < far < same


def test_quadratic_value_is_unchanged_in_logical_content_example() -> None:
    point = (0.25, 2.0, -3.0)
    curvature = (100.0, 1.0, 1.0)

    value = quadratic_value(point, curvature)

    assert math.isfinite(value)
    assert value == pytest.approx(0.5 * (100.0 * 0.25**2 + 2.0**2 + (-3.0) ** 2))


@pytest.mark.parametrize(
    ("curvature", "point"),
    [
        ((0.0, 1.0, 1.0), (1.0, 2.0, 3.0)),
        ((-1.0, 1.0, 1.0), (1.0, 2.0, 3.0)),
        ((math.inf, 1.0, 1.0), (1.0, 2.0, 3.0)),
    ],
)
def test_invalid_curvature_fails_closed(
    curvature: tuple[float, float, float],
    point: tuple[float, float, float],
) -> None:
    with pytest.raises(ValueError, match="curvature"):
        verify_diagonal_preconditioning(curvature, point)


def test_invalid_contraction_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="lam"):
        contract_n((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, 1)
    with pytest.raises(ValueError, match="steps"):
        contract_n((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5, -1)
    with pytest.raises(ValueError, match="sigma"):
        locality_kernel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), sigma=0.0)
