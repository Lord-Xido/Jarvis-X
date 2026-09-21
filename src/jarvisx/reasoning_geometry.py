"""Geometric verification primitives for representation-optimized reasoning.

The purpose of this module is deliberately narrow: demonstrate, with bounded
3D mathematics, that an unchanged logical objective can become better
conditioned under a change of representation, and that locality/contraction
metrics can be verified independently.

It does not claim that every 3D embedding improves reasoning. Candidate
representations must demonstrate measurable improvement before an enclosing
Jarvis-X authority layer accepts them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

Point3 = Tuple[float, float, float]


def _positive3(values: Point3, name: str) -> None:
    if any((not math.isfinite(v)) or v <= 0.0 for v in values):
        raise ValueError(f"{name} must contain three finite positive values")


def diagonal_condition_number(curvature: Point3) -> float:
    """Condition number for a positive diagonal 3D quadratic form."""

    _positive3(curvature, "curvature")
    return max(curvature) / min(curvature)


def whitening_scale(curvature: Point3) -> Point3:
    """Return P for x = P u such that P^T diag(curvature) P = I."""

    _positive3(curvature, "curvature")
    return tuple(1.0 / math.sqrt(v) for v in curvature)  # type: ignore[return-value]


def original_to_whitened(point: Point3, curvature: Point3) -> Point3:
    """Map x to u for x = P u using the exact diagonal whitening transform."""

    _positive3(curvature, "curvature")
    return tuple(x * math.sqrt(h) for x, h in zip(point, curvature))  # type: ignore[return-value]


def quadratic_value(point: Point3, curvature: Point3) -> float:
    """Evaluate J(x) = 1/2 x^T H x for diagonal H."""

    _positive3(curvature, "curvature")
    return 0.5 * sum(h * x * x for x, h in zip(point, curvature))


def whitened_quadratic_value(point_u: Point3) -> float:
    """Evaluate the equivalent whitened objective 1/2 ||u||^2."""

    return 0.5 * sum(u * u for u in point_u)


def euclidean_distance(a: Point3, b: Point3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def locality_kernel(a: Point3, b: Point3, sigma: float) -> float:
    """Gaussian locality coupling exp(-||a-b||^2 / (2 sigma^2))."""

    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    distance = euclidean_distance(a, b)
    return math.exp(-(distance * distance) / (2.0 * sigma * sigma))


def inward_contract(point: Point3, center: Point3, lam: float) -> Point3:
    """Apply Phi_lambda(r) = c + lambda(r-c), with 0 <= lambda < 1."""

    if not math.isfinite(lam) or not 0.0 <= lam < 1.0:
        raise ValueError("lam must satisfy 0 <= lam < 1")
    return tuple(c + lam * (x - c) for x, c in zip(point, center))  # type: ignore[return-value]


def contract_n(point: Point3, center: Point3, lam: float, steps: int) -> Point3:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    result = point
    for _ in range(steps):
        result = inward_contract(result, center, lam)
    return result


@dataclass(frozen=True)
class ConditioningVerification:
    original_condition: float
    transformed_condition: float
    objective_before: float
    objective_after: float
    condition_improved: bool
    objective_preserved: bool

    @property
    def accepted(self) -> bool:
        return self.condition_improved and self.objective_preserved


def verify_diagonal_preconditioning(
    curvature: Point3,
    point: Point3,
    *,
    tolerance: float = 1e-12,
) -> ConditioningVerification:
    """Verify the ellipsoid-to-sphere representation transformation.

    For positive diagonal H and P = H^(-1/2):

        x = P u
        P^T H P = I

    so the logical quadratic objective is preserved while its condition number
    becomes 1.
    """

    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")

    original_condition = diagonal_condition_number(curvature)
    point_u = original_to_whitened(point, curvature)
    before = quadratic_value(point, curvature)
    after = whitened_quadratic_value(point_u)
    transformed_condition = 1.0

    return ConditioningVerification(
        original_condition=original_condition,
        transformed_condition=transformed_condition,
        objective_before=before,
        objective_after=after,
        condition_improved=transformed_condition <= original_condition,
        objective_preserved=math.isclose(before, after, rel_tol=tolerance, abs_tol=tolerance),
    )


@dataclass(frozen=True)
class ContractionVerification:
    distance_before: float
    distance_after: float
    locality_before: float
    locality_after: float
    distance_reduced: bool
    locality_increased: bool

    @property
    def accepted(self) -> bool:
        return self.distance_reduced and self.locality_increased


def verify_inward_locality(
    point: Point3,
    center: Point3,
    *,
    lam: float,
    steps: int,
    sigma: float,
) -> ContractionVerification:
    """Verify that inward contraction reduces distance to an attractor center.

    Locality is measured against the center with a Gaussian coupling kernel.
    """

    before_distance = euclidean_distance(point, center)
    before_locality = locality_kernel(point, center, sigma)
    contracted = contract_n(point, center, lam, steps)
    after_distance = euclidean_distance(contracted, center)
    after_locality = locality_kernel(contracted, center, sigma)

    return ContractionVerification(
        distance_before=before_distance,
        distance_after=after_distance,
        locality_before=before_locality,
        locality_after=after_locality,
        distance_reduced=after_distance <= before_distance,
        locality_increased=after_locality >= before_locality,
    )
