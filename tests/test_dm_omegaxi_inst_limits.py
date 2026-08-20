"""Arithmetic checks for the DM–OmegaXi instantaneous-limit research contract.

These tests deliberately use only the Python standard library. They verify the
mathematical distinctions documented in
``docs/research/DM_OMEGAXI_INST_LIMIT_VERIFICATION.md``.
"""

from __future__ import annotations

import cmath
import math

import pytest


def linear_interval_average(a: float, b: float, dt: float) -> float:
    """Return (1/dt) * integral_0^dt (a + b*t) dt exactly."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return a + 0.5 * b * dt


def gaussian_delta_mass(lo: float, hi: float, epsilon: float) -> float:
    """Exact mass of a normalized Gaussian delta regularization on [lo, hi]."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    scale = math.sqrt(2.0) * epsilon
    return 0.5 * (math.erf(hi / scale) - math.erf(lo / scale))


def regular_hamiltonian_phase(h: float, dt: float, hbar: float = 1.0) -> complex:
    """Scalar analogue of U(dt) = exp(-i H dt / hbar)."""

    return cmath.exp(-1j * h * dt / hbar)


def impulsive_hamiltonian_jump(k: float, hbar: float = 1.0) -> complex:
    """Scalar analogue of the finite jump exp(-i K / hbar)."""

    return cmath.exp(-1j * k / hbar)


def matmul2(a: tuple[tuple[float, float], tuple[float, float]], b):
    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ),
    )


def test_vanishing_interval_average_recovers_origin_rate() -> None:
    a = 3.25
    b = -7.0

    coarse = abs(linear_interval_average(a, b, 1e-2) - a)
    fine = abs(linear_interval_average(a, b, 1e-8) - a)

    assert fine < coarse
    assert linear_interval_average(a, b, 1e-8) == pytest.approx(a, abs=1e-7)


def test_regular_state_displacement_vanishes_with_dt() -> None:
    # For bounded regular rate F ~= 5, DeltaXi ~= F*dt -> 0.
    rate = 5.0
    coarse = abs(rate * 1e-3)
    fine = abs(rate * 1e-9)

    assert fine < coarse
    assert fine == pytest.approx(0.0, abs=1e-8)


def test_finite_group_velocity_does_not_create_finite_instantaneous_distance() -> None:
    group_velocity = 2.5e8
    coarse_distance = group_velocity * 1e-6
    fine_distance = group_velocity * 1e-15

    assert fine_distance < coarse_distance
    assert fine_distance == pytest.approx(0.0, abs=1e-6)


def test_symmetric_delta_regularization_has_unit_total_mass() -> None:
    epsilon = 1e-3
    extent = 10.0 * epsilon

    mass = gaussian_delta_mass(-extent, extent, epsilon)
    assert mass == pytest.approx(1.0, abs=1e-12)


def test_delta_on_boundary_has_half_mass_under_symmetric_regularization() -> None:
    epsilon = 1e-3
    extent = 10.0 * epsilon

    right_half_mass = gaussian_delta_mass(0.0, extent, epsilon)
    assert right_half_mass == pytest.approx(0.5, abs=1e-12)


def test_regular_hamiltonian_evolution_tends_to_identity() -> None:
    h = 11.0

    coarse = regular_hamiltonian_phase(h, 1e-3)
    fine = regular_hamiltonian_phase(h, 1e-12)

    assert abs(fine - 1.0) < abs(coarse - 1.0)
    assert fine == pytest.approx(1.0 + 0.0j, abs=1e-10)


def test_impulsive_hamiltonian_can_produce_finite_jump() -> None:
    jump = impulsive_hamiltonian_jump(math.pi / 2.0)

    # The finite jump is not forced to the identity by a separate dt -> 0 limit.
    assert abs(jump - 1.0) > 0.1
    assert abs(jump) == pytest.approx(1.0)


def test_algebraic_jump_is_not_automatically_a_projector() -> None:
    projector = ((1.0, 0.0), (0.0, 0.0))
    non_projector = ((2.0, 0.0), (0.0, 1.0))

    assert matmul2(projector, projector) == projector
    assert matmul2(non_projector, non_projector) != non_projector
