import math

import pytest

from jarvisx.moagi_continuum import (
    ContinuumConfig,
    continuum_step,
    homogeneous_recurrence,
    regularized_pseudoinverse,
)


def diag(x: float, y: float, z: float):
    return ((x, 0.0, 0.0), (0.0, y, 0.0), (0.0, 0.0, z))


def test_single_voxel_matches_operational_equation():
    cfg = ContinuumConfig(gamma=0.2, inverse_epsilon=1e-12)
    psi = continuum_step(
        [(2.0, 4.0, 6.0)],
        [diag(2.0, 4.0, 3.0)],
        theta_core=(0.5, 0.2, -0.1),
        t=3.0,
        config=cfg,
    )
    decay = math.exp(-0.6)
    assert psi == pytest.approx((0.5 + decay, 0.2 + decay, -0.1 + 2.0 * decay))


def test_discrete_integral_scales_with_voxel_volume():
    cfg = ContinuumConfig(gamma=0.0, voxel_volume=0.5, inverse_epsilon=1e-12)
    identity = diag(1.0, 1.0, 1.0)
    psi = continuum_step(
        [(1.0, 2.0, 3.0), (3.0, 2.0, 1.0)],
        [identity, identity],
        theta_core=(1.0, 1.0, 1.0),
        t=0.0,
        config=cfg,
    )
    assert psi == pytest.approx((3.0, 3.0, 3.0))


def test_regularized_pseudoinverse_handles_singular_transform():
    singular = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    pinv = regularized_pseudoinverse(singular, epsilon=1e-6)
    assert all(math.isfinite(value) for row in pinv for value in row)
    assert pinv[0][0] == pytest.approx(1.0, rel=1e-5)
    assert pinv[1][1] == pytest.approx(0.0, abs=1e-12)
    assert pinv[2][2] == pytest.approx(0.5, rel=1e-5)


def test_large_time_converges_to_theta_core():
    cfg = ContinuumConfig(gamma=1.0)
    theta = (0.5, -0.25, 0.125)
    psi = continuum_step(
        [(100.0, 100.0, 100.0)],
        [diag(1.0, 1.0, 1.0)],
        theta_core=theta,
        t=100.0,
        config=cfg,
    )
    assert psi == pytest.approx(theta, abs=1e-12)


def test_homogeneous_inward_recurrence_is_contracting_for_identity_field():
    cfg = ContinuumConfig(gamma=math.log(2.0), inverse_epsilon=1e-12)
    history = homogeneous_recurrence(
        psi0=(8.0, 4.0, 2.0),
        lambda_field=[diag(1.0, 1.0, 1.0)] * 8,
        theta_core=(0.0, 0.0, 0.0),
        steps=4,
        dt=1.0,
        config=cfg,
    )
    assert history[-1] == pytest.approx((0.5, 0.25, 0.125), rel=1e-9)


def test_field_lengths_must_match():
    with pytest.raises(ValueError, match="equal length"):
        continuum_step(
            [(1.0, 2.0, 3.0)],
            [],
            theta_core=(0.0, 0.0, 0.0),
            t=0.0,
        )
