import math

import pytest

from jarvisx.dm_operator_transfer import (
    DMOperatorConfig,
    as_field3d,
    as_kernel3d,
    convolve3d_zero,
    dm_operator,
    elasticity_summary,
    operator_gain_bound,
    step,
    uniform_mode_is_contracting,
    uniform_mode_multiplier,
)


def singleton(value: float):
    return as_field3d([[[value]]])


def identity_kernel():
    return as_kernel3d([[[1.0]]])


def test_identity_kernel_preserves_field_before_gain():
    field = as_field3d([[[1.0, 2.0], [3.0, 4.0]]])
    assert convolve3d_zero(field, identity_kernel()) == field


def test_operator_matches_scalar_closed_form():
    config = DMOperatorConfig(
        nu=2.0,
        omega=4.0,
        omega0=2.0,
        xi=2.0,
        lambda_gain=2.0,
        theta_gain=4.0,
        dt=0.25,
    )
    # memory=(4/2)^2=4, constraint=8, scalar gain=2*4/8=1
    assert config.memory_gain == pytest.approx(4.0)
    assert config.scalar_gain == pytest.approx(1.0)
    assert dm_operator(singleton(3.0), identity_kernel(), config) == singleton(3.0)
    assert step(singleton(3.0), identity_kernel(), config) == singleton(3.75)


def test_near_singular_constraint_is_rejected():
    with pytest.raises(ValueError, match="too close to zero"):
        DMOperatorConfig(lambda_gain=0.0, theta_gain=1.0)


def test_fractional_power_uses_dimensionless_positive_ratio():
    config = DMOperatorConfig(omega=9.0, omega0=4.0, xi=0.5)
    assert config.memory_gain == pytest.approx(1.5)
    with pytest.raises(ValueError, match="positive"):
        DMOperatorConfig(omega=0.0)


def test_operator_is_deterministic():
    field = as_field3d([[[1.0, -2.0]], [[3.0, 4.0]]])
    kernel = identity_kernel()
    config = DMOperatorConfig(nu=-0.25, dt=0.5)
    assert step(field, kernel, config) == step(field, kernel, config)


def test_gain_bound_for_identity_kernel():
    config = DMOperatorConfig(nu=-0.25, omega=2.0, omega0=1.0, xi=1.0)
    assert operator_gain_bound(identity_kernel(), config) == pytest.approx(0.5)


def test_uniform_mode_contracts_for_negative_identity_feedback():
    config = DMOperatorConfig(nu=-0.5, dt=1.0)
    multiplier = uniform_mode_multiplier(identity_kernel(), config)
    assert multiplier == pytest.approx(0.5)
    assert uniform_mode_is_contracting(identity_kernel(), config)


def test_uniform_mode_rejects_false_contraction_claim():
    config = DMOperatorConfig(nu=0.5, dt=1.0)
    assert uniform_mode_multiplier(identity_kernel(), config) == pytest.approx(1.5)
    assert not uniform_mode_is_contracting(identity_kernel(), config)


def test_elasticities_match_closed_form():
    config = DMOperatorConfig(omega=4.0, omega0=2.0, xi=3.0)
    summary = elasticity_summary(config)
    assert summary["nu"] == 1.0
    assert summary["omega"] == 3.0
    assert summary["lambda_gain"] == -1.0
    assert summary["theta_gain"] == -1.0
    assert summary["xi"] == pytest.approx(math.log(2.0))
