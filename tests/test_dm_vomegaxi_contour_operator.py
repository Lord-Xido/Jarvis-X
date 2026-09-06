from __future__ import annotations

import math

import pytest

from jarvisx.dm_vomegaxi_contour_operator import (
    ContourOperatorConfig,
    ContourSample,
    DMvOmegaXiContourOperator,
)


def test_outer_product_is_typed_tensor_coupling():
    tensor = DMvOmegaXiContourOperator.outer((2.0, -1.0), (3.0, 4.0))

    assert tensor == ((6.0, 8.0), (-3.0, -4.0))


def test_constant_closed_form_matches_documented_arithmetic_example():
    psi = DMvOmegaXiContourOperator.constant_causal_closed_form(
        phi=(2.0, -1.0),
        grad_lambda=(3.0, 4.0),
        omega=2.0,
        xi_plus=1.0,
        dm=0.5,
        length=1.0,
    )

    assert psi[0][0] == pytest.approx(2.13839, rel=1.0e-5)
    assert psi[0][1] == pytest.approx(2.85119, rel=1.0e-5)
    assert psi[1][0] == pytest.approx(-1.06919, rel=1.0e-5)
    assert psi[1][1] == pytest.approx(-1.42559, rel=1.0e-5)


def test_discrete_recurrence_converges_to_constant_closed_form():
    operator = DMvOmegaXiContourOperator()
    n = 20_000
    dv = 1.0 / n
    samples = tuple(
        ContourSample(
            v=k * dv,
            phi=(2.0, -1.0),
            grad_lambda=(3.0, 4.0),
            omega=2.0,
            xi_plus=1.0,
            dm=0.5,
        )
        for k in range(n)
    )

    numerical = operator.integrate(samples, dv)
    exact = operator.constant_causal_closed_form(
        phi=(2.0, -1.0),
        grad_lambda=(3.0, 4.0),
        omega=2.0,
        xi_plus=1.0,
        dm=0.5,
        length=1.0,
    )

    for numerical_row, exact_row in zip(numerical, exact):
        for numerical_value, exact_value in zip(numerical_row, exact_row):
            assert numerical_value == pytest.approx(exact_value, rel=1.5e-4)


def test_zero_gradient_produces_zero_local_update():
    operator = DMvOmegaXiContourOperator()
    sample = ContourSample(
        v=0.5,
        phi=(7.0, -3.0),
        grad_lambda=(0.0, 0.0, 0.0),
        omega=1.5,
        xi_plus=2.0,
        dm=1.0,
    )

    assert operator.local_derivative(sample) == (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


def test_periodic_gate_matches_at_closed_contour_endpoints():
    operator = DMvOmegaXiContourOperator(
        ContourOperatorConfig(gate_mode="periodic", period=2.0 * math.pi)
    )
    start = ContourSample(
        v=0.0,
        phi=(1.0,),
        grad_lambda=(1.0,),
        omega=2.0,
        xi_plus=1.5,
        dm=0.25,
    )
    end = ContourSample(
        v=2.0 * math.pi,
        phi=(1.0,),
        grad_lambda=(1.0,),
        omega=2.0,
        xi_plus=1.5,
        dm=0.25,
    )

    assert operator.weight(start) == pytest.approx(operator.weight(end), rel=1.0e-14)


def test_causal_gate_is_not_periodic_and_decays_for_positive_omega_xi():
    operator = DMvOmegaXiContourOperator(ContourOperatorConfig(gate_mode="causal"))
    start = ContourSample(0.0, (1.0,), (1.0,), 2.0, 1.5, 0.25)
    later = ContourSample(1.0, (1.0,), (1.0,), 2.0, 1.5, 0.25)

    assert operator.weight(later) < operator.weight(start)


def test_causal_sensitivities_match_closed_form_derivatives():
    operator = DMvOmegaXiContourOperator()
    sample = ContourSample(
        v=0.5,
        phi=(1.0,),
        grad_lambda=(1.0,),
        omega=2.0,
        xi_plus=3.0,
        dm=0.25,
    )

    sensitivity = operator.causal_sensitivities(sample)

    assert sensitivity.d_log_weight_d_dm == pytest.approx(1.0)
    assert sensitivity.d_log_weight_d_v == pytest.approx(-6.0)
    assert sensitivity.d_log_weight_d_omega == pytest.approx(-3.0)
    assert sensitivity.d_log_weight_d_xi_plus == pytest.approx(-math.log(2.0) - 1.0)


def test_operator_rejects_singular_or_non_positive_control_values():
    with pytest.raises(ValueError, match="omega must be strictly positive"):
        ContourSample(0.0, (1.0,), (1.0,), 0.0, 1.0, 0.0)

    with pytest.raises(ValueError, match="xi_plus must be strictly positive"):
        ContourSample(0.0, (1.0,), (1.0,), 1.0, 0.0, 0.0)


def test_exponent_limit_prevents_overflow():
    operator = DMvOmegaXiContourOperator(ContourOperatorConfig(exponent_limit=100.0))
    sample = ContourSample(
        v=0.0,
        phi=(1.0,),
        grad_lambda=(1.0,),
        omega=1.0,
        xi_plus=1.0,
        dm=1.0e9,
    )

    assert math.isfinite(operator.weight(sample))
    assert operator.weight(sample) == pytest.approx(math.exp(100.0))
