import math

import pytest

from jarvisx.unified_autoencoding import (
    DrMoagiUEA,
    LinearGaussianAutoencoder,
    MoagiCoefficients,
    OperationSet,
    Signal3D,
    SignalBounds,
    phase_difference,
    signal_squared_error,
)


def test_identity_model_satisfies_reconstruction_fixed_points():
    engine = DrMoagiUEA(
        operations=OperationSet.default(),
        coefficients=MoagiCoefficients(beta=0.0),
    )
    signal = Signal3D(440.0, 0.75, math.pi - 0.1)

    report = engine.fixed_point_report(signal)
    loss = engine.loss((signal,))

    assert report.satisfied
    assert report.maximum_rms == pytest.approx(0.0, abs=1.0e-12)
    assert loss.base_reconstruction == pytest.approx(0.0, abs=1.0e-12)
    assert loss.operation_total == pytest.approx(0.0, abs=1.0e-12)


def test_kl_term_uses_diagonal_gaussian_divergence():
    engine = DrMoagiUEA(
        operations=OperationSet.identity(),
        coefficients=MoagiCoefficients(beta=0.25),
    )
    loss = engine.loss((Signal3D(1.0, 0.5, 0.25),))

    assert loss.kl_regularization > 0.0
    assert loss.total == pytest.approx(0.25 * loss.kl_regularization)


def test_phase_metric_uses_shortest_circular_residual():
    left = Signal3D(1.0, 1.0, -math.pi + 0.01)
    right = Signal3D(1.0, 1.0, math.pi - 0.01)

    assert phase_difference(left.phase, right.phase) == pytest.approx(0.02)
    assert signal_squared_error(left, right) == pytest.approx(0.02**2)


def test_input_gradient_matches_quadratic_identity_operation_case():
    zero_decoder = ((0.0, 0.0, 0.0),) * 3
    model = LinearGaussianAutoencoder(decoder_matrix=zero_decoder)
    engine = DrMoagiUEA(
        model=model,
        operations=OperationSet.identity(),
        coefficients=MoagiCoefficients(beta=0.0),
    )

    gradient = engine.input_gradient(Signal3D(0.5, -0.25, 0.2))

    # Base plus three operation terms each contribute ||s||^2, so grad = 8s.
    assert gradient == pytest.approx((4.0, -2.0, 1.6), rel=1.0e-5, abs=1.0e-6)


def test_delta_forcing_vanishes_for_identity_operations():
    engine = DrMoagiUEA(
        operations=OperationSet.identity(),
        coefficients=MoagiCoefficients(
            beta=0.0,
            gamma=0.0,
            lambda_m=1.0,
            lambda_f=2.0,
            lambda_n=3.0,
        ),
    )

    forcing = engine.operation_forcing(Signal3D(2.0, 3.0, 0.5), "delta")

    assert forcing == pytest.approx((0.0, 0.0, 0.0))


def test_equilibrium_trace_stops_at_identity_fixed_point():
    engine = DrMoagiUEA(
        operations=OperationSet.identity(),
        coefficients=MoagiCoefficients(beta=0.0),
    )

    trace = engine.run_to_equilibrium(Signal3D(1.0, 0.5, 0.25))

    assert trace.converged
    assert trace.steps == 0
    assert len(trace.states) == 1
    assert trace.derivative_norms[0] == pytest.approx(0.0, abs=1.0e-10)


def test_signal_bounds_project_linear_coordinates_and_wrap_phase():
    bounds = SignalBounds(
        minimum_frequency=0.0,
        maximum_frequency=1000.0,
        minimum_amplitude=0.0,
        maximum_amplitude=1.0,
    )

    projected = bounds.project(Signal3D(-10.0, 2.0, 4.0 * math.pi + 0.2))

    assert projected.frequency == 0.0
    assert projected.amplitude == 1.0
    assert projected.phase == pytest.approx(0.2)
