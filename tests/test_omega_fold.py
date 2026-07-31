"""Regression and adversarial tests for the bounded OmegaFold resolver."""

from dataclasses import replace

import pytest

from jarvisx.omega_fold import FoldConfig, FoldProblem, resolve, verify_result


def test_fixed_point_converges_and_verifies() -> None:
    problem = FoldProblem(
        name="scalar-contraction",
        initial_state=(0.0,),
        transition=lambda state: ((state[0] + 2.0) / 2.0,),
        residual=lambda state: abs(state[0] - 2.0),
    )
    config = FoldConfig(max_iterations=64, tolerance=1.0e-9)

    result = resolve(problem, config)

    assert result.certificate.converged
    assert result.certificate.method == "fixed_point"
    assert result.certificate.iterations <= config.max_iterations
    assert verify_result(problem, config, result)


def test_closed_form_is_verified_before_acceptance() -> None:
    problem = FoldProblem(
        name="verified-closed-form",
        initial_state=(10.0,),
        transition=lambda state: (state[0] / 2.0,),
        residual=lambda state: abs(state[0] - 3.0),
        closed_form=lambda state: (3.0,),
    )

    result = resolve(problem)

    assert result.state == (3.0,)
    assert result.certificate.method == "closed_form"
    assert result.certificate.terminal_reason == "residual_satisfied"


def test_invalid_closed_form_falls_back_to_bounded_iteration() -> None:
    problem = FoldProblem(
        name="rejected-closed-form",
        initial_state=(0.0,),
        transition=lambda state: ((state[0] + 4.0) / 2.0,),
        residual=lambda state: abs(state[0] - 4.0),
        closed_form=lambda state: (999.0,),
    )
    result = resolve(problem, FoldConfig(max_iterations=64, tolerance=1.0e-9))

    assert result.certificate.method == "fixed_point"
    assert result.certificate.converged


def test_cycle_is_detected_without_unbounded_execution() -> None:
    problem = FoldProblem(
        name="two-state-cycle",
        initial_state=(0.0,),
        transition=lambda state: (1.0 - state[0],),
        residual=lambda state: 1.0,
    )
    config = FoldConfig(max_iterations=100)

    result = resolve(problem, config)

    assert not result.certificate.converged
    assert result.certificate.terminal_reason == "cycle_detected"
    assert result.certificate.iterations == 2
    assert verify_result(problem, config, result)


def test_iteration_limit_is_explicit() -> None:
    problem = FoldProblem(
        name="bounded-drift",
        initial_state=(0.0,),
        transition=lambda state: (state[0] + 1.0,),
        residual=lambda state: 1.0,
    )
    config = FoldConfig(max_iterations=3)

    result = resolve(problem, config)

    assert result.certificate.terminal_reason == "iteration_limit"
    assert result.certificate.iterations == 3


def test_non_finite_state_fails_closed() -> None:
    problem = FoldProblem(
        name="non-finite",
        initial_state=(0.0,),
        transition=lambda state: (float("inf"),),
        residual=lambda state: 1.0,
    )

    with pytest.raises(ValueError, match="finite"):
        resolve(problem)


def test_dimension_change_is_rejected() -> None:
    problem = FoldProblem(
        name="dimension-change",
        initial_state=(0.0,),
        transition=lambda state: (0.0, 1.0),
        residual=lambda state: 1.0,
    )

    with pytest.raises(ValueError, match="dimensionality"):
        resolve(problem)


def test_certificate_tampering_is_detected() -> None:
    problem = FoldProblem(
        name="tamper-check",
        initial_state=(2.0,),
        transition=lambda state: state,
        residual=lambda state: abs(state[0] - 2.0),
    )
    config = FoldConfig()
    result = resolve(problem, config)
    forged = replace(
        result,
        certificate=replace(result.certificate, state_digest="0" * 64),
    )

    assert verify_result(problem, config, result)
    assert not verify_result(problem, config, forged)


def test_resolution_is_deterministic() -> None:
    problem = FoldProblem(
        name="deterministic",
        initial_state=(8.0,),
        transition=lambda state: (state[0] / 2.0,),
        residual=lambda state: abs(state[0] - 1.0),
    )
    config = FoldConfig(max_iterations=16)

    assert resolve(problem, config) == resolve(problem, config)
