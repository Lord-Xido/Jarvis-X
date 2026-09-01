"""Bounded fixed-point and closed-form resolver for the Moagi OmegaFold track."""

from __future__ import annotations

import math
from typing import Dict, List

from .certificates import canonical_state, config_digest, state_digest
from .contracts import FoldCertificate, FoldConfig, FoldProblem, FoldResult, State


def _measure(problem: FoldProblem, state: State) -> float:
    residual = float(problem.residual(state))
    if not math.isfinite(residual) or residual < 0.0:
        raise ValueError("residual must be finite and non-negative")
    return residual


def _result(
    problem: FoldProblem,
    config: FoldConfig,
    state: State,
    trace: List[float],
    *,
    method: str,
    iterations: int,
    terminal_reason: str,
) -> FoldResult:
    residual = _measure(problem, state)
    certificate = FoldCertificate(
        problem_name=problem.name,
        method=method,
        iterations=iterations,
        converged=residual <= config.tolerance,
        residual=residual,
        terminal_reason=terminal_reason,
        state_digest=state_digest(state),
        config_digest=config_digest(config),
    )
    return FoldResult(state=state, residual_trace=tuple(trace), certificate=certificate)


def resolve(problem: FoldProblem, config: FoldConfig = FoldConfig()) -> FoldResult:
    """Resolve a problem with finite work and independently verifiable termination.

    A supplied closed form is attempted first, but it is never trusted by name: its
    output must have the same dimension, contain finite values and satisfy the
    problem residual. If it does not converge, bounded fixed-point iteration begins
    from the canonical initial state. Repeated states terminate as cycles.
    """

    state = canonical_state(problem.initial_state, config.quantization_digits)
    initial_residual = _measure(problem, state)
    trace: List[float] = [initial_residual]

    if initial_residual <= config.tolerance:
        return _result(
            problem,
            config,
            state,
            trace,
            method="initial_state",
            iterations=0,
            terminal_reason="residual_satisfied",
        )

    if problem.closed_form is not None:
        closed_state = canonical_state(
            tuple(problem.closed_form(state)), config.quantization_digits
        )
        if len(closed_state) != len(state):
            raise ValueError("closed_form changed state dimensionality")
        closed_residual = _measure(problem, closed_state)
        trace.append(closed_residual)
        if closed_residual <= config.tolerance:
            return _result(
                problem,
                config,
                closed_state,
                trace,
                method="closed_form",
                iterations=1,
                terminal_reason="residual_satisfied",
            )

    seen: Dict[State, int] = {state: 0}
    for iteration in range(1, config.max_iterations + 1):
        next_state = canonical_state(
            tuple(problem.transition(state)), config.quantization_digits
        )
        if len(next_state) != len(state):
            raise ValueError("transition changed state dimensionality")

        residual = _measure(problem, next_state)
        trace.append(residual)
        if residual <= config.tolerance:
            return _result(
                problem,
                config,
                next_state,
                trace,
                method="fixed_point",
                iterations=iteration,
                terminal_reason="residual_satisfied",
            )

        if next_state in seen:
            return _result(
                problem,
                config,
                next_state,
                trace,
                method="fixed_point",
                iterations=iteration,
                terminal_reason="cycle_detected",
            )

        seen[next_state] = iteration
        state = next_state

    return _result(
        problem,
        config,
        state,
        trace,
        method="fixed_point",
        iterations=config.max_iterations,
        terminal_reason="iteration_limit",
    )
