"""Canonical hashing and independent verification for OmegaFold results."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from typing import Any, Dict

from .contracts import FoldConfig, FoldProblem, FoldResult, State


def canonical_state(state: State, digits: int) -> State:
    """Return a finite, rounded state suitable for deterministic comparison."""

    if not state:
        raise ValueError("state must contain at least one value")
    values = tuple(float(value) for value in state)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("state values must be finite")
    return tuple(round(value, digits) for value in values)


def _digest(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def state_digest(state: State) -> str:
    """Hash a canonical state without relying on platform-specific binary layout."""

    return _digest({"state": list(state)})


def config_digest(config: FoldConfig) -> str:
    """Hash the execution limits that govern a resolution."""

    return _digest(asdict(config))


def certificate_payload(result: FoldResult) -> Dict[str, Any]:
    """Return the canonical JSON-native certificate payload."""

    return {
        "certificate": asdict(result.certificate),
        "state": list(result.state),
        "residual_trace": list(result.residual_trace),
    }


def verify_result(problem: FoldProblem, config: FoldConfig, result: FoldResult) -> bool:
    """Verify state, residual and execution-bound claims in a result certificate."""

    try:
        state = canonical_state(result.state, config.quantization_digits)
        measured_residual = float(problem.residual(state))
    except (TypeError, ValueError, OverflowError):
        return False

    certificate = result.certificate
    if not math.isfinite(measured_residual):
        return False
    if certificate.problem_name != problem.name:
        return False
    if certificate.iterations < 0 or certificate.iterations > config.max_iterations:
        return False
    if certificate.state_digest != state_digest(state):
        return False
    if certificate.config_digest != config_digest(config):
        return False
    if not math.isclose(
        certificate.residual,
        measured_residual,
        rel_tol=0.0,
        abs_tol=max(config.tolerance * 1.0e-6, 1.0e-15),
    ):
        return False
    if certificate.converged != (measured_residual <= config.tolerance):
        return False
    if not result.residual_trace:
        return False
    if any(not math.isfinite(value) or value < 0.0 for value in result.residual_trace):
        return False
    return True
