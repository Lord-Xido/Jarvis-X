from __future__ import annotations

import math

import pytest

from jarvisx.dm_v3_prime_control import DMV3Metrics, PiHLambdaGate, VerificationPolicy


@pytest.fixture
def gate() -> PiHLambdaGate[str]:
    policy = VerificationPolicy(
        max_distortion=0.02,
        max_memory_bytes=512 * 1024 * 1024,
        max_risk=0.1,
        min_speedup=1.0,
        min_objective_improvement=0.001,
        target_speedup=1000.0,
    )
    return PiHLambdaGate(policy)


def test_accepts_verified_incremental_improvement(gate: PiHLambdaGate[str]) -> None:
    incumbent = DMV3Metrics(distortion=0.015, latency_ms=10.0, objective=0.10)
    candidate = DMV3Metrics(distortion=0.010, latency_ms=5.0, objective=0.08)

    selected, decision = gate.deploy("active", "candidate", incumbent, candidate)

    assert selected == "candidate"
    assert decision.accepted is True
    assert decision.speedup == pytest.approx(2.0)
    assert decision.speed_target_met is False
    assert decision.reasons == ()


def test_1000x_target_is_measured_separately_from_acceptance(gate: PiHLambdaGate[str]) -> None:
    incumbent = DMV3Metrics(distortion=0.015, latency_ms=1000.0, objective=0.10)
    candidate = DMV3Metrics(distortion=0.010, latency_ms=1.0, objective=0.08)

    decision = gate.evaluate(incumbent, candidate)

    assert decision.accepted is True
    assert decision.speedup == pytest.approx(1000.0)
    assert decision.speed_target_met is True


def test_rejects_unsafe_candidate_and_rolls_back(gate: PiHLambdaGate[str]) -> None:
    incumbent = DMV3Metrics(distortion=0.015, latency_ms=10.0, objective=0.10)
    candidate = DMV3Metrics(
        distortion=0.010,
        latency_ms=5.0,
        objective=0.08,
        safe=False,
    )

    selected, decision = gate.deploy("active", "candidate", incumbent, candidate)

    assert selected == "active"
    assert decision.accepted is False
    assert "candidate is not marked safe" in decision.reasons


def test_rejects_non_finite_candidate(gate: PiHLambdaGate[str]) -> None:
    incumbent = DMV3Metrics(distortion=0.015, latency_ms=10.0, objective=0.10)
    candidate = DMV3Metrics(
        distortion=math.nan,
        latency_ms=5.0,
        objective=0.08,
    )

    decision = gate.evaluate(incumbent, candidate)

    assert decision.accepted is False
    assert "candidate telemetry is non-finite" in decision.reasons


def test_policy_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        VerificationPolicy(
            max_distortion=-1.0,
            max_memory_bytes=1,
            max_risk=0.1,
        )
