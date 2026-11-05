"""DM-V3-PRIME verification and transactional deployment primitives.

This module implements the dependency-free control-plane portion of the
DM-V3-PRIME research baseline.  The neural codec is intentionally kept out of
this module so the deterministic Jarvis-X core does not acquire a mandatory
PyTorch dependency.

The governing rule is fail-closed: a candidate may replace the incumbent only
when every configured admissibility condition is satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar

StateT = TypeVar("StateT")


@dataclass(frozen=True)
class DMV3Metrics:
    """Measured state used by the DM-V3-PRIME verification plane."""

    distortion: float
    latency_ms: float
    objective: float
    memory_bytes: int = 0
    risk: float = 0.0
    stable: bool = True
    safe: bool = True

    def finite(self) -> bool:
        """Return ``True`` when all scalar telemetry is numerically finite."""

        return all(
            isfinite(value)
            for value in (
                self.distortion,
                self.latency_ms,
                self.objective,
                float(self.memory_bytes),
                self.risk,
            )
        )


@dataclass(frozen=True)
class VerificationPolicy:
    """Admissible-set configuration for :math:`Pi_{H,Lambda}`."""

    max_distortion: float
    max_memory_bytes: int
    max_risk: float
    min_speedup: float = 1.0
    min_objective_improvement: float = 0.0
    target_speedup: float = 1000.0

    def __post_init__(self) -> None:
        if self.max_distortion < 0.0:
            raise ValueError("max_distortion must be non-negative")
        if self.max_memory_bytes < 0:
            raise ValueError("max_memory_bytes must be non-negative")
        if self.max_risk < 0.0:
            raise ValueError("max_risk must be non-negative")
        if self.min_speedup <= 0.0:
            raise ValueError("min_speedup must be positive")
        if self.target_speedup <= 0.0:
            raise ValueError("target_speedup must be positive")
        if self.min_objective_improvement < 0.0:
            raise ValueError("min_objective_improvement must be non-negative")


@dataclass(frozen=True)
class VerificationDecision:
    """Result of evaluating a candidate against the active state."""

    accepted: bool
    speedup: float
    speed_target_met: bool
    reasons: tuple[str, ...]


class PiHLambdaGate(Generic[StateT]):
    """Fail-closed DM-V3-PRIME verification and deployment gate.

    ``PiHLambdaGate`` deliberately distinguishes *acceptance* from the
    aspirational 1000x target.  A candidate can be a valid incremental
    improvement without claiming that the 1000x benchmark has been reached.
    """

    def __init__(self, policy: VerificationPolicy) -> None:
        self.policy = policy

    @staticmethod
    def speedup(incumbent: DMV3Metrics, candidate: DMV3Metrics) -> float:
        """Return incumbent latency divided by candidate latency."""

        if candidate.latency_ms <= 0.0:
            return float("inf") if incumbent.latency_ms > 0.0 else 0.0
        return incumbent.latency_ms / candidate.latency_ms

    def evaluate(
        self,
        incumbent: DMV3Metrics,
        candidate: DMV3Metrics,
    ) -> VerificationDecision:
        """Evaluate one candidate transaction against the admissible set."""

        reasons: list[str] = []

        if not incumbent.finite():
            reasons.append("incumbent telemetry is non-finite")
        if not candidate.finite():
            reasons.append("candidate telemetry is non-finite")

        measured_speedup = self.speedup(incumbent, candidate)

        if candidate.distortion > self.policy.max_distortion:
            reasons.append("distortion exceeds configured maximum")
        if candidate.memory_bytes > self.policy.max_memory_bytes:
            reasons.append("resident memory exceeds configured maximum")
        if candidate.risk > self.policy.max_risk:
            reasons.append("risk exceeds configured maximum")
        if not candidate.safe:
            reasons.append("candidate is not marked safe")
        if not candidate.stable:
            reasons.append("candidate is not marked stable")
        if candidate.latency_ms <= 0.0:
            reasons.append("candidate latency must be positive")
        if measured_speedup < self.policy.min_speedup:
            reasons.append("candidate does not satisfy minimum speedup")

        objective_improvement = incumbent.objective - candidate.objective
        if objective_improvement < self.policy.min_objective_improvement:
            reasons.append("objective improvement is below the configured minimum")

        accepted = not reasons
        return VerificationDecision(
            accepted=accepted,
            speedup=measured_speedup,
            speed_target_met=accepted and measured_speedup >= self.policy.target_speedup,
            reasons=tuple(reasons),
        )

    def deploy(
        self,
        incumbent_state: StateT,
        candidate_state: StateT,
        incumbent_metrics: DMV3Metrics,
        candidate_metrics: DMV3Metrics,
    ) -> tuple[StateT, VerificationDecision]:
        """Atomically select candidate or roll back to the incumbent state."""

        decision = self.evaluate(incumbent_metrics, candidate_metrics)
        if decision.accepted:
            return candidate_state, decision
        return incumbent_state, decision
