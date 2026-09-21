from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import math
import statistics
from typing import Any, Callable, Iterable, Mapping, Sequence

from .adaptive_orchestrator import SecurityState


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class PromotionStage(str, Enum):
    REJECTED = "REJECTED"
    SHADOW = "SHADOW"
    CANARY = "CANARY"
    PRODUCTION = "PRODUCTION"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection
    weight: float = 1.0
    minimum: float | None = None
    maximum: float | None = None
    max_regression: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name is required")
        if self.weight < 0.0:
            raise ValueError("metric weight must be non-negative")
        if self.max_regression < 0.0:
            raise ValueError("max_regression must be non-negative")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("metric minimum cannot exceed maximum")


@dataclass(frozen=True)
class OptimizationCandidate:
    candidate_id: str
    parent_id: str
    description: str
    risk_score: float
    change_scope: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.parent_id:
            raise ValueError("candidate_id and parent_id are required")
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be in [0, 1]")


@dataclass(frozen=True)
class BenchmarkResult:
    subject_id: str
    samples: Mapping[str, tuple[float, ...]]
    means: Mapping[str, float]
    stdevs: Mapping[str, float]
    digest: str


@dataclass(frozen=True)
class FrontierSnapshot:
    benchmark_id: str
    benchmark_version: str
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.benchmark_id or not self.benchmark_version:
            raise ValueError("frontier benchmark identity/version is required")


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    allowed: bool
    stage: PromotionStage
    reason: str
    utility_delta: float
    conservative_utility_delta: float
    metric_deltas: Mapping[str, float]
    frontier_deltas: Mapping[str, float]


Evaluator = Callable[[str, int], Mapping[str, float]]
Invariant = Callable[[OptimizationCandidate], bool]


class BenchmarkHarness:
    """Repeatable evaluation harness with deterministic evidence receipts."""

    def __init__(self, metric_specs: Sequence[MetricSpec], repetitions: int = 5) -> None:
        if repetitions < 2:
            raise ValueError("repetitions must be at least 2")
        self.metric_specs = tuple(metric_specs)
        self.repetitions = repetitions
        names = [spec.name for spec in self.metric_specs]
        if not names or len(set(names)) != len(names):
            raise ValueError("metric specifications must be non-empty and unique")

    def evaluate(self, subject_id: str, evaluator: Evaluator) -> BenchmarkResult:
        if not subject_id:
            raise ValueError("subject_id is required")
        samples: dict[str, list[float]] = {spec.name: [] for spec in self.metric_specs}
        for repetition in range(self.repetitions):
            result = evaluator(subject_id, repetition)
            missing = set(samples) - set(result)
            if missing:
                raise ValueError(f"evaluator omitted metrics: {sorted(missing)}")
            for name in samples:
                value = float(result[name])
                if not math.isfinite(value):
                    raise ValueError(f"metric {name!r} must be finite")
                samples[name].append(value)

        frozen_samples = {name: tuple(values) for name, values in samples.items()}
        means = {name: statistics.fmean(values) for name, values in frozen_samples.items()}
        stdevs = {name: statistics.stdev(values) for name, values in frozen_samples.items()}
        body = {
            "subject_id": subject_id,
            "samples": frozen_samples,
            "means": means,
            "stdevs": stdevs,
        }
        digest = sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return BenchmarkResult(subject_id, frozen_samples, means, stdevs, digest)


class EvidencePromotionGate:
    """Champion/challenger gate: promote only measurable, bounded improvements."""

    def __init__(
        self,
        metric_specs: Sequence[MetricSpec],
        *,
        min_utility_gain: float = 0.01,
        max_risk_score: float = 0.35,
        confidence_z: float = 1.96,
        mutation_security_threshold: float = 0.75,
        invariants: Iterable[Invariant] = (),
    ) -> None:
        self.metric_specs = tuple(metric_specs)
        self.min_utility_gain = min_utility_gain
        self.max_risk_score = max_risk_score
        self.confidence_z = confidence_z
        self.mutation_security_threshold = mutation_security_threshold
        self.invariants = tuple(invariants)
        if min_utility_gain < 0.0:
            raise ValueError("min_utility_gain must be non-negative")
        if not 0.0 <= max_risk_score <= 1.0:
            raise ValueError("max_risk_score must be in [0, 1]")
        if confidence_z < 0.0:
            raise ValueError("confidence_z must be non-negative")

    def decide(
        self,
        *,
        candidate: OptimizationCandidate,
        champion: BenchmarkResult,
        challenger: BenchmarkResult,
        security: SecurityState,
        frontier: FrontierSnapshot | None = None,
    ) -> PromotionDecision:
        if candidate.parent_id != champion.subject_id:
            return self._reject(candidate, "candidate parent is not the active champion")
        if challenger.subject_id != candidate.candidate_id:
            return self._reject(candidate, "challenger benchmark identity mismatch")
        if candidate.risk_score > self.max_risk_score:
            return self._reject(candidate, "candidate exceeds risk budget")
        if not security.permits_mutation(self.mutation_security_threshold):
            return self._reject(candidate, security.reason or "security gate denied mutation")
        for invariant in self.invariants:
            if not invariant(candidate):
                return self._reject(candidate, "immutable invariant rejected candidate")

        metric_deltas: dict[str, float] = {}
        frontier_deltas: dict[str, float] = {}
        utility_delta = 0.0
        conservative_utility_delta = 0.0
        total_weight = sum(spec.weight for spec in self.metric_specs)
        if total_weight <= 0.0:
            return self._reject(candidate, "metric weights sum to zero")

        for spec in self.metric_specs:
            baseline = champion.means[spec.name]
            current = challenger.means[spec.name]
            if spec.minimum is not None and current < spec.minimum:
                return self._reject(candidate, f"{spec.name} violated minimum")
            if spec.maximum is not None and current > spec.maximum:
                return self._reject(candidate, f"{spec.name} violated maximum")

            directional_delta = (
                current - baseline
                if spec.direction is MetricDirection.MAXIMIZE
                else baseline - current
            )
            metric_deltas[spec.name] = directional_delta
            if directional_delta < -spec.max_regression:
                return self._reject(candidate, f"{spec.name} regression exceeds tolerance")

            scale = max(abs(baseline), 1.0)
            normalized = directional_delta / scale
            utility_delta += spec.weight * normalized

            n_champion = len(champion.samples[spec.name])
            n_challenger = len(challenger.samples[spec.name])
            se = math.sqrt(
                (champion.stdevs[spec.name] ** 2) / n_champion
                + (challenger.stdevs[spec.name] ** 2) / n_challenger
            )
            conservative = directional_delta - self.confidence_z * se
            conservative_utility_delta += spec.weight * (conservative / scale)

            if frontier is not None and spec.name in frontier.metrics:
                target = float(frontier.metrics[spec.name])
                frontier_deltas[spec.name] = (
                    current - target
                    if spec.direction is MetricDirection.MAXIMIZE
                    else target - current
                )

        utility_delta /= total_weight
        conservative_utility_delta /= total_weight
        if conservative_utility_delta < self.min_utility_gain:
            return PromotionDecision(
                candidate.candidate_id,
                False,
                PromotionStage.REJECTED,
                "evidence does not clear conservative utility threshold",
                utility_delta,
                conservative_utility_delta,
                metric_deltas,
                frontier_deltas,
            )
        return PromotionDecision(
            candidate.candidate_id,
            True,
            PromotionStage.SHADOW,
            "candidate admitted to shadow stage",
            utility_delta,
            conservative_utility_delta,
            metric_deltas,
            frontier_deltas,
        )

    @staticmethod
    def _reject(candidate: OptimizationCandidate, reason: str) -> PromotionDecision:
        return PromotionDecision(
            candidate.candidate_id,
            False,
            PromotionStage.REJECTED,
            reason,
            0.0,
            0.0,
            {},
            {},
        )


@dataclass(frozen=True)
class CanaryObservation:
    success: bool
    value_delta: float
    integrity_ok: bool = True


class ReleaseController:
    """Shadow -> canary -> production controller with automatic rollback semantics."""

    def __init__(
        self,
        *,
        canary_min_observations: int = 5,
        canary_min_success_rate: float = 0.99,
        canary_min_mean_value_delta: float = 0.0,
    ) -> None:
        if canary_min_observations < 1:
            raise ValueError("canary_min_observations must be positive")
        if not 0.0 <= canary_min_success_rate <= 1.0:
            raise ValueError("canary_min_success_rate must be in [0, 1]")
        self.canary_min_observations = canary_min_observations
        self.canary_min_success_rate = canary_min_success_rate
        self.canary_min_mean_value_delta = canary_min_mean_value_delta

    def advance(
        self,
        decision: PromotionDecision,
        observations: Sequence[CanaryObservation] = (),
    ) -> PromotionDecision:
        if not decision.allowed:
            return decision
        if decision.stage is PromotionStage.SHADOW:
            return PromotionDecision(
                decision.candidate_id,
                True,
                PromotionStage.CANARY,
                "shadow evidence accepted; candidate admitted to canary",
                decision.utility_delta,
                decision.conservative_utility_delta,
                decision.metric_deltas,
                decision.frontier_deltas,
            )
        if decision.stage is not PromotionStage.CANARY:
            return decision
        if len(observations) < self.canary_min_observations:
            return PromotionDecision(
                decision.candidate_id,
                False,
                PromotionStage.REJECTED,
                "insufficient canary observations; rollback",
                decision.utility_delta,
                decision.conservative_utility_delta,
                decision.metric_deltas,
                decision.frontier_deltas,
            )
        if not all(obs.integrity_ok for obs in observations):
            return self._rollback(decision, "canary integrity failure")
        success_rate = sum(obs.success for obs in observations) / len(observations)
        mean_value_delta = statistics.fmean(obs.value_delta for obs in observations)
        if success_rate < self.canary_min_success_rate:
            return self._rollback(decision, "canary success rate below threshold")
        if mean_value_delta < self.canary_min_mean_value_delta:
            return self._rollback(decision, "canary value regressed")
        return PromotionDecision(
            decision.candidate_id,
            True,
            PromotionStage.PRODUCTION,
            "canary passed; candidate promoted to production",
            decision.utility_delta,
            decision.conservative_utility_delta,
            decision.metric_deltas,
            decision.frontier_deltas,
        )

    @staticmethod
    def _rollback(decision: PromotionDecision, reason: str) -> PromotionDecision:
        return PromotionDecision(
            decision.candidate_id,
            False,
            PromotionStage.REJECTED,
            f"{reason}; rollback",
            decision.utility_delta,
            decision.conservative_utility_delta,
            decision.metric_deltas,
            decision.frontier_deltas,
        )


class ContinualOptimizer:
    """Finite-step recursive improvement controller; no candidate self-promotes."""

    def __init__(
        self,
        *,
        harness: BenchmarkHarness,
        gate: EvidencePromotionGate,
        release: ReleaseController,
    ) -> None:
        self.harness = harness
        self.gate = gate
        self.release = release
        self.generation = 0
        self.champion_id: str | None = None
        self.champion_result: BenchmarkResult | None = None

    def bootstrap(self, champion_id: str, evaluator: Evaluator) -> BenchmarkResult:
        result = self.harness.evaluate(champion_id, evaluator)
        self.champion_id = champion_id
        self.champion_result = result
        return result

    def challenge(
        self,
        candidate: OptimizationCandidate,
        *,
        evaluator: Evaluator,
        security: SecurityState,
        frontier: FrontierSnapshot | None = None,
    ) -> PromotionDecision:
        if self.champion_result is None or self.champion_id is None:
            raise RuntimeError("optimizer must be bootstrapped")
        challenger = self.harness.evaluate(candidate.candidate_id, evaluator)
        return self.gate.decide(
            candidate=candidate,
            champion=self.champion_result,
            challenger=challenger,
            security=security,
            frontier=frontier,
        )

    def commit_production(
        self,
        candidate: OptimizationCandidate,
        decision: PromotionDecision,
        *,
        evaluator: Evaluator,
    ) -> BenchmarkResult:
        if decision.candidate_id != candidate.candidate_id:
            raise ValueError("decision/candidate identity mismatch")
        if not decision.allowed or decision.stage is not PromotionStage.PRODUCTION:
            raise RuntimeError("only production-stage candidates may become champion")
        result = self.harness.evaluate(candidate.candidate_id, evaluator)
        self.champion_id = candidate.candidate_id
        self.champion_result = result
        self.generation += 1
        return result
