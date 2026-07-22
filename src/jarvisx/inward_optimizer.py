from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ObjectiveWeights:
    """Weights for a lower-is-better operational objective."""

    task_loss: float = 1.0
    latency_ms: float = 0.01
    memory_mb: float = 0.001
    rollback_rate: float = 1.0

    def __post_init__(self) -> None:
        values = (self.task_loss, self.latency_ms, self.memory_mb, self.rollback_rate)
        if any(value < 0.0 or not math.isfinite(value) for value in values):
            raise ValueError("objective weights must be finite and non-negative")
        if sum(values) == 0.0:
            raise ValueError("at least one objective weight must be positive")


@dataclass(frozen=True)
class Telemetry:
    """Measured performance and correctness signals for one mechanics state."""

    task_loss: float
    latency_ms: float
    memory_mb: float
    rollback_rate: float = 0.0
    semantic_distance: float = 0.0
    gradient_norm: float = 0.0

    def is_finite(self) -> bool:
        return all(
            math.isfinite(value)
            for value in (
                self.task_loss,
                self.latency_ms,
                self.memory_mb,
                self.rollback_rate,
                self.semantic_distance,
                self.gradient_norm,
            )
        )

    def score(self, weights: ObjectiveWeights) -> float:
        if not self.is_finite():
            return math.inf
        return (
            weights.task_loss * self.task_loss
            + weights.latency_ms * self.latency_ms
            + weights.memory_mb * self.memory_mb
            + weights.rollback_rate * self.rollback_rate
        )


@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float = 1e-3
    regularization: float = 1e-4
    moe_sparsity: float = 0.50
    top_k: int = 2


@dataclass(frozen=True)
class Architecture:
    layers: int = 4
    experts: int = 4
    kernel_size: int = 3
    latent_dim: int = 64


@dataclass(frozen=True)
class UpdateRule:
    name: str = "adam-like"
    momentum: float = 0.90
    beta2: float = 0.999
    epsilon: float = 1e-8
    meta_learning_rate: float = 1e-3


@dataclass(frozen=True)
class MechanicsState:
    """Versioned mechanics state. Instances are immutable after creation."""

    hyper: HyperParameters = HyperParameters()
    architecture: Architecture = Architecture()
    rule: UpdateRule = UpdateRule()
    version: int = 0


@dataclass(frozen=True)
class SafetyPolicy:
    max_gradient_norm: float = 100.0
    max_semantic_distance: float = 1e-6
    max_memory_mb: float = 16384.0
    max_rollback_rate: float = 0.05
    max_latency_ratio: float = 1.10
    max_task_loss_regression_ratio: float = 0.01
    min_score_improvement: float = 1e-9
    min_learning_rate: float = 1e-8
    max_learning_rate: float = 1.0
    max_layers: int = 128
    max_experts: int = 256
    max_latent_dim: int = 65536


@dataclass(frozen=True)
class Candidate:
    level: int
    transformation: str
    state: MechanicsState


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    telemetry: Telemetry
    score: float
    gate: GateDecision


@dataclass(frozen=True)
class JournalEntry:
    active_version: int
    candidate_level: int
    transformation: str
    baseline_score: float
    candidate_score: float
    accepted: bool
    committed: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class OptimizationResult:
    previous_state: MechanicsState
    active_state: MechanicsState
    baseline: Telemetry
    evaluations: Tuple[CandidateEvaluation, ...]
    committed: bool
    transformation: Optional[str]


Evaluator = Callable[[MechanicsState], Telemetry]


class InwardOptimizer:
    """Bounded Level-1/2/3 optimiser around a Level-0 execution evaluator.

    The evaluator is the execution engine boundary: it runs a candidate in a
    shadow context and returns measured telemetry. A candidate can only replace
    the active mechanics state after all safety gates pass and its weighted
    performance score improves.
    """

    def __init__(
        self,
        active_state: Optional[MechanicsState] = None,
        weights: Optional[ObjectiveWeights] = None,
        policy: Optional[SafetyPolicy] = None,
        max_candidates: int = 32,
    ) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self.active_state = active_state or MechanicsState()
        self.weights = weights or ObjectiveWeights()
        self.policy = policy or SafetyPolicy()
        self.max_candidates = max_candidates
        self.journal: List[JournalEntry] = []
        self._rollback_stack: List[MechanicsState] = []

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _level1_candidates(self, state: MechanicsState) -> List[Candidate]:
        hyper = state.hyper
        architecture = state.architecture
        proposals = [
            Candidate(
                1,
                "learning_rate_x0.5",
                replace(state, hyper=replace(hyper, learning_rate=hyper.learning_rate * 0.5)),
            ),
            Candidate(
                1,
                "learning_rate_x1.5",
                replace(state, hyper=replace(hyper, learning_rate=hyper.learning_rate * 1.5)),
            ),
            Candidate(
                1,
                "regularization_x0.5",
                replace(state, hyper=replace(hyper, regularization=hyper.regularization * 0.5)),
            ),
            Candidate(
                1,
                "regularization_x1.5",
                replace(state, hyper=replace(hyper, regularization=hyper.regularization * 1.5)),
            ),
            Candidate(
                1,
                "moe_sparsity_minus_0.05",
                replace(
                    state,
                    hyper=replace(
                        hyper,
                        moe_sparsity=self._clamp(hyper.moe_sparsity - 0.05, 0.0, 0.99),
                    ),
                ),
            ),
            Candidate(
                1,
                "moe_sparsity_plus_0.05",
                replace(
                    state,
                    hyper=replace(
                        hyper,
                        moe_sparsity=self._clamp(hyper.moe_sparsity + 0.05, 0.0, 0.99),
                    ),
                ),
            ),
            Candidate(
                1,
                "top_k_minus_1",
                replace(state, hyper=replace(hyper, top_k=max(1, hyper.top_k - 1))),
            ),
            Candidate(
                1,
                "top_k_plus_1",
                replace(
                    state,
                    hyper=replace(hyper, top_k=min(architecture.experts, hyper.top_k + 1)),
                ),
            ),
        ]
        return proposals

    def _level2_candidates(self, state: MechanicsState) -> List[Candidate]:
        architecture = state.architecture
        hyper = state.hyper
        kernels = (1, 3, 5, 7)
        kernel_index = kernels.index(architecture.kernel_size) if architecture.kernel_size in kernels else 1
        lower_kernel = kernels[max(0, kernel_index - 1)]
        upper_kernel = kernels[min(len(kernels) - 1, kernel_index + 1)]

        proposals = [
            Candidate(
                2,
                "layers_minus_1",
                replace(state, architecture=replace(architecture, layers=max(1, architecture.layers - 1))),
            ),
            Candidate(
                2,
                "layers_plus_1",
                replace(state, architecture=replace(architecture, layers=architecture.layers + 1)),
            ),
            Candidate(
                2,
                "experts_minus_1",
                replace(
                    state,
                    architecture=replace(architecture, experts=max(1, architecture.experts - 1)),
                    hyper=replace(hyper, top_k=min(hyper.top_k, max(1, architecture.experts - 1))),
                ),
            ),
            Candidate(
                2,
                "experts_plus_1",
                replace(state, architecture=replace(architecture, experts=architecture.experts + 1)),
            ),
            Candidate(
                2,
                "kernel_smaller",
                replace(state, architecture=replace(architecture, kernel_size=lower_kernel)),
            ),
            Candidate(
                2,
                "kernel_larger",
                replace(state, architecture=replace(architecture, kernel_size=upper_kernel)),
            ),
            Candidate(
                2,
                "latent_dim_minus_16",
                replace(
                    state,
                    architecture=replace(architecture, latent_dim=max(8, architecture.latent_dim - 16)),
                ),
            ),
            Candidate(
                2,
                "latent_dim_plus_16",
                replace(state, architecture=replace(architecture, latent_dim=architecture.latent_dim + 16)),
            ),
        ]
        return proposals

    def _level3_candidates(self, state: MechanicsState) -> List[Candidate]:
        rule = state.rule
        proposals = [
            Candidate(
                3,
                "momentum_minus_0.05",
                replace(state, rule=replace(rule, momentum=self._clamp(rule.momentum - 0.05, 0.0, 0.999))),
            ),
            Candidate(
                3,
                "momentum_plus_0.05",
                replace(state, rule=replace(rule, momentum=self._clamp(rule.momentum + 0.05, 0.0, 0.999))),
            ),
            Candidate(
                3,
                "beta2_minus_0.0005",
                replace(state, rule=replace(rule, beta2=self._clamp(rule.beta2 - 0.0005, 0.0, 0.999999))),
            ),
            Candidate(
                3,
                "beta2_plus_0.0005",
                replace(state, rule=replace(rule, beta2=self._clamp(rule.beta2 + 0.0005, 0.0, 0.999999))),
            ),
            Candidate(
                3,
                "epsilon_x0.1",
                replace(state, rule=replace(rule, epsilon=max(1e-16, rule.epsilon * 0.1))),
            ),
            Candidate(
                3,
                "epsilon_x10",
                replace(state, rule=replace(rule, epsilon=rule.epsilon * 10.0)),
            ),
            Candidate(
                3,
                "meta_learning_rate_x0.5",
                replace(
                    state,
                    rule=replace(rule, meta_learning_rate=rule.meta_learning_rate * 0.5),
                ),
            ),
            Candidate(
                3,
                "meta_learning_rate_x1.5",
                replace(
                    state,
                    rule=replace(rule, meta_learning_rate=rule.meta_learning_rate * 1.5),
                ),
            ),
        ]
        return proposals

    def propose(self) -> Tuple[Candidate, ...]:
        candidates = (
            self._level1_candidates(self.active_state)
            + self._level2_candidates(self.active_state)
            + self._level3_candidates(self.active_state)
        )

        unique: List[Candidate] = []
        seen = set()
        for candidate in candidates:
            key = candidate.state
            if key == self.active_state or key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
            if len(unique) >= self.max_candidates:
                break
        return tuple(unique)

    def _gate(
        self,
        baseline: Telemetry,
        candidate: Telemetry,
        candidate_state: MechanicsState,
    ) -> GateDecision:
        reasons: List[str] = []
        policy = self.policy
        hyper = candidate_state.hyper
        architecture = candidate_state.architecture
        rule = candidate_state.rule

        if not baseline.is_finite():
            reasons.append("baseline telemetry is not finite")
        if not candidate.is_finite():
            reasons.append("candidate telemetry is not finite")
        if candidate.task_loss < 0.0 or candidate.latency_ms < 0.0 or candidate.memory_mb < 0.0:
            reasons.append("candidate telemetry contains negative physical metrics")
        if candidate.gradient_norm > policy.max_gradient_norm:
            reasons.append("gradient norm exceeds policy")
        if candidate.semantic_distance > policy.max_semantic_distance:
            reasons.append("semantic distance exceeds policy")
        if candidate.memory_mb > policy.max_memory_mb:
            reasons.append("memory budget exceeded")
        if candidate.rollback_rate > policy.max_rollback_rate:
            reasons.append("rollback rate exceeds policy")

        if baseline.latency_ms > 0.0 and candidate.latency_ms > baseline.latency_ms * policy.max_latency_ratio:
            reasons.append("latency regression exceeds policy")
        if baseline.task_loss >= 0.0:
            task_loss_limit = baseline.task_loss * (1.0 + policy.max_task_loss_regression_ratio)
            task_loss_limit = max(task_loss_limit, baseline.task_loss + 1e-12)
            if candidate.task_loss > task_loss_limit:
                reasons.append("task-loss regression exceeds policy")

        if not (policy.min_learning_rate <= hyper.learning_rate <= policy.max_learning_rate):
            reasons.append("learning rate outside policy")
        if hyper.regularization < 0.0 or not (0.0 <= hyper.moe_sparsity < 1.0):
            reasons.append("invalid hyperparameter bounds")
        if hyper.top_k < 1 or hyper.top_k > architecture.experts:
            reasons.append("top_k must be within expert count")
        if architecture.layers < 1 or architecture.layers > policy.max_layers:
            reasons.append("layer count outside policy")
        if architecture.experts < 1 or architecture.experts > policy.max_experts:
            reasons.append("expert count outside policy")
        if architecture.kernel_size not in (1, 3, 5, 7):
            reasons.append("kernel size is not an allowed transformation")
        if architecture.latent_dim < 8 or architecture.latent_dim > policy.max_latent_dim:
            reasons.append("latent dimension outside policy")
        if not (0.0 <= rule.momentum < 1.0 and 0.0 <= rule.beta2 < 1.0):
            reasons.append("optimizer moments outside policy")
        if rule.epsilon <= 0.0 or rule.meta_learning_rate <= 0.0:
            reasons.append("optimizer rule contains non-positive scale")

        baseline_score = baseline.score(self.weights)
        candidate_score = candidate.score(self.weights)
        if candidate_score > baseline_score - policy.min_score_improvement:
            reasons.append("weighted performance did not improve")

        return GateDecision(accepted=not reasons, reasons=tuple(reasons))

    def optimize_once(self, evaluator: Evaluator) -> OptimizationResult:
        previous_state = self.active_state
        baseline = evaluator(previous_state)
        baseline_score = baseline.score(self.weights)
        evaluations: List[CandidateEvaluation] = []
        best: Optional[CandidateEvaluation] = None

        for candidate in self.propose():
            telemetry = evaluator(candidate.state)
            score = telemetry.score(self.weights)
            gate = self._gate(baseline, telemetry, candidate.state)
            evaluation = CandidateEvaluation(candidate, telemetry, score, gate)
            evaluations.append(evaluation)

            self.journal.append(
                JournalEntry(
                    active_version=previous_state.version,
                    candidate_level=candidate.level,
                    transformation=candidate.transformation,
                    baseline_score=baseline_score,
                    candidate_score=score,
                    accepted=gate.accepted,
                    committed=False,
                    reasons=gate.reasons,
                )
            )

            if gate.accepted and (best is None or score < best.score):
                best = evaluation

        if best is None:
            return OptimizationResult(
                previous_state=previous_state,
                active_state=self.active_state,
                baseline=baseline,
                evaluations=tuple(evaluations),
                committed=False,
                transformation=None,
            )

        self._rollback_stack.append(previous_state)
        self.active_state = replace(best.candidate.state, version=previous_state.version + 1)
        self.journal.append(
            JournalEntry(
                active_version=self.active_state.version,
                candidate_level=best.candidate.level,
                transformation=best.candidate.transformation,
                baseline_score=baseline_score,
                candidate_score=best.score,
                accepted=True,
                committed=True,
                reasons=(),
            )
        )
        return OptimizationResult(
            previous_state=previous_state,
            active_state=self.active_state,
            baseline=baseline,
            evaluations=tuple(evaluations),
            committed=True,
            transformation=best.candidate.transformation,
        )

    def optimize(self, evaluator: Evaluator, cycles: int) -> Tuple[OptimizationResult, ...]:
        if cycles < 0:
            raise ValueError("cycles must be non-negative")
        return tuple(self.optimize_once(evaluator) for _ in range(cycles))

    def rollback(self) -> bool:
        if not self._rollback_stack:
            return False
        current = self.active_state
        restored = self._rollback_stack.pop()
        self.active_state = restored
        self.journal.append(
            JournalEntry(
                active_version=current.version,
                candidate_level=-1,
                transformation="rollback",
                baseline_score=math.nan,
                candidate_score=math.nan,
                accepted=True,
                committed=True,
                reasons=("restored previous immutable mechanics state",),
            )
        )
        return True


def run_synthetic_demo(cycles: int = 4) -> Sequence[OptimizationResult]:
    """Run a deterministic demonstration without an ML framework dependency."""

    target_learning_rate = 0.0015
    target_layers = 3
    target_momentum = 0.85

    def evaluator(state: MechanicsState) -> Telemetry:
        hyper = state.hyper
        architecture = state.architecture
        rule = state.rule
        task_loss = (
            (hyper.learning_rate - target_learning_rate) ** 2 * 1_000_000.0
            + (architecture.layers - target_layers) ** 2 * 0.05
            + (rule.momentum - target_momentum) ** 2
            + 0.1
        )
        latency_ms = 5.0 + architecture.layers * 1.5 + architecture.experts * 0.4
        memory_mb = 128.0 + architecture.latent_dim * architecture.experts * 0.5
        return Telemetry(
            task_loss=task_loss,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            semantic_distance=0.0,
            gradient_norm=1.0,
        )

    optimizer = InwardOptimizer()
    return optimizer.optimize(evaluator, cycles=cycles)
