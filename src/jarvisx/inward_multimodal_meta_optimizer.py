"""Candidate-first self-optimization for the inward multimodal 3D swarm runtime.

This research-layer optimizer does not rewrite executable source code.  It treats a
bounded subset of :class:`Swarm3DConfig` as a runtime genotype, evaluates mutated
candidate configurations through a caller-supplied shadow evaluator, rejects
candidates that violate stability/fixed-point/resource gates, and promotes only
verified improvements.

The intended outer loop is::

    run -> measure -> mutate -> shadow evaluate -> verify -> promote/reject

while the inner runtime remains::

    encode -> 3D coordination -> inward E(D(z)) -> decode -> re-encode.

This separation keeps adaptation auditable and consistent with Jarvis-X's
candidate-first research/runtime boundary.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable

from .inward_multimodal_swarm3d import Swarm3DConfig


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _non_negative(value: float, *, name: str) -> float:
    value = _finite(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _unit(value: float, *, name: str) -> float:
    value = _finite(value, name=name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie within [0, 1]")
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


@dataclass(frozen=True)
class MetaFitness:
    """Measured fitness returned by a shadow runtime evaluation.

    The three positive quality terms are normalized to [0, 1].  Fixed-point
    error and resource cost are non-negative measured penalties.  ``stable`` is
    a hard gate, not merely another weighted term.
    """

    task_score: float
    semantic_coherence: float
    feature_coherence: float
    fixed_point_error: float
    resource_cost: float
    stable: bool

    def __post_init__(self) -> None:
        _unit(self.task_score, name="task_score")
        _unit(self.semantic_coherence, name="semantic_coherence")
        _unit(self.feature_coherence, name="feature_coherence")
        _non_negative(self.fixed_point_error, name="fixed_point_error")
        _non_negative(self.resource_cost, name="resource_cost")
        if not isinstance(self.stable, bool):
            raise TypeError("stable must be boolean")

    @property
    def score(self) -> float:
        """Composite ranking score; promotion still requires all hard gates."""

        stability = 1.0 / (1.0 + self.fixed_point_error)
        efficiency = 1.0 / (1.0 + self.resource_cost)
        return (
            0.30 * self.task_score
            + 0.25 * self.semantic_coherence
            + 0.20 * self.feature_coherence
            + 0.15 * stability
            + 0.10 * efficiency
        )


RuntimeEvaluator = Callable[[Swarm3DConfig], MetaFitness]


@dataclass(frozen=True)
class MetaSearchConfig:
    """Bounds for deterministic candidate-first meta optimization."""

    generations: int = 6
    branch_width: int = 10
    seed: int = 42
    mutation_fraction: float = 0.12
    improvement_threshold: float = 0.0025
    max_fixed_point_regression: float = 0.0
    max_resource_regression: float = 0.25

    def __post_init__(self) -> None:
        for name in ("generations", "branch_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if _finite(self.mutation_fraction, name="mutation_fraction") <= 0.0:
            raise ValueError("mutation_fraction must be positive")
        _non_negative(self.improvement_threshold, name="improvement_threshold")
        _non_negative(self.max_fixed_point_regression, name="max_fixed_point_regression")
        _non_negative(self.max_resource_regression, name="max_resource_regression")


@dataclass(frozen=True)
class CandidateEvaluation:
    generation: int
    candidate: int
    config: Swarm3DConfig
    fitness: MetaFitness
    eligible: bool
    reason: str


@dataclass(frozen=True)
class MetaOptimizationReport:
    baseline_config: Swarm3DConfig
    baseline_fitness: MetaFitness
    final_config: Swarm3DConfig
    final_fitness: MetaFitness
    promoted: bool
    evaluated_candidates: int
    evaluations: tuple[CandidateEvaluation, ...]

    @property
    def relative_improvement(self) -> float:
        baseline = self.baseline_fitness.score
        if baseline == 0.0:
            return math.inf if self.final_fitness.score > 0.0 else 0.0
        return self.final_fitness.score / baseline - 1.0


class InwardMultimodalMetaOptimizer:
    """Bounded outer-loop optimizer for ``Swarm3DConfig``.

    The evaluator is deliberately injected by the caller.  A production caller
    can therefore run each candidate in an isolated shadow runtime, collect task,
    semantic, feature, fixed-point and resource telemetry, and return only the
    normalized ``MetaFitness`` contract to this optimizer.
    """

    def __init__(self, search: MetaSearchConfig | None = None) -> None:
        self.search = search or MetaSearchConfig()

    def candidate_configs(
        self,
        incumbent: Swarm3DConfig,
        *,
        generation: int,
    ) -> tuple[Swarm3DConfig, ...]:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        rng = random.Random(self.search.seed + 104_729 * generation)
        return tuple(self._mutate(incumbent, rng) for _ in range(self.search.branch_width))

    def _mutate(self, config: Swarm3DConfig, rng: random.Random) -> Swarm3DConfig:
        scale = self.search.mutation_fraction

        def perturb(value: float, low: float, high: float, floor: float = 0.02) -> float:
            sigma = scale * max(abs(float(value)), floor)
            return _clamp(float(value) + rng.gauss(0.0, sigma), low, high)

        step_factor = 1.0 + rng.gauss(0.0, scale)
        max_steps = max(4, min(256, int(round(config.max_steps * step_factor))))

        return replace(
            config,
            dt=perturb(config.dt, 1.0e-4, 0.50),
            alpha_metric=perturb(config.alpha_metric, 0.0, 4.0),
            task_gain=perturb(config.task_gain, 0.0, 4.0),
            inward_gain=perturb(config.inward_gain, 0.0, 2.0),
            swarm_gain=perturb(config.swarm_gain, 0.0, 2.0),
            memory_gain=perturb(config.memory_gain, 0.0, 1.0),
            feature_mix_gain=perturb(config.feature_mix_gain, 0.0, 1.0),
            feature_similarity_gain=perturb(config.feature_similarity_gain, 0.0, 4.0),
            geometry_distance_gain=perturb(config.geometry_distance_gain, 0.0, 4.0),
            max_steps=max_steps,
        )

    def _eligible(self, incumbent: MetaFitness, candidate: MetaFitness) -> tuple[bool, str]:
        if not candidate.stable:
            return False, "candidate failed the stability gate"

        fp_limit = incumbent.fixed_point_error + self.search.max_fixed_point_regression
        if candidate.fixed_point_error > fp_limit + 1.0e-12:
            return False, "candidate regressed fixed-point integrity"

        resource_limit = incumbent.resource_cost * (1.0 + self.search.max_resource_regression)
        if candidate.resource_cost > resource_limit + 1.0e-12:
            return False, "candidate exceeded the resource-regression gate"

        required = incumbent.score * (1.0 + self.search.improvement_threshold)
        if candidate.score <= required:
            return False, "candidate did not clear the minimum improvement threshold"

        return True, "candidate passed stability, integrity, resource and fitness gates"

    def optimize(
        self,
        incumbent: Swarm3DConfig,
        evaluator: RuntimeEvaluator,
    ) -> MetaOptimizationReport:
        baseline_config = incumbent
        baseline_fitness = evaluator(incumbent)
        if not baseline_fitness.stable:
            raise ValueError("incumbent runtime must be stable before meta optimization")

        current_config = incumbent
        current_fitness = baseline_fitness
        evaluations: list[CandidateEvaluation] = []

        for generation in range(self.search.generations):
            best_config: Swarm3DConfig | None = None
            best_fitness: MetaFitness | None = None

            for index, candidate_config in enumerate(
                self.candidate_configs(current_config, generation=generation)
            ):
                candidate_fitness = evaluator(candidate_config)
                eligible, reason = self._eligible(current_fitness, candidate_fitness)
                evaluations.append(
                    CandidateEvaluation(
                        generation=generation,
                        candidate=index,
                        config=candidate_config,
                        fitness=candidate_fitness,
                        eligible=eligible,
                        reason=reason,
                    )
                )
                if eligible and (
                    best_fitness is None or candidate_fitness.score > best_fitness.score
                ):
                    best_config = candidate_config
                    best_fitness = candidate_fitness

            if best_config is not None and best_fitness is not None:
                current_config = best_config
                current_fitness = best_fitness

        return MetaOptimizationReport(
            baseline_config=baseline_config,
            baseline_fitness=baseline_fitness,
            final_config=current_config,
            final_fitness=current_fitness,
            promoted=current_config != baseline_config,
            evaluated_candidates=len(evaluations),
            evaluations=tuple(evaluations),
        )


__all__ = [
    "CandidateEvaluation",
    "InwardMultimodalMetaOptimizer",
    "MetaFitness",
    "MetaOptimizationReport",
    "MetaSearchConfig",
    "RuntimeEvaluator",
]
