"""Bounded inward 3D runtime tuner for the Dr Moagi field runtime.

The tuner turns *runtime mechanics* inward onto the field engine itself:
it observes current telemetry, generates a finite set of configuration
candidates, executes each candidate in an isolated shadow runtime, applies
semantic/resource gates, and commits only a better mechanics configuration.

It never rewrites source code, changes privileges, mutates external systems,
or bypasses the field runtime's transactional validator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping

from .dr_moagi_field_runtime import (
    Coordinate,
    DrMoagiFieldConfig,
    DrMoagiFieldRuntime,
    FieldStepMetrics,
    SparseField,
    Validator,
)

Objective = Callable[
    [Mapping[Coordinate, float], FieldStepMetrics, DrMoagiFieldConfig],
    float,
]


@dataclass(frozen=True)
class InwardTuningPolicy:
    """Declared search space and acceptance contract for mechanics tuning."""

    dt_factors: tuple[float, ...] = (0.8, 1.2)
    alpha_factors: tuple[float, ...] = (0.8, 1.2)
    lambda_factors: tuple[float, ...] = (0.8, 1.2)
    eta_factors: tuple[float, ...] = (0.8, 1.2)
    prune_increments: tuple[float, ...] = (1e-6,)

    reconstruction_weight: float = 1.0
    source_anchor_weight: float = 4.0
    support_weight: float = 0.05
    rhs_weight: float = 0.01
    stability_weight: float = 0.02

    max_source_anchor_mse: float = math.inf
    max_reconstruction_mse: float = math.inf
    max_semantic_mse: float = math.inf
    min_improvement: float = 1e-12
    memory_decay: float = 0.9

    def __post_init__(self) -> None:
        for name in (
            "reconstruction_weight",
            "source_anchor_weight",
            "support_weight",
            "rhs_weight",
            "stability_weight",
            "max_source_anchor_mse",
            "max_reconstruction_mse",
            "max_semantic_mse",
            "min_improvement",
            "memory_decay",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if math.isnan(float(value)):
                raise ValueError(f"{name} cannot be NaN")

        for name in (
            "reconstruction_weight",
            "source_anchor_weight",
            "support_weight",
            "rhs_weight",
            "stability_weight",
            "max_source_anchor_mse",
            "max_reconstruction_mse",
            "max_semantic_mse",
            "min_improvement",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.memory_decay < 1.0:
            raise ValueError("memory_decay must be in [0, 1)")

        for factors_name in (
            "dt_factors",
            "alpha_factors",
            "lambda_factors",
            "eta_factors",
        ):
            for factor in getattr(self, factors_name):
                if isinstance(factor, bool) or not isinstance(factor, (int, float)):
                    raise TypeError(f"{factors_name} entries must be numeric")
                if not math.isfinite(float(factor)) or factor <= 0.0:
                    raise ValueError(
                        f"{factors_name} entries must be finite and positive"
                    )

        for increment in self.prune_increments:
            if isinstance(increment, bool) or not isinstance(increment, (int, float)):
                raise TypeError("prune_increments entries must be numeric")
            if not math.isfinite(float(increment)) or increment < 0.0:
                raise ValueError(
                    "prune_increments entries must be finite and non-negative"
                )


@dataclass(frozen=True)
class CandidateEvaluation:
    transform_id: str
    config: DrMoagiFieldConfig
    score: float | None
    metrics: FieldStepMetrics | None
    source_anchor_mse: float | None
    semantic_mse: float | None
    admissible: bool
    rejection_reason: str | None = None


@dataclass(frozen=True)
class InwardOptimizationReport:
    cycle: int
    baseline_score: float
    baseline_metrics: FieldStepMetrics
    baseline_source_anchor_mse: float
    evaluations: tuple[CandidateEvaluation, ...]
    committed: bool
    committed_transform: str | None
    committed_score: float
    improvement: float


@dataclass(frozen=True)
class InwardCycleReport:
    optimization: InwardOptimizationReport
    field_step: FieldStepMetrics


class DrMoagiInward3DTuner:
    """Finite, shadow-evaluated mechanics tuner around ``DrMoagiFieldRuntime``."""

    def __init__(
        self,
        runtime: DrMoagiFieldRuntime,
        policy: InwardTuningPolicy | None = None,
        objective: Objective | None = None,
    ) -> None:
        self.runtime = runtime
        self.policy = policy or InwardTuningPolicy()
        self.objective = objective
        self._memory: dict[str, float] = {}

    def memory_snapshot(self) -> dict[str, float]:
        """Return EWMA improvement memory keyed by transformation id."""

        return dict(self._memory)

    def optimize_once(
        self,
        validator: Validator | None = None,
    ) -> InwardOptimizationReport:
        """Evaluate bounded mechanics candidates and commit only an improvement.

        The active 3D field is never advanced during candidate search. Baseline
        and candidates all start from the same active snapshot. A winning
        candidate changes only ``runtime.config``; the next authoritative
        ``runtime.step`` uses the committed mechanics.
        """

        snapshot = self.runtime.snapshot()
        source_anchor = self.runtime.anchor_snapshot()
        active_config = self.runtime.config

        baseline_shadow, baseline_metrics = self._shadow_step(
            active_config,
            snapshot,
            validator,
        )
        if not baseline_metrics.committed:
            raise RuntimeError(
                "baseline shadow transition was rejected; mechanics tuning cannot proceed"
            )

        baseline_state = baseline_shadow.snapshot()
        baseline_source_anchor_mse = self._mse(source_anchor, baseline_state)
        baseline_score = self._score(
            baseline_state,
            baseline_metrics,
            active_config,
            baseline_source_anchor_mse,
        )

        evaluations: list[CandidateEvaluation] = []
        best: CandidateEvaluation | None = None

        for transform_id, candidate_config in self._candidate_configs(active_config):
            evaluation = self._evaluate_candidate(
                transform_id=transform_id,
                candidate_config=candidate_config,
                snapshot=snapshot,
                source_anchor=source_anchor,
                baseline_state=baseline_state,
                validator=validator,
            )
            evaluations.append(evaluation)
            if not evaluation.admissible or evaluation.score is None:
                self._remember(transform_id, 0.0)
                continue
            if best is None or evaluation.score < best.score:  # type: ignore[operator]
                best = evaluation

        committed = False
        committed_transform: str | None = None
        committed_score = baseline_score

        if (
            best is not None
            and best.score is not None
            and baseline_score - best.score >= self.policy.min_improvement
        ):
            self._commit_config(best.config)
            committed = True
            committed_transform = best.transform_id
            committed_score = best.score
            self._remember(best.transform_id, baseline_score - best.score)

        improvement = baseline_score - committed_score
        return InwardOptimizationReport(
            cycle=self.runtime.cycle,
            baseline_score=baseline_score,
            baseline_metrics=baseline_metrics,
            baseline_source_anchor_mse=baseline_source_anchor_mse,
            evaluations=tuple(evaluations),
            committed=committed,
            committed_transform=committed_transform,
            committed_score=committed_score,
            improvement=improvement,
        )

    def optimize_then_step(
        self,
        validator: Validator | None = None,
    ) -> InwardCycleReport:
        """Run one mechanics-refinement transaction followed by one field step."""

        optimization = self.optimize_once(validator=validator)
        field_step = self.runtime.step(validator=validator)
        return InwardCycleReport(optimization=optimization, field_step=field_step)

    def run(
        self,
        cycles: int,
        validator: Validator | None = None,
    ) -> tuple[InwardCycleReport, ...]:
        """Execute a bounded sequence of coupled mechanics/field cycles."""

        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
            raise ValueError("cycles must be a non-negative integer")
        return tuple(
            self.optimize_then_step(validator=validator)
            for _ in range(cycles)
        )

    def _shadow_step(
        self,
        config: DrMoagiFieldConfig,
        snapshot: SparseField,
        validator: Validator | None,
    ) -> tuple[DrMoagiFieldRuntime, FieldStepMetrics]:
        shadow = DrMoagiFieldRuntime(self.runtime.codec, config)
        shadow.load(snapshot)
        metrics = shadow.step(validator=validator)
        return shadow, metrics

    def _evaluate_candidate(
        self,
        *,
        transform_id: str,
        candidate_config: DrMoagiFieldConfig,
        snapshot: SparseField,
        source_anchor: SparseField,
        baseline_state: SparseField,
        validator: Validator | None,
    ) -> CandidateEvaluation:
        try:
            shadow, metrics = self._shadow_step(
                candidate_config,
                snapshot,
                validator,
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return CandidateEvaluation(
                transform_id=transform_id,
                config=candidate_config,
                score=None,
                metrics=None,
                source_anchor_mse=None,
                semantic_mse=None,
                admissible=False,
                rejection_reason=str(exc),
            )

        if not metrics.committed:
            return CandidateEvaluation(
                transform_id=transform_id,
                config=candidate_config,
                score=None,
                metrics=metrics,
                source_anchor_mse=None,
                semantic_mse=None,
                admissible=False,
                rejection_reason=(
                    metrics.rejection_reason or "shadow transition rejected"
                ),
            )

        state = shadow.snapshot()
        source_anchor_mse = self._mse(source_anchor, state)
        semantic_mse = self._mse(baseline_state, state)

        if metrics.reconstruction_mse > self.policy.max_reconstruction_mse:
            reason = "reconstruction error exceeds policy"
        elif source_anchor_mse > self.policy.max_source_anchor_mse:
            reason = "source-anchor drift exceeds policy"
        elif semantic_mse > self.policy.max_semantic_mse:
            reason = "semantic distance exceeds policy"
        else:
            reason = None

        if reason is not None:
            return CandidateEvaluation(
                transform_id=transform_id,
                config=candidate_config,
                score=None,
                metrics=metrics,
                source_anchor_mse=source_anchor_mse,
                semantic_mse=semantic_mse,
                admissible=False,
                rejection_reason=reason,
            )

        score = self._score(
            state,
            metrics,
            candidate_config,
            source_anchor_mse,
        )
        return CandidateEvaluation(
            transform_id=transform_id,
            config=candidate_config,
            score=score,
            metrics=metrics,
            source_anchor_mse=source_anchor_mse,
            semantic_mse=semantic_mse,
            admissible=True,
        )

    def _score(
        self,
        state: Mapping[Coordinate, float],
        metrics: FieldStepMetrics,
        config: DrMoagiFieldConfig,
        source_anchor_mse: float,
    ) -> float:
        if self.objective is not None:
            value = float(self.objective(state, metrics, config))
            if not math.isfinite(value):
                raise ValueError("objective must return a finite score")
            return value

        support_fraction = metrics.support_cells / config.max_active_cells
        return (
            self.policy.reconstruction_weight * metrics.reconstruction_mse
            + self.policy.source_anchor_weight * source_anchor_mse
            + self.policy.support_weight * support_fraction
            + self.policy.rhs_weight * metrics.max_abs_rhs
            + self.policy.stability_weight * config.stability_load
        )

    def _candidate_configs(
        self,
        active: DrMoagiFieldConfig,
    ) -> tuple[tuple[str, DrMoagiFieldConfig], ...]:
        candidates: list[tuple[str, DrMoagiFieldConfig]] = []
        seen: set[DrMoagiFieldConfig] = {active}

        def add(transform_id: str, **changes: float) -> None:
            try:
                candidate = replace(active, **changes)
            except (TypeError, ValueError):
                return
            if candidate in seen:
                return
            seen.add(candidate)
            candidates.append((transform_id, candidate))

        for factor in self.policy.dt_factors:
            add(f"DT_X_{factor:g}", dt=active.dt * factor)
        for factor in self.policy.alpha_factors:
            add(f"ALPHA_X_{factor:g}", alpha=active.alpha * factor)
        for factor in self.policy.lambda_factors:
            add(
                f"LAMBDA_X_{factor:g}",
                lambda_residual=active.lambda_residual * factor,
            )
        for factor in self.policy.eta_factors:
            add(f"ETA_X_{factor:g}", eta=active.eta * factor)
        for increment in self.policy.prune_increments:
            add(
                f"PRUNE_PLUS_{increment:g}",
                prune_epsilon=active.prune_epsilon + increment,
            )

        return tuple(candidates)

    def _commit_config(self, candidate: DrMoagiFieldConfig) -> None:
        active = self.runtime.config
        if candidate.side != active.side:
            raise RuntimeError("runtime tuning cannot resize the logical lattice")
        if candidate.max_active_cells != active.max_active_cells:
            raise RuntimeError(
                "runtime tuning cannot change the active-cell authority budget"
            )
        if candidate.value_min != active.value_min or candidate.value_max != active.value_max:
            raise RuntimeError(
                "runtime tuning cannot change projection authority bounds"
            )

        state = self.runtime.snapshot()
        if len(state) > candidate.max_active_cells:
            raise RuntimeError(
                "candidate active-cell budget is smaller than current state"
            )
        self.runtime.config = candidate

    def _remember(self, transform_id: str, improvement: float) -> None:
        decay = self.policy.memory_decay
        previous = self._memory.get(transform_id, 0.0)
        self._memory[transform_id] = (
            decay * previous + (1.0 - decay) * improvement
        )

    @staticmethod
    def _mse(
        a: Mapping[Coordinate, float],
        b: Mapping[Coordinate, float],
    ) -> float:
        support = set(a) | set(b)
        if not support:
            return 0.0
        return sum(
            (float(a.get(c, 0.0)) - float(b.get(c, 0.0))) ** 2
            for c in support
        ) / len(support)
