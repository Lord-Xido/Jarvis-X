"""Four-scale bounded auto-evolution for the Dr Moagi 3D operating architecture.

The controller adds an architecture-policy time scale above the existing sparse
state, adaptive model, and runtime-configuration loops. It never rewrites source
code or removes constitutional verification stages.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, cast

from .dr_moagi_autoexec import HashChainJournal
from .dr_moagi_field_runtime import Coordinate, SparseField
from .dr_moagi_meta_optimizer import (
    MetaOptimizationReport,
    MetaSearchConfig,
    SelfOptimizing3DSystem,
)
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, OSCycleReport, OSLifecycle


REQUIRED_PIPELINE = (
    "sparse_state",
    "uint64_bitplane",
    "inward_fold",
    "autoexec",
    "deep_distiller",
    "fixed_point",
    "pi_lambda",
    "dmos2_verify",
    "atomic_commit",
)


@dataclass(frozen=True, order=True)
class ArchitectureVector3D:
    """Signed displacement in the orchestration-policy lattice."""

    cadence: int = 0
    search: int = 0
    resilience: int = 0

    def __post_init__(self) -> None:
        for name in ("cadence", "search", "resilience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 0, 1):
                raise ValueError(f"{name} must be one of -1, 0, +1")

    @property
    def manhattan(self) -> int:
        return abs(self.cadence) + abs(self.search) + abs(self.resilience)


@dataclass(frozen=True)
class ArchitecturePolicy:
    """Bounded operating framework for the nested adaptive loops."""

    state_cycles_per_meta: int = 8
    meta_epochs_per_architecture_review: int = 3
    max_architecture_candidates: int = 7
    max_architecture_eval_cells: int = 512
    max_eval_state_cycles: int = 4
    min_architecture_improvement: float = 0.01
    max_architecture_metric_regression: float = 0.05
    rejection_penalty: float = 1_000.0
    meta_search: MetaSearchConfig = field(default_factory=MetaSearchConfig)

    def __post_init__(self) -> None:
        for name in (
            "state_cycles_per_meta",
            "meta_epochs_per_architecture_review",
            "max_architecture_candidates",
            "max_architecture_eval_cells",
            "max_eval_state_cycles",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_architecture_candidates > 26:
            raise ValueError("max_architecture_candidates must be <= 26")
        for name in (
            "min_architecture_improvement",
            "max_architecture_metric_regression",
            "rejection_penalty",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class ArchitectureMetrics:
    score: float
    reconstruction_mse: float
    distiller_residual_rms: float
    fixed_point_residual: float
    transport_bytes_per_source_cell: float
    compute_proxy: float
    mean_phase_velocity: float
    meta_relative_improvement: float
    meta_evaluations: int
    rejected: bool
    state_cycles: int

    def as_dict(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True)
class ArchitectureCandidateResult:
    vector: ArchitectureVector3D
    policy: ArchitecturePolicy
    metrics: ArchitectureMetrics
    stage: str

    def as_dict(self) -> dict[str, object]:
        return {
            "vector": asdict(self.vector),
            "policy": self.policy.as_dict(),
            "metrics": self.metrics.as_dict(),
            "stage": self.stage,
        }


@dataclass(frozen=True)
class ArchitectureEvolutionReport:
    baseline: ArchitectureCandidateResult
    best: ArchitectureCandidateResult
    promoted: bool
    relative_improvement: float
    evaluated_candidates: int
    promoted_policy: ArchitecturePolicy | None
    evaluations: tuple[ArchitectureCandidateResult, ...]
    claim_status: str = "internal_architecture_improvement_only"

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.as_dict(),
            "best": self.best.as_dict(),
            "promoted": self.promoted,
            "relative_improvement": self.relative_improvement,
            "evaluated_candidates": self.evaluated_candidates,
            "promoted_policy": (
                self.promoted_policy.as_dict() if self.promoted_policy is not None else None
            ),
            "evaluations": [item.as_dict() for item in self.evaluations],
            "claim_status": self.claim_status,
        }


@dataclass(frozen=True)
class AutonomicRunReport:
    requested_cycles: int
    state_reports: tuple[OSCycleReport, ...]
    meta_reports: tuple[MetaOptimizationReport, ...]
    architecture_reports: tuple[ArchitectureEvolutionReport, ...]
    stopped_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_cycles": self.requested_cycles,
            "state_reports": [item.as_dict() for item in self.state_reports],
            "meta_reports": [item.as_dict() for item in self.meta_reports],
            "architecture_reports": [item.as_dict() for item in self.architecture_reports],
            "stopped_reason": self.stopped_reason,
        }


class DrMoagiArchitectureOptimizer:
    """Benchmark bounded orchestration policies around the incumbent architecture."""

    def candidate_policy(
        self, base: ArchitecturePolicy, vector: ArchitectureVector3D
    ) -> ArchitecturePolicy:
        cadence = base.state_cycles_per_meta
        search = base.meta_search
        if vector.cadence < 0:
            cadence = max(1, cadence // 2)
        elif vector.cadence > 0:
            cadence = min(64, cadence * 2)

        if vector.search < 0:
            max_candidates = max(1, search.max_candidates // 2)
            search = replace(
                search,
                max_candidates=max_candidates,
                confirm_cycles=max(1, search.confirm_cycles - 1),
                max_eval_cells=max(16, search.max_eval_cells // 2),
                survivors=max(1, min(search.survivors, max_candidates)),
            )
        elif vector.search > 0:
            max_candidates = min(26, max(search.max_candidates + 2, search.max_candidates * 2))
            search = replace(
                search,
                max_candidates=max_candidates,
                confirm_cycles=min(8, search.confirm_cycles + 1),
                max_eval_cells=min(16_384, search.max_eval_cells * 2),
                survivors=min(max_candidates, max(1, search.survivors + 1)),
            )

        if vector.resilience < 0:
            search = replace(
                search,
                min_relative_improvement=max(0.0, search.min_relative_improvement * 0.75),
                max_metric_regression=min(0.25, search.max_metric_regression * 1.25),
                rejection_penalty=max(1.0, search.rejection_penalty * 0.85),
            )
        elif vector.resilience > 0:
            search = replace(
                search,
                min_relative_improvement=min(0.25, search.min_relative_improvement * 1.25),
                max_metric_regression=max(0.0, search.max_metric_regression * 0.80),
                rejection_penalty=min(100_000.0, search.rejection_penalty * 1.15),
            )
        return replace(base, state_cycles_per_meta=cadence, meta_search=search)

    def optimize(
        self,
        source: Mapping[Coordinate, float],
        base_config: DrMoagiOSConfig,
        policy: ArchitecturePolicy,
    ) -> ArchitectureEvolutionReport:
        bounded = _bounded_source(source, policy.max_architecture_eval_cells)
        if not bounded:
            raise ValueError("architecture optimizer requires a non-empty sparse state")
        baseline = ArchitectureCandidateResult(
            ArchitectureVector3D(), policy, self._evaluate(policy, bounded, base_config), "baseline"
        )
        candidates = tuple(
            ArchitectureCandidateResult(
                vector,
                candidate,
                self._evaluate(candidate, bounded, base_config),
                "candidate",
            )
            for vector in self._vectors()[: policy.max_architecture_candidates]
            for candidate in (self.candidate_policy(policy, vector),)
        )
        best = min((baseline, *candidates), key=lambda item: (item.metrics.score, item.vector))
        relative = _relative_improvement(baseline.metrics.score, best.metrics.score)
        promoted = best.vector != ArchitectureVector3D() and self._promotion_gate(
            policy, baseline, best, relative
        )
        evaluations = (baseline, *candidates)
        return ArchitectureEvolutionReport(
            baseline=baseline,
            best=best,
            promoted=promoted,
            relative_improvement=relative,
            evaluated_candidates=len(evaluations),
            promoted_policy=best.policy if promoted else None,
            evaluations=evaluations,
        )

    def _evaluate(
        self, policy: ArchitecturePolicy, source: SparseField, base_config: DrMoagiOSConfig
    ) -> ArchitectureMetrics:
        kernel = DrMoagiOSKernel(replace(base_config, state_dir=None, auto_optimize=False))
        kernel.boot(restore=False)
        kernel.load(source)
        system = SelfOptimizing3DSystem(kernel, search=policy.meta_search)
        state_cycles = min(policy.state_cycles_per_meta, policy.max_eval_state_cycles)
        reports = cast(list[OSCycleReport], system.run(state_cycles))
        rejected = any(not item.committed for item in reports)
        count = max(1, len(reports))
        reconstruction = sum(float(item.reconstruction_mse) for item in reports) / count
        distiller = sum(float(item.distiller_residual_rms or 0.0) for item in reports) / count
        fixed_point = sum(float(item.fixed_point_residual or 0.0) for item in reports) / count
        phase = sum(float(item.phase_velocity) for item in reports) / count
        transport = float(reports[-1].transport_bytes if reports else 0) / max(1, len(source))
        compute_proxy = sum(
            (float(item.active_cells_after) + float(item.latent_cells)) / max(1, len(source))
            for item in reports
        ) / count
        meta_relative = 0.0
        meta_evaluations = 0
        if not rejected:
            meta = system.turn_inward()
            meta_relative = float(meta.relative_improvement)
            meta_evaluations = int(meta.evaluated_candidates)
        score = (
            6.0 * reconstruction
            + 4.0 * distiller * distiller
            + 2.0 * fixed_point * fixed_point
            + 0.010 * transport
            + 0.20 * compute_proxy
            + 0.25 * phase
            + 0.010 * float(state_cycles)
            + 0.002 * float(meta_evaluations)
            - 0.50 * meta_relative
            + (policy.rejection_penalty if rejected else 0.0)
        )
        return ArchitectureMetrics(
            score,
            reconstruction,
            distiller,
            fixed_point,
            transport,
            compute_proxy,
            phase,
            meta_relative,
            meta_evaluations,
            rejected,
            state_cycles,
        )

    @staticmethod
    def _vectors() -> list[ArchitectureVector3D]:
        values = (-1, 0, 1)
        vectors = [
            ArchitectureVector3D(x, y, z)
            for x in values
            for y in values
            for z in values
            if (x, y, z) != (0, 0, 0)
        ]
        return sorted(vectors, key=lambda item: (item.manhattan, item))

    @staticmethod
    def _promotion_gate(
        policy: ArchitecturePolicy,
        baseline: ArchitectureCandidateResult,
        best: ArchitectureCandidateResult,
        relative: float,
    ) -> bool:
        if best.metrics.rejected or relative < policy.min_architecture_improvement:
            return False
        regression = 1.0 + policy.max_architecture_metric_regression
        eps = 1.0e-12
        if best.metrics.reconstruction_mse > baseline.metrics.reconstruction_mse * regression + eps:
            return False
        if best.metrics.distiller_residual_rms > baseline.metrics.distiller_residual_rms * regression + eps:
            return False
        return True


class SelfEvolving3DArchitecture:
    """Autonomic controller for state, model, configuration and architecture loops."""

    def __init__(
        self, system: SelfOptimizing3DSystem, *, policy: ArchitecturePolicy | None = None
    ) -> None:
        self.system = system
        self.policy = policy or ArchitecturePolicy(meta_search=system.optimizer.search)
        self.optimizer = DrMoagiArchitectureOptimizer()
        self.architecture_epoch = 0
        self.state_cycles_since_meta = 0
        self.meta_epochs_since_architecture = 0
        self.last_architecture_report: ArchitectureEvolutionReport | None = None
        journal_path: Path | None = None
        if system.kernel.config.state_dir is not None:
            journal_path = system.kernel.config.state_dir / "architecture-journal.jsonl"
        self.architecture_journal = HashChainJournal(journal_path)

    @property
    def kernel(self) -> DrMoagiOSKernel:
        return self.system.kernel

    def step(self) -> OSCycleReport:
        return cast(OSCycleReport, self.system.step())

    def run(self, cycles: int) -> list[OSCycleReport]:
        return cast(list[OSCycleReport], self.system.run(cycles))

    def turn_inward(self) -> MetaOptimizationReport:
        report = self.system.turn_inward()
        self.state_cycles_since_meta = 0
        self.meta_epochs_since_architecture += 1
        return report

    def evolve_architecture(self) -> ArchitectureEvolutionReport:
        if self.kernel.lifecycle is OSLifecycle.RUNNING:
            raise RuntimeError("stop autorun before architecture evolution")
        if not self.kernel.loaded:
            raise RuntimeError("load a sparse state before architecture evolution")
        before_hash = str(self.kernel.status()["state_hash"])
        source = _snapshot_field(self.kernel)
        report = self.optimizer.optimize(source, self.kernel.config, self.policy)
        if str(self.kernel.status()["state_hash"]) != before_hash:
            raise RuntimeError("authoritative state changed during architecture evaluation")
        self.architecture_epoch += 1
        if report.promoted and report.promoted_policy is not None:
            self.policy = report.promoted_policy
            self.system.optimizer.search = self.policy.meta_search
        self.last_architecture_report = report
        record = report.as_dict()
        record["architecture_epoch"] = self.architecture_epoch
        record["state_hash"] = before_hash
        self.architecture_journal.append(record)
        return report

    def run_autonomic(self, cycles: int) -> AutonomicRunReport:
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer")
        if self.kernel.lifecycle is OSLifecycle.RUNNING:
            raise RuntimeError("stop autorun before autonomic evolution")
        if not self.kernel.loaded:
            raise RuntimeError("load a sparse state before autonomic evolution")
        state_reports: list[OSCycleReport] = []
        meta_reports: list[MetaOptimizationReport] = []
        architecture_reports: list[ArchitectureEvolutionReport] = []
        stopped_reason: str | None = None
        for _ in range(cycles):
            report = self.step()
            state_reports.append(report)
            if not report.committed:
                stopped_reason = report.rejection_reason or "state transaction rejected"
                break
            self.state_cycles_since_meta += 1
            if self.state_cycles_since_meta >= self.policy.state_cycles_per_meta:
                meta_reports.append(self.turn_inward())
                if (
                    self.meta_epochs_since_architecture
                    >= self.policy.meta_epochs_per_architecture_review
                ):
                    architecture_reports.append(self.evolve_architecture())
                    self.meta_epochs_since_architecture = 0
        return AutonomicRunReport(
            cycles,
            tuple(state_reports),
            tuple(meta_reports),
            tuple(architecture_reports),
            stopped_reason,
        )

    def architecture_lattice(self) -> dict[str, object]:
        measured: dict[ArchitectureVector3D, ArchitectureCandidateResult] = {}
        if self.last_architecture_report is not None:
            for item in self.last_architecture_report.evaluations:
                previous = measured.get(item.vector)
                if previous is None or item.metrics.score < previous.metrics.score:
                    measured[item.vector] = item
        nodes: list[dict[str, object]] = []
        vectors = [ArchitectureVector3D(), *self.optimizer._vectors()]
        for vector in vectors:
            candidate_policy = (
                self.policy
                if vector == ArchitectureVector3D()
                else self.optimizer.candidate_policy(self.policy, vector)
            )
            evaluation = measured.get(vector)
            nodes.append(
                {
                    "vector": asdict(vector),
                    "policy": candidate_policy.as_dict(),
                    "measured": evaluation is not None,
                    "metrics": evaluation.metrics.as_dict() if evaluation is not None else None,
                    "stage": evaluation.stage if evaluation is not None else None,
                }
            )
        return {
            "axes": {
                "x": "state-to-meta cadence",
                "y": "meta-search resource budget",
                "z": "promotion resilience",
            },
            "center": {"cadence": 0, "search": 0, "resilience": 0},
            "nodes": nodes,
            "architecture_epoch": self.architecture_epoch,
        }

    def status(self) -> dict[str, object]:
        return {
            **self.system.status(),
            "architecture_evolution": {
                "epoch": self.architecture_epoch,
                "policy": self.policy.as_dict(),
                "pipeline": list(REQUIRED_PIPELINE),
                "nested_time_scales": {
                    "t": "authoritative sparse state",
                    "u": "Omega/Theta adaptive model",
                    "n": "runtime configuration meta-optimizer",
                    "k": "architecture orchestration policy",
                },
                "state_cycles_since_meta": self.state_cycles_since_meta,
                "meta_epochs_since_architecture": self.meta_epochs_since_architecture,
                "journal_head": self.architecture_journal.head,
                "journal_valid": self.architecture_journal.verify(),
                "last_report": (
                    self.last_architecture_report.as_dict()
                    if self.last_architecture_report is not None
                    else None
                ),
                "required_pipeline_mutable": False,
                "arbitrary_source_rewrite": False,
                "external_sota_verified": False,
            },
        }

    def capabilities(self) -> dict[str, object]:
        return {
            "adaptation_scales": {
                "state": "transactional sparse 3D state evolution",
                "model": "DM-DD Omega/Theta residual learning",
                "configuration": "bounded 3D runtime meta-search",
                "architecture": "bounded orchestration-policy search",
            },
            "required_pipeline": list(REQUIRED_PIPELINE),
            "architecture_axes": {
                "x": "state-to-meta cadence",
                "y": "meta-search budget/depth",
                "z": "promotion resilience",
            },
            "autonomic_run": True,
            "transactional_state_commit": True,
            "transactional_configuration_promotion": True,
            "transactional_architecture_promotion": True,
            "arbitrary_host_commands": False,
            "self_rewriting_source": False,
            "external_sota_verified": False,
        }


def _snapshot_field(kernel: DrMoagiOSKernel) -> SparseField:
    snap = kernel.snapshot(limit=kernel.config.max_active_cells)
    rows = snap.get("cells")
    if not isinstance(rows, list):
        raise RuntimeError("kernel snapshot cells are unavailable")
    result: SparseField = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid kernel snapshot row")
        result[(int(row["x"]), int(row["y"]), int(row["z"]))] = float(row["value"])
    return result


def _bounded_source(source: Mapping[Coordinate, float], limit: int) -> SparseField:
    ordered = sorted(source.items())
    if len(ordered) <= limit:
        return {coordinate: float(value) for coordinate, value in ordered}
    stride = len(ordered) / limit
    indices = [min(len(ordered) - 1, int(index * stride)) for index in range(limit)]
    return {ordered[index][0]: float(ordered[index][1]) for index in indices}


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate >= baseline else 1.0
    return max(0.0, (baseline - candidate) / baseline)


__all__ = [
    "ArchitectureCandidateResult",
    "ArchitectureEvolutionReport",
    "ArchitectureMetrics",
    "ArchitecturePolicy",
    "ArchitectureVector3D",
    "AutonomicRunReport",
    "DrMoagiArchitectureOptimizer",
    "REQUIRED_PIPELINE",
    "SelfEvolving3DArchitecture",
]
