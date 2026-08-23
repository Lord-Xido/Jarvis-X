"""Bounded inward 3D meta-optimization for the Dr Moagi operating runtime.

The optimizer turns the runtime onto its *configuration*, not onto arbitrary
source code. It searches a three-axis neighbourhood:

    X: compression geometry  (block size, quantization, pruning)
    Y: adaptive dynamics     (DM-DD learning rate, memory and pass depth)
    Z: spatial dynamics      (fold/attenuation and fixed-point depth)

Every candidate is replayed in an isolated DrMoagiOSKernel against deterministic
robustness workloads. Promotion is allowed only when the candidate improves a
multi-metric objective without materially regressing reconstruction or anchor
fidelity. The production state remains authoritative until the meta gate accepts
and the wrapper atomically swaps to the promoted kernel configuration.

This is self-optimization of a bounded policy/model configuration. It does not
rewrite Python source, execute host commands, or assert state-of-the-art status
without matched external benchmark evidence.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

from .dr_moagi_autoexec import HashChainJournal
from .dr_moagi_field_runtime import Coordinate, SparseField
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, OSLifecycle


@dataclass(frozen=True, order=True)
class MetaVector3D:
    """A signed displacement in the bounded 3D policy lattice."""

    compression: int = 0
    adaptation: int = 0
    dynamics: int = 0

    def __post_init__(self) -> None:
        for name in ("compression", "adaptation", "dynamics"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 0, 1):
                raise ValueError(f"{name} must be one of -1, 0, +1")

    @property
    def manhattan(self) -> int:
        return abs(self.compression) + abs(self.adaptation) + abs(self.dynamics)


@dataclass(frozen=True)
class MetaSearchConfig:
    max_candidates: int = 13
    probe_cycles: int = 1
    confirm_cycles: int = 3
    max_eval_cells: int = 2_048
    survivors: int = 4
    min_relative_improvement: float = 0.01
    max_metric_regression: float = 0.05
    rejection_penalty: float = 1_000.0

    def __post_init__(self) -> None:
        for name in (
            "max_candidates",
            "probe_cycles",
            "confirm_cycles",
            "max_eval_cells",
            "survivors",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "min_relative_improvement",
            "max_metric_regression",
            "rejection_penalty",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class MetaMetrics:
    score: float
    reconstruction_mse: float
    distiller_residual_rms: float
    fixed_point_residual: float
    anchor_drift_mse: float
    transport_bytes_per_source_cell: float
    compute_proxy: float
    mean_phase_velocity: float
    rejected: bool
    workloads: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MetaCandidateResult:
    vector: MetaVector3D
    config: DrMoagiOSConfig
    metrics: MetaMetrics
    stage: str

    def as_dict(self) -> dict[str, object]:
        return {
            "vector": asdict(self.vector),
            "config": _config_dict(self.config),
            "metrics": self.metrics.as_dict(),
            "stage": self.stage,
        }


@dataclass(frozen=True)
class MetaOptimizationReport:
    baseline: MetaCandidateResult
    best: MetaCandidateResult
    promoted: bool
    relative_improvement: float
    evaluated_candidates: int
    promoted_config: DrMoagiOSConfig | None
    evaluations: tuple[MetaCandidateResult, ...]
    claim_status: str = "unverified_against_external_sota"

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.as_dict(),
            "best": self.best.as_dict(),
            "promoted": self.promoted,
            "relative_improvement": self.relative_improvement,
            "evaluated_candidates": self.evaluated_candidates,
            "promoted_config": (
                _config_dict(self.promoted_config) if self.promoted_config is not None else None
            ),
            "evaluations": [item.as_dict() for item in self.evaluations],
            "claim_status": self.claim_status,
        }


class DrMoagi3DMetaOptimizer:
    """Search a bounded 3D configuration lattice using deterministic replay."""

    def __init__(self, search: MetaSearchConfig | None = None) -> None:
        self.search = search or MetaSearchConfig()

    def optimize(
        self,
        source: Mapping[Coordinate, float],
        base_config: DrMoagiOSConfig,
    ) -> MetaOptimizationReport:
        bounded = _bounded_source(source, self.search.max_eval_cells)
        if not bounded:
            raise ValueError("meta-optimizer requires a non-empty sparse state")

        baseline_config = replace(base_config, state_dir=None, auto_optimize=False)
        baseline = MetaCandidateResult(
            vector=MetaVector3D(),
            config=baseline_config,
            metrics=self._evaluate(baseline_config, bounded, self.search.confirm_cycles),
            stage="baseline",
        )

        vectors = self._vectors()[: self.search.max_candidates]
        probes: list[MetaCandidateResult] = []
        for vector in vectors:
            candidate_config = self.candidate_config(base_config, vector)
            probes.append(
                MetaCandidateResult(
                    vector=vector,
                    config=candidate_config,
                    metrics=self._evaluate(candidate_config, bounded, self.search.probe_cycles),
                    stage="probe",
                )
            )

        ranked = sorted(probes, key=lambda result: (result.metrics.score, result.vector))
        survivor_count = min(self.search.survivors, len(ranked))
        confirmed: list[MetaCandidateResult] = []
        for probe in ranked[:survivor_count]:
            confirmed.append(
                MetaCandidateResult(
                    vector=probe.vector,
                    config=probe.config,
                    metrics=self._evaluate(probe.config, bounded, self.search.confirm_cycles),
                    stage="confirm",
                )
            )

        best = min([baseline, *confirmed], key=lambda result: (result.metrics.score, result.vector))
        relative = _relative_improvement(baseline.metrics.score, best.metrics.score)
        promoted = best.vector != MetaVector3D() and self._promotion_gate(baseline, best, relative)
        evaluations = (baseline, *probes, *confirmed)
        return MetaOptimizationReport(
            baseline=baseline,
            best=best,
            promoted=promoted,
            relative_improvement=relative,
            evaluated_candidates=len(evaluations),
            promoted_config=(best.config if promoted else None),
            evaluations=evaluations,
        )

    def candidate_config(
        self,
        base: DrMoagiOSConfig,
        vector: MetaVector3D,
    ) -> DrMoagiOSConfig:
        dx, dy, dz = vector.compression, vector.adaptation, vector.dynamics

        block = base.block_size
        quant = base.quantization
        prune = base.prune_epsilon
        if dx < 0:
            block = max(1, block // 2)
            quant = max(1.0e-9, quant * 0.5)
            prune = max(0.0, prune * 0.5)
        elif dx > 0:
            block = min(16, block * 2)
            quant = min(1.0, quant * 2.0)
            prune = min(1.0, prune + 0.005)

        lr = base.deep_distiller_learning_rate
        rho = base.deep_distiller_rho
        omega_gain = base.deep_distiller_omega_gain
        dd_passes = base.deep_distiller_passes
        latent = base.deep_distiller_max_latent_cells
        if dy < 0:
            lr *= 0.70
            rho = min(0.98, rho + 0.03)
            omega_gain *= 0.85
            dd_passes = max(1, dd_passes - 1) if base.deep_distiller_enabled else 0
            latent = max(1, int(latent * 0.85))
        elif dy > 0:
            lr *= 1.30
            rho = max(0.0, rho - 0.03)
            omega_gain *= 1.15
            dd_passes = min(4, max(1, dd_passes + 1)) if base.deep_distiller_enabled else 0
            latent = min(base.max_active_cells, max(1, int(latent * 1.15)))

        contraction = base.contraction
        attenuation = base.attenuation
        fp_passes = base.fixed_point_passes
        if dz < 0:
            contraction *= 0.75
            attenuation *= 0.75
            fp_passes = max(0, fp_passes - 1)
        elif dz > 0:
            contraction = min(0.95, contraction * 1.25)
            attenuation *= 1.25
            fp_passes = min(4, fp_passes + 1)

        return replace(
            base,
            block_size=block,
            quantization=quant,
            prune_epsilon=prune,
            deep_distiller_learning_rate=lr,
            deep_distiller_rho=rho,
            deep_distiller_omega_gain=omega_gain,
            deep_distiller_passes=dd_passes,
            deep_distiller_max_latent_cells=latent,
            contraction=contraction,
            attenuation=attenuation,
            fixed_point_passes=fp_passes,
            auto_optimize=False,
            state_dir=None,
        )

    def candidate_lattice(self, base_config: DrMoagiOSConfig) -> tuple[dict[str, object], ...]:
        """Describe the center and all 26 bounded neighbouring configurations."""

        center_config = replace(base_config, state_dir=None, auto_optimize=False)
        nodes: list[dict[str, object]] = [
            {
                "vector": asdict(MetaVector3D()),
                "role": "incumbent",
                "config": _config_dict(center_config),
            }
        ]
        for vector in self._vectors():
            nodes.append(
                {
                    "vector": asdict(vector),
                    "role": "candidate",
                    "config": _config_dict(self.candidate_config(base_config, vector)),
                }
            )
        return tuple(nodes)

    def _vectors(self) -> list[MetaVector3D]:
        values = (-1, 0, 1)
        vectors = [
            MetaVector3D(x, y, z)
            for x in values
            for y in values
            for z in values
            if (x, y, z) != (0, 0, 0)
        ]
        return sorted(vectors, key=lambda value: (value.manhattan, value))

    def _evaluate(
        self,
        config: DrMoagiOSConfig,
        source: SparseField,
        cycles: int,
    ) -> MetaMetrics:
        workload_metrics = [
            self._evaluate_workload(config, workload, cycles)
            for workload in _robustness_workloads(source, config.side)
        ]
        mean_score = sum(item.score for item in workload_metrics) / len(workload_metrics)
        worst_score = max(item.score for item in workload_metrics)

        def mean(name: str) -> float:
            return sum(float(getattr(item, name)) for item in workload_metrics) / len(
                workload_metrics
            )

        return MetaMetrics(
            score=mean_score + 0.25 * worst_score,
            reconstruction_mse=mean("reconstruction_mse"),
            distiller_residual_rms=mean("distiller_residual_rms"),
            fixed_point_residual=mean("fixed_point_residual"),
            anchor_drift_mse=mean("anchor_drift_mse"),
            transport_bytes_per_source_cell=mean("transport_bytes_per_source_cell"),
            compute_proxy=mean("compute_proxy"),
            mean_phase_velocity=mean("mean_phase_velocity"),
            rejected=any(item.rejected for item in workload_metrics),
            workloads=len(workload_metrics),
        )

    def _evaluate_workload(
        self,
        config: DrMoagiOSConfig,
        source: SparseField,
        cycles: int,
    ) -> MetaMetrics:
        kernel = DrMoagiOSKernel(replace(config, state_dir=None, auto_optimize=False))
        kernel.boot(restore=False)
        kernel.load(source)
        reports = kernel.run(cycles)
        final = _snapshot_field(kernel)
        rejected = any(not report.committed for report in reports)
        count = max(1, len(reports))
        reconstruction = sum(float(report.reconstruction_mse) for report in reports) / count
        distiller = sum(float(report.distiller_residual_rms or 0.0) for report in reports) / count
        fixed = sum(float(report.fixed_point_residual or 0.0) for report in reports) / count
        phase = sum(float(report.phase_velocity) for report in reports) / count
        transport = float(reports[-1].transport_bytes if reports else 0) / max(1, len(source))
        compute_proxy = sum(
            (float(report.active_cells_after) + float(report.latent_cells)) / max(1, len(source))
            for report in reports
        ) / count
        drift = _field_mse(source, final)
        score = (
            6.0 * reconstruction
            + 4.0 * distiller * distiller
            + 2.0 * fixed * fixed
            + 4.0 * drift
            + 0.010 * transport
            + 0.20 * compute_proxy
            + 0.25 * phase
            + (self.search.rejection_penalty if rejected else 0.0)
        )
        return MetaMetrics(
            score=score,
            reconstruction_mse=reconstruction,
            distiller_residual_rms=distiller,
            fixed_point_residual=fixed,
            anchor_drift_mse=drift,
            transport_bytes_per_source_cell=transport,
            compute_proxy=compute_proxy,
            mean_phase_velocity=phase,
            rejected=rejected,
            workloads=1,
        )

    def _promotion_gate(
        self,
        baseline: MetaCandidateResult,
        best: MetaCandidateResult,
        relative: float,
    ) -> bool:
        if best.metrics.rejected:
            return False
        if relative < self.search.min_relative_improvement:
            return False
        regression = 1.0 + self.search.max_metric_regression
        eps = 1.0e-12
        if best.metrics.reconstruction_mse > baseline.metrics.reconstruction_mse * regression + eps:
            return False
        if best.metrics.anchor_drift_mse > baseline.metrics.anchor_drift_mse * regression + eps:
            return False
        return True


class SelfOptimizing3DSystem:
    """A higher-order runtime that can benchmark and promote its own configuration."""

    def __init__(
        self,
        kernel: DrMoagiOSKernel,
        *,
        search: MetaSearchConfig | None = None,
    ) -> None:
        self.kernel = kernel
        self.optimizer = DrMoagi3DMetaOptimizer(search)
        self.meta_epoch = 0
        self.last_meta_report: MetaOptimizationReport | None = None
        journal_path: Path | None = None
        if kernel.config.state_dir is not None:
            journal_path = kernel.config.state_dir / "meta-journal.jsonl"
        self.meta_journal = HashChainJournal(journal_path)

    def turn_inward(self) -> MetaOptimizationReport:
        if self.kernel.lifecycle is OSLifecycle.RUNNING:
            raise RuntimeError("stop autorun before meta-optimization")
        if not self.kernel.loaded:
            raise RuntimeError("load a sparse state before meta-optimization")

        before_status = self.kernel.status()
        before_hash = str(before_status["state_hash"])
        source = _snapshot_field(self.kernel)
        adaptive = self.kernel._distiller.adaptive_snapshot()
        old_cycle = self.kernel.cycle
        old_state_dir = self.kernel.config.state_dir

        report = self.optimizer.optimize(source, self.kernel.config)
        self.meta_epoch += 1
        if report.promoted and report.promoted_config is not None:
            if str(self.kernel.status()["state_hash"]) != before_hash:
                raise RuntimeError("authoritative state changed during meta-optimization")
            promoted_config = replace(report.promoted_config, state_dir=old_state_dir)
            promoted = DrMoagiOSKernel(promoted_config)
            promoted.boot(restore=False)
            promoted.load(source)
            promoted._distiller.restore_adaptive_state(
                source,
                omega=adaptive.omega,
                theta=adaptive.theta,
                iteration=adaptive.iteration,
            )
            promoted._cycle = old_cycle
            promoted._transport_packet = promoted._verified_transport(source)
            promoted._persist_checkpoint()
            self.kernel = promoted

        self.last_meta_report = report
        record = report.as_dict()
        record["meta_epoch"] = self.meta_epoch
        record["state_hash"] = before_hash
        self.meta_journal.append(record)
        return report

    def step(self):
        return self.kernel.step()

    def run(self, cycles: int):
        return self.kernel.run(cycles)

    def meta_lattice(self) -> dict[str, object]:
        """Return the live 3D candidate neighbourhood and latest measured evaluations."""

        return {
            "epoch": self.meta_epoch,
            "axes": {
                "x": "compression geometry",
                "y": "adaptive dynamics",
                "z": "spatial/fixed-point dynamics",
            },
            "current_config": _config_dict(self.kernel.config),
            "nodes": list(self.optimizer.candidate_lattice(self.kernel.config)),
            "last_report": (
                self.last_meta_report.as_dict() if self.last_meta_report is not None else None
            ),
            "claim_status": (
                self.last_meta_report.claim_status
                if self.last_meta_report is not None
                else "unverified_against_external_sota"
            ),
            "external_sota_verified": False,
        }

    def status(self) -> dict[str, object]:
        return {
            **self.kernel.status(),
            "meta_optimizer": {
                "epoch": self.meta_epoch,
                "search_space": "3D bounded policy lattice",
                "axes": {
                    "x": "compression geometry",
                    "y": "adaptive dynamics",
                    "z": "spatial/fixed-point dynamics",
                },
                "current_config": _config_dict(self.kernel.config),
                "journal_head": self.meta_journal.head,
                "journal_valid": self.meta_journal.verify(),
                "last_report": (
                    self.last_meta_report.as_dict() if self.last_meta_report is not None else None
                ),
                "claim_status": (
                    self.last_meta_report.claim_status
                    if self.last_meta_report is not None
                    else "unverified_against_external_sota"
                ),
                "external_sota_verified": False,
            },
        }


def _config_dict(config: DrMoagiOSConfig) -> dict[str, object]:
    raw = asdict(config)
    state_dir = raw.get("state_dir")
    raw["state_dir"] = str(state_dir) if state_dir is not None else None
    return raw


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
        return dict(ordered)
    stride = len(ordered) / limit
    indices = [min(len(ordered) - 1, int(index * stride)) for index in range(limit)]
    return {ordered[index][0]: float(ordered[index][1]) for index in indices}


def _robustness_workloads(source: SparseField, side: int) -> tuple[SparseField, ...]:
    base = dict(source)
    amplitude: SparseField = {}
    shifted: SparseField = {}
    for coordinate, value in sorted(source.items()):
        digest = hashlib.sha256(repr(coordinate).encode("ascii")).digest()[0]
        factor = 0.97 if digest & 1 else 1.03
        amplitude[coordinate] = float(value) * factor
        x, y, z = coordinate
        shifted[((x + 1) % side, y, z)] = float(value)
    return base, amplitude, shifted


def _field_mse(reference: Mapping[Coordinate, float], candidate: Mapping[Coordinate, float]) -> float:
    keys = set(reference) | set(candidate)
    if not keys:
        return 0.0
    error = 0.0
    for key in keys:
        delta = float(reference.get(key, 0.0)) - float(candidate.get(key, 0.0))
        error += delta * delta
    return error / len(keys)


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate >= baseline else 1.0
    return max(0.0, (baseline - candidate) / baseline)


__all__ = [
    "DrMoagi3DMetaOptimizer",
    "MetaCandidateResult",
    "MetaMetrics",
    "MetaOptimizationReport",
    "MetaSearchConfig",
    "MetaVector3D",
    "SelfOptimizing3DSystem",
]
