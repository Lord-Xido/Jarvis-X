"""Canonical kinetic adapter for the Dr Moagi 3D meta-optimizer.

The adapter makes mechanics evolution obey the same candidate-first transaction
law as the canonical Jarvis-X runtime while preserving world-state authority.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Mapping

from .dr_moagi_autoexec import HashChainJournal
from .dr_moagi_field_runtime import Coordinate, SparseField
from .dr_moagi_meta_optimizer import (
    DrMoagi3DMetaOptimizer,
    MetaOptimizationReport,
    MetaSearchConfig,
)
from .dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, OSLifecycle
from .kinetic_runtime import KineticReceipt, KineticTransactionEngine, ValidatorResult


class CanonicalKineticDrMoagiSystem:
    """Run bounded mechanics optimization through the canonical kinetic gate."""

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
        self.last_kinetic_receipt: KineticReceipt | None = None
        self._kinetic_source: SparseField = {}
        self._kinetic_before_hash = ""

        journal_path: Path | None = None
        if kernel.config.state_dir is not None:
            journal_path = kernel.config.state_dir / "canonical-kinetic-meta-journal.jsonl"
        self.kinetic_journal = HashChainJournal(journal_path)

        self._engine = KineticTransactionEngine(
            snapshot=lambda config: replace(config),
            observe=self._observe,
            encode=self._encode,
            propose=self._propose,
            shadow=self._shadow,
            validators=(self._promotion_validator, self._world_state_validator),
            commit=self._commit_config,
            rollback=lambda config: config,
            state_identity=_config_dict,
            candidate_identity=lambda report: report.as_dict(),
        )

    def _observe(self, config: DrMoagiOSConfig) -> Mapping[str, object]:
        del config
        status = self.kernel.status()
        return {
            "state_hash": str(status["state_hash"]),
            "cycle": status["cycle"],
            "active_cells": status["active_cells"],
        }

    def _encode(
        self,
        config: DrMoagiOSConfig,
        observation: Mapping[str, object],
    ) -> SparseField:
        del config, observation
        return dict(self._kinetic_source)

    def _propose(
        self,
        config: DrMoagiOSConfig,
        observation: Mapping[str, object],
        encoded: SparseField,
    ) -> MetaOptimizationReport:
        del observation
        return self.optimizer.optimize(encoded, config)

    def _shadow(
        self,
        config: DrMoagiOSConfig,
        report: MetaOptimizationReport,
    ) -> Mapping[str, object]:
        del config
        return {
            "baseline_score": report.baseline.metrics.score,
            "best_score": report.best.metrics.score,
            "relative_improvement": report.relative_improvement,
            "evaluated_candidates": report.evaluated_candidates,
            "candidate_promoted_by_meta_gate": report.promoted,
            "claim_status": report.claim_status,
        }

    def _promotion_validator(
        self,
        config: DrMoagiOSConfig,
        report: MetaOptimizationReport,
    ) -> ValidatorResult:
        del config
        passed = report.promoted and report.promoted_config is not None
        return ValidatorResult(
            name="lambda_meta_promotion",
            passed=passed,
            metrics={
                "relative_improvement": report.relative_improvement,
                "evaluated_candidates": report.evaluated_candidates,
            },
            reason=(
                "bounded meta gate accepted candidate" if passed else "retain incumbent mechanics"
            ),
        )

    def _world_state_validator(
        self,
        config: DrMoagiOSConfig,
        report: MetaOptimizationReport,
    ) -> ValidatorResult:
        del config, report
        current_hash = str(self.kernel.status()["state_hash"])
        passed = current_hash == self._kinetic_before_hash
        return ValidatorResult(
            name="lambda_world_state_integrity",
            passed=passed,
            metrics={
                "before_state_hash": self._kinetic_before_hash,
                "current_state_hash": current_hash,
            },
            reason=(
                "world state remained immutable during mechanics search"
                if passed
                else "world state changed during mechanics search"
            ),
        )

    @staticmethod
    def _commit_config(
        config: DrMoagiOSConfig,
        report: MetaOptimizationReport,
    ) -> DrMoagiOSConfig:
        if report.promoted_config is None:
            return config
        return report.promoted_config

    def turn_inward(self) -> MetaOptimizationReport:
        """Execute one complete mechanics-state kinetic transaction."""

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

        self._kinetic_source = source
        self._kinetic_before_hash = before_hash
        result = self._engine.step(self.kernel.config)
        report = result.candidate

        if result.committed:
            promoted_config = replace(result.state, state_dir=old_state_dir)
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

        self.meta_epoch += 1
        self.last_meta_report = report
        self.last_kinetic_receipt = result.receipt
        self.kinetic_journal.append(
            {
                "meta_epoch": self.meta_epoch,
                "state_hash": before_hash,
                "report": report.as_dict(),
                "kinetic_receipt": result.receipt.to_dict(),
            }
        )
        return report

    def step(self):
        return self.kernel.step()

    def run(self, cycles: int):
        return self.kernel.run(cycles)

    def status(self) -> dict[str, object]:
        return {
            **self.kernel.status(),
            "canonical_kinetic_meta": {
                "epoch": self.meta_epoch,
                "receipt_head": self._engine.previous_receipt_hash,
                "journal_head": self.kinetic_journal.head,
                "journal_valid": self.kinetic_journal.verify(),
                "last_receipt": (
                    self.last_kinetic_receipt.to_dict()
                    if self.last_kinetic_receipt is not None
                    else None
                ),
                "last_report": (
                    self.last_meta_report.as_dict() if self.last_meta_report is not None else None
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
    snapshot = kernel.snapshot(limit=kernel.config.max_active_cells)
    rows = snapshot.get("cells")
    if not isinstance(rows, list):
        raise RuntimeError("kernel snapshot cells are unavailable")
    result: dict[Coordinate, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid kernel snapshot row")
        coordinate = (int(row["x"]), int(row["y"]), int(row["z"]))
        result[coordinate] = float(row["value"])
    return result


__all__ = ["CanonicalKineticDrMoagiSystem"]
