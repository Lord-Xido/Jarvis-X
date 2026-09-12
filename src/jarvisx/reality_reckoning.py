"""Reality-constrained recursive reckoning for the Dr Moagi sparse runtime.

The glyph ``⟲⊙`` is treated here as an executable contract rather than a logo:
recursive internal stabilization (⟲) may advance only while correspondence to a
declared external reference (⊙) is non-worsening. Convergence requires both an
internal fixed point and an external correspondence threshold.

This module intentionally does not claim that an arbitrary reference field is
"truth". The caller owns observation provenance. The runtime guarantees only
that the declared anchor participates in every admitted recursive transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from .dm_vomegaxi_fixed_point import (
    DMvOmegaXiFixedPointConfig,
    DMvOmegaXiFixedPointEngine,
    FixedPointReport,
    ThetaGate,
)
from .dr_moagi_field_runtime import Coordinate, SparseField


@dataclass(frozen=True)
class RealityEvaluation:
    """Measured correspondence between a candidate field and the world anchor."""

    residual: float
    evidence_cells: int
    within_tolerance: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.residual) or self.residual < 0.0:
            raise ValueError("residual must be finite and non-negative")
        if isinstance(self.evidence_cells, bool) or not isinstance(self.evidence_cells, int):
            raise TypeError("evidence_cells must be an integer")
        if self.evidence_cells < 0:
            raise ValueError("evidence_cells must be non-negative")


class SparseRealityAnchor:
    """Finite sparse 3D observation anchor with a bounded correction operator.

    ``correction_gain`` controls how strongly one recursive cycle moves a
    candidate toward the observation. A gain of 1 applies the full observed
    correction; smaller gains make correspondence contract over repeated cycles.
    """

    def __init__(
        self,
        world: Mapping[Coordinate, float],
        *,
        tolerance: float = 1.0e-6,
        correction_gain: float = 0.5,
    ) -> None:
        tolerance_f = float(tolerance)
        gain_f = float(correction_gain)
        if not math.isfinite(tolerance_f) or tolerance_f < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        if not math.isfinite(gain_f) or not 0.0 < gain_f <= 1.0:
            raise ValueError("correction_gain must be finite and in (0, 1]")

        normalized: SparseField = {}
        for coordinate, raw in world.items():
            if (
                len(coordinate) != 3
                or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in coordinate)
            ):
                raise TypeError("world coordinates must be integer triples")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("world values must be finite")
            normalized[coordinate] = value

        self._world = normalized
        self.tolerance = tolerance_f
        self.correction_gain = gain_f

    def snapshot(self) -> SparseField:
        return dict(self._world)

    def evaluate(self, candidate: Mapping[Coordinate, float]) -> RealityEvaluation:
        support = set(self._world) | set(candidate)
        if not support:
            residual = 0.0
        else:
            mse = sum(
                (
                    float(candidate.get(coordinate, 0.0))
                    - float(self._world.get(coordinate, 0.0))
                )
                ** 2
                for coordinate in support
            ) / len(support)
            residual = math.sqrt(mse)
        return RealityEvaluation(
            residual=residual,
            evidence_cells=len(support),
            within_tolerance=residual <= self.tolerance,
        )

    def correct(self, candidate: Mapping[Coordinate, float]) -> SparseField:
        """Move the candidate toward the observed field on the union support."""

        gain = self.correction_gain
        corrected: SparseField = {}
        for coordinate in sorted(set(self._world) | set(candidate)):
            value = float(candidate.get(coordinate, 0.0))
            target = float(self._world.get(coordinate, 0.0))
            next_value = value + gain * (target - value)
            if next_value != 0.0:
                corrected[coordinate] = next_value
        return corrected


@dataclass(frozen=True)
class RealityFixedPointReport(FixedPointReport):
    """One Generate→Contrast→Reckon→Verify→Correct→Recur transaction."""

    internal_converged: bool = False
    external_residual_before: float = 0.0
    external_residual_generated: float = 0.0
    external_residual_after: float = 0.0
    external_correspondence: bool = False
    reality_corrected: bool = False
    journal_hash: str = ""


class RealityConstrainedDMvOmegaXiEngine(DMvOmegaXiFixedPointEngine):
    """DM-vOmegaXi+ with the canonical ``⟲⊙`` correspondence invariant.

    A cycle is admitted when numerical/policy constraints pass and the proposed
    transition does not worsen correspondence to the supplied world anchor.
    The anchor's correction operator is applied before admission. Successful
    convergence requires both the inherited internal fixed-point tolerance and
    the anchor's external correspondence tolerance.
    """

    GLYPH = "⟲⊙"
    OPERATOR_NAME = "Reality-Constrained Recursive Reckoning"
    RECKONING_CYCLE = (
        "Generate",
        "Contrast",
        "Reckon",
        "Verify",
        "Correct",
        "Recur",
    )
    PERMEATION_STACK = ("Psi", "Phi", "Lambda^-1", "Omega", "Theta", "X_world")

    def __init__(
        self,
        reality_anchor: SparseRealityAnchor,
        config: DMvOmegaXiFixedPointConfig | None = None,
        *,
        theta_gate: ThetaGate | None = None,
        journal_path: str | Path | None = None,
    ) -> None:
        if not isinstance(reality_anchor, SparseRealityAnchor):
            raise TypeError("reality_anchor must be a SparseRealityAnchor")
        super().__init__(config, theta_gate=theta_gate, journal_path=journal_path)
        self.reality_anchor = reality_anchor
        self.reports.clear()

    def step(self) -> RealityFixedPointReport:
        self._require_loaded()
        current: SparseField = self.snapshot()
        support = tuple(sorted(current))

        # Generate: execute the inherited bounded sparse internal operator.
        latent = self.phi(current)
        bounded_latent = self.lambda_inverse(latent)
        decoded = self.codec.decode(bounded_latent, support)
        folded = self.omega_fold(decoded, support)
        generated = self.theta_project(current, folded)

        # Contrast/Reckon: measure internal and external discrepancy separately.
        before_eval = self.reality_anchor.evaluate(current)
        generated_eval = self.reality_anchor.evaluate(generated)

        # Correct: reality participates before any authoritative-state commit.
        corrected_raw = self.reality_anchor.correct(generated)
        corrected = self._bound_sparse_field(corrected_raw)
        corrected_eval = self.reality_anchor.evaluate(corrected)

        # Never accept a correction that is less reality-correspondent than the
        # internally generated proposal. This keeps the operator monotone in the
        # declared external discrepancy even for future anchor variants.
        if corrected_eval.residual <= generated_eval.residual:
            candidate = corrected
            final_eval = corrected_eval
        else:
            candidate = generated
            final_eval = generated_eval

        reconstruction_rms = self._rms(current, decoded)
        memory_rms = self._rms(candidate, folded)
        theta_rms = self._rms(candidate, current)
        representation_rms = self._rms(candidate, decoded)
        residual = max(theta_rms, memory_rms, representation_rms)
        semantic_gap = max(self.config.semantic_floor, reconstruction_rms)
        internal_converged = residual <= self.config.fixed_point_tolerance
        nonworsening_external = final_eval.residual <= before_eval.residual + 1.0e-15

        rejection_reason: str | None = None
        theta_gate_passed = True
        if len(candidate) > self.config.max_active_cells:
            rejection_reason = "active-cell budget exceeded"
        elif not self._finite(candidate):
            rejection_reason = "non-finite candidate state"
        elif self.theta_gate is not None:
            theta_gate_passed = bool(self.theta_gate(candidate))
            if not theta_gate_passed:
                rejection_reason = "Theta policy gate rejected candidate"
        if rejection_reason is None and not nonworsening_external:
            rejection_reason = "reality correspondence worsened"

        committed = rejection_reason is None
        if committed:
            self._state = candidate
            self._memory = folded
            self._iteration += 1

        external_correspondence = bool(committed and final_eval.within_tolerance)
        converged = bool(committed and internal_converged and external_correspondence)
        authoritative = self._state if committed else current
        provisional = RealityFixedPointReport(
            iteration=self._iteration,
            committed=committed,
            active_cells=len(authoritative),
            latent_cells=bounded_latent.latent_cells,
            reconstruction_rms=reconstruction_rms,
            memory_rms=memory_rms,
            theta_rms=theta_rms,
            fixed_point_residual=residual,
            semantic_gap=semantic_gap,
            converged=converged,
            theta_gate_passed=theta_gate_passed,
            rejection_reason=rejection_reason,
            state_hash=self._state_hash(authoritative),
            internal_converged=internal_converged,
            external_residual_before=before_eval.residual,
            external_residual_generated=generated_eval.residual,
            external_residual_after=final_eval.residual,
            external_correspondence=external_correspondence,
            reality_corrected=candidate != generated,
        )
        record = provisional.as_dict()
        record.pop("journal_hash", None)
        record["glyph"] = self.GLYPH
        record["operator"] = self.OPERATOR_NAME
        digest = self.journal.append(record)
        report = replace(provisional, journal_hash=digest)
        self.reports.append(report)
        return report

    def run_until_fixed_point(
        self,
        max_iterations: int | None = None,
    ) -> tuple[RealityFixedPointReport, ...]:
        self._require_loaded()
        limit = self.config.max_iterations if max_iterations is None else max_iterations
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("max_iterations must be a positive integer")
        for _ in range(limit):
            report = self.step()
            if report.converged or not report.committed:
                break
        return tuple(
            report for report in self.reports if isinstance(report, RealityFixedPointReport)
        )

    def status(self) -> dict[str, object]:
        status: dict[str, object] = super().status()
        latest_base = self.reports[-1] if self.reports else None
        latest = latest_base if isinstance(latest_base, RealityFixedPointReport) else None
        status.update(
            {
                "glyph": self.GLYPH,
                "operator": self.OPERATOR_NAME,
                "reckoning_cycle": list(self.RECKONING_CYCLE),
                "permeation_stack": list(self.PERMEATION_STACK),
                "reality_fixed_point_equation": (
                    "H* = F_DM(H*; X_world), "
                    "||H* - F_DM(H*; X_world)|| <= epsilon_i, "
                    "d(H*, X_world) <= epsilon_e"
                ),
                "external_tolerance": self.reality_anchor.tolerance,
                "external_residual": latest.external_residual_after if latest else None,
                "external_correspondence": latest.external_correspondence if latest else False,
                "internal_converged": latest.internal_converged if latest else False,
                "converged": latest.converged if latest else False,
            }
        )
        return status

    def _bound_sparse_field(self, field: Mapping[Coordinate, float]) -> SparseField:
        """Project reality correction back into the runtime's admissible domain."""

        eps = self.config.policy.prune_epsilon
        result: SparseField = {}
        for coordinate in sorted(field):
            value = max(
                self.config.value_min,
                min(self.config.value_max, float(field[coordinate])),
            )
            if abs(value) > eps:
                result[coordinate] = value
        return result
