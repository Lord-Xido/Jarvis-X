"""Operational consideration loop for the Dr Moagi DM-vOmegaXi+ formulation.

This module adds two first-class mechanics to the existing fixed-point runtime:

* U_attn(t): deterministic salience contraction over sparse active support.
* Gamma_in: dissipative damping of innovation/noise residuals.

The implementation is a bounded numerical reference model.  It does not claim
that the symbolic operators are physical quantum operators.  In particular,
the symbolic ``-i*hbar*Gamma_in`` term is represented here by a real-valued
contractive damping operator acting on the residual that is not explained by
Phi_in.

The logical spatial domain may be extremely large, but only active sparse
coordinates are materialized.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping

from .dr_moagi_autoexec import (
    AutoExecPolicy,
    HashChainJournal,
    SparseBlockCodec3D,
    SparseLatent3D,
    SparseParser3D,
)
from .dr_moagi_field_runtime import Coordinate, SparseField


@dataclass(frozen=True)
class DMvOmegaXiConsiderationConfig:
    """Configuration for a bounded sparse consideration loop."""

    side: int = 64
    max_active_cells: int = 50_000
    value_min: float = -1.0
    value_max: float = 1.0
    policy: AutoExecPolicy = field(default_factory=AutoExecPolicy)

    # U_attn(t)
    attention_keep_ratio: float = 0.5
    attention_min_cells: int = 1

    # Lambda^-1(div_Theta Omega_t)
    latent_bound: float = 1.0
    memory_retention: float = 0.75
    memory_constraint_gain: float = 0.10
    theta_constraint: float = 1.0

    # Theta_in projection
    state_gain: float = 0.5
    max_state_delta: float = 0.25

    # -i*hbar*Gamma_in -> real-valued residual damping
    semantic_hbar: float = 1.0
    dissipation_rate: float = 0.10

    fixed_point_tolerance: float = 1.0e-6
    equilibrium_tolerance: float = 1.0e-8
    max_iterations: int = 128
    logical_side: int = 1_000_000

    def __post_init__(self) -> None:
        for name in ("side", "max_active_cells", "attention_min_cells", "max_iterations", "logical_side"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        for name in (
            "value_min",
            "value_max",
            "attention_keep_ratio",
            "latent_bound",
            "memory_retention",
            "memory_constraint_gain",
            "theta_constraint",
            "state_gain",
            "max_state_delta",
            "semantic_hbar",
            "dissipation_rate",
            "fixed_point_tolerance",
            "equilibrium_tolerance",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if not 0.0 < self.attention_keep_ratio <= 1.0:
            raise ValueError("attention_keep_ratio must be in (0, 1]")
        if self.latent_bound <= 0.0:
            raise ValueError("latent_bound must be positive")
        if not 0.0 <= self.memory_retention < 1.0:
            raise ValueError("memory_retention must be in [0, 1)")
        if self.memory_constraint_gain < 0.0:
            raise ValueError("memory_constraint_gain must be non-negative")
        if self.theta_constraint < 0.0:
            raise ValueError("theta_constraint must be non-negative")
        if not 0.0 < self.state_gain <= 1.0:
            raise ValueError("state_gain must be in (0, 1]")
        if self.max_state_delta <= 0.0:
            raise ValueError("max_state_delta must be positive")
        if self.semantic_hbar < 0.0:
            raise ValueError("semantic_hbar must be non-negative")
        if not 0.0 <= self.dissipation_rate <= 1.0:
            raise ValueError("dissipation_rate must be in [0, 1]")
        if self.fixed_point_tolerance < 0.0:
            raise ValueError("fixed_point_tolerance must be non-negative")
        if self.equilibrium_tolerance < 0.0:
            raise ValueError("equilibrium_tolerance must be non-negative")


@dataclass(frozen=True)
class ConsiderationReport:
    iteration: int
    input_entropy: float
    attended_entropy: float
    entropy_contraction: float
    active_cells_before_attention: int
    active_cells_after_attention: int
    latent_cells: int
    description_rms: float
    memory_constraint_rms: float
    state_delta_rms: float
    memory_delta_rms: float
    dissipated_rms: float
    fixed_point_residual: float
    h_mmm: float
    h_mmm_delta: float
    converged: bool
    state_hash: str
    journal_hash: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DMvOmegaXiConsiderationLoop:
    """Sparse deterministic consideration loop.

    One step operationalizes the stack as::

        Psi_t
          -> U_attn(t)[Psi_t]                  salience contraction
          -> Phi_in                            structural description
          -> Lambda^-1(div_Theta Omega_t)      bounded memory correction
          -> Theta_in                          bounded state projection
          -> Gamma_in                          residual dissipation
          -> Psi_{t+1}
          -> Omega_{t+1}                       recurrent memory update

    The accepted fixed point is an internal numerical equilibrium:

        Psi* = F_DM(Psi*, Omega*)
        Omega* = Psi*

    subject to a bounded residual and a stabilized H_MMM Lyapunov-like energy.
    """

    LAW_ID = "DM-vOmegaXi+"
    OPERATOR_STACK = (
        "U_attn(t)",
        "Psi_t",
        "Phi_in",
        "Lambda_in^-1(div_Theta Omega_t)",
        "Theta_in",
        "Gamma_in",
        "Omega_t",
        "H_MMM",
    )

    def __init__(
        self,
        config: DMvOmegaXiConsiderationConfig | None = None,
        *,
        journal_path: str | Path | None = None,
    ) -> None:
        self.config = config or DMvOmegaXiConsiderationConfig()
        self.parser = SparseParser3D(
            side=self.config.side,
            max_active_cells=self.config.max_active_cells,
            value_min=self.config.value_min,
            value_max=self.config.value_max,
            prune_epsilon=self.config.policy.prune_epsilon,
        )
        self.codec = SparseBlockCodec3D(self.config.policy)
        self.journal = HashChainJournal(journal_path)
        self._state: SparseField = {}
        self._memory: SparseField = {}
        self._iteration = 0
        self._loaded = False
        self._last_h_mmm: float | None = None
        self.reports: list[ConsiderationReport] = []

    def load(self, source: Mapping[Coordinate, float]) -> SparseField:
        state = self.parser.parse(source)
        self._state = dict(state)
        self._memory = dict(state)
        self._iteration = 0
        self._last_h_mmm = None
        self.reports.clear()
        self._loaded = True
        return self.snapshot()

    def snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._state)

    def memory_snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._memory)

    def attention(self, field: Mapping[Coordinate, float]) -> SparseField:
        """U_attn(t): retain the most salient active coordinates.

        Salience is absolute amplitude with deterministic coordinate tie-breaking.
        The operator contracts support without allocating the logical dense domain.
        """
        if not field:
            return {}
        count = max(
            self.config.attention_min_cells,
            int(math.ceil(len(field) * self.config.attention_keep_ratio)),
        )
        count = min(count, len(field))
        ranked = sorted(field.items(), key=lambda item: (-abs(float(item[1])), item[0]))
        return {coordinate: float(value) for coordinate, value in ranked[:count]}

    def phi_in(self, field: Mapping[Coordinate, float]) -> tuple[SparseLatent3D, SparseField]:
        """Phi_in: compress and reconstruct the current structural description."""
        latent = self.codec.encode(field)
        bound = self.config.latent_bound
        bounded_cells = tuple(
            replace(cell, value=max(-bound, min(bound, float(cell.value))))
            for cell in latent.cells
        )
        bounded = replace(latent, cells=bounded_cells)
        support = tuple(sorted(field))
        decoded = self.codec.decode(bounded, support)
        return bounded, decoded

    def memory_constraint(self, support: set[Coordinate]) -> SparseField:
        """Lambda^-1(div_Theta Omega_t): bounded discrete memory correction.

        The divergence term is represented by a six-neighbour graph Laplacian over
        materialized memory support.  Lambda^-1 is a saturating inverse-boundary
        map, preventing the correction from growing linearly without bound.
        """
        gain = self.config.memory_constraint_gain
        theta = self.config.theta_constraint
        correction: SparseField = {}
        for coordinate in sorted(support):
            center = float(self._memory.get(coordinate, 0.0))
            x, y, z = coordinate
            neighbours = (
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            )
            present = [float(self._memory[n]) for n in neighbours if n in self._memory]
            if not present or gain == 0.0:
                continue
            divergence = sum(value - center for value in present) / len(present)
            bounded = gain * divergence / (1.0 + theta * abs(divergence))
            if bounded != 0.0:
                correction[coordinate] = bounded
        return correction

    def theta_project(
        self,
        current: Mapping[Coordinate, float],
        target: Mapping[Coordinate, float],
    ) -> SparseField:
        """Theta_in: bounded state-space projection toward the current target."""
        gain = self.config.state_gain
        cap = self.config.max_state_delta
        eps = self.config.policy.prune_epsilon
        result: SparseField = {}
        for coordinate in sorted(set(current) | set(target)):
            value = float(current.get(coordinate, 0.0))
            desired = float(target.get(coordinate, 0.0))
            delta = max(-cap, min(cap, gain * (desired - value)))
            projected = max(self.config.value_min, min(self.config.value_max, value + delta))
            if abs(projected) > eps:
                result[coordinate] = projected
        return result

    def gamma_dissipate(
        self,
        candidate: Mapping[Coordinate, float],
        description: Mapping[Coordinate, float],
    ) -> tuple[SparseField, float]:
        """Gamma_in: damp unexplained residual while preserving described signal.

        For a real-valued runtime the symbolic ``-i*hbar*Gamma`` term is mapped to
        a contraction of r = candidate - description:

            r' = (1 - hbar_semantic * gamma) r

        with the contraction factor clamped to [0, 1].
        """
        damping = min(1.0, max(0.0, self.config.semantic_hbar * self.config.dissipation_rate))
        eps = self.config.policy.prune_epsilon
        result: SparseField = {}
        removed_sq = 0.0
        support = set(candidate) | set(description)
        for coordinate in sorted(support):
            value = float(candidate.get(coordinate, 0.0))
            base = float(description.get(coordinate, 0.0))
            residual = value - base
            removed = damping * residual
            stabilized = value - removed
            removed_sq += removed * removed
            if abs(stabilized) > eps:
                result[coordinate] = stabilized
        removed_rms = math.sqrt(removed_sq / len(support)) if support else 0.0
        return result, removed_rms

    def omega_update(self, state: Mapping[Coordinate, float]) -> SparseField:
        retain = self.config.memory_retention
        inject = 1.0 - retain
        result: SparseField = {}
        for coordinate in sorted(set(self._memory) | set(state)):
            value = retain * float(self._memory.get(coordinate, 0.0)) + inject * float(
                state.get(coordinate, 0.0)
            )
            if abs(value) > self.config.policy.prune_epsilon:
                result[coordinate] = value
        return result

    def step(self) -> ConsiderationReport:
        self._require_loaded()
        current = dict(self._state)
        before_entropy = self._entropy(current)

        attended = self.attention(current)
        attended_entropy = self._entropy(attended)
        latent, description = self.phi_in(attended)

        support = set(current) | set(description) | set(self._memory)
        correction = self.memory_constraint(support)
        target = {
            coordinate: float(description.get(coordinate, 0.0)) + float(correction.get(coordinate, 0.0))
            for coordinate in sorted(set(description) | set(correction))
        }

        projected = self.theta_project(current, target)
        candidate, dissipated_rms = self.gamma_dissipate(projected, description)
        next_memory = self.omega_update(candidate)

        description_rms = self._rms(candidate, description)
        memory_constraint_rms = self._rms(correction, {})
        state_delta_rms = self._rms(candidate, current)
        memory_delta_rms = self._rms(next_memory, self._memory)
        residual = max(state_delta_rms, memory_delta_rms)

        # A non-negative Lyapunov-like equilibrium energy for the executable model.
        h_mmm = (
            state_delta_rms * state_delta_rms
            + memory_delta_rms * memory_delta_rms
            + description_rms * description_rms
        )
        h_delta = 0.0 if self._last_h_mmm is None else abs(h_mmm - self._last_h_mmm)

        self._state = candidate
        self._memory = next_memory
        self._iteration += 1
        self._last_h_mmm = h_mmm

        converged = bool(
            residual <= self.config.fixed_point_tolerance
            and h_delta <= self.config.equilibrium_tolerance
        )

        provisional = ConsiderationReport(
            iteration=self._iteration,
            input_entropy=before_entropy,
            attended_entropy=attended_entropy,
            entropy_contraction=max(0.0, before_entropy - attended_entropy),
            active_cells_before_attention=len(current),
            active_cells_after_attention=len(attended),
            latent_cells=latent.latent_cells,
            description_rms=description_rms,
            memory_constraint_rms=memory_constraint_rms,
            state_delta_rms=state_delta_rms,
            memory_delta_rms=memory_delta_rms,
            dissipated_rms=dissipated_rms,
            fixed_point_residual=residual,
            h_mmm=h_mmm,
            h_mmm_delta=h_delta,
            converged=converged,
            state_hash=self._state_hash(candidate),
        )
        record = provisional.as_dict()
        record.pop("journal_hash", None)
        digest = self.journal.append(record)
        report = replace(provisional, journal_hash=digest)
        self.reports.append(report)
        return report

    def run_until_fixed_point(self, max_iterations: int | None = None) -> tuple[ConsiderationReport, ...]:
        self._require_loaded()
        limit = self.config.max_iterations if max_iterations is None else max_iterations
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("max_iterations must be a positive integer")
        for _ in range(limit):
            report = self.step()
            if report.converged:
                break
        return tuple(self.reports)

    def status(self) -> dict[str, object]:
        self._require_loaded()
        latest = self.reports[-1] if self.reports else None
        return {
            "law": self.LAW_ID,
            "operator_stack": list(self.OPERATOR_STACK),
            "fixed_point_equation": "Psi* = F_DM(Psi*, Omega*), Omega* = Psi*",
            "logical_domain": f"{self.config.logical_side}^3",
            "logical_voxels": str(self.config.logical_side**3),
            "materialization": "sparse-active-support-only",
            "iteration": self._iteration,
            "active_cells": len(self._state),
            "fixed_point_residual": latest.fixed_point_residual if latest else None,
            "h_mmm": latest.h_mmm if latest else None,
            "h_mmm_delta": latest.h_mmm_delta if latest else None,
            "entropy_contraction": latest.entropy_contraction if latest else None,
            "converged": latest.converged if latest else False,
            "journal_head": self.journal.head,
            "journal_valid": self.journal.verify(),
        }

    @staticmethod
    def _entropy(field: Mapping[Coordinate, float]) -> float:
        weights = [abs(float(value)) for value in field.values() if abs(float(value)) > 0.0]
        total = sum(weights)
        if total <= 0.0:
            return 0.0
        entropy = 0.0
        for weight in weights:
            p = weight / total
            entropy -= p * math.log(p)
        return entropy

    @staticmethod
    def _rms(left: Mapping[Coordinate, float], right: Mapping[Coordinate, float]) -> float:
        support = set(left) | set(right)
        if not support:
            return 0.0
        mse = sum(
            (float(left.get(coordinate, 0.0)) - float(right.get(coordinate, 0.0))) ** 2
            for coordinate in support
        ) / len(support)
        return math.sqrt(mse)

    @staticmethod
    def _state_hash(field: Mapping[Coordinate, float]) -> str:
        canonical = [
            [coordinate[0], coordinate[1], coordinate[2], float(field[coordinate])]
            for coordinate in sorted(field)
        ]
        payload = json.dumps(canonical, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("load a sparse 3D field before consideration-loop execution")
