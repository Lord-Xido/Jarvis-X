"""Executable fixed-point contract for the locked Dr Moagi DM-vOmegaXi+ law.

The module translates the symbolic Psi--Phi--Lambda--Omega--Theta stack into a
bounded sparse reference runtime.  The fixed point is an *internal operator*
fixed point, H* = F_DM(H*).  It is deliberately not interpreted as identity
between an internal model and external reality: ``semantic_floor`` keeps a
strictly positive map--territory uncertainty floor.

The 10^27 TB hyper-volume is represented only as logical metadata.  This module
never allocates that volume; it materializes active sparse support only.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping

from .dr_moagi_autoexec import (
    AutoExecPolicy,
    HashChainJournal,
    SparseBlockCodec3D,
    SparseLatent3D,
    SparseParser3D,
)
from .dr_moagi_field_runtime import Coordinate, SparseField

ThetaGate = Callable[[Mapping[Coordinate, float]], bool]


@dataclass(frozen=True)
class DMvOmegaXiFixedPointConfig:
    """Bounded executable contract for DM-vOmegaXi+.

    ``logical_hypervolume_tb`` is address-space metadata, not resident memory.
    ``omega_memory`` is the historical retention coefficient in Omega_t.
    ``theta_gain`` and ``theta_max_delta`` bound each inward stabilization step.
    """

    side: int = 64
    max_active_cells: int = 50_000
    value_min: float = -1.0
    value_max: float = 1.0
    policy: AutoExecPolicy = field(default_factory=AutoExecPolicy)
    latent_bound: float = 1.0
    omega_memory: float = 0.5
    theta_gain: float = 0.5
    theta_max_delta: float = 0.25
    fixed_point_tolerance: float = 1.0e-6
    semantic_floor: float = 1.0e-12
    max_iterations: int = 64
    logical_hypervolume_tb: int = 10**27

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        if (
            isinstance(self.max_active_cells, bool)
            or not isinstance(self.max_active_cells, int)
            or self.max_active_cells <= 0
        ):
            raise ValueError("max_active_cells must be a positive integer")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, int)
            or self.max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        if self.logical_hypervolume_tb <= 0:
            raise ValueError("logical_hypervolume_tb must be positive")

        for name in (
            "value_min",
            "value_max",
            "latent_bound",
            "omega_memory",
            "theta_gain",
            "theta_max_delta",
            "fixed_point_tolerance",
            "semantic_floor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if self.latent_bound <= 0.0:
            raise ValueError("latent_bound must be positive")
        if not 0.0 <= self.omega_memory < 1.0:
            raise ValueError("omega_memory must be in [0, 1)")
        if not 0.0 < self.theta_gain <= 1.0:
            raise ValueError("theta_gain must be in (0, 1]")
        if self.theta_max_delta <= 0.0:
            raise ValueError("theta_max_delta must be positive")
        if self.fixed_point_tolerance < 0.0:
            raise ValueError("fixed_point_tolerance must be non-negative")
        if self.semantic_floor <= 0.0:
            raise ValueError("semantic_floor must be strictly positive")


@dataclass(frozen=True)
class FixedPointReport:
    iteration: int
    committed: bool
    active_cells: int
    latent_cells: int
    reconstruction_rms: float
    memory_rms: float
    theta_rms: float
    fixed_point_residual: float
    semantic_gap: float
    converged: bool
    theta_gate_passed: bool
    rejection_reason: str | None
    state_hash: str
    journal_hash: str = ""

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DMvOmegaXiFixedPointEngine:
    """Sparse inward-folded fixed-point engine.

    One iteration implements the measurable runtime interpretation:

        Psi_t
          -> Phi(Psi_t)                  spatial description / encoding
          -> Lambda^-1(Phi_t)             bounded latent projection
          -> D(Phi_t)                     sparse reconstruction
          -> Omega_t(D_t)                 recurrent inward memory fold
          -> Theta(Psi_t, Omega_{t+1})     bounded stability projection
          -> H_{t+1}

    H* is accepted when the authoritative state, decoded description, and
    recurrent memory agree within ``fixed_point_tolerance``.  The separate
    semantic gap gamma never falls below ``semantic_floor``.
    """

    LAW_ID = "DM-vOmegaXi+"
    OPERATOR_STACK = ("Psi", "Phi", "Lambda^-1", "Omega", "Theta")

    def __init__(
        self,
        config: DMvOmegaXiFixedPointConfig | None = None,
        *,
        theta_gate: ThetaGate | None = None,
        journal_path: str | Path | None = None,
    ) -> None:
        self.config = config or DMvOmegaXiFixedPointConfig()
        self.theta_gate = theta_gate
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
        self.reports: list[FixedPointReport] = []

    def load(self, source: Mapping[Coordinate, float]) -> SparseField:
        state = self.parser.parse(source)
        self._state = dict(state)
        self._memory = dict(state)
        self._iteration = 0
        self.reports.clear()
        self._loaded = True
        return self.snapshot()

    def snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._state)

    def memory_snapshot(self) -> SparseField:
        self._require_loaded()
        return dict(self._memory)

    def phi(self, field: Mapping[Coordinate, float]) -> SparseLatent3D:
        """Phi: spatial description/compression into the sparse latent field."""
        return self.codec.encode(field)

    def lambda_inverse(self, latent: SparseLatent3D) -> SparseLatent3D:
        """Lambda^-1: project latent amplitudes into the bounded bottleneck."""
        bound = self.config.latent_bound
        cells = tuple(
            replace(cell, value=max(-bound, min(bound, float(cell.value))))
            for cell in latent.cells
        )
        return replace(latent, cells=cells)

    def omega_fold(
        self,
        decoded: Mapping[Coordinate, float],
        support: tuple[Coordinate, ...],
    ) -> SparseField:
        """Omega_t: fold historical state back into local voxel coordinates."""
        retain = self.config.omega_memory
        inject = 1.0 - retain
        return {
            coordinate: retain * self._memory.get(coordinate, self._state.get(coordinate, 0.0))
            + inject * float(decoded.get(coordinate, 0.0))
            for coordinate in support
        }

    def theta_project(
        self,
        current: Mapping[Coordinate, float],
        folded: Mapping[Coordinate, float],
    ) -> SparseField:
        """Theta: bounded numerical/stability projection over active support."""
        result: SparseField = {}
        gain = self.config.theta_gain
        cap = self.config.theta_max_delta
        eps = self.config.policy.prune_epsilon
        for coordinate in sorted(set(current) | set(folded)):
            value = float(current.get(coordinate, 0.0))
            target = float(folded.get(coordinate, 0.0))
            delta = gain * (target - value)
            delta = max(-cap, min(cap, delta))
            projected = max(
                self.config.value_min,
                min(self.config.value_max, value + delta),
            )
            if abs(projected) > eps:
                result[coordinate] = projected
        return result

    def step(self) -> FixedPointReport:
        self._require_loaded()
        current = dict(self._state)
        support = tuple(sorted(current))

        latent = self.phi(current)
        bounded_latent = self.lambda_inverse(latent)
        decoded = self.codec.decode(bounded_latent, support)
        folded = self.omega_fold(decoded, support)
        candidate = self.theta_project(current, folded)

        reconstruction_rms = self._rms(current, decoded)
        memory_rms = self._rms(candidate, folded)
        theta_rms = self._rms(candidate, current)
        representation_rms = self._rms(candidate, decoded)
        residual = max(theta_rms, memory_rms, representation_rms)
        semantic_gap = max(self.config.semantic_floor, reconstruction_rms)

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

        committed = rejection_reason is None
        if committed:
            self._state = candidate
            self._memory = folded
            self._iteration += 1

        converged = bool(committed and residual <= self.config.fixed_point_tolerance)
        authoritative = self._state if committed else current
        provisional = FixedPointReport(
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
        )
        record = provisional.as_dict()
        record.pop("journal_hash", None)
        digest = self.journal.append(record)
        report = replace(provisional, journal_hash=digest)
        self.reports.append(report)
        return report

    def run_until_fixed_point(self, max_iterations: int | None = None) -> tuple[FixedPointReport, ...]:
        self._require_loaded()
        limit = self.config.max_iterations if max_iterations is None else max_iterations
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("max_iterations must be a positive integer")
        for _ in range(limit):
            report = self.step()
            if report.converged or not report.committed:
                break
        return tuple(self.reports)

    def status(self) -> dict[str, object]:
        self._require_loaded()
        latent = self.lambda_inverse(self.phi(self._state))
        latest = self.reports[-1] if self.reports else None
        return {
            "law": self.LAW_ID,
            "locked": True,
            "operator_stack": list(self.OPERATOR_STACK),
            "fixed_point_equation": "H* = F_DM(H*)",
            "iteration": self._iteration,
            "active_cells": len(self._state),
            "latent_cells": latent.latent_cells,
            "logical_hypervolume_tb": str(self.config.logical_hypervolume_tb),
            "materialization": "sparse-active-support-only",
            "semantic_floor": self.config.semantic_floor,
            "semantic_gap": latest.semantic_gap if latest else None,
            "fixed_point_residual": latest.fixed_point_residual if latest else None,
            "converged": latest.converged if latest else False,
            "state_hash": self._state_hash(self._state),
            "journal_head": self.journal.head,
            "journal_valid": self.journal.verify(),
        }

    @staticmethod
    def _rms(
        left: Mapping[Coordinate, float],
        right: Mapping[Coordinate, float],
    ) -> float:
        support = set(left) | set(right)
        if not support:
            return 0.0
        mse = sum(
            (float(left.get(coordinate, 0.0)) - float(right.get(coordinate, 0.0))) ** 2
            for coordinate in support
        ) / len(support)
        return math.sqrt(mse)

    @staticmethod
    def _finite(field: Mapping[Coordinate, float]) -> bool:
        return all(math.isfinite(float(value)) for value in field.values())

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
            raise RuntimeError("load a sparse 3D field before fixed-point execution")
