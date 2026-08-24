"""Bounded sparse runtime for the Dr Moagi 3D field equation.

This module operationalizes the same-space field law

    dPsi/dt =
        -alpha (I - D o E)[Psi]
        + lambda * Laplacian((I - D o E)[Psi])
        + eta * G_moagi * Psi

on a sparse logical 3D lattice. It is intentionally independent of the
canonical 64-bit VM: candidate field transitions are computed in a research
layer and become authoritative only after projection and optional validation.
Every completed candidate decision also emits the same deterministic
control-plane receipt used by the canonical VM.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from .control_plane import OmegaEvidenceChain, StateEnvelope

Coordinate = tuple[int, int, int]
SparseField = dict[Coordinate, float]
Validator = Callable[[Mapping[Coordinate, float], "FieldStepMetrics"], bool]


class FieldCodec(Protocol):
    """Codec boundary used by the field runtime.

    ``decode`` receives the exact support that must be reconstructed. This
    makes sparse materialization explicit and prevents a codec from silently
    allocating the full logical lattice.
    """

    def encode(self, field: Mapping[Coordinate, float]) -> Any:
        ...

    def decode(
        self, latent: Any, support: Sequence[Coordinate]
    ) -> Mapping[Coordinate, float]:
        ...


class IdentityFieldCodec:
    """Deterministic reference codec used for conformance tests."""

    def encode(self, field: Mapping[Coordinate, float]) -> dict[Coordinate, float]:
        return dict(field)

    def decode(
        self,
        latent: Mapping[Coordinate, float],
        support: Sequence[Coordinate],
    ) -> Mapping[Coordinate, float]:
        return {coordinate: float(latent.get(coordinate, 0.0)) for coordinate in support}


@dataclass(frozen=True)
class DrMoagiFieldConfig:
    """Numerical and resource contract for one field-runtime run."""

    side: int = 1000
    alpha: float = 1.0
    lambda_residual: float = 1.0
    eta: float = 0.1
    dt: float = 0.025
    value_min: float = -1.0
    value_max: float = 1.0
    max_active_cells: int = 100_000
    expand_halo: bool = True
    prune_epsilon: float = 0.0
    enforce_conservative_step_bound: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side <= 0:
            raise ValueError("side must be a positive integer")
        if (
            isinstance(self.max_active_cells, bool)
            or not isinstance(self.max_active_cells, int)
            or self.max_active_cells <= 0
        ):
            raise ValueError("max_active_cells must be a positive integer")

        for name in (
            "alpha",
            "lambda_residual",
            "eta",
            "dt",
            "value_min",
            "value_max",
            "prune_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.lambda_residual < 0.0:
            raise ValueError("lambda_residual must be non-negative")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be smaller than value_max")
        if self.prune_epsilon < 0.0:
            raise ValueError("prune_epsilon must be non-negative")

        # For a non-expansive codec, ||I-A|| <= 2, ||Delta_6|| <= 12 and
        # ||G_moagi|| <= 2. This is a conservative explicit-Euler budget,
        # not a proof of stability for arbitrary learned codecs.
        if self.enforce_conservative_step_bound and self.stability_load > 1.0:
            raise ValueError(
                "dt exceeds the conservative explicit-step budget; "
                "reduce dt or disable the guard explicitly"
            )

    @property
    def stability_load(self) -> float:
        return self.dt * (
            2.0 * self.alpha
            + 24.0 * self.lambda_residual
            + 2.0 * abs(self.eta)
        )


@dataclass(frozen=True)
class FieldStepMetrics:
    """Deterministic telemetry for one attempted transaction."""

    cycle: int
    support_cells: int
    active_cells_before: int
    active_cells_after: int
    reconstruction_mse: float
    anchor_mse: float
    max_abs_residual: float
    max_abs_rhs: float
    committed: bool
    rejection_reason: str | None = None


class DrMoagiFieldRuntime:
    """Sparse, transactional reference implementation of the field equation."""

    _NEIGHBOURS: tuple[Coordinate, ...] = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )

    def __init__(
        self,
        codec: FieldCodec,
        config: DrMoagiFieldConfig | None = None,
    ) -> None:
        self.codec = codec
        self.config = config or DrMoagiFieldConfig()
        self._state: SparseField = {}
        self._anchor: SparseField = {}
        self._cycle = 0
        self.control_plane = OmegaEvidenceChain()

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def virtual_cell_count(self) -> int:
        return self.config.side**3

    @property
    def active_cell_count(self) -> int:
        return len(self._state)

    def snapshot(self) -> SparseField:
        return dict(self._state)

    def anchor_snapshot(self) -> SparseField:
        return dict(self._anchor)

    def load(self, field: Mapping[Coordinate, float]) -> None:
        """Start a new run and freeze its immutable drift-detection anchor."""

        projected = self._project(dict(field))
        if len(projected) > self.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")
        self._state = projected
        self._anchor = dict(projected)
        self._cycle = 0
        self.control_plane.reset()

    def step(self, validator: Validator | None = None) -> FieldStepMetrics:
        """Attempt one atomic field transition.

        All terms are evaluated from one frozen snapshot. Candidate state is
        projected first and becomes authoritative only if the optional
        validator accepts it. Commit and rollback decisions are emitted into
        the shared Omega evidence chain before authoritative state is changed.
        """

        snapshot = dict(self._state)
        before = self._state_envelope(snapshot, authoritative=True)
        support = self._support(snapshot)
        cycle = self._cycle + 1

        latent = self.codec.encode(snapshot)
        reconstruction_raw = self.codec.decode(latent, support)
        reconstruction = self._validated_reconstruction(reconstruction_raw, support)

        residual = {
            coordinate: self._value(snapshot, coordinate)
            - self._value(reconstruction, coordinate)
            for coordinate in support
        }
        rhs: SparseField = {}
        for coordinate in support:
            closure = -self.config.alpha * residual[coordinate]
            holographic = self.config.lambda_residual * self._laplacian(
                residual, coordinate
            )
            permeation = self.config.eta * self._glyph(snapshot, coordinate)
            rhs[coordinate] = closure + holographic + permeation

        candidate = self._project(
            {
                coordinate: self._value(snapshot, coordinate)
                + self.config.dt * rhs[coordinate]
                for coordinate in support
            }
        )
        candidate_envelope = self._state_envelope(candidate, authoritative=False)

        if len(candidate) > self.config.max_active_cells:
            metrics = self._metrics(
                cycle,
                support,
                snapshot,
                candidate,
                reconstruction,
                residual,
                rhs,
                committed=False,
                rejection_reason="active-cell budget exceeded",
            )
            self._record_decision(
                before=before,
                candidate=candidate_envelope,
                after=self._state_envelope(snapshot, authoritative=True),
                metrics=metrics,
            )
            self._cycle = cycle
            return metrics

        provisional = self._metrics(
            cycle,
            support,
            snapshot,
            candidate,
            reconstruction,
            residual,
            rhs,
            committed=False,
        )
        if validator is not None and not bool(validator(candidate, provisional)):
            metrics = replace(provisional, rejection_reason="validator rejected candidate")
            self._record_decision(
                before=before,
                candidate=candidate_envelope,
                after=self._state_envelope(snapshot, authoritative=True),
                metrics=metrics,
            )
            self._cycle = cycle
            return metrics

        metrics = self._metrics(
            cycle,
            support,
            snapshot,
            candidate,
            reconstruction,
            residual,
            rhs,
            committed=True,
        )
        self._record_decision(
            before=before,
            candidate=candidate_envelope,
            after=self._state_envelope(candidate, authoritative=True),
            metrics=metrics,
        )
        self._state = candidate
        self._cycle = cycle
        return metrics

    def _state_payload(self, field: Mapping[Coordinate, float]) -> dict[str, object]:
        cells = [
            [coordinate[0], coordinate[1], coordinate[2], float(value)]
            for coordinate, value in sorted(field.items(), key=lambda item: self._linear_address(item[0]))
        ]
        return {
            "logical_side": self.config.side,
            "active_cells": len(cells),
            "cells": cells,
        }

    def _state_envelope(
        self, field: Mapping[Coordinate, float], *, authoritative: bool
    ) -> StateEnvelope:
        return StateEnvelope.from_payload(
            state_type="jarvisx.dr-moagi-field",
            state_version=1,
            dimensions=(self.config.side, self.config.side, self.config.side),
            payload=self._state_payload(field),
            authoritative=authoritative,
        )

    def _record_decision(
        self,
        *,
        before: StateEnvelope,
        candidate: StateEnvelope,
        after: StateEnvelope,
        metrics: FieldStepMetrics,
    ) -> None:
        self.control_plane.append(
            subsystem="dr-moagi-field-runtime",
            operation="field-step",
            decision="commit" if metrics.committed else "rollback",
            reason=metrics.rejection_reason,
            before=before,
            candidate=candidate,
            after=after,
            metrics=asdict(metrics),
        )

    def _support(self, field: Mapping[Coordinate, float]) -> tuple[Coordinate, ...]:
        support = set(field)
        if self.config.expand_halo:
            for coordinate in tuple(support):
                for neighbour in self._iter_neighbours(coordinate):
                    support.add(neighbour)
        if len(support) > self.config.max_active_cells:
            raise RuntimeError("support-closure budget exceeded")
        return tuple(sorted(support, key=self._linear_address))

    def _validated_reconstruction(
        self,
        reconstruction: Mapping[Coordinate, float],
        support: Sequence[Coordinate],
    ) -> SparseField:
        allowed = set(support)
        result: SparseField = {}
        for coordinate, value in reconstruction.items():
            coordinate = self._validate_coordinate(coordinate)
            if coordinate not in allowed:
                raise ValueError("decoder returned a coordinate outside requested support")
            result[coordinate] = self._finite(value, "decoder value")
        return result

    def _project(self, field: Mapping[Coordinate, float]) -> SparseField:
        projected: SparseField = {}
        for coordinate, raw_value in field.items():
            coordinate = self._validate_coordinate(coordinate)
            value = self._finite(raw_value, "field value")
            value = min(self.config.value_max, max(self.config.value_min, value))
            if abs(value) <= self.config.prune_epsilon:
                continue
            projected[coordinate] = value
        return projected

    def _laplacian(
        self, field: Mapping[Coordinate, float], coordinate: Coordinate
    ) -> float:
        center = self._value(field, coordinate)
        neighbour_sum = sum(self._value(field, n) for n in self._iter_neighbours(coordinate))
        return neighbour_sum - 6.0 * center

    def _glyph(self, field: Mapping[Coordinate, float], coordinate: Coordinate) -> float:
        # Canonical glyph kernel: +1 at the centre and -1/6 on the six face neighbours.
        center = self._value(field, coordinate)
        neighbour_sum = sum(self._value(field, n) for n in self._iter_neighbours(coordinate))
        return center - neighbour_sum / 6.0

    def _iter_neighbours(self, coordinate: Coordinate) -> tuple[Coordinate, ...]:
        x, y, z = coordinate
        neighbours: list[Coordinate] = []
        for dx, dy, dz in self._NEIGHBOURS:
            candidate = x + dx, y + dy, z + dz
            if self._inside(candidate):
                neighbours.append(candidate)
        return tuple(neighbours)

    def _value(self, field: Mapping[Coordinate, float], coordinate: Coordinate) -> float:
        if not self._inside(coordinate):
            return 0.0
        return float(field.get(coordinate, 0.0))

    def _inside(self, coordinate: Coordinate) -> bool:
        x, y, z = coordinate
        side = self.config.side
        return 0 <= x < side and 0 <= y < side and 0 <= z < side

    def _validate_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) for v in coordinate)
        ):
            raise TypeError("coordinates must be integer (x, y, z) tuples")
        if not self._inside(coordinate):
            raise ValueError("coordinate is outside the logical lattice")
        return coordinate

    def _linear_address(self, coordinate: Coordinate) -> int:
        x, y, z = coordinate
        side = self.config.side
        return x + side * (y + side * z)

    @staticmethod
    def _finite(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @staticmethod
    def _mse(a: Mapping[Coordinate, float], b: Mapping[Coordinate, float]) -> float:
        support = set(a) | set(b)
        if not support:
            return 0.0
        squared_error = sum(
            (float(a.get(c, 0.0)) - float(b.get(c, 0.0))) ** 2 for c in support
        )
        return squared_error / len(support)

    def _metrics(
        self,
        cycle: int,
        support: Sequence[Coordinate],
        snapshot: Mapping[Coordinate, float],
        candidate: Mapping[Coordinate, float],
        reconstruction: Mapping[Coordinate, float],
        residual: Mapping[Coordinate, float],
        rhs: Mapping[Coordinate, float],
        *,
        committed: bool,
        rejection_reason: str | None = None,
    ) -> FieldStepMetrics:
        committed_state = candidate if committed else snapshot
        return FieldStepMetrics(
            cycle=cycle,
            support_cells=len(support),
            active_cells_before=len(snapshot),
            active_cells_after=len(committed_state),
            reconstruction_mse=self._mse(snapshot, reconstruction),
            anchor_mse=self._mse(self._anchor, committed_state),
            max_abs_residual=max((abs(value) for value in residual.values()), default=0.0),
            max_abs_rhs=max((abs(value) for value in rhs.values()), default=0.0),
            committed=committed,
            rejection_reason=rejection_reason,
        )
