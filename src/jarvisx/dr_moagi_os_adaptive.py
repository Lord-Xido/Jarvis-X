"""OS integration bridge for persistent Deep Distiller adaptation.

The standalone DM-DD runtime owns its own authoritative state.  The OS needs to
stage a complete candidate cycle and only promote the adaptive state after every
later OS gate (fixed-point validation, resource bounds, transport verification)
has passed.  This bridge adds checkpoint/staging restore semantics without
changing the standalone Deep Distiller contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .dr_moagi_deep_distiller import (
    DeepDistiller,
    DeepDistillerCandidate,
    DeepDistillerTheta,
)
from .dr_moagi_field_runtime import Coordinate, SparseField


@dataclass(frozen=True)
class AdaptiveSnapshot:
    state: SparseField
    omega: SparseField
    theta: DeepDistillerTheta
    iteration: int


class OSDeepDistiller(DeepDistiller):
    """Deep Distiller with explicit OS staging/checkpoint restore semantics."""

    @property
    def iteration(self) -> int:
        return self._iteration

    def adaptive_snapshot(self) -> AdaptiveSnapshot:
        self._require_loaded()
        return AdaptiveSnapshot(
            state=dict(self._state),
            omega=dict(self._omega),
            theta=self._theta,
            iteration=self._iteration,
        )

    def restore_adaptive_state(
        self,
        state: Mapping[Coordinate, float],
        *,
        omega: Mapping[Coordinate, float] | None = None,
        theta: DeepDistillerTheta | None = None,
        iteration: int = 0,
    ) -> SparseField:
        """Restore/stage a complete adaptive state through ``Pi_Lambda``.

        This operation is intended for OS checkpoints and isolated staging
        instances.  It never bypasses state/Theta/resource validation.
        """
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("iteration must be a non-negative integer")
        parsed = self.parser.parse(state)
        selected_theta = theta or self._theta
        self._validate_theta(selected_theta)
        parsed_omega = self._validate_omega(omega or {})
        candidate = DeepDistillerCandidate(
            state=dict(parsed),
            omega=dict(parsed_omega),
            theta=selected_theta,
            latent_cells=0,
        )
        passed, reason = self.pi_lambda(candidate)
        if not passed:
            raise ValueError(f"adaptive state rejected by Pi_Lambda: {reason}")
        self._state = dict(candidate.state)
        self._omega = dict(candidate.omega)
        self._theta = candidate.theta
        self._iteration = iteration
        self.reports.clear()
        self._loaded = True
        return self.snapshot()

    def _validate_omega(self, source: Mapping[Coordinate, float]) -> SparseField:
        if len(source) > self.config.max_active_cells:
            raise ValueError("Omega active-cell budget exceeded")
        result: SparseField = {}
        side = self.config.logical_side
        for coordinate, raw_value in source.items():
            if (
                not isinstance(coordinate, tuple)
                or len(coordinate) != 3
                or any(
                    isinstance(axis, bool) or not isinstance(axis, int)
                    for axis in coordinate
                )
            ):
                raise TypeError("Omega coordinates must be integer triples")
            if any(axis < 0 or axis >= side for axis in coordinate):
                raise ValueError("Omega coordinate outside logical lattice")
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise TypeError("Omega values must be numeric")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("Omega values must be finite")
            result[coordinate] = value
        return result


__all__ = ["AdaptiveSnapshot", "OSDeepDistiller"]
