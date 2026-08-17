"""Reference implementation of the canonical Dr Moagi 3D state recurrence.

The research-layer equation is

    Xi[t+1] = Pi_Lambda(
        Xi[t]
        + P_1:M^inward(Xi[t])
        - E[t]
        + Omega[t]
        + kappa * R[t]^inward
        - eta * grad_Theta L[t]
        - zeta * grad_H C[t]
    )

This module deliberately does not mutate the canonical Jarvis-X VM. It is a
pure Layer-5 same-space operator: every additive term must inhabit the same
sparse 3D support as Xi, the proposed state is passed through Pi_Lambda, and a
candidate becomes the returned next state only after optional validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

Coordinate = tuple[int, int, int]
SparseField = dict[Coordinate, float]
FieldLike = Mapping[Coordinate, float]
Projector = Callable[[FieldLike], Mapping[Coordinate, float]]
Validator = Callable[[FieldLike], bool]


@dataclass(frozen=True)
class DrMoagiEquationConfig:
    """Scalar coefficients and resource limits for the geometric recurrence."""

    kappa: float = 1.0
    eta: float = 1.0
    zeta: float = 1.0
    max_active_cells: int = 100_000
    enforce_same_support: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_active_cells, bool)
            or not isinstance(self.max_active_cells, int)
            or self.max_active_cells <= 0
        ):
            raise ValueError("max_active_cells must be a positive integer")
        for name in ("kappa", "eta", "zeta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class DrMoagiEquationTerms:
    """The six additive fields appearing beside the current state Xi_t."""

    prediction: SparseField
    error: SparseField
    memory: SparseField
    refinement: SparseField
    loss_gradient: SparseField
    constraint_gradient: SparseField


@dataclass(frozen=True)
class DrMoagiEquationStep:
    """Auditable result of one candidate-first equation evaluation."""

    state_before: SparseField
    terms: DrMoagiEquationTerms
    raw_candidate: SparseField
    projected_candidate: SparseField
    next_state: SparseField
    branch_count: int
    committed: bool
    rejection_reason: str | None = None


def _validate_coordinate(coordinate: Coordinate) -> None:
    if not isinstance(coordinate, tuple) or len(coordinate) != 3:
        raise TypeError("3D coordinates must be (x, y, z) tuples")
    if any(isinstance(axis, bool) or not isinstance(axis, int) for axis in coordinate):
        raise TypeError("3D coordinate axes must be integers")


def _validated_field(
    name: str,
    field: FieldLike,
    *,
    support: set[Coordinate] | None = None,
    enforce_same_support: bool = True,
    max_active_cells: int | None = None,
) -> SparseField:
    if max_active_cells is not None and len(field) > max_active_cells:
        raise RuntimeError(f"{name} exceeds active-cell budget")

    result: SparseField = {}
    for coordinate, value in field.items():
        _validate_coordinate(coordinate)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} values must be numeric")
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"{name} contains a non-finite value")
        result[coordinate] = scalar

    if support is not None and enforce_same_support and set(result) != support:
        missing = len(support - set(result))
        extra = len(set(result) - support)
        raise ValueError(
            f"{name} must share Xi support exactly (missing={missing}, extra={extra})"
        )
    return result


def merge_predictive_branches(
    branches: Sequence[FieldLike],
    *,
    support: set[Coordinate],
    weights: Sequence[float] | None = None,
    enforce_same_support: bool = True,
) -> SparseField:
    """Merge P_1:M deterministically into one same-space predictive field.

    With no explicit weights, branches receive equal weight. Supplied weights
    are normalized by their positive finite sum, so the merged field remains a
    convex combination and branch magnitude does not grow merely because M
    increases.
    """

    if not branches:
        raise ValueError("at least one predictive branch is required")

    if weights is None:
        normalized = [1.0 / len(branches)] * len(branches)
    else:
        if len(weights) != len(branches):
            raise ValueError("prediction weights must match branch count")
        normalized = []
        total = 0.0
        for weight in weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError("prediction weights must be numeric")
            scalar = float(weight)
            if not math.isfinite(scalar) or scalar < 0.0:
                raise ValueError("prediction weights must be finite and non-negative")
            normalized.append(scalar)
            total += scalar
        if total <= 0.0:
            raise ValueError("prediction weights must have a positive sum")
        normalized = [weight / total for weight in normalized]

    validated = [
        _validated_field(
            f"prediction branch {index}",
            branch,
            support=support,
            enforce_same_support=enforce_same_support,
        )
        for index, branch in enumerate(branches)
    ]

    return {
        coordinate: sum(
            weight * branch[coordinate]
            for weight, branch in zip(normalized, validated, strict=True)
        )
        for coordinate in support
    }


def box_projector(value_min: float, value_max: float) -> Projector:
    """Construct a deterministic Pi_Lambda projection onto a scalar box."""

    if not math.isfinite(value_min) or not math.isfinite(value_max):
        raise ValueError("projection bounds must be finite")
    if value_min >= value_max:
        raise ValueError("value_min must be smaller than value_max")

    def project(field: FieldLike) -> SparseField:
        return {
            coordinate: min(value_max, max(value_min, float(value)))
            for coordinate, value in field.items()
        }

    return project


class DrMoagiStateEquation:
    """Pure candidate-first evaluator for the 3D Dr Moagi recurrence."""

    def __init__(
        self,
        config: DrMoagiEquationConfig | None = None,
        *,
        projector: Projector | None = None,
    ) -> None:
        self.config = config or DrMoagiEquationConfig()
        self.projector = projector or (lambda field: dict(field))

    def step(
        self,
        state: FieldLike,
        *,
        prediction_branches: Sequence[FieldLike],
        error: FieldLike,
        memory: FieldLike,
        refinement: FieldLike,
        loss_gradient: FieldLike,
        constraint_gradient: FieldLike,
        prediction_weights: Sequence[float] | None = None,
        validator: Validator | None = None,
    ) -> DrMoagiEquationStep:
        """Evaluate one projected update without mutating external state."""

        snapshot = _validated_field(
            "Xi_t",
            state,
            max_active_cells=self.config.max_active_cells,
        )
        support = set(snapshot)

        prediction = merge_predictive_branches(
            prediction_branches,
            support=support,
            weights=prediction_weights,
            enforce_same_support=self.config.enforce_same_support,
        )
        error_field = _validated_field(
            "E_t",
            error,
            support=support,
            enforce_same_support=self.config.enforce_same_support,
        )
        memory_field = _validated_field(
            "Omega_t",
            memory,
            support=support,
            enforce_same_support=self.config.enforce_same_support,
        )
        refinement_field = _validated_field(
            "R_t",
            refinement,
            support=support,
            enforce_same_support=self.config.enforce_same_support,
        )
        loss_field = _validated_field(
            "grad_Theta L_t",
            loss_gradient,
            support=support,
            enforce_same_support=self.config.enforce_same_support,
        )
        constraint_field = _validated_field(
            "grad_H C_t",
            constraint_gradient,
            support=support,
            enforce_same_support=self.config.enforce_same_support,
        )

        terms = DrMoagiEquationTerms(
            prediction=prediction,
            error=error_field,
            memory=memory_field,
            refinement=refinement_field,
            loss_gradient=loss_field,
            constraint_gradient=constraint_field,
        )

        raw_candidate = {
            coordinate: (
                snapshot[coordinate]
                + prediction[coordinate]
                - error_field[coordinate]
                + memory_field[coordinate]
                + self.config.kappa * refinement_field[coordinate]
                - self.config.eta * loss_field[coordinate]
                - self.config.zeta * constraint_field[coordinate]
            )
            for coordinate in support
        }
        raw_candidate = _validated_field(
            "raw candidate",
            raw_candidate,
            support=support,
            enforce_same_support=self.config.enforce_same_support,
            max_active_cells=self.config.max_active_cells,
        )

        projected_raw = self.projector(raw_candidate)
        projected = _validated_field(
            "Pi_Lambda candidate",
            projected_raw,
            support=support,
            enforce_same_support=self.config.enforce_same_support,
            max_active_cells=self.config.max_active_cells,
        )

        if validator is not None and not bool(validator(projected)):
            return DrMoagiEquationStep(
                state_before=snapshot,
                terms=terms,
                raw_candidate=raw_candidate,
                projected_candidate=projected,
                next_state=dict(snapshot),
                branch_count=len(prediction_branches),
                committed=False,
                rejection_reason="validator rejected Pi_Lambda candidate",
            )

        return DrMoagiEquationStep(
            state_before=snapshot,
            terms=terms,
            raw_candidate=raw_candidate,
            projected_candidate=projected,
            next_state=dict(projected),
            branch_count=len(prediction_branches),
            committed=True,
        )
