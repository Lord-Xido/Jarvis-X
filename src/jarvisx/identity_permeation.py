"""Finite 3-torus identity-permeation contract.

This module operationalizes the canonical lattice

    Lambda = Z_11 x Z_6 x Z_4

and verifies a field operator M against the identity-permeation invariants:

    M o M = M
    C*(p) = M[C*](p) for every p in Lambda
    grad(M - id) = 0 on the periodic lattice
    fundamental-cycle sums are preserved
    M permeates Lambda iff M = id_Lambda

The defect-gradient form is intentional. For a coordinate-valued smooth map,
the literal Jacobian of the identity is I, not zero, so grad(M) = 0 would be
incompatible with M = id on a non-trivial torus. The executable quantity is
therefore the identity defect delta_M = M[C] - C.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Mapping

Coordinate = tuple[int, int, int]
ScalarField = dict[Coordinate, float]
FieldOperator = Callable[[Mapping[Coordinate, float]], Mapping[Coordinate, float]]

CANONICAL_SHAPE = (11, 6, 4)
CANONICAL_CELLS = 11 * 6 * 4


@dataclass(frozen=True)
class IdentityPermeationReceipt:
    """Measured residuals for one operator verification."""

    shape: tuple[int, int, int]
    cells_checked: int
    idempotence_residual: float
    fixed_point_residual: float
    identity_defect_gradient_residual: float
    circulation_residual: float
    periodic_closure: bool
    permeates: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class TorusLattice3D:
    """Exact finite 3-torus with canonical modular coordinates."""

    def __init__(self, shape: tuple[int, int, int] = CANONICAL_SHAPE) -> None:
        if shape != CANONICAL_SHAPE:
            raise ValueError(
                "identity-permeation contract is locked to shape (11, 6, 4)"
            )
        self.shape = shape
        self._domain = tuple(
            (x, y, z)
            for z in range(shape[2])
            for y in range(shape[1])
            for x in range(shape[0])
        )

    @property
    def cells(self) -> int:
        return len(self._domain)

    @property
    def domain(self) -> tuple[Coordinate, ...]:
        return self._domain

    def wrap(self, coordinate: Coordinate) -> Coordinate:
        x, y, z = coordinate
        sx, sy, sz = self.shape
        return (x % sx, y % sy, z % sz)

    def neighbour(self, coordinate: Coordinate, axis: int, step: int = 1) -> Coordinate:
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        c = list(coordinate)
        c[axis] += step
        return self.wrap((c[0], c[1], c[2]))

    def validate_field(self, field: Mapping[Coordinate, float]) -> ScalarField:
        if set(field) != set(self._domain):
            missing = len(set(self._domain) - set(field))
            extra = len(set(field) - set(self._domain))
            raise ValueError(
                "field must contain exactly the canonical 264 coordinates "
                f"(missing={missing}, extra={extra})"
            )

        out: ScalarField = {}
        for coordinate in self._domain:
            value = field[coordinate]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("field values must be numeric scalars")
            value_f = float(value)
            if not math.isfinite(value_f):
                raise ValueError("field values must be finite")
            out[coordinate] = value_f
        return out

    def periodic_closure_holds(self) -> bool:
        for coordinate in self._domain:
            for axis, period in enumerate(self.shape):
                c = list(coordinate)
                c[axis] += period
                if self.wrap((c[0], c[1], c[2])) != coordinate:
                    return False
        return True

    def fundamental_cycle_sum(
        self,
        field: Mapping[Coordinate, float],
        axis: int,
        base: Coordinate,
    ) -> float:
        """Discrete line integral with unit edge length around one generator loop."""

        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        base = self.wrap(base)
        total = 0.0
        for i in range(self.shape[axis]):
            c = list(base)
            c[axis] = i
            total += float(field[(c[0], c[1], c[2])])
        return total


class IdentityPermeationVerifier:
    """Verify an arbitrary full-field operator against the identity contract."""

    def __init__(self, lattice: TorusLattice3D | None = None) -> None:
        self.lattice = lattice or TorusLattice3D()

    @staticmethod
    def _max_difference(
        left: Mapping[Coordinate, float],
        right: Mapping[Coordinate, float],
    ) -> float:
        return max(abs(left[c] - right[c]) for c in left)

    def _defect_gradient_residual(
        self,
        defect: Mapping[Coordinate, float],
    ) -> float:
        maximum = 0.0
        for coordinate in self.lattice.domain:
            origin = defect[coordinate]
            for axis in (0, 1, 2):
                neighbour = self.lattice.neighbour(coordinate, axis)
                maximum = max(maximum, abs(defect[neighbour] - origin))
        return maximum

    def _circulation_residual(
        self,
        before: Mapping[Coordinate, float],
        after: Mapping[Coordinate, float],
    ) -> float:
        maximum = 0.0
        sx, sy, sz = self.lattice.shape

        for y in range(sy):
            for z in range(sz):
                base = (0, y, z)
                maximum = max(
                    maximum,
                    abs(
                        self.lattice.fundamental_cycle_sum(after, 0, base)
                        - self.lattice.fundamental_cycle_sum(before, 0, base)
                    ),
                )
        for x in range(sx):
            for z in range(sz):
                base = (x, 0, z)
                maximum = max(
                    maximum,
                    abs(
                        self.lattice.fundamental_cycle_sum(after, 1, base)
                        - self.lattice.fundamental_cycle_sum(before, 1, base)
                    ),
                )
        for x in range(sx):
            for y in range(sy):
                base = (x, y, 0)
                maximum = max(
                    maximum,
                    abs(
                        self.lattice.fundamental_cycle_sum(after, 2, base)
                        - self.lattice.fundamental_cycle_sum(before, 2, base)
                    ),
                )
        return maximum

    def verify(
        self,
        operator: FieldOperator,
        field: Mapping[Coordinate, float],
        *,
        tolerance: float = 0.0,
    ) -> IdentityPermeationReceipt:
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and non-negative")

        initial = self.lattice.validate_field(field)
        once = self.lattice.validate_field(operator(initial))
        twice = self.lattice.validate_field(operator(once))

        idempotence = self._max_difference(twice, once)
        fixed_point = self._max_difference(once, initial)
        defect = {c: once[c] - initial[c] for c in self.lattice.domain}
        defect_gradient = self._defect_gradient_residual(defect)
        circulation = self._circulation_residual(initial, once)
        periodic = self.lattice.periodic_closure_holds()

        permeates = (
            periodic
            and idempotence <= tolerance
            and fixed_point <= tolerance
            and defect_gradient <= tolerance
            and circulation <= tolerance
        )

        return IdentityPermeationReceipt(
            shape=self.lattice.shape,
            cells_checked=self.lattice.cells,
            idempotence_residual=idempotence,
            fixed_point_residual=fixed_point,
            identity_defect_gradient_residual=defect_gradient,
            circulation_residual=circulation,
            periodic_closure=periodic,
            permeates=permeates,
        )


def identity_permeation(
    field: Mapping[Coordinate, float],
) -> ScalarField:
    """Canonical operator M = id_Lambda."""

    return {coordinate: float(value) for coordinate, value in field.items()}


def canonical_field() -> ScalarField:
    """Deterministic non-constant fixture spanning all 264 torus cells."""

    lattice = TorusLattice3D()
    return {
        (x, y, z): math.sin(0.37 * x) + math.cos(0.53 * y) + 0.125 * z
        for x, y, z in lattice.domain
    }


def verify_canonical_identity(
    field: Mapping[Coordinate, float] | None = None,
) -> IdentityPermeationReceipt:
    """Run the locked M = id verification contract."""

    verifier = IdentityPermeationVerifier()
    return verifier.verify(identity_permeation, field or canonical_field())
