"""Geometric auto-encoding, validation, and transactional execution.

The module implements the canonical cycle

    X -> E_G(X) -> T(E_G(X)) -> D_G(T(E_G(X)))

for finite three-dimensional lattices.  Arithmetic addresses are mapped
bijectively to coordinates, geometric transformations are represented as
permutations, and candidate states must pass validation before commit.

The implementation intentionally uses only the Python standard library so the
geometric control plane remains portable.  Accelerated numerical kernels can
be placed behind the same validated transaction boundary later.
"""

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Coordinate = Tuple[int, int, int]
Scalar = Any


@dataclass(frozen=True)
class GridGeometry:
    """Finite D x H x W lattice with an exact mixed-radix address map."""

    depth: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.depth <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("Grid dimensions must be positive integers")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.depth, self.height, self.width)

    @property
    def size(self) -> int:
        return self.depth * self.height * self.width

    def arithmetic_to_coordinate(self, address: int) -> Coordinate:
        """gamma(a) = (z, y, x)."""
        if not isinstance(address, int):
            raise TypeError("Address must be an integer")
        if address < 0 or address >= self.size:
            raise ValueError("Address outside lattice")

        plane = self.height * self.width
        z = address // plane
        remainder = address % plane
        y = remainder // self.width
        x = remainder % self.width
        return (z, y, x)

    def coordinate_to_arithmetic(self, coordinate: Coordinate) -> int:
        """gamma^-1(z, y, x) = x + W(y + Hz)."""
        if len(coordinate) != 3:
            raise ValueError("A coordinate must contain exactly three components")

        z, y, x = coordinate
        if not all(isinstance(component, int) for component in coordinate):
            raise TypeError("Coordinate components must be integers")
        if not (0 <= z < self.depth):
            raise ValueError("z coordinate outside lattice")
        if not (0 <= y < self.height):
            raise ValueError("y coordinate outside lattice")
        if not (0 <= x < self.width):
            raise ValueError("x coordinate outside lattice")

        return x + self.width * (y + self.height * z)

    def coordinates(self) -> Iterable[Coordinate]:
        for address in range(self.size):
            yield self.arithmetic_to_coordinate(address)

    def validate_round_trip(self) -> bool:
        return all(
            self.coordinate_to_arithmetic(self.arithmetic_to_coordinate(address)) == address
            for address in range(self.size)
        )

    def neighbours(self, coordinate: Coordinate) -> Tuple[Coordinate, ...]:
        """Return the in-bounds six-connected neighbourhood."""
        z, y, x = coordinate
        candidates = (
            (z - 1, y, x),
            (z + 1, y, x),
            (z, y - 1, x),
            (z, y + 1, x),
            (z, y, x - 1),
            (z, y, x + 1),
        )
        return tuple(
            candidate
            for candidate in candidates
            if 0 <= candidate[0] < self.depth
            and 0 <= candidate[1] < self.height
            and 0 <= candidate[2] < self.width
        )


@dataclass
class GeometricLatent:
    """Canonical geometric state Z_G = (C, A, E, Omega, Lambda)."""

    geometry: GridGeometry
    cells: Dict[Coordinate, Scalar]
    topology: Dict[Coordinate, Tuple[Coordinate, ...]]
    omega: Dict[Coordinate, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    version: int = 0

    def canonical_values(self) -> List[Scalar]:
        return [
            self.cells[self.geometry.arithmetic_to_coordinate(address)]
            for address in range(self.geometry.size)
        ]

    def digest(self) -> str:
        """Deterministic audit hash over shape, values, and version."""
        payload = repr(
            (
                self.geometry.shape,
                tuple(self.canonical_values()),
                self.version,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    coordinate_bijection: bool
    cardinality_valid: bool
    coordinates_valid: bool
    topology_valid: bool
    values_finite: bool
    message: str


@dataclass(frozen=True)
class PermutationTransform:
    """Bijective movement from each source address to a destination address."""

    destination_by_source: Tuple[int, ...]

    @classmethod
    def from_sequence(cls, permutation: Sequence[int]) -> "PermutationTransform":
        values = tuple(permutation)
        expected = tuple(range(len(values)))
        if tuple(sorted(values)) != expected:
            raise ValueError("Transformation must be a permutation of 0..N-1")
        return cls(values)

    @classmethod
    def identity(cls, size: int) -> "PermutationTransform":
        if size < 0:
            raise ValueError("Size must not be negative")
        return cls(tuple(range(size)))

    def inverse(self) -> "PermutationTransform":
        inverse_values = [0] * len(self.destination_by_source)
        for source, destination in enumerate(self.destination_by_source):
            inverse_values[destination] = source
        return PermutationTransform(tuple(inverse_values))

    def apply(self, values: Sequence[Scalar]) -> List[Scalar]:
        if len(values) != len(self.destination_by_source):
            raise ValueError("Value count and transformation size differ")

        transformed: List[Scalar] = [None] * len(values)
        for source, destination in enumerate(self.destination_by_source):
            transformed[destination] = values[source]
        return transformed


class GeometricAutoEncoder:
    """Exact arithmetic-to-geometric encoder and decoder."""

    def encode(
        self,
        values: Sequence[Scalar],
        shape: Tuple[int, int, int],
        constraints: Optional[Mapping[str, Any]] = None,
    ) -> GeometricLatent:
        geometry = GridGeometry(*shape)
        if len(values) != geometry.size:
            raise ValueError(
                "Input contains {0} values but geometry requires {1}".format(
                    len(values), geometry.size
                )
            )

        cells = {
            geometry.arithmetic_to_coordinate(address): value
            for address, value in enumerate(values)
        }
        topology = {
            coordinate: geometry.neighbours(coordinate)
            for coordinate in geometry.coordinates()
        }
        omega = {coordinate: 0.0 for coordinate in geometry.coordinates()}
        return GeometricLatent(
            geometry=geometry,
            cells=cells,
            topology=topology,
            omega=omega,
            constraints=dict(constraints or {}),
        )

    def decode(self, latent: GeometricLatent) -> List[Scalar]:
        report = validate_latent(latent)
        if not report.passed:
            raise ValueError("Cannot decode invalid geometry: {0}".format(report.message))
        return latent.canonical_values()

    def transform(
        self,
        latent: GeometricLatent,
        transformation: PermutationTransform,
    ) -> GeometricLatent:
        values = self.decode(latent)
        transformed = transformation.apply(values)
        result = self.encode(transformed, latent.geometry.shape, latent.constraints)
        result.omega = dict(latent.omega)
        result.version = latent.version + 1
        return result

    def inverse_transform(
        self,
        latent: GeometricLatent,
        transformation: PermutationTransform,
    ) -> GeometricLatent:
        return self.transform(latent, transformation.inverse())


def _is_finite(value: Scalar) -> bool:
    if isinstance(value, (int, bool)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def validate_latent(latent: GeometricLatent) -> ValidationReport:
    """Validate address bijection, topology, cardinality, and numeric state."""
    geometry = latent.geometry
    expected = set(geometry.coordinates())
    actual = set(latent.cells)

    coordinate_bijection = geometry.validate_round_trip()
    cardinality_valid = len(latent.cells) == geometry.size
    coordinates_valid = actual == expected

    topology_valid = set(latent.topology) == expected
    if topology_valid:
        for coordinate, neighbours in latent.topology.items():
            expected_neighbours = set(geometry.neighbours(coordinate))
            if set(neighbours) != expected_neighbours:
                topology_valid = False
                break
            if any(coordinate not in latent.topology.get(neighbour, ()) for neighbour in neighbours):
                topology_valid = False
                break

    values_finite = all(_is_finite(value) for value in latent.cells.values())
    passed = all(
        (
            coordinate_bijection,
            cardinality_valid,
            coordinates_valid,
            topology_valid,
            values_finite,
        )
    )

    failed = []
    if not coordinate_bijection:
        failed.append("coordinate bijection")
    if not cardinality_valid:
        failed.append("cardinality")
    if not coordinates_valid:
        failed.append("coordinate domain")
    if not topology_valid:
        failed.append("topology")
    if not values_finite:
        failed.append("finite values")

    message = "valid" if passed else "failed: {0}".format(", ".join(failed))
    return ValidationReport(
        passed=passed,
        coordinate_bijection=coordinate_bijection,
        cardinality_valid=cardinality_valid,
        coordinates_valid=coordinates_valid,
        topology_valid=topology_valid,
        values_finite=values_finite,
        message=message,
    )


@dataclass
class GeometricTransaction:
    candidate: GeometricLatent
    validation: ValidationReport
    previous_hash: Optional[str]
    candidate_hash: str
    committed: bool = False


class GeometricRuntime:
    """Transactional geometric execution with validation and Omega memory."""

    def __init__(self) -> None:
        self.codec = GeometricAutoEncoder()
        self.committed: Optional[GeometricLatent] = None
        self.pending: Optional[GeometricTransaction] = None
        self.journal: List[Dict[str, Any]] = []

    def load(
        self,
        values: Sequence[Scalar],
        shape: Tuple[int, int, int],
        constraints: Optional[Mapping[str, Any]] = None,
    ) -> GeometricLatent:
        latent = self.codec.encode(values, shape, constraints)
        report = validate_latent(latent)
        if not report.passed:
            raise RuntimeError(report.message)
        self.committed = latent
        self.pending = None
        self._journal("LOAD", latent, report, committed=True)
        return latent

    def propose(self, transformation: PermutationTransform) -> GeometricTransaction:
        if self.committed is None:
            raise RuntimeError("No committed geometric state")
        if len(transformation.destination_by_source) != self.committed.geometry.size:
            raise ValueError("Transformation size does not match committed geometry")

        candidate = self.codec.transform(self.committed, transformation)
        report = validate_latent(candidate)
        transaction = GeometricTransaction(
            candidate=candidate,
            validation=report,
            previous_hash=self.committed.digest(),
            candidate_hash=candidate.digest(),
        )
        self.pending = transaction
        self._journal("PROPOSE", candidate, report, committed=False)
        return transaction

    def commit(self) -> GeometricLatent:
        if self.pending is None:
            raise RuntimeError("No pending geometric transaction")
        if not self.pending.validation.passed:
            raise RuntimeError("Validation gate rejected candidate state")

        self.pending.committed = True
        self.committed = self.pending.candidate
        self._journal("COMMIT", self.committed, self.pending.validation, committed=True)
        self.pending = None
        return self.committed

    def rollback(self) -> Optional[GeometricLatent]:
        if self.pending is not None:
            self._journal(
                "ROLLBACK",
                self.pending.candidate,
                self.pending.validation,
                committed=False,
            )
        self.pending = None
        return self.committed

    def execute(self, transformation: PermutationTransform) -> GeometricLatent:
        transaction = self.propose(transformation)
        if not transaction.validation.passed:
            self.rollback()
            raise RuntimeError(transaction.validation.message)
        return self.commit()

    def update_omega(
        self,
        observed: Sequence[float],
        retention: float = 0.98,
        learning_rate: float = 0.02,
    ) -> Dict[Coordinate, float]:
        """Update spatial correction memory from observed - predicted residuals."""
        if self.committed is None:
            raise RuntimeError("No committed geometric state")
        if len(observed) != self.committed.geometry.size:
            raise ValueError("Observed state size differs from committed geometry")
        if not 0.0 <= retention <= 1.0:
            raise ValueError("Retention must lie in [0, 1]")
        if learning_rate < 0.0:
            raise ValueError("Learning rate must be non-negative")

        predicted = self.committed.canonical_values()
        for address, (actual, estimate) in enumerate(zip(observed, predicted)):
            coordinate = self.committed.geometry.arithmetic_to_coordinate(address)
            residual = float(actual) - float(estimate)
            previous = self.committed.omega.get(coordinate, 0.0)
            self.committed.omega[coordinate] = retention * previous + learning_rate * residual

        self._journal(
            "UPDATE_OMEGA",
            self.committed,
            validate_latent(self.committed),
            committed=True,
        )
        return dict(self.committed.omega)

    def round_trip(
        self,
        values: Sequence[Scalar],
        shape: Tuple[int, int, int],
        transformation: PermutationTransform,
    ) -> bool:
        encoded = self.codec.encode(values, shape)
        transformed = self.codec.transform(encoded, transformation)
        recovered = self.codec.inverse_transform(transformed, transformation)
        return self.codec.decode(recovered) == list(values)

    def _journal(
        self,
        operation: str,
        state: GeometricLatent,
        report: ValidationReport,
        committed: bool,
    ) -> None:
        self.journal.append(
            {
                "sequence": len(self.journal),
                "operation": operation,
                "version": state.version,
                "shape": state.geometry.shape,
                "state_hash": state.digest(),
                "validation_passed": report.passed,
                "committed": committed,
            }
        )
