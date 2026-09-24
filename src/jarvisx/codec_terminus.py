"""Codec -> mixing -> throughput -> geometry -> audit -> fixed-point reference kernel.

This module makes the user's TERMINUS chain executable without promoting it into
Jarvis-X's canonical VM authority.  It is a bounded Layer-5 research contract:

    X --E--> Y --T--> Y --D--> X
    R = D @ T @ E
    p_(t+1) = R p_t
    nu_eff = min(nu_syn, nu_bw, nu_q, nu_th, nu_ch)
    (x, y, z) = (position, bitplane, stage)
    C(n) = c*n with c ~= 4 as a declared coefficient
    audit = local overlap consistency + H^1 of the 1D cover nerve
    S = project o refine o audit
    P* = minimum-cost audited fixed point

Important mathematical boundary:
- stochastic matrices use column-vector probabilities and are column-stochastic;
- convergence is reported empirically for the supplied finite operator.  The
  runtime does not infer irreducibility, aperiodicity, a unique invariant
  distribution, or a spectral mixing bound unless a caller proves them;
- the cohomology calculation is exact only for the 1-dimensional nerve graph
  with constant field coefficients.  It is a conservative audit, not a general
  sheaf-cohomology engine;
- the Kleene iterator checks that the realized chain is ascending.  That runtime
  witness is not a proof that the operator is globally monotone or
  Scott-continuous.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Generic, Hashable, Mapping, Sequence, TypeVar


Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]
T = TypeVar("T")
K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def _matrix(values: Sequence[Sequence[float]], *, name: str) -> Matrix:
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if not rows or not rows[0]:
        raise ValueError(f"{name} must be non-empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"{name} must contain only finite values")
    return rows


def _shape(matrix: Matrix) -> tuple[int, int]:
    return len(matrix), len(matrix[0])


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    left_rows, left_cols = _shape(left)
    right_rows, right_cols = _shape(right)
    if left_cols != right_rows:
        raise ValueError(
            f"matrix shape mismatch: {(left_rows, left_cols)} x "
            f"{(right_rows, right_cols)}"
        )
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(left_cols))
            for j in range(right_cols)
        )
        for i in range(left_rows)
    )


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    rows, cols = _shape(matrix)
    if cols != len(vector):
        raise ValueError(f"matrix/vector shape mismatch: {(rows, cols)} x {len(vector)}")
    return tuple(sum(matrix[i][j] * vector[j] for j in range(cols)) for i in range(rows))


def _validate_column_stochastic(
    matrix: Matrix,
    *,
    rows: int | None = None,
    cols: int | None = None,
    name: str,
    atol: float = 1.0e-12,
) -> None:
    actual_rows, actual_cols = _shape(matrix)
    if rows is not None and actual_rows != rows:
        raise ValueError(f"{name} must have {rows} rows, got {actual_rows}")
    if cols is not None and actual_cols != cols:
        raise ValueError(f"{name} must have {cols} columns, got {actual_cols}")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value < -atol:
                raise ValueError(f"{name}[{i}][{j}] must be non-negative")
    for j in range(actual_cols):
        column_sum = sum(matrix[i][j] for i in range(actual_rows))
        if not math.isclose(column_sum, 1.0, rel_tol=0.0, abs_tol=atol):
            raise ValueError(
                f"{name} column {j} must sum to 1, got {column_sum!r}"
            )


def _probability_vector(values: Sequence[float], *, size: int, name: str) -> Vector:
    vector = tuple(float(value) for value in values)
    if len(vector) != size:
        raise ValueError(f"{name} must have length {size}")
    if any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    if any(value < 0.0 for value in vector):
        raise ValueError(f"{name} must be non-negative")
    total = sum(vector)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{name} must sum to 1, got {total!r}")
    return vector


@dataclass(frozen=True)
class MixingReceipt:
    probabilities: Vector
    iterations: int
    converged: bool
    l1_residual: float


@dataclass(frozen=True)
class CodecMarkovOperator:
    """Finite column-stochastic realization of R = D @ T @ E."""

    encoder: Matrix
    transport: Matrix
    decoder: Matrix
    reconstruction: Matrix

    @classmethod
    def from_components(
        cls,
        encoder: Sequence[Sequence[float]],
        transport: Sequence[Sequence[float]],
        decoder: Sequence[Sequence[float]],
    ) -> "CodecMarkovOperator":
        e = _matrix(encoder, name="encoder")
        t = _matrix(transport, name="transport")
        d = _matrix(decoder, name="decoder")

        y_rows, x_cols = _shape(e)
        t_rows, t_cols = _shape(t)
        x_rows, d_cols = _shape(d)

        if t_rows != y_rows or t_cols != y_rows:
            raise ValueError("transport must be square over Y and match encoder rows")
        if d_cols != y_rows:
            raise ValueError("decoder columns must match Y")
        if x_rows != x_cols:
            raise ValueError("decoder rows must match encoder X columns")

        _validate_column_stochastic(e, name="encoder")
        _validate_column_stochastic(t, rows=y_rows, cols=y_rows, name="transport")
        _validate_column_stochastic(d, name="decoder")

        reconstruction = _matmul(d, _matmul(t, e))
        _validate_column_stochastic(
            reconstruction,
            rows=x_cols,
            cols=x_cols,
            name="reconstruction",
        )
        return cls(e, t, d, reconstruction)

    @property
    def x_size(self) -> int:
        return len(self.reconstruction)

    @property
    def y_size(self) -> int:
        return len(self.encoder)

    def step(self, probabilities: Sequence[float]) -> Vector:
        p = _probability_vector(probabilities, size=self.x_size, name="probabilities")
        result = _matvec(self.reconstruction, p)
        total = sum(result)
        if total <= 0.0:
            raise RuntimeError("reconstruction produced an invalid probability vector")
        # Remove harmless floating-point drift while preserving positivity.
        normalized = tuple(max(0.0, value) / total for value in result)
        return _probability_vector(normalized, size=self.x_size, name="next probabilities")

    def mix(
        self,
        initial: Sequence[float],
        *,
        tolerance: float = 1.0e-10,
        max_iterations: int = 10_000,
    ) -> MixingReceipt:
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")

        current = _probability_vector(initial, size=self.x_size, name="initial")
        residual = math.inf
        for iteration in range(1, max_iterations + 1):
            candidate = self.step(current)
            residual = sum(abs(a - b) for a, b in zip(candidate, current))
            current = candidate
            if residual <= tolerance:
                return MixingReceipt(current, iteration, True, residual)
        return MixingReceipt(current, max_iterations, False, residual)


@dataclass(frozen=True)
class ThroughputBounds:
    synthesis: float
    bandwidth: float
    quantization: float
    thermal: float
    channel: float

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} rate must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} rate must be finite and positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "synthesis": float(self.synthesis),
            "bandwidth": float(self.bandwidth),
            "quantization": float(self.quantization),
            "thermal": float(self.thermal),
            "channel": float(self.channel),
        }

    @property
    def effective(self) -> float:
        return min(self.as_dict().values())

    @property
    def active_bounds(self) -> tuple[str, ...]:
        floor = self.effective
        return tuple(
            name
            for name, value in self.as_dict().items()
            if math.isclose(value, floor, rel_tol=1.0e-12, abs_tol=1.0e-12)
        )


@dataclass(frozen=True)
class AxisAssignment:
    x: str = "position"
    y: str = "bitplane"
    z: str = "stage"


@dataclass(frozen=True)
class LinearGeometricCost:
    n: int
    coefficient: float = 4.0

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if (
            isinstance(self.coefficient, bool)
            or not isinstance(self.coefficient, (int, float))
            or not math.isfinite(float(self.coefficient))
            or float(self.coefficient) <= 0.0
        ):
            raise ValueError("coefficient must be finite and positive")

    @property
    def total(self) -> float:
        return float(self.coefficient) * self.n

    @property
    def normalized(self) -> float:
        return float(self.coefficient)

    @property
    def asymptotic_class(self) -> str:
        return "Theta(n)"


@dataclass(frozen=True)
class OverlapConflict(Generic[K, V]):
    left_region: str
    right_region: str
    key: K
    left_value: V
    right_value: V


@dataclass(frozen=True)
class CechAuditReceipt(Generic[K, V]):
    passed: bool
    h1_dimension: int
    components: int
    vertices: int
    edges: int
    conflicts: tuple[OverlapConflict[K, V], ...]

    @property
    def obstruction_zero(self) -> bool:
        return self.h1_dimension == 0


def audit_local_sections(
    sections: Mapping[str, Mapping[K, V]],
) -> CechAuditReceipt[K, V]:
    """Audit overlap gluing and H^1 of the 1D cover nerve.

    The nerve has one vertex per region and an edge for every non-empty overlap.
    For a graph with constant field coefficients,

        dim H^1 = |E| - |V| + number_of_connected_components.

    This is not general sheaf cohomology; it is the exact graph-nerve special
    case used as a conservative admission gate.
    """

    names = tuple(sorted(sections))
    if not names:
        raise ValueError("sections must contain at least one region")

    parent = {name: name for name in names}

    def find(name: str) -> str:
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != name:
            nxt = parent[name]
            parent[name] = root
            name = nxt
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_count = 0
    conflicts: list[OverlapConflict[K, V]] = []

    for left_name, right_name in combinations(names, 2):
        left = sections[left_name]
        right = sections[right_name]
        overlap = set(left).intersection(right)
        if not overlap:
            continue

        edge_count += 1
        union(left_name, right_name)
        for key in sorted(overlap, key=repr):
            if left[key] != right[key]:
                conflicts.append(
                    OverlapConflict(
                        left_region=left_name,
                        right_region=right_name,
                        key=key,
                        left_value=left[key],
                        right_value=right[key],
                    )
                )

    components = len({find(name) for name in names})
    h1_dimension = max(0, edge_count - len(names) + components)
    passed = not conflicts and h1_dimension == 0
    return CechAuditReceipt(
        passed=passed,
        h1_dimension=h1_dimension,
        components=components,
        vertices=len(names),
        edges=edge_count,
        conflicts=tuple(conflicts),
    )


@dataclass(frozen=True)
class KleeneReceipt(Generic[T]):
    value: T
    iterations: int
    converged: bool


def kleene_iterate(
    bottom: T,
    operator: Callable[[T], T],
    *,
    leq: Callable[[T, T], bool],
    max_iterations: int = 256,
) -> KleeneReceipt[T]:
    """Bounded Kleene-style iteration with an ascending-chain runtime check."""

    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    current = bottom
    for iteration in range(1, max_iterations + 1):
        candidate = operator(current)
        if not leq(current, candidate):
            raise ValueError("operator violated the declared ascending order")
        if candidate == current:
            return KleeneReceipt(current, iteration, True)
        current = candidate
    return KleeneReceipt(current, max_iterations, False)


@dataclass(frozen=True)
class TerminusCandidate(Generic[T, K, V]):
    value: T
    cost: float
    audit: CechAuditReceipt[K, V]

    def __post_init__(self) -> None:
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, (int, float))
            or not math.isfinite(float(self.cost))
        ):
            raise ValueError("candidate cost must be finite")


@dataclass(frozen=True)
class TerminusReceipt(Generic[T]):
    value: T
    cost: float
    eligible_candidates: int


def select_terminus(
    candidates: Sequence[TerminusCandidate[T, K, V]],
    *,
    operator: Callable[[T], T],
) -> TerminusReceipt[T]:
    """Select the minimum-cost candidate satisfying audit and fixed-point gates."""

    eligible: list[TerminusCandidate[T, K, V]] = []
    for candidate in candidates:
        if candidate.audit.passed and operator(candidate.value) == candidate.value:
            eligible.append(candidate)

    if not eligible:
        raise ValueError("no audited fixed-point candidate is eligible")

    winner = min(eligible, key=lambda candidate: (float(candidate.cost), repr(candidate.value)))
    return TerminusReceipt(
        value=winner.value,
        cost=float(winner.cost),
        eligible_candidates=len(eligible),
    )
