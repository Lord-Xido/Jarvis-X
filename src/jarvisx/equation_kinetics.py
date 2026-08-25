"""Dependency-free reference runtime for Kinetic Theory of Mathematical Equations.

The module implements finite-dimensional equation-state kinetics for research and
conformance fixtures. It deliberately separates exact continuous identities from
the behavior of the numerical integrator used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

Vector = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]
ResidualFn = Callable[[Vector], Sequence[float]]
JacobianFn = Callable[[Vector], Sequence[Sequence[float]]]
StateValidator = Callable[["EquationKineticState"], bool]


def _finite_vector(values: Sequence[float], *, name: str, allow_empty: bool = False) -> Vector:
    result = tuple(float(value) for value in values)
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_matrix(values: Sequence[Sequence[float]], *, name: str) -> Matrix:
    rows = tuple(_finite_vector(row, name=f"{name} row") for row in values)
    if not rows:
        raise ValueError(f"{name} must not be empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _non_negative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _unit_interval(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return value


def _dot(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")
    return sum(a * b for a, b in zip(left, right))


def _norm(values: Vector) -> float:
    return math.sqrt(_dot(values, values))


def _add(left: Vector, right: Vector) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")
    return tuple(a + b for a, b in zip(left, right))


def _sub(left: Vector, right: Vector) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector dimensions must match")
    return tuple(a - b for a, b in zip(left, right))


def _scale(scale: float, values: Vector) -> Vector:
    return tuple(scale * value for value in values)


def _matvec(matrix: Matrix, vector: Vector) -> Vector:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix/vector dimensions must match")
    return tuple(sum(value * item for value, item in zip(row, vector)) for row in matrix)


def _transpose_matvec(matrix: Matrix, vector: Vector) -> Vector:
    if len(matrix) != len(vector):
        raise ValueError("matrix/vector dimensions must match")
    width = len(matrix[0])
    return tuple(sum(matrix[row][col] * vector[row] for row in range(len(matrix))) for col in range(width))


def _identity(size: int) -> Matrix:
    return tuple(tuple(1.0 if row == col else 0.0 for col in range(size)) for row in range(size))


def _clip_norm(vector: Vector, maximum: float) -> Vector:
    if maximum == 0.0:
        return tuple(0.0 for _ in vector)
    norm = _norm(vector)
    if norm <= maximum or norm == 0.0:
        return vector
    return _scale(maximum / norm, vector)


def _quadratic_form(vector: Vector, matrix: Matrix) -> float:
    return _dot(vector, _matvec(matrix, vector))


@dataclass(frozen=True)
class EquationKineticState:
    """Finite equation-state position, velocity, and bounded correction memory."""

    position: Vector
    velocity: Vector
    memory: Vector = ()
    step: int = 0

    def __post_init__(self) -> None:
        position = _finite_vector(self.position, name="position")
        velocity = _finite_vector(self.velocity, name="velocity")
        if len(position) != len(velocity):
            raise ValueError("position and velocity must have the same dimension")
        memory = _finite_vector(self.memory, name="memory", allow_empty=True)
        if memory and len(memory) != len(position):
            raise ValueError("memory must be empty or match the state dimension")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step < 0:
            raise ValueError("step must be a non-negative integer")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "memory", memory)

    @property
    def dimension(self) -> int:
        return len(self.position)

    @property
    def effective_memory(self) -> Vector:
        return self.memory or tuple(0.0 for _ in self.position)


@dataclass(frozen=True)
class EquationKineticConfig:
    """Explicit numerical and coupling bounds for the reference integrator."""

    dt: float = 0.05
    mass: float = 1.0
    damping: float = 0.25
    coupling: float = 0.0
    memory_retention: float = 0.85
    memory_gain: float = 0.05
    max_speed: float = 100.0
    max_disagreement: float | None = None

    def __post_init__(self) -> None:
        _positive(self.dt, name="dt")
        _positive(self.mass, name="mass")
        _non_negative(self.damping, name="damping")
        _non_negative(self.coupling, name="coupling")
        _unit_interval(self.memory_retention, name="memory_retention")
        _non_negative(self.memory_gain, name="memory_gain")
        _non_negative(self.max_speed, name="max_speed")
        if self.max_disagreement is not None:
            _non_negative(self.max_disagreement, name="max_disagreement")


@dataclass(frozen=True)
class StepMetrics:
    residual_norm: float
    residual_energy: float
    kinetic_energy: float
    force_norm: float
    speed: float


@dataclass(frozen=True)
class EquationStep:
    state: EquationKineticState
    metrics_before: StepMetrics
    metrics_after: StepMetrics


@dataclass(frozen=True)
class DualStepResult:
    committed: bool
    state_a: EquationKineticState
    state_b: EquationKineticState
    disagreement_before: float
    disagreement_after: float
    total_energy_before: float
    total_energy_after: float


@dataclass(frozen=True)
class PopulationMoments:
    count: int
    mean_position: Vector
    mean_velocity: Vector
    velocity_covariance_trace: float
    kinetic_temperature: float


def residual_energy(
    residual: Sequence[float],
    weight: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return 1/2 F^T W F for an explicit residual vector."""

    residual_vector = _finite_vector(residual, name="residual")
    weight_matrix = _identity(len(residual_vector)) if weight is None else _finite_matrix(weight, name="weight")
    if len(weight_matrix) != len(residual_vector) or len(weight_matrix[0]) != len(residual_vector):
        raise ValueError("weight must be square with residual dimension")
    return 0.5 * _quadratic_form(residual_vector, weight_matrix)


def residual_force(
    residual: Sequence[float],
    jacobian: Sequence[Sequence[float]],
    weight: Sequence[Sequence[float]] | None = None,
) -> Vector:
    """Return the residual force -J_F^T W F."""

    residual_vector = _finite_vector(residual, name="residual")
    jacobian_matrix = _finite_matrix(jacobian, name="jacobian")
    if len(jacobian_matrix) != len(residual_vector):
        raise ValueError("jacobian row count must match residual dimension")
    weight_matrix = _identity(len(residual_vector)) if weight is None else _finite_matrix(weight, name="weight")
    if len(weight_matrix) != len(residual_vector) or len(weight_matrix[0]) != len(residual_vector):
        raise ValueError("weight must be square with residual dimension")
    weighted_residual = _matvec(weight_matrix, residual_vector)
    return _scale(-1.0, _transpose_matvec(jacobian_matrix, weighted_residual))


def _metrics(
    state: EquationKineticState,
    residual_fn: ResidualFn,
    jacobian_fn: JacobianFn,
    *,
    mass: float,
) -> StepMetrics:
    residual = _finite_vector(residual_fn(state.position), name="residual")
    jacobian = _finite_matrix(jacobian_fn(state.position), name="jacobian")
    force = residual_force(residual, jacobian)
    return StepMetrics(
        residual_norm=_norm(residual),
        residual_energy=residual_energy(residual),
        kinetic_energy=0.5 * mass * _dot(state.velocity, state.velocity),
        force_norm=_norm(force),
        speed=_norm(state.velocity),
    )


def propose_step(
    state: EquationKineticState,
    residual_fn: ResidualFn,
    jacobian_fn: JacobianFn,
    config: EquationKineticConfig,
    *,
    external_force: Sequence[float] | None = None,
) -> EquationStep:
    """Advance one semi-implicit Euler proposal without granting authority."""

    before = _metrics(state, residual_fn, jacobian_fn, mass=config.mass)
    residual = _finite_vector(residual_fn(state.position), name="residual")
    jacobian = _finite_matrix(jacobian_fn(state.position), name="jacobian")
    local_force = residual_force(residual, jacobian)
    if len(local_force) != state.dimension:
        raise ValueError("jacobian column count must match equation-state dimension")

    memory = state.effective_memory
    total_force = _add(local_force, _scale(config.memory_gain, memory))
    if external_force is not None:
        external = _finite_vector(external_force, name="external_force")
        if len(external) != state.dimension:
            raise ValueError("external_force must match equation-state dimension")
        total_force = _add(total_force, external)

    damping_force = _scale(config.damping, state.velocity)
    acceleration = _scale(1.0 / config.mass, _sub(total_force, damping_force))
    velocity = _add(state.velocity, _scale(config.dt, acceleration))
    velocity = _clip_norm(velocity, config.max_speed)
    position = _add(state.position, _scale(config.dt, velocity))

    retained = _scale(config.memory_retention, memory)
    injected = _scale(1.0 - config.memory_retention, local_force)
    next_memory = _add(retained, injected)

    candidate = EquationKineticState(position, velocity, next_memory, state.step + 1)
    after = _metrics(candidate, residual_fn, jacobian_fn, mass=config.mass)
    return EquationStep(candidate, before, after)


def disagreement(left: EquationKineticState, right: EquationKineticState) -> float:
    """Euclidean disagreement on the declared shared state coordinates."""

    if left.dimension != right.dimension:
        raise ValueError("coupled equation-states must have the same shared dimension")
    return _norm(_sub(left.position, right.position))


def total_dual_energy(
    state_a: EquationKineticState,
    state_b: EquationKineticState,
    residual_a: ResidualFn,
    residual_b: ResidualFn,
    config: EquationKineticConfig,
) -> float:
    """Discrete telemetry counterpart of local + synchronization energy."""

    local = (
        residual_energy(residual_a(state_a.position))
        + residual_energy(residual_b(state_b.position))
        + 0.5 * config.mass * _dot(state_a.velocity, state_a.velocity)
        + 0.5 * config.mass * _dot(state_b.velocity, state_b.velocity)
    )
    sync = 0.5 * config.coupling * disagreement(state_a, state_b) ** 2
    return local + sync


def dual_synchronous_step(
    state_a: EquationKineticState,
    state_b: EquationKineticState,
    residual_a: ResidualFn,
    jacobian_a: JacobianFn,
    residual_b: ResidualFn,
    jacobian_b: JacobianFn,
    config: EquationKineticConfig,
    *,
    validator: StateValidator | None = None,
) -> DualStepResult:
    """Compute two lockstep proposals from one snapshot and commit atomically.

    Coupling uses only the pre-step positions. This preserves synchronous snapshot
    semantics and prevents either processor's proposal from racing ahead of the
    other's local computation.
    """

    if state_a.step != state_b.step:
        raise ValueError("synchronous processors must start at the same logical step")
    if state_a.dimension != state_b.dimension:
        raise ValueError("coupled equation-states must have the same shared dimension")

    before_disagreement = disagreement(state_a, state_b)
    before_energy = total_dual_energy(state_a, state_b, residual_a, residual_b, config)

    coupling_a = _scale(config.coupling, _sub(state_b.position, state_a.position))
    coupling_b = _scale(config.coupling, _sub(state_a.position, state_b.position))

    proposal_a = propose_step(
        state_a,
        residual_a,
        jacobian_a,
        config,
        external_force=coupling_a,
    ).state
    proposal_b = propose_step(
        state_b,
        residual_b,
        jacobian_b,
        config,
        external_force=coupling_b,
    ).state

    after_disagreement = disagreement(proposal_a, proposal_b)
    admissible = True
    if config.max_disagreement is not None:
        admissible = after_disagreement <= config.max_disagreement
    if validator is not None:
        admissible = admissible and validator(proposal_a) and validator(proposal_b)

    if not admissible:
        return DualStepResult(
            committed=False,
            state_a=state_a,
            state_b=state_b,
            disagreement_before=before_disagreement,
            disagreement_after=before_disagreement,
            total_energy_before=before_energy,
            total_energy_after=before_energy,
        )

    after_energy = total_dual_energy(proposal_a, proposal_b, residual_a, residual_b, config)
    return DualStepResult(
        committed=True,
        state_a=proposal_a,
        state_b=proposal_b,
        disagreement_before=before_disagreement,
        disagreement_after=after_disagreement,
        total_energy_before=before_energy,
        total_energy_after=after_energy,
    )


def population_moments(states: Sequence[EquationKineticState]) -> PopulationMoments:
    """Return finite ensemble moments and equation kinetic temperature."""

    if not states:
        raise ValueError("states must not be empty")
    dimension = states[0].dimension
    if any(state.dimension != dimension for state in states):
        raise ValueError("all states must share the same dimension")

    count = len(states)
    mean_position = tuple(sum(state.position[i] for state in states) / count for i in range(dimension))
    mean_velocity = tuple(sum(state.velocity[i] for state in states) / count for i in range(dimension))
    squared_deviation = sum(
        (state.velocity[i] - mean_velocity[i]) ** 2
        for state in states
        for i in range(dimension)
    )
    covariance_trace = squared_deviation / count
    kinetic_temperature = squared_deviation / (count * dimension)
    return PopulationMoments(
        count=count,
        mean_position=mean_position,
        mean_velocity=mean_velocity,
        velocity_covariance_trace=covariance_trace,
        kinetic_temperature=kinetic_temperature,
    )


__all__ = [
    "DualStepResult",
    "EquationKineticConfig",
    "EquationKineticState",
    "EquationStep",
    "PopulationMoments",
    "StepMetrics",
    "disagreement",
    "dual_synchronous_step",
    "population_moments",
    "propose_step",
    "residual_energy",
    "residual_force",
    "total_dual_energy",
]
