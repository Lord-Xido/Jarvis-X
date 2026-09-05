"""Inward multiparallel multimodal 3D swarm reference runtime.

This module is a bounded research layer. It treats 3D coordinates as a virtual
semantic/control chart, not literal physical space. Modality codecs provide the
boundary between real media representations and the shared 3D chart.

The runtime implements the local-chart form of

    dZ/dt = -G^-1 grad(J) + lambda(Phi(Z) - Z) - gamma L(Z) Z + F_memory

with the rank-one Riemannian metric

    G = I + alpha grad(phi) grad(phi)^T.

It also exposes an RC-network analogue of the inward and graph-relaxation terms.
That analogue is an algorithm-to-circuit correspondence; it is not a Maxwell
field solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Protocol, Sequence

Vec3 = tuple[float, float, float]
Feature = tuple[float, ...]
Matrix3 = tuple[Vec3, Vec3, Vec3]


def _finite_scalar(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _non_negative(value: float, *, name: str) -> float:
    value = _finite_scalar(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive(value: float, *, name: str) -> float:
    value = _finite_scalar(value, name=name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _vec3(values: Sequence[float], *, name: str) -> Vec3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three coordinates")
    result = tuple(_finite_scalar(value, name=name) for value in values)
    return result[0], result[1], result[2]


def _feature(values: Sequence[float], *, name: str = "feature") -> Feature:
    result = tuple(_finite_scalar(value, name=name) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _add(left: Vec3, right: Vec3) -> Vec3:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _sub(left: Vec3, right: Vec3) -> Vec3:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def _scale(value: Vec3, factor: float) -> Vec3:
    return value[0] * factor, value[1] * factor, value[2] * factor


def _dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _norm(value: Vec3) -> float:
    return math.sqrt(_dot(value, value))


def _distance2(left: Vec3, right: Vec3) -> float:
    delta = _sub(left, right)
    return _dot(delta, delta)


def _matvec(matrix: Matrix3, vector: Vec3) -> Vec3:
    return (
        _dot(matrix[0], vector),
        _dot(matrix[1], vector),
        _dot(matrix[2], vector),
    )


def _clamp_vec(value: Vec3, bound: float) -> Vec3:
    return (
        min(bound, max(-bound, value[0])),
        min(bound, max(-bound, value[1])),
        min(bound, max(-bound, value[2])),
    )


def _cap_norm(value: Vec3, maximum: float) -> Vec3:
    norm = _norm(value)
    if norm <= maximum or norm == 0.0:
        return value
    return _scale(value, maximum / norm)


def _cosine(left: Feature, right: Feature) -> float:
    if len(left) != len(right):
        raise ValueError("all particles must use the same shared feature width")
    dot = sum(a * b for a, b in zip(left, right))
    ln = math.sqrt(sum(a * a for a in left))
    rn = math.sqrt(sum(b * b for b in right))
    if ln == 0.0 or rn == 0.0:
        return 0.0
    return dot / (ln * rn)


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    maximum = max(values)
    exps = tuple(math.exp(value - maximum) for value in values)
    total = sum(exps)
    return tuple(value / total for value in exps)


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GEOMETRY = "geometry"
    CODE = "code"
    DATA = "data"


class ModalityCodec(Protocol):
    """Boundary adapter between one media modality and the shared 3D chart."""

    def encode(self, value: object) -> Vec3:
        """Map one modality value to a bounded virtual 3D coordinate."""

    def decode(self, position: Vec3) -> object:
        """Generate/project one modality value from a virtual 3D coordinate."""


FeatureEncoder = Callable[[Modality, object, Vec3], Sequence[float]]
TaskGradient = Callable[["Particle3D"], Vec3]
PotentialGradient = Callable[["Particle3D"], Vec3]
TaskEnergy = Callable[["Particle3D"], float]


@dataclass(frozen=True)
class Particle3D:
    """One modality-tagged hypothesis in the shared virtual 3D chart."""

    position: Vec3
    feature: Feature
    modality: Modality
    confidence: float = 1.0
    time_coordinate: float = 0.0
    source_id: str = ""

    def __post_init__(self) -> None:
        position = _vec3(self.position, name="position")
        feature = _feature(self.feature)
        confidence = _finite_scalar(self.confidence, name="confidence")
        time_coordinate = _finite_scalar(self.time_coordinate, name="time_coordinate")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie within [0, 1]")
        if not isinstance(self.modality, Modality):
            raise TypeError("modality must be a Modality")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "time_coordinate", time_coordinate)


@dataclass(frozen=True)
class Swarm3DConfig:
    """Numerical and resource bounds for the inward swarm."""

    dt: float = 0.05
    alpha_metric: float = 0.40
    task_gain: float = 1.0
    inward_gain: float = 0.15
    swarm_gain: float = 0.20
    memory_gain: float = 0.10
    feature_mix_gain: float = 0.10
    feature_similarity_gain: float = 1.0
    geometry_distance_gain: float = 0.25
    position_bound: float = 1.0
    max_position_step: float = 0.20
    max_steps: int = 64
    tolerance: float = 1.0e-5
    max_particles: int = 16_384

    def __post_init__(self) -> None:
        _positive(self.dt, name="dt")
        for name in (
            "alpha_metric",
            "task_gain",
            "inward_gain",
            "swarm_gain",
            "memory_gain",
            "feature_mix_gain",
            "feature_similarity_gain",
            "geometry_distance_gain",
        ):
            _non_negative(getattr(self, name), name=name)
        _positive(self.position_bound, name="position_bound")
        _positive(self.max_position_step, name="max_position_step")
        _non_negative(self.tolerance, name="tolerance")
        for name in ("max_steps", "max_particles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class Swarm3DMetrics:
    step: int
    particle_count: int
    energy: float
    task_energy: float
    cycle_energy: float
    coupling_energy: float
    fixed_point_error: float
    consensus_error: float
    max_displacement: float


@dataclass(frozen=True)
class Swarm3DState:
    particles: tuple[Particle3D, ...]
    metrics: Swarm3DMetrics

    def __post_init__(self) -> None:
        if not self.particles:
            raise ValueError("swarm must contain at least one particle")
        width = len(self.particles[0].feature)
        if any(len(particle.feature) != width for particle in self.particles):
            raise ValueError("all particles must use the same shared feature width")


@dataclass(frozen=True)
class Swarm3DResult:
    state: Swarm3DState
    energy_history: tuple[float, ...]
    converged: bool


@dataclass(frozen=True)
class ElectricalAnalogueConfig:
    """Bounded RC-network parameters for the structural hardware analogue."""

    capacitance: float = 1.0
    feedback_conductance: float = 0.15
    coupling_conductance: float = 0.20
    dt: float = 0.01
    voltage_bound: float = 1.0

    def __post_init__(self) -> None:
        _positive(self.capacitance, name="capacitance")
        _non_negative(self.feedback_conductance, name="feedback_conductance")
        _non_negative(self.coupling_conductance, name="coupling_conductance")
        _positive(self.dt, name="dt")
        _positive(self.voltage_bound, name="voltage_bound")


def riemannian_metric_inverse(gradient_phi: Vec3, *, alpha: float) -> Matrix3:
    """Exact inverse of G = I + alpha grad(phi) grad(phi)^T."""

    gradient_phi = _vec3(gradient_phi, name="gradient_phi")
    alpha = _non_negative(alpha, name="alpha")
    denominator = 1.0 + alpha * _dot(gradient_phi, gradient_phi)
    factor = alpha / denominator
    gx, gy, gz = gradient_phi
    return (
        (1.0 - factor * gx * gx, -factor * gx * gy, -factor * gx * gz),
        (-factor * gy * gx, 1.0 - factor * gy * gy, -factor * gy * gz),
        (-factor * gz * gx, -factor * gz * gy, 1.0 - factor * gz * gz),
    )


def local_riemannian_gradient(
    euclidean_gradient: Vec3,
    potential_gradient: Vec3,
    *,
    alpha: float,
) -> Vec3:
    """Return G^-1 grad(J) in the current local chart."""

    inverse = riemannian_metric_inverse(potential_gradient, alpha=alpha)
    return _matvec(inverse, _vec3(euclidean_gradient, name="euclidean_gradient"))


class InwardMultimodalSwarm3D:
    """Dependency-free multiparallel multimodal inward-relaxation runtime."""

    def __init__(
        self,
        codecs: Mapping[Modality, ModalityCodec],
        config: Swarm3DConfig | None = None,
        *,
        feature_encoder: FeatureEncoder | None = None,
    ) -> None:
        if not codecs:
            raise ValueError("at least one modality codec is required")
        normalized: dict[Modality, ModalityCodec] = {}
        for modality, codec in codecs.items():
            if not isinstance(modality, Modality):
                raise TypeError("codec keys must be Modality values")
            normalized[modality] = codec
        self.codecs = normalized
        self.config = config or Swarm3DConfig()
        self.feature_encoder = feature_encoder

    def encode_modalities(
        self,
        inputs: Mapping[Modality, Sequence[object]],
    ) -> tuple[Particle3D, ...]:
        """Encode all supplied modality items into one shared particle population."""

        particles: list[Particle3D] = []
        for modality, values in inputs.items():
            if modality not in self.codecs:
                raise KeyError(f"no codec registered for modality {modality.value}")
            codec = self.codecs[modality]
            for index, value in enumerate(values):
                position = _clamp_vec(
                    _vec3(codec.encode(value), name="encoded position"),
                    self.config.position_bound,
                )
                raw_feature = (
                    self.feature_encoder(modality, value, position)
                    if self.feature_encoder is not None
                    else position
                )
                particles.append(
                    Particle3D(
                        position=position,
                        feature=_feature(raw_feature),
                        modality=modality,
                        source_id=f"{modality.value}:{index}",
                    )
                )
                if len(particles) > self.config.max_particles:
                    raise ValueError("encoded particle count exceeds max_particles")
        if not particles:
            raise ValueError("inputs must contain at least one modality item")
        width = len(particles[0].feature)
        if any(len(particle.feature) != width for particle in particles):
            raise ValueError("feature_encoder must project all modalities to one shared width")
        return tuple(particles)

    def inward_target(self, particle: Particle3D) -> Vec3:
        """Phi(z) = E_m(D_m(z)) for the particle's modality."""

        codec = self.codecs.get(particle.modality)
        if codec is None:
            raise KeyError(f"no codec registered for modality {particle.modality.value}")
        decoded = codec.decode(particle.position)
        return _clamp_vec(
            _vec3(codec.encode(decoded), name="inward target"),
            self.config.position_bound,
        )

    def attention(self, particles: Sequence[Particle3D]) -> tuple[tuple[float, ...], ...]:
        """Dynamic graph weights from shared features and chart distance."""

        if not particles:
            raise ValueError("particles must not be empty")
        width = len(particles[0].feature)
        if any(len(particle.feature) != width for particle in particles):
            raise ValueError("all particles must use the same shared feature width")

        rows: list[tuple[float, ...]] = []
        count = len(particles)
        for index, particle in enumerate(particles):
            if count == 1:
                rows.append((1.0,))
                continue
            scores: list[float] = []
            destinations: list[int] = []
            for other_index, other in enumerate(particles):
                if other_index == index:
                    continue
                score = (
                    self.config.feature_similarity_gain
                    * _cosine(particle.feature, other.feature)
                    - self.config.geometry_distance_gain
                    * _distance2(particle.position, other.position)
                )
                scores.append(score)
                destinations.append(other_index)
            weights = _softmax(scores)
            row = [0.0] * count
            for destination, weight in zip(destinations, weights):
                row[destination] = weight
            rows.append(tuple(row))
        return tuple(rows)

    @staticmethod
    def chart_consensus(particles: Sequence[Particle3D]) -> Vec3:
        """Confidence-weighted local-chart consensus.

        This is the Euclidean/tangent approximation to a Riemannian Frechet mean;
        it is intentionally not advertised as an exact global Karcher solver.
        """

        if not particles:
            raise ValueError("particles must not be empty")
        weights = tuple(particle.confidence for particle in particles)
        total = sum(weights)
        if total == 0.0:
            weights = (1.0,) * len(particles)
            total = float(len(particles))
        return (
            sum(weight * particle.position[0] for weight, particle in zip(weights, particles))
            / total,
            sum(weight * particle.position[1] for weight, particle in zip(weights, particles))
            / total,
            sum(weight * particle.position[2] for weight, particle in zip(weights, particles))
            / total,
        )

    def decode_consensus(
        self,
        particles: Sequence[Particle3D],
        modalities: Sequence[Modality] | None = None,
    ) -> dict[Modality, object]:
        """Decode one consensus control state through multiple modality heads."""

        consensus = self.chart_consensus(particles)
        selected = tuple(modalities) if modalities is not None else tuple(self.codecs)
        generated: dict[Modality, object] = {}
        for modality in selected:
            codec = self.codecs.get(modality)
            if codec is None:
                raise KeyError(f"no codec registered for modality {modality.value}")
            generated[modality] = codec.decode(consensus)
        return generated

    def _metrics(
        self,
        particles: Sequence[Particle3D],
        inward_targets: Sequence[Vec3],
        adjacency: Sequence[Sequence[float]],
        *,
        step: int,
        task_energy: TaskEnergy | None,
        max_displacement: float,
    ) -> Swarm3DMetrics:
        count = len(particles)
        task_value = (
            sum(_finite_scalar(task_energy(particle), name="task_energy") for particle in particles)
            / count
            if task_energy is not None
            else 0.0
        )
        cycle_value = sum(
            _distance2(particle.position, target)
            for particle, target in zip(particles, inward_targets)
        ) / count
        coupling_value = 0.0
        for i, particle in enumerate(particles):
            coupling_value += sum(
                adjacency[i][j] * _distance2(particle.position, other.position)
                for j, other in enumerate(particles)
            )
        coupling_value /= count

        consensus = self.chart_consensus(particles)
        consensus_error = math.sqrt(
            sum(_distance2(particle.position, consensus) for particle in particles) / count
        )
        fixed_point_error = math.sqrt(cycle_value)
        energy = (
            self.config.task_gain * task_value
            + self.config.inward_gain * cycle_value
            + 0.5 * self.config.swarm_gain * coupling_value
        )
        return Swarm3DMetrics(
            step=step,
            particle_count=count,
            energy=energy,
            task_energy=task_value,
            cycle_energy=cycle_value,
            coupling_energy=coupling_value,
            fixed_point_error=fixed_point_error,
            consensus_error=consensus_error,
            max_displacement=max_displacement,
        )

    def initial_state(
        self,
        particles: Sequence[Particle3D],
        *,
        task_energy: TaskEnergy | None = None,
    ) -> Swarm3DState:
        particles = tuple(particles)
        if not particles:
            raise ValueError("particles must not be empty")
        if len(particles) > self.config.max_particles:
            raise ValueError("particle count exceeds max_particles")
        width = len(particles[0].feature)
        if any(len(particle.feature) != width for particle in particles):
            raise ValueError("all particles must use the same shared feature width")
        targets = tuple(self.inward_target(particle) for particle in particles)
        adjacency = self.attention(particles)
        metrics = self._metrics(
            particles,
            targets,
            adjacency,
            step=0,
            task_energy=task_energy,
            max_displacement=0.0,
        )
        return Swarm3DState(particles, metrics)

    def step(
        self,
        state: Swarm3DState,
        *,
        task_gradient: TaskGradient | None = None,
        potential_gradient: PotentialGradient | None = None,
        task_energy: TaskEnergy | None = None,
        memory_targets: Mapping[str, Vec3] | None = None,
    ) -> Swarm3DState:
        """Advance the local-chart swarm by one bounded explicit-Euler step."""

        particles = state.particles
        adjacency = self.attention(particles)
        inward_targets = tuple(self.inward_target(particle) for particle in particles)
        memory_targets = memory_targets or {}
        updated: list[Particle3D] = []
        maximum_displacement = 0.0

        for i, particle in enumerate(particles):
            euclidean_gradient = (
                _vec3(task_gradient(particle), name="task gradient")
                if task_gradient is not None
                else (0.0, 0.0, 0.0)
            )
            metric_gradient = (
                _vec3(potential_gradient(particle), name="potential gradient")
                if potential_gradient is not None
                else euclidean_gradient
            )
            riemannian_gradient = local_riemannian_gradient(
                euclidean_gradient,
                metric_gradient,
                alpha=self.config.alpha_metric,
            )
            task_force = _scale(riemannian_gradient, -self.config.task_gain)
            inward_force = _scale(
                _sub(inward_targets[i], particle.position),
                self.config.inward_gain,
            )

            swarm_force: Vec3 = (0.0, 0.0, 0.0)
            neighbor_feature = [0.0] * len(particle.feature)
            for j, other in enumerate(particles):
                weight = adjacency[i][j]
                if weight == 0.0:
                    continue
                swarm_force = _add(
                    swarm_force,
                    _scale(_sub(other.position, particle.position), weight),
                )
                for k, value in enumerate(other.feature):
                    neighbor_feature[k] += weight * value
            swarm_force = _scale(swarm_force, self.config.swarm_gain)

            memory_force: Vec3 = (0.0, 0.0, 0.0)
            if particle.source_id in memory_targets:
                target = _vec3(memory_targets[particle.source_id], name="memory target")
                memory_force = _scale(
                    _sub(target, particle.position),
                    self.config.memory_gain,
                )

            force = _add(_add(task_force, inward_force), _add(swarm_force, memory_force))
            displacement = _cap_norm(
                _scale(force, self.config.dt),
                self.config.max_position_step,
            )
            new_position = _clamp_vec(
                _add(particle.position, displacement),
                self.config.position_bound,
            )
            actual_displacement = _norm(_sub(new_position, particle.position))
            maximum_displacement = max(maximum_displacement, actual_displacement)

            if len(particles) == 1:
                new_feature = particle.feature
            else:
                feature_rate = self.config.dt * self.config.feature_mix_gain
                new_feature = tuple(
                    current + feature_rate * (neighbor - current)
                    for current, neighbor in zip(particle.feature, neighbor_feature)
                )
            updated.append(
                Particle3D(
                    position=new_position,
                    feature=new_feature,
                    modality=particle.modality,
                    confidence=particle.confidence,
                    time_coordinate=particle.time_coordinate + self.config.dt,
                    source_id=particle.source_id,
                )
            )

        updated_particles = tuple(updated)
        updated_targets = tuple(self.inward_target(particle) for particle in updated_particles)
        updated_adjacency = self.attention(updated_particles)
        metrics = self._metrics(
            updated_particles,
            updated_targets,
            updated_adjacency,
            step=state.metrics.step + 1,
            task_energy=task_energy,
            max_displacement=maximum_displacement,
        )
        return Swarm3DState(updated_particles, metrics)

    def relax(
        self,
        particles: Sequence[Particle3D],
        *,
        task_gradient: TaskGradient | None = None,
        potential_gradient: PotentialGradient | None = None,
        task_energy: TaskEnergy | None = None,
        memory_targets: Mapping[str, Vec3] | None = None,
    ) -> Swarm3DResult:
        """Run bounded inward relaxation until tolerance or max_steps."""

        state = self.initial_state(particles, task_energy=task_energy)
        history = [state.metrics.energy]
        converged = False

        for _ in range(self.config.max_steps):
            state = self.step(
                state,
                task_gradient=task_gradient,
                potential_gradient=potential_gradient,
                task_energy=task_energy,
                memory_targets=memory_targets,
            )
            history.append(state.metrics.energy)
            movement_ok = state.metrics.max_displacement <= self.config.tolerance
            inward_ok = (
                self.config.inward_gain == 0.0
                or state.metrics.fixed_point_error <= self.config.tolerance
            )
            if movement_ok and inward_ok:
                converged = True
                break

        return Swarm3DResult(state=state, energy_history=tuple(history), converged=converged)


def electrical_rhs(
    voltages: Sequence[Vec3],
    phi_voltages: Sequence[Vec3],
    adjacency: Sequence[Sequence[float]],
    config: ElectricalAnalogueConfig,
    *,
    external_currents: Sequence[Vec3] | None = None,
) -> tuple[Vec3, ...]:
    """Return dV/dt for the RC analogue.

    Implements, component-wise,

        C dV_i/dt =
            g_phi (V_phi_i - V_i)
            + g_c sum_j A_ij (V_j - V_i)
            + I_ext_i.

    Voltages encode algorithmic state. This helper does not solve Maxwell's
    field equations and does not identify semantic coordinates with spacetime.
    """

    values = tuple(_vec3(value, name="voltage") for value in voltages)
    targets = tuple(_vec3(value, name="phi_voltage") for value in phi_voltages)
    if not values:
        raise ValueError("voltages must not be empty")
    if len(targets) != len(values):
        raise ValueError("phi_voltages must match voltage count")
    if len(adjacency) != len(values) or any(len(row) != len(values) for row in adjacency):
        raise ValueError("adjacency must be a square matrix matching voltage count")
    for row in adjacency:
        if any(not math.isfinite(float(weight)) or float(weight) < 0.0 for weight in row):
            raise ValueError("adjacency weights must be finite and non-negative")

    if external_currents is None:
        currents = ((0.0, 0.0, 0.0),) * len(values)
    else:
        currents = tuple(_vec3(value, name="external current") for value in external_currents)
        if len(currents) != len(values):
            raise ValueError("external_currents must match voltage count")

    derivatives: list[Vec3] = []
    for i, value in enumerate(values):
        feedback = _scale(
            _sub(targets[i], value),
            config.feedback_conductance,
        )
        coupling: Vec3 = (0.0, 0.0, 0.0)
        for j, other in enumerate(values):
            coupling = _add(
                coupling,
                _scale(_sub(other, value), float(adjacency[i][j])),
            )
        coupling = _scale(coupling, config.coupling_conductance)
        total_current = _add(_add(feedback, coupling), currents[i])
        derivatives.append(_scale(total_current, 1.0 / config.capacitance))
    return tuple(derivatives)


def electrical_step(
    voltages: Sequence[Vec3],
    phi_voltages: Sequence[Vec3],
    adjacency: Sequence[Sequence[float]],
    config: ElectricalAnalogueConfig,
    *,
    external_currents: Sequence[Vec3] | None = None,
) -> tuple[Vec3, ...]:
    """Advance the bounded RC analogue by one explicit-Euler integration step."""

    values = tuple(_vec3(value, name="voltage") for value in voltages)
    derivatives = electrical_rhs(
        values,
        phi_voltages,
        adjacency,
        config,
        external_currents=external_currents,
    )
    return tuple(
        _clamp_vec(_add(value, _scale(rate, config.dt)), config.voltage_bound)
        for value, rate in zip(values, derivatives)
    )


__all__ = [
    "ElectricalAnalogueConfig",
    "Feature",
    "InwardMultimodalSwarm3D",
    "Matrix3",
    "Modality",
    "ModalityCodec",
    "Particle3D",
    "Swarm3DConfig",
    "Swarm3DMetrics",
    "Swarm3DResult",
    "Swarm3DState",
    "Vec3",
    "electrical_rhs",
    "electrical_step",
    "local_riemannian_gradient",
    "riemannian_metric_inverse",
]
