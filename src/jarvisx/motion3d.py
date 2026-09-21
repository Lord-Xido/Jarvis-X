"""Deterministic Dr Moagi 3D motion engine.

The engine advances rigid-body state on SE(3) using bounded force/torque
integration, optional observation residuals, retained correction memory, and a
constraint projector. It is intentionally dependency-free and Python 3.8+
compatible.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]  # (w, x, y, z)

_EPSILON = 1e-12


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("%s must be finite" % name)
    return result


def _vec3(values: Sequence[float], name: str) -> Vec3:
    if len(values) != 3:
        raise ValueError("%s must contain exactly three values" % name)
    result = tuple(_finite(value, name) for value in values)
    return (result[0], result[1], result[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(values: Vec3, scalar: float) -> Vec3:
    return (values[0] * scalar, values[1] * scalar, values[2] * scalar)


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(values: Vec3) -> float:
    return math.sqrt(_dot(values, values))


def _clamp_norm(values: Vec3, maximum: float) -> Vec3:
    magnitude = _norm(values)
    if magnitude <= maximum or magnitude <= _EPSILON:
        return values
    return _scale(values, maximum / magnitude)


def quaternion_normalize(value: Sequence[float]) -> Quaternion:
    if len(value) != 4:
        raise ValueError("orientation must contain exactly four values")
    raw = tuple(_finite(component, "orientation") for component in value)
    magnitude = math.sqrt(sum(component * component for component in raw))
    if magnitude <= _EPSILON:
        raise ValueError("orientation quaternion must be non-zero")
    return (
        raw[0] / magnitude,
        raw[1] / magnitude,
        raw[2] / magnitude,
        raw[3] / magnitude,
    )


def quaternion_multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quaternion_delta(angular_velocity: Vec3, dt: float) -> Quaternion:
    angle = _norm(angular_velocity) * dt
    if angle <= _EPSILON:
        half_dt = 0.5 * dt
        return quaternion_normalize(
            (
                1.0,
                angular_velocity[0] * half_dt,
                angular_velocity[1] * half_dt,
                angular_velocity[2] * half_dt,
            )
        )
    axis = _scale(angular_velocity, 1.0 / _norm(angular_velocity))
    half = 0.5 * angle
    sine = math.sin(half)
    return (math.cos(half), axis[0] * sine, axis[1] * sine, axis[2] * sine)


def quaternion_nlerp(a: Quaternion, b: Quaternion, weight: float) -> Quaternion:
    weight = max(0.0, min(1.0, float(weight)))
    if sum(a[index] * b[index] for index in range(4)) < 0.0:
        b = (-b[0], -b[1], -b[2], -b[3])
    return quaternion_normalize(
        tuple((1.0 - weight) * a[index] + weight * b[index] for index in range(4))
    )


@dataclass(frozen=True)
class MotionConstraints:
    world_min: Vec3 = (-1_000_000.0, -1_000_000.0, -1_000_000.0)
    world_max: Vec3 = (1_000_000.0, 1_000_000.0, 1_000_000.0)
    max_speed: float = 10_000.0
    max_acceleration: float = 10_000.0
    max_angular_speed: float = 1_000.0
    max_angular_acceleration: float = 1_000.0
    floor_z: Optional[float] = None
    restitution: float = 0.0
    max_dt: float = 1.0

    def validate(self) -> None:
        minimum = _vec3(self.world_min, "world_min")
        maximum = _vec3(self.world_max, "world_max")
        if any(minimum[index] > maximum[index] for index in range(3)):
            raise ValueError("world_min must not exceed world_max")
        for name in (
            "max_speed",
            "max_acceleration",
            "max_angular_speed",
            "max_angular_acceleration",
            "max_dt",
        ):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError("%s must be positive" % name)
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError("restitution must be inside [0, 1]")
        if self.floor_z is not None:
            _finite(self.floor_z, "floor_z")


@dataclass(frozen=True)
class MotionObservation:
    position: Optional[Vec3] = None
    orientation: Optional[Quaternion] = None
    velocity: Optional[Vec3] = None
    angular_velocity: Optional[Vec3] = None
    confidence: float = 1.0

    def validate(self) -> None:
        if self.position is not None:
            _vec3(self.position, "observation.position")
        if self.orientation is not None:
            quaternion_normalize(self.orientation)
        if self.velocity is not None:
            _vec3(self.velocity, "observation.velocity")
        if self.angular_velocity is not None:
            _vec3(self.angular_velocity, "observation.angular_velocity")
        if not 0.0 <= _finite(self.confidence, "observation.confidence") <= 1.0:
            raise ValueError("observation confidence must be inside [0, 1]")


@dataclass(frozen=True)
class MotionState:
    position: Vec3 = (0.0, 0.0, 0.0)
    orientation: Quaternion = (1.0, 0.0, 0.0, 0.0)
    velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    acceleration: Vec3 = (0.0, 0.0, 0.0)
    angular_acceleration: Vec3 = (0.0, 0.0, 0.0)
    force: Vec3 = (0.0, 0.0, 0.0)
    torque: Vec3 = (0.0, 0.0, 0.0)
    mass: float = 1.0
    inertia: Vec3 = (1.0, 1.0, 1.0)
    memory_position: Vec3 = (0.0, 0.0, 0.0)
    memory_velocity: Vec3 = (0.0, 0.0, 0.0)
    time_seconds: float = 0.0
    step_index: int = 0

    def validate(self) -> None:
        _vec3(self.position, "position")
        quaternion_normalize(self.orientation)
        _vec3(self.velocity, "velocity")
        _vec3(self.angular_velocity, "angular_velocity")
        _vec3(self.acceleration, "acceleration")
        _vec3(self.angular_acceleration, "angular_acceleration")
        _vec3(self.force, "force")
        _vec3(self.torque, "torque")
        _vec3(self.memory_position, "memory_position")
        _vec3(self.memory_velocity, "memory_velocity")
        if _finite(self.mass, "mass") <= 0.0:
            raise ValueError("mass must be positive")
        inertia = _vec3(self.inertia, "inertia")
        if any(value <= 0.0 for value in inertia):
            raise ValueError("inertia components must be positive")
        if _finite(self.time_seconds, "time_seconds") < 0.0:
            raise ValueError("time_seconds must be non-negative")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True)
class MotionStepResult:
    state: MotionState
    residual_position: Vec3
    residual_velocity: Vec3
    kinetic_energy: float
    state_hash: str

    def snapshot(self) -> Dict[str, object]:
        return {
            "state": asdict(self.state),
            "residual_position": self.residual_position,
            "residual_velocity": self.residual_velocity,
            "kinetic_energy": self.kinetic_energy,
            "state_hash": self.state_hash,
        }


class DrMoagiMotionEngine:
    """Bounded SE(3) propagation with residual correction and retained memory."""

    def __init__(
        self,
        constraints: Optional[MotionConstraints] = None,
        *,
        position_gain: float = 0.25,
        velocity_gain: float = 0.20,
        orientation_gain: float = 0.20,
        angular_velocity_gain: float = 0.20,
        memory_retention: float = 0.95,
        memory_gain: float = 0.05,
    ) -> None:
        self.constraints = constraints or MotionConstraints()
        self.constraints.validate()
        for name, value in (
            ("position_gain", position_gain),
            ("velocity_gain", velocity_gain),
            ("orientation_gain", orientation_gain),
            ("angular_velocity_gain", angular_velocity_gain),
            ("memory_retention", memory_retention),
            ("memory_gain", memory_gain),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must be inside [0, 1]" % name)
        self.position_gain = float(position_gain)
        self.velocity_gain = float(velocity_gain)
        self.orientation_gain = float(orientation_gain)
        self.angular_velocity_gain = float(angular_velocity_gain)
        self.memory_retention = float(memory_retention)
        self.memory_gain = float(memory_gain)

    @staticmethod
    def _state_hash(state: MotionState) -> str:
        payload = asdict(state)

        def encode(value):
            if isinstance(value, float):
                return value.hex()
            if isinstance(value, tuple):
                return [encode(item) for item in value]
            if isinstance(value, list):
                return [encode(item) for item in value]
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            return value

        canonical = json.dumps(
            encode(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()

    def _project(
        self,
        position: Vec3,
        orientation: Quaternion,
        velocity: Vec3,
        angular_velocity: Vec3,
    ) -> Tuple[Vec3, Quaternion, Vec3, Vec3]:
        minimum = self.constraints.world_min
        maximum = self.constraints.world_max
        bounded_position = tuple(
            max(minimum[index], min(maximum[index], position[index]))
            for index in range(3)
        )
        bounded_velocity = _clamp_norm(velocity, self.constraints.max_speed)
        bounded_angular = _clamp_norm(
            angular_velocity, self.constraints.max_angular_speed
        )
        if (
            self.constraints.floor_z is not None
            and bounded_position[2] < self.constraints.floor_z
        ):
            bounded_position = (
                bounded_position[0],
                bounded_position[1],
                self.constraints.floor_z,
            )
            if bounded_velocity[2] < 0.0:
                bounded_velocity = (
                    bounded_velocity[0],
                    bounded_velocity[1],
                    -bounded_velocity[2] * self.constraints.restitution,
                )
        return (
            _vec3(bounded_position, "projected.position"),
            quaternion_normalize(orientation),
            _vec3(bounded_velocity, "projected.velocity"),
            _vec3(bounded_angular, "projected.angular_velocity"),
        )

    def step(
        self,
        state: MotionState,
        dt: float,
        *,
        external_force: Vec3 = (0.0, 0.0, 0.0),
        external_torque: Vec3 = (0.0, 0.0, 0.0),
        observation: Optional[MotionObservation] = None,
    ) -> MotionStepResult:
        state.validate()
        dt = _finite(dt, "dt")
        if dt <= 0.0 or dt > self.constraints.max_dt:
            raise ValueError("dt must be inside (0, max_dt]")
        force = _add(
            _vec3(state.force, "force"),
            _vec3(external_force, "external_force"),
        )
        torque = _add(
            _vec3(state.torque, "torque"),
            _vec3(external_torque, "external_torque"),
        )
        acceleration = _clamp_norm(
            _scale(force, 1.0 / state.mass), self.constraints.max_acceleration
        )
        predicted_velocity = _clamp_norm(
            _add(state.velocity, _scale(acceleration, dt)),
            self.constraints.max_speed,
        )
        predicted_position = _add(state.position, _scale(predicted_velocity, dt))

        inertia_omega = (
            state.inertia[0] * state.angular_velocity[0],
            state.inertia[1] * state.angular_velocity[1],
            state.inertia[2] * state.angular_velocity[2],
        )
        gyroscopic = _cross(state.angular_velocity, inertia_omega)
        angular_acceleration = (
            (torque[0] - gyroscopic[0]) / state.inertia[0],
            (torque[1] - gyroscopic[1]) / state.inertia[1],
            (torque[2] - gyroscopic[2]) / state.inertia[2],
        )
        angular_acceleration = _clamp_norm(
            angular_acceleration, self.constraints.max_angular_acceleration
        )
        predicted_angular_velocity = _clamp_norm(
            _add(state.angular_velocity, _scale(angular_acceleration, dt)),
            self.constraints.max_angular_speed,
        )
        midpoint_angular_velocity = _scale(
            _add(state.angular_velocity, predicted_angular_velocity), 0.5
        )
        predicted_orientation = quaternion_normalize(
            quaternion_multiply(
                quaternion_normalize(state.orientation),
                quaternion_delta(midpoint_angular_velocity, dt),
            )
        )

        residual_position = (0.0, 0.0, 0.0)
        residual_velocity = (0.0, 0.0, 0.0)
        corrected_position = _add(
            predicted_position, _scale(state.memory_position, self.memory_gain)
        )
        corrected_velocity = _add(
            predicted_velocity, _scale(state.memory_velocity, self.memory_gain)
        )
        corrected_orientation = predicted_orientation
        corrected_angular_velocity = predicted_angular_velocity
        confidence = 0.0
        if observation is not None:
            observation.validate()
            confidence = observation.confidence
            if observation.position is not None:
                residual_position = _sub(observation.position, predicted_position)
                corrected_position = _add(
                    corrected_position,
                    _scale(residual_position, self.position_gain * confidence),
                )
            if observation.velocity is not None:
                residual_velocity = _sub(observation.velocity, predicted_velocity)
                corrected_velocity = _add(
                    corrected_velocity,
                    _scale(residual_velocity, self.velocity_gain * confidence),
                )
            if observation.orientation is not None:
                corrected_orientation = quaternion_nlerp(
                    predicted_orientation,
                    quaternion_normalize(observation.orientation),
                    self.orientation_gain * confidence,
                )
            if observation.angular_velocity is not None:
                angular_residual = _sub(
                    observation.angular_velocity, predicted_angular_velocity
                )
                corrected_angular_velocity = _add(
                    predicted_angular_velocity,
                    _scale(
                        angular_residual,
                        self.angular_velocity_gain * confidence,
                    ),
                )

        memory_position = _add(
            _scale(state.memory_position, self.memory_retention),
            _scale(residual_position, (1.0 - self.memory_retention) * confidence),
        )
        memory_velocity = _add(
            _scale(state.memory_velocity, self.memory_retention),
            _scale(residual_velocity, (1.0 - self.memory_retention) * confidence),
        )
        (
            corrected_position,
            corrected_orientation,
            corrected_velocity,
            corrected_angular_velocity,
        ) = self._project(
            corrected_position,
            corrected_orientation,
            corrected_velocity,
            corrected_angular_velocity,
        )
        next_state = MotionState(
            position=corrected_position,
            orientation=corrected_orientation,
            velocity=corrected_velocity,
            angular_velocity=corrected_angular_velocity,
            acceleration=acceleration,
            angular_acceleration=angular_acceleration,
            force=state.force,
            torque=state.torque,
            mass=state.mass,
            inertia=state.inertia,
            memory_position=memory_position,
            memory_velocity=memory_velocity,
            time_seconds=state.time_seconds + dt,
            step_index=state.step_index + 1,
        )
        next_state.validate()
        kinetic_energy = 0.5 * state.mass * _dot(
            corrected_velocity, corrected_velocity
        ) + 0.5 * sum(
            state.inertia[index]
            * corrected_angular_velocity[index]
            * corrected_angular_velocity[index]
            for index in range(3)
        )
        return MotionStepResult(
            state=next_state,
            residual_position=residual_position,
            residual_velocity=residual_velocity,
            kinetic_energy=kinetic_energy,
            state_hash=self._state_hash(next_state),
        )

    def run(
        self,
        initial: MotionState,
        dt: float,
        steps: int,
        *,
        external_force: Vec3 = (0.0, 0.0, 0.0),
        external_torque: Vec3 = (0.0, 0.0, 0.0),
        observations: Optional[Iterable[Optional[MotionObservation]]] = None,
    ) -> Tuple[MotionStepResult, ...]:
        if steps <= 0:
            raise ValueError("steps must be positive")
        observation_values = tuple(observations or ())
        if observation_values and len(observation_values) != steps:
            raise ValueError("observations must match the requested step count")
        state = initial
        results = []
        for index in range(steps):
            result = self.step(
                state,
                dt,
                external_force=external_force,
                external_torque=external_torque,
                observation=(
                    observation_values[index] if observation_values else None
                ),
            )
            results.append(result)
            state = result.state
        return tuple(results)
