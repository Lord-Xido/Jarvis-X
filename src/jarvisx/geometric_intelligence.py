"""Bounded 3D geometric autoencoding intelligence reference kernel.

This module operationalizes ADR-013 with a dependency-free, deterministic
reference implementation.  It intentionally uses a flat latent metric
(`G = I`) and a local six-neighbour stencil so the mechanics are explicit and
falsifiable.  A configured sub-nanosecond period is a design target for one
local primitive; this module does not claim that Python executes at that rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

C_M_PER_S = 299_792_458.0


def _validate_real(name: str, value: float, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return lower if value < lower else upper if value > upper else value


def coordinate_to_index(x: int, y: int, z: int, side: int) -> int:
    if isinstance(side, bool) or not isinstance(side, int) or side < 2:
        raise ValueError("side must be an integer >= 2")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (x, y, z)):
        raise TypeError("coordinates must be integers")
    if not all(0 <= v < side for v in (x, y, z)):
        raise ValueError("coordinate is outside the lattice")
    return (x * side + y) * side + z


def index_to_coordinate(index: int, side: int) -> tuple[int, int, int]:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if not 0 <= index < side**3:
        raise ValueError("index is outside the lattice")
    x, remainder = divmod(index, side * side)
    y, z = divmod(remainder, side)
    return x, y, z


def six_neighbor_laplacian(field: Sequence[float], side: int) -> tuple[float, ...]:
    """Return a clamped-boundary six-neighbour Laplacian.

    Missing neighbours are replaced by the center value.  This implements a
    zero-normal-gradient (Neumann-like) boundary in the finite-difference
    reference and keeps constant fields exactly invariant.
    """

    if len(field) != side**3:
        raise ValueError("field length must equal side**3")
    values = tuple(_validate_real("field value", value) for value in field)
    result: list[float] = []
    for index, center in enumerate(values):
        x, y, z = index_to_coordinate(index, side)
        total = 0.0
        for dx, dy, dz in (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < side and 0 <= ny < side and 0 <= nz < side:
                total += values[coordinate_to_index(nx, ny, nz, side)]
            else:
                total += center
        result.append(total - 6.0 * center)
    return tuple(result)


def encode_scalar(value: float) -> float:
    """Bounded scalar encoder used by the reference kernel."""

    return math.tanh(_validate_real("value", value))


def decode_scalar(latent: float) -> float:
    """Inverse of ``tanh`` on a numerically safe open interval."""

    bounded = _clamp(_validate_real("latent", latent), -0.999_999_999, 0.999_999_999)
    return math.atanh(bounded)


def encode_field(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(encode_scalar(value) for value in values)


def decode_field(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(decode_scalar(value) for value in values)


@dataclass(frozen=True)
class GeometricIntelligenceConfig:
    side: int = 4
    local_period_ps: float = 100.0
    damping: float = 0.08
    restoring_gain: float = 0.35
    diffusion_gain: float = 0.04
    memory_gain: float = 0.05
    memory_decay: float = 0.90
    major_frequency_hz: float = 1.0e6
    micro_frequency_hz: float = 1.0e9
    velocity_limit: float = 0.25

    def __post_init__(self) -> None:
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side < 2:
            raise ValueError("side must be an integer >= 2")
        period = _validate_real("local_period_ps", self.local_period_ps, positive=True)
        if period >= 1_000.0:
            raise ValueError("local_period_ps must be < 1000 ps for the ADR-013 micro-loop target")
        for name in (
            "damping",
            "restoring_gain",
            "diffusion_gain",
            "memory_gain",
            "major_frequency_hz",
            "micro_frequency_hz",
            "velocity_limit",
        ):
            _validate_real(name, getattr(self, name), positive=True)
        decay = _validate_real("memory_decay", self.memory_decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("memory_decay must be in [0, 1)")

    @property
    def node_count(self) -> int:
        return self.side**3

    @property
    def local_period_s(self) -> float:
        return self.local_period_ps * 1.0e-12

    @property
    def absolute_propagation_bound_m(self) -> float:
        """Vacuum light-travel ceiling for the declared local period."""

        return C_M_PER_S * self.local_period_s


@dataclass(frozen=True)
class GeometricState:
    latent: tuple[float, ...]
    velocity: tuple[float, ...]
    memory: tuple[float, ...]
    major_phase: float = 0.0
    micro_phase: float = 0.0
    step_index: int = 0


@dataclass(frozen=True)
class GeometricTransition:
    previous: GeometricState
    current: GeometricState
    encoded_observation: tuple[float, ...]
    reconstruction: tuple[float, ...]
    residual: tuple[float, ...]
    laplacian: tuple[float, ...]
    reconstruction_mse: float
    residual_rms: float


class GeometricIntelligenceKernel:
    """Deterministic flat-manifold specialization of the ADR-013 micro-loop."""

    def __init__(self, config: GeometricIntelligenceConfig | None = None) -> None:
        self.config = config or GeometricIntelligenceConfig()
        zeros = (0.0,) * self.config.node_count
        self._state = GeometricState(latent=zeros, velocity=zeros, memory=zeros)

    @property
    def state(self) -> GeometricState:
        return self._state

    def reset(self) -> GeometricState:
        zeros = (0.0,) * self.config.node_count
        self._state = GeometricState(latent=zeros, velocity=zeros, memory=zeros)
        return self._state

    def _validate_observation(self, observation: Sequence[float]) -> tuple[float, ...]:
        if len(observation) != self.config.node_count:
            raise ValueError("observation length must equal side**3")
        return tuple(_validate_real("observation value", value) for value in observation)

    def step(self, observation: Sequence[float]) -> GeometricTransition:
        """Advance one bounded local geometric state transition.

        The reference uses a normalized integration step of one kernel tick.
        ``local_period_ps`` remains explicit metadata for phase and propagation
        calculations; Python wall-clock execution is not interpreted as that
        hardware latency.
        """

        obs = self._validate_observation(observation)
        encoded = encode_field(obs)
        previous = self._state
        lap = six_neighbor_laplacian(previous.latent, self.config.side)

        next_velocity: list[float] = []
        next_latent: list[float] = []
        for z, v, target, memory_value, lap_value in zip(
            previous.latent,
            previous.velocity,
            encoded,
            previous.memory,
            lap,
            strict=True,
        ):
            acceleration = (
                -self.config.damping * v
                -self.config.restoring_gain * (z - target)
                +self.config.diffusion_gain * lap_value
                -self.config.memory_gain * memory_value
            )
            v_next = _clamp(
                v + acceleration,
                -self.config.velocity_limit,
                self.config.velocity_limit,
            )
            z_next = _clamp(z + v_next, -0.999_999_999, 0.999_999_999)
            next_velocity.append(v_next)
            next_latent.append(z_next)

        reconstruction = decode_field(next_latent)
        residual = tuple(x - xhat for x, xhat in zip(obs, reconstruction, strict=True))
        next_memory = tuple(
            self.config.memory_decay * old + (1.0 - self.config.memory_decay) * error
            for old, error in zip(previous.memory, residual, strict=True)
        )

        dt = self.config.local_period_s
        major_phase = math.fmod(
            previous.major_phase + 2.0 * math.pi * self.config.major_frequency_hz * dt,
            2.0 * math.pi,
        )
        micro_phase = math.fmod(
            previous.micro_phase + 2.0 * math.pi * self.config.micro_frequency_hz * dt,
            2.0 * math.pi,
        )

        current = GeometricState(
            latent=tuple(next_latent),
            velocity=tuple(next_velocity),
            memory=next_memory,
            major_phase=major_phase,
            micro_phase=micro_phase,
            step_index=previous.step_index + 1,
        )
        self._state = current

        mse = sum(error * error for error in residual) / len(residual)
        rms = math.sqrt(mse)
        return GeometricTransition(
            previous=previous,
            current=current,
            encoded_observation=encoded,
            reconstruction=reconstruction,
            residual=residual,
            laplacian=lap,
            reconstruction_mse=mse,
            residual_rms=rms,
        )
