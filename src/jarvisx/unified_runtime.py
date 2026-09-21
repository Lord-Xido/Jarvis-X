"""Bounded shared-state coordination runtime for JARVIS-X.

This module unifies the symbolic Psi-Phi-Lambda-Omega-Theta vocabulary with an
explicit, measurable state transition. It is intentionally dependency-free and
does not replace the canonical CodexVM, ANN implementations, policy engine, or
external-world verification. Instead it provides a deterministic coordination
state that those surfaces can observe and drive.
"""

from __future__ import annotations

import hashlib
import math
import struct
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

Vector = tuple[float, ...]


@dataclass(frozen=True)
class UnifiedRuntimeConfig:
    """Numerical and resource bounds for one shared-state runtime."""

    latent_block_size: int = 2
    omega_retention: float = 0.85
    psi_memory_gain: float = 0.25
    theta_gain: float = 0.05
    theta_limit: float = 1.0
    stability_epsilon: float = 1.0e-6
    max_dimensions: int = 256
    value_limit: float = 1.0e6
    resource_weight: float = 1.0e-3

    def __post_init__(self) -> None:
        if isinstance(self.latent_block_size, bool) or not isinstance(
            self.latent_block_size, int
        ):
            raise TypeError("latent_block_size must be an integer")
        if self.latent_block_size < 1:
            raise ValueError("latent_block_size must be at least 1")
        if isinstance(self.max_dimensions, bool) or not isinstance(self.max_dimensions, int):
            raise TypeError("max_dimensions must be an integer")
        if not 1 <= self.max_dimensions <= 4096:
            raise ValueError("max_dimensions must be in [1, 4096]")

        bounded_unit = {
            "omega_retention": self.omega_retention,
            "psi_memory_gain": self.psi_memory_gain,
        }
        for name, value in bounded_unit.items():
            _finite_number(name, value)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        positive = {
            "theta_gain": self.theta_gain,
            "theta_limit": self.theta_limit,
            "stability_epsilon": self.stability_epsilon,
            "value_limit": self.value_limit,
            "resource_weight": self.resource_weight,
        }
        for name, value in positive.items():
            _finite_number(name, value)
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True)
class UnifiedRuntimeState:
    """One committed Psi-Phi-Lambda-Omega-Theta state."""

    cycle: int
    psi: Vector
    phi: Vector
    lambda_state: Vector
    omega: Vector
    theta: float
    latent: Vector
    reconstruction: Vector
    error: Vector
    reconstruction_mse: float
    latent_cycle_mse: float
    state_delta: float
    h_mmm: float
    verified: bool
    stable: bool
    state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "psi": list(self.psi),
            "phi": list(self.phi),
            "lambda": list(self.lambda_state),
            "omega": list(self.omega),
            "theta": self.theta,
            "latent": list(self.latent),
            "reconstruction": list(self.reconstruction),
            "error": list(self.error),
            "metrics": {
                "reconstruction_mse": self.reconstruction_mse,
                "latent_cycle_mse": self.latent_cycle_mse,
                "state_delta": self.state_delta,
                "h_mmm": self.h_mmm,
            },
            "verified": self.verified,
            "stable": self.stable,
            "state_hash": self.state_hash,
        }


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validate_values(values: Sequence[float], config: UnifiedRuntimeConfig) -> Vector:
    if not values:
        raise ValueError("values must not be empty")
    if len(values) > config.max_dimensions:
        raise ValueError(f"values exceed max_dimensions={config.max_dimensions}")

    cleaned: list[float] = []
    for index, value in enumerate(values):
        number = _finite_number(f"values[{index}]", value)
        if abs(number) > config.value_limit:
            raise ValueError(
                f"values[{index}] exceeds configured magnitude limit {config.value_limit}"
            )
        cleaned.append(number)
    return tuple(cleaned)


def _mse(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("MSE vectors must have equal length")
    if not left:
        return 0.0
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) / len(left)


def _rms_delta(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(_mse(left, right))


def _encode_blocks(values: Vector, block_size: int) -> Vector:
    latent: list[float] = []
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        latent.append(sum(block) / len(block))
    return tuple(latent)


def _decode_blocks(latent: Vector, length: int, block_size: int) -> Vector:
    decoded: list[float] = []
    for value in latent:
        decoded.extend([value] * min(block_size, length - len(decoded)))
        if len(decoded) >= length:
            break
    return tuple(decoded)


def _state_hash(
    *,
    cycle: int,
    vectors: Sequence[Vector],
    scalars: Sequence[float],
    verified: bool,
    stable: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack(">Q", cycle))
    for vector in vectors:
        digest.update(struct.pack(">I", len(vector)))
        for value in vector:
            digest.update(struct.pack(">d", float(value)))
    for value in scalars:
        digest.update(struct.pack(">d", float(value)))
    digest.update(bytes((int(verified), int(stable))))
    return digest.hexdigest()


class UnifiedRuntime:
    """Deterministic inward coordination loop over a bounded numeric observation."""

    def __init__(self, config: UnifiedRuntimeConfig | None = None) -> None:
        self.config = config or UnifiedRuntimeConfig()
        self._state: UnifiedRuntimeState | None = None
        self._lock = threading.RLock()

    @property
    def state(self) -> UnifiedRuntimeState | None:
        with self._lock:
            return self._state

    def reset(self) -> None:
        with self._lock:
            self._state = None

    def step(self, values: Sequence[float]) -> UnifiedRuntimeState:
        observation = _validate_values(values, self.config)
        with self._lock:
            previous = self._state
            if previous is not None and len(previous.psi) != len(observation):
                raise ValueError("observation dimensionality cannot change within a runtime")

            n = len(observation)
            previous_omega = previous.omega if previous is not None else (0.0,) * n
            previous_theta = previous.theta if previous is not None else 0.0
            previous_psi = previous.psi if previous is not None else (0.0,) * n

            psi = tuple(
                math.tanh(value + self.config.psi_memory_gain * memory)
                for value, memory in zip(observation, previous_omega)
            )

            if n == 1:
                phi = psi
            else:
                phi = tuple(
                    0.5 * psi[index]
                    + 0.25 * psi[(index - 1) % n]
                    + 0.25 * psi[(index + 1) % n]
                    for index in range(n)
                )

            lambda_state = tuple(
                math.tanh(structured + previous_theta * state)
                for structured, state in zip(phi, psi)
            )
            latent = _encode_blocks(lambda_state, self.config.latent_block_size)
            reconstruction = _decode_blocks(
                latent, len(lambda_state), self.config.latent_block_size
            )
            error = tuple(
                state - decoded for state, decoded in zip(lambda_state, reconstruction)
            )

            reconstruction_mse = _mse(lambda_state, reconstruction)
            cycle_latent = _encode_blocks(reconstruction, self.config.latent_block_size)
            latent_cycle_mse = _mse(latent, cycle_latent)
            state_delta = _rms_delta(psi, previous_psi)

            omega = tuple(
                self.config.omega_retention * memory
                + (1.0 - self.config.omega_retention) * residual
                for memory, residual in zip(previous_omega, error)
            )

            theta_drive = reconstruction_mse - self.config.stability_epsilon
            theta = max(
                -self.config.theta_limit,
                min(
                    self.config.theta_limit,
                    previous_theta + self.config.theta_gain * theta_drive,
                ),
            )

            resource_penalty = self.config.resource_weight * n / self.config.max_dimensions
            h_mmm = (
                reconstruction_mse
                + 0.25 * latent_cycle_mse
                + 0.10 * state_delta
                + resource_penalty
            )

            vectors = (
                psi,
                phi,
                lambda_state,
                omega,
                latent,
                reconstruction,
                error,
            )
            scalars = (
                theta,
                reconstruction_mse,
                latent_cycle_mse,
                state_delta,
                h_mmm,
            )
            verified = all(
                math.isfinite(value) for vector in vectors for value in vector
            ) and all(math.isfinite(value) for value in scalars)
            stable = (
                verified
                and reconstruction_mse <= self.config.stability_epsilon
                and state_delta <= self.config.stability_epsilon
            )
            cycle = 1 if previous is None else previous.cycle + 1
            digest = _state_hash(
                cycle=cycle,
                vectors=vectors,
                scalars=scalars,
                verified=verified,
                stable=stable,
            )
            committed = UnifiedRuntimeState(
                cycle=cycle,
                psi=psi,
                phi=phi,
                lambda_state=lambda_state,
                omega=omega,
                theta=theta,
                latent=latent,
                reconstruction=reconstruction,
                error=error,
                reconstruction_mse=reconstruction_mse,
                latent_cycle_mse=latent_cycle_mse,
                state_delta=state_delta,
                h_mmm=h_mmm,
                verified=verified,
                stable=stable,
                state_hash=digest,
            )
            self._state = committed
            return committed


class UnifiedRuntimeRegistry:
    """Bounded in-process registry for browser/API runtime sessions."""

    def __init__(self, max_sessions: int = 16) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
            raise TypeError("max_sessions must be an integer")
        if not 1 <= max_sessions <= 256:
            raise ValueError("max_sessions must be in [1, 256]")
        self.max_sessions = max_sessions
        self._sessions: dict[str, UnifiedRuntime] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def create(self, config: UnifiedRuntimeConfig | None = None) -> dict[str, Any]:
        identifier = uuid.uuid4().hex
        runtime = UnifiedRuntime(config)
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                oldest = self._order.pop(0)
                self._sessions.pop(oldest, None)
            self._sessions[identifier] = runtime
            self._order.append(identifier)
        return {
            "session_id": identifier,
            "config": _config_dict(runtime.config),
            "state": None,
        }

    def _get(self, session_id: str) -> UnifiedRuntime:
        with self._lock:
            runtime = self._sessions.get(session_id)
        if runtime is None:
            raise KeyError(session_id)
        return runtime

    def step(self, session_id: str, values: Sequence[float]) -> dict[str, Any]:
        runtime = self._get(session_id)
        state = runtime.step(values)
        return {
            "session_id": session_id,
            "config": _config_dict(runtime.config),
            "state": state.to_dict(),
        }

    def status(self, session_id: str) -> dict[str, Any]:
        runtime = self._get(session_id)
        state = runtime.state
        return {
            "session_id": session_id,
            "config": _config_dict(runtime.config),
            "state": None if state is None else state.to_dict(),
        }

    def reset(self, session_id: str) -> dict[str, Any]:
        runtime = self._get(session_id)
        runtime.reset()
        return self.status(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            if session_id in self._order:
                self._order.remove(session_id)
            return existed


def _config_dict(config: UnifiedRuntimeConfig) -> dict[str, Any]:
    return {
        "latent_block_size": config.latent_block_size,
        "omega_retention": config.omega_retention,
        "psi_memory_gain": config.psi_memory_gain,
        "theta_gain": config.theta_gain,
        "theta_limit": config.theta_limit,
        "stability_epsilon": config.stability_epsilon,
        "max_dimensions": config.max_dimensions,
        "value_limit": config.value_limit,
        "resource_weight": config.resource_weight,
    }
