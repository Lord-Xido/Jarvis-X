"""Bounded DM-vOmegaXi+ pulse-domain mechatronic control reference.

The module turns the symbolic Psi -> Phi -> Lambda -> Omega -> Theta chain
into deterministic software primitives.  It emits hardware-neutral H-bridge
commands; it does not claim transistor, gate-driver, or mechanical timing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _require_bit(value: int) -> int:
    if value not in (0, 1):
        raise ValueError("bit values must be 0 or 1")
    return value


def xnor_popcount(lhs: Sequence[int], rhs: Sequence[int]) -> int:
    """Count equal bit positions in two non-empty, equal-width vectors."""

    if not lhs or len(lhs) != len(rhs):
        raise ValueError("vectors must be non-empty and have equal width")
    return sum(_require_bit(a) == _require_bit(b) for a, b in zip(lhs, rhs))


def bipolar_dot(lhs: Sequence[int], rhs: Sequence[int]) -> int:
    """Exact {-1,+1} dot product represented through XNOR/popcount."""

    return 2 * xnor_popcount(lhs, rhs) - len(lhs)


class DeltaSigmaBank:
    """First-order one-bit delta-sigma description operator."""

    def __init__(self, channels: int) -> None:
        if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
            raise ValueError("channels must be a positive integer")
        self._integrators = [0.0] * channels

    @property
    def channels(self) -> int:
        return len(self._integrators)

    def encode(self, values: Sequence[float]) -> tuple[int, ...]:
        if len(values) != self.channels:
            raise ValueError("sensor vector width does not match channel count")
        bits: list[int] = []
        for index, raw in enumerate(values):
            value = float(raw)
            if not -1.0 <= value <= 1.0:
                raise ValueError("normalized sensor values must be within [-1, 1]")
            self._integrators[index] += value
            bit = 1 if self._integrators[index] >= 0.0 else 0
            self._integrators[index] -= 1.0 if bit else -1.0
            bits.append(bit)
        return tuple(bits)


def rotate_u16(value: int, places: int) -> int:
    """Rotate a value inside an explicit 16-bit recurrent register."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if isinstance(places, bool) or not isinstance(places, int):
        raise TypeError("places must be an integer")
    places %= 16
    value &= 0xFFFF
    return ((value << places) | (value >> ((16 - places) % 16))) & 0xFFFF


@dataclass
class RecurrentRegister:
    value: int = 0
    rotation: int = 1

    def update(self, bits: Sequence[int], error_bit: int) -> int:
        injection = 0
        for index, bit in enumerate(bits[:16]):
            injection |= _require_bit(bit) << index
        injection ^= _require_bit(error_bit) << 15
        self.value = rotate_u16(self.value, self.rotation) ^ injection
        self.value &= 0xFFFF
        return self.value


@dataclass(frozen=True)
class SafetyLimits:
    max_duty_ppm: int = 800_000
    max_slew_ppm: int = 100_000
    reversal_dead_ticks: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.max_duty_ppm <= 1_000_000:
            raise ValueError("max_duty_ppm must be within [0, 1_000_000]")
        if not 0 <= self.max_slew_ppm <= 1_000_000:
            raise ValueError("max_slew_ppm must be within [0, 1_000_000]")
        if self.reversal_dead_ticks < 0:
            raise ValueError("reversal_dead_ticks must be non-negative")


@dataclass(frozen=True)
class GateCommand:
    signed_duty_ppm: int
    left_high_pwm: int
    left_low: bool
    right_high_pwm: int
    right_low: bool
    inhibited: bool
    reason: str

    @property
    def shoot_through_safe(self) -> bool:
        return not (
            (self.left_high_pwm and self.left_low)
            or (self.right_high_pwm and self.right_low)
        )


class ConstraintGovernor:
    """Theta admissibility layer with clamp, slew, stop, and reversal dead time."""

    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()
        self._last_duty = 0
        self._dead_ticks = 0

    @staticmethod
    def _coast(reason: str) -> GateCommand:
        return GateCommand(0, 0, False, 0, False, True, reason)

    def apply(self, requested_duty_ppm: int, *, emergency_stop: bool = False) -> GateCommand:
        if isinstance(requested_duty_ppm, bool) or not isinstance(requested_duty_ppm, int):
            raise TypeError("requested_duty_ppm must be an integer")
        if emergency_stop:
            self._last_duty = 0
            self._dead_ticks = 0
            return self._coast("emergency-stop")

        limit = self.limits.max_duty_ppm
        requested = max(-limit, min(limit, requested_duty_ppm))
        previous = self._last_duty
        reversing = previous != 0 and requested != 0 and (previous > 0) != (requested > 0)
        if reversing and self.limits.reversal_dead_ticks:
            self._last_duty = 0
            self._dead_ticks = self.limits.reversal_dead_ticks - 1
            return self._coast("reversal-dead-time")
        if self._dead_ticks:
            self._dead_ticks -= 1
            self._last_duty = 0
            return self._coast("reversal-dead-time")

        slew = self.limits.max_slew_ppm
        duty = max(previous - slew, min(previous + slew, requested))
        self._last_duty = duty
        magnitude = abs(duty)
        if duty > 0:
            command = GateCommand(duty, magnitude, False, 0, True, False, "forward")
        elif duty < 0:
            command = GateCommand(duty, 0, True, magnitude, False, False, "reverse")
        else:
            command = self._coast("zero-command")
        if not command.shoot_through_safe:
            raise AssertionError("unsafe H-bridge command generated")
        return command


@dataclass(frozen=True)
class ControlTrace:
    tick: int
    pulse_bits: tuple[int, ...]
    popcount: int
    signed_dot: int
    prediction_bit: int
    error_bit: int
    memory_u16: int
    requested_duty_ppm: int
    command: GateCommand


class DMVomegaxiMechatronicLoop:
    """Closed computational path from normalized sensors to a safe gate command."""

    def __init__(
        self,
        weights: Sequence[int],
        *,
        limits: SafetyLimits | None = None,
        memory_rotation: int = 1,
    ) -> None:
        if not weights:
            raise ValueError("weights must not be empty")
        self.weights = tuple(_require_bit(bit) for bit in weights)
        self.phi = DeltaSigmaBank(len(self.weights))
        self.omega = RecurrentRegister(rotation=memory_rotation)
        self.theta = ConstraintGovernor(limits)
        self.tick = 0

    def step(
        self,
        sensor_values: Sequence[float],
        *,
        target_bit: int,
        emergency_stop: bool = False,
    ) -> ControlTrace:
        target_bit = _require_bit(target_bit)
        pulses = self.phi.encode(sensor_values)
        matches = xnor_popcount(pulses, self.weights)
        signed = 2 * matches - len(self.weights)
        prediction = 1 if signed >= 0 else 0
        error = target_bit ^ prediction
        memory = self.omega.update(pulses, error)

        # Lambda: normalize exact Boolean dot product to signed parts-per-million.
        requested = int(round((signed / len(self.weights)) * 1_000_000))
        command = self.theta.apply(requested, emergency_stop=emergency_stop)
        self.tick += 1
        return ControlTrace(
            tick=self.tick,
            pulse_bits=pulses,
            popcount=matches,
            signed_dot=signed,
            prediction_bit=prediction,
            error_bit=error,
            memory_u16=memory,
            requested_duty_ppm=requested,
            command=command,
        )
