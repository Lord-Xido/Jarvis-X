"""Literal electronic actuation mappings for Jarvis-X research runtimes.

This module bridges bounded algorithmic state to explicit electrical quantities:

    scalar -> voltage -> logic/bit state -> switching estimate -> PWM command

It is intentionally a *model and encoding layer*. It performs no GPIO writes,
network control, device-driver calls, or physical hardware actuation.

Semantic/latent variables remain algorithmic variables. Voltages, charges,
currents, switching energies and duty cycles are physical implementation
quantities used to encode or actuate those variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def _finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value: float, *, name: str) -> float:
    value = _finite(value, name=name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class VoltageCodec:
    """Affine map between a bounded algorithmic scalar and node voltage."""

    scalar_min: float = -1.0
    scalar_max: float = 1.0
    voltage_min: float = 0.0
    voltage_max: float = 1.0

    def __post_init__(self) -> None:
        s0 = _finite(self.scalar_min, name="scalar_min")
        s1 = _finite(self.scalar_max, name="scalar_max")
        v0 = _finite(self.voltage_min, name="voltage_min")
        v1 = _finite(self.voltage_max, name="voltage_max")
        if not s0 < s1:
            raise ValueError("scalar_min must be less than scalar_max")
        if not v0 < v1:
            raise ValueError("voltage_min must be less than voltage_max")

    def encode(self, value: float) -> float:
        value = _finite(value, name="value")
        if not self.scalar_min <= value <= self.scalar_max:
            raise ValueError("value lies outside scalar bounds")
        u = (value - self.scalar_min) / (self.scalar_max - self.scalar_min)
        return self.voltage_min + u * (self.voltage_max - self.voltage_min)

    def decode(self, voltage: float) -> float:
        voltage = _finite(voltage, name="voltage")
        if not self.voltage_min <= voltage <= self.voltage_max:
            raise ValueError("voltage lies outside voltage bounds")
        u = (voltage - self.voltage_min) / (self.voltage_max - self.voltage_min)
        return self.scalar_min + u * (self.scalar_max - self.scalar_min)


@dataclass(frozen=True)
class CMOSLogicModel:
    """Minimal bounded CMOS logic-level and switching-energy model."""

    v_low: float = 0.0
    v_high: float = 1.0
    threshold: float = 0.5
    effective_capacitance_f: float = 1.0e-15
    activity_factor: float = 0.10
    clock_hz: float = 1.0e9

    def __post_init__(self) -> None:
        v_low = _finite(self.v_low, name="v_low")
        v_high = _finite(self.v_high, name="v_high")
        threshold = _finite(self.threshold, name="threshold")
        if not v_low < threshold < v_high:
            raise ValueError("threshold must lie strictly between v_low and v_high")
        _positive(self.effective_capacitance_f, name="effective_capacitance_f")
        activity = _finite(self.activity_factor, name="activity_factor")
        if not 0.0 <= activity <= 1.0:
            raise ValueError("activity_factor must lie within [0, 1]")
        _positive(self.clock_hz, name="clock_hz")

    @property
    def delta_v(self) -> float:
        return self.v_high - self.v_low

    def bit_voltage(self, bit: int) -> float:
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        return self.v_high if bit else self.v_low

    def decode_bit(self, voltage: float) -> int:
        voltage = _finite(voltage, name="voltage")
        return int(voltage >= self.threshold)

    def switching_energy_j(self) -> float:
        """Representative 0->1 capacitive energy, 1/2 C (Delta V)^2."""

        return 0.5 * self.effective_capacitance_f * self.delta_v**2

    def dynamic_power_w(self) -> float:
        """Conventional alpha C V^2 f activity estimate."""

        return (
            self.activity_factor
            * self.effective_capacitance_f
            * self.delta_v**2
            * self.clock_hz
        )


@dataclass(frozen=True)
class PWMCommand:
    """Normalized electronic actuator command before any hardware driver."""

    duty_cycle: float
    direction: int
    enable: bool

    def __post_init__(self) -> None:
        duty = _finite(self.duty_cycle, name="duty_cycle")
        if not 0.0 <= duty <= 1.0:
            raise ValueError("duty_cycle must lie within [0, 1]")
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        if self.direction == 0 and duty != 0.0:
            raise ValueError("zero direction requires zero duty cycle")


@dataclass(frozen=True)
class PWMMapper:
    """Map a bounded signed control scalar to PWM magnitude and direction."""

    deadband: float = 0.02
    max_duty: float = 1.0

    def __post_init__(self) -> None:
        deadband = _finite(self.deadband, name="deadband")
        max_duty = _finite(self.max_duty, name="max_duty")
        if not 0.0 <= deadband < 1.0:
            raise ValueError("deadband must lie within [0, 1)")
        if not 0.0 < max_duty <= 1.0:
            raise ValueError("max_duty must lie within (0, 1]")

    def map(self, control: float) -> PWMCommand:
        control = _finite(control, name="control")
        if not -1.0 <= control <= 1.0:
            raise ValueError("control must lie within [-1, 1]")
        magnitude = abs(control)
        if magnitude <= self.deadband:
            return PWMCommand(duty_cycle=0.0, direction=0, enable=False)
        scaled = (magnitude - self.deadband) / (1.0 - self.deadband)
        duty = min(self.max_duty, self.max_duty * scaled)
        return PWMCommand(
            duty_cycle=duty,
            direction=1 if control > 0.0 else -1,
            enable=True,
        )


@dataclass(frozen=True)
class ElectronicActuationTrace:
    """Auditable algorithmic-to-electronic mapping for one control scalar."""

    control_scalar: float
    encoded_voltage_v: float
    quantized_bit: int
    bit_voltage_v: float
    switching_energy_j: float
    dynamic_power_w: float
    pwm: PWMCommand


def trace_control_scalar(
    control: float,
    *,
    voltage_codec: VoltageCodec | None = None,
    logic: CMOSLogicModel | None = None,
    pwm_mapper: PWMMapper | None = None,
) -> ElectronicActuationTrace:
    """Trace one normalized control value through the electronic mapping stack."""

    codec = voltage_codec or VoltageCodec()
    model = logic or CMOSLogicModel()
    mapper = pwm_mapper or PWMMapper()
    voltage = codec.encode(control)
    bit = model.decode_bit(voltage)
    return ElectronicActuationTrace(
        control_scalar=float(control),
        encoded_voltage_v=voltage,
        quantized_bit=bit,
        bit_voltage_v=model.bit_voltage(bit),
        switching_energy_j=model.switching_energy_j(),
        dynamic_power_w=model.dynamic_power_w(),
        pwm=mapper.map(control),
    )


def trace_vector(
    controls: Sequence[float],
    *,
    voltage_codec: VoltageCodec | None = None,
    logic: CMOSLogicModel | None = None,
    pwm_mapper: PWMMapper | None = None,
) -> tuple[ElectronicActuationTrace, ...]:
    """Trace a vector of bounded controls independently and deterministically."""

    return tuple(
        trace_control_scalar(
            value,
            voltage_codec=voltage_codec,
            logic=logic,
            pwm_mapper=pwm_mapper,
        )
        for value in controls
    )


__all__ = [
    "CMOSLogicModel",
    "ElectronicActuationTrace",
    "PWMCommand",
    "PWMMapper",
    "VoltageCodec",
    "trace_control_scalar",
    "trace_vector",
]
