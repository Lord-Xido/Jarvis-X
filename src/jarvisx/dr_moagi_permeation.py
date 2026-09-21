"""Deterministic software transport model for Dr Moagi permeation.

This module models encode -> modulate -> scalar free-space channel -> demodulate
-> verify -> reconstruct. Carrier frequency and propagation are simulation
parameters only; the implementation does not access RF hardware or radiate.
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence, cast

PROTOCOL = "jarvisx.dr-moagi-permeation.v1"


class PermeationIntegrityError(ValueError):
    """Raised when a received frame cannot be reconstructed with integrity."""


@dataclass(frozen=True, slots=True)
class PermeationConfig:
    """Bounded software-channel parameters for one permeation transaction."""

    carrier_hz: float = 333_330_000.0
    propagation_speed_m_s: float = 299_792_458.0
    source_strength: float = 0.941
    range_m: float = 1.0
    coherence: float = 0.967
    omni_weight: float = 0.6
    quadrupole_weight: float = 0.4
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    receiver_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    noise_std: float = 0.0
    noise_seed: int = 0
    max_payload_bytes: int = 65_536

    def __post_init__(self) -> None:
        for name in (
            "carrier_hz",
            "propagation_speed_m_s",
            "source_strength",
            "range_m",
            "coherence",
            "omni_weight",
            "quadrupole_weight",
            "noise_std",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.carrier_hz <= 0 or self.propagation_speed_m_s <= 0 or self.range_m <= 0:
            raise ValueError("carrier_hz, propagation_speed_m_s and range_m must be positive")
        if self.source_strength <= 0:
            raise ValueError("source_strength must be positive")
        if not 0.0 < self.coherence <= 1.0:
            raise ValueError("coherence must be in (0, 1]")
        if self.omni_weight < 0 or self.quadrupole_weight < 0:
            raise ValueError("angular weights must be non-negative")
        if self.omni_weight + self.quadrupole_weight <= 0:
            raise ValueError("at least one angular weight must be positive")
        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative")
        if isinstance(self.noise_seed, bool) or not isinstance(self.noise_seed, int):
            raise TypeError("noise_seed must be an integer")
        if isinstance(self.max_payload_bytes, bool) or not isinstance(self.max_payload_bytes, int):
            raise TypeError("max_payload_bytes must be an integer")
        if self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        _unit_vector(self.axis, "axis")
        _unit_vector(self.receiver_direction, "receiver_direction")
        if abs(self.angular_gain) < 1e-12:
            raise ValueError("configured angular pattern has a channel null at the receiver")

    @property
    def wavelength_m(self) -> float:
        return self.propagation_speed_m_s / self.carrier_hz

    @property
    def wave_number_rad_m(self) -> float:
        return 2.0 * math.pi / self.wavelength_m

    @property
    def propagation_delay_ns(self) -> float:
        return self.range_m / self.propagation_speed_m_s * 1e9

    @property
    def angular_gain(self) -> float:
        """Return an axis-rotated l=0 plus l=2 Legendre-pattern gain.

        This is a scalar quadrupole abstraction, not an antenna solver. A pure
        l=2 pattern is symmetric about +/-axis; one-sided beamforming requires
        a different phased-array model.
        """

        axis = _unit_vector(self.axis, "axis")
        receiver = _unit_vector(self.receiver_direction, "receiver_direction")
        cosine = sum(a * b for a, b in zip(axis, receiver))
        p2 = 0.5 * (3.0 * cosine * cosine - 1.0)
        return self.omni_weight + self.quadrupole_weight * p2

    @property
    def channel_coefficient(self) -> complex:
        amplitude = self.source_strength * self.angular_gain / (4.0 * math.pi * self.range_m)
        phase = self.wave_number_rad_m * self.range_m
        return amplitude * cmath.exp(1j * phase)

    def focused(self, axis: Sequence[float]) -> "PermeationConfig":
        """Implement FOCUS Phi by rotating the quadrupole axis in the simulation."""

        return replace(self, axis=_unit_vector(axis, "axis"))


@dataclass(frozen=True, slots=True)
class ModulatedFrame:
    payload_digest: str
    payload_length: int
    symbols: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ReceivedFrame:
    payload_digest: str
    payload_length: int
    samples: tuple[complex, ...]
    channel_coefficient: complex


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def modulate(payload: Mapping[str, Any], config: PermeationConfig) -> ModulatedFrame:
    """Implement MODULATE Phi using deterministic BPSK over canonical JSON."""

    encoded = canonical_json_bytes(payload)
    if len(encoded) > config.max_payload_bytes:
        raise ValueError("payload exceeds max_payload_bytes")
    digest = hashlib.sha256(encoded).hexdigest()
    symbols: list[float] = []
    for byte in encoded:
        for shift in range(7, -1, -1):
            bit = (byte >> shift) & 1
            symbols.append(1.0 if bit else -1.0)
    return ModulatedFrame(digest, len(encoded), tuple(symbols))


def propagate(frame: ModulatedFrame, config: PermeationConfig) -> ReceivedFrame:
    """Apply scalar 1/r propagation, carrier phase and deterministic optional noise."""

    coefficient = config.channel_coefficient
    rng = random.Random(config.noise_seed)
    samples: list[complex] = []
    for symbol in frame.symbols:
        noise = 0j
        if config.noise_std:
            noise = complex(
                rng.gauss(0.0, config.noise_std),
                rng.gauss(0.0, config.noise_std),
            )
        samples.append(coefficient * symbol + noise)
    return ReceivedFrame(frame.payload_digest, frame.payload_length, tuple(samples), coefficient)


def absorb(frame: ReceivedFrame) -> dict[str, Any]:
    """Implement ABSORB Phi by equalizing, demodulating and verifying the frame."""

    coefficient = frame.channel_coefficient
    if abs(coefficient) < 1e-15:
        raise PermeationIntegrityError("channel coefficient is too small to equalize")
    bits: list[int] = []
    for sample in frame.samples:
        equalized = sample / coefficient
        bits.append(1 if equalized.real >= 0.0 else 0)
    if len(bits) % 8:
        raise PermeationIntegrityError("symbol count is not byte aligned")

    decoded = bytearray()
    for offset in range(0, len(bits), 8):
        value = 0
        for bit in bits[offset : offset + 8]:
            value = (value << 1) | bit
        decoded.append(value)
    if len(decoded) != frame.payload_length:
        raise PermeationIntegrityError("payload length mismatch")
    digest = hashlib.sha256(bytes(decoded)).hexdigest()
    if digest != frame.payload_digest:
        raise PermeationIntegrityError("payload digest mismatch")
    try:
        value = json.loads(bytes(decoded).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermeationIntegrityError("reconstructed payload is not valid JSON") from error
    if not isinstance(value, dict):
        raise PermeationIntegrityError("reconstructed payload must be a JSON object")
    return cast(dict[str, Any], value)


def simulate_round_trip(payload: Mapping[str, Any], config: PermeationConfig) -> dict[str, Any]:
    """Run the full software permeation path and emit auditable telemetry."""

    frame = modulate(payload, config)
    received = propagate(frame, config)
    reconstructed = absorb(received)
    return {
        "protocol": PROTOCOL,
        "physical_rf": False,
        "model": "deterministic-bpsk-scalar-free-space-simulation",
        "carrier_hz": config.carrier_hz,
        "wavelength_m": config.wavelength_m,
        "wave_number_rad_m": config.wave_number_rad_m,
        "range_m": config.range_m,
        "propagation_delay_ns": config.propagation_delay_ns,
        "source_strength": config.source_strength,
        "coherence": config.coherence,
        "angular_gain": config.angular_gain,
        "amplitude_at_receiver": abs(config.channel_coefficient),
        "payload_bytes": frame.payload_length,
        "symbol_count": len(frame.symbols),
        "payload_digest": frame.payload_digest,
        "verified": reconstructed == dict(payload),
        "reconstructed": reconstructed,
    }


def _unit_vector(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three components")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} components must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} components must be finite")
        converted.append(numeric)
    norm = math.sqrt(sum(value * value for value in converted))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return (
        converted[0] / norm,
        converted[1] / norm,
        converted[2] / norm,
    )
