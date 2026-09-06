"""Bounded bit-serial reference for the DM-vOmegaXi+ control pipeline.

The module maps normalized sensor samples to a first-order delta-sigma stream,
executes binary XNOR/popcount inference, folds the result through 16-bit Omega
register memory, applies a Theta mask, evaluates independent hardware interlocks,
and emits logic-level pulse-density frames.

This is a deterministic software reference. It does not access GPIO, a gate
driver, an H-bridge, or any other actuator, and it is not a substitute for
certified hardware protection or plant-specific control validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

WORD_BITS = 16
WORD_MASK = (1 << WORD_BITS) - 1


class MixedSignalError(ValueError):
    """Raised when a mixed-signal contract or input is malformed."""


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise MixedSignalError(f"{name} must be a {qualifier} integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MixedSignalError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MixedSignalError(f"{name} must be finite")
    return result


def _bits(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
        for value in result
    ):
        raise MixedSignalError(f"{name} must contain only integer bits 0 or 1")
    return result


@dataclass(frozen=True)
class HardwareInterlockLimits:
    """Plant-specific limits that must be supplied by the integration boundary."""

    max_abs_current_a: float
    max_abs_voltage_v: float
    max_temperature_c: float
    watchdog_timeout_ticks: int
    min_dead_time_ticks: int

    def __post_init__(self) -> None:
        current = _finite(self.max_abs_current_a, "max_abs_current_a")
        voltage = _finite(self.max_abs_voltage_v, "max_abs_voltage_v")
        temperature = _finite(self.max_temperature_c, "max_temperature_c")
        if current <= 0.0 or voltage <= 0.0:
            raise MixedSignalError("current and voltage limits must be positive")
        object.__setattr__(self, "max_abs_current_a", current)
        object.__setattr__(self, "max_abs_voltage_v", voltage)
        object.__setattr__(self, "max_temperature_c", temperature)
        _positive_int(self.watchdog_timeout_ticks, "watchdog_timeout_ticks")
        _positive_int(self.min_dead_time_ticks, "min_dead_time_ticks", allow_zero=True)


@dataclass(frozen=True)
class HardwareTelemetry:
    """Measured driver/plant telemetry consumed by the independent interlock."""

    current_a: float
    voltage_v: float
    temperature_c: float
    watchdog_age_ticks: int
    observed_dead_time_ticks: int
    emergency_stop: bool = False
    bridge_overlap_detected: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "current_a", _finite(self.current_a, "current_a"))
        object.__setattr__(self, "voltage_v", _finite(self.voltage_v, "voltage_v"))
        object.__setattr__(self, "temperature_c", _finite(self.temperature_c, "temperature_c"))
        _positive_int(self.watchdog_age_ticks, "watchdog_age_ticks", allow_zero=True)
        _positive_int(
            self.observed_dead_time_ticks,
            "observed_dead_time_ticks",
            allow_zero=True,
        )
        if not isinstance(self.emergency_stop, bool):
            raise MixedSignalError("emergency_stop must be Boolean")
        if not isinstance(self.bridge_overlap_detected, bool):
            raise MixedSignalError("bridge_overlap_detected must be Boolean")


@dataclass(frozen=True)
class MixedSignalConfig:
    """Finite execution and representation bounds for the digital reference."""

    sensor_channels: int = 1
    oversample: int = 16
    sensor_min: float = -1.0
    sensor_max: float = 1.0
    score_bound: int = 16
    memory_words: int = 4
    omega_rotate_bits: int = 1
    omega_persistence_mask: int = 0
    pdm_period: int = 32
    max_duty_cycle: float = 0.9
    max_outputs: int = 64
    fixed_point_tolerance: float = 0.0

    def __post_init__(self) -> None:
        _positive_int(self.sensor_channels, "sensor_channels")
        _positive_int(self.oversample, "oversample")
        low = _finite(self.sensor_min, "sensor_min")
        high = _finite(self.sensor_max, "sensor_max")
        if low >= high:
            raise MixedSignalError("sensor_min must be smaller than sensor_max")
        object.__setattr__(self, "sensor_min", low)
        object.__setattr__(self, "sensor_max", high)
        _positive_int(self.score_bound, "score_bound")
        _positive_int(self.memory_words, "memory_words")
        rotate = _positive_int(self.omega_rotate_bits, "omega_rotate_bits", allow_zero=True)
        if rotate >= WORD_BITS:
            raise MixedSignalError("omega_rotate_bits must be in [0, 15]")
        mask = _positive_int(
            self.omega_persistence_mask,
            "omega_persistence_mask",
            allow_zero=True,
        )
        if mask > WORD_MASK:
            raise MixedSignalError("omega_persistence_mask must fit in 16 bits")
        _positive_int(self.pdm_period, "pdm_period")
        duty = _finite(self.max_duty_cycle, "max_duty_cycle")
        if not 0.0 <= duty <= 1.0:
            raise MixedSignalError("max_duty_cycle must be in [0, 1]")
        object.__setattr__(self, "max_duty_cycle", duty)
        _positive_int(self.max_outputs, "max_outputs")
        tolerance = _finite(self.fixed_point_tolerance, "fixed_point_tolerance")
        if tolerance < 0.0:
            raise MixedSignalError("fixed_point_tolerance must be non-negative")
        object.__setattr__(self, "fixed_point_tolerance", tolerance)

    @property
    def input_width(self) -> int:
        return self.sensor_channels * self.oversample


@dataclass(frozen=True)
class HBridgeGateFrame:
    """Logic-level mutually exclusive gate intent for one output channel."""

    direction: int
    duty_cycle: float
    positive_gate: tuple[int, ...]
    negative_gate: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise MixedSignalError("direction must be -1, 0, or 1")
        duty = _finite(self.duty_cycle, "duty_cycle")
        if not 0.0 <= duty <= 1.0:
            raise MixedSignalError("duty_cycle must be in [0, 1]")
        positive = _bits(self.positive_gate, "positive_gate")
        negative = _bits(self.negative_gate, "negative_gate")
        if len(positive) != len(negative):
            raise MixedSignalError("gate vectors must have equal length")
        if any(a & b for a, b in zip(positive, negative)):
            raise MixedSignalError("positive and negative gates cannot overlap")
        if self.direction == 0 and duty != 0.0:
            raise MixedSignalError("an idle frame must have zero duty")
        if self.direction == 0 and (any(positive) or any(negative)):
            raise MixedSignalError("an idle frame cannot contain active gates")
        if self.direction > 0 and any(negative):
            raise MixedSignalError("positive direction cannot drive the negative gate")
        if self.direction < 0 and any(positive):
            raise MixedSignalError("negative direction cannot drive the positive gate")
        object.__setattr__(self, "duty_cycle", duty)
        object.__setattr__(self, "positive_gate", positive)
        object.__setattr__(self, "negative_gate", negative)


@dataclass(frozen=True)
class MixedSignalStepReport:
    iteration: int
    bitstream: tuple[int, ...]
    xnor_matches: tuple[int, ...]
    raw_scores: tuple[int, ...]
    bounded_scores: tuple[int, ...]
    latent_bits: tuple[int, ...]
    omega_words: tuple[int, ...]
    theta_candidate_bits: tuple[int, ...]
    theta_output_bits: tuple[int, ...]
    emitted_bits: tuple[int, ...]
    target_bits: tuple[int, ...] | None
    hamming_error_bits: int | None
    hamming_error_rate: float | None
    interlock_trips: tuple[str, ...]
    actuation_permitted: bool
    emission_active: bool
    duty_cycles: tuple[float, ...]
    gate_frames: tuple[HBridgeGateFrame, ...]
    state_gap_bits: int
    state_gap_numeric: float
    internal_fixed_point: bool
    state_sha256: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def xnor_popcount(left: Sequence[int], right: Sequence[int]) -> tuple[int, int]:
    """Return matching-bit count and its bipolar signed-dot equivalent."""

    lhs = _bits(left, "left")
    rhs = _bits(right, "right")
    if not lhs or len(lhs) != len(rhs):
        raise MixedSignalError("XNOR operands must be non-empty and equally sized")
    matches = sum(a == b for a, b in zip(lhs, rhs))
    return matches, 2 * matches - len(lhs)


def rotate_left_16(value: int, amount: int) -> int:
    """Rotate one unsigned 16-bit word."""

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= WORD_MASK:
        raise MixedSignalError("value must be an unsigned 16-bit integer")
    if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
        raise MixedSignalError("rotation amount must be a non-negative integer")
    shift = amount % WORD_BITS
    if shift == 0:
        return value
    return ((value << shift) | (value >> (WORD_BITS - shift))) & WORD_MASK


def pack_bits_16(bits: Sequence[int], *, word_count: int) -> tuple[int, ...]:
    """Pack least-significant-bit first into a fixed number of 16-bit words."""

    normalized = _bits(bits, "bits")
    count = _positive_int(word_count, "word_count")
    if len(normalized) > count * WORD_BITS:
        raise MixedSignalError("bit vector exceeds the fixed word capacity")
    words = [0] * count
    for index, bit in enumerate(normalized):
        words[index // WORD_BITS] |= bit << (index % WORD_BITS)
    return tuple(words)


def unpack_bits_16(words: Sequence[int], *, bit_count: int) -> tuple[int, ...]:
    """Unpack a bounded number of least-significant-bit-first values."""

    count = _positive_int(bit_count, "bit_count", allow_zero=True)
    normalized = tuple(words)
    if any(
        isinstance(word, bool) or not isinstance(word, int) or not 0 <= word <= WORD_MASK
        for word in normalized
    ):
        raise MixedSignalError("words must contain unsigned 16-bit integers")
    if count > len(normalized) * WORD_BITS:
        raise MixedSignalError("bit_count exceeds the available word capacity")
    return tuple(
        (normalized[index // WORD_BITS] >> (index % WORD_BITS)) & 1 for index in range(count)
    )


class DeltaSigmaBank:
    """Stateful first-order one-bit delta-sigma acquisition bank."""

    def __init__(self, config: MixedSignalConfig) -> None:
        self.config = config
        self._accumulators = [0.0] * config.sensor_channels

    @property
    def state(self) -> tuple[float, ...]:
        return tuple(self._accumulators)

    def encode(self, samples: Sequence[float]) -> tuple[int, ...]:
        values = tuple(_finite(value, "sensor sample") for value in samples)
        if len(values) != self.config.sensor_channels:
            raise MixedSignalError("sensor sample count does not match sensor_channels")
        scale = self.config.sensor_max - self.config.sensor_min
        normalized: list[float] = []
        for value in values:
            if value < self.config.sensor_min or value > self.config.sensor_max:
                raise MixedSignalError("sensor sample is outside the declared range")
            normalized.append(2.0 * (value - self.config.sensor_min) / scale - 1.0)

        stream: list[int] = []
        for _ in range(self.config.oversample):
            for channel, sample in enumerate(normalized):
                accumulator = self._accumulators[channel] + sample
                bit = 1 if accumulator >= 0.0 else 0
                feedback = 1.0 if bit else -1.0
                self._accumulators[channel] = accumulator - feedback
                stream.append(bit)
        return tuple(stream)

    def reset(self) -> None:
        self._accumulators = [0.0] * self.config.sensor_channels


class OmegaRegisterBank:
    """Finite recurrent memory using XOR, rotate, and persistence masking."""

    def __init__(self, config: MixedSignalConfig) -> None:
        self.config = config
        self._words = [0] * config.memory_words

    @property
    def state(self) -> tuple[int, ...]:
        return tuple(self._words)

    def update(self, incoming: Sequence[int]) -> tuple[int, ...]:
        values = tuple(incoming)
        if len(values) != self.config.memory_words:
            raise MixedSignalError("incoming Omega word count does not match memory_words")
        if any(
            isinstance(word, bool) or not isinstance(word, int) or not 0 <= word <= WORD_MASK
            for word in values
        ):
            raise MixedSignalError("incoming Omega words must be unsigned 16-bit integers")
        retain = self.config.omega_persistence_mask
        replace_mask = WORD_MASK ^ retain
        next_words: list[int] = []
        for previous, injected in zip(self._words, values):
            rotated = rotate_left_16(previous ^ injected, self.config.omega_rotate_bits)
            next_words.append(((previous & retain) | (rotated & replace_mask)) & WORD_MASK)
        self._words = next_words
        return self.state

    def reset(self) -> None:
        self._words = [0] * self.config.memory_words


class HardwareInterlock:
    """Independent fail-closed check that cannot be overridden by Theta output."""

    def __init__(self, limits: HardwareInterlockLimits) -> None:
        if not isinstance(limits, HardwareInterlockLimits):
            raise MixedSignalError("limits must be a HardwareInterlockLimits value")
        self.limits = limits

    def evaluate(self, telemetry: HardwareTelemetry) -> tuple[str, ...]:
        trips: list[str] = []
        if telemetry.emergency_stop:
            trips.append("emergency-stop")
        if abs(telemetry.current_a) > self.limits.max_abs_current_a:
            trips.append("overcurrent")
        if abs(telemetry.voltage_v) > self.limits.max_abs_voltage_v:
            trips.append("overvoltage")
        if telemetry.temperature_c > self.limits.max_temperature_c:
            trips.append("overtemperature")
        if telemetry.watchdog_age_ticks > self.limits.watchdog_timeout_ticks:
            trips.append("watchdog-timeout")
        if telemetry.bridge_overlap_detected:
            trips.append("bridge-overlap")
        if telemetry.observed_dead_time_ticks < self.limits.min_dead_time_ticks:
            trips.append("insufficient-dead-time")
        return tuple(trips)


class PulseDensityBank:
    """Stateful deterministic pulse-density encoder for bounded duty commands."""

    def __init__(self, channels: int, period: int) -> None:
        self.channels = _positive_int(channels, "channels")
        self.period = _positive_int(period, "period")
        self._accumulators = [0.0] * self.channels

    @property
    def state(self) -> tuple[float, ...]:
        return tuple(self._accumulators)

    def encode(self, duties: Sequence[float]) -> tuple[tuple[int, ...], ...]:
        values = tuple(_finite(value, "duty") for value in duties)
        if len(values) != self.channels:
            raise MixedSignalError("duty count does not match pulse-density channels")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise MixedSignalError("duties must be in [0, 1]")

        patterns: list[tuple[int, ...]] = []
        for channel, duty in enumerate(values):
            accumulator = self._accumulators[channel]
            pattern: list[int] = []
            for _ in range(self.period):
                accumulator += duty
                if accumulator >= 1.0:
                    pattern.append(1)
                    accumulator -= 1.0
                else:
                    pattern.append(0)
            self._accumulators[channel] = accumulator
            patterns.append(tuple(pattern))
        return tuple(patterns)

    def reset(self) -> None:
        self._accumulators = [0.0] * self.channels


class DMvOmegaXiMixedSignalEngine:
    """Execute one bounded sensor-to-logic-frame recursion."""

    OPERATOR_STACK = (
        "Psi",
        "Phi",
        "Lambda",
        "Omega",
        "Theta",
        "hardware-interlock",
        "PDM",
    )

    def __init__(
        self,
        weights: Sequence[Sequence[int]],
        interlock_limits: HardwareInterlockLimits,
        *,
        config: MixedSignalConfig | None = None,
        theta_mask: Sequence[int] | None = None,
        actuator_polarities: Sequence[int] | None = None,
    ) -> None:
        self.config = config or MixedSignalConfig()
        rows = tuple(_bits(row, "weight row") for row in weights)
        if not rows:
            raise MixedSignalError("weights must contain at least one output row")
        if any(len(row) != self.config.input_width for row in rows):
            raise MixedSignalError("every weight row must match the oversampled input width")
        if len(rows) > self.config.max_outputs:
            raise MixedSignalError("weight rows exceed max_outputs")
        if len(rows) > self.config.memory_words * WORD_BITS:
            raise MixedSignalError("weight rows exceed Omega register capacity")
        self.weights = rows
        outputs = len(rows)
        active_theta_mask = (1,) * outputs if theta_mask is None else theta_mask
        self.theta_mask = _bits(active_theta_mask, "theta_mask")
        if len(self.theta_mask) != outputs:
            raise MixedSignalError("theta_mask length must match output rows")
        polarities = tuple((1,) * outputs if actuator_polarities is None else actuator_polarities)
        if len(polarities) != outputs or any(
            isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1)
            for value in polarities
        ):
            raise MixedSignalError("actuator_polarities must contain one -1 or 1 per output")
        self.actuator_polarities = polarities
        self.delta_sigma = DeltaSigmaBank(self.config)
        self.omega = OmegaRegisterBank(self.config)
        self.interlock = HardwareInterlock(interlock_limits)
        self.pdm = PulseDensityBank(outputs, self.config.pdm_period)
        self._previous_emitted = (0,) * outputs
        self._iteration = 0

    def step(
        self,
        sensor_samples: Sequence[float],
        telemetry: HardwareTelemetry,
        *,
        target_bits: Sequence[int] | None = None,
    ) -> MixedSignalStepReport:
        if not isinstance(telemetry, HardwareTelemetry):
            raise MixedSignalError("telemetry must be a HardwareTelemetry value")
        normalized_target: tuple[int, ...] | None = None
        if target_bits is not None:
            normalized_target = _bits(target_bits, "target_bits")
            if len(normalized_target) != len(self.weights):
                raise MixedSignalError("target_bits length must match output rows")

        before_delta = self.delta_sigma.state
        before_omega = self.omega.state
        before_pdm = self.pdm.state
        before_emitted = self._previous_emitted

        bitstream = self.delta_sigma.encode(sensor_samples)
        matches_and_scores = tuple(xnor_popcount(bitstream, row) for row in self.weights)
        matches = tuple(item[0] for item in matches_and_scores)
        raw_scores = tuple(item[1] for item in matches_and_scores)
        bound = self.config.score_bound
        bounded_scores = tuple(max(-bound, min(bound, score)) for score in raw_scores)
        latent_bits = tuple(int(score > 0) for score in bounded_scores)

        incoming = pack_bits_16(latent_bits, word_count=self.config.memory_words)
        omega_words = self.omega.update(incoming)
        theta_candidate = unpack_bits_16(omega_words, bit_count=len(self.weights))
        theta_output = tuple(bit & mask for bit, mask in zip(theta_candidate, self.theta_mask))

        hamming_error: int | None = None
        hamming_rate: float | None = None
        if normalized_target is not None:
            hamming_error = sum(a ^ b for a, b in zip(theta_output, normalized_target))
            hamming_rate = hamming_error / len(theta_output)

        trips = self.interlock.evaluate(telemetry)
        emitted = (0,) * len(theta_output) if trips else theta_output
        duties = tuple(
            (
                min(
                    self.config.max_duty_cycle,
                    self.config.max_duty_cycle * max(0.0, score / bound),
                )
                if enabled
                else 0.0
            )
            for enabled, score in zip(emitted, bounded_scores)
        )
        if trips:
            self.pdm.reset()
        patterns = self.pdm.encode(duties)
        frames: list[HBridgeGateFrame] = []
        for enabled, duty, polarity, pattern in zip(
            emitted,
            duties,
            self.actuator_polarities,
            patterns,
        ):
            direction = polarity if enabled and duty > 0.0 else 0
            zeros = (0,) * self.config.pdm_period
            frames.append(
                HBridgeGateFrame(
                    direction=direction,
                    duty_cycle=duty,
                    positive_gate=pattern if direction > 0 else zeros,
                    negative_gate=pattern if direction < 0 else zeros,
                )
            )

        after_delta = self.delta_sigma.state
        after_omega = self.omega.state
        after_pdm = self.pdm.state
        word_gap = sum((left ^ right).bit_count() for left, right in zip(before_omega, after_omega))
        output_gap = sum(left ^ right for left, right in zip(before_emitted, emitted))
        numeric_deltas = [abs(left - right) for left, right in zip(before_delta, after_delta)]
        numeric_deltas.extend(abs(left - right) for left, right in zip(before_pdm, after_pdm))
        numeric_gap = max(numeric_deltas, default=0.0)
        state_gap_bits = word_gap + output_gap
        internal_fixed_point = bool(
            state_gap_bits == 0 and numeric_gap <= self.config.fixed_point_tolerance
        )

        self._previous_emitted = emitted
        self._iteration += 1
        state_hash = self._state_hash(after_delta, after_omega, after_pdm, emitted)
        return MixedSignalStepReport(
            iteration=self._iteration,
            bitstream=bitstream,
            xnor_matches=matches,
            raw_scores=raw_scores,
            bounded_scores=bounded_scores,
            latent_bits=latent_bits,
            omega_words=omega_words,
            theta_candidate_bits=theta_candidate,
            theta_output_bits=theta_output,
            emitted_bits=emitted,
            target_bits=normalized_target,
            hamming_error_bits=hamming_error,
            hamming_error_rate=hamming_rate,
            interlock_trips=trips,
            actuation_permitted=not trips,
            emission_active=any(frame.direction for frame in frames),
            duty_cycles=duties,
            gate_frames=tuple(frames),
            state_gap_bits=state_gap_bits,
            state_gap_numeric=numeric_gap,
            internal_fixed_point=internal_fixed_point,
            state_sha256=state_hash,
        )

    def reset(self) -> None:
        self.delta_sigma.reset()
        self.omega.reset()
        self.pdm.reset()
        self._previous_emitted = (0,) * len(self.weights)
        self._iteration = 0

    def status(self) -> dict[str, object]:
        return {
            "law": "DM-vOmegaXi+",
            "mode": "bit-serial-mixed-signal-reference",
            "operator_stack": list(self.OPERATOR_STACK),
            "input_width": self.config.input_width,
            "outputs": len(self.weights),
            "iteration": self._iteration,
            "omega_words": list(self.omega.state),
            "hardware_io": False,
            "hardware_limits_required": True,
        }

    @staticmethod
    def _state_hash(
        delta_sigma: Sequence[float],
        omega: Sequence[int],
        pdm: Sequence[float],
        emitted: Sequence[int],
    ) -> str:
        state = {
            "delta_sigma": [float(value).hex() for value in delta_sigma],
            "emitted": list(emitted),
            "omega": list(omega),
            "pdm": [float(value).hex() for value in pdm],
        }
        payload = json.dumps(
            state,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DMvOmegaXiMixedSignalEngine",
    "DeltaSigmaBank",
    "HBridgeGateFrame",
    "HardwareInterlock",
    "HardwareInterlockLimits",
    "HardwareTelemetry",
    "MixedSignalConfig",
    "MixedSignalError",
    "MixedSignalStepReport",
    "OmegaRegisterBank",
    "PulseDensityBank",
    "pack_bits_16",
    "rotate_left_16",
    "unpack_bits_16",
    "xnor_popcount",
]
