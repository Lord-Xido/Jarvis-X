"""Deterministic, proof-ready multi-stream forensic arithmetic for Jarvis-X.

Hyperion aligns semantically compatible event observations, fuses them with
confidence-aware fixed-point arithmetic, computes time-correct derivatives,
applies robust anomaly filters, and emits tamper-evident commitments.

A commitment proves deterministic computation over committed inputs; it does
not prove that a sensor or source system reported reality truthfully. External
attestation and chain-of-custody controls remain necessary.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping, Sequence

FILTER_NAMES = (
    "continuity",
    "acceleration_spike",
    "precision_strike",
    "ghost_entity",
    "bytecode_divergence",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _robust_z(value: float, history: Sequence[float], epsilon: float) -> float:
    if len(history) < 5:
        return 0.0
    center = statistics.median(history)
    mad = statistics.median(abs(item - center) for item in history)
    if mad <= epsilon:
        if abs(value - center) <= epsilon:
            return 0.0
        return math.copysign(math.inf, value - center)
    return 0.6744897501960817 * (value - center) / mad


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _round_div_nearest(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return sign * quotient


def merkle_root(leaves: Sequence[str]) -> str:
    """Return a deterministic SHA-256 Merkle root for hexadecimal leaves."""

    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes.fromhex(leaf) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


@dataclass(frozen=True, slots=True)
class Observation:
    """One source observation with an explicit semantic dimension."""

    source: str
    timestamp_ms: int
    value: float
    quantity: str
    unit: str
    correlation_id: str | None = None
    confidence: float = 1.0
    available: bool = True
    label: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be non-empty")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not math.isfinite(self.value):
            raise ValueError("observation value must be finite")
        if not self.quantity or not self.unit:
            raise ValueError("quantity and unit must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FusionTerm:
    source: str
    value_int: int
    effective_weight_int: int


@dataclass(frozen=True, slots=True)
class ArithmeticWitness:
    """Fixed-point weighted-average witness for a proof backend."""

    event_id: str
    timestamp_ms: int
    quantity: str
    unit: str
    value_scale: int
    weight_scale: int
    terms: tuple[FusionTerm, ...]
    numerator: int
    denominator: int
    fused_int: int
    remainder: int
    commitment: str

    @classmethod
    def build(
        cls,
        *,
        event_id: str,
        timestamp_ms: int,
        quantity: str,
        unit: str,
        observations: Sequence[Observation],
        source_weights: Mapping[str, float],
        value_scale: int,
        weight_scale: int,
    ) -> "ArithmeticWitness":
        terms: list[FusionTerm] = []
        for observation in sorted(
            observations,
            key=lambda item: (item.source, item.timestamp_ms, item.value),
        ):
            if not observation.available or observation.confidence <= 0.0:
                continue
            source_weight = float(source_weights.get(observation.source, 0.0))
            if not math.isfinite(source_weight) or source_weight < 0.0:
                raise ValueError(f"invalid source weight for {observation.source!r}")
            effective_weight_int = round(
                source_weight * observation.confidence * weight_scale
            )
            if effective_weight_int <= 0:
                continue
            terms.append(
                FusionTerm(
                    source=observation.source,
                    value_int=round(observation.value * value_scale),
                    effective_weight_int=effective_weight_int,
                )
            )
        if not terms:
            raise ValueError("event has no positive-weight compatible observations")

        numerator = sum(term.value_int * term.effective_weight_int for term in terms)
        denominator = sum(term.effective_weight_int for term in terms)
        fused_int = _round_div_nearest(numerator, denominator)
        remainder = numerator - denominator * fused_int
        payload = {
            "event_id": event_id,
            "timestamp_ms": timestamp_ms,
            "quantity": quantity,
            "unit": unit,
            "value_scale": value_scale,
            "weight_scale": weight_scale,
            "terms": [asdict(term) for term in terms],
            "numerator": numerator,
            "denominator": denominator,
            "fused_int": fused_int,
            "remainder": remainder,
        }
        return cls(
            event_id=event_id,
            timestamp_ms=timestamp_ms,
            quantity=quantity,
            unit=unit,
            value_scale=value_scale,
            weight_scale=weight_scale,
            terms=tuple(terms),
            numerator=numerator,
            denominator=denominator,
            fused_int=fused_int,
            remainder=remainder,
            commitment=sha256_hex(payload),
        )

    @property
    def fused_value(self) -> float:
        return self.fused_int / self.value_scale

    def circuit_inputs(self) -> dict[str, object]:
        """Export the integer relation consumed by a SNARK/STARK circuit."""

        return {
            "event_id": self.event_id,
            "timestamp_ms": self.timestamp_ms,
            "value_scale": self.value_scale,
            "weight_scale": self.weight_scale,
            "value_ints": [term.value_int for term in self.terms],
            "weight_ints": [term.effective_weight_int for term in self.terms],
            "numerator": self.numerator,
            "denominator": self.denominator,
            "fused_int": self.fused_int,
            "remainder": self.remainder,
        }

    def verify(self) -> bool:
        if self.denominator <= 0 or not self.terms:
            return False
        if sum(term.effective_weight_int for term in self.terms) != self.denominator:
            return False
        if (
            sum(term.value_int * term.effective_weight_int for term in self.terms)
            != self.numerator
        ):
            return False
        if self.numerator != self.denominator * self.fused_int + self.remainder:
            return False
        if 2 * abs(self.remainder) > self.denominator:
            return False
        payload = {
            "event_id": self.event_id,
            "timestamp_ms": self.timestamp_ms,
            "quantity": self.quantity,
            "unit": self.unit,
            "value_scale": self.value_scale,
            "weight_scale": self.weight_scale,
            "terms": [asdict(term) for term in self.terms],
            "numerator": self.numerator,
            "denominator": self.denominator,
            "fused_int": self.fused_int,
            "remainder": self.remainder,
        }
        return self.commitment == sha256_hex(payload)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    severities: Mapping[str, float]
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")


@dataclass(frozen=True, slots=True)
class ScoreModel:
    """Versioned, bounded logistic anomaly aggregator."""

    bias: float = -3.0
    weights: tuple[float, ...] = (1.35, 1.25, 1.30, 1.10, 1.50)
    version: int = 1
    training_digest: str | None = None

    def __post_init__(self) -> None:
        if len(self.weights) != len(FILTER_NAMES):
            raise ValueError("one model weight is required per filter")
        if not math.isfinite(self.bias):
            raise ValueError("bias must be finite")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in self.weights):
            raise ValueError("weights must be finite and non-negative")
        if self.version < 1:
            raise ValueError("version must be positive")

    @property
    def model_hash(self) -> str:
        return sha256_hex(
            {
                "bias": self.bias,
                "weights": self.weights,
                "version": self.version,
                "training_digest": self.training_digest,
                "features": FILTER_NAMES,
            }
        )

    def score(self, severities: Mapping[str, float]) -> float:
        logit = self.bias + sum(
            weight * _clamp(float(severities.get(name, 0.0)), 0.0, 1.0)
            for name, weight in zip(FILTER_NAMES, self.weights)
        )
        return _sigmoid(logit)

    def fit_supervised(
        self,
        examples: Sequence[TrainingExample],
        *,
        learning_rate: float = 0.08,
        l2: float = 1.0e-3,
        max_iterations: int = 2_000,
        convergence_delta: float = 1.0e-3,
        convergence_patience: int = 10,
    ) -> "ScoreModel":
        """Fit only on explicit labels using projected gradient descent."""

        if not examples:
            raise ValueError("supervised fitting requires labelled examples")
        if learning_rate <= 0.0 or max_iterations < 1:
            raise ValueError("invalid training controls")

        weights = list(self.weights)
        bias = self.bias
        stable_iterations = 0
        for _ in range(max_iterations):
            gradient_weights = [0.0] * len(weights)
            gradient_bias = 0.0
            for example in examples:
                features = [
                    _clamp(float(example.severities.get(name, 0.0)), 0.0, 1.0)
                    for name in FILTER_NAMES
                ]
                probability = _sigmoid(
                    bias
                    + sum(
                        weight * feature
                        for weight, feature in zip(weights, features)
                    )
                )
                error = probability - example.label
                gradient_bias += error
                for index, feature in enumerate(features):
                    gradient_weights[index] += error * feature
            count = len(examples)
            new_bias = _clamp(
                bias - learning_rate * gradient_bias / count,
                -12.0,
                12.0,
            )
            new_weights = [
                _clamp(
                    weight
                    - learning_rate * (gradient / count + l2 * weight),
                    0.0,
                    12.0,
                )
                for weight, gradient in zip(weights, gradient_weights)
            ]
            change = max(
                [abs(new_bias - bias)]
                + [abs(new - old) for new, old in zip(new_weights, weights)]
            )
            bias, weights = new_bias, new_weights
            if change < convergence_delta:
                stable_iterations += 1
                if stable_iterations >= convergence_patience:
                    break
            else:
                stable_iterations = 0

        digest = sha256_hex(
            [
                {
                    "severities": {
                        name: float(example.severities.get(name, 0.0))
                        for name in FILTER_NAMES
                    },
                    "label": example.label,
                }
                for example in examples
            ]
        )
        return ScoreModel(
            bias=bias,
            weights=tuple(weights),
            version=self.version + 1,
            training_digest=digest,
        )


@dataclass(frozen=True, slots=True)
class HyperionConfig:
    target_quantity: str = "amount"
    target_unit: str = "ZAR"
    source_weights: Mapping[str, float] = field(
        default_factory=lambda: {"csv": 1.0, "audio": 1.0, "cpu": 1.0}
    )
    event_tolerance_ms: int = 25
    minimum_dt_ms: int = 1
    allow_unconfigured_sources: bool = False
    value_scale: int = 100
    weight_scale: int = 1_000_000
    robust_window: int = 30
    continuity_absolute_tolerance: float = 0.01
    continuity_relative_tolerance: float = 0.01
    robust_z_threshold: float = 3.5
    bytecode_absolute_tolerance: float = 0.01
    bytecode_relative_tolerance: float = 0.01
    precision_quantile: float = 0.80
    precision_limit_fraction: float = 0.02
    lower_bound: float | None = None
    critical_threshold: float = 0.75
    ghs_exposure_weight: float = 0.70
    ghs_frequency_weight: float = 0.30
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        if not self.target_quantity or not self.target_unit:
            raise ValueError("target quantity and unit are required")
        if self.event_tolerance_ms < 1:
            raise ValueError("event_tolerance_ms must be positive")
        if self.minimum_dt_ms < 1:
            raise ValueError("minimum_dt_ms must be positive")
        if self.value_scale < 1 or self.weight_scale < 1:
            raise ValueError("fixed-point scales must be positive")
        if self.robust_window < 5:
            raise ValueError("robust_window must be at least 5")
        if not 0.0 < self.precision_quantile < 1.0:
            raise ValueError("precision_quantile must be in (0, 1)")
        if not 0.0 < self.critical_threshold < 1.0:
            raise ValueError("critical_threshold must be in (0, 1)")
        if not math.isclose(
            self.ghs_exposure_weight + self.ghs_frequency_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("GHS weights must sum to 1")


@dataclass(frozen=True, slots=True)
class AuditPoint:
    event_id: str
    timestamp_ms: int
    fused_value: float
    fusion_confidence: float
    delta_value: float
    velocity: float
    acceleration: float
    jerk: float
    predicted_value: float
    continuity_residual: float
    severities: Mapping[str, float]
    flags: Mapping[str, bool]
    cas: float
    critical: bool
    label_present: bool
    witness_commitment: str

    @property
    def digest(self) -> str:
        return sha256_hex(
            {
                "event_id": self.event_id,
                "timestamp_ms": self.timestamp_ms,
                "fused_value": self.fused_value,
                "fusion_confidence": self.fusion_confidence,
                "delta_value": self.delta_value,
                "velocity": self.velocity,
                "acceleration": self.acceleration,
                "jerk": self.jerk,
                "predicted_value": self.predicted_value,
                "continuity_residual": self.continuity_residual,
                "severities": dict(self.severities),
                "flags": dict(self.flags),
                "cas": self.cas,
                "critical": self.critical,
                "label_present": self.label_present,
                "witness_commitment": self.witness_commitment,
            }
        )


@dataclass(frozen=True, slots=True)
class AuditReport:
    points: tuple[AuditPoint, ...]
    witnesses: tuple[ArithmeticWitness, ...]
    geometric_health_score: float
    model_hash: str
    configuration_hash: str
    input_root: str
    output_root: str
    witness_root: str
    report_digest: str

    def verify(self) -> bool:
        if len(self.points) != len(self.witnesses):
            return False
        if any(not witness.verify() for witness in self.witnesses):
            return False
        if any(
            point.witness_commitment != witness.commitment
            for point, witness in zip(self.points, self.witnesses)
        ):
            return False
        if self.output_root != merkle_root([point.digest for point in self.points]):
            return False
        if self.witness_root != merkle_root(
            [witness.commitment for witness in self.witnesses]
        ):
            return False
        expected = sha256_hex(
            {
                "ghs": self.geometric_health_score,
                "model_hash": self.model_hash,
                "configuration_hash": self.configuration_hash,
                "input_root": self.input_root,
                "output_root": self.output_root,
                "witness_root": self.witness_root,
            }
        )
        return expected == self.report_digest


@dataclass(slots=True)
class _Event:
    event_id: str
    timestamp_ms: int
    observations: list[Observation]


class HyperionEngine:
    """Deterministic event-time forensic arithmetic engine."""

    def __init__(
        self,
        config: HyperionConfig | None = None,
        model: ScoreModel | None = None,
    ) -> None:
        self.config = config or HyperionConfig()
        self.model = model or ScoreModel()

    @property
    def configuration_hash(self) -> str:
        return sha256_hex(
            {
                **asdict(self.config),
                "source_weights": dict(sorted(self.config.source_weights.items())),
            }
        )

    def _event_key(self, observation: Observation) -> str:
        if observation.correlation_id:
            return f"id:{observation.correlation_id}"
        bucket = observation.timestamp_ms // self.config.event_tolerance_ms
        return f"time:{bucket}"

    def align(self, observations: Iterable[Observation]) -> tuple[_Event, ...]:
        groups: dict[str, list[Observation]] = {}
        for observation in observations:
            groups.setdefault(self._event_key(observation), []).append(observation)

        events: list[_Event] = []
        for key, grouped in groups.items():
            compatible = [
                observation
                for observation in grouped
                if observation.quantity == self.config.target_quantity
                and observation.unit == self.config.target_unit
                and observation.available
            ]
            if not compatible:
                continue
            timestamp_ms = round(
                statistics.median(item.timestamp_ms for item in compatible)
            )
            event_id = key.removeprefix("id:")
            events.append(
                _Event(
                    event_id=event_id,
                    timestamp_ms=timestamp_ms,
                    observations=sorted(
                        grouped,
                        key=lambda item: (item.source, item.timestamp_ms, item.value),
                    ),
                )
            )
        events.sort(key=lambda event: (event.timestamp_ms, event.event_id))
        return tuple(events)

    def _compatible(self, event: _Event) -> list[Observation]:
        candidates = [
            observation
            for observation in event.observations
            if observation.available
            and observation.quantity == self.config.target_quantity
            and observation.unit == self.config.target_unit
            and (
                self.config.allow_unconfigured_sources
                or observation.source in self.config.source_weights
            )
        ]
        selected: dict[str, Observation] = {}
        for observation in candidates:
            current = selected.get(observation.source)
            if current is None or (
                observation.confidence,
                -abs(observation.timestamp_ms - event.timestamp_ms),
                observation.value,
            ) > (
                current.confidence,
                -abs(current.timestamp_ms - event.timestamp_ms),
                current.value,
            ):
                selected[observation.source] = observation
        return [selected[source] for source in sorted(selected)]

    @staticmethod
    def _stream_value(
        observations: Sequence[Observation], source: str
    ) -> float | None:
        candidates = [
            observation
            for observation in observations
            if observation.source == source and observation.available
        ]
        if not candidates:
            return None
        best = max(
            candidates,
            key=lambda item: (item.confidence, -item.timestamp_ms, item.value),
        )
        return best.value

    def audit(self, observations: Iterable[Observation]) -> AuditReport:
        materialized = tuple(observations)
        input_leaves = [
            sha256_hex(
                {
                    "source": item.source,
                    "timestamp_ms": item.timestamp_ms,
                    "value": item.value,
                    "quantity": item.quantity,
                    "unit": item.unit,
                    "correlation_id": item.correlation_id,
                    "confidence": item.confidence,
                    "available": item.available,
                    "label": item.label,
                    "metadata": dict(item.metadata),
                }
            )
            for item in sorted(
                materialized,
                key=lambda item: (
                    item.timestamp_ms,
                    item.correlation_id or "",
                    item.source,
                    item.value,
                ),
            )
        ]
        input_root = merkle_root(input_leaves)

        events = self.align(materialized)
        points: list[AuditPoint] = []
        witnesses: list[ArithmeticWitness] = []
        fused_history: list[float] = []
        acceleration_history: list[float] = []
        negative_delta_history: list[float] = []

        for event in events:
            compatible = self._compatible(event)
            witness = ArithmeticWitness.build(
                event_id=event.event_id,
                timestamp_ms=event.timestamp_ms,
                quantity=self.config.target_quantity,
                unit=self.config.target_unit,
                observations=compatible,
                source_weights=self.config.source_weights,
                value_scale=self.config.value_scale,
                weight_scale=self.config.weight_scale,
            )
            witnesses.append(witness)
            fused_value = witness.fused_value
            effective_weight_sum = sum(
                float(self.config.source_weights.get(item.source, 1.0))
                * item.confidence
                for item in compatible
            )
            nominal_weight_sum = sum(
                max(0.0, float(self.config.source_weights.get(item.source, 1.0)))
                for item in compatible
            )
            fusion_confidence = (
                _clamp(effective_weight_sum / nominal_weight_sum, 0.0, 1.0)
                if nominal_weight_sum > self.config.epsilon
                else 0.0
            )

            if not points:
                delta_value = 0.0
                velocity = 0.0
                acceleration = 0.0
                jerk = 0.0
                predicted = fused_value
            else:
                previous = points[-1]
                delta_ms = event.timestamp_ms - previous.timestamp_ms
                if delta_ms < 0:
                    raise ValueError("aligned event timestamps must be non-decreasing")
                dt_seconds = max(delta_ms, self.config.minimum_dt_ms) / 1000.0
                delta_value = fused_value - previous.fused_value
                velocity = delta_value / dt_seconds
                acceleration = (velocity - previous.velocity) / dt_seconds
                jerk = (acceleration - previous.acceleration) / dt_seconds
                predicted = (
                    previous.fused_value
                    + previous.velocity * dt_seconds
                    + 0.5 * previous.acceleration * dt_seconds * dt_seconds
                )

            residual = fused_value - predicted
            continuity_scale = (
                self.config.continuity_absolute_tolerance
                + self.config.continuity_relative_tolerance
                * max(abs(predicted), abs(points[-1].fused_value) if points else 0.0)
            )
            continuity_ratio = abs(residual) / max(
                continuity_scale,
                self.config.epsilon,
            )
            continuity_severity = _clamp(continuity_ratio / 3.0, 0.0, 1.0)
            continuity_flag = continuity_ratio > 1.0

            acceleration_window = acceleration_history[-self.config.robust_window :]
            acceleration_z = _robust_z(
                acceleration,
                acceleration_window,
                self.config.epsilon,
            )
            acceleration_severity = (
                1.0
                if math.isinf(acceleration_z)
                else _clamp(
                    abs(acceleration_z) / self.config.robust_z_threshold,
                    0.0,
                    1.0,
                )
            )
            acceleration_flag = abs(acceleration_z) > self.config.robust_z_threshold

            precision_flag = False
            precision_severity = 0.0
            if (
                points
                and self.config.lower_bound is not None
                and delta_value < 0.0
                and negative_delta_history
            ):
                debit_threshold = _quantile(
                    negative_delta_history[-self.config.robust_window :],
                    self.config.precision_quantile,
                )
                projected_balance = points[-1].fused_value + delta_value
                buffer = projected_balance - self.config.lower_bound
                buffer_limit = self.config.precision_limit_fraction * abs(
                    self.config.lower_bound
                )
                magnitude_ratio = abs(delta_value) / max(
                    debit_threshold,
                    self.config.epsilon,
                )
                proximity = (
                    _clamp(
                        1.0 - buffer / max(buffer_limit, self.config.epsilon),
                        0.0,
                        1.0,
                    )
                    if buffer >= 0.0
                    else 0.0
                )
                precision_severity = _clamp(
                    min(1.0, magnitude_ratio) * proximity,
                    0.0,
                    1.0,
                )
                precision_flag = (
                    abs(delta_value) >= debit_threshold
                    and 0.0 <= buffer < buffer_limit
                )

            label_present = any(
                bool(observation.label and observation.label.strip())
                for observation in event.observations
            )
            value_window = fused_history[-self.config.robust_window :]
            magnitude_z = _robust_z(
                fused_value,
                value_window,
                self.config.epsilon,
            )
            magnitude_severity = (
                1.0
                if math.isinf(magnitude_z)
                else _clamp(
                    abs(magnitude_z) / self.config.robust_z_threshold,
                    0.0,
                    1.0,
                )
            )
            ghost_severity = magnitude_severity if not label_present else 0.0
            ghost_flag = (
                not label_present and abs(magnitude_z) > self.config.robust_z_threshold
            )

            csv_value = self._stream_value(compatible, "csv")
            cpu_value = self._stream_value(compatible, "cpu")
            bytecode_flag = False
            bytecode_severity = 0.0
            if csv_value is not None and cpu_value is not None:
                divergence = abs(csv_value - cpu_value)
                tolerance = (
                    self.config.bytecode_absolute_tolerance
                    + self.config.bytecode_relative_tolerance * abs(csv_value)
                )
                divergence_ratio = divergence / max(tolerance, self.config.epsilon)
                bytecode_severity = _clamp(divergence_ratio / 3.0, 0.0, 1.0)
                bytecode_flag = divergence_ratio > 1.0

            severities = {
                "continuity": continuity_severity,
                "acceleration_spike": acceleration_severity,
                "precision_strike": precision_severity,
                "ghost_entity": ghost_severity,
                "bytecode_divergence": bytecode_severity,
            }
            flags = {
                "continuity": continuity_flag,
                "acceleration_spike": acceleration_flag,
                "precision_strike": precision_flag,
                "ghost_entity": ghost_flag,
                "bytecode_divergence": bytecode_flag,
            }
            cas = self.model.score(severities)
            point = AuditPoint(
                event_id=event.event_id,
                timestamp_ms=event.timestamp_ms,
                fused_value=fused_value,
                fusion_confidence=fusion_confidence,
                delta_value=delta_value,
                velocity=velocity,
                acceleration=acceleration,
                jerk=jerk,
                predicted_value=predicted,
                continuity_residual=residual,
                severities=severities,
                flags=flags,
                cas=cas,
                critical=cas >= self.config.critical_threshold,
                label_present=label_present,
                witness_commitment=witness.commitment,
            )
            points.append(point)
            fused_history.append(fused_value)
            acceleration_history.append(acceleration)
            if delta_value < 0.0:
                negative_delta_history.append(abs(delta_value))

        total_exposure = sum(abs(point.fused_value) for point in points)
        critical_exposure = sum(
            abs(point.fused_value) for point in points if point.critical
        )
        exposure_ratio = (
            critical_exposure / total_exposure
            if total_exposure > self.config.epsilon
            else 0.0
        )
        frequency_ratio = (
            sum(point.critical for point in points) / len(points) if points else 0.0
        )
        ghs = 100.0 * _clamp(
            1.0
            - self.config.ghs_exposure_weight * exposure_ratio
            - self.config.ghs_frequency_weight * frequency_ratio,
            0.0,
            1.0,
        )

        output_root = merkle_root([point.digest for point in points])
        witness_root = merkle_root([witness.commitment for witness in witnesses])
        digest = sha256_hex(
            {
                "ghs": ghs,
                "model_hash": self.model.model_hash,
                "configuration_hash": self.configuration_hash,
                "input_root": input_root,
                "output_root": output_root,
                "witness_root": witness_root,
            }
        )
        return AuditReport(
            points=tuple(points),
            witnesses=tuple(witnesses),
            geometric_health_score=ghs,
            model_hash=self.model.model_hash,
            configuration_hash=self.configuration_hash,
            input_root=input_root,
            output_root=output_root,
            witness_root=witness_root,
            report_digest=digest,
        )


def binary_cross_entropy(
    model: ScoreModel,
    examples: Sequence[TrainingExample],
) -> float:
    if not examples:
        raise ValueError("examples are required")
    total = 0.0
    for example in examples:
        probability = _clamp(
            model.score(example.severities),
            1.0e-12,
            1.0 - 1.0e-12,
        )
        total -= example.label * math.log(probability)
        total -= (1 - example.label) * math.log(1.0 - probability)
    return total / len(examples)
