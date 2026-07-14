"""Operational MM3D auto-encoding/decoding cycle for Jarvis-X.

The implementation intentionally treats the AED equation as a deterministic
representation transform. It does not equate a decoded representation with
physical reality; the type-level semantic gap is preserved in every state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


Vector = Tuple[float, ...]
QuantizedVector = Tuple[int, ...]
ConstraintVector = Tuple[Tuple[float, float], ...]


@dataclass(frozen=True)
class AEDConfig:
    """Numerical and invariant controls for one synchronous AED cycle."""

    ambient_min: float = 0.0
    ambient_max: float = 255.0
    latent_min: int = -4
    latent_max: int = 3
    memory_coupling: float = 0.25
    intent_gain: float = 0.50
    semantic_gap_epsilon: float = 1.0e-9

    def __post_init__(self) -> None:
        if self.ambient_min != 0.0 or self.ambient_max != 255.0:
            raise ValueError("the reference AED decoder requires the 8-bit ambient domain [0, 255]")
        if self.latent_max <= self.latent_min:
            raise ValueError("latent_max must be greater than latent_min")
        if not 0.0 <= self.memory_coupling <= 1.0:
            raise ValueError("memory_coupling must be in [0, 1]")
        if self.intent_gain < 0.0:
            raise ValueError("intent_gain must be non-negative")
        if self.semantic_gap_epsilon <= 0.0:
            raise ValueError("semantic_gap_epsilon must be positive")


@dataclass(frozen=True)
class AEDState:
    """Immutable record committed after a front/back-buffer swap."""

    cycle: int
    ambient_input: Vector
    latent_encoded: QuantizedVector
    latent_coupled: Vector
    latent_projected: Vector
    ambient_output: QuantizedVector
    reconstruction_rmse: float
    semantic_gap: float
    reality_separation: float
    representation_tag: str = "SIMULATION_NOT_TERRITORY"


class MM3DAEDEngine:
    """Deterministic, constant-stage AED transform with double buffering.

    A cycle has a fixed number of operators:

        encode -> Q3 quantize -> memory couple -> HSLF project -> decode -> swap

    The number of stages is constant, while arithmetic work is O(n) in the
    number of ambient coordinates.
    """

    def __init__(self, config: Optional[AEDConfig] = None) -> None:
        self.config = config or AEDConfig()
        self._front: Optional[AEDState] = None
        self._back: Optional[AEDState] = None
        self._cycle = 0

    @property
    def front_state(self) -> Optional[AEDState]:
        return self._front

    @property
    def back_state(self) -> Optional[AEDState]:
        return self._back

    def encode(self, ambient: Sequence[float]) -> QuantizedVector:
        """Map ambient coordinates to the signed 3-bit set Q3={-4,...,3}."""

        values = self._coerce_nonempty(ambient, "ambient")
        cfg = self.config
        ambient_span = cfg.ambient_max - cfg.ambient_min
        latent_span = cfg.latent_max - cfg.latent_min
        encoded = []
        for value in values:
            clipped = self._clip(value, cfg.ambient_min, cfg.ambient_max)
            ratio = (clipped - cfg.ambient_min) / ambient_span
            mapped = cfg.latent_min + ratio * latent_span
            quantized = int(math.floor(mapped + 0.5))
            encoded.append(int(self._clip(quantized, cfg.latent_min, cfg.latent_max)))
        return tuple(encoded)

    def spin_couple(
        self,
        latent: Sequence[float],
        memory: Optional[Sequence[float]] = None,
    ) -> Vector:
        """Couple the latent vector to bounded historical memory Ω."""

        z = self._coerce_nonempty(latent, "latent")
        omega = self._broadcast_or_validate(memory, len(z), 0.0, "memory")
        gain = self.config.memory_coupling
        return tuple(
            self._clip(
                value + gain * (memory_value - value),
                self.config.latent_min,
                self.config.latent_max,
            )
            for value, memory_value in zip(z, omega)
        )

    def hslf_project(
        self,
        latent: Sequence[float],
        intent: Optional[Sequence[float]] = None,
        constraints: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Vector:
        """Apply the HSLF log-map, semantic translation, inverse map, and gate.

        The HSLF stage is implemented as a direct bounded map, not as an
        unbounded iterative solver.
        """

        z = self._coerce_nonempty(latent, "latent")
        theta = self._broadcast_or_validate(intent, len(z), 0.0, "intent")
        bounds = self._constraints_or_default(constraints, len(z))
        projected = []
        for value, semantic_vector, (lower, upper) in zip(z, theta, bounds):
            log_coordinate = math.copysign(math.log1p(abs(value)), value)
            translated = log_coordinate + self.config.intent_gain * semantic_vector
            expanded = math.copysign(math.expm1(abs(translated)), translated)
            projected.append(self._clip(expanded, lower, upper))
        return tuple(projected)

    def decode(self, latent: Sequence[float]) -> QuantizedVector:
        """Lift stabilized latent coordinates into clipped 8-bit ambient space."""

        z = self._coerce_nonempty(latent, "latent")
        cfg = self.config
        latent_span = cfg.latent_max - cfg.latent_min
        ambient_span = cfg.ambient_max - cfg.ambient_min
        decoded = []
        for value in z:
            clipped = self._clip(value, cfg.latent_min, cfg.latent_max)
            ratio = (clipped - cfg.latent_min) / latent_span
            mapped = cfg.ambient_min + ratio * ambient_span
            decoded.append(int(self._clip(math.floor(mapped + 0.5), 0, 255)))
        return tuple(decoded)

    def cycle(
        self,
        ambient: Sequence[float],
        *,
        memory: Optional[Sequence[float]] = None,
        intent: Optional[Sequence[float]] = None,
        constraints: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> AEDState:
        """Execute one complete synchronous AED mapping and commit by swap."""

        ambient_values = self._coerce_nonempty(ambient, "ambient")
        encoded = self.encode(ambient_values)
        coupled = self.spin_couple(encoded, memory)
        projected = self.hslf_project(coupled, intent, constraints)
        decoded = self.decode(projected)
        rmse = math.sqrt(
            sum((source - target) ** 2 for source, target in zip(ambient_values, decoded))
            / len(ambient_values)
        )

        self._cycle += 1
        self._back = AEDState(
            cycle=self._cycle,
            ambient_input=ambient_values,
            latent_encoded=encoded,
            latent_coupled=coupled,
            latent_projected=projected,
            ambient_output=decoded,
            reconstruction_rmse=rmse,
            semantic_gap=self.config.semantic_gap_epsilon,
            reality_separation=math.inf,
        )
        self._front, self._back = self._back, self._front
        return self._front

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)

    @staticmethod
    def _coerce_nonempty(values: Sequence[float], name: str) -> Vector:
        result = tuple(float(value) for value in values)
        if not result:
            raise ValueError(f"{name} must contain at least one coordinate")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} must contain only finite coordinates")
        return result

    @staticmethod
    def _broadcast_or_validate(
        values: Optional[Sequence[float]],
        size: int,
        default: float,
        name: str,
    ) -> Vector:
        if values is None:
            return (default,) * size
        result = tuple(float(value) for value in values)
        if len(result) == 1:
            result = result * size
        if len(result) != size:
            raise ValueError(f"{name} must have length 1 or {size}")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} must contain only finite coordinates")
        return result

    def _constraints_or_default(
        self,
        constraints: Optional[Sequence[Tuple[float, float]]],
        size: int,
    ) -> ConstraintVector:
        if constraints is None:
            return ((float(self.config.latent_min), float(self.config.latent_max)),) * size
        result = tuple((float(lower), float(upper)) for lower, upper in constraints)
        if len(result) == 1:
            result = result * size
        if len(result) != size:
            raise ValueError(f"constraints must have length 1 or {size}")
        for lower, upper in result:
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError("constraint bounds must be finite")
            if lower > upper:
                raise ValueError("constraint lower bound cannot exceed upper bound")
            if lower < self.config.latent_min or upper > self.config.latent_max:
                raise ValueError("constraints must remain inside the Q3 latent domain")
        return result
