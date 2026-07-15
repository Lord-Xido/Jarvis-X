"""Deterministic hierarchical cognitive kernel for Jarvis-X.

The kernel operationalises the canonical cycle:

    encode -> condense -> predict -> compare -> update Omega
    -> project Lambda -> decode -> commit/rollback

All committed state is integer-valued and hash chained. Candidate state is
validated before it is exposed through the VM register bridge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

Number = Union[int, float]
Q3_MIN = -4
Q3_MAX = 3


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _div_round_nearest(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def quantize_q3(value: Number, scale: float = 1.0) -> int:
    """Quantize a scalar into the signed 3-bit domain {-4, ..., 3}."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("input values must be finite")
    return _clamp(_round_half_away_from_zero(numeric * scale), Q3_MIN, Q3_MAX)


@dataclass(frozen=True)
class CognitiveConfig:
    branch_factor: int = 4
    input_scale: float = 1.0
    max_levels: int = 16
    max_input_values: int = 65536
    retention_numerator: int = 7
    retention_denominator: int = 8
    learning_numerator: int = 1
    learning_denominator: int = 2
    omega_prediction_scale: int = 4
    omega_limit: int = 127
    max_residual_l1: Optional[int] = None
    max_active_nodes: Optional[int] = None
    journal_limit: int = 128

    def validate(self) -> None:
        if self.branch_factor < 2:
            raise ValueError("branch_factor must be >= 2")
        if self.max_levels < 1:
            raise ValueError("max_levels must be >= 1")
        if self.max_input_values < 1:
            raise ValueError("max_input_values must be positive")
        if self.input_scale <= 0 or not math.isfinite(self.input_scale):
            raise ValueError("input_scale must be positive and finite")
        if self.retention_denominator <= 0 or self.learning_denominator <= 0:
            raise ValueError("update denominators must be positive")
        if not 0 <= self.retention_numerator <= self.retention_denominator:
            raise ValueError("retention ratio must be between 0 and 1")
        if self.learning_numerator < 0:
            raise ValueError("learning_numerator must be non-negative")
        if self.omega_prediction_scale <= 0:
            raise ValueError("omega_prediction_scale must be positive")
        if self.omega_limit < 1:
            raise ValueError("omega_limit must be positive")
        if self.max_residual_l1 is not None and self.max_residual_l1 < 0:
            raise ValueError("max_residual_l1 must be non-negative")
        if self.max_active_nodes is not None and self.max_active_nodes < 0:
            raise ValueError("max_active_nodes must be non-negative")
        if self.journal_limit < 1:
            raise ValueError("journal_limit must be positive")


@dataclass(frozen=True)
class CognitiveState:
    cycle: int = 0
    encoded: Tuple[int, ...] = field(default_factory=tuple)
    hierarchy: Tuple[Tuple[int, ...], ...] = field(default_factory=tuple)
    prediction: Tuple[int, ...] = field(default_factory=tuple)
    residual: Tuple[int, ...] = field(default_factory=tuple)
    omega: Tuple[int, ...] = field(default_factory=tuple)
    decoded: Tuple[int, ...] = field(default_factory=tuple)
    state_hash: str = "GENESIS"


@dataclass(frozen=True)
class CycleResult:
    cycle: int
    committed: bool
    reason: str
    encoded: Tuple[int, ...]
    hierarchy: Tuple[Tuple[int, ...], ...]
    prediction: Tuple[int, ...]
    residual: Tuple[int, ...]
    omega_before: Tuple[int, ...]
    omega_after: Tuple[int, ...]
    decoded: Tuple[int, ...]
    metrics: Dict[str, float]
    previous_hash: str
    candidate_hash: str
    state_hash: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class CognitiveKernel:
    """Transactional signed-3-bit hierarchical predictive runtime."""

    def __init__(self, config: Optional[CognitiveConfig] = None) -> None:
        self.config = config or CognitiveConfig()
        self.config.validate()
        self.state = CognitiveState()
        self.journal: List[CycleResult] = []

    def encode(self, values: Union[Number, Sequence[Number]]) -> Tuple[int, ...]:
        if isinstance(values, (int, float)):
            source: Iterable[Number] = (values,)
        else:
            source = values
        encoded = tuple(quantize_q3(value, self.config.input_scale) for value in source)
        if not encoded:
            raise ValueError("at least one input value is required")
        if len(encoded) > self.config.max_input_values:
            raise ValueError("input exceeds max_input_values")
        return encoded

    def condense(self, encoded: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
        levels: List[Tuple[int, ...]] = [encoded]
        current = encoded

        while len(current) > 1:
            if len(levels) >= self.config.max_levels:
                raise ValueError("hierarchy exceeds max_levels")
            parent: List[int] = []
            for start in range(0, len(current), self.config.branch_factor):
                group = current[start : start + self.config.branch_factor]
                mean = _div_round_nearest(sum(group), len(group))
                parent.append(_clamp(mean, Q3_MIN, Q3_MAX))
            current = tuple(parent)
            levels.append(current)

        return tuple(levels)

    def predict(self, encoded_length: int) -> Tuple[int, ...]:
        previous = self.state.encoded
        omega = self.state.omega
        prediction: List[int] = []
        for index in range(encoded_length):
            base = previous[index] if index < len(previous) else 0
            memory = omega[index] if index < len(omega) else 0
            correction = _div_round_nearest(memory, self.config.omega_prediction_scale)
            prediction.append(_clamp(base + correction, Q3_MIN, Q3_MAX))
        return tuple(prediction)

    @staticmethod
    def compare(encoded: Tuple[int, ...], prediction: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(actual - expected for actual, expected in zip(encoded, prediction))

    def update_omega(self, residual: Tuple[int, ...]) -> Tuple[int, ...]:
        updated: List[int] = []
        for index, error in enumerate(residual):
            old = self.state.omega[index] if index < len(self.state.omega) else 0
            retained = _div_round_nearest(
                old * self.config.retention_numerator,
                self.config.retention_denominator,
            )
            learned = _div_round_nearest(
                error * self.config.learning_numerator,
                self.config.learning_denominator,
            )
            updated.append(
                _clamp(retained + learned, -self.config.omega_limit, self.config.omega_limit)
            )
        return tuple(updated)

    def decode(self, hierarchy: Tuple[Tuple[int, ...], ...]) -> Tuple[int, ...]:
        """Reconstruct the leaf width from the finest available condensed level."""
        if len(hierarchy) == 1:
            return hierarchy[0]

        reconstructed = hierarchy[1]
        expanded: List[int] = []
        for value in reconstructed:
            expanded.extend([value] * self.config.branch_factor)
        return tuple(
            _clamp(value, Q3_MIN, Q3_MAX) for value in expanded[: len(hierarchy[0])]
        )

    def _validate_candidate(
        self,
        encoded: Tuple[int, ...],
        hierarchy: Tuple[Tuple[int, ...], ...],
        residual: Tuple[int, ...],
        omega: Tuple[int, ...],
    ) -> Tuple[bool, str]:
        if not hierarchy or hierarchy[0] != encoded or len(hierarchy[-1]) != 1:
            return False, "invalid hierarchy"
        if len(hierarchy) > self.config.max_levels:
            return False, "hierarchy depth exceeded"
        if any(value < Q3_MIN or value > Q3_MAX for level in hierarchy for value in level):
            return False, "q3 range violation"
        if any(abs(value) > self.config.omega_limit for value in omega):
            return False, "omega bound violation"

        residual_l1 = sum(abs(value) for value in residual)
        if self.config.max_residual_l1 is not None and residual_l1 > self.config.max_residual_l1:
            return False, "residual budget exceeded"

        active_nodes = sum(1 for level in hierarchy for value in level if value != 0)
        if self.config.max_active_nodes is not None and active_nodes > self.config.max_active_nodes:
            return False, "active-node budget exceeded"

        return True, "committed"

    @staticmethod
    def _hash_candidate(payload: Dict[str, object]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _metrics(
        self,
        encoded: Tuple[int, ...],
        hierarchy: Tuple[Tuple[int, ...], ...],
        residual: Tuple[int, ...],
        omega: Tuple[int, ...],
        decoded: Tuple[int, ...],
    ) -> Dict[str, float]:
        raw_nodes = len(encoded)
        root_nodes = len(hierarchy[-1])
        hierarchy_nodes = sum(len(level) for level in hierarchy)
        active_nodes = sum(1 for level in hierarchy for value in level if value != 0)
        reconstruction_error = sum(abs(a - b) for a, b in zip(encoded, decoded))
        return {
            "raw_nodes": float(raw_nodes),
            "raw_bits": float(raw_nodes * 3),
            "root_nodes": float(root_nodes),
            "root_bits": float(root_nodes * 3),
            "hierarchy_nodes": float(hierarchy_nodes),
            "hierarchy_levels": float(len(hierarchy)),
            "condensation_ratio": float(raw_nodes) / float(root_nodes),
            "materialization_ratio": float(hierarchy_nodes) / float(raw_nodes),
            "active_fraction": float(active_nodes) / float(hierarchy_nodes),
            "residual_l1": float(sum(abs(value) for value in residual)),
            "memory_l1": float(sum(abs(value) for value in omega)),
            "reconstruction_error_l1": float(reconstruction_error),
        }

    def step(self, values: Union[Number, Sequence[Number]]) -> CycleResult:
        previous_hash = self.state.state_hash
        encoded = self.encode(values)
        hierarchy = self.condense(encoded)
        prediction = self.predict(len(encoded))
        residual = self.compare(encoded, prediction)
        omega_before = self.state.omega
        omega_after = self.update_omega(residual)
        decoded = self.decode(hierarchy)
        metrics = self._metrics(encoded, hierarchy, residual, omega_after, decoded)
        cycle = self.state.cycle + 1

        candidate_payload: Dict[str, object] = {
            "config": asdict(self.config),
            "cycle": cycle,
            "encoded": encoded,
            "hierarchy": hierarchy,
            "prediction": prediction,
            "residual": residual,
            "omega": omega_after,
            "decoded": decoded,
            "previous_hash": previous_hash,
        }
        candidate_hash = self._hash_candidate(candidate_payload)
        valid, reason = self._validate_candidate(encoded, hierarchy, residual, omega_after)

        if valid:
            self.state = CognitiveState(
                cycle=cycle,
                encoded=encoded,
                hierarchy=hierarchy,
                prediction=prediction,
                residual=residual,
                omega=omega_after,
                decoded=decoded,
                state_hash=candidate_hash,
            )

        result = CycleResult(
            cycle=cycle,
            committed=valid,
            reason=reason,
            encoded=encoded,
            hierarchy=hierarchy,
            prediction=prediction,
            residual=residual,
            omega_before=omega_before,
            omega_after=omega_after,
            decoded=decoded,
            metrics=metrics,
            previous_hash=previous_hash,
            candidate_hash=candidate_hash,
            state_hash=self.state.state_hash,
        )
        self.journal.append(result)
        if len(self.journal) > self.config.journal_limit:
            del self.journal[: len(self.journal) - self.config.journal_limit]
        return result

    def run(self, stream: Iterable[Union[Number, Sequence[Number]]]) -> List[CycleResult]:
        return [self.step(values) for values in stream]

    def snapshot(self) -> Dict[str, object]:
        return asdict(self.state)


class CognitiveVMBridge:
    """Expose only committed kernel state through the Jarvis-X register bank."""

    def __init__(self, registers: object, kernel: Optional[CognitiveKernel] = None) -> None:
        self.registers = registers
        self.kernel = kernel or CognitiveKernel()

    def cycle(self, values: Union[Number, Sequence[Number]]) -> CycleResult:
        result = self.kernel.step(values)
        if not result.committed:
            self.registers["Λ"] = 0
            return result

        before = self.registers.snapshot()
        try:
            root = result.hierarchy[-1][0]
            self.registers["Ξ"] = sum(result.encoded)
            self.registers["Ψ"] = root
            self.registers["Φ"] = sum(result.prediction)
            self.registers["Λ"] = 1
            self.registers["Ω"] = sum(result.omega_after)
            self.registers["Θ"] = self.kernel.config.learning_numerator
            self.registers["𝒮"] = int(result.metrics["residual_l1"])
            self.registers["Π"] = sum(result.decoded)
        except Exception:
            for name, value in before.items():
                self.registers[name] = value
            raise
        return result
