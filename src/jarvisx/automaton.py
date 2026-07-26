"""Sparse 3-D auto-encoding/decoding simulation automaton.

The nominal universe contains ``(1000**1000)**3 == 10**9000`` cells. The
runtime never allocates that tensor. Coordinates are exact Python integers,
untouched cells are reconstructed from a deterministic procedural field, and
only a bounded causal frontier is materialised.

The committed update law is the executable form of::

    Sigma[t+1] = Pi_Lambda(
        Sigma[t]
        + P_theta(E_theta(Sigma[t]))
        - K * error[t]
        + Omega[t]
        + input[t]
    )

All proposals are calculated against an immutable snapshot, verified, and then
committed atomically. A failed verification leaves the prior state untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

RADIX = 1000
ADDRESS_DEPTH = 1000
AXIS_SIZE = RADIX ** ADDRESS_DEPTH
VIRTUAL_CELL_EXPONENT = 9000
LOCAL_INPUT_DIM = 7


@dataclass(frozen=True, order=True)
class Coordinate3D:
    """One exact coordinate in the ``1000**1000`` cubical address space."""

    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not isinstance(value, int):
                raise TypeError("{} must be an integer".format(name))
            if value < 0 or value >= AXIS_SIZE:
                raise ValueError("{} must satisfy 0 <= {} < 1000**1000".format(name, name))

    def offset(self, dx: int = 0, dy: int = 0, dz: int = 0) -> "Coordinate3D":
        return Coordinate3D(
            (self.x + dx) % AXIS_SIZE,
            (self.y + dy) % AXIS_SIZE,
            (self.z + dz) % AXIS_SIZE,
        )

    def radix_prefix(self, digits: int = 4) -> Tuple[Tuple[int, ...], ...]:
        if digits < 1 or digits > ADDRESS_DEPTH:
            raise ValueError("digits must be in [1, 1000]")
        divisor = RADIX ** (ADDRESS_DEPTH - digits)

        def axis_prefix(value: int) -> Tuple[int, ...]:
            head = value // divisor
            out = [0] * digits
            for index in range(digits - 1, -1, -1):
                out[index] = head % RADIX
                head //= RADIX
            return tuple(out)

        return axis_prefix(self.x), axis_prefix(self.y), axis_prefix(self.z)


@dataclass(frozen=True)
class CellState:
    value: float
    omega: float = 0.0
    inactive_steps: int = 0
    revision: int = 0


@dataclass(frozen=True)
class Mechanics:
    diffusion: float = 0.06
    error_gain: float = 0.55
    omega_retention: float = 0.88
    omega_rate: float = 0.12
    time_step: float = 0.18
    activation_threshold: float = 0.03
    prune_after: int = 8
    max_abs_value: float = 4.0
    max_energy: float = 1000000.0
    max_active_cells: int = 10000

    def validate(self) -> None:
        bounded = {
            "diffusion": (self.diffusion, 0.0, 1.0),
            "error_gain": (self.error_gain, 0.0, 4.0),
            "omega_retention": (self.omega_retention, 0.0, 1.0),
            "omega_rate": (self.omega_rate, 0.0, 1.0),
            "time_step": (self.time_step, 0.0, 1.0),
            "activation_threshold": (self.activation_threshold, 0.0, self.max_abs_value),
        }
        for name, (value, lower, upper) in bounded.items():
            if not math.isfinite(value) or value < lower or value > upper:
                raise ValueError("{} must be finite and in [{}, {}]".format(name, lower, upper))
        if self.prune_after < 1:
            raise ValueError("prune_after must be positive")
        if self.max_abs_value <= 0.0 or not math.isfinite(self.max_abs_value):
            raise ValueError("max_abs_value must be finite and positive")
        if self.max_energy <= 0.0 or not math.isfinite(self.max_energy):
            raise ValueError("max_energy must be finite and positive")
        if self.max_active_cells < 1:
            raise ValueError("max_active_cells must be positive")


@dataclass(frozen=True)
class StepMetrics:
    cycle: int
    materialised_cells: int
    active_cells: int
    frontier_cells: int
    reconstruction_mse: float
    energy: float
    committed: bool
    journal_hash: str
    rollback_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationResult:
    adopted: bool
    baseline_score: float
    candidate_score: float
    previous_mechanics: Mechanics
    selected_mechanics: Mechanics


class DeterministicAutoencoder:
    """Dependency-free deterministic ANN for a local 3-D neighbourhood."""

    def __init__(self, latent_dim: int = 8, seed: int = 1337) -> None:
        if latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")
        self.input_dim = LOCAL_INPUT_DIM
        self.latent_dim = int(latent_dim)
        rng = random.Random(int(seed))
        encoder_scale = 1.0 / math.sqrt(float(self.input_dim))
        transition_scale = 0.45 / math.sqrt(float(self.latent_dim))
        self._encoder = tuple(
            tuple(rng.uniform(-encoder_scale, encoder_scale) for _ in range(self.input_dim))
            for _ in range(self.latent_dim)
        )
        self._encoder_bias = tuple(rng.uniform(-0.02, 0.02) for _ in range(self.latent_dim))
        self._transition = tuple(
            tuple(rng.uniform(-transition_scale, transition_scale) for _ in range(self.latent_dim))
            for _ in range(self.latent_dim)
        )
        normalizers = []
        for input_index in range(self.input_dim):
            norm = sum(
                self._encoder[latent_index][input_index] ** 2
                for latent_index in range(self.latent_dim)
            )
            normalizers.append(max(norm, 1e-9))
        self._decoder = tuple(
            tuple(
                self._encoder[latent_index][input_index] / normalizers[input_index]
                for latent_index in range(self.latent_dim)
            )
            for input_index in range(self.input_dim)
        )

    @staticmethod
    def _dot(weights: Sequence[float], values: Sequence[float]) -> float:
        return sum(weight * value for weight, value in zip(weights, values))

    def encode(self, neighbourhood: Sequence[float]) -> Tuple[float, ...]:
        if len(neighbourhood) != self.input_dim:
            raise ValueError("neighbourhood must contain exactly 7 values")
        return tuple(
            math.tanh(self._dot(row, neighbourhood) + bias)
            for row, bias in zip(self._encoder, self._encoder_bias)
        )

    def evolve(self, latent: Sequence[float], omega: float = 0.0) -> Tuple[float, ...]:
        if len(latent) != self.latent_dim:
            raise ValueError("latent state has the wrong dimension")
        bounded_omega = math.tanh(float(omega))
        return tuple(
            math.tanh(0.72 * latent[index] + self._dot(row, latent) + 0.08 * bounded_omega)
            for index, row in enumerate(self._transition)
        )

    def decode(self, latent: Sequence[float]) -> Tuple[float, ...]:
        if len(latent) != self.latent_dim:
            raise ValueError("latent state has the wrong dimension")
        return tuple(math.tanh(self._dot(row, latent)) for row in self._decoder)


class Sparse3DAutomaton:
    """Transactional sparse simulation of the 10**9000-cell universe."""

    _OFFSETS = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )

    def __init__(
        self,
        seed: int = 1337,
        latent_dim: int = 8,
        mechanics: Optional[Mechanics] = None,
    ) -> None:
        self.seed = int(seed)
        self.mechanics = mechanics or Mechanics()
        self.mechanics.validate()
        self.autoencoder = DeterministicAutoencoder(latent_dim=latent_dim, seed=self.seed)
        self._cells: Dict[Coordinate3D, CellState] = {}
        self.cycle = 0
        self.journal_hash = "0" * 64
        self.last_metrics = StepMetrics(
            cycle=0,
            materialised_cells=0,
            active_cells=0,
            frontier_cells=0,
            reconstruction_mse=0.0,
            energy=0.0,
            committed=True,
            journal_hash=self.journal_hash,
        )

    @property
    def materialised_cells(self) -> int:
        return len(self._cells)

    @property
    def cells(self) -> Mapping[Coordinate3D, CellState]:
        return dict(self._cells)

    @staticmethod
    def universe_descriptor() -> Dict[str, object]:
        return {
            "radix": RADIX,
            "address_depth": ADDRESS_DEPTH,
            "axis_size": "1000^1000",
            "axis_power_of_ten": 3000,
            "axis_decimal_digits": 3001,
            "virtual_cells": "10^9000",
            "virtual_cell_exponent": VIRTUAL_CELL_EXPONENT,
            "storage_model": "procedural-universe/sparse-active-frontier",
        }

    def _coordinate_bytes(self, coordinate: Coordinate3D) -> bytes:
        byte_width = (AXIS_SIZE.bit_length() + 7) // 8
        return b"".join(
            value.to_bytes(byte_width, byteorder="big", signed=False)
            for value in (coordinate.x, coordinate.y, coordinate.z)
        )

    def procedural_value(self, coordinate: Coordinate3D) -> float:
        key = int(self.seed & ((1 << 128) - 1)).to_bytes(16, "big", signed=False)
        digest = hashlib.blake2b(
            self._coordinate_bytes(coordinate), key=key, digest_size=8
        ).digest()
        raw = int.from_bytes(digest, byteorder="big", signed=False)
        unit = raw / float((1 << 64) - 1)
        return (unit - 0.5) * 0.02

    def sample(self, coordinate: Coordinate3D) -> float:
        state = self._cells.get(coordinate)
        return state.value if state is not None else self.procedural_value(coordinate)

    def inject(self, coordinate: Coordinate3D, value: float) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("injected value must be finite")
        current = self._cells.get(coordinate)
        if current is None:
            current = CellState(value=self.procedural_value(coordinate))
        proposal = current.value + value
        proposal = max(-self.mechanics.max_abs_value, min(self.mechanics.max_abs_value, proposal))
        self._cells[coordinate] = CellState(
            value=proposal,
            omega=current.omega,
            inactive_steps=0,
            revision=current.revision + 1,
        )

    def active_coordinates(self) -> Tuple[Coordinate3D, ...]:
        threshold = self.mechanics.activation_threshold
        return tuple(
            sorted(
                coordinate
                for coordinate, state in self._cells.items()
                if abs(state.value) + abs(state.omega) >= threshold
            )
        )

    def _neighbours(self, coordinate: Coordinate3D) -> Tuple[Coordinate3D, ...]:
        return tuple(coordinate.offset(*offset) for offset in self._OFFSETS)

    def _value_from(
        self, cells: Mapping[Coordinate3D, CellState], coordinate: Coordinate3D
    ) -> float:
        state = cells.get(coordinate)
        return state.value if state is not None else self.procedural_value(coordinate)

    def _frontier(
        self, cells: Mapping[Coordinate3D, CellState]
    ) -> Tuple[Coordinate3D, ...]:
        threshold = self.mechanics.activation_threshold
        active = {
            coordinate
            for coordinate, state in cells.items()
            if abs(state.value) + abs(state.omega) >= threshold or state.inactive_steps == 0
        }
        frontier = set(active)
        for coordinate in active:
            frontier.update(self._neighbours(coordinate))
        if len(frontier) <= self.mechanics.max_active_cells:
            return tuple(sorted(frontier))

        def score(coordinate: Coordinate3D) -> Tuple[float, Coordinate3D]:
            state = cells.get(coordinate)
            magnitude = (
                abs(state.value) + abs(state.omega)
                if state is not None
                else abs(self.procedural_value(coordinate))
            )
            return (-magnitude, coordinate)

        selected = sorted(frontier, key=score)[: self.mechanics.max_active_cells]
        return tuple(sorted(selected))

    def _cap_candidate(
        self, candidate: Mapping[Coordinate3D, CellState]
    ) -> Dict[Coordinate3D, CellState]:
        if len(candidate) <= self.mechanics.max_active_cells:
            return dict(candidate)
        ranked = sorted(
            candidate.items(),
            key=lambda item: (-(abs(item[1].value) + abs(item[1].omega)), item[0]),
        )
        return dict(ranked[: self.mechanics.max_active_cells])

    def _verify(
        self, candidate: Mapping[Coordinate3D, CellState]
    ) -> Tuple[bool, Optional[str], float]:
        if len(candidate) > self.mechanics.max_active_cells:
            return False, "active-cell budget exceeded", math.inf
        energy = 0.0
        for coordinate, state in candidate.items():
            if not isinstance(coordinate, Coordinate3D):
                return False, "invalid coordinate type", math.inf
            if not math.isfinite(state.value) or not math.isfinite(state.omega):
                return False, "non-finite state", math.inf
            if abs(state.value) > self.mechanics.max_abs_value + 1e-12:
                return False, "value bound exceeded", math.inf
            energy += state.value * state.value + state.omega * state.omega
        if not math.isfinite(energy) or energy > self.mechanics.max_energy:
            return False, "energy budget exceeded", energy
        return True, None, energy

    def _journal(
        self, cells: Mapping[Coordinate3D, CellState], cycle: int
    ) -> str:
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(self.journal_hash))
        digest.update(struct.pack(">Q", cycle))
        mechanics_payload = json.dumps(
            asdict(self.mechanics), sort_keys=True, separators=(",", ":")
        )
        digest.update(mechanics_payload.encode("utf-8"))
        for coordinate in sorted(cells):
            digest.update(self._coordinate_bytes(coordinate))
            state = cells[coordinate]
            digest.update(
                struct.pack(
                    ">ddII",
                    state.value,
                    state.omega,
                    state.inactive_steps,
                    state.revision,
                )
            )
        return digest.hexdigest()

    def step(
        self,
        injections: Optional[Mapping[Coordinate3D, float]] = None,
    ) -> StepMetrics:
        before = dict(self._cells)
        working = dict(before)
        if injections:
            for coordinate in sorted(injections):
                value = float(injections[coordinate])
                if not math.isfinite(value):
                    return self._rollback_metrics("non-finite injection", 0, 0.0)
                current = working.get(coordinate)
                if current is None:
                    current = CellState(value=self.procedural_value(coordinate))
                injected = max(
                    -self.mechanics.max_abs_value,
                    min(self.mechanics.max_abs_value, current.value + value),
                )
                working[coordinate] = CellState(
                    value=injected,
                    omega=current.omega,
                    inactive_steps=0,
                    revision=current.revision + 1,
                )

        frontier = self._frontier(working)
        proposed_updates: Dict[Coordinate3D, CellState] = {}
        squared_error = 0.0
        for coordinate in frontier:
            centre_state = working.get(coordinate)
            centre_value = self._value_from(working, coordinate)
            centre_omega = centre_state.omega if centre_state is not None else 0.0
            neighbours = self._neighbours(coordinate)
            neighbour_values = tuple(
                self._value_from(working, neighbour) for neighbour in neighbours
            )
            neighbourhood = (centre_value,) + neighbour_values
            latent = self.autoencoder.encode(neighbourhood)
            predicted_latent = self.autoencoder.evolve(latent, centre_omega)
            reconstruction = self.autoencoder.decode(predicted_latent)
            residual = reconstruction[0] - centre_value
            squared_error += residual * residual
            laplacian = sum(value - centre_value for value in neighbour_values)
            omega_next = (
                self.mechanics.omega_retention * centre_omega
                - self.mechanics.omega_rate * residual
            )
            proposal = centre_value + self.mechanics.time_step * (
                self.mechanics.diffusion * laplacian
                - self.mechanics.error_gain * residual
                + omega_next
            )
            proposal = max(
                -self.mechanics.max_abs_value,
                min(self.mechanics.max_abs_value, proposal),
            )
            activity = abs(proposal) + abs(omega_next)
            inactive_steps = (
                0
                if activity >= self.mechanics.activation_threshold
                else ((centre_state.inactive_steps + 1) if centre_state is not None else 1)
            )
            proposed_updates[coordinate] = CellState(
                value=proposal,
                omega=omega_next,
                inactive_steps=inactive_steps,
                revision=(centre_state.revision if centre_state is not None else 0) + 1,
            )

        candidate: Dict[Coordinate3D, CellState] = {}
        frontier_set = set(frontier)
        for coordinate, state in working.items():
            if coordinate in frontier_set:
                continue
            aged = replace(state, inactive_steps=state.inactive_steps + 1)
            baseline = self.procedural_value(coordinate)
            should_prune = (
                aged.inactive_steps >= self.mechanics.prune_after
                and abs(aged.value - baseline) < self.mechanics.activation_threshold
                and abs(aged.omega) < self.mechanics.activation_threshold
            )
            if not should_prune:
                candidate[coordinate] = aged
        for coordinate, state in proposed_updates.items():
            baseline = self.procedural_value(coordinate)
            should_prune = (
                state.inactive_steps >= self.mechanics.prune_after
                and abs(state.value - baseline) < self.mechanics.activation_threshold
                and abs(state.omega) < self.mechanics.activation_threshold
            )
            if not should_prune:
                candidate[coordinate] = state

        candidate = self._cap_candidate(candidate)
        valid, reason, energy = self._verify(candidate)
        mse = squared_error / float(max(1, len(frontier)))
        if not valid:
            self._cells = before
            return self._rollback_metrics(
                reason or "verification failed", len(frontier), mse, energy
            )

        self._cells = candidate
        self.cycle += 1
        self.journal_hash = self._journal(candidate, self.cycle)
        active_count = len(self.active_coordinates())
        self.last_metrics = StepMetrics(
            cycle=self.cycle,
            materialised_cells=len(candidate),
            active_cells=active_count,
            frontier_cells=len(frontier),
            reconstruction_mse=mse,
            energy=energy,
            committed=True,
            journal_hash=self.journal_hash,
        )
        return self.last_metrics

    def _rollback_metrics(
        self,
        reason: str,
        frontier_cells: int,
        mse: float,
        energy: float = 0.0,
    ) -> StepMetrics:
        self.last_metrics = StepMetrics(
            cycle=self.cycle,
            materialised_cells=len(self._cells),
            active_cells=len(self.active_coordinates()),
            frontier_cells=frontier_cells,
            reconstruction_mse=mse,
            energy=energy,
            committed=False,
            journal_hash=self.journal_hash,
            rollback_reason=reason,
        )
        return self.last_metrics

    def fork(self, mechanics: Optional[Mechanics] = None) -> "Sparse3DAutomaton":
        clone = Sparse3DAutomaton(
            seed=self.seed,
            latent_dim=self.autoencoder.latent_dim,
            mechanics=mechanics or self.mechanics,
        )
        clone._cells = dict(self._cells)
        clone.cycle = self.cycle
        clone.journal_hash = self.journal_hash
        clone.last_metrics = self.last_metrics
        return clone

    def state_digest(self) -> str:
        return self._journal(self._cells, self.cycle)

    def snapshot(self) -> Dict[str, object]:
        active = self.active_coordinates()
        return {
            "universe": self.universe_descriptor(),
            "cycle": self.cycle,
            "materialised_cells": len(self._cells),
            "active_cells": len(active),
            "journal_hash": self.journal_hash,
            "mechanics": asdict(self.mechanics),
            "last_metrics": self.last_metrics.to_dict(),
        }


class BoundedMechanicsOptimizer:
    """Shadow-test declared mechanics and adopt only a better valid result."""

    def __init__(self, active_cell_penalty: float = 1e-5) -> None:
        if active_cell_penalty < 0.0:
            raise ValueError("active_cell_penalty must be non-negative")
        self.active_cell_penalty = float(active_cell_penalty)

    def _score(
        self,
        engine: Sparse3DAutomaton,
        workload: Sequence[Mapping[Coordinate3D, float]],
    ) -> float:
        score = 0.0
        for injections in workload:
            metrics = engine.step(injections)
            if not metrics.committed:
                return math.inf
            score += metrics.reconstruction_mse
            score += self.active_cell_penalty * metrics.materialised_cells
        return score

    def optimize(
        self,
        engine: Sparse3DAutomaton,
        workload: Sequence[Mapping[Coordinate3D, float]],
        candidates: Optional[Iterable[Mechanics]] = None,
    ) -> OptimizationResult:
        if not workload:
            raise ValueError("workload must contain at least one transaction")
        previous = engine.mechanics
        baseline_score = self._score(engine.fork(previous), workload)
        if candidates is None:
            candidates = (
                replace(previous, diffusion=max(0.0, previous.diffusion - 0.02)),
                replace(previous, diffusion=min(1.0, previous.diffusion + 0.02)),
                replace(previous, error_gain=max(0.0, previous.error_gain - 0.10)),
                replace(previous, error_gain=min(4.0, previous.error_gain + 0.10)),
                replace(
                    previous,
                    activation_threshold=min(
                        previous.max_abs_value,
                        previous.activation_threshold * 1.25,
                    ),
                ),
            )
        best_score = baseline_score
        best = previous
        for candidate in candidates:
            candidate.validate()
            score = self._score(engine.fork(candidate), workload)
            if score + 1e-15 < best_score:
                best_score = score
                best = candidate
        adopted = best != previous
        if adopted:
            engine.mechanics = best
        return OptimizationResult(
            adopted=adopted,
            baseline_score=baseline_score,
            candidate_score=best_score,
            previous_mechanics=previous,
            selected_mechanics=best,
        )


def make_echo_injections(
    side: int = 3,
    amplitude: float = 1.0,
    origin: Optional[Coordinate3D] = None,
) -> Dict[Coordinate3D, float]:
    if side < 1 or side > 21 or side % 2 == 0:
        raise ValueError("side must be an odd integer in [1, 21]")
    if not math.isfinite(amplitude):
        raise ValueError("amplitude must be finite")
    origin = origin or Coordinate3D(0, 0, 0)
    radius = side // 2
    injections: Dict[Coordinate3D, float] = {}
    for dz in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                distance_squared = dx * dx + dy * dy + dz * dz
                value = amplitude * math.exp(-0.65 * distance_squared)
                injections[origin.offset(dx, dy, dz)] = value
    return injections
