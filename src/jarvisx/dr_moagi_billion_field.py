"""Sparse operational runtime for the Dr Moagi billion-instance 3D field.

The default logical lattice contains ``1000 ** 3`` addressable coordinates. The runtime never
allocates a dense billion-cell tensor; it materializes only the configured active support and,
optionally, a bounded neighbour halo. Missing coordinates are deterministic zero background.

The synchronous transaction is:

    snapshot -> support closure -> encode -> Q3 -> reason/couple/control
    -> residual -> omega -> decode -> validate -> commit/rollback -> journal

This is a bounded mathematical reference, not a claim that one billion proprietary neural models
are instantiated.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple, cast

Coordinate = Tuple[int, int, int]


@dataclass(frozen=True)
class BillionFieldConfig:
    """Numerical, geometric, and execution constraints for one sparse field."""

    side: int = 1000
    block_side: int = 32
    encoder_gain: float = 1.0
    context_gain: float = 0.25
    reasoning_gain: float = 0.50
    coupling_gain: float = 0.08
    consensus_gain: float = 0.0
    omega_decay: float = 0.875
    omega_gain: float = 0.0625
    reasoning_steps: int = 3
    residual_threshold: float = 1.50
    value_min: float = -1.0
    value_max: float = 1.0
    latent_min: int = -4
    latent_max: int = 3
    max_active_cells: int = 100_000
    halo_depth: int = 0
    prune_epsilon: float = 0.0

    def __post_init__(self) -> None:
        self._require_positive_int(self.side, "side")
        self._require_positive_int(self.block_side, "block_side")
        self._require_positive_int(self.reasoning_steps, "reasoning_steps")
        self._require_positive_int(self.max_active_cells, "max_active_cells")
        self._require_non_negative_int(self.halo_depth, "halo_depth")

        if isinstance(self.latent_min, bool) or not isinstance(self.latent_min, int):
            raise TypeError("latent_min must be an integer")
        if isinstance(self.latent_max, bool) or not isinstance(self.latent_max, int):
            raise TypeError("latent_max must be an integer")
        if (self.latent_min, self.latent_max) != (-4, 3):
            raise ValueError("the canonical signed Q3 range is exactly [-4, 3]")

        for name in (
            "encoder_gain",
            "context_gain",
            "reasoning_gain",
            "coupling_gain",
            "consensus_gain",
            "omega_decay",
            "omega_gain",
            "residual_threshold",
            "value_min",
            "value_max",
            "prune_epsilon",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.value_min != -1.0 or self.value_max != 1.0:
            raise ValueError("the canonical field range is exactly [-1, 1]")
        if self.encoder_gain <= 0.0:
            raise ValueError("encoder_gain must be positive")
        if self.context_gain < 0.0:
            raise ValueError("context_gain must be non-negative")
        if not 0.0 <= self.reasoning_gain <= 1.0:
            raise ValueError("reasoning_gain must be in [0, 1]")
        if self.coupling_gain < 0.0:
            raise ValueError("coupling_gain must be non-negative")
        if self.consensus_gain < 0.0:
            raise ValueError("consensus_gain must be non-negative")
        if not 0.0 <= self.omega_decay <= 1.0:
            raise ValueError("omega_decay must be in [0, 1]")
        if self.omega_gain < 0.0:
            raise ValueError("omega_gain must be non-negative")
        if self.residual_threshold < 0.0:
            raise ValueError("residual_threshold must be non-negative")
        if self.prune_epsilon < 0.0:
            raise ValueError("prune_epsilon must be non-negative")

    @staticmethod
    def _require_positive_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    @staticmethod
    def _require_non_negative_int(value: int, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class CellState:
    """Committed persistent state plus diagnostics from the latest attempted cycle."""

    observed: float = 0.0
    latent: int = 0
    decoded: float = 0.0
    predicted: float = 0.0
    residual: float = 0.0
    omega: float = 0.0
    committed: float = 0.0
    valid: bool = True


@dataclass(frozen=True)
class FieldMetrics:
    """Deterministic aggregate measurements for one completed cycle."""

    cycle: int
    virtual_cells: int
    active_cells: int
    active_ratio: float
    mean_absolute_residual: float
    reconstruction_loss: float
    coherence: float
    valid_cells: int
    rejected_cells: int
    journal_digest: str
    state_digest: str


_ZERO_STATE = CellState()
_CHECKPOINT_VERSION = 1


class SparseBillionField:
    """Synchronous sparse implementation of the billion-address Dr Moagi field.

    Every update reads a frozen snapshot and writes a new mapping. Canonical address ordering is
    used for support construction, metrics, serialization, and hashing, so results do not depend
    on dictionary insertion order.
    """

    _NEIGHBOUR_OFFSETS: Tuple[Coordinate, ...] = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )

    def __init__(self, config: BillionFieldConfig | None = None) -> None:
        self.config = config or BillionFieldConfig()
        self._cells: Dict[Coordinate, CellState] = {}
        self._cycle = 0
        self._journal_digest = self._initial_journal_digest()

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def virtual_cell_count(self) -> int:
        return self.config.side**3

    @property
    def active_cell_count(self) -> int:
        return len(self._cells)

    @property
    def padded_block_count(self) -> int:
        blocks_per_axis = math.ceil(self.config.side / self.config.block_side)
        return blocks_per_axis**3

    def estimate_dense_state_bytes(self, bytes_per_cell: int = 32) -> int:
        if isinstance(bytes_per_cell, bool) or not isinstance(bytes_per_cell, int):
            raise TypeError("bytes_per_cell must be an integer")
        if bytes_per_cell <= 0:
            raise ValueError("bytes_per_cell must be positive")
        return self.virtual_cell_count * bytes_per_cell

    def address(self, coordinate: Coordinate) -> int:
        x, y, z = self._validate_coordinate(coordinate)
        side = self.config.side
        return x + side * (y + side * z)

    def coordinate(self, address: int) -> Coordinate:
        if isinstance(address, bool) or not isinstance(address, int):
            raise TypeError("address must be an integer")
        if address < 0 or address >= self.virtual_cell_count:
            raise ValueError("address is outside the virtual lattice")
        side = self.config.side
        return address % side, (address // side) % side, address // (side * side)

    def block_address(self, coordinate: Coordinate) -> Coordinate:
        x, y, z = self._validate_coordinate(coordinate)
        block = self.config.block_side
        return x // block, y // block, z // block

    def activate(self, coordinate: Coordinate, observed: float = 0.0) -> CellState:
        coordinate = self._validate_coordinate(coordinate)
        if coordinate not in self._cells and len(self._cells) >= self.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")
        value = self._clip_value(self._require_finite(observed, "observed"))
        previous = self._cells.get(coordinate, _ZERO_STATE)
        state = CellState(
            observed=value,
            latent=previous.latent,
            decoded=previous.decoded,
            predicted=previous.predicted,
            residual=previous.residual,
            omega=previous.omega,
            committed=previous.committed,
            valid=previous.valid,
        )
        self._cells[coordinate] = state
        return state

    def deactivate(self, coordinate: Coordinate) -> None:
        self._cells.pop(self._validate_coordinate(coordinate), None)

    def state(self, coordinate: Coordinate) -> CellState:
        return self._cells.get(self._validate_coordinate(coordinate), _ZERO_STATE)

    def active_coordinates(self) -> Tuple[Coordinate, ...]:
        return tuple(sorted(self._cells, key=self.address))

    def iter_active(self) -> Iterator[Tuple[Coordinate, CellState]]:
        for coordinate in self.active_coordinates():
            yield coordinate, self._cells[coordinate]

    def encoded_snapshot(self) -> Dict[Coordinate, int]:
        return {coordinate: state.latent for coordinate, state in self.iter_active()}

    def decoded_value(self, coordinate: Coordinate) -> float:
        return self.state(coordinate).committed

    def step(
        self,
        observations: Mapping[Coordinate, float] | None = None,
        controls: Mapping[Coordinate, float] | None = None,
    ) -> FieldMetrics:
        """Execute one synchronous transaction.

        New observation or control coordinates become active. Existing observations persist when
        omitted. ``controls`` are transient external inputs applied only during this cycle.
        ``halo_depth`` expands the computational support by the requested bounded neighbour rings.
        """

        normalized_inputs = self._normalize_values(observations or {}, "observation")
        normalized_controls = self._normalize_values(controls or {}, "control")
        snapshot = dict(self._cells)
        seed_support = set(snapshot).union(normalized_inputs).union(normalized_controls)
        active = self._support_closure(seed_support, self.config.halo_depth)
        if len(active) > self.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")

        observed = {
            coordinate: normalized_inputs.get(
                coordinate, snapshot.get(coordinate, _ZERO_STATE).observed
            )
            for coordinate in active
        }
        control = {coordinate: normalized_controls.get(coordinate, 0.0) for coordinate in active}

        latents: Dict[Coordinate, int] = {}
        decoded: Dict[Coordinate, float] = {}
        for coordinate in active:
            context = self._neighbour_mean(observed, coordinate)
            activation = self.config.encoder_gain * (
                observed[coordinate] + self.config.context_gain * context
            )
            latent = self._quantize_q3(activation)
            latents[coordinate] = latent
            decoded[coordinate] = self._decode_q3(latent)

        staged: Dict[Coordinate, CellState] = {}
        for coordinate in active:
            previous = snapshot.get(coordinate, _ZERO_STATE)
            laplacian = self._laplacian(snapshot, coordinate)
            consensus = self._latent_consensus(latents, coordinate)

            prediction = previous.committed
            for _ in range(self.config.reasoning_steps):
                prediction += self.config.reasoning_gain * (decoded[coordinate] - prediction)
                prediction += self.config.coupling_gain * laplacian
                prediction += self.config.consensus_gain * consensus
                prediction += control[coordinate]
                prediction = self._clip_value(prediction)

            pre_correction_residual = observed[coordinate] - prediction
            omega_candidate = self._clip_value(
                self.config.omega_decay * previous.omega
                + self.config.omega_gain * pre_correction_residual
            )
            candidate = self._clip_value(prediction + omega_candidate)
            residual = observed[coordinate] - candidate
            valid = (
                math.isfinite(candidate)
                and math.isfinite(residual)
                and self.config.latent_min <= latents[coordinate] <= self.config.latent_max
                and abs(residual) <= self.config.residual_threshold
            )

            # Only persistent numeric state crosses the transaction boundary. Rejected candidate
            # diagnostics remain visible, but committed and omega both roll back atomically.
            committed = candidate if valid else previous.committed
            omega = omega_candidate if valid else previous.omega
            staged[coordinate] = CellState(
                observed=observed[coordinate],
                latent=latents[coordinate],
                decoded=decoded[coordinate],
                predicted=prediction,
                residual=residual,
                omega=omega,
                committed=committed,
                valid=valid,
            )

        staged = self._prune(staged, protected=set(normalized_inputs).union(normalized_controls))
        next_cycle = self._cycle + 1
        next_digest = self._calculate_journal_digest(next_cycle, staged)
        self._cells = staged
        self._cycle = next_cycle
        self._journal_digest = next_digest
        return self.metrics()

    def run(
        self,
        cycles: int,
        observations: Mapping[Coordinate, float] | None = None,
        controls: Mapping[Coordinate, float] | None = None,
    ) -> FieldMetrics:
        if isinstance(cycles, bool) or not isinstance(cycles, int):
            raise TypeError("cycles must be an integer")
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        metrics = self.metrics()
        for cycle_index in range(cycles):
            metrics = self.step(
                observations if cycle_index == 0 else None,
                controls if cycle_index == 0 else None,
            )
        return metrics

    def metrics(self) -> FieldMetrics:
        active_cells = len(self._cells)
        if active_cells:
            residuals = [state.residual for state in self._cells.values()]
            mean_absolute_residual = sum(abs(value) for value in residuals) / active_cells
            reconstruction_loss = sum(value * value for value in residuals) / active_cells
            valid_cells = sum(1 for state in self._cells.values() if state.valid)
        else:
            mean_absolute_residual = 0.0
            reconstruction_loss = 0.0
            valid_cells = 0

        coherence = 1.0 / (1.0 + mean_absolute_residual)
        return FieldMetrics(
            cycle=self._cycle,
            virtual_cells=self.virtual_cell_count,
            active_cells=active_cells,
            active_ratio=active_cells / self.virtual_cell_count,
            mean_absolute_residual=mean_absolute_residual,
            reconstruction_loss=reconstruction_loss,
            coherence=coherence,
            valid_cells=valid_cells,
            rejected_cells=active_cells - valid_cells,
            journal_digest=self._journal_digest,
            state_digest=self.state_digest(),
        )

    def state_digest(self) -> str:
        """Return a canonical SHA-256 digest of config, cycle, journal, and sparse state."""

        digest = hashlib.sha256()
        digest.update(b"jarvisx-dr-moagi-state-v1\0")
        digest.update(self._config_bytes())
        digest.update(struct.pack(">Q", self._cycle))
        digest.update(bytes.fromhex(self._journal_digest))
        self._update_digest_with_states(digest, self._cells)
        return digest.hexdigest()

    def checkpoint(self) -> Dict[str, object]:
        """Return a JSON-serializable checkpoint with an independently verifiable state digest."""

        return {
            "version": _CHECKPOINT_VERSION,
            "config": asdict(self.config),
            "cycle": self._cycle,
            "journal_digest": self._journal_digest,
            "state_digest": self.state_digest(),
            "cells": [
                {
                    "coordinate": list(coordinate),
                    "state": asdict(state),
                }
                for coordinate, state in self.iter_active()
            ],
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, object]) -> "SparseBillionField":
        """Restore a checkpoint and reject malformed or tampered state."""

        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint must be a mapping")
        version = checkpoint.get("version")
        if isinstance(version, bool) or version != _CHECKPOINT_VERSION:
            raise ValueError("unsupported checkpoint version")

        config_data = checkpoint.get("config")
        if not isinstance(config_data, Mapping):
            raise TypeError("checkpoint config must be a mapping")
        field = cls(BillionFieldConfig(**cast(Any, dict(config_data))))

        cycle = checkpoint.get("cycle")
        if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
            raise ValueError("checkpoint cycle must be a non-negative integer")
        journal_digest = checkpoint.get("journal_digest")
        state_digest = checkpoint.get("state_digest")
        if not cls._is_sha256_hex(journal_digest):
            raise ValueError("checkpoint journal_digest must be a SHA-256 hex digest")
        if not cls._is_sha256_hex(state_digest):
            raise ValueError("checkpoint state_digest must be a SHA-256 hex digest")

        cells_data = checkpoint.get("cells")
        if not isinstance(cells_data, list):
            raise TypeError("checkpoint cells must be a list")
        if len(cells_data) > field.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")

        restored: Dict[Coordinate, CellState] = {}
        for item in cells_data:
            if not isinstance(item, Mapping):
                raise TypeError("checkpoint cell entry must be a mapping")
            raw_coordinate = item.get("coordinate")
            if not isinstance(raw_coordinate, list) or len(raw_coordinate) != 3:
                raise TypeError("checkpoint coordinate must be a three-integer list")
            coordinate = field._validate_coordinate(cast(Coordinate, tuple(raw_coordinate)))
            if coordinate in restored:
                raise ValueError("checkpoint contains duplicate coordinates")
            raw_state = item.get("state")
            if not isinstance(raw_state, Mapping):
                raise TypeError("checkpoint state must be a mapping")
            state = field._validated_cell_state(raw_state)
            restored[coordinate] = state

        field._cells = restored
        field._cycle = cycle
        field._journal_digest = str(journal_digest)
        if field.state_digest() != state_digest:
            raise ValueError("checkpoint state digest mismatch")
        return field

    def _normalize_values(
        self, values: Mapping[Coordinate, float], name: str
    ) -> Dict[Coordinate, float]:
        if not isinstance(values, Mapping):
            raise TypeError(f"{name}s must be a mapping")
        normalized: Dict[Coordinate, float] = {}
        for coordinate, value in values.items():
            checked_coordinate = self._validate_coordinate(coordinate)
            checked_value = self._clip_value(self._require_finite(value, name))
            normalized[checked_coordinate] = checked_value
        return normalized

    def _validate_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if not isinstance(coordinate, tuple) or len(coordinate) != 3:
            raise TypeError("coordinate must be a three-integer tuple")
        if not all(
            isinstance(component, int) and not isinstance(component, bool)
            for component in coordinate
        ):
            raise TypeError("coordinate components must be integers")
        if not all(0 <= component < self.config.side for component in coordinate):
            raise ValueError("coordinate is outside the virtual lattice")
        return coordinate

    def _neighbours(self, coordinate: Coordinate) -> Iterable[Coordinate]:
        x, y, z = coordinate
        side = self.config.side
        for dx, dy, dz in self._NEIGHBOUR_OFFSETS:
            neighbour = x + dx, y + dy, z + dz
            if all(0 <= component < side for component in neighbour):
                yield neighbour

    def _support_closure(
        self, coordinates: Iterable[Coordinate], depth: int
    ) -> Tuple[Coordinate, ...]:
        support = set(coordinates)
        frontier = set(support)
        for _ in range(depth):
            expanded = set(frontier)
            for coordinate in frontier:
                expanded.update(self._neighbours(coordinate))
            frontier = expanded - support
            support.update(expanded)
            if len(support) > self.config.max_active_cells:
                raise RuntimeError("active-cell budget exceeded")
        return tuple(sorted(support, key=self.address))

    def _neighbour_mean(
        self, observed: Mapping[Coordinate, float], coordinate: Coordinate
    ) -> float:
        neighbours = tuple(self._neighbours(coordinate))
        if not neighbours:
            return 0.0
        return sum(observed.get(neighbour, 0.0) for neighbour in neighbours) / len(neighbours)

    def _laplacian(
        self, snapshot: Mapping[Coordinate, CellState], coordinate: Coordinate
    ) -> float:
        center = snapshot.get(coordinate, _ZERO_STATE).committed
        return sum(
            snapshot.get(neighbour, _ZERO_STATE).committed - center
            for neighbour in self._neighbours(coordinate)
        )

    def _latent_consensus(
        self, latents: Mapping[Coordinate, int], coordinate: Coordinate
    ) -> float:
        neighbours = tuple(self._neighbours(coordinate))
        if not neighbours:
            return 0.0
        center = latents[coordinate]
        return sum(latents.get(neighbour, 0) - center for neighbour in neighbours) / len(neighbours)

    def _quantize_q3(self, value: float) -> int:
        # Piecewise scaling uses every signed 3-bit code over the canonical [-1, 1] field.
        scaled = 4.0 * value if value < 0.0 else 3.0 * value
        quantized = self._round_half_away_from_zero(scaled)
        return max(self.config.latent_min, min(self.config.latent_max, quantized))

    def _decode_q3(self, latent: int) -> float:
        if latent < self.config.latent_min or latent > self.config.latent_max:
            raise ValueError("latent is outside the canonical signed Q3 range")
        return latent / 4.0 if latent < 0 else latent / 3.0

    def _clip_value(self, value: float) -> float:
        return max(self.config.value_min, min(self.config.value_max, value))

    def _prune(
        self, states: Dict[Coordinate, CellState], protected: set[Coordinate]
    ) -> Dict[Coordinate, CellState]:
        epsilon = self.config.prune_epsilon
        if epsilon <= 0.0:
            return states
        return {
            coordinate: state
            for coordinate, state in states.items()
            if coordinate in protected
            or not state.valid
            or max(
                abs(state.observed),
                abs(state.omega),
                abs(state.committed),
                abs(state.residual),
            )
            > epsilon
        }

    @staticmethod
    def _round_half_away_from_zero(value: float) -> int:
        return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)

    @staticmethod
    def _require_finite(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _initial_journal_digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"jarvisx-dr-moagi-field-v2\0")
        digest.update(self._config_bytes())
        return digest.hexdigest()

    def _config_bytes(self) -> bytes:
        return json.dumps(
            asdict(self.config), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    def _calculate_journal_digest(
        self, cycle: int, states: Mapping[Coordinate, CellState]
    ) -> str:
        digest = hashlib.sha256()
        digest.update(b"jarvisx-dr-moagi-journal-v2\0")
        digest.update(bytes.fromhex(self._journal_digest))
        digest.update(self._config_bytes())
        digest.update(struct.pack(">Q", cycle))
        self._update_digest_with_states(digest, states)
        return digest.hexdigest()

    def _update_digest_with_states(
        self, digest: Any, states: Mapping[Coordinate, CellState]
    ) -> None:
        for coordinate in sorted(states, key=self.address):
            state = states[coordinate]
            digest.update(struct.pack(">Q", self.address(coordinate)))
            digest.update(struct.pack(">d", state.observed))
            digest.update(struct.pack(">b", state.latent))
            digest.update(struct.pack(">d", state.decoded))
            digest.update(struct.pack(">d", state.predicted))
            digest.update(struct.pack(">d", state.residual))
            digest.update(struct.pack(">d", state.omega))
            digest.update(struct.pack(">d", state.committed))
            digest.update(b"\x01" if state.valid else b"\x00")

    def _validated_cell_state(self, raw: Mapping[str, object]) -> CellState:
        expected = {
            "observed",
            "latent",
            "decoded",
            "predicted",
            "residual",
            "omega",
            "committed",
            "valid",
        }
        if set(raw) != expected:
            raise ValueError("checkpoint cell state has unexpected fields")
        latent = raw["latent"]
        valid = raw["valid"]
        if isinstance(latent, bool) or not isinstance(latent, int):
            raise TypeError("checkpoint latent must be an integer")
        if not self.config.latent_min <= latent <= self.config.latent_max:
            raise ValueError("checkpoint latent is outside Q3 range")
        if not isinstance(valid, bool):
            raise TypeError("checkpoint valid must be boolean")

        numeric = {
            name: self._require_finite(raw[name], f"checkpoint {name}")
            for name in expected - {"latent", "valid"}
        }
        for name in ("observed", "decoded", "predicted", "omega", "committed"):
            if not self.config.value_min <= numeric[name] <= self.config.value_max:
                raise ValueError(f"checkpoint {name} is outside field bounds")
        return CellState(
            observed=numeric["observed"],
            latent=latent,
            decoded=numeric["decoded"],
            predicted=numeric["predicted"],
            residual=numeric["residual"],
            omega=numeric["omega"],
            committed=numeric["committed"],
            valid=valid,
        )

    @staticmethod
    def _is_sha256_hex(value: object) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            bytes.fromhex(value)
        except ValueError:
            return False
        return True
