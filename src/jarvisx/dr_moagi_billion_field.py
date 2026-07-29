"""Sparse reference runtime for the Dr Moagi billion-instance 3D field.

The logical lattice contains ``1000 ** 3`` addressable coordinates by default.  This module
never allocates a dense billion-cell tensor.  It materializes only active coordinates and
interprets absent coordinates as a deterministic zero background.

The synchronous update implements the operational sequence documented in
``docs/DR_MOAGI_3D_BILLION_INSTANCE_AUTOENCODER_EQUATION.md``:

    snapshot -> encode -> Q3 -> reason -> couple -> residual -> omega
    -> decode -> validate -> commit/rollback -> journal

It is a bounded mathematical reference, not a claim that one billion proprietary neural
models are instantiated.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Mapping, Tuple

Coordinate = Tuple[int, int, int]


@dataclass(frozen=True)
class BillionFieldConfig:
    """Numerical and execution constraints for one sparse virtual field."""

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

    def __post_init__(self) -> None:
        if self.side <= 0:
            raise ValueError("side must be positive")
        if self.block_side <= 0:
            raise ValueError("block_side must be positive")
        if self.reasoning_steps <= 0:
            raise ValueError("reasoning_steps must be positive")
        if self.max_active_cells <= 0:
            raise ValueError("max_active_cells must be positive")
        if self.value_min >= self.value_max:
            raise ValueError("value_min must be less than value_max")
        if self.latent_min >= self.latent_max:
            raise ValueError("latent_min must be less than latent_max")
        if not 0.0 <= self.reasoning_gain <= 1.0:
            raise ValueError("reasoning_gain must be in [0, 1]")
        if not 0.0 <= self.omega_decay <= 1.0:
            raise ValueError("omega_decay must be in [0, 1]")
        if self.context_gain < 0.0:
            raise ValueError("context_gain must be non-negative")
        if self.coupling_gain < 0.0:
            raise ValueError("coupling_gain must be non-negative")
        if self.consensus_gain < 0.0:
            raise ValueError("consensus_gain must be non-negative")
        if self.omega_gain < 0.0:
            raise ValueError("omega_gain must be non-negative")
        if self.residual_threshold < 0.0:
            raise ValueError("residual_threshold must be non-negative")


@dataclass(frozen=True)
class CellState:
    """Committed and diagnostic state for one active coordinate."""

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
    """Deterministic aggregate measurements for one committed cycle."""

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


_ZERO_STATE = CellState()


class SparseBillionField:
    """Synchronous sparse implementation of the billion-address Dr Moagi field.

    Missing cells are logical zero-valued cells.  Every update reads from a frozen snapshot and
    writes into a new mapping, so results do not depend on dictionary iteration order.
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
        self._journal_digest = hashlib.sha256(b"jarvisx-dr-moagi-field-v1").hexdigest()

    @property
    def cycle(self) -> int:
        """Return the number of committed synchronous cycles."""

        return self._cycle

    @property
    def virtual_cell_count(self) -> int:
        """Return the full logical address count without allocating it."""

        return self.config.side**3

    @property
    def active_cell_count(self) -> int:
        """Return the number of physically materialized coordinates."""

        return len(self._cells)

    @property
    def padded_block_count(self) -> int:
        """Return the number of blocks in the padded logical block grid."""

        blocks_per_axis = math.ceil(self.config.side / self.config.block_side)
        return blocks_per_axis**3

    def estimate_dense_state_bytes(self, bytes_per_cell: int = 32) -> int:
        """Estimate dense storage for a chosen per-cell representation."""

        if bytes_per_cell <= 0:
            raise ValueError("bytes_per_cell must be positive")
        return self.virtual_cell_count * bytes_per_cell

    def address(self, coordinate: Coordinate) -> int:
        """Map ``(x, y, z)`` to its canonical row-major scalar address."""

        x, y, z = self._validate_coordinate(coordinate)
        side = self.config.side
        return x + side * (y + side * z)

    def coordinate(self, address: int) -> Coordinate:
        """Invert the canonical row-major scalar address."""

        if not isinstance(address, int):
            raise TypeError("address must be an integer")
        if address < 0 or address >= self.virtual_cell_count:
            raise ValueError("address is outside the virtual lattice")
        side = self.config.side
        x = address % side
        y = (address // side) % side
        z = address // (side * side)
        return x, y, z

    def block_address(self, coordinate: Coordinate) -> Coordinate:
        """Return the padded sparse block coordinate that owns ``coordinate``."""

        x, y, z = self._validate_coordinate(coordinate)
        block = self.config.block_side
        return x // block, y // block, z // block

    def activate(self, coordinate: Coordinate, observed: float = 0.0) -> CellState:
        """Materialize or update one coordinate without advancing the field cycle."""

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
        """Remove one materialized coordinate, restoring logical zero background."""

        coordinate = self._validate_coordinate(coordinate)
        self._cells.pop(coordinate, None)

    def state(self, coordinate: Coordinate) -> CellState:
        """Return one state, or the immutable zero background for an inactive cell."""

        coordinate = self._validate_coordinate(coordinate)
        return self._cells.get(coordinate, _ZERO_STATE)

    def active_coordinates(self) -> Tuple[Coordinate, ...]:
        """Return active coordinates in canonical scalar-address order."""

        return tuple(sorted(self._cells, key=self.address))

    def iter_active(self) -> Iterator[Tuple[Coordinate, CellState]]:
        """Iterate over active states in canonical scalar-address order."""

        for coordinate in self.active_coordinates():
            yield coordinate, self._cells[coordinate]

    def encoded_snapshot(self) -> Dict[Coordinate, int]:
        """Return a copy of the currently committed signed 3-bit latent field."""

        return {coordinate: state.latent for coordinate, state in self.iter_active()}

    def decoded_value(self, coordinate: Coordinate) -> float:
        """Return the last committed reconstruction at ``coordinate``."""

        return self.state(coordinate).committed

    def step(self, observations: Mapping[Coordinate, float] | None = None) -> FieldMetrics:
        """Execute one complete synchronous encode/reason/decode/commit transaction.

        New coordinates supplied by ``observations`` become active.  Existing active coordinates
        retain their previous observation when not supplied in the current mapping.
        """

        normalized_inputs = self._normalize_observations(observations or {})
        prospective_count = len(set(self._cells).union(normalized_inputs))
        if prospective_count > self.config.max_active_cells:
            raise RuntimeError("active-cell budget exceeded")

        snapshot = dict(self._cells)
        active = tuple(sorted(set(snapshot).union(normalized_inputs), key=self.address))
        observed = {
            coordinate: normalized_inputs.get(
                coordinate,
                snapshot.get(coordinate, _ZERO_STATE).observed,
            )
            for coordinate in active
        }

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
                prediction += self.config.reasoning_gain * (
                    decoded[coordinate] - prediction
                )
                prediction += self.config.coupling_gain * laplacian
                prediction += self.config.consensus_gain * consensus
                prediction = self._clip_value(prediction)

            pre_correction_residual = observed[coordinate] - prediction
            omega = self._clip_value(
                self.config.omega_decay * previous.omega
                + self.config.omega_gain * pre_correction_residual
            )
            candidate = self._clip_value(prediction + omega)
            residual = observed[coordinate] - candidate
            valid = (
                math.isfinite(candidate)
                and math.isfinite(residual)
                and self.config.latent_min <= latents[coordinate] <= self.config.latent_max
                and abs(residual) <= self.config.residual_threshold
            )
            committed = candidate if valid else previous.committed

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
    ) -> FieldMetrics:
        """Run a fixed number of deterministic cycles and return final metrics."""

        if cycles <= 0:
            raise ValueError("cycles must be positive")
        metrics = self.metrics()
        for cycle_index in range(cycles):
            metrics = self.step(observations if cycle_index == 0 else None)
        return metrics

    def metrics(self) -> FieldMetrics:
        """Measure the currently committed sparse field."""

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
        )

    def _normalize_observations(
        self,
        observations: Mapping[Coordinate, float],
    ) -> Dict[Coordinate, float]:
        normalized: Dict[Coordinate, float] = {}
        for coordinate, value in observations.items():
            checked_coordinate = self._validate_coordinate(coordinate)
            checked_value = self._clip_value(self._require_finite(value, "observation"))
            normalized[checked_coordinate] = checked_value
        return normalized

    def _validate_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if not isinstance(coordinate, tuple) or len(coordinate) != 3:
            raise TypeError("coordinate must be a three-integer tuple")
        if not all(isinstance(component, int) for component in coordinate):
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

    def _neighbour_mean(
        self,
        observed: Mapping[Coordinate, float],
        coordinate: Coordinate,
    ) -> float:
        neighbours = tuple(self._neighbours(coordinate))
        if not neighbours:
            return 0.0
        return sum(observed.get(neighbour, 0.0) for neighbour in neighbours) / len(neighbours)

    def _laplacian(
        self,
        snapshot: Mapping[Coordinate, CellState],
        coordinate: Coordinate,
    ) -> float:
        center = snapshot.get(coordinate, _ZERO_STATE).committed
        return sum(
            snapshot.get(neighbour, _ZERO_STATE).committed - center
            for neighbour in self._neighbours(coordinate)
        )

    def _latent_consensus(
        self,
        latents: Mapping[Coordinate, int],
        coordinate: Coordinate,
    ) -> float:
        neighbours = tuple(self._neighbours(coordinate))
        if not neighbours:
            return 0.0
        center = latents[coordinate]
        return sum(latents.get(neighbour, 0) - center for neighbour in neighbours) / len(
            neighbours
        )

    def _quantize_q3(self, value: float) -> int:
        quantized = self._round_half_away_from_zero(3.0 * value)
        return max(self.config.latent_min, min(self.config.latent_max, quantized))

    def _decode_q3(self, latent: int) -> float:
        return self._clip_value(latent / 3.0)

    def _clip_value(self, value: float) -> float:
        return max(self.config.value_min, min(self.config.value_max, value))

    @staticmethod
    def _round_half_away_from_zero(value: float) -> int:
        if value >= 0.0:
            return math.floor(value + 0.5)
        return math.ceil(value - 0.5)

    @staticmethod
    def _require_finite(value: float, name: str) -> float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def _calculate_journal_digest(
        self,
        cycle: int,
        states: Mapping[Coordinate, CellState],
    ) -> str:
        digest = hashlib.sha256()
        digest.update(bytes.fromhex(self._journal_digest))
        digest.update(struct.pack(">Q", cycle))
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
        return digest.hexdigest()
