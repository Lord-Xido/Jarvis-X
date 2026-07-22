"""Deterministic sparse 30D auto-encoding/decoding runtime.

The engine models a very large virtual manifold without allocating a dense
30-dimensional tensor.  Coordinates are materialised only when deterministic
routing activates them.  Each active cell carries latent, predictive,
residual, persistent-memory, and decoded state.

The implementation intentionally uses only the Python standard library so it
can sit beneath Jarvis-X's VM, policy gate, ledger, and sandbox layers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, Union


Coordinate30D = Tuple[int, ...]
NumericInput = Union[str, bytes, Sequence[float]]


@dataclass(frozen=True)
class Engine30DConfig:
    dimensions: int = 30
    latent_width: int = 16
    max_active_cells: int = 256
    coordinate_modulus: int = 65521
    omega_decay: float = 0.95
    error_learning_rate: float = 0.20
    latent_retention: float = 0.05
    correction_gain: float = 0.50
    memory_gain: float = 0.10
    coherence_limit: float = 1.0

    def __post_init__(self) -> None:
        if self.dimensions != 30:
            raise ValueError("the Jarvis-X manifold is fixed at 30 dimensions")
        if self.latent_width <= 0:
            raise ValueError("latent_width must be positive")
        if self.max_active_cells <= 0:
            raise ValueError("max_active_cells must be positive")
        if self.coordinate_modulus <= 1:
            raise ValueError("coordinate_modulus must be greater than one")


@dataclass
class Cell30D:
    latent: List[float]
    prediction: List[float]
    residual: List[float]
    omega: List[float]
    decoded: List[float]
    visits: int = 0

    @classmethod
    def empty(cls, width: int) -> "Cell30D":
        zeros = [0.0] * width
        return cls(
            latent=list(zeros),
            prediction=list(zeros),
            residual=list(zeros),
            omega=list(zeros),
            decoded=list(zeros),
        )

    def clone(self) -> "Cell30D":
        return Cell30D(
            latent=list(self.latent),
            prediction=list(self.prediction),
            residual=list(self.residual),
            omega=list(self.omega),
            decoded=list(self.decoded),
            visits=self.visits,
        )


@dataclass(frozen=True)
class Cycle30DResult:
    cycle: int
    active_coordinates: Tuple[Coordinate30D, ...]
    output: Tuple[float, ...]
    reconstruction_error: float
    prediction_error: float
    coherence: float
    committed: bool


@dataclass
class Sparse30DManifold:
    width: int
    max_active_cells: int
    cells: MutableMapping[Coordinate30D, Cell30D] = field(default_factory=dict)

    def get_or_create(self, coordinate: Coordinate30D) -> Cell30D:
        cell = self.cells.get(coordinate)
        if cell is not None:
            return cell
        if len(self.cells) >= self.max_active_cells:
            self._evict_least_visited()
        cell = Cell30D.empty(self.width)
        self.cells[coordinate] = cell
        return cell

    def _evict_least_visited(self) -> None:
        if not self.cells:
            return
        victim = min(self.cells, key=lambda key: (self.cells[key].visits, key))
        del self.cells[victim]


class ThirtyDAutoEncodingEngine:
    """Closed 30D encode -> predict -> correct -> decode runtime.

    The cycle is transactional.  Only touched cells are snapshotted; a failed
    coherence check restores them and leaves the committed cycle counter
    unchanged.
    """

    def __init__(self, config: Engine30DConfig = Engine30DConfig()) -> None:
        self.config = config
        self.manifold = Sparse30DManifold(
            width=config.latent_width,
            max_active_cells=config.max_active_cells,
        )
        self.cycles = 0
        self.last_result: Union[Cycle30DResult, None] = None

    @property
    def active_coordinates(self) -> Tuple[Coordinate30D, ...]:
        return tuple(sorted(self.manifold.cells))

    def cycle(self, observation: NumericInput) -> Cycle30DResult:
        vector = self._normalise(observation)
        routes = self._route(vector)
        snapshot = self._snapshot(routes)

        outputs: List[float] = []
        reconstruction_sq = 0.0
        prediction_sq = 0.0
        samples = 0

        try:
            for coordinate, chunk in routes:
                cell = self.manifold.get_or_create(coordinate)
                target = self._pad(chunk)
                latent = self._encode(target, cell.omega)
                prediction = self._predict(cell.latent, cell.omega)
                residual = [target[i] - prediction[i] for i in range(self.config.latent_width)]
                omega = self._update_omega(cell.omega, latent, residual)
                corrected = self._correct(latent, residual, omega)
                decoded = self._decode(corrected, omega)

                cell.latent = corrected
                cell.prediction = prediction
                cell.residual = residual
                cell.omega = omega
                cell.decoded = decoded
                cell.visits += 1

                emitted = decoded[: len(chunk)]
                outputs.extend(emitted)
                reconstruction_sq += sum((chunk[i] - emitted[i]) ** 2 for i in range(len(chunk)))
                prediction_sq += sum(residual[i] ** 2 for i in range(len(chunk)))
                samples += len(chunk)

            output = tuple(outputs[: len(vector)])
            coherence = self._coherence(routes)
            if not self._is_coherent(coherence):
                raise RuntimeError("30D candidate state failed coherence projection")

            self.cycles += 1
            denominator = max(samples, 1)
            result = Cycle30DResult(
                cycle=self.cycles,
                active_coordinates=tuple(coordinate for coordinate, _ in routes),
                output=output,
                reconstruction_error=math.sqrt(reconstruction_sq / denominator),
                prediction_error=math.sqrt(prediction_sq / denominator),
                coherence=coherence,
                committed=True,
            )
            self.last_result = result
            return result
        except Exception:
            self._restore(snapshot, routes)
            raise

    def observe_vm_state(self, opcode: int, registers: Mapping[str, int]) -> Cycle30DResult:
        """Encode a VM transition without mutating the VM register file."""
        ordered = [float(opcode)]
        for name in sorted(registers):
            ordered.append(float(registers[name]))
        return self.cycle(ordered)

    def _normalise(self, observation: NumericInput) -> List[float]:
        if isinstance(observation, str):
            raw = observation.encode("utf-8")
            values = [((byte / 255.0) * 2.0) - 1.0 for byte in raw]
        elif isinstance(observation, bytes):
            values = [((byte / 255.0) * 2.0) - 1.0 for byte in observation]
        else:
            values = [float(value) for value in observation]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("observation values must be finite")
            scale = max((abs(value) for value in values), default=1.0)
            scale = max(scale, 1.0)
            values = [max(-1.0, min(1.0, value / scale)) for value in values]

        return values or [0.0]

    def _route(self, vector: Sequence[float]) -> List[Tuple[Coordinate30D, List[float]]]:
        routes: List[Tuple[Coordinate30D, List[float]]] = []
        width = self.config.latent_width
        for index, start in enumerate(range(0, len(vector), width)):
            chunk = list(vector[start : start + width])
            coordinate = self._coordinate_for(chunk, index)
            routes.append((coordinate, chunk))
            if len(routes) >= self.config.max_active_cells:
                break
        return routes

    def _coordinate_for(self, chunk: Sequence[float], index: int) -> Coordinate30D:
        payload = ",".join("{:.12g}".format(value) for value in chunk)
        seed = ("{}:{}".format(index, payload)).encode("ascii")
        digest = hashlib.blake2b(seed, digest_size=60).digest()
        coordinate = tuple(
            int.from_bytes(digest[offset : offset + 2], "big") % self.config.coordinate_modulus
            for offset in range(0, 60, 2)
        )
        if len(coordinate) != self.config.dimensions:
            raise AssertionError("coordinate router did not emit 30 dimensions")
        return coordinate

    def _pad(self, chunk: Sequence[float]) -> List[float]:
        return list(chunk) + [0.0] * (self.config.latent_width - len(chunk))

    def _encode(self, target: Sequence[float], omega: Sequence[float]) -> List[float]:
        return [math.tanh(target[i] + self.config.memory_gain * omega[i]) for i in range(len(target))]

    def _predict(self, previous_latent: Sequence[float], omega: Sequence[float]) -> List[float]:
        return [
            math.tanh((0.75 * previous_latent[i]) + (0.25 * omega[i]))
            for i in range(self.config.latent_width)
        ]

    def _update_omega(
        self,
        previous: Sequence[float],
        latent: Sequence[float],
        residual: Sequence[float],
    ) -> List[float]:
        return [
            math.tanh(
                (self.config.omega_decay * previous[i])
                + (self.config.error_learning_rate * residual[i])
                + (self.config.latent_retention * latent[i])
            )
            for i in range(self.config.latent_width)
        ]

    def _correct(
        self,
        latent: Sequence[float],
        residual: Sequence[float],
        omega: Sequence[float],
    ) -> List[float]:
        return [
            math.tanh(
                latent[i]
                + (self.config.correction_gain * residual[i])
                + (self.config.memory_gain * omega[i])
            )
            for i in range(self.config.latent_width)
        ]

    def _decode(self, corrected: Sequence[float], omega: Sequence[float]) -> List[float]:
        return [
            math.tanh(corrected[i] + (self.config.memory_gain * omega[i]))
            for i in range(self.config.latent_width)
        ]

    def _coherence(self, routes: Iterable[Tuple[Coordinate30D, Sequence[float]]]) -> float:
        magnitudes: List[float] = []
        for coordinate, _ in routes:
            cell = self.manifold.cells[coordinate]
            magnitudes.extend(abs(value) for value in cell.latent)
            magnitudes.extend(abs(value) for value in cell.omega)
            magnitudes.extend(abs(value) for value in cell.decoded)
        if not magnitudes:
            return 1.0
        peak = max(magnitudes)
        return max(0.0, 1.0 - (peak / (self.config.coherence_limit + 1.0)))

    def _is_coherent(self, coherence: float) -> bool:
        if not math.isfinite(coherence):
            return False
        if len(self.manifold.cells) > self.config.max_active_cells:
            return False
        for cell in self.manifold.cells.values():
            fields = (cell.latent, cell.prediction, cell.residual, cell.omega, cell.decoded)
            if any(not math.isfinite(value) for field in fields for value in field):
                return False
            if any(abs(value) > self.config.coherence_limit for field in fields for value in field):
                return False
        return True

    def _snapshot(self, routes: Iterable[Tuple[Coordinate30D, Sequence[float]]]) -> Dict[Coordinate30D, Union[Cell30D, None]]:
        snapshot: Dict[Coordinate30D, Union[Cell30D, None]] = {}
        for coordinate, _ in routes:
            cell = self.manifold.cells.get(coordinate)
            snapshot[coordinate] = cell.clone() if cell is not None else None
        return snapshot

    def _restore(
        self,
        snapshot: Mapping[Coordinate30D, Union[Cell30D, None]],
        routes: Iterable[Tuple[Coordinate30D, Sequence[float]]],
    ) -> None:
        for coordinate, _ in routes:
            previous = snapshot.get(coordinate)
            if previous is None:
                self.manifold.cells.pop(coordinate, None)
            else:
                self.manifold.cells[coordinate] = previous
