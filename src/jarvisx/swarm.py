"""Sparse 30-dimensional reaction-diffusion swarm for Jarvis-X.

The virtual lattice may contain L**3 addresses, but only active voxels are
materialised.  The implementation is dependency-free and deterministic so it
can be audited and embedded in the Jarvis-X VM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt, tanh
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

Coord = Tuple[int, int, int]
Vector = List[float]

NEIGHBOURS: Tuple[Coord, ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)


def zeros(dimensions: int) -> Vector:
    return [0.0] * dimensions


def l2(vector: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in vector))


@dataclass(frozen=True)
class SwarmConfig:
    side_length: int = 1000
    dimensions: int = 30
    diffusion: float = 0.1
    reaction: float = 0.01
    learning_rate: float = 0.001
    regularization: float = 0.001
    epsilon: float = 1e-8
    boundary: str = "zero"  # zero or periodic

    def __post_init__(self) -> None:
        if self.side_length <= 0 or self.dimensions <= 0:
            raise ValueError("side_length and dimensions must be positive")
        if self.diffusion < 0 or self.reaction < 0:
            raise ValueError("diffusion and reaction must be non-negative")
        if self.learning_rate < 0 or self.regularization < 0:
            raise ValueError("learning_rate and regularization must be non-negative")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.boundary not in {"zero", "periodic"}:
            raise ValueError("boundary must be 'zero' or 'periodic'")


@dataclass
class Voxel:
    theta: Vector
    state: Vector
    memory: Vector = field(default_factory=list)

    def validate(self, dimensions: int) -> None:
        if len(self.theta) != dimensions or len(self.state) != dimensions:
            raise ValueError(f"theta and state must contain {dimensions} values")
        if self.memory and len(self.memory) != dimensions:
            raise ValueError(f"memory must be empty or contain {dimensions} values")


@dataclass(frozen=True)
class StepMetrics:
    iteration: int
    active_before: int
    active_after: int
    activated: int
    pruned: int
    prediction_mse: float
    regularization_loss: float
    total_loss: float
    residual_l2: float


class SparseSwarm30D:
    """Dictionary-backed sparse field over a virtual 3D lattice."""

    def __init__(
        self,
        config: SwarmConfig | None = None,
        predictor_weights: Sequence[Sequence[float]] | None = None,
        predictor_bias: Sequence[float] | None = None,
    ) -> None:
        self.config = config or SwarmConfig()
        d = self.config.dimensions
        self.predictor_weights = (
            [list(row) for row in predictor_weights]
            if predictor_weights is not None
            else [[0.0] * d for _ in range(d)]
        )
        self.predictor_bias = list(predictor_bias) if predictor_bias is not None else zeros(d)
        if len(self.predictor_weights) != d or any(len(row) != d for row in self.predictor_weights):
            raise ValueError(f"predictor_weights must have shape {d}x{d}")
        if len(self.predictor_bias) != d:
            raise ValueError(f"predictor_bias must contain {d} values")
        self.voxels: Dict[Coord, Voxel] = {}
        self.iteration = 0

    @property
    def virtual_voxels(self) -> int:
        return self.config.side_length ** 3

    def _normalise_coord(self, coord: Coord) -> Coord | None:
        length = self.config.side_length
        if self.config.boundary == "periodic":
            return tuple(axis % length for axis in coord)  # type: ignore[return-value]
        if all(0 <= axis < length for axis in coord):
            return coord
        return None

    def set_voxel(
        self,
        coord: Coord,
        theta: Sequence[float],
        state: Sequence[float] | None = None,
        memory: Sequence[float] | None = None,
    ) -> None:
        normalised = self._normalise_coord(coord)
        if normalised is None:
            raise ValueError(f"coordinate {coord!r} lies outside the virtual lattice")
        voxel = Voxel(
            theta=list(theta),
            state=list(state) if state is not None else zeros(self.config.dimensions),
            memory=list(memory) if memory is not None else [],
        )
        voxel.validate(self.config.dimensions)
        if l2(voxel.theta) > self.config.epsilon or l2(voxel.state) > self.config.epsilon:
            self.voxels[normalised] = voxel
        else:
            self.voxels.pop(normalised, None)

    def global_latent(self) -> Vector:
        if not self.voxels:
            return zeros(self.config.dimensions)
        latent = zeros(self.config.dimensions)
        for voxel in self.voxels.values():
            for index, value in enumerate(voxel.theta):
                latent[index] += value
        scale = 1.0 / len(self.voxels)
        return [value * scale for value in latent]

    def _prediction_delta(self, latent: Sequence[float]) -> Vector:
        encoded = [tanh(value) for value in latent]
        return [
            sum(weight * encoded[index] for index, weight in enumerate(row)) + self.predictor_bias[out]
            for out, row in enumerate(self.predictor_weights)
        ]

    def _candidate_coords(self) -> set[Coord]:
        candidates = set(self.voxels)
        for x, y, z in tuple(self.voxels):
            for dx, dy, dz in NEIGHBOURS:
                coord = self._normalise_coord((x + dx, y + dy, z + dz))
                if coord is not None:
                    candidates.add(coord)
        return candidates

    def _theta_at(self, coord: Coord) -> Sequence[float]:
        voxel = self.voxels.get(coord)
        return voxel.theta if voxel is not None else zeros(self.config.dimensions)

    def step(self) -> StepMetrics:
        cfg = self.config
        active_before = len(self.voxels)
        latent = self.global_latent()
        prediction_delta = self._prediction_delta(latent)
        candidates = self._candidate_coords()
        next_voxels: Dict[Coord, Voxel] = {}
        squared_error = 0.0
        squared_theta = 0.0
        residual_squared = 0.0

        for coord in sorted(candidates):
            current = self.voxels.get(coord)
            theta = current.theta if current is not None else zeros(cfg.dimensions)
            state = current.state if current is not None else zeros(cfg.dimensions)
            memory = current.memory[:] if current is not None else []

            neighbour_sum = zeros(cfg.dimensions)
            x, y, z = coord
            for dx, dy, dz in NEIGHBOURS:
                neighbour = self._normalise_coord((x + dx, y + dy, z + dz))
                if neighbour is None:
                    continue
                neighbour_theta = self._theta_at(neighbour)
                for index, value in enumerate(neighbour_theta):
                    neighbour_sum[index] += value

            actual = zeros(cfg.dimensions)
            predicted = zeros(cfg.dimensions)
            residual = zeros(cfg.dimensions)
            for index in range(cfg.dimensions):
                laplacian = neighbour_sum[index] - 6.0 * theta[index]
                reaction_term = tanh(theta[index]) * state[index]
                predicted[index] = theta[index] + prediction_delta[index]
                actual[index] = theta[index] + cfg.diffusion * laplacian + cfg.reaction * reaction_term
                residual[index] = actual[index] - predicted[index]

                # Stable local descent approximation: minimise residual energy
                # plus L2 regularisation without materialising a global Jacobian.
                gradient = 2.0 * residual[index] + 2.0 * cfg.regularization * actual[index]
                actual[index] -= cfg.learning_rate * gradient

                squared_error += residual[index] ** 2
                residual_squared += residual[index] ** 2
                squared_theta += theta[index] ** 2

            if l2(actual) > cfg.epsilon or l2(state) > cfg.epsilon:
                next_voxels[coord] = Voxel(theta=actual, state=list(state), memory=memory)

        active_after = len(next_voxels)
        denominator = max(active_before, 1)
        prediction_mse = squared_error / denominator
        regularization_loss = cfg.regularization * squared_theta / denominator
        self.voxels = next_voxels
        self.iteration += 1
        return StepMetrics(
            iteration=self.iteration,
            active_before=active_before,
            active_after=active_after,
            activated=max(active_after - active_before, 0),
            pruned=max(active_before - active_after, 0),
            prediction_mse=prediction_mse,
            regularization_loss=regularization_loss,
            total_loss=prediction_mse + regularization_loss,
            residual_l2=sqrt(residual_squared),
        )

    def run(self, iterations: int, tolerance: float | None = None) -> List[StepMetrics]:
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        history: List[StepMetrics] = []
        for _ in range(iterations):
            metrics = self.step()
            history.append(metrics)
            if tolerance is not None and metrics.residual_l2 <= tolerance:
                break
        return history

    def snapshot(self) -> Mapping[Coord, Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        return {
            coord: (tuple(voxel.theta), tuple(voxel.state))
            for coord, voxel in sorted(self.voxels.items())
        }
