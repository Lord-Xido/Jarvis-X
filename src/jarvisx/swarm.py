"""Sparse deterministic reaction-diffusion runtime for Jarvis-X.

The virtual lattice may contain ``L**3`` addresses, but only active voxels are
materialised.  Each step is an auditable transaction:

observe -> predict -> evolve -> residual -> update Ω -> project Λ -> budget -> commit

The implementation is dependency-free, deterministic for a fixed manifest,
and deliberately separates the fast voxel field from the predictor parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite, sqrt, tanh
import json
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int, int]
Vector = List[float]
Matrix = List[List[float]]

NEIGHBOURS: Tuple[Coord, ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def zeros(dimensions: int) -> Vector:
    return [0.0] * dimensions


def l2(vector: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def _finite(vector: Sequence[float]) -> bool:
    return all(isfinite(value) for value in vector)


def _matrix_or_zeros(matrix: Optional[Sequence[Sequence[float]]], dimensions: int) -> Matrix:
    if matrix is None:
        return [[0.0] * dimensions for _ in range(dimensions)]
    result = [list(row) for row in matrix]
    if len(result) != dimensions or any(len(row) != dimensions for row in result):
        raise ValueError("predictor matrices must have shape dimensions x dimensions")
    if not all(_finite(row) for row in result):
        raise ValueError("predictor matrices must contain finite values")
    return result


@dataclass(frozen=True)
class SwarmConfig:
    """Numerical, sparse-allocation, memory, and constraint contract."""

    side_length: int = 1000
    dimensions: int = 30
    diffusion: float = 0.1
    reaction: float = 0.01
    time_step: float = 1.0
    spacing: float = 1.0
    learning_rate: float = 0.001
    regularization: float = 0.001
    memory_decay: float = 0.95
    memory_rate: float = 0.05
    memory_coupling: float = 0.0
    epsilon: float = 1e-8
    active_budget: int = 1_000_000
    max_abs_state: Optional[float] = None
    boundary: str = "zero"  # zero or periodic

    def __post_init__(self) -> None:
        if self.side_length <= 0 or self.dimensions <= 0:
            raise ValueError("side_length and dimensions must be positive")
        if self.diffusion < 0 or self.reaction < 0:
            raise ValueError("diffusion and reaction must be non-negative")
        if self.time_step <= 0 or self.spacing <= 0:
            raise ValueError("time_step and spacing must be positive")
        if self.learning_rate < 0 or self.regularization < 0:
            raise ValueError("learning_rate and regularization must be non-negative")
        if not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError("memory_decay must lie in [0, 1]")
        if self.memory_rate < 0 or self.memory_coupling < 0:
            raise ValueError("memory rates must be non-negative")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.active_budget <= 0:
            raise ValueError("active_budget must be positive")
        if self.max_abs_state is not None and self.max_abs_state <= 0:
            raise ValueError("max_abs_state must be positive when supplied")
        if self.boundary not in {"zero", "periodic"}:
            raise ValueError("boundary must be 'zero' or 'periodic'")
        if self.diffusion_number > (1.0 / 6.0) + 1e-12:
            raise ValueError(
                "explicit 3D diffusion is unstable: diffusion*time_step/spacing**2 must be <= 1/6"
            )

    @property
    def diffusion_number(self) -> float:
        return self.diffusion * self.time_step / (self.spacing * self.spacing)

    @property
    def diffusion_stability_margin(self) -> float:
        return 1.0 - 6.0 * self.diffusion_number


@dataclass
class Voxel:
    """Fast state q, fixed/companion state s, and persistent correction memory Ω."""

    theta: Vector
    state: Vector
    memory: Vector = field(default_factory=list)

    def validate(self, dimensions: int) -> None:
        if len(self.theta) != dimensions or len(self.state) != dimensions:
            raise ValueError("theta and state must match configured dimensions")
        if len(self.memory) != dimensions:
            raise ValueError("memory must match configured dimensions")
        if not (_finite(self.theta) and _finite(self.state) and _finite(self.memory)):
            raise ValueError("voxel values must be finite")


@dataclass(frozen=True)
class StepMetrics:
    iteration: int
    active_before: int
    active_after: int
    activated: int
    pruned: int
    prediction_mse: float
    regularization_loss: float
    constraint_loss: float
    sparse_budget_loss: float
    total_loss: float
    residual_l2: float
    motion_l2: float
    constraint_l2: float
    diffusion_number: float
    stability_margin: float
    budget_utilization: float
    journal_hash: str


@dataclass
class _Candidate:
    voxel: Voxel
    priority: float


class SparseSwarm30D:
    """Dictionary-backed sparse field over a virtual 3D lattice.

    A fixed ordering of coordinates and neighbours makes each step deterministic.
    Predictor matrices are slow model parameters; voxel ``theta`` is the fast
    runtime field retained for backwards compatibility with the initial prototype.
    """

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        predictor_weights: Optional[Sequence[Sequence[float]]] = None,
        predictor_bias: Optional[Sequence[float]] = None,
        local_predictor_weights: Optional[Sequence[Sequence[float]]] = None,
        memory_predictor_weights: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.config = config or SwarmConfig()
        dimensions = self.config.dimensions
        self.predictor_weights = _matrix_or_zeros(predictor_weights, dimensions)
        self.local_predictor_weights = _matrix_or_zeros(local_predictor_weights, dimensions)
        self.memory_predictor_weights = _matrix_or_zeros(memory_predictor_weights, dimensions)
        self.predictor_bias = (
            list(predictor_bias) if predictor_bias is not None else zeros(dimensions)
        )
        if len(self.predictor_bias) != dimensions or not _finite(self.predictor_bias):
            raise ValueError("predictor_bias must contain finite values for every dimension")
        self.voxels: Dict[Coord, Voxel] = {}
        self.iteration = 0
        self.journal_hash = "0" * 64

    @property
    def virtual_voxels(self) -> int:
        return self.config.side_length ** 3

    @property
    def active_voxels(self) -> int:
        return len(self.voxels)

    def manifest(self) -> Mapping[str, object]:
        """Return the execution manifest needed to reproduce numerical semantics."""

        cfg = self.config
        return {
            "runtime": "jarvisx.sparse-swarm-v1",
            "side_length": cfg.side_length,
            "dimensions": cfg.dimensions,
            "boundary": cfg.boundary,
            "diffusion": cfg.diffusion,
            "reaction": cfg.reaction,
            "time_step": cfg.time_step,
            "spacing": cfg.spacing,
            "learning_rate": cfg.learning_rate,
            "regularization": cfg.regularization,
            "memory_decay": cfg.memory_decay,
            "memory_rate": cfg.memory_rate,
            "memory_coupling": cfg.memory_coupling,
            "active_budget": cfg.active_budget,
            "max_abs_state": cfg.max_abs_state,
        }

    def _normalise_coord(self, coord: Coord) -> Optional[Coord]:
        length = self.config.side_length
        if self.config.boundary == "periodic":
            return (coord[0] % length, coord[1] % length, coord[2] % length)
        if all(0 <= axis < length for axis in coord):
            return coord
        return None

    def set_voxel(
        self,
        coord: Coord,
        theta: Sequence[float],
        state: Optional[Sequence[float]] = None,
        memory: Optional[Sequence[float]] = None,
    ) -> None:
        normalised = self._normalise_coord(coord)
        if normalised is None:
            raise ValueError("coordinate lies outside the virtual lattice")
        dimensions = self.config.dimensions
        voxel = Voxel(
            theta=list(theta),
            state=list(state) if state is not None else zeros(dimensions),
            memory=list(memory) if memory is not None else zeros(dimensions),
        )
        voxel.validate(dimensions)
        if self._is_active(voxel):
            if normalised not in self.voxels and len(self.voxels) >= self.config.active_budget:
                raise MemoryError("active voxel budget exceeded")
            self.voxels[normalised] = voxel
        else:
            self.voxels.pop(normalised, None)

    def _is_active(self, voxel: Voxel) -> bool:
        epsilon = self.config.epsilon
        return (
            l2(voxel.theta) > epsilon
            or l2(voxel.state) > epsilon
            or l2(voxel.memory) > epsilon
        )

    def global_latent(self) -> Vector:
        if not self.voxels:
            return zeros(self.config.dimensions)
        latent = zeros(self.config.dimensions)
        for coord in sorted(self.voxels):
            voxel = self.voxels[coord]
            for index, value in enumerate(voxel.theta):
                latent[index] += value
        scale = 1.0 / len(self.voxels)
        return [value * scale for value in latent]

    @staticmethod
    def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
        return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]

    def _prediction_delta(
        self,
        latent: Sequence[float],
        local_context: Sequence[float],
        memory: Sequence[float],
    ) -> Vector:
        global_term = self._matvec(self.predictor_weights, [tanh(value) for value in latent])
        local_term = self._matvec(
            self.local_predictor_weights, [tanh(value) for value in local_context]
        )
        memory_term = self._matvec(
            self.memory_predictor_weights, [tanh(value) for value in memory]
        )
        return [
            global_term[index]
            + local_term[index]
            + memory_term[index]
            + self.predictor_bias[index]
            for index in range(self.config.dimensions)
        ]

    def _candidate_coords(self) -> Set[Coord]:
        candidates: Set[Coord] = set(self.voxels)
        for x, y, z in sorted(self.voxels):
            for dx, dy, dz in NEIGHBOURS:
                coord = self._normalise_coord((x + dx, y + dy, z + dz))
                if coord is not None:
                    candidates.add(coord)
        return candidates

    def _theta_at(self, coord: Coord) -> Sequence[float]:
        voxel = self.voxels.get(coord)
        return voxel.theta if voxel is not None else zeros(self.config.dimensions)

    def _neighbour_sum(self, coord: Coord) -> Vector:
        result = zeros(self.config.dimensions)
        x, y, z = coord
        for dx, dy, dz in NEIGHBOURS:
            neighbour = self._normalise_coord((x + dx, y + dy, z + dz))
            if neighbour is None:
                continue
            values = self._theta_at(neighbour)
            for index, value in enumerate(values):
                result[index] += value
        return result

    def _project(self, vector: Sequence[float]) -> Tuple[Vector, float]:
        if not _finite(vector):
            raise FloatingPointError("non-finite swarm state rejected before commit")
        limit = self.config.max_abs_state
        if limit is None:
            return list(vector), 0.0
        projected = [max(-limit, min(limit, value)) for value in vector]
        delta = [before - after for before, after in zip(vector, projected)]
        return projected, sum(value * value for value in delta)

    def _priority(self, voxel: Voxel, residual: Sequence[float]) -> float:
        # Residual, state magnitude, and retained correction memory determine value.
        return l2(residual) + l2(voxel.theta) + 0.5 * l2(voxel.memory)

    def _commit_hash(self, metrics_payload: Mapping[str, object]) -> str:
        voxel_payload = [
            [coord, voxel.theta, voxel.state, voxel.memory]
            for coord, voxel in sorted(self.voxels.items())
        ]
        payload = {
            "previous": self.journal_hash,
            "manifest": self.manifest(),
            "iteration": self.iteration,
            "metrics": metrics_payload,
            "voxels": voxel_payload,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()

    def step(self) -> StepMetrics:
        cfg = self.config
        active_before_set = set(self.voxels)
        latent = self.global_latent()
        candidates = sorted(self._candidate_coords())
        provisional: Dict[Coord, _Candidate] = {}

        squared_error = 0.0
        squared_theta = 0.0
        residual_squared = 0.0
        motion_squared = 0.0
        constraint_squared = 0.0

        for coord in candidates:
            current = self.voxels.get(coord)
            theta = current.theta if current is not None else zeros(cfg.dimensions)
            state = current.state if current is not None else zeros(cfg.dimensions)
            memory = current.memory if current is not None else zeros(cfg.dimensions)
            neighbour_sum = self._neighbour_sum(coord)
            local_context = [value / 6.0 for value in neighbour_sum]
            prediction_delta = self._prediction_delta(latent, local_context, memory)

            actual_raw = zeros(cfg.dimensions)
            predicted = zeros(cfg.dimensions)
            residual = zeros(cfg.dimensions)
            updated_memory = zeros(cfg.dimensions)

            for index in range(cfg.dimensions):
                laplacian = neighbour_sum[index] - 6.0 * theta[index]
                diffusion_term = cfg.diffusion * laplacian / (cfg.spacing * cfg.spacing)
                reaction_term = cfg.reaction * tanh(theta[index]) * state[index]
                memory_term = cfg.memory_coupling * memory[index]
                predicted[index] = theta[index] + prediction_delta[index]
                actual_raw[index] = theta[index] + cfg.time_step * (
                    diffusion_term + reaction_term + memory_term
                )
                residual[index] = actual_raw[index] - predicted[index]

                # Stable local descent approximation. Full autodiff through the
                # sparse transition graph is intentionally a later backend.
                gradient = (
                    2.0 * residual[index]
                    + 2.0 * cfg.regularization * actual_raw[index]
                )
                actual_raw[index] -= cfg.learning_rate * gradient
                updated_memory[index] = (
                    cfg.memory_decay * memory[index] + cfg.memory_rate * residual[index]
                )

                squared_error += residual[index] * residual[index]
                residual_squared += residual[index] * residual[index]
                squared_theta += theta[index] * theta[index]

            projected, projection_error = self._project(actual_raw)
            constraint_squared += projection_error
            motion_squared += sum(
                (next_value - previous_value) ** 2
                for next_value, previous_value in zip(projected, theta)
            )
            voxel = Voxel(theta=projected, state=list(state), memory=updated_memory)
            voxel.validate(cfg.dimensions)
            if self._is_active(voxel):
                provisional[coord] = _Candidate(
                    voxel=voxel,
                    priority=self._priority(voxel, residual),
                )

        # Λ resource projection: preserve only the highest-value active addresses.
        if len(provisional) > cfg.active_budget:
            ranked = sorted(
                provisional.items(),
                key=lambda item: (-item[1].priority, item[0]),
            )[: cfg.active_budget]
            next_voxels = {coord: candidate.voxel for coord, candidate in ranked}
        else:
            next_voxels = {
                coord: provisional[coord].voxel for coord in sorted(provisional)
            }

        self.voxels = next_voxels
        self.iteration += 1
        active_after_set = set(self.voxels)

        element_denominator = max(len(candidates) * cfg.dimensions, 1)
        state_denominator = max(len(candidates), 1)
        prediction_mse = squared_error / element_denominator
        regularization_loss = cfg.regularization * squared_theta / element_denominator
        constraint_loss = constraint_squared / element_denominator
        budget_utilization = len(self.voxels) / cfg.active_budget
        sparse_budget_loss = budget_utilization * budget_utilization
        total_loss = (
            prediction_mse
            + regularization_loss
            + constraint_loss
            + sparse_budget_loss
        )

        metrics_payload = {
            "prediction_mse": prediction_mse,
            "regularization_loss": regularization_loss,
            "constraint_loss": constraint_loss,
            "sparse_budget_loss": sparse_budget_loss,
            "residual_l2": sqrt(residual_squared),
            "motion_l2": sqrt(motion_squared),
            "constraint_l2": sqrt(constraint_squared),
            "active_after": len(self.voxels),
        }
        self.journal_hash = self._commit_hash(metrics_payload)

        return StepMetrics(
            iteration=self.iteration,
            active_before=len(active_before_set),
            active_after=len(active_after_set),
            activated=len(active_after_set - active_before_set),
            pruned=len(active_before_set - active_after_set),
            prediction_mse=prediction_mse,
            regularization_loss=regularization_loss,
            constraint_loss=constraint_loss,
            sparse_budget_loss=sparse_budget_loss,
            total_loss=total_loss,
            residual_l2=sqrt(residual_squared),
            motion_l2=sqrt(motion_squared),
            constraint_l2=sqrt(constraint_squared),
            diffusion_number=cfg.diffusion_number,
            stability_margin=cfg.diffusion_stability_margin,
            budget_utilization=budget_utilization,
            journal_hash=self.journal_hash,
        )

    def run(
        self,
        iterations: int,
        tolerance: Optional[float] = None,
        motion_tolerance: Optional[float] = None,
    ) -> List[StepMetrics]:
        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        history: List[StepMetrics] = []
        for _ in range(iterations):
            metrics = self.step()
            history.append(metrics)
            prediction_converged = (
                tolerance is not None and metrics.residual_l2 <= tolerance
            )
            motion_converged = (
                motion_tolerance is None or metrics.motion_l2 <= motion_tolerance
            )
            if prediction_converged and motion_converged:
                break
        return history

    def snapshot(self) -> Mapping[Coord, Tuple[Tuple[float, ...], Tuple[float, ...]]]:
        """Backwards-compatible state snapshot without Ω memory."""

        return {
            coord: (tuple(voxel.theta), tuple(voxel.state))
            for coord, voxel in sorted(self.voxels.items())
        }

    def full_snapshot(
        self,
    ) -> Mapping[Coord, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]]:
        return {
            coord: (
                tuple(voxel.theta),
                tuple(voxel.state),
                tuple(voxel.memory),
            )
            for coord, voxel in sorted(self.voxels.items())
        }
