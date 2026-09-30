from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Callable, Iterable, Sequence

LATENT_BYTES = 1024
WORLD_EXTENT = 1024
ADDRESS_BITS = 30
DEFAULT_LATENT_SHAPE = (8, 8, 16)


@dataclass(frozen=True)
class LatentFieldConfig:
    world_extent: int = WORLD_EXTENT
    latent_shape: tuple[int, int, int] = DEFAULT_LATENT_SHAPE
    latent_bytes: int = LATENT_BYTES
    max_backtracks: int = 8
    minimum_learning_rate: float = 1.0e-6

    def __post_init__(self) -> None:
        if isinstance(self.world_extent, bool) or not isinstance(self.world_extent, int):
            raise TypeError("world_extent must be an integer")
        if self.world_extent < 2:
            raise ValueError("world_extent must be at least 2")
        if len(self.latent_shape) != 3:
            raise ValueError("latent_shape must contain exactly three dimensions")
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in self.latent_shape
        ):
            raise TypeError("latent_shape dimensions must be integers")
        if any(value < 2 for value in self.latent_shape):
            raise ValueError("latent_shape dimensions must be at least 2")
        if self.latent_bytes != LATENT_BYTES:
            raise ValueError("this codec requires exactly 1,024 latent bytes")
        latent_cells = self.latent_shape[0] * self.latent_shape[1] * self.latent_shape[2]
        if latent_cells != self.latent_bytes:
            raise ValueError("latent_shape product must equal latent_bytes")
        if isinstance(self.max_backtracks, bool) or not isinstance(self.max_backtracks, int):
            raise TypeError("max_backtracks must be an integer")
        if self.max_backtracks < 1:
            raise ValueError("max_backtracks must be positive")
        if not isfinite(self.minimum_learning_rate) or self.minimum_learning_rate <= 0.0:
            raise ValueError("minimum_learning_rate must be finite and positive")


@dataclass(frozen=True)
class LatentFieldState:
    payload: bytes
    revision: int = 0

    def __post_init__(self) -> None:
        if len(self.payload) != LATENT_BYTES:
            raise ValueError("payload must be exactly 1,024 bytes")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 0:
            raise ValueError("revision cannot be negative")


@dataclass(frozen=True)
class FieldObservation:
    x: int
    y: int
    z: int
    target: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not isfinite(self.target) or not 0.0 <= self.target <= 1.0:
            raise ValueError("target must be finite and within [0, 1]")
        if not isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("weight must be finite and positive")


@dataclass(frozen=True)
class RefinementReport:
    state: LatentFieldState
    committed: bool
    loss_before: float
    loss_after: float
    learning_rate_used: float
    backtracks: int


class LatentFieldCodec:
    """Deterministic 1 KiB latent codec for a lazily queried virtual 3D scalar field."""

    def __init__(self, config: LatentFieldConfig | None = None) -> None:
        self.config = config or LatentFieldConfig()
        self._lx, self._ly, self._lz = self.config.latent_shape

    @property
    def latent_bytes(self) -> int:
        return self.config.latent_bytes

    @property
    def virtual_voxels(self) -> int:
        return self.config.world_extent**3

    @property
    def raw_u8_storage_bytes(self) -> int:
        return self.virtual_voxels

    @property
    def logical_expansion_ratio(self) -> int:
        return self.raw_u8_storage_bytes // self.latent_bytes

    def linear_address(self, x: int, y: int, z: int) -> int:
        self._validate_coordinate(x, y, z)
        extent = self.config.world_extent
        return x + extent * y + extent * extent * z

    def encode_field(self, field: Callable[[int, int, int], float]) -> LatentFieldState:
        if not callable(field):
            raise TypeError("field must be callable")
        payload = bytearray(self.latent_bytes)
        for iz in range(self._lz):
            z = self._latent_to_world(iz, self._lz)
            for iy in range(self._ly):
                y = self._latent_to_world(iy, self._ly)
                for ix in range(self._lx):
                    x = self._latent_to_world(ix, self._lx)
                    value = field(x, y, z)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError("field samples must be numeric")
                    value = float(value)
                    if not isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError("field samples must be finite and within [0, 1]")
                    payload[self._latent_index(ix, iy, iz)] = round(value * 255.0)
        return LatentFieldState(bytes(payload))

    def decode_voxel(self, state: LatentFieldState, x: int, y: int, z: int) -> float:
        self._validate_state(state)
        self._validate_coordinate(x, y, z)
        tx = self._world_to_latent(x, self._lx)
        ty = self._world_to_latent(y, self._ly)
        tz = self._world_to_latent(z, self._lz)
        return self._trilinear(state.payload, tx, ty, tz)

    def decode_points(
        self,
        state: LatentFieldState,
        coordinates: Iterable[tuple[int, int, int]],
    ) -> tuple[float, ...]:
        return tuple(self.decode_voxel(state, *coordinate) for coordinate in coordinates)

    def materialize_slice(
        self,
        state: LatentFieldState,
        *,
        axis: str,
        index: int,
        resolution: int = 64,
    ) -> tuple[tuple[float, ...], ...]:
        self._validate_state(state)
        axis = axis.lower()
        if axis not in {"x", "y", "z"}:
            raise ValueError("axis must be one of: x, y, z")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an integer")
        if not 0 <= index < self.config.world_extent:
            raise ValueError("index is outside the virtual world")
        if isinstance(resolution, bool) or not isinstance(resolution, int):
            raise TypeError("resolution must be an integer")
        if not 2 <= resolution <= self.config.world_extent:
            raise ValueError("resolution must be between 2 and world_extent")

        world = tuple(
            round(i * (self.config.world_extent - 1) / (resolution - 1))
            for i in range(resolution)
        )
        rows: list[tuple[float, ...]] = []
        for b in world:
            row: list[float] = []
            for a in world:
                coordinate = {
                    "x": (index, a, b),
                    "y": (a, index, b),
                    "z": (a, b, index),
                }[axis]
                row.append(self.decode_voxel(state, *coordinate))
            rows.append(tuple(row))
        return tuple(rows)

    def self_consistency_error(self, state: LatentFieldState) -> float:
        self._validate_state(state)
        squared_error = 0.0
        count = 0
        for iz in range(self._lz):
            z = self._latent_to_world(iz, self._lz)
            for iy in range(self._ly):
                y = self._latent_to_world(iy, self._ly)
                for ix in range(self._lx):
                    x = self._latent_to_world(ix, self._lx)
                    expected = state.payload[self._latent_index(ix, iy, iz)] / 255.0
                    residual = self.decode_voxel(state, x, y, z) - expected
                    squared_error += residual * residual
                    count += 1
        return (squared_error / count) ** 0.5

    def refine(
        self,
        state: LatentFieldState,
        observations: Sequence[FieldObservation],
        *,
        learning_rate: float = 0.25,
    ) -> RefinementReport:
        self._validate_state(state)
        if not observations:
            raise ValueError("observations cannot be empty")
        if not isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        for observation in observations:
            self._validate_observation_coordinate(observation)

        loss_before = self._observation_loss(state, observations)
        gradient = [0.0] * self.latent_bytes

        for observation in observations:
            prediction = self.decode_voxel(state, observation.x, observation.y, observation.z)
            residual = prediction - observation.target
            tx = self._world_to_latent(observation.x, self._lx)
            ty = self._world_to_latent(observation.y, self._ly)
            tz = self._world_to_latent(observation.z, self._lz)
            for latent_index, weight in self._neighbor_weights(tx, ty, tz):
                gradient[latent_index] += 2.0 * observation.weight * residual * weight

        total_weight = sum(observation.weight for observation in observations)
        gradient = [value / total_weight for value in gradient]

        rate = learning_rate
        for backtracks in range(self.config.max_backtracks):
            candidate_payload = bytearray(state.payload)
            for index, derivative in enumerate(gradient):
                normalized = state.payload[index] / 255.0 - rate * derivative
                candidate_payload[index] = round(clamp01(normalized) * 255.0)
            candidate = LatentFieldState(bytes(candidate_payload), revision=state.revision + 1)
            loss_after = self._observation_loss(candidate, observations)
            if loss_after <= loss_before:
                return RefinementReport(
                    state=candidate,
                    committed=True,
                    loss_before=loss_before,
                    loss_after=loss_after,
                    learning_rate_used=rate,
                    backtracks=backtracks,
                )
            rate *= 0.5
            if rate < self.config.minimum_learning_rate:
                break

        return RefinementReport(
            state=state,
            committed=False,
            loss_before=loss_before,
            loss_after=loss_before,
            learning_rate_used=0.0,
            backtracks=self.config.max_backtracks,
        )

    def _observation_loss(
        self,
        state: LatentFieldState,
        observations: Sequence[FieldObservation],
    ) -> float:
        weighted_squared_error = 0.0
        total_weight = 0.0
        for observation in observations:
            residual = (
                self.decode_voxel(state, observation.x, observation.y, observation.z)
                - observation.target
            )
            weighted_squared_error += observation.weight * residual * residual
            total_weight += observation.weight
        return weighted_squared_error / total_weight

    def _neighbor_weights(
        self,
        tx: float,
        ty: float,
        tz: float,
    ) -> tuple[tuple[int, float], ...]:
        x0, x1, fx = interpolation_pair(tx, self._lx)
        y0, y1, fy = interpolation_pair(ty, self._ly)
        z0, z1, fz = interpolation_pair(tz, self._lz)

        result: list[tuple[int, float]] = []
        for iz, wz in ((z0, 1.0 - fz), (z1, fz)):
            for iy, wy in ((y0, 1.0 - fy), (y1, fy)):
                for ix, wx in ((x0, 1.0 - fx), (x1, fx)):
                    weight = wx * wy * wz
                    if weight > 0.0:
                        result.append((self._latent_index(ix, iy, iz), weight))
        return tuple(result)

    def _trilinear(self, payload: bytes, tx: float, ty: float, tz: float) -> float:
        return sum(
            payload[index] / 255.0 * weight
            for index, weight in self._neighbor_weights(tx, ty, tz)
        )

    def _world_to_latent(self, value: int, latent_extent: int) -> float:
        return value * (latent_extent - 1) / (self.config.world_extent - 1)

    def _latent_to_world(self, value: int, latent_extent: int) -> int:
        return round(value * (self.config.world_extent - 1) / (latent_extent - 1))

    def _latent_index(self, x: int, y: int, z: int) -> int:
        return x + self._lx * y + self._lx * self._ly * z

    def _validate_state(self, state: LatentFieldState) -> None:
        if not isinstance(state, LatentFieldState):
            raise TypeError("state must be a LatentFieldState")

    def _validate_coordinate(self, x: int, y: int, z: int) -> None:
        for name, value in (("x", x), ("y", y), ("z", z)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= value < self.config.world_extent:
                raise ValueError(f"{name} is outside the virtual world")

    def _validate_observation_coordinate(self, observation: FieldObservation) -> None:
        if not isinstance(observation, FieldObservation):
            raise TypeError("observations must contain FieldObservation values")
        self._validate_coordinate(observation.x, observation.y, observation.z)


def interpolation_pair(value: float, extent: int) -> tuple[int, int, float]:
    lower = floor(value)
    upper = min(extent - 1, lower + 1)
    fraction = value - lower
    return lower, upper, fraction


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
