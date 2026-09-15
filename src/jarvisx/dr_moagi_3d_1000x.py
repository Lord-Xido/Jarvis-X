"""Sparse 3D multimodal engine with a 1000x algorithmic work-reduction target.

The engine exposes a virtual 1000^3 address space but materializes only an
active fraction of 3D tiles.  With the canonical active fraction of 0.001 and
10^3-voxel tiles, one cycle touches 1,000,000 logical voxels instead of
1,000,000,000, giving a 1000x *work-reduction target*.  This is deliberately
separate from any wall-clock performance claim.

NumPy is an optional acceleration backend.  Install ``jarvisx[accel]`` to use
this module operationally.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by environments without accel extra
    np = None  # type: ignore[assignment]

CUBE_EDGE = 1000
VIRTUAL_VOXELS = CUBE_EDGE**3
VOXEL_BITS = 64
DEFAULT_TILE_EDGE = 10
DEFAULT_ACTIVE_FRACTION = 0.001
FEATURES_PER_TILE = 8


def _require_numpy() -> Any:
    if np is None:
        raise RuntimeError(
            "DrMoagi3D1000xEngine requires the optional acceleration backend; "
            "install with `python -m pip install -e '.[accel]'`"
        )
    return np


@dataclass(slots=True)
class EngineConfig:
    cube_edge: int = CUBE_EDGE
    tile_edge: int = DEFAULT_TILE_EDGE
    active_fraction: float = DEFAULT_ACTIVE_FRACTION
    latent_dim: int = 4
    omega_decay: float = 0.95
    residual_threshold: float = 1e-3
    correction_gain: float = 0.15
    max_active_tiles: int | None = None
    seed: int = 7

    def validate(self) -> None:
        if self.cube_edge < 1:
            raise ValueError("cube_edge must be positive")
        if not 1 <= self.tile_edge <= self.cube_edge:
            raise ValueError("tile_edge must be in [1, cube_edge]")
        if not 0.0 < self.active_fraction <= 1.0:
            raise ValueError("active_fraction must be in (0, 1]")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        if not 0.0 <= self.omega_decay < 1.0:
            raise ValueError("omega_decay must be in [0, 1)")
        if not 0.0 <= self.correction_gain <= 1.0:
            raise ValueError("correction_gain must be in [0, 1]")
        if self.max_active_tiles is not None and self.max_active_tiles < 1:
            raise ValueError("max_active_tiles must be positive when supplied")


@dataclass(slots=True)
class CycleTelemetry:
    cycle: int
    active_tiles: int
    active_voxels: int
    virtual_voxels: int
    work_reduction_factor: float
    encode_ms: float
    decode_ms: float
    residual_ms: float
    scheduling_ms: float
    total_ms: float
    residual_mse: float
    stable_fraction: float


@dataclass(slots=True)
class BenchmarkResult:
    scalar_ms: float
    vectorized_ms: float
    vectorized_speedup: float
    sparse_work_reduction: float
    effective_upper_bound: float
    sample_values: int


def _tile_grid(cube_edge: int, tile_edge: int) -> tuple[int, int]:
    tiles_per_axis = math.ceil(cube_edge / tile_edge)
    return tiles_per_axis, tiles_per_axis**3


def _tile_id_to_xyz(tile_ids: Any, tiles_per_axis: int) -> Any:
    backend = _require_numpy()
    ids = backend.asarray(tile_ids, dtype=backend.int64)
    z = ids // (tiles_per_axis * tiles_per_axis)
    rem = ids % (tiles_per_axis * tiles_per_axis)
    y = rem // tiles_per_axis
    x = rem % tiles_per_axis
    return backend.stack((x, y, z), axis=1)


def pack_voxel64(
    payload: Any,
    feature: Any,
    memory: Any,
    activation: Any,
    residual: Any,
    opcode: Any,
) -> Any:
    """Vector-pack the canonical 64-bit Cube64 control word.

    Layout: payload[16] | feature[12] | memory[8] | activation[8] |
    residual[12] | opcode[8].
    """

    backend = _require_numpy()
    fields = [
        backend.asarray(payload, dtype=backend.uint64),
        backend.asarray(feature, dtype=backend.uint64),
        backend.asarray(memory, dtype=backend.uint64),
        backend.asarray(activation, dtype=backend.uint64),
        backend.asarray(residual, dtype=backend.uint64),
        backend.asarray(opcode, dtype=backend.uint64),
    ]
    limits = (0xFFFF, 0xFFF, 0xFF, 0xFF, 0xFFF, 0xFF)
    names = ("payload", "feature", "memory", "activation", "residual", "opcode")
    for name, value, limit in zip(names, fields, limits):
        if backend.any(value > limit):
            raise ValueError(f"{name} overflow")

    p, f, m, a, r, o = fields
    word = p
    word = (word << backend.uint64(12)) | f
    word = (word << backend.uint64(8)) | m
    word = (word << backend.uint64(8)) | a
    word = (word << backend.uint64(12)) | r
    word = (word << backend.uint64(8)) | o
    return word


class MultimodalDescriptor:
    """Dependency-light byte front-end shared by text/image/audio/video/code/3D."""

    @staticmethod
    def from_bytes(payload: bytes, width: int = 4096) -> Any:
        backend = _require_numpy()
        if width < 1:
            raise ValueError("width must be positive")
        if not payload:
            return backend.zeros((width,), dtype=backend.float32)

        raw = backend.frombuffer(payload, dtype=backend.uint8)
        if raw.size >= width:
            usable = (raw.size // width) * width
            values = raw[:usable].reshape(width, -1).mean(axis=1)
        else:
            repeats = math.ceil(width / raw.size)
            values = backend.tile(raw, repeats)[:width].astype(backend.float32)

        values = values.astype(backend.float32) / 255.0
        digest = backend.frombuffer(
            hashlib.blake2b(payload, digest_size=64).digest(), dtype=backend.uint8
        ).astype(backend.float32) / 255.0
        digest = backend.resize(digest, width)
        return backend.clip(0.85 * values + 0.15 * digest, 0.0, 1.0).astype(
            backend.float32
        )


class DrMoagi3D1000xEngine:
    """Sparse, vectorized, residual-scheduled reference engine."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        backend = _require_numpy()
        self.config = config or EngineConfig()
        self.config.validate()
        self.tiles_per_axis, self.virtual_tiles = _tile_grid(
            self.config.cube_edge, self.config.tile_edge
        )
        self.voxels_per_tile = self.config.tile_edge**3
        self.virtual_voxels = self.config.cube_edge**3

        target_tiles = max(1, math.ceil(self.virtual_tiles * self.config.active_fraction))
        if self.config.max_active_tiles is not None:
            target_tiles = min(target_tiles, self.config.max_active_tiles)
        self.target_active_tiles = target_tiles

        self.rng = backend.random.default_rng(self.config.seed)
        self.active_tile_ids = backend.arange(target_tiles, dtype=backend.int64)
        self.tile_features = backend.zeros(
            (target_tiles, FEATURES_PER_TILE), dtype=backend.float32
        )
        self.omega = backend.zeros_like(self.tile_features)
        self.residual_score = backend.ones((target_tiles,), dtype=backend.float32)

        self.latent_dim = min(self.config.latent_dim, FEATURES_PER_TILE)
        q, _ = backend.linalg.qr(
            self.rng.normal(size=(FEATURES_PER_TILE, FEATURES_PER_TILE)).astype(
                backend.float32
            )
        )
        self.encoder_matrix = q[:, : self.latent_dim].astype(backend.float32)
        self.decoder_matrix = self.encoder_matrix.T.copy()
        self.cycle_index = 0

    @property
    def active_voxels(self) -> int:
        return int(self.active_tile_ids.size * self.voxels_per_tile)

    @property
    def work_reduction_factor(self) -> float:
        return self.virtual_voxels / max(1, self.active_voxels)

    def ingest_descriptor(self, descriptor: Sequence[float] | Any) -> None:
        backend = _require_numpy()
        values = backend.asarray(descriptor, dtype=backend.float32).reshape(-1)
        if values.size == 0:
            raise ValueError("descriptor must not be empty")
        required = self.target_active_tiles * FEATURES_PER_TILE
        repeated = backend.resize(values, required)
        self.tile_features[:] = repeated.reshape(self.target_active_tiles, FEATURES_PER_TILE)

    def ingest_bytes(self, payload: bytes) -> None:
        self.ingest_descriptor(MultimodalDescriptor.from_bytes(payload))

    def _encode(self) -> Any:
        return self.tile_features @ self.encoder_matrix

    def _decode(self, latent: Any) -> Any:
        return latent @ self.decoder_matrix

    def _update_omega(self) -> None:
        rho = self.config.omega_decay
        self.omega *= rho
        self.omega += (1.0 - rho) * self.tile_features

    def _schedule(self, residual: Any) -> None:
        backend = _require_numpy()
        scores = residual.mean(axis=1)
        order = backend.argsort(scores)[::-1]
        self.tile_features[:] = self.tile_features[order]
        self.omega[:] = self.omega[order]
        self.residual_score[:] = scores[order]
        self.active_tile_ids[:] = self.active_tile_ids[order]

    def cycle(self) -> CycleTelemetry:
        backend = _require_numpy()
        started = time.perf_counter()

        t0 = time.perf_counter()
        latent = self._encode()
        self._update_omega()
        latent += 0.05 * (self.omega @ self.encoder_matrix)
        encode_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        reconstruction = self._decode(latent)
        decode_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        delta = self.tile_features - reconstruction
        residual = backend.abs(delta)
        residual_mse = float(backend.mean(delta * delta))
        stable_fraction = float(
            backend.mean(backend.mean(residual, axis=1) <= self.config.residual_threshold)
        )
        residual_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._schedule(residual)
        self.tile_features -= self.config.correction_gain * (
            self.tile_features - reconstruction
        )
        backend.clip(self.tile_features, 0.0, 1.0, out=self.tile_features)
        scheduling_ms = (time.perf_counter() - t0) * 1000.0

        self.cycle_index += 1
        total_ms = (time.perf_counter() - started) * 1000.0
        return CycleTelemetry(
            cycle=self.cycle_index,
            active_tiles=int(self.active_tile_ids.size),
            active_voxels=self.active_voxels,
            virtual_voxels=self.virtual_voxels,
            work_reduction_factor=self.work_reduction_factor,
            encode_ms=encode_ms,
            decode_ms=decode_ms,
            residual_ms=residual_ms,
            scheduling_ms=scheduling_ms,
            total_ms=total_ms,
            residual_mse=residual_mse,
            stable_fraction=stable_fraction,
        )

    def packed_control_words(self, opcode: int = 8) -> Any:
        backend = _require_numpy()
        n = self.active_tile_ids.size
        feature = backend.clip(backend.rint(self.tile_features.mean(axis=1) * 4095), 0, 4095)
        residual = backend.clip(backend.rint(self.residual_score * 4095), 0, 4095)
        activation = backend.clip(backend.rint(self.tile_features.max(axis=1) * 255), 0, 255)
        memory = backend.clip(backend.rint(self.omega.mean(axis=1) * 255), 0, 255)
        payload = self.active_tile_ids % 65536
        opcodes = backend.full(n, opcode, dtype=backend.uint64)
        return pack_voxel64(payload, feature, memory, activation, residual, opcodes)

    def cloud_plan(self, workers: int = 8) -> dict[str, Any]:
        if workers < 1:
            raise ValueError("workers must be positive")
        assignments = [self.active_tile_ids[i::workers].tolist() for i in range(workers)]
        return {
            "workers": workers,
            "active_tiles": int(self.active_tile_ids.size),
            "virtual_tiles": int(self.virtual_tiles),
            "assignments": assignments,
            "rule": "move compute to resident tiles; synchronize summaries and halos",
        }

    def export_active_tile_centers(self, path: Path) -> None:
        backend = _require_numpy()
        xyz = _tile_id_to_xyz(self.active_tile_ids, self.tiles_per_axis)
        centers = (xyz * self.config.tile_edge + self.config.tile_edge / 2.0).astype(
            backend.float32
        )
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\nformat ascii 1.0\n")
            handle.write(f"element vertex {len(centers)}\n")
            handle.write("property float x\nproperty float y\nproperty float z\n")
            handle.write("property float residual\nend_header\n")
            for point, score in zip(centers, self.residual_score):
                handle.write(f"{point[0]} {point[1]} {point[2]} {float(score)}\n")

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "virtual_voxels": self.virtual_voxels,
            "active_voxels": self.active_voxels,
            "active_tiles": int(self.active_tile_ids.size),
            "work_reduction_factor": self.work_reduction_factor,
            "cycle": self.cycle_index,
            "wall_clock_1000x_guaranteed": False,
        }


def _scalar_roundtrip(x: Any, matrix: Any) -> Any:
    backend = _require_numpy()
    rows, features = x.shape
    latent_dim = matrix.shape[1]
    z = [[0.0] * latent_dim for _ in range(rows)]
    for i in range(rows):
        for j in range(latent_dim):
            for k in range(features):
                z[i][j] += float(x[i, k]) * float(matrix[k, j])
    output = backend.empty_like(x)
    for i in range(rows):
        for k in range(features):
            value = 0.0
            for j in range(latent_dim):
                value += z[i][j] * float(matrix[k, j])
            output[i, k] = value
    return output


def benchmark(sample_values: int = 50_000) -> BenchmarkResult:
    """Microbenchmark vectorization separately from sparse work reduction."""

    backend = _require_numpy()
    rng = backend.random.default_rng(123)
    rows = max(1, min(sample_values // FEATURES_PER_TILE, 20_000))
    x = rng.random((rows, FEATURES_PER_TILE), dtype=backend.float32)
    q, _ = backend.linalg.qr(
        rng.normal(size=(FEATURES_PER_TILE, FEATURES_PER_TILE)).astype(backend.float32)
    )
    matrix = q[:, :4].astype(backend.float32)

    t0 = time.perf_counter()
    scalar = _scalar_roundtrip(x, matrix)
    scalar_ms = (time.perf_counter() - t0) * 1000.0

    _ = (x @ matrix) @ matrix.T
    best_ms = float("inf")
    vectorized = None
    for _ in range(5):
        t0 = time.perf_counter()
        vectorized = (x @ matrix) @ matrix.T
        best_ms = min(best_ms, (time.perf_counter() - t0) * 1000.0)

    assert vectorized is not None
    if not backend.allclose(scalar, vectorized, atol=2e-5, rtol=2e-5):
        raise AssertionError("scalar and vectorized reference transforms diverged")

    vectorized_speedup = scalar_ms / max(best_ms, 1e-9)
    sparse_work_reduction = 1.0 / DEFAULT_ACTIVE_FRACTION
    return BenchmarkResult(
        scalar_ms=scalar_ms,
        vectorized_ms=best_ms,
        vectorized_speedup=vectorized_speedup,
        sparse_work_reduction=sparse_work_reduction,
        effective_upper_bound=vectorized_speedup * sparse_work_reduction,
        sample_values=rows * FEATURES_PER_TILE,
    )


__all__ = [
    "BenchmarkResult",
    "CycleTelemetry",
    "DEFAULT_ACTIVE_FRACTION",
    "DrMoagi3D1000xEngine",
    "EngineConfig",
    "MultimodalDescriptor",
    "VIRTUAL_VOXELS",
    "benchmark",
    "pack_voxel64",
]
