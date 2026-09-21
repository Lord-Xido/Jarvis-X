"""Bounded reference for the Jarvis-X 3D volumetric bytecode pipeline.

This module deliberately separates astronomical *logical* iteration/address
metadata from bounded physical execution. It is a research/reference layer and
does not replace the canonical JX3DVM1 bytecode authority boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import math
from typing import Mapping

Coord3 = tuple[int, int, int]
VoxelKey = tuple[str, int, int, int]
SparseField = dict[VoxelKey, int]


class VolumetricBytecodeError(ValueError):
    """Raised when a volumetric program or state violates its bounded contract."""


class VolumetricOp(str, Enum):
    INIT_LATTICE = "INIT_LATTICE"
    PACK_MODAL = "PACK_MODAL"
    SPARSE_MASK = "SPARSE_MASK"
    TRAVERSE_OCT = "TRAVERSE_OCT"
    SIMD_ENCODE = "SIMD_ENCODE"
    FOLD_XYZ_MIRROR = "FOLD_XYZ_MIRROR"
    INJECT_MIRROR_UNION = "INJECT_MIRROR_UNION"
    SIMD_DECODE = "SIMD_DECODE"
    DELTA_XOR = "DELTA_XOR"
    HAMMING_CHECK = "HAMMING_CHECK"
    VERIFY_CTR = "VERIFY_CTR"
    PERMEATE = "PERMEATE"
    HALT = "HALT"


REFERENCE_CYCLE_OPS: tuple[VolumetricOp, ...] = (
    VolumetricOp.SPARSE_MASK,
    VolumetricOp.TRAVERSE_OCT,
    VolumetricOp.SIMD_ENCODE,
    VolumetricOp.FOLD_XYZ_MIRROR,
    VolumetricOp.INJECT_MIRROR_UNION,
    VolumetricOp.SIMD_DECODE,
    VolumetricOp.DELTA_XOR,
    VolumetricOp.HAMMING_CHECK,
    VolumetricOp.VERIFY_CTR,
    VolumetricOp.PERMEATE,
)


@dataclass(frozen=True, slots=True)
class SymbolicIterationTarget:
    """Unevaluated ``base ** exponent`` iteration target.

    The numeric tower is metadata only. ``value`` is intentionally never
    materialized by the runtime.
    """

    base: int = 1_000_000
    exponent: int = 1_000_000

    def __post_init__(self) -> None:
        if self.base < 2:
            raise VolumetricBytecodeError("symbolic iteration base must be >= 2")
        if self.exponent < 1:
            raise VolumetricBytecodeError("symbolic iteration exponent must be positive")

    @property
    def log2_iterations(self) -> float:
        return self.exponent * math.log2(self.base)

    @property
    def decimal_exponent(self) -> int | None:
        """Return ``k`` when the target is exactly ``10 ** k``."""

        power = 0
        value = self.base
        while value > 1 and value % 10 == 0:
            value //= 10
            power += 1
        if value != 1:
            return None
        return power * self.exponent

    @property
    def description(self) -> str:
        decimal_exponent = self.decimal_exponent
        if decimal_exponent is not None:
            return f"10^{decimal_exponent}"
        return f"{self.base}^{self.exponent}"


@dataclass(frozen=True, slots=True)
class VolumetricConfig:
    """Logical geometry plus bounded physical execution policy."""

    axis_extent: int = 10**24
    lane_bits: int = 8
    octree_depth: int = 3
    max_active_voxels: int = 65_536
    max_iterations: int = 64
    epsilon_active_change: float = 0.0
    symbolic_target: SymbolicIterationTarget = SymbolicIterationTarget()

    def __post_init__(self) -> None:
        if self.axis_extent < 2:
            raise VolumetricBytecodeError("axis_extent must be >= 2")
        if self.lane_bits != 8:
            raise VolumetricBytecodeError("reference implementation currently supports 8-bit lanes")
        if not 0 <= self.octree_depth <= 30:
            raise VolumetricBytecodeError("octree_depth must be within [0, 30]")
        if self.max_active_voxels <= 0:
            raise VolumetricBytecodeError("max_active_voxels must be positive")
        if self.max_iterations <= 0:
            raise VolumetricBytecodeError("max_iterations must be positive")
        if not 0.0 <= self.epsilon_active_change <= 1.0:
            raise VolumetricBytecodeError("epsilon_active_change must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class CycleReceipt:
    iteration: int
    active_voxels_before: int
    active_voxels_after: int
    octree_tiles: int
    changed_bits: int
    active_change_fraction: float
    logical_change_fraction: float
    codec_roundtrip_error_bits: int
    committed: bool
    converged: bool
    state_hash: str
    executed_ops: tuple[VolumetricOp, ...] = REFERENCE_CYCLE_OPS


@dataclass(frozen=True, slots=True)
class RunResult:
    receipts: tuple[CycleReceipt, ...]
    converged: bool
    budget_exhausted: bool
    symbolic_target: str
    physical_iterations: int
    final_state_hash: str


@dataclass(frozen=True, slots=True)
class PermeationSnapshot:
    iteration: int
    active_voxels: int
    state_hash: str
    symbolic_target: str
    logical_axis_extent: int


def _lane_mask(bits: int) -> int:
    return (1 << bits) - 1


def _rotl(value: int, shift: int, bits: int) -> int:
    shift %= bits
    mask = _lane_mask(bits)
    value &= mask
    return ((value << shift) | (value >> (bits - shift))) & mask


def _rotr(value: int, shift: int, bits: int) -> int:
    return _rotl(value, bits - (shift % bits), bits)


def _validate_coord(coord: Coord3, axis_extent: int) -> None:
    if len(coord) != 3:
        raise VolumetricBytecodeError("3D coordinates require exactly three axes")
    if any(axis < 0 or axis >= axis_extent for axis in coord):
        raise VolumetricBytecodeError(f"coordinate {coord!r} lies outside logical lattice")


def _state_hash(field: Mapping[VoxelKey, int]) -> str:
    digest = sha256()
    for key, value in sorted(field.items()):
        modality, x, y, z = key
        digest.update(modality.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(x).encode("ascii"))
        digest.update(b",")
        digest.update(str(y).encode("ascii"))
        digest.update(b",")
        digest.update(str(z).encode("ascii"))
        digest.update(b"=")
        digest.update(bytes((value,)))
        digest.update(b"\n")
    return digest.hexdigest()


def _xor_delta(left: Mapping[VoxelKey, int], right: Mapping[VoxelKey, int]) -> SparseField:
    delta: SparseField = {}
    for key in set(left) | set(right):
        value = left.get(key, 0) ^ right.get(key, 0)
        if value:
            delta[key] = value
    return delta


def _hamming_bits(left: Mapping[VoxelKey, int], right: Mapping[VoxelKey, int]) -> int:
    return sum(value.bit_count() for value in _xor_delta(left, right).values())


class VolumetricBytecodeVM:
    """Sparse, fixed-budget executor for the supplied volumetric bytecode law.

    ``FOLD_XYZ_MIRROR`` uses exact central coordinates. ``INJECT`` is defined as
    a bitwise union between each encoded voxel and its mirror. That closure is
    monotone and reaches a mirror-symmetric fixed point without allocating the
    full logical lattice.
    """

    def __init__(self, config: VolumetricConfig | None = None) -> None:
        self.config = config or VolumetricConfig()
        self.state: SparseField = {}
        self.iteration = 0
        self.modalities: set[str] = set()

    def load_modalities(self, modalities: Mapping[str, Mapping[Coord3, int]]) -> None:
        packed: SparseField = {}
        mask = _lane_mask(self.config.lane_bits)
        for modality, field in modalities.items():
            if not modality:
                raise VolumetricBytecodeError("modality names must be non-empty")
            self.modalities.add(modality)
            for coord, value in field.items():
                _validate_coord(coord, self.config.axis_extent)
                if isinstance(value, bool) or not isinstance(value, int):
                    raise VolumetricBytecodeError("voxel values must be integer lane values")
                if not 0 <= value <= mask:
                    raise VolumetricBytecodeError(
                        f"voxel value {value} does not fit {self.config.lane_bits}-bit lane"
                    )
                if value:
                    x, y, z = coord
                    packed[(modality, x, y, z)] = value

        if len(packed) > self.config.max_active_voxels:
            raise VolumetricBytecodeError("initial active set exceeds max_active_voxels")
        self.state = packed
        self.iteration = 0

    def mirror_key(self, key: VoxelKey) -> VoxelKey:
        modality, x, y, z = key
        n = self.config.axis_extent - 1
        return modality, n - x, n - y, n - z

    def sparse_mask(self, field: Mapping[VoxelKey, int]) -> SparseField:
        return {key: value for key, value in field.items() if value != 0}

    def octree_tiles(self, field: Mapping[VoxelKey, int]) -> set[tuple[str, int, int, int]]:
        bins = 1 << self.config.octree_depth
        tile_edge = max(1, (self.config.axis_extent + bins - 1) // bins)
        tiles: set[tuple[str, int, int, int]] = set()
        for modality, x, y, z in field:
            tiles.add(
                (
                    modality,
                    min(x // tile_edge, bins - 1),
                    min(y // tile_edge, bins - 1),
                    min(z // tile_edge, bins - 1),
                )
            )
        return tiles

    def encode(self, field: Mapping[VoxelKey, int]) -> SparseField:
        return {
            key: _rotl(value, 1, self.config.lane_bits)
            for key, value in field.items()
            if value
        }

    def decode(self, field: Mapping[VoxelKey, int]) -> SparseField:
        return {
            key: _rotr(value, 1, self.config.lane_bits)
            for key, value in field.items()
            if value
        }

    def is_mirror_symmetric(self, field: Mapping[VoxelKey, int]) -> bool:
        """Return True when every active voxel has an equal XYZ mirror."""

        return all(
            field.get(self.mirror_key(key), 0) == value
            for key, value in field.items()
        )

    def fold_xyz_mirror_union(self, latent: Mapping[VoxelKey, int]) -> SparseField:
        """Close the latent field under XYZ mirror union, one pair at a time."""

        folded: SparseField = {}
        visited: set[VoxelKey] = set()
        for key, value in latent.items():
            if key in visited:
                continue
            mirror = self.mirror_key(key)
            union = value | latent.get(mirror, 0)
            if union:
                folded[key] = union
                if mirror != key:
                    folded[mirror] = union
            visited.add(key)
            visited.add(mirror)
        return folded

    def _logical_bits(self) -> int:
        modality_count = max(1, len(self.modalities))
        return (
            self.config.axis_extent**3
            * self.config.lane_bits
            * modality_count
        )

    def step(self) -> CycleReceipt:
        before = self.sparse_mask(self.state)
        if len(before) > self.config.max_active_voxels:
            raise VolumetricBytecodeError("active set exceeds max_active_voxels")

        tiles = self.octree_tiles(before)
        latent = self.encode(before)
        codec_roundtrip = self.decode(latent)
        codec_error = _hamming_bits(before, codec_roundtrip)

        if self.is_mirror_symmetric(before):
            # Fixed-point fast path: codec verification still runs, but an
            # already symmetric sparse field does not need another mirror
            # closure allocation and decode pass.
            candidate = before
        else:
            folded = self.fold_xyz_mirror_union(latent)
            candidate = self.sparse_mask(self.decode(folded))
        if len(candidate) > self.config.max_active_voxels:
            raise VolumetricBytecodeError("candidate active set exceeds max_active_voxels")

        delta = _xor_delta(before, candidate)
        changed_bits = sum(value.bit_count() for value in delta.values())
        active_denominator = max(
            self.config.lane_bits,
            self.config.lane_bits * len(set(before) | set(candidate)),
        )
        active_fraction = changed_bits / active_denominator
        logical_fraction = changed_bits / self._logical_bits()

        # CTR-style local verification: the encode/decode pair must remain exact,
        # all bounds must already have passed, and only then is the candidate
        # promoted to the live sparse state.
        committed = codec_error == 0
        if committed:
            self.state = candidate
        self.iteration += 1

        converged = committed and active_fraction <= self.config.epsilon_active_change
        return CycleReceipt(
            iteration=self.iteration,
            active_voxels_before=len(before),
            active_voxels_after=len(self.state),
            octree_tiles=len(tiles),
            changed_bits=changed_bits,
            active_change_fraction=active_fraction,
            logical_change_fraction=logical_fraction,
            codec_roundtrip_error_bits=codec_error,
            committed=committed,
            converged=converged,
            state_hash=_state_hash(self.state),
        )

    def run(self) -> RunResult:
        receipts: list[CycleReceipt] = []
        converged = False
        for _ in range(self.config.max_iterations):
            receipt = self.step()
            receipts.append(receipt)
            if receipt.converged:
                converged = True
                break

        return RunResult(
            receipts=tuple(receipts),
            converged=converged,
            budget_exhausted=not converged,
            symbolic_target=self.config.symbolic_target.description,
            physical_iterations=len(receipts),
            final_state_hash=_state_hash(self.state),
        )

    def permeate_snapshot(self) -> PermeationSnapshot:
        """Publish verified state metadata without introducing new authority."""

        return PermeationSnapshot(
            iteration=self.iteration,
            active_voxels=len(self.state),
            state_hash=_state_hash(self.state),
            symbolic_target=self.config.symbolic_target.description,
            logical_axis_extent=self.config.axis_extent,
        )
