"""Sparse 64-bit virtual-cube contract for the Dr Moagi cloud runtime.

The module deliberately separates *virtual address-space size* from resident
memory.  A 1000^3 cube contains one billion logical addresses, but callers
materialize only bounded sparse records selected by the cloud policy.

The 64-bit word is a control/state descriptor.  It does not replace the
existing Q16.16x3 (96-bit) QVector numerical data plane.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

VOXEL_WORD_BITS = 64
VOXEL_WORD_BYTES = 8
COORD_BITS_PER_AXIS = 10
COORD_RECORD_BYTES = 4
SPARSE_RECORD_BYTES = COORD_RECORD_BYTES + VOXEL_WORD_BYTES
ONE_MIB = 1 << 20
DEFAULT_CUBE_SHAPE = (1000, 1000, 1000)
DEFAULT_TILE_SHAPE = (10, 10, 10)


def _shape3(shape: Sequence[int], *, maximum_axis: int | None = None) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError("shape must contain exactly three dimensions")
    result = tuple(int(axis) for axis in shape)
    if min(result) < 1:
        raise ValueError("shape dimensions must be positive")
    if maximum_axis is not None and max(result) > maximum_axis:
        raise ValueError(f"shape dimensions cannot exceed {maximum_axis}")
    return result  # type: ignore[return-value]


def _volume(shape: Sequence[int]) -> int:
    x, y, z = _shape3(shape)
    return x * y * z


def _bounded(name: str, value: int, bits: int) -> int:
    integer = int(value)
    maximum = (1 << bits) - 1
    if integer < 0 or integer > maximum:
        raise ValueError(f"{name} must fit unsigned {bits} bits")
    return integer


@dataclass(frozen=True, slots=True)
class VoxelWord64:
    """Exact 64-bit sparse voxel control/state descriptor.

    Bit layout, most-significant to least-significant::

        payload[16] | feature[12] | memory[8] | activation[8]
        | residual[12] | opcode[8]
    """

    payload: int = 0
    feature: int = 0
    memory: int = 0
    activation: int = 0
    residual: int = 0
    opcode: int = 0

    def __post_init__(self) -> None:
        _bounded("payload", self.payload, 16)
        _bounded("feature", self.feature, 12)
        _bounded("memory", self.memory, 8)
        _bounded("activation", self.activation, 8)
        _bounded("residual", self.residual, 12)
        _bounded("opcode", self.opcode, 8)

    @property
    def word(self) -> int:
        return (
            (self.payload << 48)
            | (self.feature << 36)
            | (self.memory << 28)
            | (self.activation << 20)
            | (self.residual << 8)
            | self.opcode
        )

    def to_bytes(self) -> bytes:
        return self.word.to_bytes(VOXEL_WORD_BYTES, "big", signed=False)

    @classmethod
    def from_word(cls, word: int) -> "VoxelWord64":
        raw = _bounded("word", word, VOXEL_WORD_BITS)
        return cls(
            payload=(raw >> 48) & 0xFFFF,
            feature=(raw >> 36) & 0xFFF,
            memory=(raw >> 28) & 0xFF,
            activation=(raw >> 20) & 0xFF,
            residual=(raw >> 8) & 0xFFF,
            opcode=raw & 0xFF,
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> "VoxelWord64":
        if len(payload) != VOXEL_WORD_BYTES:
            raise ValueError("voxel word must contain exactly 8 bytes")
        return cls.from_word(int.from_bytes(payload, "big", signed=False))


def pack_coord30(x: int, y: int, z: int) -> int:
    """Pack three 0..1023 coordinates into the low 30 bits of a uint32."""

    bx = _bounded("x", x, COORD_BITS_PER_AXIS)
    by = _bounded("y", y, COORD_BITS_PER_AXIS)
    bz = _bounded("z", z, COORD_BITS_PER_AXIS)
    return bx | (by << 10) | (bz << 20)


def unpack_coord30(value: int) -> tuple[int, int, int]:
    raw = _bounded("packed coordinate", value, 32)
    if raw >> 30:
        raise ValueError("packed coordinate uses reserved high bits")
    return (raw & 0x3FF, (raw >> 10) & 0x3FF, (raw >> 20) & 0x3FF)


@dataclass(frozen=True, slots=True)
class SparseVoxelRecord64:
    """Twelve-byte global sparse record: uint32 coordinate + uint64 state."""

    coordinate: tuple[int, int, int]
    state: VoxelWord64

    def __post_init__(self) -> None:
        pack_coord30(*self.coordinate)

    def to_bytes(self) -> bytes:
        return struct.pack(">IQ", pack_coord30(*self.coordinate), self.state.word)

    @classmethod
    def from_bytes(cls, payload: bytes) -> "SparseVoxelRecord64":
        if len(payload) != SPARSE_RECORD_BYTES:
            raise ValueError("sparse voxel record must contain exactly 12 bytes")
        coordinate, state = struct.unpack(">IQ", payload)
        return cls(unpack_coord30(coordinate), VoxelWord64.from_word(state))


@dataclass(frozen=True, slots=True)
class CubeWorkingSetPlan:
    shape: tuple[int, int, int]
    tile_shape: tuple[int, int, int]
    budget_bytes: int
    record_bytes: int
    virtual_cells: int
    dense_descriptor_bytes: int
    max_active_records: int
    max_active_fraction: float
    tile_cells: int
    max_full_tiles: int
    committed_record_bytes: int
    unused_budget_bytes: int


def plan_sparse_cube64(
    shape: Sequence[int] = DEFAULT_CUBE_SHAPE,
    *,
    budget_bytes: int = ONE_MIB,
    tile_shape: Sequence[int] = DEFAULT_TILE_SHAPE,
    record_bytes: int = SPARSE_RECORD_BYTES,
) -> CubeWorkingSetPlan:
    """Plan a bounded sparse working set without allocating the virtual cube."""

    shape3 = _shape3(shape, maximum_axis=1 << COORD_BITS_PER_AXIS)
    tile3 = _shape3(tile_shape)
    budget = int(budget_bytes)
    record = int(record_bytes)
    if budget < 1:
        raise ValueError("budget_bytes must be positive")
    if record < VOXEL_WORD_BYTES:
        raise ValueError("record_bytes cannot be smaller than the 8-byte voxel state")

    virtual_cells = _volume(shape3)
    max_active_records = budget // record
    tile_cells = _volume(tile3)
    max_full_tiles = max_active_records // tile_cells
    committed = max_active_records * record

    return CubeWorkingSetPlan(
        shape=shape3,
        tile_shape=tile3,
        budget_bytes=budget,
        record_bytes=record,
        virtual_cells=virtual_cells,
        dense_descriptor_bytes=virtual_cells * VOXEL_WORD_BYTES,
        max_active_records=max_active_records,
        max_active_fraction=max_active_records / virtual_cells,
        tile_cells=tile_cells,
        max_full_tiles=max_full_tiles,
        committed_record_bytes=committed,
        unused_budget_bytes=budget - committed,
    )


__all__ = [
    "COORD_BITS_PER_AXIS",
    "CubeWorkingSetPlan",
    "DEFAULT_CUBE_SHAPE",
    "DEFAULT_TILE_SHAPE",
    "ONE_MIB",
    "SPARSE_RECORD_BYTES",
    "SparseVoxelRecord64",
    "VOXEL_WORD_BITS",
    "VOXEL_WORD_BYTES",
    "VoxelWord64",
    "pack_coord30",
    "plan_sparse_cube64",
    "unpack_coord30",
]
