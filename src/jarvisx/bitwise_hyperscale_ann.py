"""Bitwise hyperscale 3D ANN address, LOD, and inward-folding reference layer.

This module operationalizes the uploaded 60-bit spatial-address contract without
materializing the logical 3D volume. It is intentionally a bounded reference:

* 20 bits per axis, interleaved X:Y:Z into a 60-bit Morton key;
* 20-level octree prefix extraction for deterministic LOD grouping;
* the source-defined fixed-point viewport projection and inward bitwise step;
* the source-defined FP32 top-byte truncation as a structural bit codec.

The top-byte truncation is not conventional calibrated neural-network INT8
quantization. It preserves the source document's bitwise operation while making
that capability boundary explicit.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

AXIS_BITS = 20
MORTON_BITS = 60
AXIS_LIMIT = 1 << AXIS_BITS
MORTON_LIMIT = 1 << MORTON_BITS
TARGET_SIDE = 1_000_000
OCTREE_LEVELS = AXIS_BITS
CORE_THRESHOLD = 0x00000800
UINT60_MASK = MORTON_LIMIT - 1


def _validate_axis(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < AXIS_LIMIT:
        raise ValueError(f"{name} must fit the 20-bit coordinate contract")


def _validate_key(key: int) -> None:
    if isinstance(key, bool) or not isinstance(key, int):
        raise TypeError("Morton key must be an integer")
    if not 0 <= key < MORTON_LIMIT:
        raise ValueError("Morton key must fit in 60 bits")


def morton60_encode(x: int, y: int, z: int) -> int:
    """Encode three 20-bit coordinates using source ordering X:Y:Z.

    Within every three-bit octree group, X is the most-significant bit, Y is
    the middle bit, and Z is the least-significant bit. Therefore the highest
    source group is bits 59..57 = x19:y19:z19.
    """

    _validate_axis("x", x)
    _validate_axis("y", y)
    _validate_axis("z", z)
    code = 0
    for bit in range(AXIS_BITS):
        code |= ((x >> bit) & 1) << (3 * bit + 2)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit)
    return code


def morton60_decode(key: int) -> tuple[int, int, int]:
    """Decode the source-ordered 60-bit Morton key."""

    _validate_key(key)
    x = y = z = 0
    for bit in range(AXIS_BITS):
        x |= ((key >> (3 * bit + 2)) & 1) << bit
        y |= ((key >> (3 * bit + 1)) & 1) << bit
        z |= ((key >> (3 * bit)) & 1) << bit
    return x, y, z


def octree_child_index(key: int, level: int) -> int:
    """Return the 3-bit child selector at root-relative level 0..19."""

    _validate_key(key)
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError("level must be an integer")
    if not 0 <= level < OCTREE_LEVELS:
        raise ValueError("level must be in [0, 19]")
    return (key >> (57 - 3 * level)) & 0b111


def lod_mask(depth: int) -> int:
    """Return the 60-bit mask retaining the first depth octree groups."""

    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if not 1 <= depth <= OCTREE_LEVELS:
        raise ValueError("depth must be in [1, 20]")
    trailing = MORTON_BITS - 3 * depth
    low = (1 << trailing) - 1 if trailing else 0
    return UINT60_MASK ^ low


def lod_key(key: int, depth: int) -> int:
    """Return the source document's zero-filled LOD cluster key."""

    _validate_key(key)
    return key & lod_mask(depth)


def lod_prefix(key: int, depth: int) -> int:
    """Return the compact octree prefix as a 3*depth bit integer."""

    _validate_key(key)
    lod_mask(depth)
    return key >> (MORTON_BITS - 3 * depth)


@dataclass(frozen=True)
class LODCluster:
    """Geometric meaning of one masked Morton prefix."""

    depth: int
    prefix: int
    key: int
    minimum: tuple[int, int, int]
    maximum: tuple[int, int, int]
    centroid: tuple[float, float, float]
    edge: int
    population: int

    @property
    def cloud_tile_id(self) -> str:
        width = max(1, math.ceil(3 * self.depth / 4))
        return f"m60:d{self.depth}:p{self.prefix:0{width}x}"


def lod_cluster(key: int, depth: int) -> LODCluster:
    """Resolve a Morton key to its deterministic octree/LOD cluster."""

    masked = lod_key(key, depth)
    minimum = morton60_decode(masked)
    edge = 1 << (AXIS_BITS - depth)
    min_x, min_y, min_z = minimum
    maximum = (min_x + edge - 1, min_y + edge - 1, min_z + edge - 1)
    half = (edge - 1) / 2.0
    centroid = (min_x + half, min_y + half, min_z + half)
    return LODCluster(
        depth=depth,
        prefix=lod_prefix(key, depth),
        key=masked,
        minimum=minimum,
        maximum=maximum,
        centroid=centroid,
        edge=edge,
        population=edge**3,
    )


def reference_q16_projection(value: float) -> int:
    """Apply the source document's fixed-point viewport projection exactly.

    The source labels this Q16.16, but its stated equation maps [-10, 10] into
    a 16-bit normalized interval and then shifts left by four additional bits.
    This preserves the stated equation rather than silently replacing it with
    conventional round(value * 2**16) Q16.16 encoding.
    """

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if not -10.0 <= value <= 10.0:
        raise ValueError("value must be in [-10, 10]")
    normalized = math.floor((value + 10.0) * (1 << 16) / 20.0)
    return normalized << 4


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def inward_vortex_step(point: tuple[int, int, int]) -> tuple[int, int, int]:
    """Execute one source-defined bitwise inward-vortex update."""

    if len(point) != 3:
        raise ValueError("point must have exactly three coordinates")
    x_fx, y_fx, z_fx = (_int32(int(value)) for value in point)
    dx = (y_fx >> 3) ^ (x_fx >> 5)
    dy = (-(x_fx >> 3)) ^ (y_fx >> 5)
    x_next = _int32((x_fx - dx) - (x_fx >> 7))
    y_next = _int32((y_fx - dy) - (y_fx >> 7))
    z_next = _int32(z_fx - (z_fx >> 6))
    return x_next, y_next, z_next


def inside_singularity_core(
    point: tuple[int, int, int],
    threshold: int = CORE_THRESHOLD,
) -> bool:
    """Test the source-defined axis-aligned singularity-core threshold."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return all(abs(int(value)) <= threshold for value in point)


def fp32_top_byte(value: float) -> int:
    """Return bits 31..24 of an IEEE-754 binary32 representation."""

    raw = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    return (raw >> 24) & 0xFF


def bit_truncate_fp32_to_int8(value: float) -> int:
    """Interpret source-defined FP32 top-byte extraction as signed INT8.

    This is a bit-structural codec operation, not scale/zero-point ANN
    quantization.
    """

    byte = fp32_top_byte(value)
    return byte - 256 if byte >= 128 else byte


def prune_truncated_int8(value: int, threshold: int) -> int:
    """Apply the source prose's absolute-magnitude pruning rule.

    Section 4's prose specifies abs(W_int8) < Threshold while its schematic
    mask omits the absolute value. This implementation follows the explicit
    absolute-magnitude rule and documents the discrepancy rather than hiding it.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    if not -128 <= value <= 127:
        raise ValueError("value must fit signed INT8")
    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise TypeError("threshold must be an integer")
    if not 0 <= threshold <= 128:
        raise ValueError("threshold must be in [0, 128]")
    return 0 if abs(value) < threshold else value


@dataclass(frozen=True)
class BitwiseAddress:
    """One logical 3D address plus its exact 60-bit Morton representation."""

    x: int
    y: int
    z: int
    key: int

    @classmethod
    def from_xyz(cls, x: int, y: int, z: int) -> "BitwiseAddress":
        return cls(x=x, y=y, z=z, key=morton60_encode(x, y, z))

    def cluster(self, depth: int) -> LODCluster:
        return lod_cluster(self.key, depth)
