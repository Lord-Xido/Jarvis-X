"""Exynos-inspired heterogeneous 3D bit-fabric reference layer.

This module maps finite 32-bit machine words into the canonical Jarvis-X
1000 x 1000 x 1000 virtual lattice without dense allocation.

It deliberately does not model proprietary Samsung firmware or a physical
Exynos implementation. The CPU/GPU/NPU/ISP/CODEC/DSP/MODEM/DISPLAY labels are
heterogeneous-compute roles used to test routing, bit mapping, reversible
transport encoding, XOR/Hamming verification, and branchless commit semantics.

The numerical 3D autoencoding runtime remains SparseBillionField. This module
is an orthogonal bit/data mapping adapter around that same logical geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

SIDE: Final = 1000
VIRTUAL_CELLS: Final = SIDE**3
TILE_SIDE: Final = 10
TILES_PER_AXIS: Final = SIDE // TILE_SIDE
VIRTUAL_TILES: Final = TILES_PER_AXIS**3
WORD_BITS: Final = 32
UINT32_MASK: Final = (1 << WORD_BITS) - 1

Coordinate = tuple[int, int, int]


class ProcessingDomain(IntEnum):
    """Heterogeneous execution roles for the virtual SoC fabric."""

    CPU = 0
    GPU = 1
    NPU = 2
    ISP = 3
    CODEC = 4
    DSP = 5
    MODEM = 6
    DISPLAY = 7


DOMAIN_ANCHORS: Final[dict[ProcessingDomain, Coordinate]] = {
    ProcessingDomain.CPU: (120, 500, 500),
    ProcessingDomain.GPU: (300, 500, 500),
    ProcessingDomain.NPU: (500, 500, 500),
    ProcessingDomain.ISP: (680, 350, 500),
    ProcessingDomain.CODEC: (680, 650, 500),
    ProcessingDomain.DSP: (500, 250, 500),
    ProcessingDomain.MODEM: (500, 750, 500),
    ProcessingDomain.DISPLAY: (880, 500, 500),
}


@dataclass(frozen=True)
class SpatialAddress:
    """Exact coordinate, linear, tile, and intra-tile mapping."""

    coordinate: Coordinate
    linear: int
    tile: Coordinate
    tile_id: int
    local: Coordinate
    local_id: int


@dataclass(frozen=True)
class BitResidual:
    """Keep representation error separate from numeric-word error."""

    xor_mask: int
    hamming_distance: int
    numeric_delta: int


@dataclass(frozen=True)
class BitTransitionReceipt:
    """One candidate-first bit transition through an external Lambda decision."""

    spatial: SpatialAddress
    domain: ProcessingDomain
    before: int
    candidate: int
    committed: int
    residual: BitResidual
    accepted: bool


def _require_coordinate(coordinate: Coordinate) -> Coordinate:
    if not isinstance(coordinate, tuple) or len(coordinate) != 3:
        raise TypeError("coordinate must be a three-integer tuple")
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in coordinate):
        raise TypeError("coordinate components must be integers")
    if not all(0 <= v < SIDE for v in coordinate):
        raise ValueError("coordinate is outside the 1000^3 logical lattice")
    return coordinate


def _require_u32(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= UINT32_MASK:
        raise ValueError(f"{name} must fit unsigned 32 bits")
    return value


def linear_address(coordinate: Coordinate) -> int:
    """Map (x,y,z) bijectively into [0, 10^9)."""

    x, y, z = _require_coordinate(coordinate)
    return x + SIDE * (y + SIDE * z)


def coordinate_from_address(address: int) -> Coordinate:
    """Inverse of linear_address."""

    if isinstance(address, bool) or not isinstance(address, int):
        raise TypeError("address must be an integer")
    if not 0 <= address < VIRTUAL_CELLS:
        raise ValueError("address is outside the 1000^3 logical lattice")
    return address % SIDE, (address // SIDE) % SIDE, address // (SIDE * SIDE)


def spatial_address(coordinate: Coordinate) -> SpatialAddress:
    """Return the exact 10^3-tile decomposition for one logical coordinate."""

    x, y, z = _require_coordinate(coordinate)
    tx, ty, tz = x // TILE_SIDE, y // TILE_SIDE, z // TILE_SIDE
    lx, ly, lz = x % TILE_SIDE, y % TILE_SIDE, z % TILE_SIDE
    tile_id = tx + TILES_PER_AXIS * (ty + TILES_PER_AXIS * tz)
    local_id = lx + TILE_SIDE * (ly + TILE_SIDE * lz)
    return SpatialAddress(
        coordinate=(x, y, z),
        linear=linear_address((x, y, z)),
        tile=(tx, ty, tz),
        tile_id=tile_id,
        local=(lx, ly, lz),
        local_id=local_id,
    )


def coordinate_from_tile(tile_id: int, local_id: int) -> Coordinate:
    """Reconstruct a coordinate from exact tile and intra-tile IDs."""

    if isinstance(tile_id, bool) or not isinstance(tile_id, int):
        raise TypeError("tile_id must be an integer")
    if isinstance(local_id, bool) or not isinstance(local_id, int):
        raise TypeError("local_id must be an integer")
    if not 0 <= tile_id < VIRTUAL_TILES:
        raise ValueError("tile_id is outside the virtual tile lattice")
    if not 0 <= local_id < TILE_SIDE**3:
        raise ValueError("local_id is outside the tile")

    tx = tile_id % TILES_PER_AXIS
    ty = (tile_id // TILES_PER_AXIS) % TILES_PER_AXIS
    tz = tile_id // (TILES_PER_AXIS * TILES_PER_AXIS)
    lx = local_id % TILE_SIDE
    ly = (local_id // TILE_SIDE) % TILE_SIDE
    lz = local_id // (TILE_SIDE * TILE_SIDE)
    return tx * TILE_SIDE + lx, ty * TILE_SIDE + ly, tz * TILE_SIDE + lz


def route_domain(coordinate: Coordinate) -> ProcessingDomain:
    """Route a coordinate to the nearest declared heterogeneous-domain anchor."""

    x, y, z = _require_coordinate(coordinate)

    def rank(item: tuple[ProcessingDomain, Coordinate]) -> tuple[int, int]:
        domain, anchor = item
        ax, ay, az = anchor
        distance2 = (x - ax) ** 2 + (y - ay) ** 2 + (z - az) ** 2
        return distance2, int(domain)

    return min(DOMAIN_ANCHORS.items(), key=rank)[0]


def _mix32(value: int) -> int:
    value &= UINT32_MASK
    value ^= value >> 16
    value = (value * 0x7FEB352D) & UINT32_MASK
    value ^= value >> 15
    value = (value * 0x846CA68B) & UINT32_MASK
    value ^= value >> 16
    return value & UINT32_MASK


def procedural_word(coordinate: Coordinate, seed: int = 0x5EED3D26) -> int:
    """Materialize one deterministic virtual word without allocating 10^9 words."""

    _require_u32(seed, "seed")
    address = linear_address(coordinate)
    domain = int(route_domain(coordinate))
    return _mix32(address ^ seed ^ ((domain + 1) * 0x9E3779B9))


def bit_residual(before: int, candidate: int) -> BitResidual:
    """Measure representation and numeric residuals independently."""

    before = _require_u32(before, "before")
    candidate = _require_u32(candidate, "candidate")
    xor_mask = before ^ candidate
    return BitResidual(
        xor_mask=xor_mask,
        hamming_distance=xor_mask.bit_count(),
        numeric_delta=int(candidate) - int(before),
    )


def _rotl32(value: int, shift: int) -> int:
    shift &= 31
    if shift == 0:
        return value & UINT32_MASK
    return ((value << shift) | (value >> (32 - shift))) & UINT32_MASK


def _rotr32(value: int, shift: int) -> int:
    shift &= 31
    if shift == 0:
        return value & UINT32_MASK
    return ((value >> shift) | (value << (32 - shift))) & UINT32_MASK


def encode_transport_word(coordinate: Coordinate, word: int) -> int:
    """Reversibly encode a word for spatial transport.

    This is a structural bit codec, not compression and not a learned
    autoencoder. The learned/numerical autoencoding contract remains in the
    existing Jarvis-X field and codec runtimes.
    """

    word = _require_u32(word, "word")
    domain = route_domain(coordinate)
    key = procedural_word(coordinate, seed=0xA5A53D26)
    shift = 1 + (3 * int(domain)) % 31
    return _rotl32(word ^ key, shift)


def decode_transport_word(coordinate: Coordinate, encoded: int) -> int:
    """Inverse of encode_transport_word."""

    encoded = _require_u32(encoded, "encoded")
    domain = route_domain(coordinate)
    key = procedural_word(coordinate, seed=0xA5A53D26)
    shift = 1 + (3 * int(domain)) % 31
    return _rotr32(encoded, shift) ^ key


def branchless_commit(before: int, candidate: int, accepted: bool) -> int:
    """Select candidate or baseline with a full-width digital mask."""

    before = _require_u32(before, "before")
    candidate = _require_u32(candidate, "candidate")
    if not isinstance(accepted, bool):
        raise TypeError("accepted must be a bool")
    mask = UINT32_MASK if accepted else 0
    return ((candidate & mask) | (before & (~mask & UINT32_MASK))) & UINT32_MASK


def transition(
    coordinate: Coordinate,
    before: int,
    candidate: int,
    *,
    accepted: bool,
) -> BitTransitionReceipt:
    """Create one candidate-first transition receipt.

    The caller owns the Lambda admission decision. Rejection is guaranteed to
    preserve the exact baseline word.
    """

    before = _require_u32(before, "before")
    candidate = _require_u32(candidate, "candidate")
    spatial = spatial_address(coordinate)
    return BitTransitionReceipt(
        spatial=spatial,
        domain=route_domain(coordinate),
        before=before,
        candidate=candidate,
        committed=branchless_commit(before, candidate, accepted),
        residual=bit_residual(before, candidate),
        accepted=accepted,
    )
