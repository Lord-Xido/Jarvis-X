"""B3D1 cross-language binary contract for the isolated BitCube3D C++ SDK.

This module implements the native BitCube3D v2 snapshot wire format, *not*
the unrelated DM3DROM1 64^3-byte ROM protocol. It uses only Python stdlib.
The sparse 10^27-bit address domain is virtual; only active 16^3-bit
bricks are allocated. B3D1 is neither encrypted nor authenticated.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Mapping

SIDE = 1_000_000_000
BRICK_EDGE = 16
BRICK_BITS = BRICK_EDGE**3
BRICK_BYTES = BRICK_BITS // 8
BRICKS_PER_AXIS = SIDE // BRICK_EDGE
MAX_BRICKS = 100_000
MAX_TRANSFORM_ROUNDS = 32
HEADER = struct.Struct("<4sIIQ")
BRICK_HEADER = struct.Struct("<III")
MAGIC = b"B3D1"

Coord = tuple[int, int, int]
BrickCoord = tuple[int, int, int]


def _check_coord(point: Coord) -> None:
    if (
        not isinstance(point, tuple)
        or len(point) != 3
        or any(type(v) is not int or not 0 <= v < SIDE for v in point)
    ):
        raise ValueError("B3D1 voxel coordinate outside [0, 10^9-1]")


def _check_rounds(rounds: int, *, transform: bool = False) -> None:
    limit = MAX_TRANSFORM_ROUNDS if transform else 10_000
    if type(rounds) is not int or not 0 <= rounds <= limit:
        raise ValueError("B3D1 invalid round count")


def _check_limit(max_bricks: int) -> None:
    if type(max_bricks) is not int or max_bricks < 0 or max_bricks > MAX_BRICKS:
        raise ValueError("B3D1 max_bricks must be in [0, 100000]")


def _passes(brick: bytearray, *, inverse: bool = False) -> None:
    axes = (2, 1, 0) if inverse else (0, 1, 2)
    strides = (8, 4, 2, 1) if inverse else (1, 2, 4, 8)
    for axis in axes:
        for stride in strides:
            delta = (1, 16, 256)[axis] * stride
            for z in range(16):
                for y in range(16):
                    for x in range(16):
                        if (x, y, z)[axis] & stride:
                            continue
                        lo = x + 16 * y + 256 * z
                        if brick[lo >> 3] & (1 << (lo & 7)):
                            hi = lo + delta
                            brick[hi >> 3] ^= 1 << (hi & 7)


def _transformed(bricks: Mapping[BrickCoord, bytes], rounds: int, *,
                 inverse: bool = False) -> dict[BrickCoord, bytes]:
    _check_rounds(rounds, transform=True)
    out: dict[BrickCoord, bytes] = {}
    for coord, data in bricks.items():
        state = bytearray(data)
        for _ in range(rounds):
            _passes(state, inverse=inverse)
        out[coord] = bytes(state)
    return out


@dataclass(frozen=True)
class B3D1Snapshot:
    """A decoded snapshot; 'rounds' records outstanding native XOR passes."""

    rounds: int
    bricks: dict[BrickCoord, bytes]

    def serialize(self, *, max_bricks: int = MAX_BRICKS) -> bytes:
        _check_rounds(self.rounds)
        _check_limit(max_bricks)
        if len(self.bricks) > max_bricks:
            raise ValueError("B3D1 active brick count exceeds limit")
        result = bytearray(HEADER.pack(MAGIC, SIDE, self.rounds, len(self.bricks)))
        for coordinate, data in sorted(self.bricks.items()):
            if (
                len(coordinate) != 3
                or any(type(v) is not int or not 0 <= v < BRICKS_PER_AXIS for v in coordinate)
            ):
                raise ValueError("B3D1 invalid brick coordinate")
            if not isinstance(data, bytes) or len(data) != BRICK_BYTES or not any(data):
                raise ValueError("B3D1 requires nonempty immutable 512-byte bricks")
            result.extend(BRICK_HEADER.pack(*coordinate))
            result.extend(data)
        return bytes(result)

    def active_points(self, *, inverse_rounds: bool = False) -> list[Coord]:
        if inverse_rounds:
            data = _transformed(self.bricks, self.rounds, inverse=True)
        else:
            data = self.bricks
        result: list[Coord] = []
        for (bx, by, bz), brick in sorted(data.items()):
            for i, byte in enumerate(brick):
                for shift in range(8):
                    if byte & (1 << shift):
                        bit = i * 8 + shift
                        x, y, z = bit % 16, (bit // 16) % 16, bit // 256
                        result.append((bx * 16 + x, by * 16 + y, bz * 16 + z))
        return sorted(result)


def from_points(points: Iterable[Coord], *, rounds: int = 0,
                max_bricks: int = MAX_BRICKS) -> B3D1Snapshot:
    """Build the same bit-plane and forward XOR lifting as native BitCube3D."""
    _check_rounds(rounds, transform=True)
    _check_limit(max_bricks)
    bricks: dict[BrickCoord, bytearray] = {}
    for point in points:
        _check_coord(point)
        x, y, z = point
        key = (x // 16, y // 16, z // 16)
        if key not in bricks:
            if len(bricks) >= max_bricks:
                raise ValueError("B3D1 active brick count exceeds limit")
            bricks[key] = bytearray(BRICK_BYTES)
        index = x % 16 + 16 * (y % 16) + 256 * (z % 16)
        bricks[key][index >> 3] |= 1 << (index & 7)
    raw = {coord: bytes(data) for coord, data in bricks.items()}
    return B3D1Snapshot(rounds, _transformed(raw, rounds))


def deserialize(payload: bytes, *, max_bricks: int = MAX_BRICKS) -> B3D1Snapshot:
    """Strict canonical reader: bounded length, sorted unique nonzero bricks."""
    _check_limit(max_bricks)
    if not isinstance(payload, bytes):
        raise TypeError("B3D1 payload must be immutable bytes")
    if len(payload) < HEADER.size:
        raise ValueError("B3D1 truncated header")
    magic, side, rounds, count = HEADER.unpack_from(payload)
    if magic != MAGIC or side != SIDE:
        raise ValueError("B3D1 unsupported magic or side")
    _check_rounds(rounds)
    if count > max_bricks:
        raise ValueError("B3D1 active brick count exceeds limit")
    if len(payload) != HEADER.size + count * (BRICK_HEADER.size + BRICK_BYTES):
        raise ValueError("B3D1 byte length/brick count mismatch")
    bricks: dict[BrickCoord, bytes] = {}
    offset = HEADER.size
    previous: BrickCoord | None = None
    for _ in range(count):
        coord = BRICK_HEADER.unpack_from(payload, offset)
        offset += BRICK_HEADER.size
        if any(v >= BRICKS_PER_AXIS for v in coord):
            raise ValueError("B3D1 brick coordinate outside virtual cube")
        if previous is not None and coord <= previous:
            raise ValueError("B3D1 brick coordinates must be strictly sorted")
        brick = payload[offset:offset + BRICK_BYTES]
        if not any(brick):
            raise ValueError("B3D1 empty brick is forbidden")
        bricks[coord] = brick
        previous = coord
        offset += BRICK_BYTES
    return B3D1Snapshot(rounds, bricks)
