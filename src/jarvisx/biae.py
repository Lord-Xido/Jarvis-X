"""Packed 3D-BIAE-Ω v1 reference profile. See docs/research/3d_biae_omega."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from .dm3d_rom import morton3_local

SIDES = (4, 8, 16, 32, 64, 128, 256, 512, 1024)
SEED = 0x7FFF0133
DOMAIN = b"JX/3D-BIAE-Omega/v1\x00"
BLOCK_BYTES = 1 << 16
MAX_TICK = (1 << 64) - 1


def packed_size(side: int) -> int:
    if type(side) is not int or side not in SIDES:
        raise ValueError("side must be a power of two from 4 through 1024")
    axis_bits = side.bit_length() - 1
    return 1 << (3 * axis_bits - 3)


def _check_plane(field: bytes, size: int) -> None:
    if not isinstance(field, bytes) or len(field) != size:
        raise ValueError(f"expected exactly {size} immutable packed bytes per field")


def seeded_field(side: int, seed: int = SEED) -> bytes:
    """Create a reproducible test volume from a little-endian unsigned seed."""
    size = packed_size(side)
    if type(seed) is not int or not 0 <= seed < 1 << 32:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    return hashlib.shake_256(DOMAIN + b"seed\x00" + struct.pack("<I", seed)).digest(size)


@dataclass(frozen=True)
class Packet:
    latent: bytes
    residual: bytes
    drift: bytes

    def check_size(self, size: int) -> None:
        for plane in (self.latent, self.residual, self.drift):
            _check_plane(plane, size)


@dataclass(frozen=True)
class State:
    side: int
    field: bytes
    tick: int = 0
    omega: bytes = bytes(32)
    packet: Packet | None = None

    def __post_init__(self) -> None:
        size = packed_size(self.side)
        _check_plane(self.field, size)
        if type(self.tick) is not int or not 0 <= self.tick <= MAX_TICK:
            raise ValueError("tick must fit in an unsigned 64-bit integer")
        _check_plane(self.omega, 32)
        if self.packet is not None:
            self.packet.check_size(size)


def _blocks(size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, size, BLOCK_BYTES):
        yield start, min(start + BLOCK_BYTES, size)


@lru_cache(maxsize=16)
def _masks(length: int) -> tuple[int, int]:
    # Each Morton octet has one bit per (x0, y0, z0) triple, X bit first.
    return (1 << (length << 3)) - 1, int.from_bytes(b"\x96" * length, "little")


def _window(field: bytes, start: int, end: int) -> int:
    # Two halo bytes on each end suffice for every shift in E and D.
    before = field[start - 2 : start] if start else field[-2:]
    after = field[end : end + 2] if end < len(field) else field[:2]
    return int.from_bytes(before + field[start:end] + after, "little")


def _encode_block(field: bytes, start: int, end: int) -> tuple[int, int]:
    mask, coordinates = _masks(end - start)
    window = _window(field, start, end)
    center = window >> 16
    latent = (center ^ (window >> (16 - 7)) ^ (window >> (16 + 13))) & mask
    drift = (center ^ (window >> (16 - 3)) ^ coordinates) & mask
    return latent, drift


def _predict_block(latent: bytes, drift: bytes, start: int, end: int) -> int:
    mask, _ = _masks(end - start)
    warped = _window(latent, start, end) ^ _window(drift, start, end)
    return ((warped >> 16) ^ (warped >> (16 - 3)) ^ (warped >> (16 + 11))) & mask


def _corrected_blocks(latent: bytes, drift: bytes, correction: bytes) -> Iterator[bytes]:
    for start, end in _blocks(len(latent)):
        prediction = _predict_block(latent, drift, start, end)
        other = int.from_bytes(correction[start:end], "little")
        yield (prediction ^ other).to_bytes(end - start, "little")


def encode(side: int, field: bytes) -> Packet:
    """Produce the three same-width fields needed by the standalone decoder."""
    size = packed_size(side)
    _check_plane(field, size)
    latent_chunks, drift_chunks = [], []
    for start, end in _blocks(size):
        latent, drift = _encode_block(field, start, end)
        latent_chunks.append(latent.to_bytes(end - start, "little"))
        drift_chunks.append(drift.to_bytes(end - start, "little"))
    latent_field = b"".join(latent_chunks)
    del latent_chunks
    drift_field = b"".join(drift_chunks)
    del drift_chunks
    residual = b"".join(_corrected_blocks(latent_field, drift_field, field))
    return Packet(latent_field, residual, drift_field)


def decode(side: int, packet: Packet) -> bytes:
    """Reconstruct a packed lattice using only the packet and geometry."""
    packet.check_size(packed_size(side))
    return b"".join(_corrected_blocks(packet.latent, packet.drift, packet.residual))


def diffusion_bit(side: int, field: bytes, x: int, y: int, z: int) -> int:
    """Evaluate the optional six-face XOR stencil at one periodic voxel."""
    _check_plane(field, packed_size(side))
    if any(type(c) is not int or not 0 <= c < side for c in (x, y, z)):
        raise ValueError("voxel coordinate outside the lattice")
    wrap = side - 1
    neighbors = (
        ((x - 1) & wrap, y, z),
        ((x + 1) & wrap, y, z),
        (x, (y - 1) & wrap, z),
        (x, (y + 1) & wrap, z),
        (x, y, (z - 1) & wrap),
        (x, y, (z + 1) & wrap),
    )
    parity = 0
    for neighbor in neighbors:
        index = morton3_local(*neighbor)
        parity ^= (field[index >> 3] >> (index & 7)) & 1
    return parity


class GateRejected(ValueError):
    """The proposed packet failed verification. The caller retains its state."""


def commit(state: State, packet: Packet) -> tuple[State, dict[str, int | str]]:
    """Verify a proposal and return a new immutable state and a replay receipt."""
    size = len(state.field)
    packet.check_size(size)
    if state.tick == MAX_TICK:
        raise GateRejected("tick counter exhausted")
    for start, end in _blocks(size):
        latent, drift = _encode_block(state.field, start, end)
        if latent != int.from_bytes(packet.latent[start:end], "little"):
            raise GateRejected("latent differs from E(B)")
        if drift != int.from_bytes(packet.drift[start:end], "little"):
            raise GateRejected("drift differs from the coordinate warp")
    parts = _corrected_blocks(packet.latent, packet.drift, packet.residual)
    for (start, end), part in zip(_blocks(size), parts, strict=True):
        if part != state.field[start:end]:
            raise GateRejected("reconstruction differs from B")

    # Bytewise equality permits the next state to share the verified input buffer.
    names = ("B", "Z", "epsilon", "delta")
    planes = (state.field, packet.latent, packet.residual, packet.drift)
    digests = tuple(hashlib.sha3_256(plane).digest() for plane in planes)
    gate = 1
    preimage = (
        DOMAIN
        + state.omega
        + struct.pack("<I", state.side)
        + b"".join(digests)
        + struct.pack("<QB", state.tick, gate)
    )
    hashed = hashlib.sha3_256(preimage).digest()
    omega = bytes(old ^ new for old, new in zip(state.omega, hashed, strict=True))
    next_state = State(state.side, state.field, state.tick + 1, omega, packet)
    errors = sum(
        int.from_bytes(packet.residual[start:end], "little").bit_count()
        for start, end in _blocks(size)
    )
    receipt: dict[str, int | str] = {
        "side": state.side,
        "packed_bytes": size,
        "input_tick": state.tick,
        "gate_mask": gate,
        "pre_correction_errors": errors,
        "changed_bits": 0,
        "omega_in": state.omega.hex(),
        "omega_out": omega.hex(),
    }
    receipt.update({key: value.hex() for key, value in zip(names, digests, strict=True)})
    return next_state, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, choices=SIDES, default=32)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=SEED)
    parser.add_argument("--ticks", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.ticks <= MAX_TICK:
        parser.error("--ticks must be a positive unsigned 64-bit count")
    try:
        state = State(args.side, seeded_field(args.side, args.seed))
    except ValueError as exc:
        parser.error(str(exc))
    for _ in range(args.ticks):
        state, receipt = commit(state, encode(state.side, state.field))
        print(json.dumps({"seed": args.seed, **receipt}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
