"""Bit-packed 3D VME reference layer for the Dr Moagi geometric ANN.

The logical domain is 4000^3 = 64,000,000,000 voxels. Only active sparse
cells are materialized. This module therefore distinguishes logical extent,
physical residency, and measured execution cost.

Voxel control word (64 bits):
    x:12 | y:12 | z:12 | value:4 | modality:2 | residual:8 | flags:14

Latent recurrence uses bounded signed INT8/Q7-like arithmetic. The module is
a deterministic software reference, not a hardware-throughput claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping

LOGICAL_SIDE = 4000
LOGICAL_CELLS = LOGICAL_SIDE ** 3
ONE_GIB_BYTES = 1 << 30
ONE_GIB_BITS = ONE_GIB_BYTES * 8


class Modality(IntEnum):
    MESH = 0
    AUDIO = 1
    VIDEO = 2
    TEXT = 3


class Topology(IntEnum):
    TORUS = 0
    KLEIN = 1
    SPHERE = 2
    RESERVED = 3


class BitOpcode(IntEnum):
    NOP = 0x00
    LOAD_VOX = 0x01
    STORE_VOX = 0x02
    SHIFT_XP = 0x10
    SHIFT_XM = 0x11
    SHIFT_YP = 0x12
    SHIFT_YM = 0x13
    SHIFT_ZP = 0x14
    SHIFT_ZM = 0x15
    DIFFUSE6 = 0x20
    LAPLACE3D = 0x21
    ENCODE_INT8 = 0x30
    GATE = 0x31
    RECURRENT = 0x32
    TOPOLOGY = 0x40
    GEOM_MAP = 0x41
    SONIFY = 0x50
    FEEDBACK = 0x60
    OMEGA = 0x61
    RECUR = 0x62
    HALT = 0xFF


@dataclass(frozen=True)
class Instruction64:
    opcode: BitOpcode
    dst: int = 0
    src_a: int = 0
    src_b: int = 0
    immediate: int = 0

    def pack(self) -> int:
        for name, value in (("dst", self.dst), ("src_a", self.src_a), ("src_b", self.src_b)):
            if not 0 <= value <= 0xFF:
                raise ValueError(f"{name} must fit in 8 bits")
        if not 0 <= self.immediate <= 0xFFFFFFFF:
            raise ValueError("immediate must fit in 32 bits")
        return (
            (int(self.opcode) << 56)
            | (self.dst << 48)
            | (self.src_a << 40)
            | (self.src_b << 32)
            | self.immediate
        )

    @classmethod
    def unpack(cls, word: int) -> "Instruction64":
        if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("word must fit in 64 bits")
        return cls(
            opcode=BitOpcode((word >> 56) & 0xFF),
            dst=(word >> 48) & 0xFF,
            src_a=(word >> 40) & 0xFF,
            src_b=(word >> 32) & 0xFF,
            immediate=word & 0xFFFFFFFF,
        )


@dataclass(frozen=True)
class VoxelWord:
    x: int
    y: int
    z: int
    value: int
    modality: Modality
    residual: int = 0
    flags: int = 0

    def pack(self) -> int:
        for name, value in (("x", self.x), ("y", self.y), ("z", self.z)):
            if not 0 <= value < LOGICAL_SIDE:
                raise ValueError(f"{name} must be in [0, {LOGICAL_SIDE})")
        if not 0 <= self.value <= 0xF:
            raise ValueError("value must fit in 4 bits")
        if not 0 <= self.residual <= 0xFF:
            raise ValueError("residual must fit in 8 bits")
        if not 0 <= self.flags <= 0x3FFF:
            raise ValueError("flags must fit in 14 bits")
        return (
            (self.x << 52)
            | (self.y << 40)
            | (self.z << 28)
            | (self.value << 24)
            | (int(self.modality) << 22)
            | (self.residual << 14)
            | self.flags
        )

    @classmethod
    def unpack(cls, word: int) -> "VoxelWord":
        if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("word must fit in 64 bits")
        return cls(
            x=(word >> 52) & 0xFFF,
            y=(word >> 40) & 0xFFF,
            z=(word >> 28) & 0xFFF,
            value=(word >> 24) & 0xF,
            modality=Modality((word >> 22) & 0x3),
            residual=(word >> 14) & 0xFF,
            flags=word & 0x3FFF,
        )


def morton3d(x: int, y: int, z: int) -> int:
    """Interleave 12 bits from x/y/z into a 36-bit Morton address."""
    for axis in (x, y, z):
        if not 0 <= axis < LOGICAL_SIDE:
            raise ValueError("coordinate outside logical 4000^3 domain")
    code = 0
    for bit in range(12):
        code |= ((x >> bit) & 1) << (3 * bit)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit + 2)
    return code


def unmorton3d(code: int) -> tuple[int, int, int]:
    if not 0 <= code < (1 << 36):
        raise ValueError("Morton code must fit in 36 bits")
    x = y = z = 0
    for bit in range(12):
        x |= ((code >> (3 * bit)) & 1) << bit
        y |= ((code >> (3 * bit + 1)) & 1) << bit
        z |= ((code >> (3 * bit + 2)) & 1) << bit
    return x, y, z


def sat_u4(value: int) -> int:
    return max(0, min(15, int(value)))


def sat_i8(value: int) -> int:
    return max(-128, min(127, int(value)))


def tanh_q7(value: int) -> int:
    """Integer saturating tanh-like nonlinearity in signed Q7 range."""
    sign = -1 if value < 0 else 1
    magnitude = abs(int(value))
    mapped = (magnitude * 127) // (127 + magnitude)
    return sat_i8(sign * mapped)


@dataclass(frozen=True)
class BitwiseVMEConfig:
    latent_dim: int = 8
    diffusion_alpha_q8: int = 46  # 46/256 ~= 0.18
    max_refine_steps: int = 32
    fixed_point_l1_tolerance: int = 1

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not 0 <= self.diffusion_alpha_q8 <= 255:
            raise ValueError("diffusion_alpha_q8 must be in [0,255]")
        if self.max_refine_steps <= 0:
            raise ValueError("max_refine_steps must be positive")
        if self.fixed_point_l1_tolerance < 0:
            raise ValueError("fixed_point_l1_tolerance must be non-negative")


@dataclass(frozen=True)
class BitwiseCycleReceipt:
    active_cells: int
    logical_cells: int
    physical_bytes: int
    latent: tuple[int, ...]
    fixed_point_l1: int
    physical_refine_steps: int
    converged: bool


class BitwiseVME3D:
    """Sparse active-set VME over a 4000^3 logical domain."""

    def __init__(self, config: BitwiseVMEConfig | None = None) -> None:
        self.config = config or BitwiseVMEConfig()
        self.active: dict[int, int] = {}
        self.latent = [0] * self.config.latent_dim
        self.omega = [0] * self.config.latent_dim
        self.topology = Topology.TORUS

    def load(self, cells: Mapping[tuple[int, int, int], int]) -> None:
        active: dict[int, int] = {}
        for (x, y, z), raw in cells.items():
            value = sat_u4(raw)
            if value:
                active[morton3d(x, y, z)] = value
        self.active = active

    @staticmethod
    def _neighbors(x: int, y: int, z: int) -> tuple[tuple[int, int, int], ...]:
        n = LOGICAL_SIDE
        return (
            ((x + 1) % n, y, z),
            ((x - 1) % n, y, z),
            (x, (y + 1) % n, z),
            (x, (y - 1) % n, z),
            (x, y, (z + 1) % n),
            (x, y, (z - 1) % n),
        )

    def diffuse6(self) -> None:
        alpha = self.config.diffusion_alpha_q8
        keep = 256 - alpha
        nxt: dict[int, int] = {}
        for address, value in self.active.items():
            x, y, z = unmorton3d(address)
            neighbor_sum = sum(
                self.active.get(morton3d(nx, ny, nz), 0)
                for nx, ny, nz in self._neighbors(x, y, z)
            )
            neighbor_mean = neighbor_sum // 6
            updated = sat_u4((keep * value + alpha * neighbor_mean) >> 8)
            if updated:
                nxt[address] = updated
        self.active = nxt

    def encode_int8(self) -> list[int]:
        if not self.active:
            return [0] * self.config.latent_dim
        items = sorted(self.active.items())
        count = len(items)
        total = sum(value for _, value in items)
        maximum = max(value for _, value in items)
        minimum = min(value for _, value in items)
        base = [
            sat_i8((total * 8) // count - 64),
            sat_i8(maximum * 8 - 64),
            sat_i8(minimum * 8 - 64),
            sat_i8((count.bit_length() - 1) * 8),
        ]
        checksum = 0
        for address, value in items:
            checksum ^= (address ^ value) & 0x7F
        base.append(sat_i8(checksum))
        while len(base) < self.config.latent_dim:
            j = len(base)
            base.append(sat_i8((base[j % 5] * (j + 3)) >> 2))
        return base[: self.config.latent_dim]

    def _latent_step(self, current: list[int], seed: list[int]) -> list[int]:
        result: list[int] = []
        d = self.config.latent_dim
        for j in range(d):
            accum = (
                96 * current[j]
                + 48 * current[(j + 1) % d]
                + 40 * seed[j]
                + 32 * self.omega[j]
            ) >> 7
            proposal = tanh_q7(accum)
            mixed = (3 * current[j] + proposal) >> 2
            result.append(sat_i8(mixed))
        return result

    def refine(self, seed: list[int]) -> tuple[list[int], int, int, bool]:
        current = seed[:]
        residual = 0
        converged = False
        steps = 0
        for steps in range(1, self.config.max_refine_steps + 1):
            nxt = self._latent_step(current, seed)
            residual = sum(abs(a - b) for a, b in zip(nxt, current))
            current = nxt
            if residual <= self.config.fixed_point_l1_tolerance:
                converged = True
                break
        return current, steps, residual, converged

    def update_omega(self, latent: list[int]) -> None:
        self.omega = [
            sat_i8((7 * old + new) >> 3)
            for old, new in zip(self.omega, latent)
        ]

    def feedback(self, latent: list[int]) -> None:
        if not self.active:
            return
        bias = sum(latent) // max(1, len(latent))
        updated: dict[int, int] = {}
        for address, value in self.active.items():
            corrected = sat_u4(value + (bias >> 5))
            if corrected:
                updated[address] = corrected
        self.active = updated

    def run_cycle(self) -> BitwiseCycleReceipt:
        self.diffuse6()
        seed = self.encode_int8()
        latent, steps, residual, converged = self.refine(seed)
        self.latent = latent
        self.update_omega(latent)
        self.feedback(latent)
        physical_bytes = len(self.active) * 8
        return BitwiseCycleReceipt(
            active_cells=len(self.active),
            logical_cells=LOGICAL_CELLS,
            physical_bytes=physical_bytes,
            latent=tuple(latent),
            fixed_point_l1=residual,
            physical_refine_steps=steps,
            converged=converged,
        )


def dense_capacity_for_one_gib(bits_per_voxel: int) -> int:
    if bits_per_voxel <= 0:
        raise ValueError("bits_per_voxel must be positive")
    return ONE_GIB_BITS // bits_per_voxel


__all__ = [
    "BitOpcode",
    "BitwiseCycleReceipt",
    "BitwiseVME3D",
    "BitwiseVMEConfig",
    "Instruction64",
    "LOGICAL_CELLS",
    "LOGICAL_SIDE",
    "Modality",
    "ONE_GIB_BITS",
    "ONE_GIB_BYTES",
    "Topology",
    "VoxelWord",
    "dense_capacity_for_one_gib",
    "morton3d",
    "sat_i8",
    "sat_u4",
    "tanh_q7",
    "unmorton3d",
]
