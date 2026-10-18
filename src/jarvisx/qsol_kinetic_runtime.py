from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from itertools import product
from typing import Any

import numpy as np


class Opcode(IntEnum):
    DMA_BURST_ST = 0x10
    DMA_BURST_LD = 0x11
    DMA_DESC = 0x12
    BARRIER = 0x18
    ENC_3D_VOXEL = 0x20
    DEC_3D_MESH = 0x21
    FIX_POINT_CHK = 0x30
    RECON_ERROR = 0x31
    VERIFY = 0x32
    COMMIT = 0x33
    ROLLBACK = 0x34
    STATE_HASH = 0x35
    REFINE_NORM = 0x40
    LAPLACE_3D = 0x41
    AUTO_ENCODE = 0x50
    LATENT_LOOP = 0x60
    HALT = 0xFF


@dataclass(frozen=True)
class PackedInstruction:
    """Canonical QSOL 64-bit instruction word.

    Layout: opcode[63:56] | flags[55:48] | regdst[47:40] | payload[39:0].
    """

    opcode: int
    flags: int = 0
    regdst: int = 0
    payload: int = 0

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("opcode", self.opcode, 0xFF),
            ("flags", self.flags, 0xFF),
            ("regdst", self.regdst, 0xFF),
            ("payload", self.payload, (1 << 40) - 1),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= limit:
                raise ValueError(f"{name} is outside its encoded range")

    def pack(self) -> int:
        return (
            (self.opcode << 56)
            | (self.flags << 48)
            | (self.regdst << 40)
            | self.payload
        )

    @classmethod
    def unpack(cls, word: int) -> "PackedInstruction":
        if not isinstance(word, int) or isinstance(word, bool) or not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("instruction word must be an unsigned 64-bit integer")
        return cls(
            opcode=(word >> 56) & 0xFF,
            flags=(word >> 48) & 0xFF,
            regdst=(word >> 40) & 0xFF,
            payload=word & ((1 << 40) - 1),
        )

    @staticmethod
    def spatial_payload(x: int, y: int, z: int, immediate: int = 0) -> int:
        for name, value in (("x", x), ("y", y), ("z", z), ("immediate", immediate)):
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 1024:
                raise ValueError(f"{name} must fit in 10 bits")
        return (x << 30) | (y << 20) | (z << 10) | immediate

    @staticmethod
    def unpack_spatial_payload(payload: int) -> tuple[int, int, int, int]:
        if not isinstance(payload, int) or isinstance(payload, bool) or not 0 <= payload < (1 << 40):
            raise ValueError("payload must fit in 40 bits")
        return (
            (payload >> 30) & 0x3FF,
            (payload >> 20) & 0x3FF,
            (payload >> 10) & 0x3FF,
            payload & 0x3FF,
        )


@dataclass(frozen=True)
class DMADescriptor:
    channel: int
    origin: tuple[int, int, int]
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.channel, int) or isinstance(self.channel, bool) or not 0 <= self.channel < 8:
            raise ValueError("channel must be in [0, 7]")
        for name, values in (("origin", self.origin), ("shape", self.shape)):
            if len(values) != 3:
                raise ValueError(f"{name} must have three dimensions")
            for value in values:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"{name} entries must be integers")
        if any(value < 0 for value in self.origin):
            raise ValueError("origin cannot contain negative coordinates")
        if any(value < 1 for value in self.shape):
            raise ValueError("shape dimensions must be positive")


@dataclass(frozen=True)
class KineticConfig:
    latent_dim: int = 64
    dt: float = 0.20
    source_gain: float = 1.00
    nonlinear_coupling: float = 0.20
    damping: float = 0.35
    memory_gain: float = 0.08
    memory_decay: float = 0.90
    velocity_limit: float = 2.0
    latent_tolerance: float = 1.0e-3
    regression_tolerance: float = 1.0e-12
    max_cycles: int = 256
    major_phase_rate: float = 0.05
    micro_phase_rate: float = 0.50

    def __post_init__(self) -> None:
        if not isinstance(self.latent_dim, int) or isinstance(self.latent_dim, bool) or self.latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        if not isinstance(self.max_cycles, int) or isinstance(self.max_cycles, bool) or self.max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")
        for name in (
            "dt",
            "source_gain",
            "damping",
            "velocity_limit",
            "latent_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "nonlinear_coupling",
            "memory_gain",
            "major_phase_rate",
            "micro_phase_rate",
            "regression_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self.memory_decay < 1.0:
            raise ValueError("memory_decay must be in [0, 1)")
        if self.nonlinear_coupling >= 1.0:
            raise ValueError("nonlinear_coupling must be < 1 to retain a source-anchored fixed point")


@dataclass(frozen=True)
class KineticReceipt:
    cycle: int
    committed: bool
    converged: bool
    candidate_mse: float
    authoritative_mse: float
    latent_rms: float
    velocity_rms: float
    bytes_transferred: int
    major_phase: float
    micro_phase: float
    state_hash: str


@dataclass(frozen=True)
class KineticResult:
    reconstruction: np.ndarray
    latent: np.ndarray
    velocity: np.ndarray
    memory: np.ndarray
    receipts: tuple[KineticReceipt, ...]
    converged: bool
    cycles: int
    bytes_transferred: int


class DMAFabric:
    """Deterministic software reference for an eight-channel DMA fabric.

    The reference deliberately performs a NumPy copy. It models transfer
    semantics and byte accounting; it does not claim zero-copy or hardware
    bandwidth.
    """

    def __init__(self, channels: int = 8) -> None:
        if not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 8:
            raise ValueError("channels must be in [1, 8]")
        self.channels = channels
        self.bytes_by_channel = [0 for _ in range(channels)]

    @property
    def bytes_transferred(self) -> int:
        return sum(self.bytes_by_channel)

    def read(self, volume: np.ndarray, descriptor: DMADescriptor) -> np.ndarray:
        if descriptor.channel >= self.channels:
            raise ValueError("descriptor targets an unavailable DMA channel")
        if volume.ndim != 3:
            raise ValueError("DMA source must be a three-dimensional volume")
        ox, oy, oz = descriptor.origin
        sx, sy, sz = descriptor.shape
        ex, ey, ez = ox + sx, oy + sy, oz + sz
        if ex > volume.shape[0] or ey > volume.shape[1] or ez > volume.shape[2]:
            raise ValueError("DMA descriptor exceeds source volume bounds")
        block = volume[ox:ex, oy:ey, oz:ez].copy()
        self.bytes_by_channel[descriptor.channel] += int(block.nbytes)
        return block


class QSOLKineticRuntime:
    """Source-anchored kinetic 3D autoencoding reference runtime.

    Operational loop:

        DMA -> encode -> kinetic update -> decode -> residual -> verify
            -> commit/rollback -> feedback -> repeat

    Candidate dynamics are second order in latent space and remain anchored to
    the immutable source encoding z0. A candidate is authoritative only when
    reconstruction error does not regress against the last committed state.
    """

    def __init__(self, config: KineticConfig | None = None) -> None:
        self.config = config or KineticConfig()
        self.dma = DMAFabric()
        self._basis_cache: dict[tuple[tuple[int, int, int], int], np.ndarray] = {}

    @staticmethod
    def _normalize(volume: np.ndarray) -> np.ndarray:
        if volume.dtype != np.uint8:
            raise TypeError("QSOL voxel volumes must use uint8 storage")
        if volume.ndim != 3:
            raise ValueError("voxel volume must be three-dimensional")
        return volume.astype(np.float64) / 127.5 - 1.0

    @staticmethod
    def _denormalize(field: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        clipped = np.clip(field, -1.0, 1.0)
        return np.rint((clipped.reshape(shape) + 1.0) * 127.5).astype(np.uint8)

    @staticmethod
    def _axis_basis(length: int) -> np.ndarray:
        n = np.arange(length, dtype=np.float64)
        k = np.arange(length, dtype=np.float64)[:, None]
        basis = np.cos(math.pi * (n[None, :] + 0.5) * k / length)
        basis[0, :] *= math.sqrt(1.0 / length)
        if length > 1:
            basis[1:, :] *= math.sqrt(2.0 / length)
        return basis

    def _basis(self, shape: tuple[int, int, int], latent_dim: int) -> np.ndarray:
        node_count = int(np.prod(shape))
        if latent_dim > node_count:
            raise ValueError("latent_dim cannot exceed the voxel count of the DMA block")
        key = (shape, latent_dim)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached

        bx = self._axis_basis(shape[0])
        by = self._axis_basis(shape[1])
        bz = self._axis_basis(shape[2])
        frequencies = list(product(range(shape[0]), range(shape[1]), range(shape[2])))
        frequencies.sort(key=lambda item: (sum(item), max(item), item[0], item[1], item[2]))

        rows: list[np.ndarray] = []
        for kx, ky, kz in frequencies[:latent_dim]:
            tensor = np.einsum("i,j,k->ijk", bx[kx], by[ky], bz[kz])
            rows.append(tensor.reshape(-1))
        basis = np.stack(rows, axis=0)
        self._basis_cache[key] = basis
        return basis

    @staticmethod
    def _state_hash(reconstruction: np.ndarray, cycle: int, mse: float, latent_rms: float) -> str:
        metadata = json.dumps(
            {"cycle": cycle, "mse": mse, "latent_rms": latent_rms},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(metadata)
        digest.update(reconstruction.tobytes(order="C"))
        return digest.hexdigest()

    def run(
        self,
        volume: np.ndarray,
        *,
        descriptor: DMADescriptor | None = None,
    ) -> KineticResult:
        source = np.asarray(volume)
        if source.dtype != np.uint8:
            raise TypeError("volume must be a uint8 array")
        if source.ndim != 3:
            raise ValueError("volume must be three-dimensional")

        resolved_descriptor = descriptor or DMADescriptor(
            channel=0,
            origin=(0, 0, 0),
            shape=tuple(int(value) for value in source.shape),
        )
        block = self.dma.read(source, resolved_descriptor)
        normalized = self._normalize(block)
        flat_source = normalized.reshape(-1)
        basis = self._basis(tuple(int(value) for value in block.shape), self.config.latent_dim)
        z0 = basis @ flat_source

        latent = np.zeros_like(z0)
        velocity = np.zeros_like(z0)
        memory = np.zeros_like(z0)

        initial_field = basis.T @ latent
        authoritative_mse = float(np.mean((flat_source - initial_field) ** 2))
        authoritative_latent = latent.copy()
        authoritative_reconstruction = self._denormalize(initial_field, block.shape)

        receipts: list[KineticReceipt] = []
        major_phase = 0.0
        micro_phase = 0.0
        converged = False

        for cycle in range(1, self.config.max_cycles + 1):
            error = latent - z0
            desired = z0 + self.config.nonlinear_coupling * np.tanh(error)
            acceleration = (
                self.config.source_gain * (desired - latent)
                - self.config.damping * velocity
                - self.config.memory_gain * memory
            )
            candidate_velocity = np.clip(
                velocity + self.config.dt * acceleration,
                -self.config.velocity_limit,
                self.config.velocity_limit,
            )
            candidate_latent = latent + self.config.dt * candidate_velocity
            candidate_memory = (
                self.config.memory_decay * memory
                + (1.0 - self.config.memory_decay) * (candidate_latent - z0)
            )

            candidate_field = basis.T @ candidate_latent
            candidate_mse = float(np.mean((flat_source - candidate_field) ** 2))
            latent_rms = float(np.sqrt(np.mean((candidate_latent - z0) ** 2)))
            velocity_rms = float(np.sqrt(np.mean(candidate_velocity**2)))

            finite = all(
                np.isfinite(value)
                for value in (candidate_mse, latent_rms, velocity_rms)
            )
            committed = bool(
                finite
                and candidate_mse
                <= authoritative_mse + self.config.regression_tolerance
            )

            if committed:
                latent = candidate_latent
                velocity = candidate_velocity
                memory = candidate_memory
                authoritative_latent = latent.copy()
                authoritative_mse = candidate_mse
                authoritative_reconstruction = self._denormalize(candidate_field, block.shape)
            else:
                latent = authoritative_latent.copy()
                velocity = -0.25 * velocity
                memory = 0.5 * memory

            major_phase = math.fmod(major_phase + self.config.major_phase_rate, 2.0 * math.pi)
            micro_phase = math.fmod(micro_phase + self.config.micro_phase_rate, 2.0 * math.pi)
            converged = bool(committed and latent_rms <= self.config.latent_tolerance)

            receipt = KineticReceipt(
                cycle=cycle,
                committed=committed,
                converged=converged,
                candidate_mse=candidate_mse,
                authoritative_mse=authoritative_mse,
                latent_rms=latent_rms,
                velocity_rms=velocity_rms,
                bytes_transferred=self.dma.bytes_transferred,
                major_phase=major_phase,
                micro_phase=micro_phase,
                state_hash=self._state_hash(
                    authoritative_reconstruction,
                    cycle,
                    authoritative_mse,
                    latent_rms,
                ),
            )
            receipts.append(receipt)
            if converged:
                break

        return KineticResult(
            reconstruction=authoritative_reconstruction,
            latent=latent.copy(),
            velocity=velocity.copy(),
            memory=memory.copy(),
            receipts=tuple(receipts),
            converged=converged,
            cycles=len(receipts),
            bytes_transferred=self.dma.bytes_transferred,
        )


__all__ = [
    "DMADescriptor",
    "DMAFabric",
    "KineticConfig",
    "KineticReceipt",
    "KineticResult",
    "Opcode",
    "PackedInstruction",
    "QSOLKineticRuntime",
]
