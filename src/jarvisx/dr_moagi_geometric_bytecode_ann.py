"""Dr Moagi 3D geometric bytecode ANN reference runtime.

This module implements a deterministic, bounded software abstraction of the
3D inward autoencoding/reconstruction loop. Astronomical recursion counts are
represented as logical metadata; the runtime never claims to physically execute
those counts. Physical execution is bounded by max_refine_steps and may collapse
deep logical recurrence into a converged fixed-point approximation.

Canonical operation:
    diffuse -> encode -> fixed-point refine -> memory -> decode
    -> residual -> correct -> geometric map -> recur

The module is a reference laboratory for ADR-017 semantics. It does not by
itself establish trained-model quality, hardware throughput, global convergence,
or external-world correctness.
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Sequence


SEPTILLION = 10**24
LOGICAL_RECURSION_EXPRESSION = "Q^(Q^Q), Q=10^24"


class Op(IntEnum):
    INGEST = 0x01
    DIFFUSE6 = 0x02
    ENCODE = 0x03
    LATENT_REFINE = 0x04
    MEMORY = 0x05
    DECODE = 0x06
    RESIDUAL = 0x07
    CORRECT = 0x08
    ADAPT = 0x09
    SCHEDULE = 0x0A
    RECUR = 0x0B
    GEOM_MAP = 0x0C
    HALT = 0xFF


@dataclass(frozen=True)
class Instruction:
    opcode: Op
    a: int = 0
    b: int = 0
    c: int = 0
    immediate: int = 0

    def pack(self) -> int:
        for name, value in (("a", self.a), ("b", self.b), ("c", self.c)):
            if not 0 <= value <= 0xFF:
                raise ValueError(f"{name} must fit in 8 bits")
        if not 0 <= self.immediate <= 0xFFFFFFFF:
            raise ValueError("immediate must fit in 32 bits")
        return (
            (int(self.opcode) << 56)
            | (self.a << 48)
            | (self.b << 40)
            | (self.c << 32)
            | self.immediate
        )

    @classmethod
    def unpack(cls, word: int) -> "Instruction":
        if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("instruction word must fit in 64 bits")
        return cls(
            Op((word >> 56) & 0xFF),
            (word >> 48) & 0xFF,
            (word >> 40) & 0xFF,
            (word >> 32) & 0xFF,
            word & 0xFFFFFFFF,
        )


def _q16(value: float) -> int:
    if not 0.0 <= value <= 1.0:
        raise ValueError("Q16 scalar must be in [0, 1]")
    return min(0xFFFF, int(round(value * 0xFFFF)))


def _from_q16(value: int) -> float:
    return (value & 0xFFFF) / 0xFFFF


@dataclass(frozen=True)
class ANNConfig:
    side: int = 4
    channels: int = 2
    latent_dim: int = 8
    diffusion_alpha: float = 0.18
    latent_relaxation: float = 0.55
    memory_rho: float = 0.88
    correction_gamma: float = 0.65
    learning_rate: float = 0.001
    fixed_point_tolerance: float = 1.0e-6
    max_refine_steps: int = 64
    active_threshold: float = 1.0e-8
    seed: int = 7

    def __post_init__(self) -> None:
        if self.side <= 1:
            raise ValueError("side must be greater than 1")
        if self.channels <= 0 or self.latent_dim <= 0:
            raise ValueError("channels and latent_dim must be positive")
        if not 0.0 <= self.diffusion_alpha <= 1.0:
            raise ValueError("diffusion_alpha must be in [0, 1]")
        if not 0.0 < self.latent_relaxation <= 1.0:
            raise ValueError("latent_relaxation must be in (0, 1]")
        if not 0.0 <= self.memory_rho < 1.0:
            raise ValueError("memory_rho must be in [0, 1)")
        if not 0.0 <= self.correction_gamma <= 1.0:
            raise ValueError("correction_gamma must be in [0, 1]")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative")
        if self.fixed_point_tolerance < 0.0:
            raise ValueError("fixed_point_tolerance must be non-negative")
        if self.max_refine_steps <= 0:
            raise ValueError("max_refine_steps must be positive")


@dataclass
class ANNState:
    x: list[float]
    x_mixed: list[float]
    z: list[float]
    omega: list[float]
    x_hat: list[float]
    residual: list[float]
    corrected: list[float]
    cycle: int = 0
    logical_recursion: str = LOGICAL_RECURSION_EXPRESSION
    fixed_point_residual: float = math.inf
    physical_refine_steps: int = 0
    converged: bool = False


@dataclass(frozen=True)
class GeometricPoint:
    x: float
    y: float
    z: float
    radius: float
    residual_dx: float
    residual_dy: float
    residual_dz: float


@dataclass(frozen=True)
class CycleReceipt:
    cycle: int
    mse: float
    fixed_point_residual: float
    physical_refine_steps: int
    converged: bool
    active_voxels: int
    logical_recursion: str


class GeometricBytecodeANN:
    """Deterministic 3D ANN VM with bounded fixed-point collapse."""

    def __init__(self, config: ANNConfig | None = None) -> None:
        self.config = config or ANNConfig()
        self.voxels = self.config.side**3
        self.input_dim = self.voxels * self.config.channels
        self.rng = random.Random(self.config.seed)
        scale_e = 1.0 / math.sqrt(self.input_dim)
        scale_z = 1.0 / math.sqrt(self.config.latent_dim)
        self.w_enc = [
            [self.rng.uniform(-scale_e, scale_e) for _ in range(self.input_dim)]
            for _ in range(self.config.latent_dim)
        ]
        self.b_enc = [0.0] * self.config.latent_dim
        self.w_latent = self._contractive_latent_matrix(target_row_sum=0.72)
        self.w_dec = [
            [self.rng.uniform(-scale_z, scale_z) for _ in range(self.config.latent_dim)]
            for _ in range(self.input_dim)
        ]
        self.b_dec = [0.0] * self.input_dim
        zero_x = [0.0] * self.input_dim
        zero_z = [0.0] * self.config.latent_dim
        self.state = ANNState(
            x=zero_x[:],
            x_mixed=zero_x[:],
            z=zero_z[:],
            omega=zero_z[:],
            x_hat=zero_x[:],
            residual=zero_x[:],
            corrected=zero_x[:],
        )
        self.geometry: list[GeometricPoint] = []
        self.receipts: list[CycleReceipt] = []
        self.program = self.default_program()
        self.halted = False

    def _contractive_latent_matrix(self, target_row_sum: float) -> list[list[float]]:
        matrix: list[list[float]] = []
        for _ in range(self.config.latent_dim):
            row = [self.rng.uniform(-1.0, 1.0) for _ in range(self.config.latent_dim)]
            norm = sum(abs(value) for value in row) or 1.0
            matrix.append([target_row_sum * value / norm for value in row])
        return matrix

    def default_program(self) -> tuple[Instruction, ...]:
        return (
            Instruction(Op.INGEST),
            Instruction(Op.DIFFUSE6, immediate=_q16(self.config.diffusion_alpha)),
            Instruction(Op.ENCODE),
            Instruction(Op.LATENT_REFINE),
            Instruction(Op.MEMORY, immediate=_q16(self.config.memory_rho)),
            Instruction(Op.DECODE),
            Instruction(Op.RESIDUAL),
            Instruction(Op.SCHEDULE),
            Instruction(Op.CORRECT, immediate=_q16(self.config.correction_gamma)),
            Instruction(Op.ADAPT),
            Instruction(Op.GEOM_MAP),
            Instruction(Op.RECUR),
            Instruction(Op.HALT),
        )

    def load(self, values: Sequence[float]) -> None:
        if len(values) != self.input_dim:
            raise ValueError(f"expected {self.input_dim} values")
        finite = [float(value) for value in values]
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("input values must be finite")
        self.state.x = finite[:]
        self.state.x_mixed = finite[:]
        self.state.x_hat = [0.0] * self.input_dim
        self.state.residual = [0.0] * self.input_dim
        self.state.corrected = finite[:]
        self.state.cycle = 0
        self.state.fixed_point_residual = math.inf
        self.state.physical_refine_steps = 0
        self.state.converged = False
        self.receipts.clear()
        self.geometry.clear()
        self.halted = False

    def disassemble(self) -> list[str]:
        rows: list[str] = []
        for pc, instruction in enumerate(self.program):
            rows.append(f"{pc:02d} 0x{instruction.pack():016X} {instruction.opcode.name}")
        return rows

    def _index(self, x: int, y: int, z: int, c: int) -> int:
        side = self.config.side
        return (((z * side) + y) * side + x) * self.config.channels + c

    def _neighbor_mean(self, values: Sequence[float], x: int, y: int, z: int, c: int) -> float:
        side = self.config.side
        total = 0.0
        count = 0
        for dx, dy, dz in (
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < side and 0 <= ny < side and 0 <= nz < side:
                total += values[self._index(nx, ny, nz, c)]
                count += 1
        return total / count if count else values[self._index(x, y, z, c)]

    def _diffuse6(self, alpha: float) -> None:
        src = self.state.x
        dst = [0.0] * self.input_dim
        side = self.config.side
        for z in range(side):
            for y in range(side):
                for x in range(side):
                    for c in range(self.config.channels):
                        idx = self._index(x, y, z, c)
                        local = src[idx]
                        mean = self._neighbor_mean(src, x, y, z, c)
                        dst[idx] = (1.0 - alpha) * local + alpha * mean
        self.state.x_mixed = dst

    @staticmethod
    def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
        return [sum(weight * value for weight, value in zip(row, vector)) for row in matrix]

    def _encode(self) -> None:
        projected = self._matvec(self.w_enc, self.state.x_mixed)
        self.state.z = [
            math.tanh(value + bias) for value, bias in zip(projected, self.b_enc)
        ]

    def _latent_step(self, z: Sequence[float]) -> list[float]:
        recurrent = self._matvec(self.w_latent, z)
        relaxation = self.config.latent_relaxation
        proposal = [
            math.tanh(recurrent[i] + 0.15 * self.state.omega[i] + 0.20 * self.state.z[i])
            for i in range(self.config.latent_dim)
        ]
        return [(1.0 - relaxation) * z[i] + relaxation * proposal[i] for i in range(len(z))]

    @staticmethod
    def _l2_delta(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def _refine(self) -> None:
        current = self.state.z[:]
        converged = False
        residual = math.inf
        steps = 0
        for steps in range(1, self.config.max_refine_steps + 1):
            nxt = self._latent_step(current)
            residual = self._l2_delta(nxt, current)
            current = nxt
            if residual <= self.config.fixed_point_tolerance:
                converged = True
                break
        self.state.z = current
        self.state.fixed_point_residual = residual
        self.state.physical_refine_steps = steps
        self.state.converged = converged

    def _memory(self, rho: float) -> None:
        self.state.omega = [
            rho * old + (1.0 - rho) * latent
            for old, latent in zip(self.state.omega, self.state.z)
        ]

    def _decode(self) -> None:
        output: list[float] = []
        for row, bias in zip(self.w_dec, self.b_dec):
            output.append(math.tanh(sum(w * z for w, z in zip(row, self.state.z)) + bias))
        self.state.x_hat = output

    def _residual(self) -> None:
        self.state.residual = [x - y for x, y in zip(self.state.x, self.state.x_hat)]

    def _correct(self, gamma: float) -> None:
        self.state.corrected = [
            estimate + gamma * error
            for estimate, error in zip(self.state.x_hat, self.state.residual)
        ]

    def _adapt(self) -> None:
        # Deterministic bounded decoder-only laboratory update.
        eta = self.config.learning_rate
        if eta == 0.0:
            return
        for i, error in enumerate(self.state.residual):
            bounded_error = max(-1.0, min(1.0, error))
            row = self.w_dec[i]
            for j, latent in enumerate(self.state.z):
                row[j] += eta * bounded_error * latent
                row[j] = max(-1.0, min(1.0, row[j]))
            self.b_dec[i] = max(
                -1.0,
                min(1.0, self.b_dec[i] + eta * bounded_error),
            )

    def _geom_map(self) -> None:
        side = self.config.side
        latent_energy = sum(abs(value) for value in self.state.z) / max(1, len(self.state.z))
        contraction = max(0.15, 1.0 - 0.55 * min(1.0, latent_energy))
        residual_channels = self.config.channels
        points: list[GeometricPoint] = []
        for z in range(side):
            nz = -1.0 + 2.0 * z / (side - 1)
            for y in range(side):
                ny = -1.0 + 2.0 * y / (side - 1)
                for x in range(side):
                    nx = -1.0 + 2.0 * x / (side - 1)
                    base = self._index(x, y, z, 0)
                    values = self.state.x[base : base + residual_channels]
                    errors = self.state.residual[base : base + residual_channels]
                    energy = math.sqrt(sum(value * value for value in values))
                    padded = list(errors[:3]) + [0.0, 0.0, 0.0]
                    points.append(
                        GeometricPoint(
                            x=contraction * nx,
                            y=contraction * ny,
                            z=contraction * nz,
                            radius=1.0 + energy,
                            residual_dx=padded[0],
                            residual_dy=padded[1],
                            residual_dz=padded[2],
                        )
                    )
        self.geometry = points

    def _mse(self) -> float:
        return sum(error * error for error in self.state.residual) / len(self.state.residual)

    def _record_receipt(self) -> None:
        active = 0
        for voxel in range(self.voxels):
            start = voxel * self.config.channels
            stop = start + self.config.channels
            if any(abs(value) > self.config.active_threshold for value in self.state.x[start:stop]):
                active += 1
        self.receipts.append(
            CycleReceipt(
                cycle=self.state.cycle,
                mse=self._mse(),
                fixed_point_residual=self.state.fixed_point_residual,
                physical_refine_steps=self.state.physical_refine_steps,
                converged=self.state.converged,
                active_voxels=active,
                logical_recursion=self.state.logical_recursion,
            )
        )

    def step_instruction(self, instruction: Instruction) -> None:
        op = instruction.opcode
        if op is Op.INGEST:
            return
        if op is Op.DIFFUSE6:
            self._diffuse6(_from_q16(instruction.immediate))
        elif op is Op.ENCODE:
            self._encode()
        elif op is Op.LATENT_REFINE:
            self._refine()
        elif op is Op.MEMORY:
            self._memory(_from_q16(instruction.immediate))
        elif op is Op.DECODE:
            self._decode()
        elif op is Op.RESIDUAL:
            self._residual()
        elif op is Op.SCHEDULE:
            # Scheduling is intentionally bounded. Deep logical recursion is not
            # converted into an impossible physical loop count.
            pass
        elif op is Op.CORRECT:
            self._correct(_from_q16(instruction.immediate))
        elif op is Op.ADAPT:
            self._adapt()
        elif op is Op.GEOM_MAP:
            self._geom_map()
        elif op is Op.RECUR:
            self.state.x = self.state.corrected[:]
            self.state.cycle += 1
            self._record_receipt()
        elif op is Op.HALT:
            self.halted = True
        else:
            raise ValueError(f"unsupported opcode: {op}")

    def run_cycle(self) -> CycleReceipt:
        self.halted = False
        for instruction in self.program:
            self.step_instruction(instruction)
            if self.halted:
                break
        if not self.receipts:
            raise RuntimeError("program did not execute RECUR")
        return self.receipts[-1]

    def run(self, cycles: int) -> tuple[CycleReceipt, ...]:
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        return tuple(self.run_cycle() for _ in range(cycles))


def synthetic_field(config: ANNConfig, value: float = 0.5) -> list[float]:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return [float(value)] * (config.side**3 * config.channels)


def encode_program_binary(program: Iterable[Instruction]) -> bytes:
    return b"".join(struct.pack(">Q", instruction.pack()) for instruction in program)


def decode_program_binary(payload: bytes) -> tuple[Instruction, ...]:
    if len(payload) % 8 != 0:
        raise ValueError("bytecode payload length must be a multiple of 8")
    return tuple(
        Instruction.unpack(struct.unpack(">Q", payload[offset : offset + 8])[0])
        for offset in range(0, len(payload), 8)
    )


__all__ = [
    "ANNConfig",
    "ANNState",
    "CycleReceipt",
    "GeometricBytecodeANN",
    "GeometricPoint",
    "Instruction",
    "LOGICAL_RECURSION_EXPRESSION",
    "Op",
    "SEPTILLION",
    "decode_program_binary",
    "encode_program_binary",
    "synthetic_field",
]
