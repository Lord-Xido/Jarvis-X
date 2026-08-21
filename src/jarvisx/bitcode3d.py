from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from hashlib import sha256
from math import isfinite
from struct import pack
from time import perf_counter
from typing import Sequence

Q_FRAC_BITS = 16
Q_SCALE = 1 << Q_FRAC_BITS
Q_MIN = -(1 << 31)
Q_MAX = (1 << 31) - 1
MAX_VOXELS = 1_048_576

Shape3D = tuple[int, int, int]


class Opcode(IntEnum):
    """Compact 32-bit edge/lowering ISA opcodes for the 3D runtime."""

    NOP = 0x00
    NET_RX = 0x10
    HOST_STAGE = 0x11
    Q16_CONVERT = 0x20
    PACK3D = 0x21
    ENCODE3D = 0x30
    LATENT_WRITE = 0x31
    DECODE3D = 0x40
    VERIFY = 0x50
    TELEMETRY = 0x60
    EMIT = 0x70
    HALT = 0xFF


@dataclass(frozen=True)
class Instruction32:
    """32-bit instruction: opcode[31:24], dst[23:20], src1[19:16], src2[15:12], imm[11:0]."""

    opcode: Opcode
    dst: int = 0
    src1: int = 0
    src2: int = 0
    imm: int = 0

    def encode(self) -> int:
        for name, value in (("dst", self.dst), ("src1", self.src1), ("src2", self.src2)):
            if not 0 <= value <= 0xF:
                raise ValueError(f"{name} must fit in 4 bits")
        if not -2048 <= self.imm <= 2047:
            raise ValueError("imm must fit in signed 12 bits")
        return (
            (int(self.opcode) << 24)
            | (self.dst << 20)
            | (self.src1 << 16)
            | (self.src2 << 12)
            | (self.imm & 0xFFF)
        )

    @classmethod
    def decode(cls, word: int) -> "Instruction32":
        if not 0 <= word <= 0xFFFFFFFF:
            raise ValueError("word must fit in unsigned 32 bits")
        opcode_raw = (word >> 24) & 0xFF
        imm_raw = word & 0xFFF
        imm = imm_raw - 0x1000 if imm_raw & 0x800 else imm_raw
        try:
            opcode = Opcode(opcode_raw)
        except ValueError as exc:
            raise ValueError(f"unknown opcode 0x{opcode_raw:02X}") from exc
        return cls(
            opcode=opcode,
            dst=(word >> 20) & 0xF,
            src1=(word >> 16) & 0xF,
            src2=(word >> 12) & 0xF,
            imm=imm,
        )


@dataclass(frozen=True)
class SpatialInstruction:
    word: int
    x: int
    y: int
    z: int
    stage: str

    @property
    def instruction(self) -> Instruction32:
        return Instruction32.decode(self.word)

    def as_payload(self) -> dict[str, object]:
        decoded = self.instruction
        return {
            "word": self.word,
            "hex": f"0x{self.word:08X}",
            "opcode": decoded.opcode.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class Verification:
    mse: float
    max_abs_error: float
    tolerance: float
    passed: bool
    checksum_sha256: str

    def as_payload(self) -> dict[str, object]:
        return {
            "mse": self.mse,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass(frozen=True)
class Telemetry:
    cycles: int
    active_cells: int
    latent_cells: int
    clipped_values: int
    elapsed_ms: float
    stages: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "active_cells": self.active_cells,
            "latent_cells": self.latent_cells,
            "clipped_values": self.clipped_values,
            "elapsed_ms": self.elapsed_ms,
            "stages": list(self.stages),
        }


@dataclass(frozen=True)
class BitCode3DResult:
    input_shape: Shape3D
    latent_shape: Shape3D
    latent: tuple[float, ...]
    reconstructed: tuple[float, ...]
    bytecode: tuple[int, ...]
    spatial_program: tuple[SpatialInstruction, ...]
    verification: Verification
    telemetry: Telemetry

    def as_payload(self) -> dict[str, object]:
        return {
            "input_shape": list(self.input_shape),
            "latent_shape": list(self.latent_shape),
            "latent": list(self.latent),
            "reconstructed": list(self.reconstructed),
            "bytecode": [f"0x{word:08X}" for word in self.bytecode],
            "spatial_program": [item.as_payload() for item in self.spatial_program],
            "verification": self.verification.as_payload(),
            "telemetry": self.telemetry.as_payload(),
        }


@dataclass
class _ExecutionState:
    values: list[float]
    shape: Shape3D
    pool: int
    tolerance: float
    q_input: list[int] = field(default_factory=list)
    latent_q: list[int] = field(default_factory=list)
    latent_shape: Shape3D = (0, 0, 0)
    reconstructed_q: list[int] = field(default_factory=list)
    clipped_values: int = 0
    verification: Verification | None = None


def quantize_q16_16(value: float) -> tuple[int, bool]:
    if not isfinite(value):
        raise ValueError("all input values must be finite")
    scaled = round(value * Q_SCALE)
    clipped = scaled < Q_MIN or scaled > Q_MAX
    return min(max(scaled, Q_MIN), Q_MAX), clipped


def dequantize_q16_16(value: int) -> float:
    return value / Q_SCALE


def _voxel_count(shape: Shape3D) -> int:
    sx, sy, sz = shape
    return sx * sy * sz


def _index(shape: Shape3D, x: int, y: int, z: int) -> int:
    sx, sy, _ = shape
    return (z * sy + y) * sx + x


def _round_div(numerator: int, denominator: int) -> int:
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _pool3d(values: Sequence[int], shape: Shape3D, factor: int) -> tuple[list[int], Shape3D]:
    sx, sy, sz = shape
    latent_shape = (
        (sx + factor - 1) // factor,
        (sy + factor - 1) // factor,
        (sz + factor - 1) // factor,
    )
    lx, ly, lz = latent_shape
    latent: list[int] = []
    for z0 in range(lz):
        for y0 in range(ly):
            for x0 in range(lx):
                total = 0
                count = 0
                for dz in range(factor):
                    z = z0 * factor + dz
                    if z >= sz:
                        break
                    for dy in range(factor):
                        y = y0 * factor + dy
                        if y >= sy:
                            break
                        for dx in range(factor):
                            x = x0 * factor + dx
                            if x >= sx:
                                break
                            total += values[_index(shape, x, y, z)]
                            count += 1
                latent.append(_round_div(total, count))
    return latent, latent_shape


def _expand3d(
    values: Sequence[int], latent_shape: Shape3D, output_shape: Shape3D, factor: int
) -> list[int]:
    sx, sy, sz = output_shape
    reconstructed = [0] * _voxel_count(output_shape)
    for z in range(sz):
        for y in range(sy):
            for x in range(sx):
                source = _index(latent_shape, x // factor, y // factor, z // factor)
                reconstructed[_index(output_shape, x, y, z)] = values[source]
    return reconstructed


def _checksum_q16(values: Sequence[int]) -> str:
    digest = sha256()
    for value in values:
        digest.update(pack("<i", value))
    return digest.hexdigest()


def compile_spatial_program(pool: int) -> tuple[SpatialInstruction, ...]:
    """Compile the fixed end-to-end request path into 3D-addressed 32-bit words."""

    if not 1 <= pool <= 16:
        raise ValueError("pool must be between 1 and 16")
    stages = (
        (Opcode.NET_RX, 0, "network-ingress"),
        (Opcode.HOST_STAGE, 1, "host-orchestration"),
        (Opcode.Q16_CONVERT, 2, "q16.16-transmutation"),
        (Opcode.PACK3D, 2, "spatial-pack"),
        (Opcode.ENCODE3D, 3, "latent-contract"),
        (Opcode.LATENT_WRITE, 4, "latent-core"),
        (Opcode.DECODE3D, 3, "latent-expand"),
        (Opcode.VERIFY, 2, "closed-loop-verify"),
        (Opcode.TELEMETRY, 1, "out-of-band-telemetry"),
        (Opcode.EMIT, 0, "network-egress"),
        (Opcode.HALT, 0, "halt"),
    )
    program: list[SpatialInstruction] = []
    for tick, (opcode, z, stage) in enumerate(stages):
        imm = pool if opcode in {Opcode.ENCODE3D, Opcode.DECODE3D} else 0
        word = Instruction32(opcode=opcode, imm=imm).encode()
        program.append(SpatialInstruction(word=word, x=tick, y=tick, z=z, stage=stage))
    return tuple(program)


class BitCode3DRuntime:
    """Deterministic sparse 3D Bit Code reference runtime.

    The implementation is intentionally backend-neutral: the instruction stream,
    Q16.16 representation, validation semantics, and telemetry remain stable while
    ENCODE3D/DECODE3D can later be lowered to CUDA kernels without changing the API.
    """

    def __init__(self, *, max_voxels: int = MAX_VOXELS) -> None:
        if max_voxels < 1:
            raise ValueError("max_voxels must be positive")
        self.max_voxels = max_voxels

    def execute(
        self,
        values: Sequence[float],
        shape: Shape3D,
        *,
        pool: int = 2,
        tolerance: float = 1.0,
    ) -> BitCode3DResult:
        if len(shape) != 3 or any(dimension < 1 for dimension in shape):
            raise ValueError("shape must contain exactly three positive dimensions")
        count = _voxel_count(shape)
        if count > self.max_voxels:
            raise ValueError(f"voxel count {count} exceeds runtime limit {self.max_voxels}")
        if len(values) != count:
            raise ValueError(f"values length {len(values)} does not match shape voxel count {count}")
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be a finite non-negative number")

        program = compile_spatial_program(pool)
        state = _ExecutionState(
            values=[float(value) for value in values],
            shape=shape,
            pool=pool,
            tolerance=tolerance,
        )
        stages: list[str] = []
        started = perf_counter()
        cycles = 0

        for spatial in program:
            instruction = spatial.instruction
            stages.append(spatial.stage)
            cycles += 1

            if instruction.opcode is Opcode.NET_RX:
                continue
            if instruction.opcode is Opcode.HOST_STAGE:
                continue
            if instruction.opcode is Opcode.Q16_CONVERT:
                quantized: list[int] = []
                clipped = 0
                for value in state.values:
                    q_value, was_clipped = quantize_q16_16(value)
                    quantized.append(q_value)
                    clipped += int(was_clipped)
                state.q_input = quantized
                state.clipped_values = clipped
                continue
            if instruction.opcode is Opcode.PACK3D:
                if len(state.q_input) != count:
                    raise RuntimeError("PACK3D requires a complete Q16.16 input volume")
                continue
            if instruction.opcode is Opcode.ENCODE3D:
                state.latent_q, state.latent_shape = _pool3d(
                    state.q_input, state.shape, instruction.imm
                )
                continue
            if instruction.opcode is Opcode.LATENT_WRITE:
                if not state.latent_q:
                    raise RuntimeError("LATENT_WRITE requires encoded state")
                continue
            if instruction.opcode is Opcode.DECODE3D:
                state.reconstructed_q = _expand3d(
                    state.latent_q, state.latent_shape, state.shape, instruction.imm
                )
                continue
            if instruction.opcode is Opcode.VERIFY:
                if len(state.reconstructed_q) != count:
                    raise RuntimeError("VERIFY requires a complete reconstruction")
                errors = [
                    dequantize_q16_16(source - reconstructed)
                    for source, reconstructed in zip(state.q_input, state.reconstructed_q)
                ]
                mse = sum(error * error for error in errors) / count
                max_abs_error = max((abs(error) for error in errors), default=0.0)
                state.verification = Verification(
                    mse=mse,
                    max_abs_error=max_abs_error,
                    tolerance=tolerance,
                    passed=max_abs_error <= tolerance,
                    checksum_sha256=_checksum_q16(state.reconstructed_q),
                )
                continue
            if instruction.opcode in {Opcode.TELEMETRY, Opcode.EMIT}:
                continue
            if instruction.opcode is Opcode.HALT:
                break
            raise RuntimeError(f"unsupported 3D Bit Code opcode {instruction.opcode.name}")

        verification = state.verification
        if verification is None:
            raise RuntimeError("program halted without verification")

        elapsed_ms = (perf_counter() - started) * 1000.0
        telemetry = Telemetry(
            cycles=cycles,
            active_cells=count,
            latent_cells=len(state.latent_q),
            clipped_values=state.clipped_values,
            elapsed_ms=elapsed_ms,
            stages=tuple(stages),
        )
        return BitCode3DResult(
            input_shape=shape,
            latent_shape=state.latent_shape,
            latent=tuple(dequantize_q16_16(value) for value in state.latent_q),
            reconstructed=tuple(dequantize_q16_16(value) for value in state.reconstructed_q),
            bytecode=tuple(item.word for item in program),
            spatial_program=program,
            verification=verification,
            telemetry=telemetry,
        )
