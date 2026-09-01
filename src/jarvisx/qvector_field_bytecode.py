"""128-bit geometric-field coprocessor for the Dr Moagi QVector VM.

The coprocessor shares vector-field handles and vector registers with the v1
QVector bytecode engine when one is supplied.  It adds deterministic field-level
operators without changing the stable v1 opcode map.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence

from .qvector3d import Q_ONE, QVector3Q16, QVectorField3D
from .qvector_bytecode3d import DrMoagiQVectorBytecodeEngine3D
from .qvector_v2 import QNumericPolicy, QRoundMode, QScalarKernel3D, QVectorFieldOps3D

FIELD_INSTRUCTION_BYTES = 16
FIELD_REGISTER_COUNT = 16
FIELD_PROTOCOL = "jarvisx.dr-moagi-qvector-field-bytecode.v2"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _volume(shape: Sequence[int]) -> int:
    if len(shape) != 3:
        raise ValueError("field program shape must contain exactly three dimensions")
    x, y, z = (int(axis) for axis in shape)
    if min(x, y, z) < 1:
        raise ValueError("field program dimensions must be positive")
    return x * y * z


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    parsed = (int(shape[0]), int(shape[1]), int(shape[2]))
    _volume(parsed)
    if max(parsed) > 0xFFFF:
        raise ValueError("field ROM dimensions cannot exceed 65535")
    return parsed


def _signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


class QVectorFieldOpcode(IntEnum):
    NOP = 0x00

    QSETMODE = 0x50
    QCLRSTATUS = 0x51
    QSTATUS = 0x52

    VGRADX = 0x60
    VGRADY = 0x61
    VGRADZ = 0x62
    VDIV = 0x63
    VCURL = 0x64
    VLAPLACE = 0x65
    VCONV3D = 0x66

    VERIFY = 0x70
    SEAL = 0x71
    HALT = 0xFF


@dataclass(frozen=True, slots=True)
class QVectorFieldInstruction128:
    opcode: QVectorFieldOpcode
    flags: int = 0
    x: int = 0
    y: int = 0
    z: int = 0
    a: int = 0
    b: int = 0
    imm: int = 0

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("flags", self.flags, 0xFF),
            ("x", self.x, 0xFFFF),
            ("y", self.y, 0xFFFF),
            ("z", self.z, 0xFFFF),
            ("a", self.a, 0xFFFF),
            ("b", self.b, 0xFFFF),
            ("imm", self.imm, 0xFFFFFFFF),
        ):
            if int(value) < 0 or int(value) > limit:
                raise ValueError(f"{name} must be in the range 0..{limit}")

    @property
    def destination_register(self) -> int:
        return self.flags & 0x0F

    def to_int(self) -> int:
        return (
            (int(self.opcode) << 120)
            | (self.flags << 112)
            | (self.x << 96)
            | (self.y << 80)
            | (self.z << 64)
            | (self.a << 48)
            | (self.b << 32)
            | self.imm
        )

    def to_bytes(self) -> bytes:
        return self.to_int().to_bytes(FIELD_INSTRUCTION_BYTES, "big", signed=False)

    @classmethod
    def from_bytes(cls, encoded: bytes) -> "QVectorFieldInstruction128":
        if len(encoded) != FIELD_INSTRUCTION_BYTES:
            raise ValueError("one field instruction must contain exactly 16 bytes")
        word = int.from_bytes(encoded, "big", signed=False)
        raw_opcode = (word >> 120) & 0xFF
        try:
            opcode = QVectorFieldOpcode(raw_opcode)
        except ValueError as error:
            raise ValueError(f"unknown QVector field opcode 0x{raw_opcode:02x}") from error
        return cls(
            opcode=opcode,
            flags=(word >> 112) & 0xFF,
            x=(word >> 96) & 0xFFFF,
            y=(word >> 80) & 0xFFFF,
            z=(word >> 64) & 0xFFFF,
            a=(word >> 48) & 0xFFFF,
            b=(word >> 32) & 0xFFFF,
            imm=word & 0xFFFFFFFF,
        )


@dataclass(frozen=True, slots=True)
class QVectorFieldProgram3D:
    shape: tuple[int, int, int]
    instructions: tuple[QVectorFieldInstruction128, ...]

    def __post_init__(self) -> None:
        if _shape3(self.shape) != self.shape:
            raise ValueError("field program shape must be normalized integers")
        if len(self.instructions) != _volume(self.shape):
            raise ValueError("field instruction count must equal ROM volume")

    @classmethod
    def from_instructions(
        cls,
        instructions: Iterable[QVectorFieldInstruction128],
        shape: Sequence[int] | None = None,
    ) -> "QVectorFieldProgram3D":
        items = tuple(instructions)
        if not items:
            raise ValueError("field program must contain at least one instruction")
        shape3 = _shape3(shape or (len(items), 1, 1))
        volume = _volume(shape3)
        if len(items) > volume:
            raise ValueError("field program contains more instructions than its ROM")
        if len(items) < volume:
            items = items + (QVectorFieldInstruction128(QVectorFieldOpcode.NOP),) * (
                volume - len(items)
            )
        return cls(shape3, items)

    def to_bytes(self) -> bytes:
        return b"".join(instruction.to_bytes() for instruction in self.instructions)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def pc_xyz(self, pc: int) -> tuple[int, int, int]:
        if pc < 0 or pc >= len(self.instructions):
            raise IndexError("field program counter is outside the ROM")
        sx, sy, _ = self.shape
        return (pc % sx, (pc // sx) % sy, pc // (sx * sy))


@dataclass(frozen=True, slots=True)
class QVectorFieldExecutionReport3D:
    protocol: str
    program_digest: str
    rom_shape: tuple[int, int, int]
    cycles: int
    halted: bool
    final_pc: int
    state_digest: str
    trace_digest: str
    numeric_status: tuple[bool, bool, bool, bool]


class DrMoagiQVectorFieldBytecodeEngine3D:
    """Field-operator coprocessor sharing state with a QVector bytecode engine."""

    def __init__(self, base: DrMoagiQVectorBytecodeEngine3D | None = None) -> None:
        self.base = base or DrMoagiQVectorBytecodeEngine3D()
        self.ops = QVectorFieldOps3D()
        self.kernels: dict[int, QScalarKernel3D] = {}
        self.program: QVectorFieldProgram3D | None = None
        self.pc = 0
        self.cycles = 0
        self.halted = False
        self.trace_digest = "0" * 64

    @property
    def vector_fields(self) -> dict[int, QVectorField3D]:
        return self.base.vector_fields

    def mount_vector_field(self, handle: int, field: QVectorField3D) -> None:
        self.base.mount_vector_field(handle, field)

    def mount_kernel(self, handle: int, kernel: QScalarKernel3D) -> None:
        handle = int(handle)
        if handle < 0 or handle > 0xFFFF:
            raise ValueError("kernel handle must be in the range 0..65535")
        self.kernels[handle] = kernel

    def _field(self, handle: int) -> QVectorField3D:
        try:
            return self.vector_fields[int(handle)]
        except KeyError as error:
            raise KeyError(f"vector field handle {handle} is not mounted") from error

    def _write_status_vector(self, instruction: QVectorFieldInstruction128) -> None:
        register = instruction.destination_register
        coordinate = (instruction.x, instruction.y, instruction.z)
        status = self.ops.status
        vector = QVector3Q16(
            Q_ONE if status.saturated else 0,
            Q_ONE if status.accumulator_saturated else 0,
            Q_ONE if status.inexact else 0,
        )
        self.base._write_vector(coordinate, register, vector)

    def _spacing(self, instruction: QVectorFieldInstruction128) -> int:
        raw = _signed32(instruction.imm)
        return Q_ONE if raw == 0 else raw

    def _state_digest(self) -> str:
        return _digest(
            {
                "fields": [
                    {"handle": handle, "digest": field.digest}
                    for handle, field in sorted(self.vector_fields.items())
                ],
                "kernels": [
                    {
                        "handle": handle,
                        "shape": list(kernel.shape),
                        "weights_q16": list(kernel.weights_q16),
                    }
                    for handle, kernel in sorted(self.kernels.items())
                ],
                "status": {
                    "saturated": self.ops.status.saturated,
                    "accumulator_saturated": self.ops.status.accumulator_saturated,
                    "inexact": self.ops.status.inexact,
                    "divide_by_zero": self.ops.status.divide_by_zero,
                },
                "rounding": int(self.ops.policy.rounding),
            }
        )

    def load(self, program: QVectorFieldProgram3D) -> None:
        self.program = program
        self.pc = 0
        self.cycles = 0
        self.halted = False
        self.trace_digest = "0" * 64

    def _execute(self, instruction: QVectorFieldInstruction128) -> None:
        opcode = instruction.opcode
        if opcode == QVectorFieldOpcode.NOP:
            return
        if opcode == QVectorFieldOpcode.QSETMODE:
            try:
                rounding = QRoundMode(instruction.imm & 0xFF)
            except ValueError as error:
                raise ValueError("QSETMODE contains an invalid rounding mode") from error
            self.ops.policy = QNumericPolicy(
                rounding=rounding,
                saturate=self.ops.policy.saturate,
                accumulator_saturate=self.ops.policy.accumulator_saturate,
            )
            return
        if opcode == QVectorFieldOpcode.QCLRSTATUS:
            self.ops.clear_status()
            return
        if opcode == QVectorFieldOpcode.QSTATUS:
            self._write_status_vector(instruction)
            return

        spacing = self._spacing(instruction)
        if opcode == QVectorFieldOpcode.VGRADX:
            self.vector_fields[instruction.b] = self.ops.directional_derivative(
                self._field(instruction.a), 0, spacing_q16=spacing
            )
        elif opcode == QVectorFieldOpcode.VGRADY:
            self.vector_fields[instruction.b] = self.ops.directional_derivative(
                self._field(instruction.a), 1, spacing_q16=spacing
            )
        elif opcode == QVectorFieldOpcode.VGRADZ:
            self.vector_fields[instruction.b] = self.ops.directional_derivative(
                self._field(instruction.a), 2, spacing_q16=spacing
            )
        elif opcode == QVectorFieldOpcode.VDIV:
            self.vector_fields[instruction.b] = self.ops.divergence(
                self._field(instruction.a), spacing_q16=spacing
            )
        elif opcode == QVectorFieldOpcode.VCURL:
            self.vector_fields[instruction.b] = self.ops.curl(
                self._field(instruction.a), spacing_q16=spacing
            )
        elif opcode == QVectorFieldOpcode.VLAPLACE:
            self.vector_fields[instruction.b] = self.ops.laplacian(
                self._field(instruction.a), spacing_q16=spacing
            )
        elif opcode == QVectorFieldOpcode.VCONV3D:
            kernel_handle = instruction.imm & 0xFFFF
            try:
                kernel = self.kernels[kernel_handle]
            except KeyError as error:
                raise KeyError(f"QVector kernel handle {kernel_handle} is not mounted") from error
            self.vector_fields[instruction.b] = self.ops.convolve(
                self._field(instruction.a), kernel
            )
        elif opcode == QVectorFieldOpcode.VERIFY:
            raw = Q_ONE if self.base.cloud.cloud.ledger.verify() else 0
            self.base._write_vector(
                (instruction.x, instruction.y, instruction.z),
                instruction.destination_register,
                QVector3Q16(raw, raw, raw),
            )
        elif opcode == QVectorFieldOpcode.SEAL:
            assert self.program is not None
            self.base.cloud.cloud.ledger.append(
                "qvector.field.bytecode.sealed",
                {
                    "protocol": FIELD_PROTOCOL,
                    "program_digest": self.program.digest,
                    "pc": self.pc,
                    "pc_xyz": list(self.program.pc_xyz(self.pc)),
                    "state_digest": self._state_digest(),
                    "trace_digest": self.trace_digest,
                },
            )
        elif opcode == QVectorFieldOpcode.HALT:
            self.halted = True
        else:  # pragma: no cover
            raise ValueError(f"unsupported field opcode {opcode!r}")

    def step(self) -> None:
        if self.program is None:
            raise RuntimeError("no QVector field program is loaded")
        if self.halted:
            raise RuntimeError("QVector field coprocessor is halted")
        if self.pc < 0 or self.pc >= len(self.program.instructions):
            raise IndexError("field program counter escaped ROM without HALT")
        instruction = self.program.instructions[self.pc]
        before = self._state_digest()
        current_pc = self.pc
        self._execute(instruction)
        after = self._state_digest()
        trace_record = {
            "pc": current_pc,
            "pc_xyz": list(self.program.pc_xyz(current_pc)),
            "instruction": instruction.to_bytes().hex(),
            "before": before,
            "after": after,
            "previous": self.trace_digest,
        }
        self.trace_digest = _digest(trace_record)
        self.cycles += 1
        if not self.halted:
            self.pc += 1

    def run(self, *, max_cycles: int = 100_000) -> QVectorFieldExecutionReport3D:
        if self.program is None:
            raise RuntimeError("no QVector field program is loaded")
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")
        while not self.halted:
            if self.cycles >= max_cycles:
                raise RuntimeError("QVector field coprocessor exceeded cycle limit")
            self.step()
        state_digest = self._state_digest()
        status = self.ops.status
        self.base.cloud.cloud.ledger.append(
            "qvector.field.bytecode.run.committed",
            {
                "protocol": FIELD_PROTOCOL,
                "program_digest": self.program.digest,
                "cycles": self.cycles,
                "state_digest": state_digest,
                "trace_digest": self.trace_digest,
            },
        )
        return QVectorFieldExecutionReport3D(
            protocol=FIELD_PROTOCOL,
            program_digest=self.program.digest,
            rom_shape=self.program.shape,
            cycles=self.cycles,
            halted=self.halted,
            final_pc=self.pc,
            state_digest=state_digest,
            trace_digest=self.trace_digest,
            numeric_status=(
                status.saturated,
                status.accumulator_saturated,
                status.inexact,
                status.divide_by_zero,
            ),
        )


__all__ = [
    "DrMoagiQVectorFieldBytecodeEngine3D",
    "FIELD_INSTRUCTION_BYTES",
    "FIELD_PROTOCOL",
    "QVectorFieldExecutionReport3D",
    "QVectorFieldInstruction128",
    "QVectorFieldOpcode",
    "QVectorFieldProgram3D",
]
