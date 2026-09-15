"""128-bit bytecode VM for Q16.16 x Q16.16 x Q16.16 vector computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence

from .qvector3d import Q_ONE, QVector3Q16, QVectorAutoencoder3D, QVectorField3D, q16_from_float
from .qvector_cloud import DrMoagiQVectorCloudEngine3D

QVECTOR_BYTECODE_PROTOCOL = "jarvisx.dr-moagi-qvector-bytecode3d.v1"
INSTRUCTION_BYTES = 16
VECTOR_REGISTER_COUNT = 16
ZERO_DIGEST = "0" * 64


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
        raise ValueError("shape must contain exactly three dimensions")
    x, y, z = (int(axis) for axis in shape)
    if min(x, y, z) < 1:
        raise ValueError("shape dimensions must be positive")
    return x * y * z


def _shape3(shape: Sequence[int]) -> tuple[int, int, int]:
    _volume(shape)
    parsed = (int(shape[0]), int(shape[1]), int(shape[2]))
    if max(parsed) > 0xFFFF:
        raise ValueError("3D ROM dimensions cannot exceed 65535")
    return parsed


def _signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def pack_shape(shape: Sequence[int]) -> int:
    x, y, z = (int(axis) for axis in shape)
    if min(x, y, z) < 1 or max(x, y, z) > 1023:
        raise ValueError("packed shape dimensions must be in the range 1..1023")
    return x | (y << 10) | (z << 20)


def unpack_shape(value: int) -> tuple[int, int, int]:
    x = value & 0x3FF
    y = (value >> 10) & 0x3FF
    z = (value >> 20) & 0x3FF
    if min(x, y, z) < 1:
        raise ValueError("packed shape contains a zero dimension")
    return (x, y, z)


def pack_coordinate(x: int, y: int, z: int) -> int:
    if min(x, y, z) < 0 or max(x, y, z) > 1023:
        raise ValueError("packed coordinates must be in the range 0..1023")
    return x | (y << 10) | (z << 20)


def unpack_coordinate(value: int) -> tuple[int, int, int]:
    return (value & 0x3FF, (value >> 10) & 0x3FF, (value >> 20) & 0x3FF)


def _pack_signed10(value: int) -> int:
    if value < -512 or value > 511:
        raise ValueError("neighbor offsets must be in the range -512..511")
    return value & 0x3FF


def _unpack_signed10(value: int) -> int:
    value &= 0x3FF
    return value - 1024 if value & 0x200 else value


def pack_offset(dx: int, dy: int, dz: int) -> int:
    return _pack_signed10(dx) | (_pack_signed10(dy) << 10) | (_pack_signed10(dz) << 20)


def unpack_offset(value: int) -> tuple[int, int, int]:
    return (
        _unpack_signed10(value),
        _unpack_signed10(value >> 10),
        _unpack_signed10(value >> 20),
    )


def _metric_vector(x: float, y: float, z: float) -> QVector3Q16:
    return QVector3Q16.from_floats(x, y, z, saturate=True)


class QVectorOpcode(IntEnum):
    NOP = 0x00
    VFETCH = 0x01
    VMOV = 0x02
    VMOVE = 0x03
    VSTORE = 0x04

    VADD = 0x10
    VSUB = 0x11
    VMUL = 0x12
    VSCALE = 0x13

    VFENCODE = 0x20
    VFDECODE = 0x21
    VFROUND = 0x22
    VFERR = 0x23
    VFAUTO = 0x24

    VERIFY = 0x30
    SEAL = 0x31

    JMP = 0x40
    JNZ = 0x41

    HALT = 0xFF


@dataclass(frozen=True, slots=True)
class QVectorInstruction128:
    opcode: QVectorOpcode
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
        return self.to_int().to_bytes(INSTRUCTION_BYTES, "big", signed=False)

    @classmethod
    def from_bytes(cls, encoded: bytes) -> "QVectorInstruction128":
        if len(encoded) != INSTRUCTION_BYTES:
            raise ValueError("one instruction must contain exactly 16 bytes")
        word = int.from_bytes(encoded, "big", signed=False)
        raw_opcode = (word >> 120) & 0xFF
        try:
            opcode = QVectorOpcode(raw_opcode)
        except ValueError as error:
            raise ValueError(f"unknown QVector opcode 0x{raw_opcode:02x}") from error
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
class QVectorProgram3D:
    shape: tuple[int, int, int]
    instructions: tuple[QVectorInstruction128, ...]

    def __post_init__(self) -> None:
        normalized = _shape3(self.shape)
        if normalized != self.shape:
            raise ValueError("program shape must contain normalized integer dimensions")
        if len(self.instructions) != _volume(self.shape):
            raise ValueError("instruction count must equal the 3D ROM volume")

    @classmethod
    def from_instructions(
        cls,
        instructions: Iterable[QVectorInstruction128],
        shape: Sequence[int] | None = None,
    ) -> "QVectorProgram3D":
        items = tuple(instructions)
        if not items:
            raise ValueError("program must contain at least one instruction")
        target_shape = _shape3(shape or (len(items), 1, 1))
        volume = _volume(target_shape)
        if len(items) > volume:
            raise ValueError("program contains more instructions than the ROM volume")
        if len(items) < volume:
            items = items + (QVectorInstruction128(QVectorOpcode.NOP),) * (volume - len(items))
        return cls(target_shape, items)

    @classmethod
    def from_bytes(cls, encoded: bytes, shape: Sequence[int]) -> "QVectorProgram3D":
        shape3 = _shape3(shape)
        expected = _volume(shape3) * INSTRUCTION_BYTES
        if len(encoded) != expected:
            raise ValueError(f"bytecode contains {len(encoded)} bytes; expected {expected}")
        instructions = tuple(
            QVectorInstruction128.from_bytes(encoded[offset : offset + INSTRUCTION_BYTES])
            for offset in range(0, len(encoded), INSTRUCTION_BYTES)
        )
        return cls(shape3, instructions)

    def to_bytes(self) -> bytes:
        return b"".join(instruction.to_bytes() for instruction in self.instructions)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def pc_xyz(self, pc: int) -> tuple[int, int, int]:
        if pc < 0 or pc >= len(self.instructions):
            raise IndexError("program counter is outside the QVector ROM")
        sx, sy, _ = self.shape
        return (pc % sx, (pc // sx) % sy, pc // (sx * sy))


@dataclass(frozen=True, slots=True)
class QVectorExecutionReport3D:
    protocol: str
    program_digest: str
    rom_shape: tuple[int, int, int]
    cycles: int
    halted: bool
    final_pc: int
    final_state_digest: str
    trace_digest: str
    register_voxels: int
    vector_field_handles: tuple[int, ...]


class DrMoagiQVectorBytecodeEngine3D:
    """Auto-executing 3D VM whose registers and field cells are Q16.16x3 vectors."""

    def __init__(
        self,
        cloud: DrMoagiQVectorCloudEngine3D | None = None,
        *,
        ledger_path: Path | None = None,
        default_node_cells: int = 3_000_000,
    ) -> None:
        if cloud is not None and ledger_path is not None:
            raise ValueError("ledger_path cannot be supplied with an existing vector cloud engine")
        self.cloud = cloud or DrMoagiQVectorCloudEngine3D(
            ledger_path=ledger_path,
            default_node_cells=default_node_cells,
        )
        self.encoder = QVectorAutoencoder3D()
        self.registers: dict[tuple[int, int, int], list[QVector3Q16]] = {}
        self.vector_fields: dict[int, QVectorField3D] = {}
        self.trace: list[dict[str, object]] = []
        self.trace_digest = ZERO_DIGEST
        self.program: QVectorProgram3D | None = None
        self.pc = 0
        self.cycles = 0
        self.halted = False

    @staticmethod
    def _validate_handle(handle: int) -> int:
        handle = int(handle)
        if handle < 0 or handle > 0xFFFF:
            raise ValueError("vector field handle must be in the range 0..65535")
        return handle

    @staticmethod
    def _validate_register(register: int) -> int:
        register = int(register)
        if register < 0 or register >= VECTOR_REGISTER_COUNT:
            raise ValueError(
                f"vector register must be in the range 0..{VECTOR_REGISTER_COUNT - 1}"
            )
        return register

    def _voxel(self, coordinate: tuple[int, int, int]) -> list[QVector3Q16]:
        if coordinate not in self.registers:
            self.registers[coordinate] = [
                QVector3Q16.zero() for _ in range(VECTOR_REGISTER_COUNT)
            ]
        return self.registers[coordinate]

    def read_vector(self, x: int, y: int, z: int, register: int) -> QVector3Q16:
        register = self._validate_register(register)
        return self._voxel((int(x), int(y), int(z)))[register]

    def read_vector_floats(
        self,
        x: int,
        y: int,
        z: int,
        register: int,
    ) -> tuple[float, float, float]:
        return self.read_vector(x, y, z, register).to_floats()

    def _write_vector(
        self,
        coordinate: tuple[int, int, int],
        register: int,
        value: QVector3Q16,
    ) -> None:
        register = self._validate_register(register)
        self._voxel(coordinate)[register] = value

    def mount_vector_field(self, handle: int, field: QVectorField3D) -> None:
        self.vector_fields[self._validate_handle(handle)] = field

    def _field(self, handle: int) -> QVectorField3D:
        handle = self._validate_handle(handle)
        try:
            return self.vector_fields[handle]
        except KeyError as error:
            raise KeyError(f"vector field handle {handle} is not mounted") from error

    @staticmethod
    def _payload_field(payload: object) -> QVectorField3D:
        if not isinstance(payload, dict):
            raise TypeError("vector cloud field result must be a dictionary")
        shape = payload.get("shape")
        vectors_q16 = payload.get("vectors_q16")
        if not isinstance(shape, list) or not isinstance(vectors_q16, list):
            raise TypeError(
                "vector cloud field result must contain shape and vectors_q16 lists"
            )
        return QVectorField3D.from_raw(vectors_q16, shape)

    def _state_digest(self) -> str:
        return _digest(
            {
                "registers": [
                    {
                        "xyz": list(coordinate),
                        "vectors_q16": [
                            [value.x, value.y, value.z] for value in registers
                        ],
                    }
                    for coordinate, registers in sorted(self.registers.items())
                ],
                "vector_fields": [
                    {"handle": handle, "digest": self.vector_fields[handle].digest}
                    for handle in sorted(self.vector_fields)
                ],
            }
        )

    def load(self, program: QVectorProgram3D, *, reset_state: bool = False) -> None:
        self.program = program
        self.pc = 0
        self.cycles = 0
        self.halted = False
        self.trace.clear()
        self.trace_digest = ZERO_DIGEST
        if reset_state:
            self.registers.clear()
            self.vector_fields.clear()

    def _request_id(
        self,
        instruction: QVectorInstruction128,
        source: QVectorField3D,
    ) -> str:
        assert self.program is not None
        return "qvbc3d-" + _digest(
            {
                "program": self.program.digest,
                "pc": self.pc,
                "opcode": int(instruction.opcode),
                "source": source.digest,
                "a": instruction.a,
                "b": instruction.b,
                "imm": instruction.imm,
            }
        )[:32]

    def _jump_target(self, target: int) -> int:
        assert self.program is not None
        if target < 0 or target >= len(self.program.instructions):
            raise IndexError(f"jump target {target} is outside the QVector ROM")
        return target

    def _execute(
        self,
        instruction: QVectorInstruction128,
        next_pc: int,
    ) -> tuple[int, bool]:
        coordinate = (instruction.x, instruction.y, instruction.z)
        destination = instruction.destination_register
        opcode = instruction.opcode
        branch_taken = False

        if opcode == QVectorOpcode.NOP:
            pass
        elif opcode == QVectorOpcode.VFETCH:
            source_xyz = unpack_coordinate(instruction.imm)
            self._write_vector(
                coordinate,
                destination,
                self._field(instruction.a).at(*source_xyz),
            )
        elif opcode == QVectorOpcode.VSTORE:
            source_register = self._validate_register(instruction.a)
            target_xyz = unpack_coordinate(instruction.imm)
            updated = self._field(instruction.b).replace(
                *target_xyz,
                self._voxel(coordinate)[source_register],
            )
            self.mount_vector_field(instruction.b, updated)
        elif opcode == QVectorOpcode.VMOV:
            source_register = self._validate_register(instruction.a)
            self._write_vector(
                coordinate,
                destination,
                self._voxel(coordinate)[source_register],
            )
        elif opcode == QVectorOpcode.VMOVE:
            source_register = self._validate_register(instruction.a)
            dx, dy, dz = unpack_offset(instruction.imm)
            source_coordinate = (
                coordinate[0] + dx,
                coordinate[1] + dy,
                coordinate[2] + dz,
            )
            if min(source_coordinate) < 0 or max(source_coordinate) > 0xFFFF:
                raise IndexError(
                    "vector neighbor move points outside the 16-bit voxel lattice"
                )
            self._write_vector(
                coordinate,
                destination,
                self._voxel(source_coordinate)[source_register],
            )
        elif opcode in {QVectorOpcode.VADD, QVectorOpcode.VSUB, QVectorOpcode.VMUL}:
            left = self._voxel(coordinate)[self._validate_register(instruction.a)]
            right = self._voxel(coordinate)[self._validate_register(instruction.b)]
            if opcode == QVectorOpcode.VADD:
                value = left.add(right)
            elif opcode == QVectorOpcode.VSUB:
                value = left.sub(right)
            else:
                value = left.hadamard(right)
            self._write_vector(coordinate, destination, value)
        elif opcode == QVectorOpcode.VSCALE:
            source = self._voxel(coordinate)[self._validate_register(instruction.a)]
            self._write_vector(
                coordinate,
                destination,
                source.scale(_signed32(instruction.imm)),
            )
        elif opcode == QVectorOpcode.VFENCODE:
            latent = self.encoder.encode(
                self._field(instruction.a),
                unpack_shape(instruction.imm),
            )
            self.mount_vector_field(instruction.b, latent)
        elif opcode == QVectorOpcode.VFDECODE:
            decoded = self.encoder.decode(
                self._field(instruction.a),
                unpack_shape(instruction.imm),
            )
            self.mount_vector_field(instruction.b, decoded)
        elif opcode == QVectorOpcode.VFERR:
            axis_mse, _, _ = self.encoder.error_metrics(
                self._field(instruction.a),
                self._field(instruction.b),
            )
            self._write_vector(coordinate, destination, _metric_vector(*axis_mse))
        elif opcode == QVectorOpcode.VFROUND:
            if instruction.b == 0xFFFF:
                raise ValueError(
                    "VFROUND output handle must leave room for the latent handle"
                )
            source = self._field(instruction.a)
            job = self.cloud.round_trip(
                source,
                unpack_shape(instruction.imm),
                request_id=self._request_id(instruction, source),
            )
            if job.result is None:
                raise RuntimeError("vector cloud round-trip completed without a result")
            self.mount_vector_field(
                instruction.b,
                self._payload_field(job.result.get("reconstruction")),
            )
            self.mount_vector_field(
                instruction.b + 1,
                self._payload_field(job.result.get("latent")),
            )
            axis_mse = job.result.get("axis_mse")
            if not isinstance(axis_mse, list) or len(axis_mse) != 3:
                raise TypeError("vector cloud result is missing axis_mse")
            self._write_vector(
                coordinate,
                destination,
                _metric_vector(
                    float(axis_mse[0]),
                    float(axis_mse[1]),
                    float(axis_mse[2]),
                ),
            )
        elif opcode == QVectorOpcode.VFAUTO:
            if instruction.b == 0xFFFF:
                raise ValueError(
                    "VFAUTO output handle must leave room for the latent handle"
                )
            complexity_weight = _signed32(instruction.imm) / Q_ONE
            if complexity_weight < 0.0:
                raise ValueError("VFAUTO complexity weight must be non-negative")
            source = self._field(instruction.a)
            job = self.cloud.auto_optimize(
                source,
                request_id=self._request_id(instruction, source),
                complexity_weight=complexity_weight,
            )
            if job.result is None:
                raise RuntimeError("vector cloud optimizer completed without a result")
            self.mount_vector_field(
                instruction.b,
                self._payload_field(job.result.get("reconstruction")),
            )
            self.mount_vector_field(
                instruction.b + 1,
                self._payload_field(job.result.get("latent")),
            )
            objective = job.result.get("objective")
            component_mse = job.result.get("component_mse")
            compression_ratio = job.result.get("compression_ratio")
            if not all(
                isinstance(value, (int, float))
                for value in (objective, component_mse, compression_ratio)
            ):
                raise TypeError(
                    "vector cloud optimizer result is missing numeric metrics"
                )
            self._write_vector(
                coordinate,
                destination,
                _metric_vector(
                    float(objective),
                    float(component_mse),
                    float(compression_ratio),
                ),
            )
        elif opcode == QVectorOpcode.VERIFY:
            raw = q16_from_float(1.0) if self.cloud.cloud.ledger.verify() else 0
            self._write_vector(coordinate, destination, QVector3Q16(raw, raw, raw))
        elif opcode == QVectorOpcode.SEAL:
            self.cloud.cloud.ledger.append(
                "qvector.bytecode.sealed",
                {
                    "protocol": QVECTOR_BYTECODE_PROTOCOL,
                    "program_digest": self.program.digest if self.program else "",
                    "pc": self.pc,
                    "pc_xyz": list(self.program.pc_xyz(self.pc)) if self.program else [],
                    "state_digest": self._state_digest(),
                    "trace_digest": self.trace_digest,
                },
            )
        elif opcode == QVectorOpcode.JMP:
            next_pc = self._jump_target(instruction.imm)
            branch_taken = True
        elif opcode == QVectorOpcode.JNZ:
            value = self._voxel(coordinate)[
                self._validate_register(instruction.a)
            ]
            if value.x != 0 or value.y != 0 or value.z != 0:
                next_pc = self._jump_target(instruction.imm)
                branch_taken = True
        elif opcode == QVectorOpcode.HALT:
            self.halted = True
        else:  # pragma: no cover
            raise ValueError(f"unsupported QVector opcode {opcode!r}")

        return next_pc, branch_taken

    def step(self) -> dict[str, object]:
        if self.program is None:
            raise RuntimeError("no QVector bytecode program is loaded")
        if self.halted:
            raise RuntimeError("QVector bytecode engine is halted")
        if self.pc < 0 or self.pc >= len(self.program.instructions):
            raise IndexError("program counter escaped the QVector ROM without HALT")

        current_pc = self.pc
        instruction = self.program.instructions[current_pc]
        before = self._state_digest()
        next_pc, branch_taken = self._execute(instruction, current_pc + 1)
        after = self._state_digest()
        entry: dict[str, object] = {
            "protocol": QVECTOR_BYTECODE_PROTOCOL,
            "cycle": self.cycles,
            "pc": current_pc,
            "pc_xyz": list(self.program.pc_xyz(current_pc)),
            "target_xyz": [instruction.x, instruction.y, instruction.z],
            "opcode": instruction.opcode.name,
            "instruction": instruction.to_bytes().hex(),
            "before": before,
            "after": after,
            "next_pc": next_pc,
            "branch_taken": branch_taken,
            "previous_trace_digest": self.trace_digest,
        }
        self.trace_digest = _digest(entry)
        entry["trace_digest"] = self.trace_digest
        self.trace.append(entry)
        self.cycles += 1
        self.pc = next_pc
        return dict(entry)

    def run(self, *, max_cycles: int = 100_000) -> QVectorExecutionReport3D:
        if self.program is None:
            raise RuntimeError("no QVector bytecode program is loaded")
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")

        self.cloud.cloud.ledger.append(
            "qvector.bytecode.run.started",
            {
                "protocol": QVECTOR_BYTECODE_PROTOCOL,
                "program_digest": self.program.digest,
                "rom_shape": list(self.program.shape),
                "max_cycles": max_cycles,
            },
        )
        try:
            while not self.halted:
                if self.cycles >= max_cycles:
                    raise RuntimeError("QVector bytecode cycle limit exceeded")
                self.step()

            report = QVectorExecutionReport3D(
                protocol=QVECTOR_BYTECODE_PROTOCOL,
                program_digest=self.program.digest,
                rom_shape=self.program.shape,
                cycles=self.cycles,
                halted=self.halted,
                final_pc=self.pc,
                final_state_digest=self._state_digest(),
                trace_digest=self.trace_digest,
                register_voxels=len(self.registers),
                vector_field_handles=tuple(sorted(self.vector_fields)),
            )
            self.cloud.cloud.ledger.append(
                "qvector.bytecode.run.committed",
                {
                    "protocol": QVECTOR_BYTECODE_PROTOCOL,
                    "program_digest": report.program_digest,
                    "cycles": report.cycles,
                    "final_pc": report.final_pc,
                    "final_state_digest": report.final_state_digest,
                    "trace_digest": report.trace_digest,
                },
            )
            if not self.cloud.cloud.ledger.verify():
                raise RuntimeError(
                    "ledger verification failed after QVector bytecode commit"
                )
            return report
        except Exception as error:
            self.cloud.cloud.ledger.append(
                "qvector.bytecode.run.failed",
                {
                    "protocol": QVECTOR_BYTECODE_PROTOCOL,
                    "program_digest": self.program.digest,
                    "cycles": self.cycles,
                    "pc": self.pc,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


__all__ = [
    "INSTRUCTION_BYTES",
    "QVECTOR_BYTECODE_PROTOCOL",
    "VECTOR_REGISTER_COUNT",
    "DrMoagiQVectorBytecodeEngine3D",
    "QVectorExecutionReport3D",
    "QVectorInstruction128",
    "QVectorOpcode",
    "QVectorProgram3D",
    "pack_coordinate",
    "pack_offset",
    "pack_shape",
    "unpack_coordinate",
    "unpack_offset",
    "unpack_shape",
]
