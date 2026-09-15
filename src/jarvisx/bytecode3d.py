"""Deterministic 128-bit 3D bytecode engine for the Dr Moagi Cloud OS.

The engine treats execution as movement through two coupled 3D spaces:

* a dense 3D ROM, addressed by the program counter; and
* a sparse 3D voxel register lattice, addressed by each instruction.

Each 128-bit instruction is encoded as::

    opcode[8] | flags[8] | x[16] | y[16] | z[16] | a[16] | b[16] | imm[32]

The low four flag bits select the destination register (R0..R15). Arithmetic
uses deterministic Q16.16 fixed-point values. Field instructions operate on
``Field3D`` handles and cloud round-trip/optimization instructions execute
through ``DrMoagiCloudOS`` so node capacity, journaling and verify-before-
commit semantics remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence

from .cloud_os import DrMoagiCloudOS, Field3D

BYTECODE_PROTOCOL = "jarvisx.dr-moagi-bytecode3d.v1"
INSTRUCTION_BYTES = 16
REGISTER_COUNT = 16
Q_SHIFT = 16
Q_ONE = 1 << Q_SHIFT
I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
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
    if any(axis > 0xFFFF for axis in parsed):
        raise ValueError("3D ROM dimensions cannot exceed 65535")
    return parsed


def _i64(value: int) -> int:
    return min(I64_MAX, max(I64_MIN, int(value)))


def _trunc_div(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def q_from_float(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("Q16.16 values must be finite")
    raw = int(round(value * Q_ONE))
    if raw < -(1 << 31) or raw > (1 << 31) - 1:
        raise OverflowError("Q16.16 immediate does not fit signed 32 bits")
    return raw


def q_to_float(raw: int) -> float:
    return int(raw) / Q_ONE


def _signed32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value & 0x80000000 else value


def pack_shape(shape: Sequence[int]) -> int:
    """Pack a bounded 3D shape into 30 immediate bits (10 bits per axis)."""

    x, y, z = (int(axis) for axis in shape)
    if min(x, y, z) < 1 or max(x, y, z) > 1023:
        raise ValueError("packed field dimensions must be in the range 1..1023")
    return x | (y << 10) | (z << 20)


def unpack_shape(value: int) -> tuple[int, int, int]:
    x = value & 0x3FF
    y = (value >> 10) & 0x3FF
    z = (value >> 20) & 0x3FF
    if min(x, y, z) < 1:
        raise ValueError("packed shape contains a zero dimension")
    return (x, y, z)


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


class Opcode(IntEnum):
    NOP = 0x00
    QLOAD = 0x01
    QMOV = 0x02
    VMOVE = 0x03

    QADD = 0x10
    QSUB = 0x11
    QMUL = 0x12
    QDIV = 0x13

    FENCODE = 0x20
    FDECODE = 0x21
    FROUND = 0x22
    FERR = 0x23
    FAUTO = 0x24

    VERIFY = 0x30
    SEAL = 0x31

    JMP = 0x40
    JNZ = 0x41

    HALT = 0xFF


@dataclass(frozen=True, slots=True)
class Instruction128:
    opcode: Opcode
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
    def from_bytes(cls, encoded: bytes) -> "Instruction128":
        if len(encoded) != INSTRUCTION_BYTES:
            raise ValueError("one instruction must contain exactly 16 bytes")
        word = int.from_bytes(encoded, "big", signed=False)
        raw_opcode = (word >> 120) & 0xFF
        try:
            opcode = Opcode(raw_opcode)
        except ValueError as error:
            raise ValueError(f"unknown opcode 0x{raw_opcode:02x}") from error
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
class BytecodeProgram3D:
    shape: tuple[int, int, int]
    instructions: tuple[Instruction128, ...]

    def __post_init__(self) -> None:
        shape = _shape3(self.shape)
        if shape != self.shape:
            raise ValueError("program shape must contain normalized integer dimensions")
        if len(self.instructions) != _volume(self.shape):
            raise ValueError("instruction count must equal the 3D ROM volume")

    @classmethod
    def from_instructions(
        cls,
        instructions: Iterable[Instruction128],
        shape: Sequence[int] | None = None,
    ) -> "BytecodeProgram3D":
        items = tuple(instructions)
        if not items:
            raise ValueError("program must contain at least one instruction")
        target_shape = _shape3(shape or (len(items), 1, 1))
        volume = _volume(target_shape)
        if len(items) > volume:
            raise ValueError("program contains more instructions than the ROM volume")
        if len(items) < volume:
            items = items + (Instruction128(Opcode.NOP),) * (volume - len(items))
        return cls(shape=target_shape, instructions=items)

    @classmethod
    def from_bytes(cls, encoded: bytes, shape: Sequence[int]) -> "BytecodeProgram3D":
        shape3 = _shape3(shape)
        expected = _volume(shape3) * INSTRUCTION_BYTES
        if len(encoded) != expected:
            raise ValueError(f"bytecode contains {len(encoded)} bytes; expected {expected}")
        instructions = tuple(
            Instruction128.from_bytes(encoded[offset : offset + INSTRUCTION_BYTES])
            for offset in range(0, len(encoded), INSTRUCTION_BYTES)
        )
        return cls(shape=shape3, instructions=instructions)

    def to_bytes(self) -> bytes:
        return b"".join(instruction.to_bytes() for instruction in self.instructions)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def pc_xyz(self, pc: int) -> tuple[int, int, int]:
        if pc < 0 or pc >= len(self.instructions):
            raise IndexError("program counter is outside the ROM")
        sx, sy, _ = self.shape
        x = pc % sx
        y = (pc // sx) % sy
        z = pc // (sx * sy)
        return (x, y, z)


@dataclass(frozen=True, slots=True)
class ExecutionReport3D:
    protocol: str
    program_digest: str
    rom_shape: tuple[int, int, int]
    cycles: int
    halted: bool
    final_pc: int
    final_state_digest: str
    trace_digest: str
    register_voxels: int
    field_handles: tuple[int, ...]


class DrMoagiBytecodeEngine3D:
    """Bounded auto-executing 3D bytecode virtual machine."""

    def __init__(
        self,
        cloud: DrMoagiCloudOS | None = None,
        *,
        ledger_path: Path | None = None,
        default_node_cells: int = 1_000_000,
    ) -> None:
        if default_node_cells < 1:
            raise ValueError("default_node_cells must be positive")
        if cloud is not None and ledger_path is not None:
            raise ValueError("ledger_path cannot be supplied with an existing cloud runtime")
        self.cloud = cloud or DrMoagiCloudOS(ledger_path=ledger_path)
        if not self.cloud.nodes:
            self.cloud.register_node(
                "bytecode-local",
                max_cells=default_node_cells,
                max_concurrency=1,
            )
        self.registers: dict[tuple[int, int, int], list[int]] = {}
        self.fields: dict[int, Field3D] = {}
        self._field_digests: dict[int, str] = {}
        self.trace: list[dict[str, object]] = []
        self.trace_digest = ZERO_DIGEST
        self.program: BytecodeProgram3D | None = None
        self.pc = 0
        self.cycles = 0
        self.halted = False

    @staticmethod
    def _validate_handle(handle: int) -> int:
        handle = int(handle)
        if handle < 0 or handle > 0xFFFF:
            raise ValueError("field handle must be in the range 0..65535")
        return handle

    @staticmethod
    def _validate_register(register: int) -> int:
        register = int(register)
        if register < 0 or register >= REGISTER_COUNT:
            raise ValueError(f"register must be in the range 0..{REGISTER_COUNT - 1}")
        return register

    def _voxel(self, coordinate: tuple[int, int, int]) -> list[int]:
        if coordinate not in self.registers:
            self.registers[coordinate] = [0] * REGISTER_COUNT
        return self.registers[coordinate]

    def read_register(self, x: int, y: int, z: int, register: int) -> int:
        register = self._validate_register(register)
        return self._voxel((int(x), int(y), int(z)))[register]

    def read_q(self, x: int, y: int, z: int, register: int) -> float:
        return q_to_float(self.read_register(x, y, z, register))

    def _write_register(self, coordinate: tuple[int, int, int], register: int, value: int) -> None:
        register = self._validate_register(register)
        self._voxel(coordinate)[register] = _i64(value)

    @staticmethod
    def _field_digest(field: Field3D) -> str:
        return _digest({"shape": list(field.shape), "values": list(field.values)})

    def mount_field(self, handle: int, field: Field3D) -> None:
        handle = self._validate_handle(handle)
        self.fields[handle] = field
        self._field_digests[handle] = self._field_digest(field)

    def _store_field(self, handle: int, field: Field3D) -> None:
        self.mount_field(handle, field)

    def _field(self, handle: int) -> Field3D:
        handle = self._validate_handle(handle)
        try:
            return self.fields[handle]
        except KeyError as error:
            raise KeyError(f"field handle {handle} is not mounted") from error

    @staticmethod
    def _payload_field(payload: object) -> Field3D:
        if not isinstance(payload, dict):
            raise TypeError("cloud field result must be a dictionary")
        shape = payload.get("shape")
        values = payload.get("values")
        if not isinstance(shape, list) or not isinstance(values, list):
            raise TypeError("cloud field result must contain shape and values lists")
        return Field3D.from_values(values, shape)

    def _state_digest(self) -> str:
        return _digest(
            {
                "registers": [
                    {"xyz": list(coordinate), "values": values}
                    for coordinate, values in sorted(self.registers.items())
                ],
                "fields": [
                    {"handle": handle, "digest": self._field_digests[handle]}
                    for handle in sorted(self.fields)
                ],
            }
        )

    def load(self, program: BytecodeProgram3D, *, reset_state: bool = False) -> None:
        self.program = program
        self.pc = 0
        self.cycles = 0
        self.halted = False
        self.trace.clear()
        self.trace_digest = ZERO_DIGEST
        if reset_state:
            self.registers.clear()
            self.fields.clear()
            self._field_digests.clear()

    def _request_id(self, instruction: Instruction128, source: Field3D) -> str:
        assert self.program is not None
        return "bc3d-" + _digest(
            {
                "program": self.program.digest,
                "pc": self.pc,
                "opcode": int(instruction.opcode),
                "source": self._field_digest(source),
                "a": instruction.a,
                "b": instruction.b,
                "imm": instruction.imm,
            }
        )[:32]

    def _jump_target(self, target: int) -> int:
        assert self.program is not None
        if target < 0 or target >= len(self.program.instructions):
            raise IndexError(f"jump target {target} is outside the 3D ROM")
        return target

    def _execute(self, instruction: Instruction128, next_pc: int) -> tuple[int, bool]:
        coordinate = (instruction.x, instruction.y, instruction.z)
        destination = instruction.destination_register
        opcode = instruction.opcode
        branch_taken = False

        if opcode == Opcode.NOP:
            pass
        elif opcode == Opcode.QLOAD:
            self._write_register(coordinate, destination, _signed32(instruction.imm))
        elif opcode == Opcode.QMOV:
            source = self._validate_register(instruction.a)
            self._write_register(coordinate, destination, self._voxel(coordinate)[source])
        elif opcode == Opcode.VMOVE:
            source_register = self._validate_register(instruction.a)
            dx, dy, dz = unpack_offset(instruction.imm)
            source_coordinate = (
                coordinate[0] + dx,
                coordinate[1] + dy,
                coordinate[2] + dz,
            )
            if min(source_coordinate) < 0 or max(source_coordinate) > 0xFFFF:
                raise IndexError("neighbor move points outside the 16-bit voxel lattice")
            value = self._voxel(source_coordinate)[source_register]
            self._write_register(coordinate, destination, value)
        elif opcode in {Opcode.QADD, Opcode.QSUB, Opcode.QMUL, Opcode.QDIV}:
            register_a = self._validate_register(instruction.a)
            register_b = self._validate_register(instruction.b)
            left = self._voxel(coordinate)[register_a]
            right = self._voxel(coordinate)[register_b]
            if opcode == Opcode.QADD:
                value = left + right
            elif opcode == Opcode.QSUB:
                value = left - right
            elif opcode == Opcode.QMUL:
                value = _trunc_div(left * right, Q_ONE)
            else:
                value = _trunc_div(left << Q_SHIFT, right)
            self._write_register(coordinate, destination, value)
        elif opcode == Opcode.FENCODE:
            latent = self.cloud.encoder.encode(self._field(instruction.a), unpack_shape(instruction.imm))
            self._store_field(instruction.b, latent)
        elif opcode == Opcode.FDECODE:
            decoded = self.cloud.encoder.decode(self._field(instruction.a), unpack_shape(instruction.imm))
            self._store_field(instruction.b, decoded)
        elif opcode == Opcode.FROUND:
            if instruction.b == 0xFFFF:
                raise ValueError("FROUND output handle must leave room for the latent handle")
            source = self._field(instruction.a)
            job = self.cloud.round_trip(
                source,
                unpack_shape(instruction.imm),
                request_id=self._request_id(instruction, source),
            )
            if job.result is None:
                raise RuntimeError("cloud round-trip completed without a result")
            reconstruction = self._payload_field(job.result.get("reconstruction"))
            latent = self._payload_field(job.result.get("latent"))
            self._store_field(instruction.b, reconstruction)
            self._store_field(instruction.b + 1, latent)
            mse = job.result.get("mse")
            if not isinstance(mse, (int, float)):
                raise TypeError("cloud round-trip result is missing numeric mse")
            self._write_register(coordinate, destination, q_from_float(float(mse)))
        elif opcode == Opcode.FERR:
            left = self._field(instruction.a)
            right = self._field(instruction.b)
            if left.shape != right.shape:
                raise ValueError("FERR requires fields with identical shapes")
            mse = sum((a - b) ** 2 for a, b in zip(left.values, right.values)) / left.cells
            self._write_register(coordinate, destination, q_from_float(mse))
        elif opcode == Opcode.FAUTO:
            if instruction.b == 0xFFFF:
                raise ValueError("FAUTO output handle must leave room for the latent handle")
            complexity_weight = q_to_float(_signed32(instruction.imm))
            if complexity_weight < 0.0:
                raise ValueError("FAUTO complexity weight must be non-negative")
            source = self._field(instruction.a)
            job = self.cloud.auto_optimize(
                source,
                request_id=self._request_id(instruction, source),
                complexity_weight=complexity_weight,
            )
            if job.result is None:
                raise RuntimeError("cloud optimizer completed without a result")
            reconstruction = self._payload_field(job.result.get("reconstruction"))
            latent = self._payload_field(job.result.get("latent"))
            self._store_field(instruction.b, reconstruction)
            self._store_field(instruction.b + 1, latent)
            objective = job.result.get("objective")
            if not isinstance(objective, (int, float)):
                raise TypeError("cloud optimizer result is missing numeric objective")
            self._write_register(coordinate, destination, q_from_float(float(objective)))
        elif opcode == Opcode.VERIFY:
            self._write_register(coordinate, destination, Q_ONE if self.cloud.ledger.verify() else 0)
        elif opcode == Opcode.SEAL:
            self.cloud.ledger.append(
                "bytecode.sealed",
                {
                    "program_digest": self.program.digest if self.program else "",
                    "pc": self.pc,
                    "pc_xyz": list(self.program.pc_xyz(self.pc)) if self.program else [],
                    "state_digest": self._state_digest(),
                    "trace_digest": self.trace_digest,
                },
            )
        elif opcode == Opcode.JMP:
            next_pc = self._jump_target(instruction.imm)
            branch_taken = True
        elif opcode == Opcode.JNZ:
            source = self._validate_register(instruction.a)
            if self._voxel(coordinate)[source] != 0:
                next_pc = self._jump_target(instruction.imm)
                branch_taken = True
        elif opcode == Opcode.HALT:
            self.halted = True
        else:  # pragma: no cover - IntEnum decoding prevents this path.
            raise ValueError(f"unsupported opcode {opcode!r}")

        return next_pc, branch_taken

    def step(self) -> dict[str, object]:
        if self.program is None:
            raise RuntimeError("no bytecode program is loaded")
        if self.halted:
            raise RuntimeError("bytecode engine is halted")
        if self.pc < 0 or self.pc >= len(self.program.instructions):
            raise IndexError("program counter escaped the 3D ROM without HALT")

        current_pc = self.pc
        instruction = self.program.instructions[current_pc]
        before = self._state_digest()
        next_pc, branch_taken = self._execute(instruction, current_pc + 1)
        after = self._state_digest()
        entry: dict[str, object] = {
            "protocol": BYTECODE_PROTOCOL,
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

    def run(self, *, max_cycles: int = 100_000) -> ExecutionReport3D:
        if self.program is None:
            raise RuntimeError("no bytecode program is loaded")
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")

        self.cloud.ledger.append(
            "bytecode.run.started",
            {
                "program_digest": self.program.digest,
                "rom_shape": list(self.program.shape),
                "max_cycles": max_cycles,
            },
        )
        try:
            while not self.halted:
                if self.cycles >= max_cycles:
                    raise RuntimeError("bytecode cycle limit exceeded")
                self.step()

            report = ExecutionReport3D(
                protocol=BYTECODE_PROTOCOL,
                program_digest=self.program.digest,
                rom_shape=self.program.shape,
                cycles=self.cycles,
                halted=self.halted,
                final_pc=self.pc,
                final_state_digest=self._state_digest(),
                trace_digest=self.trace_digest,
                register_voxels=len(self.registers),
                field_handles=tuple(sorted(self.fields)),
            )
            self.cloud.ledger.append(
                "bytecode.run.committed",
                {
                    "program_digest": report.program_digest,
                    "cycles": report.cycles,
                    "final_pc": report.final_pc,
                    "final_state_digest": report.final_state_digest,
                    "trace_digest": report.trace_digest,
                },
            )
            if not self.cloud.ledger.verify():
                raise RuntimeError("ledger verification failed after bytecode commit")
            return report
        except Exception as error:
            self.cloud.ledger.append(
                "bytecode.run.failed",
                {
                    "program_digest": self.program.digest,
                    "cycles": self.cycles,
                    "pc": self.pc,
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


__all__ = [
    "BYTECODE_PROTOCOL",
    "INSTRUCTION_BYTES",
    "Q_ONE",
    "BytecodeProgram3D",
    "DrMoagiBytecodeEngine3D",
    "ExecutionReport3D",
    "Instruction128",
    "Opcode",
    "pack_offset",
    "pack_shape",
    "q_from_float",
    "q_to_float",
    "unpack_offset",
    "unpack_shape",
]
