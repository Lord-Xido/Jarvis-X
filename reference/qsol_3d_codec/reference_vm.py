"""Deterministic QSOL 3D auto-encoding/decoding bytecode reference VM.

This module implements the bounded research semantics from ADR-010.  It has no
external device I/O: ACTUATE_E writes only an in-memory register bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

Q16_ONE = 1 << 16
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

Coord3 = Tuple[int, int, int]
NeighborVector = Tuple[int, int, int, int, int, int]
RegisterValue = Union[int, NeighborVector]


class Opcode(IntEnum):
    HALT_LOOP = 0x00
    LDF_PSI = 0x01
    FETCH_NB = 0x02
    LAPLACE_3D = 0x03
    ENCODE_3D = 0x04
    DECODE_3D = 0x05
    ACTUATE_E = 0x06
    HAMILTON_CHK = 0x07
    SYNC_COMMIT = 0x08


# Byte 3 is a selector for these Q16.16 constants when an opcode requires a
# scalar coefficient.  This is deliberately not described as an 8-bit
# immediate, because the represented constants are wider than one byte.
CONST_POOL: Mapping[int, int] = {
    0x10: 0x00010000,  # 1.0
    0x33: 0x00003333,  # ~0.2
    0x40: 0x00004000,  # 0.25
    0x9A: 0x0000D99A,  # ~0.85
}

MEM_NULL = 0x00
MEM_OFF = 0x01


# Preserved exactly as submitted.  It is intentionally non-canonical because
# several source-register bytes encode decimal register labels as hexadecimal
# values (for example R48 as 0x48 rather than decimal 48 == 0x30).
PROGRAM_SUBMITTED_DRAFT = bytes(
    [
        0x01,
        0x10,
        0x00,
        0x10,
        0x01,
        0x11,
        0x01,
        0x33,
        0x02,
        0x20,
        0x10,
        0x06,
        0x03,
        0x30,
        0x20,
        0x10,
        0x04,
        0x40,
        0x48,
        0x40,
        0x07,
        0x50,
        0x64,
        0x00,
        0x05,
        0x60,
        0x64,
        0x9A,
        0x06,
        0x70,
        0x96,
        0x00,
        0x08,
        0x10,
        0x70,
        0x01,
        0x00,
        0x00,
        0x00,
        0x00,
    ]
)


# Corrected executable stream for the 256-register four-byte profile.
PROGRAM_CANONICAL_V1 = bytes(
    [
        0x01,
        0x10,
        0x00,
        0x10,  # LDF_PSI R16, MEM_NULL, CONST[0x10]
        0x01,
        0x11,
        0x01,
        0x33,  # LDF_PSI R17, MEM_OFF, CONST[0x33]
        0x02,
        0x20,
        0x10,
        0x06,  # FETCH_NB R32, R16, 6
        0x03,
        0x30,
        0x20,
        0x10,  # LAPLACE_3D R48, R32, center R16
        0x04,
        0x40,
        0x30,
        0x40,  # ENCODE_3D R64, R48, gain 0.25
        0x07,
        0x50,
        0x40,
        0x00,  # HAMILTON_CHK R80, R64, profile 0
        0x05,
        0x60,
        0x40,
        0x9A,  # DECODE_3D R96, R64, gain ~0.85
        0x06,
        0x70,
        0x60,
        0x00,  # ACTUATE_E R112, R96, internal sink
        0x08,
        0x10,
        0x70,
        0x01,  # SYNC_COMMIT R16, R112, residual-add
        0x00,
        0x00,
        0x00,
        0x00,  # HALT_LOOP epoch fence
    ]
)


REGISTER_CORRECTIONS: Mapping[int, Tuple[int, int]] = {
    # word_index: (submitted_source_byte, canonical_source_byte)
    4: (0x48, 0x30),  # R48 decimal == 0x30
    5: (0x64, 0x40),  # R64 decimal == 0x40
    6: (0x64, 0x40),  # R64 decimal == 0x40
    7: (0x96, 0x60),  # R96 decimal == 0x60
}


def sat_i32(value: int) -> int:
    """Saturate an integer to signed 32-bit range."""

    return max(INT32_MIN, min(INT32_MAX, int(value)))


def trunc_div(numerator: int, denominator: int) -> int:
    """Integer division truncated toward zero, independent of Python // sign rules."""

    if denominator == 0:
        raise ZeroDivisionError("division by zero")
    quotient = abs(numerator) // abs(denominator)
    return -quotient if (numerator < 0) ^ (denominator < 0) else quotient


def q16_mul(lhs: int, rhs: int) -> int:
    """Saturating signed Q16.16 multiply with a wide intermediate."""

    product = int(lhs) * int(rhs)
    return sat_i32(trunc_div(product, Q16_ONE))


def q16_to_float(value: int) -> float:
    return int(value) / Q16_ONE


@dataclass(frozen=True)
class Instruction:
    opcode: int
    dest: int
    src: int
    modifier: int

    @classmethod
    def decode(cls, word: bytes) -> "Instruction":
        if len(word) != 4:
            raise ValueError("instruction must be exactly four bytes")
        return cls(*word)

    def encode(self) -> bytes:
        return bytes((self.opcode, self.dest, self.src, self.modifier))


@dataclass(frozen=True)
class TraceRecord:
    cycle: int
    pc: int
    opcode: Opcode
    note: str


class QSOL3DVM:
    """Small SIMD-style toroidal reference interpreter.

    Each lattice coordinate owns a 256-entry register bank.  Instructions are
    applied over the complete lattice from a consistent source snapshot where
    neighborhood or atomic-commit semantics require it.
    """

    def __init__(self, shape: Tuple[int, int, int] = (3, 3, 3)) -> None:
        if len(shape) != 3 or any(axis <= 0 for axis in shape):
            raise ValueError("shape must contain three positive extents")

        self.shape = shape
        self.coords: List[Coord3] = [
            (x, y, z)
            for z in range(shape[2])
            for y in range(shape[1])
            for x in range(shape[0])
        ]
        self.registers: Dict[Coord3, List[RegisterValue]] = {
            coord: [0] * 256 for coord in self.coords
        }
        self.actuation_register: Dict[Coord3, int] = {coord: 0 for coord in self.coords}
        self.trace: List[TraceRecord] = []
        self.halted = False
        self.commit_generation = 0
        self.last_commit_drift = 0

    def neighbor_coords(self, coord: Coord3) -> NeighborVector:  # type: ignore[override]
        """Return six wrapped coordinate tuples.

        The return annotation is intentionally replaced below at runtime by the
        actual tuple-of-coordinate shape; keeping the implementation explicit
        avoids any hidden addressing rules.
        """

        x, y, z = coord
        sx, sy, sz = self.shape
        return (  # type: ignore[return-value]
            ((x + 1) % sx, y, z),
            ((x - 1) % sx, y, z),
            (x, (y + 1) % sy, z),
            (x, (y - 1) % sy, z),
            (x, y, (z + 1) % sz),
            (x, y, (z - 1) % sz),
        )

    def _scalar_snapshot(self, register_id: int) -> Dict[Coord3, int]:
        snapshot: Dict[Coord3, int] = {}
        for coord in self.coords:
            value = self.registers[coord][register_id]
            if not isinstance(value, int):
                raise TypeError(f"R{register_id} does not contain scalar Q16.16 data")
            snapshot[coord] = value
        return snapshot

    @staticmethod
    def _constant(selector: int) -> int:
        try:
            return CONST_POOL[selector]
        except KeyError as exc:
            raise ValueError(f"unknown Q16.16 constant selector 0x{selector:02X}") from exc

    def execute(self, instruction: Instruction, *, cycle: int, pc: int) -> None:
        try:
            opcode = Opcode(instruction.opcode)
        except ValueError as exc:
            raise ValueError(f"unknown opcode 0x{instruction.opcode:02X}") from exc

        note = ""

        if opcode == Opcode.HALT_LOOP:
            self.halted = True
            note = "end-of-epoch fence"

        elif opcode == Opcode.LDF_PSI:
            if instruction.src not in (MEM_NULL, MEM_OFF):
                raise ValueError("LDF_PSI source tag must be MEM_NULL or MEM_OFF in v1")
            value = self._constant(instruction.modifier)
            for coord in self.coords:
                self.registers[coord][instruction.dest] = value
            note = f"R{instruction.dest}=0x{value & 0xFFFFFFFF:08X}"

        elif opcode == Opcode.FETCH_NB:
            if instruction.modifier != 0x06:
                raise ValueError("FETCH_NB v1 requires exactly six orthogonal neighbors")
            source = self._scalar_snapshot(instruction.src)
            for coord in self.coords:
                neighbors = tuple(source[n] for n in self.neighbor_coords(coord))
                self.registers[coord][instruction.dest] = neighbors  # type: ignore[assignment]
            note = f"six-neighbor toroidal stencil from R{instruction.src}"

        elif opcode == Opcode.LAPLACE_3D:
            center = self._scalar_snapshot(instruction.modifier)
            for coord in self.coords:
                value = self.registers[coord][instruction.src]
                if not (isinstance(value, tuple) and len(value) == 6):
                    raise TypeError("LAPLACE_3D source must contain a six-value neighbor vector")
                neighbor_mean = trunc_div(sum(value), 6)
                self.registers[coord][instruction.dest] = sat_i32(neighbor_mean - center[coord])
            note = (
                f"normalized Laplacian mean(R{instruction.src})-R{instruction.modifier}"
            )

        elif opcode == Opcode.ENCODE_3D:
            gain = self._constant(instruction.modifier)
            source = self._scalar_snapshot(instruction.src)
            for coord in self.coords:
                self.registers[coord][instruction.dest] = q16_mul(source[coord], gain)
            note = f"Q16 encode gain={q16_to_float(gain):.9f}"

        elif opcode == Opcode.DECODE_3D:
            gain = self._constant(instruction.modifier)
            source = self._scalar_snapshot(instruction.src)
            for coord in self.coords:
                self.registers[coord][instruction.dest] = q16_mul(source[coord], gain)
            note = f"Q16 decode gain={q16_to_float(gain):.9f}"

        elif opcode == Opcode.ACTUATE_E:
            if instruction.modifier != 0x00:
                raise ValueError("ACTUATE_E v1 supports only internal-sink profile 0")
            source = self._scalar_snapshot(instruction.src)
            for coord in self.coords:
                value = source[coord]
                self.registers[coord][instruction.dest] = value
                self.actuation_register[coord] = value
            note = "internal simulated actuation register only"

        elif opcode == Opcode.HAMILTON_CHK:
            if instruction.modifier != 0x00:
                raise ValueError("HAMILTON_CHK v1 supports only profile 0")
            source = self._scalar_snapshot(instruction.src)
            max_energy = 0
            for coord in self.coords:
                squared = q16_mul(source[coord], source[coord])
                energy = trunc_div(squared, 2)
                self.registers[coord][instruction.dest] = energy
                max_energy = max(max_energy, abs(energy))
            note = f"H_eff proxy max={q16_to_float(max_energy):.9f}"

        elif opcode == Opcode.SYNC_COMMIT:
            source = self._scalar_snapshot(instruction.src)
            previous = self._scalar_snapshot(instruction.dest)
            staged: Dict[Coord3, int] = {}

            if instruction.modifier == 0x01:
                mode = "residual-add"
                for coord in self.coords:
                    staged[coord] = sat_i32(previous[coord] + source[coord])
            elif instruction.modifier == 0x00:
                mode = "replace"
                staged = dict(source)
            else:
                raise ValueError("SYNC_COMMIT modifier must be 0x00 or 0x01")

            self.last_commit_drift = max(
                abs(staged[coord] - previous[coord]) for coord in self.coords
            )
            for coord, value in staged.items():
                self.registers[coord][instruction.dest] = value
            self.commit_generation += 1
            note = (
                f"atomic {mode}; max drift={q16_to_float(self.last_commit_drift):.9f}"
            )

        self.trace.append(TraceRecord(cycle=cycle, pc=pc, opcode=opcode, note=note))

    def run_once(self, program: bytes = PROGRAM_CANONICAL_V1) -> Sequence[TraceRecord]:
        if len(program) % 4 != 0:
            raise ValueError("program length must be a multiple of four bytes")

        self.halted = False
        self.trace = []

        for pc in range(0, len(program), 4):
            cycle = (pc // 4) + 1
            instruction = Instruction.decode(program[pc : pc + 4])
            self.execute(instruction, cycle=cycle, pc=pc)
            if self.halted:
                break

        return tuple(self.trace)


def iter_instructions(program: bytes) -> Iterable[Instruction]:
    if len(program) % 4 != 0:
        raise ValueError("program length must be a multiple of four bytes")
    for pc in range(0, len(program), 4):
        yield Instruction.decode(program[pc : pc + 4])


def audit_submitted_register_bytes() -> List[str]:
    """Return the known source-register corrections in human-readable form."""

    draft = list(iter_instructions(PROGRAM_SUBMITTED_DRAFT))
    canonical = list(iter_instructions(PROGRAM_CANONICAL_V1))
    findings: List[str] = []

    for word_index, (submitted, corrected) in REGISTER_CORRECTIONS.items():
        if draft[word_index].src != submitted:
            raise AssertionError("submitted compatibility fixture changed unexpectedly")
        if canonical[word_index].src != corrected:
            raise AssertionError("canonical register correction changed unexpectedly")
        findings.append(
            f"word {word_index}: source 0x{submitted:02X} -> 0x{corrected:02X}"
        )

    return findings


def format_program_hex(program: bytes = PROGRAM_CANONICAL_V1) -> str:
    return " ".join(program[i : i + 4].hex().upper() for i in range(0, len(program), 4))


def main() -> int:
    vm = QSOL3DVM()
    trace = vm.run_once()

    print("QSOL 3D codec bytecode v1")
    print(format_program_hex())
    print()
    for record in trace:
        print(
            f"[CYCLE {record.cycle:03d}] PC 0x{record.pc:04X} "
            f"{record.opcode.name:<13} -> {record.note}"
        )

    print()
    print(f"commit_generation={vm.commit_generation}")
    print(f"max_commit_drift_q16={vm.last_commit_drift}")
    print(f"fixed_point={vm.last_commit_drift == 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
