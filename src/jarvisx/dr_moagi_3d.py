"""Sparse, bit-accurate Dr Moagi M.M ROM Ω³ 6400³ reference runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Dict, List, Sequence, Tuple

AXIS = 6400
CUBE_EDGE = 64
CUBES_PER_AXIS = 100
CELL_BYTES = 16
TOTAL_CELLS = AXIS**3
TOTAL_BYTES = TOTAL_CELLS * CELL_BYTES
CUBE_BYTES = CUBE_EDGE**3 * CELL_BYTES
Q14_SHIFT = 14
Q14_ONE = 1 << Q14_SHIFT
I8_MIN, I8_MAX = -128, 127
I16_MIN, I16_MAX = -32768, 32767
U38_MASK = (1 << 38) - 1
U48_MASK = (1 << 48) - 1
U50_MASK = (1 << 50) - 1
U128_MASK = (1 << 128) - 1


def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def sat_i8(v: int) -> int:
    return clamp(v, I8_MIN, I8_MAX)


def sat_i16(v: int) -> int:
    return clamp(v, I16_MIN, I16_MAX)


def rounded_shift(v: int, shift: int) -> int:
    if shift < 0:
        raise ValueError("shift must be non-negative")
    if shift == 0:
        return v
    half = 1 << (shift - 1)
    return (v + half) >> shift if v >= 0 else -(((-v) + half) >> shift)


def q14_from_byte(v: int) -> int:
    if not 0 <= v <= 255:
        raise ValueError("byte must be in [0, 255]")
    return (v - 128) << 7


def q14_to_byte(v: int) -> int:
    return clamp((sat_i16(v) >> 7) + 128, 0, 255)


def q14_mul(a: int, b: int) -> int:
    return sat_i16(rounded_shift(a * b, Q14_SHIFT))


def quantize_q14_to_i8(v: int) -> int:
    return sat_i8(rounded_shift(sat_i16(v), 7))


def dequantize_i8_to_q14(v: int) -> int:
    if not I8_MIN <= v <= I8_MAX:
        raise ValueError("value must fit signed int8")
    return v << 7


def _u8(v: int) -> int:
    if not I8_MIN <= v <= I8_MAX:
        raise ValueError("value must fit signed int8")
    return v & 255


def _i8(v: int) -> int:
    return v - 256 if v & 128 else v


@dataclass(frozen=True)
class Coordinate3D:
    x: int
    y: int
    z: int

    def __post_init__(self) -> None:
        if any(not 0 <= v < AXIS for v in (self.x, self.y, self.z)):
            raise ValueError("coordinates must be in [0, 6399]")

    @property
    def cell_address(self) -> int:
        return self.x + AXIS * (self.y + AXIS * self.z)

    @property
    def byte_address(self) -> int:
        return self.cell_address << 4

    @property
    def cube_coordinate(self) -> Tuple[int, int, int]:
        return self.x >> 6, self.y >> 6, self.z >> 6

    @property
    def local_coordinate(self) -> Tuple[int, int, int]:
        return self.x & 63, self.y & 63, self.z & 63

    @property
    def cube_id(self) -> int:
        cx, cy, cz = self.cube_coordinate
        return cx + 100 * (cy + 100 * cz)

    @property
    def local_address(self) -> int:
        lx, ly, lz = self.local_coordinate
        return lx | (ly << 6) | (lz << 12)

    @property
    def sparse_key(self) -> int:
        return (self.cube_id << 18) | self.local_address

    @classmethod
    def from_cell_address(cls, address: int) -> "Coordinate3D":
        if not 0 <= address < TOTAL_CELLS:
            raise ValueError("cell address out of range")
        z, rem = divmod(address, AXIS * AXIS)
        y, x = divmod(rem, AXIS)
        return cls(x, y, z)


class CellClass(IntEnum):
    RAW_INPUT = 1
    FEATURE = 2
    LATENT = 3
    BYTECODE = 4
    OUTPUT = 5
    RESIDUAL = 6
    OMEGA = 7
    TRACE = 8
    CANDIDATE = 9
    VERIFICATION = 10
    JOURNAL = 11
    INTEGRITY = 12


class CellState(IntFlag):
    VALID = 1 << 0
    DIRTY = 1 << 1
    CANDIDATE = 1 << 2
    VERIFIED = 1 << 3
    COMMITTED = 1 << 4
    IMMUTABLE = 1 << 5
    TOMBSTONE = 1 << 6


def pack_i8x6(values: Sequence[int]) -> int:
    if len(values) != 6:
        raise ValueError("six signed int8 values required")
    out = 0
    for i, value in enumerate(values):
        out |= _u8(value) << (8 * i)
    return out


def unpack_i8x6(payload: int) -> Tuple[int, int, int, int, int, int]:
    if not 0 <= payload <= U48_MASK:
        raise ValueError("payload must fit 48 bits")
    values = [_i8((payload >> (8 * i)) & 255) for i in range(6)]
    return tuple(values)  # type: ignore[return-value]


@dataclass(frozen=True)
class RomCell:
    cell_class: CellClass
    state: CellState
    version: int
    evidence: int
    parent: int
    payload: int

    def __post_init__(self) -> None:
        if not 0 <= self.version <= 0xFFFF:
            raise ValueError("version must fit 16 bits")
        if not 0 <= self.evidence <= 0xFFFF:
            raise ValueError("evidence must fit 16 bits")
        if not 0 <= self.parent <= 0xFFFFFFFF:
            raise ValueError("parent must fit 32 bits")
        if not 0 <= self.payload <= U48_MASK:
            raise ValueError("payload must fit 48 bits")

    def pack(self) -> int:
        return ((int(self.cell_class) << 120) | (int(self.state) << 112) |
                (self.version << 96) | (self.evidence << 80) |
                (self.parent << 48) | self.payload)

    @classmethod
    def unpack(cls, value: int) -> "RomCell":
        if not 0 <= value <= U128_MASK:
            raise ValueError("cell must fit 128 bits")
        return cls(CellClass((value >> 120) & 255), CellState((value >> 112) & 255),
                   (value >> 96) & 0xFFFF, (value >> 80) & 0xFFFF,
                   (value >> 48) & 0xFFFFFFFF, value & U48_MASK)

    @classmethod
    def latent(cls, values: Sequence[int], version: int = 0, parent: int = 0) -> "RomCell":
        return cls(CellClass.LATENT, CellState.VALID | CellState.CANDIDATE,
                   version, 0, parent, pack_i8x6(values))


class Opcode(IntEnum):
    OBSERVE = 0x01
    NORMALISE = 0x02
    ENCODE = 0x03
    QUANTISE = 0x04
    ROUTE3D = 0x05
    READROM = 0x06
    WRITETX = 0x07
    FUSE = 0x08
    EXECUTE = 0x09
    DECODE = 0x0A
    COMPARE = 0x0B
    OMEGA = 0x0C
    VERIFY = 0x0D
    COMMIT = 0x0E
    ROLLBACK = 0x0F
    SELF_SCAN = 0x20
    SELF_ENCODE = 0x21
    SELF_FORK = 0x22
    SELF_MUTATE = 0x23
    SELF_COMPILE = 0x24
    SELF_TEST = 0x25
    SELF_SCORE = 0x26
    SELF_SEAL = 0x27


@dataclass(frozen=True)
class Instruction:
    opcode: Opcode
    mode: int = 0
    dst: int = 0
    src: int = 0
    coordinate: int = 0
    immediate: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.mode <= 255:
            raise ValueError("mode must fit 8 bits")
        if not 0 <= self.dst <= 0xFFF or not 0 <= self.src <= 0xFFF:
            raise ValueError("registers must fit 12 bits")
        if not 0 <= self.coordinate < TOTAL_CELLS:
            raise ValueError("coordinate out of range")
        if not 0 <= self.immediate <= U50_MASK:
            raise ValueError("immediate must fit 50 bits")

    def pack(self) -> int:
        return ((int(self.opcode) << 120) | (self.mode << 112) |
                (self.dst << 100) | (self.src << 88) |
                (self.coordinate << 50) | self.immediate)

    @classmethod
    def unpack(cls, value: int) -> "Instruction":
        if not 0 <= value <= U128_MASK:
            raise ValueError("instruction must fit 128 bits")
        return cls(Opcode((value >> 120) & 255), (value >> 112) & 255,
                   (value >> 100) & 0xFFF, (value >> 88) & 0xFFF,
                   (value >> 50) & U38_MASK, value & U50_MASK)


class VerificationBit(IntFlag):
    PARSED = 1 << 0
    SHAPES = 1 << 1
    MEMORY_BOUNDS = 1 << 2
    OPCODE_LEGALITY = 1 << 3
    DETERMINISTIC_REPLAY = 1 << 4
    UNIT_TESTS = 1 << 5
    INTEGRATION_TESTS = 1 << 6
    HELD_OUT_TESTS = 1 << 7
    SECURITY_TESTS = 1 << 8
    RESOURCE_LIMITS = 1 << 9
    SEMANTIC_EQUIVALENCE = 1 << 10
    NO_REGRESSION = 1 << 11
    AUTHORISED = 1 << 12


REQUIRED_VERIFICATION = VerificationBit((1 << 13) - 1)


def bitwise_select_u128(active: int, candidate: int, commit: bool) -> int:
    if any(not 0 <= v <= U128_MASK for v in (active, candidate)):
        raise ValueError("states must fit 128 bits")
    mask = U128_MASK if commit else 0
    return (mask & candidate) | ((U128_MASK ^ mask) & active)


def commit_gate(active: int, candidate: int, verification: VerificationBit, *,
                candidate_loss: int, active_loss: int,
                required_margin: int = 0) -> Tuple[int, bool]:
    verified = (verification & REQUIRED_VERIFICATION) == REQUIRED_VERIFICATION
    commit = bool(verified and candidate_loss + required_margin < active_loss)
    return bitwise_select_u128(active, candidate, commit), commit


@dataclass
class SparseRom:
    """Only touched cells are materialised; the logical lattice remains 6400³."""
    cells: Dict[int, int] = field(default_factory=dict)
    immutable: set = field(default_factory=set)

    def read(self, coordinate: Coordinate3D) -> int:
        return self.cells.get(coordinate.cell_address, 0)

    def begin(self) -> "RomTransaction":
        return RomTransaction(self)


@dataclass
class RomTransaction:
    rom: SparseRom
    staged: Dict[int, int] = field(default_factory=dict)
    closed: bool = False

    def write(self, coordinate: Coordinate3D, value: int) -> None:
        if self.closed:
            raise RuntimeError("transaction closed")
        address = coordinate.cell_address
        if address in self.rom.immutable:
            raise PermissionError("immutable cell")
        if not 0 <= value <= U128_MASK:
            raise ValueError("cell must fit 128 bits")
        self.staged[address] = value

    def rollback(self) -> None:
        self.staged.clear()
        self.closed = True

    def commit(self, verification: VerificationBit, *, candidate_loss: int,
               active_loss: int, required_margin: int = 0,
               make_immutable: bool = False) -> bool:
        if self.closed:
            raise RuntimeError("transaction closed")
        accepted = ((verification & REQUIRED_VERIFICATION) == REQUIRED_VERIFICATION and
                    candidate_loss + required_margin < active_loss)
        if accepted:
            self.rom.cells.update(self.staged)
            if make_immutable:
                self.rom.immutable.update(self.staged)
        self.staged.clear()
        self.closed = True
        return bool(accepted)


@dataclass(frozen=True)
class FixedPointLinear:
    weights: Tuple[Tuple[int, ...], ...]
    bias: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.weights or not self.weights[0]:
            raise ValueError("weights cannot be empty")
        width = len(self.weights[0])
        if any(len(row) != width for row in self.weights):
            raise ValueError("ragged weight matrix")
        if len(self.bias) != len(self.weights):
            raise ValueError("bias/output mismatch")

    @property
    def input_width(self) -> int:
        return len(self.weights[0])

    @property
    def output_width(self) -> int:
        return len(self.weights)

    def __call__(self, values: Sequence[int]) -> Tuple[int, ...]:
        if len(values) != self.input_width:
            raise ValueError("input width mismatch")
        output: List[int] = []
        for row, bias in zip(self.weights, self.bias):
            acc = bias << Q14_SHIFT
            for weight, value in zip(row, values):
                acc += weight * value
            output.append(sat_i16(rounded_shift(acc, Q14_SHIFT)))
        return tuple(output)

    @classmethod
    def identity(cls, width: int) -> "FixedPointLinear":
        rows = tuple(tuple(Q14_ONE if i == j else 0 for j in range(width))
                     for i in range(width))
        return cls(rows, tuple(0 for _ in range(width)))


@dataclass(frozen=True)
class EncodedBlock:
    coordinate: Coordinate3D
    latent_q14: Tuple[int, ...]
    quantized: Tuple[int, int, int, int, int, int]
    cell: RomCell


@dataclass(frozen=True)
class GeometricCodec:
    encoder: FixedPointLinear
    decoder: FixedPointLinear

    def __post_init__(self) -> None:
        if self.encoder.output_width != 6 or self.decoder.input_width != 6:
            raise ValueError("one 128-bit cell requires six latent channels")
        if self.decoder.output_width != self.encoder.input_width:
            raise ValueError("encoder/decoder width mismatch")

    @classmethod
    def identity(cls, width: int = 6) -> "GeometricCodec":
        if width != 6:
            raise ValueError("reference cell width is six")
        linear = FixedPointLinear.identity(width)
        return cls(linear, linear)

    @staticmethod
    def _coordinate(latent: Sequence[int]) -> Coordinate3D:
        def project(v: int) -> int:
            return min(AXIS - 1, ((sat_i16(v) + 16384) * AXIS) >> 15)
        return Coordinate3D(project(latent[0]), project(latent[1]), project(latent[2]))

    def encode(self, data: bytes, *, version: int = 0, parent: int = 0) -> EncodedBlock:
        if len(data) != self.encoder.input_width:
            raise ValueError("input block width mismatch")
        latent = self.encoder(tuple(q14_from_byte(v) for v in data))
        quantized = tuple(quantize_q14_to_i8(v) for v in latent)
        return EncodedBlock(self._coordinate(latent), latent, quantized,  # type: ignore[arg-type]
                            RomCell.latent(quantized, version, parent))

    def decode(self, cell_value: int) -> bytes:
        cell = RomCell.unpack(cell_value)
        if cell.cell_class != CellClass.LATENT:
            raise ValueError("not a latent cell")
        latent = tuple(dequantize_i8_to_q14(v) for v in unpack_i8x6(cell.payload))
        return bytes(q14_to_byte(v) for v in self.decoder(latent))

    def encode_into(self, tx: RomTransaction, data: bytes, *, version: int = 0) -> EncodedBlock:
        block = self.encode(data, version=version)
        tx.write(block.coordinate, block.cell.pack())
        return block


def omega_update(omega_q14: int, error_q14: int) -> int:
    return sat_i16(omega_q14 - (omega_q14 >> 3) + (error_q14 >> 4))


def reconstruction_error(actual: bytes, decoded: bytes) -> Tuple[int, ...]:
    if len(actual) != len(decoded):
        raise ValueError("length mismatch")
    return tuple(sat_i16(q14_from_byte(a) - q14_from_byte(b))
                 for a, b in zip(actual, decoded))


@dataclass
class DrMoagiEngine:
    codec: GeometricCodec = field(default_factory=GeometricCodec.identity)
    rom: SparseRom = field(default_factory=SparseRom)
    omega: Tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    version: int = 0

    def cycle(self, data: bytes, verification: VerificationBit, *,
              candidate_loss: int, active_loss: int,
              required_margin: int = 0) -> Tuple[bytes, bool, Coordinate3D]:
        tx = self.rom.begin()
        block = self.codec.encode_into(tx, data, version=self.version + 1)
        decoded = self.codec.decode(block.cell.pack())
        errors = reconstruction_error(data, decoded)
        candidate_omega = tuple(omega_update(o, e) for o, e in zip(self.omega, errors))
        committed = tx.commit(verification, candidate_loss=candidate_loss,
                              active_loss=active_loss, required_margin=required_margin)
        if committed:
            self.version += 1
            self.omega = candidate_omega
        return decoded, committed, block.coordinate

    def self_encode(self, authorised_state: bytes) -> EncodedBlock:
        """Encode one authorised six-byte self-state frame through the same path."""
        return self.codec.encode(authorised_state, version=self.version)


__all__ = [
    "AXIS", "CELL_BYTES", "CUBE_BYTES", "TOTAL_BYTES", "Coordinate3D",
    "CellClass", "CellState", "RomCell", "Opcode", "Instruction",
    "VerificationBit", "REQUIRED_VERIFICATION", "SparseRom", "RomTransaction",
    "FixedPointLinear", "GeometricCodec", "EncodedBlock", "DrMoagiEngine",
    "q14_from_byte", "q14_to_byte", "q14_mul", "quantize_q14_to_i8",
    "dequantize_i8_to_q14", "omega_update", "commit_gate",
]
