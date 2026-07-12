from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


class Opcode(IntEnum):
    NOP = 0x00
    LOAD_INPUT = 0x01
    NORMALIZE = 0x02
    VOXELIZE = 0x03
    PREFETCH3D = 0x04
    ENCODE3D = 0x10
    PREDICT = 0x11
    COMPARE = 0x12
    UPDATE_OMEGA = 0x13
    DECODE3D = 0x14
    PROJECT_LAMBDA = 0x20
    STAGE = 0x21
    VERIFY = 0x22
    COMMIT = 0x23
    RENDER = 0x24
    SAMPLE_METRICS = 0x30
    OPTIMIZE_POLICY = 0x31
    JOURNAL = 0x32
    ADVANCE = 0x40
    JMP3D = 0x41
    HALT = 0xFF


class NumericFormat(IntEnum):
    INT4 = 0
    INT8 = 1
    FP16 = 2
    BF16 = 3
    FP32 = 4
    FP64 = 5
    SYMBOLIC = 15


class Phase(IntEnum):
    FETCH = 0
    SELECT = 1
    ENCODE = 2
    PREDICT = 3
    RESIDUAL = 4
    UPDATE = 5
    DECODE = 6
    COMMIT = 7
    OPTIMIZE = 8
    HALT = 15


def _crc8(payload: bytes) -> int:
    crc = 0
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
    return crc


@dataclass(frozen=True, slots=True)
class Instruction:
    """Fixed-width 128-bit VANN-ROM bytecode instruction.

    Layout: OP:8, FORMAT:4, PHASE:4, DST:8, SRCA:8, SRCB:8,
    LAMBDA:16, GEO:24, LENGTH:16, IMMEDIATE:24, CRC:8.
    """

    opcode: Opcode
    numeric_format: NumericFormat = NumericFormat.FP32
    phase: Phase = Phase.FETCH
    dst: int = 0
    src_a: int = 0
    src_b: int = 0
    lambda_mask: int = 0xFFFF
    geo: int = 0
    length: int = 0
    immediate: int = 0

    BYTE_WIDTH = 16

    def __post_init__(self) -> None:
        limits = {
            "dst": (self.dst, 0xFF),
            "src_a": (self.src_a, 0xFF),
            "src_b": (self.src_b, 0xFF),
            "lambda_mask": (self.lambda_mask, 0xFFFF),
            "geo": (self.geo, 0xFFFFFF),
            "length": (self.length, 0xFFFF),
        }
        for name, (value, maximum) in limits.items():
            if not 0 <= value <= maximum:
                raise ValueError(f"{name} outside valid range")
        if not -(2**23) <= self.immediate < 2**23:
            raise ValueError("immediate must fit signed 24-bit range")

    def encode(self) -> bytes:
        payload = bytearray()
        payload.append(int(self.opcode))
        payload.append((int(self.numeric_format) << 4) | int(self.phase))
        payload.extend((self.dst, self.src_a, self.src_b))
        payload.extend(self.lambda_mask.to_bytes(2, "big"))
        payload.extend(self.geo.to_bytes(3, "big"))
        payload.extend(self.length.to_bytes(2, "big"))
        payload.extend((self.immediate & 0xFFFFFF).to_bytes(3, "big"))
        payload.append(_crc8(bytes(payload)))
        return bytes(payload)

    @classmethod
    def decode(cls, data: bytes) -> "Instruction":
        if len(data) != cls.BYTE_WIDTH:
            raise ValueError("instruction must be exactly 16 bytes")
        if _crc8(data[:-1]) != data[-1]:
            raise ValueError("instruction CRC mismatch")
        fmt_phase = data[1]
        immediate = int.from_bytes(data[12:15], "big")
        if immediate & 0x800000:
            immediate -= 1 << 24
        return cls(
            opcode=Opcode(data[0]),
            numeric_format=NumericFormat((fmt_phase >> 4) & 0xF),
            phase=Phase(fmt_phase & 0xF),
            dst=data[2],
            src_a=data[3],
            src_b=data[4],
            lambda_mask=int.from_bytes(data[5:7], "big"),
            geo=int.from_bytes(data[7:10], "big"),
            length=int.from_bytes(data[10:12], "big"),
            immediate=immediate,
        )

    @classmethod
    def decode_stream(cls, data: bytes) -> list["Instruction"]:
        if not data:
            raise ValueError("bytecode image cannot be empty")
        if len(data) % cls.BYTE_WIDTH:
            raise ValueError("bytecode image length must be a multiple of 16 bytes")
        return [
            cls.decode(data[offset : offset + cls.BYTE_WIDTH])
            for offset in range(0, len(data), cls.BYTE_WIDTH)
        ]

    @classmethod
    def encode_stream(cls, instructions: Iterable["Instruction"]) -> bytes:
        encoded = b"".join(instruction.encode() for instruction in instructions)
        if not encoded:
            raise ValueError("instruction stream cannot be empty")
        return encoded
