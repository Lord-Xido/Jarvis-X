"""Deterministic 64-byte Dr. Moagi 3D bytecode ROM image.

The ROM is a compact instruction descriptor. It does not implement convolution,
quantization, GPU dispatch, or neural-network kernels by itself; a host runtime
must bind those opcode semantics explicitly.
"""

from __future__ import annotations

import argparse
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MAGIC = b"MOAG"
WORD_COUNT = 16
ROM_BYTES = WORD_COUNT * 4
CRC_OFFSET = ROM_BYTES - 4
SUPPLIED_CRC32 = 0x7E3F91A2

# Fifteen body words followed by a computed IEEE CRC-32 word.
ROM_BODY_WORDS: tuple[int, ...] = (
    0x4D4F4147,  # [0x0000] MAGIC "MOAG"
    0x01000010,  # [0x0004] HEADER: total image length = 16 words
    0x04000040,  # [0x0008] INIT_GRID: side length = 64
    0x04200000,  # [0x000C] LOAD_COORD
    0x04400001,  # [0x0010] CLEAR_ACC
    0x08610800,  # [0x0014] ENC_PROJ layer 1
    0x0C831004,  # [0x0018] QUANT_Z
    0x08A41800,  # [0x001C] ENC_PROJ layer 2
    0x10C52000,  # [0x0020] DEC_DECONV layer 1
    0x10E62800,  # [0x0024] DEC_DECONV layer 2
    0x14E73000,  # [0x0028] SOL_CURL
    0x18E70008,  # [0x002C] DAMP_REG
    0x1CE70000,  # [0x0030] SYNC_OUT
    0x00000000,  # [0x0034] NOP by slot contract
    0x00000000,  # [0x0038] HALT by slot contract
)

SLOT_OPERATIONS: tuple[str, ...] = (
    "MAGIC",
    "HEADER",
    "INIT_GRID",
    "LOAD_COORD",
    "CLEAR_ACC",
    "ENC_PROJ_1",
    "QUANT_Z",
    "ENC_PROJ_2",
    "DEC_DECONV_1",
    "DEC_DECONV_2",
    "SOL_CURL",
    "DAMP_REG",
    "SYNC_OUT",
    "NOP",
    "HALT",
    "CRC32",
)


@dataclass(frozen=True)
class DecodedWord:
    """Bit-field view of one 32-bit ROM word."""

    index: int
    address: int
    word: int
    opcode: int
    rd: int
    rs1: int
    rs2: int
    immediate: int
    symbolic_operation: str


@dataclass(frozen=True)
class VerificationResult:
    """Deterministic verification result for a candidate ROM image."""

    valid: bool
    magic_valid: bool
    length_valid: bool
    declared_word_count_valid: bool
    crc_valid: bool
    expected_crc32: int
    stored_crc32: int


def _pack_words(words: Sequence[int]) -> bytes:
    if len(words) != WORD_COUNT:
        raise ValueError(f"expected {WORD_COUNT} words, received {len(words)}")
    if any(word < 0 or word > 0xFFFFFFFF for word in words):
        raise ValueError("every ROM word must fit in an unsigned 32-bit integer")
    return struct.pack(f">{WORD_COUNT}I", *words)


def compute_crc32(payload: bytes) -> int:
    """Return the unsigned IEEE CRC-32 used by Python/zlib."""

    return zlib.crc32(payload) & 0xFFFFFFFF


def build_rom() -> bytes:
    """Build the canonical big-endian ROM image with a computed checksum."""

    body = struct.pack(f">{len(ROM_BODY_WORDS)}I", *ROM_BODY_WORDS)
    checksum = compute_crc32(body)
    return body + struct.pack(">I", checksum)


CANONICAL_ROM_BYTES = build_rom()
CANONICAL_CRC32 = struct.unpack(">I", CANONICAL_ROM_BYTES[CRC_OFFSET:])[0]
CANONICAL_WORDS: tuple[int, ...] = struct.unpack(f">{WORD_COUNT}I", CANONICAL_ROM_BYTES)

SUPPLIED_ROM_BYTES = struct.pack(
    f">{WORD_COUNT}I",
    *ROM_BODY_WORDS,
    SUPPLIED_CRC32,
)


def verify_rom(rom: bytes) -> VerificationResult:
    """Verify image length, magic, header word count, and terminal CRC-32."""

    length_valid = len(rom) == ROM_BYTES
    if not length_valid:
        return VerificationResult(
            valid=False,
            magic_valid=False,
            length_valid=False,
            declared_word_count_valid=False,
            crc_valid=False,
            expected_crc32=0,
            stored_crc32=0,
        )

    words = struct.unpack(f">{WORD_COUNT}I", rom)
    magic_valid = rom[:4] == MAGIC
    declared_word_count_valid = (words[1] & 0xFFF) == WORD_COUNT
    expected_crc32 = compute_crc32(rom[:CRC_OFFSET])
    stored_crc32 = words[-1]
    crc_valid = stored_crc32 == expected_crc32
    valid = magic_valid and declared_word_count_valid and crc_valid
    return VerificationResult(
        valid=valid,
        magic_valid=magic_valid,
        length_valid=True,
        declared_word_count_valid=declared_word_count_valid,
        crc_valid=crc_valid,
        expected_crc32=expected_crc32,
        stored_crc32=stored_crc32,
    )


def decode_word(index: int, rom: bytes = CANONICAL_ROM_BYTES) -> DecodedWord:
    """Decode the declared [31:26]/register/immediate bit layout for one slot."""

    if len(rom) != ROM_BYTES:
        raise ValueError(f"ROM must be exactly {ROM_BYTES} bytes")
    if not 0 <= index < WORD_COUNT:
        raise IndexError(f"ROM word index must be in [0, {WORD_COUNT - 1}]")

    word = struct.unpack_from(">I", rom, index * 4)[0]
    return DecodedWord(
        index=index,
        address=index * 4,
        word=word,
        opcode=(word >> 26) & 0x3F,
        rd=(word >> 21) & 0x1F,
        rs1=(word >> 16) & 0x1F,
        rs2=(word >> 11) & 0x1F,
        immediate=word & 0x7FF,
        symbolic_operation=SLOT_OPERATIONS[index],
    )


def decode_rom(rom: bytes = CANONICAL_ROM_BYTES) -> tuple[DecodedWord, ...]:
    """Decode all sixteen slots, including metadata and checksum slots."""

    return tuple(decode_word(index, rom) for index in range(WORD_COUNT))


def execute_rom_step(
    pc: int,
    registers: list[int],
    rom: bytes = CANONICAL_ROM_BYTES,
) -> int:
    """Trace one ROM slot and advance the word-indexed program counter.

    This function intentionally does not pretend to execute the mathematical
    kernels named by the opcodes. A host runtime must bind those operations.
    """

    if len(registers) < 32:
        raise ValueError("the declared instruction layout requires at least 32 registers")
    decoded = decode_word(pc, rom)
    print(
        f"[PC: 0x{decoded.address:04X}] Word: 0x{decoded.word:08X} "
        f"| Slot: {decoded.symbolic_operation} | Opcode: 0x{decoded.opcode:02X} "
        f"| Rd: R{decoded.rd}"
    )
    return pc + 1


def write_rom(path: str | Path) -> Path:
    """Write the canonical raw binary image and return its path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(CANONICAL_ROM_BYTES)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dr_moagi_3d_aed_v1.rom"),
        help="raw 64-byte ROM output path",
    )
    args = parser.parse_args(argv)

    output = write_rom(args.output)
    result = verify_rom(output.read_bytes())
    print(f"wrote {output} ({ROM_BYTES} bytes)")
    print(f"magic={MAGIC.decode('ascii')} words={WORD_COUNT} crc32=0x{CANONICAL_CRC32:08X}")
    return 0 if result.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
