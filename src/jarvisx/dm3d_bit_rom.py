"""Exact 1 MiB 1024^3-bit ROM profile for the Dr Moagi/Jarvis-X 3D engine.

This module is a bounded ROM *profile adapter*. It does not replace the
canonical Jarvis-X authority VM. The ROM compactly describes a virtual
1024 x 1024 x 1024 bit field and a nested traversal over that field.

The dense logical field would require 128 MiB of working state. The ROM is
exactly 1 MiB and contains a 4 KiB header plus 64-bit instructions padded
with deterministic NOP words.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, TypedDict

ROM_SIZE = 1024 * 1024
HEADER_SIZE = 4096
WORD_SIZE = 8

NX = 1024
NY = 1024
NZ = 1024
TOTAL_BITS = NX * NY * NZ
STATE_BYTES = TOTAL_BITS // 8

MAGIC = b"M3DROM1\x00"
ABI_MAJOR = 1
ABI_MINOR = 0

OP_NOP = 0x00
OP_CFG_DIM = 0x01
OP_CFG_STRIDE = 0x02
OP_CFG_PARAM = 0x03
OP_CLR_REG = 0x04
OP_LOAD_BIT = 0x10
OP_NEIGHBOR6 = 0x11
OP_ENC_3D_BIT = 0x20
OP_DEC_3D_BIT = 0x30
OP_RESIDUAL = 0x40
OP_INWARD_FOLD = 0x50
OP_OMEGA_ACCUM = 0x51
OP_VERIFY = 0x60
OP_CORRECT = 0x61
OP_STORE_BIT = 0x62
OP_INC = 0x70
OP_JLT = 0x71
OP_JMP = 0x72
OP_PASS_END = 0x7E
OP_HALT = 0xFF

OP_NAMES: Dict[int, str] = {
    OP_NOP: "NOP",
    OP_CFG_DIM: "CFG_DIM",
    OP_CFG_STRIDE: "CFG_STRIDE",
    OP_CFG_PARAM: "CFG_PARAM",
    OP_CLR_REG: "CLR_REG",
    OP_LOAD_BIT: "LOAD_BIT",
    OP_NEIGHBOR6: "NEIGHBOR6",
    OP_ENC_3D_BIT: "ENC_3D_BIT",
    OP_DEC_3D_BIT: "DEC_3D_BIT",
    OP_RESIDUAL: "RESIDUAL",
    OP_INWARD_FOLD: "INWARD_FOLD",
    OP_OMEGA_ACCUM: "OMEGA_ACCUM",
    OP_VERIFY: "VERIFY",
    OP_CORRECT: "CORRECT",
    OP_STORE_BIT: "STORE_BIT",
    OP_INC: "INC",
    OP_JLT: "JLT",
    OP_JMP: "JMP",
    OP_PASS_END: "PASS_END",
    OP_HALT: "HALT",
}

EXPECTED_SHA256 = "bb61432c3c952742f2c7fb81b47e1590b87e451efcf85fb87d896e39237f14e0"


class HeaderInfo(TypedDict):
    magic: str
    abi_major: int
    abi_minor: int
    header_size: int
    rom_size: int
    word_size: int
    active_program_words: int
    shape: tuple[int, int, int]
    total_bits: int
    dense_state_bytes: int
    entry_offset: int
    code_crc32: int
    sha256: str


def pack_word(
    opcode: int,
    dst: int = 0,
    src_a: int = 0,
    src_b: int = 0,
    imm12: int = 0,
    arg32: int = 0,
) -> int:
    """Pack one 64-bit profile instruction.

    Layout:
        [63:56 opcode][55:52 dst][51:48 srcA][47:44 srcB]
        [43:32 imm12][31:0 arg32]
    """

    if not 0 <= opcode < 256:
        raise ValueError("opcode must fit in 8 bits")
    if not 0 <= dst < 16 or not 0 <= src_a < 16 or not 0 <= src_b < 16:
        raise ValueError("register indices must fit in 4 bits")
    if not 0 <= imm12 < 4096:
        raise ValueError("imm12 must fit in 12 bits")
    if not 0 <= arg32 < 2**32:
        raise ValueError("arg32 must fit in 32 bits")

    return (
        ((opcode & 0xFF) << 56)
        | ((dst & 0xF) << 52)
        | ((src_a & 0xF) << 48)
        | ((src_b & 0xF) << 44)
        | ((imm12 & 0xFFF) << 32)
        | (arg32 & 0xFFFFFFFF)
    )


def build_program() -> List[int]:
    """Return the compact bounded traversal program."""

    program = [
        pack_word(OP_CFG_DIM, dst=0, arg32=NX),
        pack_word(OP_CFG_DIM, dst=1, arg32=NY),
        pack_word(OP_CFG_DIM, dst=2, arg32=NZ),
        pack_word(OP_CFG_STRIDE, dst=0, arg32=1),
        pack_word(OP_CFG_STRIDE, dst=1, arg32=NX),
        pack_word(OP_CFG_STRIDE, dst=2, arg32=NX * NY),
        pack_word(OP_CFG_PARAM, dst=0, imm12=3072),  # lambda ~= 0.75
        pack_word(OP_CFG_PARAM, dst=1, imm12=3482),  # rho ~= 0.85
        pack_word(OP_CFG_PARAM, dst=2, imm12=1024),  # alpha ~= 0.25
        pack_word(OP_CLR_REG, dst=0),
        pack_word(OP_CLR_REG, dst=1),
        pack_word(OP_CLR_REG, dst=2),
        pack_word(OP_CLR_REG, dst=7),
        pack_word(OP_CLR_REG, dst=11),
    ]

    voxel_loop = len(program)
    program.extend(
        [
            pack_word(OP_LOAD_BIT, dst=3, src_a=0, src_b=1, imm12=2),
            pack_word(OP_NEIGHBOR6, dst=4, src_a=3, src_b=0),
            pack_word(OP_ENC_3D_BIT, dst=5, src_a=3, src_b=4),
            pack_word(OP_INWARD_FOLD, dst=6, src_a=5, src_b=0, imm12=3072),
            pack_word(OP_OMEGA_ACCUM, dst=7, src_a=6, src_b=7, imm12=3482),
            pack_word(OP_DEC_3D_BIT, dst=8, src_a=6, src_b=7),
            pack_word(OP_RESIDUAL, dst=9, src_a=3, src_b=8),
            pack_word(OP_VERIFY, dst=10, src_a=8, src_b=3),
            pack_word(OP_CORRECT, dst=10, src_a=10, src_b=9, imm12=1024),
            pack_word(OP_STORE_BIT, dst=10, src_a=0, src_b=1, imm12=2),
            pack_word(OP_INC, dst=0),
            pack_word(OP_JLT, dst=0, imm12=voxel_loop, arg32=NX),
            pack_word(OP_CLR_REG, dst=0),
            pack_word(OP_INC, dst=1),
            pack_word(OP_JLT, dst=1, imm12=voxel_loop, arg32=NY),
            pack_word(OP_CLR_REG, dst=1),
            pack_word(OP_INC, dst=2),
            pack_word(OP_JLT, dst=2, imm12=voxel_loop, arg32=NZ),
            pack_word(OP_CLR_REG, dst=2),
            pack_word(OP_PASS_END, dst=11),
            pack_word(OP_HALT),
        ]
    )
    return program


def _opcode_header_table() -> bytes:
    table = bytearray()
    for opcode, name in OP_NAMES.items():
        record = f"{opcode:02X} {name}".encode("ascii")[:31]
        table.extend(record.ljust(32, b"\x00"))
    return bytes(table)


def build_rom() -> bytes:
    """Build the deterministic exact-1-MiB ROM image."""

    program = build_program()
    capacity_words = (ROM_SIZE - HEADER_SIZE) // WORD_SIZE
    if len(program) > capacity_words:
        raise ValueError("program exceeds 1 MiB ROM capacity")

    code = b"".join(struct.pack("<Q", word) for word in program)
    nop = struct.pack("<Q", pack_word(OP_NOP))
    code += nop * (capacity_words - len(program))

    header = bytearray(HEADER_SIZE)
    header[0:8] = MAGIC
    struct.pack_into("<I", header, 8, ABI_MAJOR)
    struct.pack_into("<I", header, 12, ABI_MINOR)
    struct.pack_into("<I", header, 16, HEADER_SIZE)
    struct.pack_into("<I", header, 20, ROM_SIZE)
    struct.pack_into("<I", header, 24, WORD_SIZE)
    struct.pack_into("<I", header, 28, len(program))
    struct.pack_into("<III", header, 32, NX, NY, NZ)
    struct.pack_into("<Q", header, 48, TOTAL_BITS)
    struct.pack_into("<Q", header, 56, STATE_BYTES)
    struct.pack_into("<Q", header, 64, HEADER_SIZE)
    struct.pack_into("<Q", header, 72, 0)
    struct.pack_into("<Q", header, 80, 0x1)  # recursive AE/AD profile
    struct.pack_into("<Q", header, 88, 0x2)  # six-neighbour feature
    struct.pack_into("<Q", header, 96, 0x4)  # verify/correct feature

    name = b"Moagi 3D Bit Iteration Autoencode/Decode VM"
    header[128 : 128 + len(name)] = name

    opcode_table = _opcode_header_table()
    header[512 : 512 + len(opcode_table)] = opcode_table

    rom = header + code
    crc = zlib.crc32(rom[HEADER_SIZE:]) & 0xFFFFFFFF
    struct.pack_into("<I", rom, 104, crc)

    if len(rom) != ROM_SIZE:
        raise AssertionError("ROM builder violated exact-size invariant")
    return bytes(rom)


def parse_header(rom: bytes) -> HeaderInfo:
    """Parse and validate fixed header fields."""

    if len(rom) != ROM_SIZE:
        raise ValueError("ROM must be exactly 1 MiB")
    if rom[:8] != MAGIC:
        raise ValueError("invalid ROM magic")

    return {
        "magic": rom[:8].rstrip(b"\x00").decode("ascii"),
        "abi_major": struct.unpack_from("<I", rom, 8)[0],
        "abi_minor": struct.unpack_from("<I", rom, 12)[0],
        "header_size": struct.unpack_from("<I", rom, 16)[0],
        "rom_size": struct.unpack_from("<I", rom, 20)[0],
        "word_size": struct.unpack_from("<I", rom, 24)[0],
        "active_program_words": struct.unpack_from("<I", rom, 28)[0],
        "shape": struct.unpack_from("<III", rom, 32),
        "total_bits": struct.unpack_from("<Q", rom, 48)[0],
        "dense_state_bytes": struct.unpack_from("<Q", rom, 56)[0],
        "entry_offset": struct.unpack_from("<Q", rom, 64)[0],
        "code_crc32": struct.unpack_from("<I", rom, 104)[0],
        "sha256": hashlib.sha256(rom).hexdigest(),
    }


def iter_program_words(rom: bytes) -> Iterable[int]:
    header = parse_header(rom)
    count = header["active_program_words"]
    offset = header["entry_offset"]
    for index in range(count):
        yield struct.unpack_from("<Q", rom, offset + index * WORD_SIZE)[0]


def validate_rom(rom: bytes, require_reference_hash: bool = False) -> HeaderInfo:
    """Validate size, geometry, code CRC, and optionally the reference hash."""

    header = parse_header(rom)
    expected_crc = zlib.crc32(rom[HEADER_SIZE:]) & 0xFFFFFFFF
    if header["code_crc32"] != expected_crc:
        raise ValueError("ROM code CRC mismatch")
    if header["shape"] != (NX, NY, NZ):
        raise ValueError("ROM geometry mismatch")
    if header["total_bits"] != TOTAL_BITS:
        raise ValueError("ROM logical bit count mismatch")
    if header["dense_state_bytes"] != STATE_BYTES:
        raise ValueError("ROM dense-state byte count mismatch")
    if require_reference_hash and header["sha256"] != EXPECTED_SHA256:
        raise ValueError("ROM reference SHA-256 mismatch")
    return header


def write_rom(path: Path) -> HeaderInfo:
    rom = build_rom()
    path.write_bytes(rom)
    return validate_rom(rom, require_reference_hash=True)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect the Jarvis-X 1024^3-bit ROM profile")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build", help="emit the exact 1 MiB reference ROM")
    build.add_argument("path", nargs="?", default="moagi_3d_1024cube_1MiB.rom")

    inspect_cmd = sub.add_parser("inspect", help="inspect and validate a ROM")
    inspect_cmd.add_argument("path")

    args = parser.parse_args(argv)

    if args.cmd == "build":
        meta = write_rom(Path(args.path))
        print(json.dumps(meta, indent=2))
        return 0

    if args.cmd == "inspect":
        rom = Path(args.path).read_bytes()
        print(json.dumps(validate_rom(rom), indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
