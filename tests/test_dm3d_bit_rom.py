from __future__ import annotations

import hashlib
import struct
import zlib

import pytest

from jarvisx.dm3d_bit_rom import (
    EXPECTED_SHA256,
    HEADER_SIZE,
    NX,
    NY,
    NZ,
    ROM_SIZE,
    STATE_BYTES,
    TOTAL_BITS,
    build_program,
    build_rom,
    iter_program_words,
    pack_word,
    parse_header,
    validate_rom,
)


def test_1024cube_geometry_and_exact_rom_size() -> None:
    rom = build_rom()
    header = validate_rom(rom, require_reference_hash=True)

    assert len(rom) == ROM_SIZE == 1_048_576
    assert header["shape"] == (1024, 1024, 1024)
    assert (NX, NY, NZ) == (1024, 1024, 1024)
    assert TOTAL_BITS == 1_073_741_824
    assert STATE_BYTES == 134_217_728


def test_reference_rom_is_deterministic() -> None:
    first = build_rom()
    second = build_rom()

    assert first == second
    assert hashlib.sha256(first).hexdigest() == EXPECTED_SHA256


def test_code_crc_covers_the_padded_code_region() -> None:
    rom = build_rom()
    header = parse_header(rom)

    expected = zlib.crc32(rom[HEADER_SIZE:]) & 0xFFFFFFFF
    assert header["code_crc32"] == expected


def test_compact_program_uses_nested_bounded_loops() -> None:
    program = build_program()
    rom = build_rom()
    words = list(iter_program_words(rom))

    assert words == program
    assert len(program) == 35

    # JLT opcode is 0x71 in bits 63:56. Three JLT words bound x, y and z.
    jlt_words = [word for word in words if (word >> 56) == 0x71]
    assert len(jlt_words) == 3
    assert [word & 0xFFFFFFFF for word in jlt_words] == [1024, 1024, 1024]


def test_invalid_instruction_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="opcode"):
        pack_word(256)
    with pytest.raises(ValueError, match="register"):
        pack_word(0, dst=16)
    with pytest.raises(ValueError, match="imm12"):
        pack_word(0, imm12=4096)
    with pytest.raises(ValueError, match="arg32"):
        pack_word(0, arg32=2**32)


def test_corrupt_code_crc_is_rejected() -> None:
    rom = bytearray(build_rom())
    rom[-1] ^= 0x01

    with pytest.raises(ValueError, match="CRC"):
        validate_rom(bytes(rom))


def test_header_declares_dense_state_without_embedding_it() -> None:
    rom = build_rom()
    header = parse_header(rom)

    assert header["dense_state_bytes"] == 128 * 1024 * 1024
    assert header["dense_state_bytes"] > len(rom)
    assert struct.unpack_from("<Q", rom, 56)[0] == STATE_BYTES
