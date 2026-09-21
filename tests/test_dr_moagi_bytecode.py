from __future__ import annotations

import struct

import pytest

from jarvisx.dr_moagi_bytecode import (
    CANONICAL_CRC32,
    CANONICAL_ROM_BYTES,
    CANONICAL_WORDS,
    ROM_BYTES,
    SUPPLIED_CRC32,
    SUPPLIED_ROM_BYTES,
    compute_crc32,
    decode_word,
    execute_rom_step,
    verify_rom,
    write_rom,
)


def test_canonical_rom_is_exactly_64_bytes_and_crc_valid() -> None:
    assert len(CANONICAL_ROM_BYTES) == ROM_BYTES == 64
    assert CANONICAL_ROM_BYTES[:4] == b"MOAG"
    assert CANONICAL_CRC32 == 0x96FDFC2F
    assert compute_crc32(CANONICAL_ROM_BYTES[:60]) == CANONICAL_CRC32
    assert verify_rom(CANONICAL_ROM_BYTES).valid


def test_supplied_checksum_is_preserved_but_rejected() -> None:
    result = verify_rom(SUPPLIED_ROM_BYTES)

    assert result.valid is False
    assert result.crc_valid is False
    assert result.stored_crc32 == SUPPLIED_CRC32 == 0x7E3F91A2
    assert result.expected_crc32 == CANONICAL_CRC32


def test_declared_instruction_fields_decode_consistently() -> None:
    encoder = decode_word(5)
    assert encoder.address == 0x0014
    assert encoder.opcode == 0x02
    assert (encoder.rd, encoder.rs1, encoder.rs2, encoder.immediate) == (3, 1, 1, 0)

    curl = decode_word(10)
    assert curl.address == 0x0028
    assert curl.opcode == 0x05
    assert (curl.rd, curl.rs1, curl.rs2, curl.immediate) == (7, 7, 6, 0)


def test_nop_and_halt_require_slot_level_semantics() -> None:
    nop = decode_word(13)
    halt = decode_word(14)

    assert nop.word == halt.word == 0
    assert nop.symbolic_operation == "NOP"
    assert halt.symbolic_operation == "HALT"


def test_write_rom_emits_exact_big_endian_binary(tmp_path) -> None:
    output = write_rom(tmp_path / "image.rom")

    assert output.read_bytes() == CANONICAL_ROM_BYTES
    assert struct.unpack(">16I", output.read_bytes()) == CANONICAL_WORDS


def test_trace_step_validates_register_file_and_advances(capsys) -> None:
    assert execute_rom_step(5, [0] * 32) == 6
    assert "ENC_PROJ_1" in capsys.readouterr().out

    with pytest.raises(ValueError, match="at least 32"):
        execute_rom_step(5, [0] * 8)
