import hashlib
from pathlib import Path

import pytest

from jarvisx.assembler import OPCODES
from jarvisx.chrysalis_rom import (
    ChrysalisROM,
    ROMCapacityError,
    ROMFormatError,
    ROMIntegrityError,
    assemble_source,
    mutate_immediate,
    pack_bytecode,
    rom_from_source,
    unpack_bytecode,
    words_from_rom,
)


def test_payload_round_trip_and_addressing(tmp_path: Path):
    payload = bytes(range(256)) * 3
    rom = ChrysalisROM.from_payload(
        payload,
        engines=2,
        grid_shape=(2, 3, 4),
        cell_bytes=16,
        compress=False,
    )

    assert rom.payload() == payload
    assert rom.verify()
    assert len(rom.cell(0, 0, 0, 0)) == 16
    assert rom.address_for_stored_offset(0) == (0, 0, 0, 0, 0)
    assert rom.address_for_stored_offset(16) == (0, 0, 0, 1, 0)

    path = tmp_path / "image.jrom"
    rom.write(path)
    restored = ChrysalisROM.read(path)
    assert restored.payload() == payload
    assert restored.statistics()["sha256"] == hashlib.sha256(payload).hexdigest()


def test_bytecode_round_trip_is_big_endian_and_exact():
    words = [0, 1, 0x0102030405060708, (1 << 64) - 1]
    image = pack_bytecode(words)
    assert image[:4] == b"JXBC"
    assert image[-8:] == b"\xff" * 8
    assert unpack_bytecode(image) == words


def test_source_to_rom_round_trip():
    source = "SET A 7\nSET B 5\nADD C A B\nHALT"
    expected = assemble_source(source)
    rom = rom_from_source(source, engines=1, grid_shape=(2, 2, 2))
    assert words_from_rom(rom) == expected


def test_bounded_mutation_changes_only_set_immediate():
    rom = rom_from_source("SET A 7\nHALT", grid_shape=(1, 1, 2), cell_bytes=16)
    original = words_from_rom(rom)
    candidate = mutate_immediate(rom, word_index=0, delta=5)
    mutated = words_from_rom(candidate)

    assert ((mutated[0] >> 56) & 0xFF) == OPCODES["SET"]
    assert ((mutated[0] >> 40) & 0xFF) == ((original[0] >> 40) & 0xFF)
    assert ((mutated[0] >> 8) & 0xFFFF) == 12
    assert mutated[1] == original[1]
    assert candidate.verify()


def test_mutation_rejects_non_set_instruction():
    rom = rom_from_source("HALT", grid_shape=(1, 1, 1), cell_bytes=32)
    with pytest.raises(ValueError, match="SET"):
        mutate_immediate(rom, word_index=0, delta=1)


def test_capacity_is_enforced():
    with pytest.raises(ROMCapacityError):
        ChrysalisROM.from_payload(
            b"too large", engines=1, grid_shape=(1, 1, 1), cell_bytes=2
        )


def test_payload_tampering_is_detected():
    rom = ChrysalisROM.from_payload(b"integrity", grid_shape=(1, 1, 1), cell_bytes=16)
    serialized = bytearray(rom.serialize())
    serialized[-16] ^= 0x01
    tampered = ChrysalisROM.deserialize(bytes(serialized))
    with pytest.raises(ROMIntegrityError, match="SHA-256"):
        tampered.payload()


def test_malformed_bytecode_length_is_rejected():
    image = pack_bytecode([1, 2])
    with pytest.raises(ROMFormatError, match="expected"):
        unpack_bytecode(image[:-1])
