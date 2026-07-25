import pytest

from jarvisx.dr_moagi_3d import (
    AXIS,
    CUBE_BYTES,
    REQUIRED_VERIFICATION,
    TOTAL_BYTES,
    CellClass,
    Coordinate3D,
    DrMoagiEngine,
    GeometricCodec,
    Instruction,
    Opcode,
    RomCell,
    VerificationBit,
    commit_gate,
    dequantize_i8_to_q14,
    omega_update,
    q14_from_byte,
    q14_to_byte,
    quantize_q14_to_i8,
)


def test_capacity_and_cube_geometry() -> None:
    assert AXIS == 6400
    assert CUBE_BYTES == 4 * 1024 * 1024
    assert TOTAL_BYTES == 4_194_304_000_000


def test_coordinate_round_trip_and_bounds() -> None:
    coordinate = Coordinate3D(6399, 6399, 6399)
    assert Coordinate3D.from_cell_address(coordinate.cell_address) == coordinate
    assert coordinate.cell_address == AXIS**3 - 1
    assert coordinate.byte_address == TOTAL_BYTES - 16
    assert coordinate.cube_coordinate == (99, 99, 99)
    assert coordinate.local_coordinate == (63, 63, 63)

    with pytest.raises(ValueError):
        Coordinate3D(6400, 0, 0)


def test_byte_q14_quantization_round_trip() -> None:
    for value in range(256):
        q14 = q14_from_byte(value)
        quantized = quantize_q14_to_i8(q14)
        restored = dequantize_i8_to_q14(quantized)
        assert q14_to_byte(restored) == value


def test_rom_cell_pack_round_trip() -> None:
    block = GeometricCodec.identity().encode(b"ABCDEF", version=7, parent=11)
    restored = RomCell.unpack(block.cell.pack())
    assert restored == block.cell
    assert restored.cell_class == CellClass.LATENT


def test_instruction_pack_round_trip() -> None:
    coordinate = Coordinate3D(123, 456, 789)
    instruction = Instruction(
        Opcode.SELF_ENCODE,
        mode=2,
        dst=31,
        src=17,
        coordinate=coordinate.cell_address,
        immediate=0x12345,
    )
    assert Instruction.unpack(instruction.pack()) == instruction


def test_identity_codec_is_bit_exact() -> None:
    codec = GeometricCodec.identity()
    encoded = codec.encode(b"ABCDEF")
    assert codec.decode(encoded.cell.pack()) == b"ABCDEF"


def test_commit_requires_all_verification_bits_and_improvement() -> None:
    active = 0x1111
    candidate = 0x2222

    selected, committed = commit_gate(
        active,
        candidate,
        REQUIRED_VERIFICATION,
        candidate_loss=9,
        active_loss=10,
    )
    assert committed is True
    assert selected == candidate

    selected, committed = commit_gate(
        active,
        candidate,
        VerificationBit.PARSED,
        candidate_loss=0,
        active_loss=10,
    )
    assert committed is False
    assert selected == active

    selected, committed = commit_gate(
        active,
        candidate,
        REQUIRED_VERIFICATION,
        candidate_loss=10,
        active_loss=10,
    )
    assert committed is False
    assert selected == active


def test_engine_commit_and_rollback() -> None:
    engine = DrMoagiEngine()

    decoded, committed, coordinate = engine.cycle(
        b"ABCDEF",
        REQUIRED_VERIFICATION,
        candidate_loss=1,
        active_loss=2,
    )
    assert decoded == b"ABCDEF"
    assert committed is True
    assert engine.rom.read(coordinate) != 0
    assert engine.version == 1

    decoded, committed, rejected_coordinate = engine.cycle(
        b"UVWXYZ",
        VerificationBit.PARSED,
        candidate_loss=0,
        active_loss=2,
    )
    assert decoded == b"UVWXYZ"
    assert committed is False
    assert engine.rom.read(rejected_coordinate) == 0
    assert engine.version == 1


def test_omega_shift_exact_update() -> None:
    assert omega_update(0, 1600) == 100
    assert omega_update(800, 0) == 700
