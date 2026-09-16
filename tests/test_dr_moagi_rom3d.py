"""Tests for the sparse 3D 1 MiB-cubed ROM reference runtime."""

import struct

import pytest

from jarvisx.dr_moagi_rom3d import (
    ADDRESS_BITS,
    AXIS,
    BLOCK_VOLUME,
    BOOT_PROGRAM,
    LOGICAL_BYTES,
    MAGIC,
    PAGE_SIZE,
    VERSION,
    InwardAutoEncoder,
    Instruction,
    Op,
    ROMVM,
    VirtualROM,
    descriptor_bytes,
    morton3_decode,
    morton3_encode,
    self_test,
)


def test_logical_geometry_is_exactly_one_eib() -> None:
    assert AXIS == 1 << 20
    assert LOGICAL_BYTES == 1 << 60
    assert ADDRESS_BITS == 60
    assert BLOCK_VOLUME == 1000


@pytest.mark.parametrize(
    "xyz",
    [
        (0, 0, 0),
        (1, 2, 3),
        (12345, 54321, 999999),
        (AXIS - 1, AXIS - 1, AXIS - 1),
    ],
)
def test_morton_addressing_is_bijective(xyz: tuple[int, int, int]) -> None:
    address = morton3_encode(*xyz)
    assert 0 <= address < LOGICAL_BYTES
    assert morton3_decode(address) == xyz


def test_morton_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        morton3_encode(AXIS, 0, 0)
    with pytest.raises(ValueError):
        morton3_decode(1 << ADDRESS_BITS)


def test_sparse_rom_is_deterministic() -> None:
    left = VirtualROM(seed=b"same-seed")
    right = VirtualROM(seed=b"same-seed")
    other = VirtualROM(seed=b"other-seed")

    address = morton3_encode(1024, 2048, 4096)
    assert left.read_linear(address) == right.read_linear(address)
    assert left._synthesize_page(7) == right._synthesize_page(7)
    assert left._synthesize_page(7) != other._synthesize_page(7)


def test_block_is_10_cubed_bytes() -> None:
    rom = VirtualROM()
    block = rom.read_block(AXIS - 3, AXIS - 3, AXIS - 3)
    assert len(block) == BLOCK_VOLUME


def test_instruction_round_trip() -> None:
    instruction = Instruction(Op.JNZ, a=7, b=65535, c=0xDEADBEEF)
    packed = instruction.pack()
    assert len(packed) == 8
    assert Instruction.unpack(packed) == instruction


def test_descriptor_header_and_boot_program() -> None:
    blob = descriptor_bytes()
    header_size = struct.calcsize(">8sIQQIIII32s")
    magic, version, axis, logical_bytes, bits, page_size, edge, boot_size, _ = struct.unpack(
        ">8sIQQIIII32s", blob[:header_size]
    )
    assert magic == MAGIC
    assert version == VERSION
    assert axis == AXIS
    assert logical_bytes == LOGICAL_BYTES
    assert bits == ADDRESS_BITS
    assert page_size == PAGE_SIZE
    assert edge == 10
    assert boot_size == len(BOOT_PROGRAM) * 8
    assert len(blob) == header_size + boot_size


def test_autoencoder_is_bounded_and_deterministic() -> None:
    autoencoder = InwardAutoEncoder(latent_dim=8)
    block = bytes(range(64))
    first = autoencoder.encode(block)
    second = autoencoder.encode(block)
    assert first.z == second.z

    refined = autoencoder.refine_fixed_point(first, max_iter=4, eps=1e-12)
    decoded = autoencoder.decode(refined, len(block))
    learned = autoencoder.learn_from_residual(block, decoded, refined)

    assert len(refined.z) == 8
    assert len(decoded) == len(block)
    assert 0.0 <= learned.error <= 1.0
    assert learned.iterations == 4


def test_boot_vm_executes_to_halt() -> None:
    rom = VirtualROM()
    rom.install_boot_program()
    vm = ROMVM(rom, InwardAutoEncoder(latent_dim=8))
    steps = vm.run(max_steps=len(BOOT_PROGRAM) + 2)
    assert steps == len(BOOT_PROGRAM)
    assert vm.halted
    assert len(vm.last_block) == BLOCK_VOLUME
    assert len(vm.prediction) == BLOCK_VOLUME


def test_public_self_test_reports_bounded_execution() -> None:
    report = self_test()
    assert report["logical_bytes"] == LOGICAL_BYTES
    assert report["logical_eib"] == 1.0
    assert report["block_volume"] == BLOCK_VOLUME
    assert report["vm_halted"] is True
