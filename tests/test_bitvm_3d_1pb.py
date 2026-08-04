"""Invariant tests for the sparse transactional 3D 1 PB BitVM reference."""

import copy

import pytest

from jarvisx.bitvm_3d_1pb import (
    CANONICAL_VIRTUAL_BITS,
    CANONICAL_VIRTUAL_BYTES,
    AddressClass,
    BitAddress,
    BitInstruction,
    BitOpcode,
    BitVMConfig,
    Capability,
    Sparse3DBitVM,
    TransactionRejected,
)


def address(
    *,
    access: AddressClass,
    x: int = 0,
    y: int = 0,
    z: int = 0,
    byte: int = 0,
    bit: int = 0,
    asid: int = 0,
) -> BitAddress:
    return BitAddress(asid, int(access), x, y, z, byte, bit)


def test_default_geometry_is_one_petabyte_without_allocation() -> None:
    vm = Sparse3DBitVM()

    assert vm.virtual_brick_count == 1_000_000_000
    assert vm.virtual_byte_count == CANONICAL_VIRTUAL_BYTES == 1_000_000_000_000_000
    assert vm.virtual_bit_count == CANONICAL_VIRTUAL_BITS == 8_000_000_000_000_000
    assert vm.config.lines_per_brick == 15_625
    assert vm.resident_brick_count == 0
    assert vm.resident_payload_bytes == 0


def test_address_pack_unpack_round_trip() -> None:
    values = (
        address(access=AddressClass.READ),
        address(
            access=AddressClass.CONTROL,
            asid=255,
            x=999,
            y=998,
            z=997,
            byte=999_999,
            bit=7,
        ),
    )

    for item in values:
        assert BitAddress.unpack(item.pack()) == item
        assert 0 <= item.pack() < 1 << 64

    with pytest.raises(ValueError, match="byte_offset"):
        BitAddress(0, 0, 0, 0, 0, 1_000_000, 0)


def test_absent_read_is_zero_and_does_not_materialize_a_brick() -> None:
    vm = Sparse3DBitVM()

    assert vm.read_bit(address(access=AddressClass.READ, x=500, y=500, z=500)) == 0
    assert vm.resident_brick_count == 0


def test_set_clear_and_zero_brick_pruning() -> None:
    vm = Sparse3DBitVM(BitVMConfig(brick_bytes=64))
    target = address(access=AddressClass.WRITE, byte=3, bit=6)

    set_receipt = vm.execute(BitInstruction(BitOpcode.BSET, destination=target, length_bits=3))

    assert set_receipt.committed is True
    assert vm.resident_brick_count == 1
    assert vm.read_bit(address(access=AddressClass.READ, byte=3, bit=6)) == 1
    assert vm.read_bit(address(access=AddressClass.READ, byte=3, bit=7)) == 1
    assert vm.read_bit(address(access=AddressClass.READ, byte=4, bit=0)) == 1

    vm.execute(BitInstruction(BitOpcode.BCLR, destination=target, length_bits=3))

    assert vm.resident_brick_count == 0
    assert vm.read_bit(address(access=AddressClass.READ, byte=4, bit=0)) == 0


def test_copy_and_xor_across_a_brick_boundary() -> None:
    vm = Sparse3DBitVM(
        BitVMConfig(
            side=3,
            brick_bytes=8,
            vector_line_bytes=8,
            max_resident_bricks=8,
        )
    )
    source = address(access=AddressClass.WRITE, byte=7, bit=6)
    vm.execute(BitInstruction(BitOpcode.BSET, destination=source, length_bits=4))

    destination = address(access=AddressClass.WRITE, x=2, byte=7, bit=6)
    readable_source = address(access=AddressClass.READ, byte=7, bit=6)
    vm.execute(
        BitInstruction(
            BitOpcode.BCOPY,
            destination=destination,
            source0=readable_source,
            length_bits=4,
        )
    )

    assert vm.read_bit(address(access=AddressClass.READ, x=2, byte=7, bit=6)) == 1
    assert vm.read_bit(address(access=AddressClass.READ, y=1, byte=0, bit=0)) == 1

    xor_destination = address(access=AddressClass.WRITE, x=1, byte=0, bit=0)
    vm.execute(
        BitInstruction(
            BitOpcode.BXOR,
            destination=xor_destination,
            source0=address(access=AddressClass.READ, x=0, byte=7, bit=6),
            source1=address(access=AddressClass.READ, x=2, byte=7, bit=6),
            length_bits=1,
        )
    )
    assert vm.read_bit(address(access=AddressClass.READ, x=1, byte=0, bit=0)) == 0


def test_popcount_and_hash_are_deterministic_and_non_mutating() -> None:
    vm = Sparse3DBitVM(BitVMConfig(brick_bytes=64))
    target = address(access=AddressClass.WRITE, byte=0, bit=0)
    vm.execute(BitInstruction(BitOpcode.BSET, destination=target, length_bits=5))
    before = vm.state_digest()

    pop = vm.execute(
        BitInstruction(
            BitOpcode.BPOPCNT,
            source0=address(access=AddressClass.READ),
            length_bits=8,
        )
    )
    hashed = vm.execute(
        BitInstruction(
            BitOpcode.BHASH,
            source0=address(access=AddressClass.READ),
            length_bits=8,
        )
    )

    assert pop.result == 5
    assert isinstance(hashed.result, str) and len(hashed.result) == 64
    assert vm.state_digest() == before


def test_rejected_instruction_rolls_back_state_but_advances_journal() -> None:
    capability = Capability(x_range=(0, 0), y_range=(0, 0), z_range=(0, 0))
    vm = Sparse3DBitVM(BitVMConfig(side=2, brick_bytes=64), capability)
    before_state = vm.state_digest()
    before_journal = vm.journal_digest

    with pytest.raises(TransactionRejected, match="outside the capability") as rejected:
        vm.execute(
            BitInstruction(
                BitOpcode.BSET,
                destination=address(access=AddressClass.WRITE, x=1),
            )
        )

    assert rejected.value.receipt.committed is False
    assert vm.state_digest() == before_state
    assert vm.journal_digest != before_journal
    assert vm.resident_brick_count == 0
    assert vm.journal[-1].after_state_digest == before_state


def test_resident_budget_failure_is_atomic() -> None:
    vm = Sparse3DBitVM(
        BitVMConfig(
            side=2,
            brick_bytes=8,
            vector_line_bytes=8,
            max_resident_bricks=1,
        ),
        Capability(x_range=(0, 1), y_range=(0, 1), z_range=(0, 1), max_accessed_bricks=2),
    )
    before = vm.state_digest()

    with pytest.raises(TransactionRejected, match="resident-brick budget exceeded"):
        vm.execute(
            BitInstruction(
                BitOpcode.BSET,
                destination=address(access=AddressClass.WRITE, byte=7, bit=7),
                length_bits=2,
            )
        )

    assert vm.state_digest() == before
    assert vm.resident_brick_count == 0


def test_checkpoint_round_trip_and_tamper_detection() -> None:
    vm = Sparse3DBitVM(BitVMConfig(side=4, brick_bytes=64, max_resident_bricks=4))
    vm.execute(
        BitInstruction(
            BitOpcode.BSET,
            destination=address(access=AddressClass.WRITE, x=2, byte=1, bit=3),
            length_bits=7,
        )
    )
    checkpoint = vm.checkpoint()

    restored = Sparse3DBitVM.from_checkpoint(checkpoint)

    assert restored.config == vm.config
    assert restored.state_digest() == vm.state_digest()
    assert restored.journal_digest == vm.journal_digest
    assert restored.read_bit(address(access=AddressClass.READ, x=2, byte=1, bit=3)) == 1

    tampered = copy.deepcopy(checkpoint)
    payload = tampered["bricks"][0]["payload_b64"]
    tampered["bricks"][0]["payload_b64"] = ("A" if payload[0] != "A" else "B") + payload[1:]
    with pytest.raises(ValueError, match="digest mismatch|base64"):
        Sparse3DBitVM.from_checkpoint(tampered)


def test_same_instruction_stream_produces_identical_state_and_journal() -> None:
    config = BitVMConfig(side=4, brick_bytes=64, max_resident_bricks=4)
    first = Sparse3DBitVM(config)
    second = Sparse3DBitVM(config)
    program = (
        BitInstruction(
            BitOpcode.BSET,
            destination=address(access=AddressClass.WRITE, x=1, byte=2, bit=1),
            length_bits=9,
        ),
        BitInstruction(
            BitOpcode.BNOT,
            destination=address(access=AddressClass.WRITE, x=2, byte=0, bit=0),
            source0=address(access=AddressClass.READ, x=1, byte=2, bit=1),
            length_bits=9,
        ),
    )

    for instruction in program:
        first.execute(instruction)
        second.execute(instruction)

    assert first.state_digest() == second.state_digest()
    assert first.journal_digest == second.journal_digest
    assert first.checkpoint() == second.checkpoint()
