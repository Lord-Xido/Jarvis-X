"""Adversarial hardening tests for the 3D 1 PB BitVM."""

import base64
import copy
import hashlib

import pytest

from jarvisx.bitvm_3d_1pb import (
    AddressClass,
    BitAddress,
    BitInstruction,
    BitOpcode,
    BitVMConfig,
    Capability,
    Sparse3DBitVM,
    TransactionRejected,
)


def address(*, access: AddressClass, x=0, y=0, z=0, byte=0, bit=0, asid=0):
    return BitAddress(asid, int(access), x, y, z, byte, bit)


def tiny_config(**overrides):
    values = dict(side=4, brick_bytes=64, vector_line_bytes=64, max_resident_bricks=8)
    values.update(overrides)
    return BitVMConfig(**values)


def rehash_journal(checkpoint):
    vm = Sparse3DBitVM(BitVMConfig(**checkpoint["config"]))
    previous = vm._initial_journal_digest()
    for raw in checkpoint["journal"]:
        raw["previous_hash"] = previous
        payload = dict(raw)
        payload.pop("hash", None)
        raw["hash"] = hashlib.sha256(vm._canonical_json(payload)).hexdigest()
        previous = raw["hash"]
    checkpoint["journal_digest"] = previous


def test_read_bit_rejects_write_class_even_when_capability_allows_it():
    capability = Capability(
        allowed_classes=(int(AddressClass.WRITE),),
        allowed_opcodes=(BitOpcode.BPOPCNT,),
    )
    vm = Sparse3DBitVM(tiny_config(), capability)

    with pytest.raises(PermissionError, match="readable"):
        vm.read_bit(address(access=AddressClass.WRITE))


def test_malformed_address_is_rejected_and_journaled_safely():
    vm = Sparse3DBitVM(tiny_config())
    malformed = BitInstruction(
        BitOpcode.BSET,
        destination="not-an-address",  # type: ignore[arg-type]
    )

    with pytest.raises(TransactionRejected, match="BitAddress"):
        vm.execute(malformed)

    assert len(vm.journal) == 1
    assert vm.journal[0].destination == "<invalid:builtins.str>"
    assert vm.journal[0].committed is False


def test_extraneous_operands_are_rejected():
    vm = Sparse3DBitVM(tiny_config())
    instruction = BitInstruction(
        BitOpcode.BSET,
        destination=address(access=AddressClass.WRITE),
        source0=address(access=AddressClass.READ),
    )

    with pytest.raises(TransactionRejected, match="extraneous"):
        vm.execute(instruction)


def test_checkpoint_rejects_duplicate_brick_keys():
    vm = Sparse3DBitVM(tiny_config())
    vm.execute(
        BitInstruction(
            BitOpcode.BSET,
            destination=address(access=AddressClass.WRITE),
        )
    )
    checkpoint = vm.checkpoint()
    checkpoint["bricks"].append(copy.deepcopy(checkpoint["bricks"][0]))

    with pytest.raises(ValueError, match="duplicate brick"):
        Sparse3DBitVM.from_checkpoint(checkpoint)


def test_checkpoint_rejects_non_monotonic_sequence_even_when_rehashed():
    vm = Sparse3DBitVM(tiny_config())
    vm.execute(
        BitInstruction(
            BitOpcode.BSET,
            destination=address(access=AddressClass.WRITE),
        )
    )
    checkpoint = vm.checkpoint()
    checkpoint["journal"][0]["sequence"] = 999
    rehash_journal(checkpoint)
    checkpoint["sequence"] = 1

    with pytest.raises(ValueError, match="sequence"):
        Sparse3DBitVM.from_checkpoint(checkpoint)


def test_checkpoint_rejects_broken_state_continuity_even_when_rehashed():
    vm = Sparse3DBitVM(tiny_config())
    target = address(access=AddressClass.WRITE)
    vm.execute(BitInstruction(BitOpcode.BSET, destination=target))
    vm.execute(BitInstruction(BitOpcode.BCLR, destination=target))
    checkpoint = vm.checkpoint()
    checkpoint["journal"][1]["before_state_digest"] = "0" * 64
    rehash_journal(checkpoint)

    with pytest.raises(ValueError, match="continuity"):
        Sparse3DBitVM.from_checkpoint(checkpoint)


def test_checkpoint_rejects_state_not_produced_by_journal_tip():
    vm = Sparse3DBitVM(tiny_config())
    vm.execute(
        BitInstruction(
            BitOpcode.BSET,
            destination=address(access=AddressClass.WRITE),
        )
    )
    checkpoint = vm.checkpoint()
    raw = bytearray(64)
    raw[0] = 2
    checkpoint["bricks"][0]["payload_b64"] = base64.b64encode(raw).decode("ascii")

    probe = Sparse3DBitVM(tiny_config())
    probe._bricks[(0, 0, 0, 0)] = bytes(raw)
    checkpoint["state_digest"] = probe.state_digest()

    with pytest.raises(ValueError, match="journal tip"):
        Sparse3DBitVM.from_checkpoint(checkpoint)


def test_checkpoint_rejects_malformed_touched_bricks():
    vm = Sparse3DBitVM(tiny_config())
    vm.execute(
        BitInstruction(
            BitOpcode.BSET,
            destination=address(access=AddressClass.WRITE),
        )
    )
    checkpoint = vm.checkpoint()
    checkpoint["journal"][0]["touched_bricks"] = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    rehash_journal(checkpoint)

    with pytest.raises(ValueError, match="sorted and unique"):
        Sparse3DBitVM.from_checkpoint(checkpoint)
