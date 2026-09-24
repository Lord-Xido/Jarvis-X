from __future__ import annotations

import pytest

from jarvisx.recursive_bits3d import (
    BitAddress3D,
    Bits3DConfig,
    CellAddress3D,
    RecursiveBits3DVM,
    SymbolicPowerTower,
    address_to_bit_index,
    address_to_scalar_index,
    bit_index_to_address,
    scalar_index_to_address,
)
from jarvisx.volumetric_bytecode import VolumetricBytecodeError


def test_default_64_cubed_four_channel_state_is_one_mib() -> None:
    config = Bits3DConfig()

    assert config.scalar_cells == 64**3 * 4 == 1_048_576
    assert config.raw_bytes == 1_048_576
    assert config.raw_bits == 8_388_608


def test_scalar_address_mapping_round_trips() -> None:
    config = Bits3DConfig(side=8, channels=4)
    indices = (0, 1, 17, config.scalar_cells - 1)

    for index in indices:
        address = scalar_index_to_address(index, config)
        assert address_to_scalar_index(address, config) == index

    assert scalar_index_to_address(config.scalar_cells - 1, config) == CellAddress3D(
        x=7, y=7, z=7, channel=3
    )


def test_bit_address_mapping_round_trips() -> None:
    config = Bits3DConfig(side=4, channels=3)

    for index in (0, 7, 8, 37, config.raw_bits - 1):
        address = bit_index_to_address(index, config)
        assert address_to_bit_index(address, config) == index

    last = bit_index_to_address(config.raw_bits - 1, config)
    assert last == BitAddress3D(x=3, y=3, z=3, channel=2, bit=7)


def test_septillion_power_tower_stays_symbolic() -> None:
    target = SymbolicPowerTower()

    assert target.definition == "a=1000000000000000000000000"
    assert target.description == "a^(a^(a))"
    assert target.decimal_power_description == "10^(24*10^(24*10^(24)))"


def test_bitstream_adapter_runs_bounded_recursive_cycle() -> None:
    vm = RecursiveBits3DVM(
        Bits3DConfig(side=8, channels=4, max_iterations=4, epsilon_active_change=0.0)
    )
    payload = bytes([0x03, 0x00, 0x55, 0x80])
    vm.load_bytes(payload)

    result = vm.run()

    assert result.converged is True
    assert result.physical_iterations == 2
    assert result.symbolic_iteration_target == "a^(a^(a))"
    assert result.symbolic_decimal_target == "10^(24*10^(24*10^(24)))"
    assert result.changed_bits_total > 0
    assert result.receipts[-1].changed_bits == 0
    assert result.receipts[-1].codec_roundtrip_error_bits == 0
    assert vm.read_prefix() == payload


def test_symbolic_horizon_never_overrides_physical_budget() -> None:
    vm = RecursiveBits3DVM(
        Bits3DConfig(side=8, channels=4, max_iterations=1, epsilon_active_change=0.0)
    )
    vm.load_bytes(bytes([1]))

    result = vm.run()

    assert result.physical_iterations == 1
    assert result.budget_exhausted is True
    assert result.converged is False
    assert result.symbolic_iteration_target == "a^(a^(a))"


def test_zero_bytes_are_sparse_and_not_materialized() -> None:
    vm = RecursiveBits3DVM(Bits3DConfig(side=4, channels=4, max_active_cells=4))
    vm.load_bytes(bytes([0, 0, 7, 0]))

    assert vm.active_cells == 1
    assert vm.sparse_bytes() == ((2, 7),)
    assert vm.read_prefix() == bytes([0, 0, 7, 0])


def test_oversized_payload_fails_closed() -> None:
    config = Bits3DConfig(side=2, channels=1)
    vm = RecursiveBits3DVM(config)

    with pytest.raises(VolumetricBytecodeError, match="payload exceeds"):
        vm.load_bytes(bytes(config.scalar_cells + 1))


def test_out_of_range_bit_address_fails_closed() -> None:
    config = Bits3DConfig(side=4, channels=4)

    with pytest.raises(IndexError, match="bit index"):
        bit_index_to_address(config.raw_bits, config)

    with pytest.raises(IndexError, match="bit lane"):
        address_to_bit_index(BitAddress3D(0, 0, 0, 0, 8), config)
