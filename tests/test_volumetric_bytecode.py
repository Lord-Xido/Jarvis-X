from __future__ import annotations

import math

import pytest

from jarvisx.volumetric_bytecode import (
    REFERENCE_CYCLE_OPS,
    SymbolicIterationTarget,
    VolumetricBytecodeError,
    VolumetricBytecodeVM,
    VolumetricConfig,
    VolumetricOp,
)


def test_symbolic_target_is_described_without_materializing_power() -> None:
    target = SymbolicIterationTarget(1_000_000, 1_000_000)

    assert target.description == "10^6000000"
    assert target.decimal_exponent == 6_000_000
    assert target.log2_iterations == pytest.approx(6_000_000 * math.log2(10))


def test_sparse_modalities_do_not_allocate_logical_domain() -> None:
    vm = VolumetricBytecodeVM(VolumetricConfig(axis_extent=10**24, max_active_voxels=16))
    vm.load_modalities(
        {
            "gui": {(0, 0, 0): 7, (1, 2, 3): 0},
            "audio": {(9, 8, 7): 11},
        }
    )

    assert len(vm.state) == 2
    assert vm.permeate_snapshot().logical_axis_extent == 10**24


def test_exact_xyz_mirror_uses_logical_center() -> None:
    vm = VolumetricBytecodeVM(VolumetricConfig(axis_extent=16))

    assert vm.mirror_key(("gui", 1, 2, 3)) == ("gui", 14, 13, 12)
    assert vm.mirror_key(vm.mirror_key(("gui", 1, 2, 3))) == ("gui", 1, 2, 3)


def test_encode_decode_is_exact_for_all_byte_values() -> None:
    vm = VolumetricBytecodeVM(VolumetricConfig(axis_extent=8))
    field = {("gui", value % 8, 0, 0): value for value in range(1, 256, 32)}

    assert vm.decode(vm.encode(field)) == field


def test_mirror_injection_reaches_symmetric_fixed_point() -> None:
    vm = VolumetricBytecodeVM(
        VolumetricConfig(axis_extent=8, max_iterations=4, epsilon_active_change=0.0)
    )
    vm.load_modalities({"gui": {(0, 1, 2): 0b0000_0011}})

    result = vm.run()

    assert result.converged is True
    assert result.physical_iterations == 2
    assert result.symbolic_target == "10^6000000"
    assert vm.state[("gui", 0, 1, 2)] == 0b0000_0011
    assert vm.state[("gui", 7, 6, 5)] == 0b0000_0011
    assert result.receipts[0].changed_bits > 0
    assert result.receipts[1].changed_bits == 0
    assert result.receipts[1].active_change_fraction == 0.0


def test_octree_traversal_only_counts_active_tiles() -> None:
    vm = VolumetricBytecodeVM(VolumetricConfig(axis_extent=64, octree_depth=3))
    vm.load_modalities(
        {
            "gui": {(0, 0, 0): 1, (63, 63, 63): 2},
            "audio": {(0, 0, 0): 3},
        }
    )

    receipt = vm.step()

    assert receipt.octree_tiles == 3
    assert receipt.active_voxels_before == 3


def test_active_change_fraction_not_huge_logical_fraction_drives_convergence() -> None:
    vm = VolumetricBytecodeVM(
        VolumetricConfig(axis_extent=10**24, max_active_voxels=16, max_iterations=1)
    )
    vm.load_modalities({"gui": {(0, 0, 0): 1}})

    receipt = vm.step()

    assert receipt.changed_bits > 0
    assert receipt.logical_change_fraction < 1e-60
    assert receipt.active_change_fraction > 0.0
    assert receipt.converged is False


def test_execution_budget_is_physical_and_finite() -> None:
    vm = VolumetricBytecodeVM(
        VolumetricConfig(axis_extent=8, max_iterations=1, epsilon_active_change=0.0)
    )
    vm.load_modalities({"gui": {(0, 0, 0): 1}})

    result = vm.run()

    assert result.converged is False
    assert result.budget_exhausted is True
    assert result.physical_iterations == 1


def test_invalid_lane_and_active_set_fail_closed() -> None:
    with pytest.raises(VolumetricBytecodeError, match="8-bit lanes"):
        VolumetricConfig(lane_bits=16)

    vm = VolumetricBytecodeVM(VolumetricConfig(axis_extent=4, max_active_voxels=1))
    with pytest.raises(VolumetricBytecodeError, match="max_active_voxels"):
        vm.load_modalities({"gui": {(0, 0, 0): 1, (1, 1, 1): 2}})


def test_reference_trace_contains_verify_and_permeate_without_halt_per_cycle() -> None:
    assert VolumetricOp.VERIFY_CTR in REFERENCE_CYCLE_OPS
    assert VolumetricOp.PERMEATE in REFERENCE_CYCLE_OPS
    assert VolumetricOp.HALT not in REFERENCE_CYCLE_OPS


def test_mirror_symmetric_state_skips_redundant_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    vm = VolumetricBytecodeVM(
        VolumetricConfig(axis_extent=8, max_iterations=1, epsilon_active_change=0.0)
    )
    vm.load_modalities(
        {
            "gui": {
                (0, 1, 2): 0b0000_0011,
                (7, 6, 5): 0b0000_0011,
            }
        }
    )

    def unexpected_fold(_latent: object) -> object:
        raise AssertionError("mirror fold should not run for a symmetric fixed point")

    monkeypatch.setattr(vm, "fold_xyz_mirror_union", unexpected_fold)

    receipt = vm.step()

    assert receipt.converged is True
    assert receipt.changed_bits == 0
    assert receipt.codec_roundtrip_error_bits == 0


def test_pairwise_mirror_union_matches_expected_closure() -> None:
    vm = VolumetricBytecodeVM(VolumetricConfig(axis_extent=8))
    latent = {
        ("gui", 0, 1, 2): 0b0000_0011,
        ("gui", 7, 6, 5): 0b0000_0101,
    }

    folded = vm.fold_xyz_mirror_union(latent)

    assert folded == {
        ("gui", 0, 1, 2): 0b0000_0111,
        ("gui", 7, 6, 5): 0b0000_0111,
    }
