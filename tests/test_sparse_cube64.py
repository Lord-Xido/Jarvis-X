from __future__ import annotations

import pytest

from jarvisx.sparse_cube64 import (
    ONE_MIB,
    SPARSE_RECORD_BYTES,
    SparseVoxelRecord64,
    VoxelWord64,
    pack_coord30,
    plan_sparse_cube64,
    unpack_coord30,
)


def test_voxel_word64_round_trip_preserves_all_fields() -> None:
    state = VoxelWord64(
        payload=0xBEEF,
        feature=0xABC,
        memory=0x7D,
        activation=0x91,
        residual=0xFED,
        opcode=0x42,
    )

    assert state.word.bit_length() <= 64
    assert len(state.to_bytes()) == 8
    assert VoxelWord64.from_word(state.word) == state
    assert VoxelWord64.from_bytes(state.to_bytes()) == state


def test_voxel_word64_rejects_field_overflow() -> None:
    with pytest.raises(ValueError, match="feature"):
        VoxelWord64(feature=1 << 12)


def test_coordinate_pack_round_trip_for_1000_cube_boundary() -> None:
    coordinate = (999, 999, 999)
    packed = pack_coord30(*coordinate)

    assert packed < 1 << 30
    assert unpack_coord30(packed) == coordinate


def test_sparse_record_is_exactly_twelve_bytes() -> None:
    record = SparseVoxelRecord64(
        (999, 500, 0),
        VoxelWord64(payload=7, residual=13, opcode=2),
    )
    payload = record.to_bytes()

    assert SPARSE_RECORD_BYTES == 12
    assert len(payload) == 12
    assert SparseVoxelRecord64.from_bytes(payload) == record


def test_default_1000_cube_plan_is_virtual_not_dense() -> None:
    plan = plan_sparse_cube64()

    assert plan.shape == (1000, 1000, 1000)
    assert plan.virtual_cells == 1_000_000_000
    assert plan.dense_descriptor_bytes == 8_000_000_000
    assert plan.budget_bytes == ONE_MIB
    assert plan.max_active_records == 87_381
    assert plan.tile_shape == (10, 10, 10)
    assert plan.tile_cells == 1_000
    assert plan.max_full_tiles == 87
    assert plan.committed_record_bytes <= ONE_MIB
    assert plan.committed_record_bytes + plan.unused_budget_bytes == ONE_MIB
    assert plan.max_active_fraction < 0.0001


def test_planner_rejects_unaddressable_axis_and_invalid_budget() -> None:
    with pytest.raises(ValueError, match="cannot exceed 1024"):
        plan_sparse_cube64((1025, 1, 1))
    with pytest.raises(ValueError, match="budget_bytes"):
        plan_sparse_cube64(budget_bytes=0)
    with pytest.raises(ValueError, match="record_bytes"):
        plan_sparse_cube64(record_bytes=7)
