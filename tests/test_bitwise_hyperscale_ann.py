from __future__ import annotations

import math

import pytest

from jarvisx.bitwise_hyperscale_ann import (
    AXIS_LIMIT,
    CORE_THRESHOLD,
    MORTON_LIMIT,
    BitwiseAddress,
    bit_truncate_fp32_to_int8,
    fp32_top_byte,
    inside_singularity_core,
    inward_vortex_step,
    lod_cluster,
    lod_key,
    lod_mask,
    morton60_decode,
    morton60_encode,
    octree_child_index,
    prune_truncated_int8,
    reference_q16_projection,
)
from jarvisx.cloud_control_plane import CloudControlPlane, TileTask, WorkerState


def worker(worker_id: str, resident_tiles: frozenset[str]) -> WorkerState:
    return WorkerState(
        worker_id=worker_id,
        region="test",
        kind="gpu",
        capacity_units=16.0,
        free_units=12.0,
        queue_depth=0,
        latency_ms=2.0,
        resident_tiles=resident_tiles,
    )


def test_morton60_roundtrip_and_source_bit_order() -> None:
    assert morton60_decode(morton60_encode(0, 0, 0)) == (0, 0, 0)
    assert morton60_decode(morton60_encode(999_999, 456_789, 12_345)) == (
        999_999,
        456_789,
        12_345,
    )
    assert morton60_encode(AXIS_LIMIT - 1, AXIS_LIMIT - 1, AXIS_LIMIT - 1) == (
        MORTON_LIMIT - 1
    )

    assert morton60_encode(1 << 19, 0, 0) == 1 << 59
    assert morton60_encode(0, 1 << 19, 0) == 1 << 58
    assert morton60_encode(0, 0, 1 << 19) == 1 << 57


def test_octree_child_indices_follow_root_to_leaf_groups() -> None:
    key = morton60_encode((1 << 19) | 1, 1 << 18, 1 << 17)

    assert octree_child_index(key, 0) == 0b100
    assert octree_child_index(key, 1) == 0b010
    assert octree_child_index(key, 2) == 0b001
    assert octree_child_index(key, 19) == 0b100


def test_lod_mask_cluster_geometry_and_cloud_tile_id() -> None:
    address = BitwiseAddress.from_xyz(900_000, 400_000, 100_000)
    cluster = address.cluster(10)

    assert lod_key(address.key, 10) == address.key & lod_mask(10)
    assert cluster.edge == 1 << 10
    assert cluster.population == cluster.edge**3
    assert all(
        low <= value <= high
        for value, low, high in zip(
            (address.x, address.y, address.z),
            cluster.minimum,
            cluster.maximum,
        )
    )
    assert cluster.cloud_tile_id.startswith("m60:d10:p")


def test_lod_depth_twenty_resolves_one_logical_coordinate() -> None:
    key = morton60_encode(123, 456, 789)
    cluster = lod_cluster(key, 20)

    assert cluster.edge == 1
    assert cluster.population == 1
    assert cluster.minimum == (123, 456, 789)
    assert cluster.maximum == (123, 456, 789)
    assert cluster.centroid == (123.0, 456.0, 789.0)


def test_reference_fixed_point_projection_preserves_source_equation() -> None:
    assert reference_q16_projection(-10.0) == 0
    assert reference_q16_projection(0.0) == (1 << 19)
    assert reference_q16_projection(10.0) == (1 << 20)

    with pytest.raises(ValueError):
        reference_q16_projection(math.inf)
    with pytest.raises(ValueError):
        reference_q16_projection(10.01)


def test_inward_vortex_step_is_deterministic_and_contracts_positive_z() -> None:
    point = (120_000, -80_000, 64_000)
    first = inward_vortex_step(point)
    second = inward_vortex_step(point)

    assert first == second
    assert abs(first[2]) < abs(point[2])


def test_singularity_core_uses_axis_aligned_source_threshold() -> None:
    assert inside_singularity_core((CORE_THRESHOLD, 0, -CORE_THRESHOLD))
    assert not inside_singularity_core((CORE_THRESHOLD + 1, 0, 0))


def test_fp32_top_byte_truncation_is_structural_not_numeric_quantization() -> None:
    assert fp32_top_byte(1.0) == 0x3F
    assert bit_truncate_fp32_to_int8(1.0) == 63
    assert fp32_top_byte(-1.0) == 0xBF
    assert bit_truncate_fp32_to_int8(-1.0) == -65


def test_absolute_magnitude_pruning_rule() -> None:
    assert prune_truncated_int8(7, 8) == 0
    assert prune_truncated_int8(-7, 8) == 0
    assert prune_truncated_int8(8, 8) == 8
    assert prune_truncated_int8(-8, 8) == -8


def test_lod_prefix_can_route_as_cloud_control_plane_tile() -> None:
    cluster = BitwiseAddress.from_xyz(700_000, 200_000, 300_000).cluster(8)
    tile_id = cluster.cloud_tile_id
    plane = CloudControlPlane(
        {
            "gpu-local": worker("gpu-local", frozenset({tile_id})),
            "gpu-remote": worker("gpu-remote", frozenset()),
        }
    )
    plane.register_tile(tile_id, {"prefix": cluster.prefix})

    task = TileTask(
        task_id="morton-task",
        tile_id=tile_id,
        operation="INWARD_FOLD",
        parent_version=0,
        bytes_estimate=2_000_000_000,
        preferred_kind="gpu",
    )
    result = plane.dispatch(
        task,
        executor=lambda _task, _worker, parent: {
            "prefix": parent.value["prefix"],
            "folded": True,
        },
        verifier=lambda *_: True,
    )

    assert result.route.worker_id == "gpu-local"
    assert result.receipt.decision == "commit"
    assert plane.tile_state(tile_id).version == 1
