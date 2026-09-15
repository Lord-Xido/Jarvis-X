from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from jarvisx.dr_moagi_3d_1000x import (  # noqa: E402
    DEFAULT_ACTIVE_FRACTION,
    DrMoagi3D1000xEngine,
    EngineConfig,
    MultimodalDescriptor,
    pack_voxel64,
)


def test_canonical_geometry_targets_exact_1000x_work_reduction() -> None:
    engine = DrMoagi3D1000xEngine(
        EngineConfig(
            cube_edge=1000,
            tile_edge=10,
            active_fraction=0.001,
            latent_dim=4,
        )
    )

    assert engine.virtual_voxels == 1_000_000_000
    assert engine.virtual_tiles == 1_000_000
    assert engine.target_active_tiles == 1_000
    assert engine.active_voxels == 1_000_000
    assert engine.work_reduction_factor == pytest.approx(1000.0)


def test_engine_never_materializes_dense_cube() -> None:
    engine = DrMoagi3D1000xEngine(EngineConfig(max_active_tiles=32))

    assert engine.tile_features.shape == (32, 8)
    assert engine.omega.shape == (32, 8)
    assert engine.tile_features.size == 256
    assert engine.tile_features.size << engine.virtual_voxels


def test_multimodal_bytes_cycle_and_control_words() -> None:
    engine = DrMoagi3D1000xEngine(EngineConfig(max_active_tiles=64, latent_dim=4))
    engine.ingest_bytes(
        b"text image audio video 3D code -> encode -> omega -> decode -> residual"
    )

    report = engine.cycle()
    words = engine.packed_control_words(opcode=8)

    assert report.cycle == 1
    assert report.active_tiles == 64
    assert report.active_voxels == 64_000
    assert math.isfinite(report.residual_mse)
    assert report.total_ms >= 0.0
    assert words.dtype == np.uint64
    assert words.shape == (64,)


def test_cloud_plan_partitions_every_active_tile_once() -> None:
    engine = DrMoagi3D1000xEngine(EngineConfig(max_active_tiles=17))
    plan = engine.cloud_plan(workers=4)

    assigned = [tile for worker in plan["assignments"] for tile in worker]
    assert sorted(assigned) == list(range(17))
    assert len(set(assigned)) == 17
    assert plan["active_tiles"] == 17


def test_voxel64_pack_matches_documented_bit_layout() -> None:
    word = pack_voxel64(
        np.array([0x1234]),
        np.array([0xABC]),
        np.array([0x56]),
        np.array([0x78]),
        np.array([0xDEF]),
        np.array([0x90]),
    )[0]

    expected = 0x1234
    expected = (expected << 12) | 0xABC
    expected = (expected << 8) | 0x56
    expected = (expected << 8) | 0x78
    expected = (expected << 12) | 0xDEF
    expected = (expected << 8) | 0x90
    assert int(word) == expected


def test_descriptor_is_deterministic() -> None:
    payload = b"same multimedia payload"
    left = MultimodalDescriptor.from_bytes(payload, width=128)
    right = MultimodalDescriptor.from_bytes(payload, width=128)

    assert np.array_equal(left, right)
    assert left.shape == (128,)
    assert np.all((left >= 0.0) & (left <= 1.0))


def test_snapshot_does_not_claim_wall_clock_1000x() -> None:
    engine = DrMoagi3D1000xEngine(EngineConfig(max_active_tiles=8))
    snapshot = engine.snapshot()

    assert snapshot["wall_clock_1000x_guaranteed"] is False
    assert snapshot["work_reduction_factor"] > 0.0


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        EngineConfig(active_fraction=0.0).validate()
    with pytest.raises(ValueError):
        EngineConfig(tile_edge=1001).validate()
    with pytest.raises(ValueError):
        EngineConfig(omega_decay=1.0).validate()


def test_default_active_fraction_is_one_per_thousand() -> None:
    assert DEFAULT_ACTIVE_FRACTION == pytest.approx(1.0 / 1000.0)
