from __future__ import annotations

import math

import pytest

from jarvisx.dr_moagi_worker_fabric import FabricConfig, LogicalWorkerFabric


def test_exact_6400_cube_geometry() -> None:
    fabric = LogicalWorkerFabric(FabricConfig(physical_workers=2))
    status = fabric.status()

    assert status["logical_dimensions"] == [6400, 6400, 6400]
    assert status["logical_workers"] == 262_144_000_000
    assert status["brick_side"] == 64
    assert status["brick_dimensions"] == [100, 100, 100]
    assert status["logical_bricks"] == 1_000_000
    assert status["workers_per_brick"] == 262_144


@pytest.mark.parametrize(
    "xyz",
    [
        (0, 0, 0),
        (6399, 0, 0),
        (0, 6399, 0),
        (0, 0, 6399),
        (6399, 6399, 6399),
        (123, 4567, 3201),
    ],
)
def test_worker_id_round_trip(xyz: tuple[int, int, int]) -> None:
    fabric = LogicalWorkerFabric(FabricConfig(physical_workers=1))
    worker_id = fabric.linear_id(*xyz)
    assert fabric.coordinate(worker_id) == xyz


def test_step_separates_logical_active_resident_and_physical_counts() -> None:
    fabric = LogicalWorkerFabric(
        FabricConfig(
            physical_workers=3,
            active_worker_budget=512,
            resident_brick_limit=16,
        )
    )
    receipt = fabric.step(sample_size=32)

    assert receipt["logical_workers"] == 262_144_000_000
    assert receipt["active_workers_executed"] == 512
    assert receipt["physical_workers"] == 3
    assert 1 <= receipt["resident_bricks"] <= 16
    assert receipt["active_bricks_touched"] >= receipt["resident_bricks"]
    assert receipt["execution_fraction"] == pytest.approx(512 / 262_144_000_000)
    assert math.isfinite(receipt["residual_rms"])
    assert math.isfinite(receipt["checksum"])
    assert receipt["elapsed_ms"] >= 0.0
    assert len(receipt["sample"]) == 32
    for point in receipt["sample"]:
        x, y, z = point["xyz"]
        assert 0 <= x < 6400
        assert 0 <= y < 6400
        assert 0 <= z < 6400


def test_first_cycle_is_deterministic_for_same_config() -> None:
    config = FabricConfig(physical_workers=2, active_worker_budget=128, seed=99)
    left = LogicalWorkerFabric(config).step(sample_size=8)
    right = LogicalWorkerFabric(config).step(sample_size=8)

    assert left["checksum"] == pytest.approx(right["checksum"])
    assert left["residual_rms"] == pytest.approx(right["residual_rms"])
    assert left["sample"] == right["sample"]


def test_invalid_physical_materialization_claim_is_rejected_by_config_bounds() -> None:
    with pytest.raises(ValueError):
        FabricConfig(physical_workers=6400).validate()
