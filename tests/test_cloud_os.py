from pathlib import Path

import pytest

from jarvisx.cloud_os import DrMoagiCloudOS, Field3D, GeometricAutoencoder3D


def test_constant_field_round_trip_is_exact() -> None:
    field = Field3D.from_values([7.0] * 64, (4, 4, 4))
    result = GeometricAutoencoder3D().round_trip(field, (2, 2, 2))

    assert result.latent.shape == (2, 2, 2)
    assert result.latent.values == (7.0,) * 8
    assert result.reconstruction.values == field.values
    assert result.mse == 0.0
    assert result.compression_ratio == 0.125


def test_geometric_partition_reconstructs_region_means() -> None:
    field = Field3D.from_values(range(8), (2, 2, 2))
    result = GeometricAutoencoder3D().round_trip(field, (1, 1, 1))

    assert result.latent.values == (3.5,)
    assert result.reconstruction.values == (3.5,) * 8
    assert result.mse == pytest.approx(5.25)


def test_cloud_auto_optimize_commits_verified_job(tmp_path: Path) -> None:
    runtime = DrMoagiCloudOS(tmp_path / "ledger.jsonl")
    runtime.register_node("node-a", max_cells=512, max_concurrency=2)

    field = Field3D.from_values(
        [float((x + y + z) % 3) for z in range(4) for y in range(4) for x in range(4)],
        (4, 4, 4),
    )
    job = runtime.auto_optimize(
        field,
        request_id="optimize-001",
        complexity_weight=0.1,
        candidates=((1, 1, 1), (2, 2, 2), (4, 4, 4)),
    )

    snapshot = runtime.job_snapshot(job.job_id)
    assert snapshot["status"] == "succeeded"
    assert snapshot["node_id"] == "node-a"
    assert snapshot["result"]["selected_latent_shape"] in (
        [1, 1, 1],
        [2, 2, 2],
        [4, 4, 4],
    )
    assert runtime.ledger.verify()
    assert (tmp_path / "ledger.jsonl").exists()


def test_request_id_is_idempotent() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("node", max_cells=8)
    field = Field3D.from_values(range(8), (2, 2, 2))

    first = runtime.round_trip(field, (1, 1, 1), request_id="same-request")
    second = runtime.round_trip(field, (1, 1, 1), request_id="same-request")

    assert first is second
    assert len(runtime.jobs) == 1
    assert len([record for record in runtime.ledger.records if record["event"] == "job.committed"]) == 1


def test_scheduler_prefers_smallest_eligible_idle_node() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("large", max_cells=128)
    runtime.register_node("small", max_cells=16)
    field = Field3D.from_values(range(8), (2, 2, 2))

    job = runtime.round_trip(field, (1, 1, 1), request_id="schedule")

    assert job.node_id == "small"


def test_field_and_candidate_bounds_are_enforced() -> None:
    with pytest.raises(ValueError):
        Field3D.from_values([1.0, 2.0], (1, 1, 1))

    runtime = DrMoagiCloudOS()
    runtime.register_node("tiny", max_cells=1)
    field = Field3D.from_values(range(8), (2, 2, 2))
    with pytest.raises(RuntimeError):
        runtime.round_trip(field, (1, 1, 1), request_id="too-large")
