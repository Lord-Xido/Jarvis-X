from __future__ import annotations

import time

import pytest

from jarvisx.cloud import (
    HierarchicalAddress,
    HyperCloudWorker,
    MediaEnvelope,
    MediaKind,
    OperationalHyperCloud,
    ShardCoordinate,
    SQLiteStateStore,
)
from jarvisx.cloud.inference import LocalReferenceBackend


def _state(tmp_path):
    return SQLiteStateStore(
        str(tmp_path / "state.db"),
        str(tmp_path / "objects"),
    )


def test_sparse_parameters_survive_store_restart(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    address = HierarchicalAddress((17, 42, 999_999))

    service.set_parameter("tenant-a", address, 0.875)
    assert service.get_parameter("tenant-a", address) == pytest.approx(0.875)
    state.close()

    reopened = _state(tmp_path)
    service2 = OperationalHyperCloud(state=reopened)
    assert service2.get_parameter("tenant-a", address) == pytest.approx(0.875)
    assert reopened.parameter_count() == 1
    reopened.close()


def test_media_is_content_addressed_and_integrity_checked(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    media = MediaEnvelope(
        kind=MediaKind.VIDEO,
        payload=b"frame-stream-" * 200,
        content_type="video/example",
    )

    record = service.ingest("demo", media)
    restored = state.get_media("demo", media.digest)

    assert record["sha256"] == media.digest
    assert restored is not None
    assert restored.payload == media.payload
    assert state.media_count() == 1
    state.close()


def test_identical_media_can_exist_in_multiple_namespaces_without_metadata_collision(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    media = MediaEnvelope(
        kind=MediaKind.AUDIO,
        payload=b"same-audio-bytes" * 50,
        content_type="audio/example",
    )

    service.ingest("tenant-a", media)
    service.ingest("tenant-b", media)

    tenant_a = state.media_record("tenant-a", media.digest)
    tenant_b = state.media_record("tenant-b", media.digest)
    assert tenant_a is not None and tenant_a["namespace"] == "tenant-a"
    assert tenant_b is not None and tenant_b["namespace"] == "tenant-b"
    assert state.media_count() == 2
    state.close()


def test_codec_job_runs_end_to_end_through_durable_worker(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    media = MediaEnvelope(
        kind=MediaKind.IMAGE,
        payload=b"repeating-pixel-data" * 500,
        content_type="image/example",
    )
    service.ingest("demo", media)
    queued = service.enqueue_codec("demo", media.digest)

    assert queued.target is not None
    worker = HyperCloudWorker(
        state,
        backend=LocalReferenceBackend(),
        worker_id="codec-worker",
        coordinate=queued.target,
    )
    assert worker.run_once() is True

    completed = service.job(queued.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result is not None
    assert completed.result["lossless"] is True
    assert completed.result["reconstructed_sha256"] == media.digest
    assert completed.result["compression_ratio"] < 1.0
    assert completed.attempts == 1
    assert completed.lease_owner is None
    state.close()


def test_codec_job_cannot_cross_namespace_boundary(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    media = MediaEnvelope(
        kind=MediaKind.BINARY,
        payload=b"private-tenant-object",
        content_type="application/octet-stream",
    )
    service.ingest("tenant-a", media)

    with pytest.raises(KeyError):
        service.enqueue_codec("tenant-b", media.digest)
    state.close()


def test_chat_job_runs_end_to_end_without_external_credentials(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    queued = service.enqueue_chat(
        "demo",
        "Explain sparse virtual parameters.",
        "Be concise",
    )

    assert queued.target is not None
    worker = HyperCloudWorker(
        state,
        backend=LocalReferenceBackend(),
        worker_id="chat-worker",
        coordinate=queued.target,
    )
    assert worker.run_once() is True

    completed = service.job(queued.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result is not None
    assert completed.result["backend"] == "local-reference-non-llm"
    assert completed.result["neural_model"] is False
    assert "sparse virtual parameters" in str(completed.result["text"])
    state.close()


def test_worker_only_claims_supported_operations(tmp_path) -> None:
    state = _state(tmp_path)
    bad = state.create_job("demo", "not-supported", {})
    worker = HyperCloudWorker(
        state,
        backend=LocalReferenceBackend(),
        worker_id="bounded-worker",
        capabilities=("chat", "codec"),
    )

    assert worker.run_once() is False
    untouched = state.get_job(bad.id)
    assert untouched is not None
    assert untouched.status == "queued"
    assert untouched.attempts == 0
    state.close()


def test_worker_registry_and_placement_choose_nearest_capable_worker(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    queued = service.enqueue_chat("demo", "Route me geometrically")
    assert queued.target is not None

    target = queued.target
    far = ShardCoordinate((target.x + 7) % 16, (target.y + 7) % 16, (target.z + 7) % 16)
    state.register_worker(
        worker_id="far-worker",
        coordinate=far,
        capabilities=("chat",),
        backend="test",
        load=0.0,
    )
    state.register_worker(
        worker_id="near-worker",
        coordinate=target,
        capabilities=("chat",),
        backend="test",
        load=0.2,
    )
    state.register_worker(
        worker_id="codec-only",
        coordinate=target,
        capabilities=("codec",),
        backend="test",
        load=0.0,
    )

    decision = service.placement_preview(queued.id)
    assert decision is not None
    assert decision.worker_id == "near-worker"
    assert decision.distance == 0
    assert len(state.active_workers()) == 3
    state.close()


def test_claim_prefers_spatially_local_job_over_older_remote_job(tmp_path) -> None:
    state = _state(tmp_path)
    remote = state.create_job(
        "demo",
        "chat",
        {"prompt": "remote"},
        target=ShardCoordinate(15, 15, 15),
    )
    local = state.create_job(
        "demo",
        "chat",
        {"prompt": "local"},
        target=ShardCoordinate(1, 1, 1),
    )

    claimed = state.claim_next_job(
        worker_id="worker-a",
        coordinate=ShardCoordinate(1, 1, 1),
        operations=("chat",),
        lease_seconds=10.0,
    )

    assert claimed is not None
    assert claimed.id == local.id
    assert claimed.id != remote.id
    assert claimed.lease_owner == "worker-a"
    assert claimed.attempts == 1
    state.close()


def test_expired_job_lease_is_requeued_and_reclaimable(tmp_path) -> None:
    state = _state(tmp_path)
    queued = state.create_job(
        "demo",
        "chat",
        {"prompt": "recover me"},
        target=ShardCoordinate(2, 2, 2),
        max_attempts=2,
    )
    first = state.claim_next_job(
        worker_id="worker-dead",
        coordinate=ShardCoordinate(2, 2, 2),
        operations=("chat",),
        lease_seconds=0.01,
    )
    assert first is not None and first.id == queued.id

    time.sleep(0.02)
    assert state.requeue_expired_leases() == 1
    recovered = state.get_job(queued.id)
    assert recovered is not None
    assert recovered.status == "queued"
    assert recovered.lease_owner is None

    second = state.claim_next_job(
        worker_id="worker-replacement",
        coordinate=ShardCoordinate(2, 2, 2),
        operations=("chat",),
        lease_seconds=10.0,
    )
    assert second is not None
    assert second.id == queued.id
    assert second.attempts == 2
    state.close()


def test_expired_lease_fails_job_after_maximum_attempts(tmp_path) -> None:
    state = _state(tmp_path)
    queued = state.create_job(
        "demo",
        "chat",
        {"prompt": "one shot"},
        max_attempts=1,
    )
    claimed = state.claim_next_job(
        worker_id="worker-dead",
        coordinate=ShardCoordinate(0, 0, 0),
        operations=("chat",),
        lease_seconds=0.01,
    )
    assert claimed is not None

    time.sleep(0.02)
    state.requeue_expired_leases()
    failed = state.get_job(queued.id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error == "worker lease expired after maximum attempts"
    state.close()


def test_lease_owner_fences_stale_worker_completion(tmp_path) -> None:
    state = _state(tmp_path)
    queued = state.create_job("demo", "chat", {"prompt": "fence me"})
    claimed = state.claim_next_job(
        worker_id="worker-owner",
        coordinate=ShardCoordinate(0, 0, 0),
        operations=("chat",),
        lease_seconds=10.0,
    )
    assert claimed is not None

    assert state.complete_job(queued.id, {"text": "stale"}, worker_id="worker-other") is False
    still_running = state.get_job(queued.id)
    assert still_running is not None and still_running.status == "running"

    assert state.complete_job(queued.id, {"text": "committed"}, worker_id="worker-owner") is True
    completed = state.get_job(queued.id)
    assert completed is not None and completed.status == "succeeded"
    assert completed.result == {"text": "committed"}
    state.close()


def test_runtime_description_reports_permeated_boundaries(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)
    HyperCloudWorker(
        state,
        backend=LocalReferenceBackend(),
        worker_id="registered-worker",
        coordinate=ShardCoordinate(3, 4, 5),
    )

    description = service.describe(backend_name="local-reference-non-llm")

    assert description["status"] == "operational-permeated"
    assert description["deployment_lattice"]["active_workers"] == 1
    assert description["claims"]["durable_local_state"] is True
    assert description["claims"]["leased_worker_execution"] is True
    assert description["claims"]["abandoned_job_recovery"] is True
    assert description["claims"]["topology_aware_3d_placement"] is True
    assert description["claims"]["lossless_multimodal_codec"] is True
    assert description["claims"]["dense_parameter_allocation"] is False
    assert description["claims"]["distributed_accelerator_backend"] is False
    state.close()
