from __future__ import annotations

import pytest

from jarvisx.cloud import (
    HierarchicalAddress,
    HyperCloudWorker,
    MediaEnvelope,
    MediaKind,
    OperationalHyperCloud,
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

    assert state.media_record("tenant-a", media.digest)["namespace"] == "tenant-a"
    assert state.media_record("tenant-b", media.digest)["namespace"] == "tenant-b"
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

    worker = HyperCloudWorker(state, backend=LocalReferenceBackend())
    assert worker.run_once() is True

    completed = service.job(queued.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result is not None
    assert completed.result["lossless"] is True
    assert completed.result["reconstructed_sha256"] == media.digest
    assert completed.result["compression_ratio"] < 1.0
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

    worker = HyperCloudWorker(state, backend=LocalReferenceBackend())
    assert worker.run_once() is True

    completed = service.job(queued.id)
    assert completed is not None
    assert completed.status == "succeeded"
    assert completed.result is not None
    assert completed.result["backend"] == "local-reference-non-llm"
    assert completed.result["neural_model"] is False
    assert "sparse virtual parameters" in str(completed.result["text"])
    state.close()


def test_worker_persists_failure_for_unknown_operation(tmp_path) -> None:
    state = _state(tmp_path)
    bad = state.create_job("demo", "not-supported", {})
    worker = HyperCloudWorker(state, backend=LocalReferenceBackend())

    assert worker.run_once() is True
    failed = state.get_job(bad.id)

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error is not None
    assert "unsupported job operation" in failed.error
    state.close()


def test_runtime_description_reports_operational_boundaries(tmp_path) -> None:
    state = _state(tmp_path)
    service = OperationalHyperCloud(state=state)

    description = service.describe(backend_name="local-reference-non-llm")

    assert description["status"] == "operational-reference"
    assert description["claims"]["durable_local_state"] is True
    assert description["claims"]["lossless_multimodal_codec"] is True
    assert description["claims"]["dense_parameter_allocation"] is False
    assert description["claims"]["distributed_accelerator_backend"] is False
    state.close()
