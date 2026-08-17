from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvisx.dr_moagi_cloud_runtime import (
    AtomicJobStore,
    DrMoagiCloudCoordinator,
    EchoExecutor,
    JobPolicy,
    JobState,
    ResourceLimits,
)


def coordinator(tmp_path: Path, verifier=None):
    kwargs = {}
    if verifier is not None:
        kwargs["verifier"] = verifier
    return DrMoagiCloudCoordinator(
        executors={"echo.v1": EchoExecutor()},
        policy=JobPolicy(
            allowed_operations=frozenset({"echo.v1"}),
            limits=ResourceLimits(
                max_input_bytes=1024,
                max_output_bytes=2048,
                max_runtime_ms=1000,
            ),
        ),
        store=AtomicJobStore(tmp_path),
        clock_ms=iter(range(1000, 2000)).__next__,
        **kwargs,
    )


def test_successful_job_commits_with_replayable_hash_chain(tmp_path):
    runtime = coordinator(tmp_path)
    job = runtime.submit(principal="user:doctor", operation="echo.v1", payload={"x": 7})

    assert job["state"] == JobState.COMMITTED.value
    assert job["result"] == {"echo": {"x": 7}}
    assert [event["state"] for event in job["events"]] == [
        "RECEIVED",
        "VALIDATED",
        "PLANNED",
        "DISPATCHED",
        "RUNNING",
        "VERIFIED",
        "COMMITTED",
    ]
    verification = runtime.verify_job(job["job_id"])
    assert verification == {
        "job_id": job["job_id"],
        "verified": True,
        "reason": "verified",
        "state": "COMMITTED",
        "event_count": 7,
    }


def test_verifier_rejection_prevents_commit(tmp_path):
    runtime = coordinator(
        tmp_path,
        verifier=lambda result, context: (False, "policy test failed"),
    )
    job = runtime.submit(principal="user:doctor", operation="echo.v1", payload={"x": 7})

    assert job["state"] == JobState.REJECTED.value
    assert job["result"] is None
    assert job["result_digest"] is None
    assert job["verification"] == {"passed": False, "reason": "policy test failed"}
    assert job["events"][-1]["details"]["reason"].startswith("verification failed")


def test_input_limit_fails_before_job_creation(tmp_path):
    runtime = coordinator(tmp_path)
    with pytest.raises(ValueError, match="max_input_bytes"):
        runtime.submit(
            principal="user:doctor",
            operation="echo.v1",
            payload={"x": "z" * 2000},
        )
    assert runtime.store.count() == 0


def test_operation_policy_is_fail_closed(tmp_path):
    runtime = coordinator(tmp_path)
    with pytest.raises(PermissionError, match="not allowed"):
        runtime.submit(principal="user:doctor", operation="shell.exec", payload={})
    assert runtime.store.count() == 0


def test_tampering_is_detected(tmp_path):
    runtime = coordinator(tmp_path)
    job = runtime.submit(principal="user:doctor", operation="echo.v1", payload={"x": 7})
    path = tmp_path / f"{job['job_id']}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["echo"]["x"] = 8
    path.write_text(json.dumps(payload), encoding="utf-8")

    verification = runtime.verify_job(job["job_id"])
    assert not verification["verified"]
    assert verification["reason"] == "envelope digest mismatch"


def test_executor_exception_is_journalled_failed(tmp_path):
    class Boom:
        def execute(self, payload, limits):
            raise RuntimeError("boom")

    runtime = DrMoagiCloudCoordinator(
        executors={"boom.v1": Boom()},
        policy=JobPolicy(allowed_operations=frozenset({"boom.v1"})),
        store=AtomicJobStore(tmp_path),
    )
    with pytest.raises(RuntimeError, match="boom"):
        runtime.submit(principal="user:doctor", operation="boom.v1", payload={})

    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    failed = json.loads(paths[0].read_text(encoding="utf-8"))
    assert failed["state"] == JobState.FAILED.value
    assert failed["events"][-1]["state"] == JobState.FAILED.value
