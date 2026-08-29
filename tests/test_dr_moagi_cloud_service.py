from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from jarvisx.dr_moagi_cloud_runtime import (
    AtomicJobStore,
    DrMoagiCloudCoordinator,
    EchoExecutor,
    JobPolicy,
)
from jarvisx.dr_moagi_cloud_service import CloudServiceSettings, create_app


def make_client(tmp_path: Path, *, auth: bool = False) -> TestClient:
    settings = CloudServiceSettings(
        data_dir=tmp_path,
        api_key="secret" if auth else None,
        require_api_key=auth,
    )
    runtime = DrMoagiCloudCoordinator(
        executors={"echo.v1": EchoExecutor()},
        policy=JobPolicy(allowed_operations=frozenset({"echo.v1"})),
        store=AtomicJobStore(tmp_path),
    )
    return TestClient(create_app(settings, runtime))


def test_end_to_end_job_api(tmp_path):
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/jobs",
        json={"operation": "echo.v1", "input": {"value": 42}, "request_id": "r-1"},
    )
    assert response.status_code == 201
    job = response.json()
    assert job["state"] == "COMMITTED"

    fetched = client.get(f"/api/v1/jobs/{job['job_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["result"] == {"echo": {"value": 42}}

    verified = client.post(f"/api/v1/jobs/{job['job_id']}/verify")
    assert verified.status_code == 200
    assert verified.json()["verified"] is True

    events = client.get(f"/api/v1/jobs/{job['job_id']}/events")
    assert events.status_code == 200
    assert events.json()["events"][-1]["state"] == "COMMITTED"


def test_api_key_and_principal_are_enforced(tmp_path):
    client = make_client(tmp_path, auth=True)
    request = {"operation": "echo.v1", "input": {"value": 1}}
    assert client.post("/api/v1/jobs", json=request).status_code == 401

    response = client.post(
        "/api/v1/jobs",
        json=request,
        headers={"X-Dr-Moagi-Key": "secret", "X-Dr-Moagi-Principal": "user:test"},
    )
    assert response.status_code == 201
    assert response.json()["principal"] == "user:test"


def test_policy_denial_is_403(tmp_path):
    client = make_client(tmp_path)
    response = client.post("/api/v1/jobs", json={"operation": "shell.exec", "input": {}})
    assert response.status_code == 403


def test_health_and_metrics(tmp_path):
    client = make_client(tmp_path)
    assert client.get("/health/live").json()["status"] == "live"
    assert client.get("/health/ready").json()["store_ready"] is True
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "dr_moagi_cloud_jobs_stored 0" in metrics.text
