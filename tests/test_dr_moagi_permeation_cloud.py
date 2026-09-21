from __future__ import annotations

from fastapi.testclient import TestClient

from jarvisx.dr_moagi_cloud_service import CloudServiceSettings, create_app


def _state() -> dict[str, object]:
    return {
        "latent": [index / 10.0 for index in range(16)],
        "geometry": {"vpx_density": 0.72},
        "intent": "externalize-and-reconstruct",
    }


def test_permeation_job_commits_and_replays_through_cloud_gate(tmp_path):
    settings = CloudServiceSettings(
        data_dir=tmp_path,
        api_key="test-secret",
        require_api_key=True,
    )
    client = TestClient(create_app(settings=settings))
    headers = {
        "X-Dr-Moagi-Key": "test-secret",
        "X-Dr-Moagi-Principal": "permeation-test",
    }

    response = client.post(
        "/api/v1/jobs",
        headers=headers,
        json={
            "operation": "permeate-roundtrip.v1",
            "request_id": "permeation-001",
            "input": {
                "state": _state(),
                "config": {
                    "carrier_hz": 333_330_000.0,
                    "range_m": 1.0,
                    "axis": [0.0, 1.0, 0.0],
                    "receiver_direction": [0.0, 1.0, 0.0],
                },
            },
        },
    )

    assert response.status_code == 201
    job = response.json()
    assert job["state"] == "COMMITTED"
    assert job["principal"] == "permeation-test"
    result = job["result"]
    assert result["operation"] == "permeate-roundtrip.v1"
    assert result["physical_rf"] is False
    assert result["verified"] is True
    assert result["reconstructed"] == _state()

    job_id = job["job_id"]
    verify_response = client.post(f"/api/v1/jobs/{job_id}/verify", headers=headers)
    assert verify_response.status_code == 200
    assert verify_response.json()["verified"] is True

    events_response = client.get(f"/api/v1/jobs/{job_id}/events", headers=headers)
    assert events_response.status_code == 200
    assert events_response.json()["events"][-1]["state"] == "COMMITTED"


def test_permeation_rejects_unknown_transport_configuration(tmp_path):
    settings = CloudServiceSettings(data_dir=tmp_path)
    client = TestClient(create_app(settings=settings))

    response = client.post(
        "/api/v1/jobs",
        json={
            "operation": "permeate-roundtrip.v1",
            "input": {
                "state": _state(),
                "config": {"physical_transmitter_gain": 1000},
            },
        },
    )

    assert response.status_code == 422
    assert "unknown permeation config keys" in response.json()["detail"]
