from pathlib import Path

import pytest
from fastapi import HTTPException

from jarvisx.cloud_os import DrMoagiCloudOS
from jarvisx.cloud_service import (
    FieldPayload,
    NodeRequest,
    OptimizeRequest,
    RoundTripRequest,
    _auth_dependency,
    create_app,
)


def _endpoint(app, path: str, method: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"missing {method} {path}")


def test_service_executes_roundtrip_and_optimization() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("api-node", max_cells=64)
    app = create_app(runtime)

    health = _endpoint(app, "/health", "GET")()
    assert health["status"] == "ok"
    assert health["ledger_valid"] is True

    roundtrip = _endpoint(app, "/v1/roundtrip", "POST")
    response = roundtrip(
        RoundTripRequest(
            request_id="api-roundtrip",
            field=FieldPayload(shape=(2, 2, 2), values=list(range(8))),
            latent_shape=(1, 1, 1),
        )
    )
    assert response["status"] == "succeeded"

    optimize = _endpoint(app, "/v1/auto-optimize", "POST")
    optimized = optimize(
        OptimizeRequest(
            request_id="api-optimize",
            field=FieldPayload(shape=(2, 2, 2), values=list(range(8))),
            complexity_weight=0.1,
            candidates=[(1, 1, 1), (2, 2, 2)],
        )
    )
    assert optimized["status"] == "succeeded"

    job = _endpoint(app, "/v1/jobs/{job_id}", "GET")(optimized["job_id"])
    assert job["job_id"] == optimized["job_id"]

    verified = _endpoint(app, "/v1/ledger/verify", "GET")()
    assert verified["valid"] is True
    assert verified["records"] >= 5


def test_service_registers_nodes_and_reports_conflict() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("seed", max_cells=8)
    app = create_app(runtime)
    register = _endpoint(app, "/v1/nodes", "POST")
    list_nodes = _endpoint(app, "/v1/nodes", "GET")

    created = register(NodeRequest(node_id="worker", max_cells=16, max_concurrency=2))
    assert created["node_id"] == "worker"
    assert len(list_nodes()) == 2

    with pytest.raises(HTTPException) as error:
        register(NodeRequest(node_id="worker", max_cells=16))
    assert error.value.status_code == 409


def test_service_maps_runtime_errors_to_http_errors() -> None:
    runtime = DrMoagiCloudOS()
    runtime.register_node("tiny", max_cells=1)
    app = create_app(runtime)

    roundtrip = _endpoint(app, "/v1/roundtrip", "POST")
    with pytest.raises(HTTPException) as error:
        roundtrip(
            RoundTripRequest(
                request_id="too-large",
                field=FieldPayload(shape=(2, 2, 2), values=list(range(8))),
                latent_shape=(1, 1, 1),
            )
        )
    assert error.value.status_code == 422

    get_job = _endpoint(app, "/v1/jobs/{job_id}", "GET")
    with pytest.raises(HTTPException) as missing:
        get_job("missing")
    assert missing.value.status_code == 404


def test_optional_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVISX_CLOUD_API_KEY", "secret")
    authorize = _auth_dependency()

    with pytest.raises(HTTPException) as error:
        authorize("wrong")
    assert error.value.status_code == 401

    authorize("secret")


def test_service_can_use_persistent_ledger(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    runtime = DrMoagiCloudOS(path)
    runtime.register_node("persisted", max_cells=8)
    assert runtime.ledger.verify()

    reloaded = DrMoagiCloudOS(path)
    assert reloaded.ledger.verify()
    assert len(reloaded.ledger.records) == 1
