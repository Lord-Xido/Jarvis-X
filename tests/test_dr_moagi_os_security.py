"""Focused access-gate regression tests using an isolated ASGI test fixture."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jarvisx.dr_moagi_os_security import (
    authorise_bearer,
    control_access_middleware,
    requires_control_auth,
)

TOKEN = "test-only-control-token-of-at-least-32-chars"


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("JARVISX_OS_CONTROL_TOKEN", raising=False)
    app = FastAPI()
    app.middleware("http")(control_access_middleware)
    calls = {"writes": 0, "exports": 0}

    @app.get("/healthz")
    def health():
        return {"ok": True}

    @app.get("/v1/os/status")
    def status():
        return {"ready": True}

    @app.get("/v1/os/export")
    def export():
        calls["exports"] += 1
        return {"secret": "sensitive"}

    @app.post("/v1/os/boot")
    def boot():
        calls["writes"] += 1
        return {"state": "booted"}

    @app.delete("/other/path")
    def delete():
        calls["writes"] += 1
        return {"state": "deleted"}

    with TestClient(app) as client:
        yield client, calls, monkeypatch


def test_path_policy():
    assert requires_control_auth("POST", "/v1/os/boot")
    assert requires_control_auth("DELETE", "/other/path")
    assert requires_control_auth("GET", "/v1/os/export")
    assert requires_control_auth("GET", "/v1/os/snapshot")
    assert not requires_control_auth("GET", "/healthz")
    assert not requires_control_auth("GET", "/v1/os/status")


def test_fail_closed_when_not_configured(api):
    client, calls, _ = api
    assert client.post("/v1/os/boot").status_code == 503
    assert calls["writes"] == 0


def test_fail_closed_for_short_secret(api):
    client, calls, env = api
    env.setenv("JARVISX_OS_CONTROL_TOKEN", "weak")
    assert client.post("/v1/os/boot", headers={"Authorization": "Bearer weak"}).status_code == 503
    assert calls["writes"] == 0


def test_missing_token(api):
    client, calls, env = api
    env.setenv("JARVISX_OS_CONTROL_TOKEN", TOKEN)
    response = client.post("/v1/os/boot")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert calls["writes"] == 0


def test_invalid_token(api):
    client, calls, env = api
    env.setenv("JARVISX_OS_CONTROL_TOKEN", TOKEN)
    assert client.post("/v1/os/boot", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert calls["writes"] == 0


def test_authorized_post(api):
    client, calls, env = api
    env.setenv("JARVISX_OS_CONTROL_TOKEN", TOKEN)
    response = client.post("/v1/os/boot", headers={"Authorization": "Bearer " + TOKEN})
    assert response.status_code == 200
    assert response.json()["state"] == "booted"
    assert calls["writes"] == 1


def test_authorized_sensitive_read(api):
    client, calls, env = api
    env.setenv("JARVISX_OS_CONTROL_TOKEN", TOKEN)
    assert client.get("/v1/os/export").status_code == 401
    assert calls["exports"] == 0
    assert client.get("/v1/os/export", headers={"Authorization": "Bearer " + TOKEN}).status_code == 200
    assert calls["exports"] == 1


def test_health_and_status_remain_readable(api):
    client, _, _ = api
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/os/status").status_code == 200


def test_every_write_method_fails_closed(api):
    client, calls, _ = api
    assert client.delete("/other/path").status_code == 503
    assert calls["writes"] == 0


def test_auth_check_is_exact():
    assert authorise_bearer("Bearer " + TOKEN, TOKEN) == 200
    assert authorise_bearer("Bearer " + TOKEN + "X", TOKEN) == 401
    assert authorise_bearer("bearer " + TOKEN, TOKEN) == 401
    assert authorise_bearer("Bearer " + "x" * 1025, TOKEN) == 401


def test_installed_api_rejects_mutation_before_handler(monkeypatch):
    """Exercise the real production FastAPI app, not only a synthetic fixture."""
    from jarvisx.dr_moagi_os_api import app

    monkeypatch.delenv("JARVISX_OS_CONTROL_TOKEN", raising=False)
    with TestClient(app) as client:
        response = client.post("/v1/os/boot")
        assert response.status_code == 503
        assert client.get("/healthz").status_code == 200
