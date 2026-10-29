"""Fail-closed control-plane tests without optional HTTP client dependencies."""
from __future__ import annotations

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse

from jarvisx.dr_moagi_os_security import (
    authorise_bearer,
    control_access_middleware,
    requires_control_auth,
)

TOKEN = "test-only-control-token-of-at-least-32-chars"


def call_gate(monkeypatch, method, path, *, authorization=None, configured=None):
    """Exercise the real middleware on a Starlette request and tracked handler."""
    if configured is None:
        monkeypatch.delenv("JARVISX_OS_CONTROL_TOKEN", raising=False)
    else:
        monkeypatch.setenv("JARVISX_OS_CONTROL_TOKEN", configured)
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("ascii")))
    request = Request({
        "type": "http", "method": method, "scheme": "http",
        "path": path, "root_path": "", "query_string": b"", "headers": headers,
        "server": ("localhost", 80), "client": ("127.0.0.1", 1234),
    })
    calls = []

    async def handler(_request):
        calls.append(method + " " + path)
        return JSONResponse({"accepted": True})

    response = asyncio.run(control_access_middleware(request, handler))
    return response, calls


def test_path_policy():
    assert requires_control_auth("POST", "/v1/os/boot")
    assert requires_control_auth("DELETE", "/other/path")
    assert requires_control_auth("GET", "/v1/os/export")
    assert requires_control_auth("GET", "/v1/os/snapshot")
    assert requires_control_auth("GET", "/v1/os/bitplane")
    assert requires_control_auth("GET", "/v1/os/meta/lattice")
    assert requires_control_auth("GET", "/v1/os/new-state-endpoint")
    assert not requires_control_auth("GET", "/healthz")
    assert not requires_control_auth("GET", "/v1/os/status")
    assert not requires_control_auth("GET", "/v1/os/capabilities")
    assert not requires_control_auth("GET", "/v1/os/meta/status")


def test_fail_closed_when_unconfigured(monkeypatch):
    response, calls = call_gate(monkeypatch, "POST", "/v1/os/boot")
    assert response.status_code == 503
    assert not calls


def test_fail_closed_for_short_secret(monkeypatch):
    response, calls = call_gate(monkeypatch, "POST", "/v1/os/boot",
                                configured="weak", authorization="Bearer weak")
    assert response.status_code == 503
    assert not calls


def test_missing_token_rejected(monkeypatch):
    response, calls = call_gate(monkeypatch, "POST", "/v1/os/boot", configured=TOKEN)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert not calls


def test_invalid_token_rejected(monkeypatch):
    response, calls = call_gate(monkeypatch, "POST", "/v1/os/boot",
                                configured=TOKEN, authorization="Bearer wrong")
    assert response.status_code == 401
    assert not calls


def test_correct_token_allows_mutation(monkeypatch):
    response, calls = call_gate(monkeypatch, "POST", "/v1/os/boot",
                                configured=TOKEN, authorization="Bearer " + TOKEN)
    assert response.status_code == 200
    assert calls == ["POST /v1/os/boot"]


def test_sensitive_read_requires_auth(monkeypatch):
    denied, denied_calls = call_gate(monkeypatch, "GET", "/v1/os/export", configured=TOKEN)
    allowed, allowed_calls = call_gate(monkeypatch, "GET", "/v1/os/export",
                                      configured=TOKEN, authorization="Bearer " + TOKEN)
    assert denied.status_code == 401 and not denied_calls
    assert allowed.status_code == 200 and allowed_calls


def test_health_and_status_remain_readable(monkeypatch):
    for path in ("/healthz", "/v1/os/status"):
        response, calls = call_gate(monkeypatch, "GET", path)
        assert response.status_code == 200
        assert calls


def test_every_write_method_fails_closed(monkeypatch):
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        response, calls = call_gate(monkeypatch, method, "/new/handler")
        assert response.status_code == 503
        assert not calls


def test_bearer_exactness():
    assert authorise_bearer("Bearer " + TOKEN, TOKEN) == 200
    assert authorise_bearer("Bearer " + TOKEN + "X", TOKEN) == 401
    assert authorise_bearer("bearer " + TOKEN, TOKEN) == 401
    assert authorise_bearer("Bearer " + "x" * 1025, TOKEN) == 401


def test_production_app_mounts_access_gate():
    from jarvisx.dr_moagi_os_api import app

    assert any(
        getattr(layer, "kwargs", {}).get("dispatch") is control_access_middleware
        for layer in app.user_middleware
    )

def test_future_state_read_fails_closed(monkeypatch):
    response, calls = call_gate(monkeypatch, "GET", "/v1/os/new-state-endpoint", configured=TOKEN)
    assert response.status_code == 401
    assert not calls


def test_meta_lattice_is_protected(monkeypatch):
    response, calls = call_gate(monkeypatch, "GET", "/v1/os/meta/lattice", configured=TOKEN)
    assert response.status_code == 401
    assert not calls
