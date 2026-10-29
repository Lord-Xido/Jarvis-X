"""Fail-closed HTTP access control for the Dr Moagi operating control plane.

A bearer token is required for every state-changing HTTP method and for raw
state export/snapshot endpoints. Reading health and aggregate telemetry remains
unauthenticated, so deployments must additionally restrict network exposure.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Request
from starlette.responses import JSONResponse

CONTROL_TOKEN_ENV = "JARVISX_OS_CONTROL_TOKEN"
# Future state-read endpoints are protected unless explicitly reviewed as public.
PUBLIC_READ_PATHS = frozenset({
    "/v1/os/capabilities", "/v1/os/status", "/v1/os/meta/status",
})
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def requires_control_auth(method: str, path: str) -> bool:
    """Protect writes irrespective of route name; protect explicit data exports."""
    return method.upper() in WRITE_METHODS or (
        path.startswith("/v1/os/") and path not in PUBLIC_READ_PATHS
    )


def authorise_bearer(authorization: str | None, configured_secret: str | None) -> int:
    """Return 200 on success, 401 on bad credentials, or 503 if disabled.

    The mandatory minimum length prevents deploying with a low-entropy example
    token. The operator must supply a random secret; length alone is insufficient.
    """
    if configured_secret is None or len(configured_secret) < 32 or not configured_secret.isascii():
        return 503
    if not authorization or not authorization.startswith("Bearer "):
        return 401
    supplied = authorization[len("Bearer "):]
    if not supplied or len(supplied) > 1024 or not supplied.isascii():
        return 401
    return 200 if hmac.compare_digest(supplied, configured_secret) else 401


async def control_access_middleware(request: Request, call_next):
    """FastAPI HTTP middleware, applied before handler side effects."""
    if not requires_control_auth(request.method, request.url.path):
        return await call_next(request)
    result = authorise_bearer(
        request.headers.get("authorization"), os.environ.get(CONTROL_TOKEN_ENV)
    )
    if result == 503:
        return JSONResponse(
            status_code=503,
            content={"detail": "Control mutations disabled until operator token is configured"},
            headers={"Cache-Control": "no-store"},
        )
    if result == 401:
        return JSONResponse(
            status_code=401,
            content={"detail": "Control token required"},
            headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
        )
    return await call_next(request)
