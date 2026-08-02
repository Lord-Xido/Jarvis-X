"""Network smoke test for a running Hyperion service."""

from __future__ import annotations

import json
import os
import urllib.request

BASE_URL = os.getenv("HYPERION_SMOKE_URL", "http://127.0.0.1:8080")
API_KEY = os.getenv("HYPERION_API_KEY", "ci-secret")


def request(path: str, *, method: str = "GET", payload: object | None = None) -> object:
    data = None
    headers = {"X-Hyperion-Key": API_KEY}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    message = urllib.request.Request(
        BASE_URL + path,
        data=data,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(message, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


health = request("/healthz")
if not isinstance(health, dict) or health.get("status") != "ok":
    raise SystemExit("health check failed")

created = request(
    "/v1/hyperion/audits",
    method="POST",
    payload={
        "request_id": "ci-smoke",
        "observations": [
            {
                "source": "csv",
                "timestamp_ms": 0,
                "value": 100.0,
                "quantity": "amount",
                "unit": "ZAR",
                "correlation_id": "ci-0",
                "label": "known",
            },
            {
                "source": "cpu",
                "timestamp_ms": 1,
                "value": 100.0,
                "quantity": "amount",
                "unit": "ZAR",
                "correlation_id": "ci-0",
                "label": "known",
            },
        ],
    },
)
if not isinstance(created, dict) or not created.get("verified"):
    raise SystemExit("audit creation failed")
report_digest = created.get("report_digest")
if not isinstance(report_digest, str):
    raise SystemExit("report digest missing")

verified = request(f"/v1/hyperion/audits/{report_digest}/verify", method="POST")
if not isinstance(verified, dict) or not verified.get("verified"):
    raise SystemExit("report replay failed")

print(json.dumps({"health": True, "report_digest": report_digest, "verified": True}))
