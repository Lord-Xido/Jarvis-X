"""Minimal Jarvis-X HyperCloud client using only the Python standard library."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any
from urllib.request import Request, urlopen


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_job(base_url: str, job_id: str, api_key: str | None, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = request_json("GET", f"{base_url}/v1/jobs/{job_id}", api_key=api_key)
        if job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(0.25)
    raise TimeoutError(f"job {job_id} did not complete within {timeout} seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a HyperCloud chat job")
    parser.add_argument("prompt")
    parser.add_argument("--namespace", default="demo")
    parser.add_argument("--base-url", default=os.getenv("JARVISX_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--api-key", default=os.getenv("JARVISX_API_KEY"))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    queued = request_json(
        "POST",
        f"{args.base_url.rstrip('/')}/v1/chat",
        {"namespace": args.namespace, "prompt": args.prompt},
        args.api_key,
    )
    completed = wait_for_job(
        args.base_url.rstrip("/"),
        str(queued["id"]),
        args.api_key,
        args.timeout,
    )
    print(json.dumps(completed, indent=2, sort_keys=True))
    return 0 if completed["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
