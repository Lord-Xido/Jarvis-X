from __future__ import annotations

from .api import app, start_api


def start_web(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Compatibility entrypoint for the unified FastAPI dashboard/API service."""

    start_api(host=host, port=port)


__all__ = ["app", "start_web"]
