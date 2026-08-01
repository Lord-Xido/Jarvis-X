"""Packaged experimental 3D auto-poietic runtime for Jarvis-X.

The browser runtime is an observability and bounded-adaptation laboratory. It
is deliberately isolated from the authoritative CodexVM execution path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import Final

from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse

PROTOCOL: Final[str] = "jarvisx.autopoietic-web.v1"
DEFAULT_ROUTE: Final[str] = "/research/autopoietic-runtime"


@dataclass(frozen=True, slots=True)
class AutoPoieticWebManifest:
    """Stable host contract for the packaged browser runtime."""

    protocol: str = PROTOCOL
    maturity: str = "experimental"
    authoritative: bool = False
    deterministic_core: bool = False
    persistence: str = "browser-local-storage"
    bridge_object: str = "window.JarvisXAutopoieticRuntime"
    state_event: str = "jarvisx:autopoietic-state"
    safety_boundary: str = "bounded-parameter-mutation-only"

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


def runtime_asset() -> object:
    """Return the importlib resource handle for the standalone HTML runtime."""

    return files("jarvisx").joinpath("web/autopoietic_runtime.html")


def runtime_html() -> str:
    """Load the packaged single-file browser runtime as UTF-8 text."""

    asset = runtime_asset()
    if not asset.is_file():
        raise FileNotFoundError("packaged auto-poietic runtime asset is missing")
    return asset.read_text(encoding="utf-8")


def create_autopoietic_router(
    *,
    route: str = DEFAULT_ROUTE,
    include_manifest: bool = True,
) -> APIRouter:
    """Create an isolated FastAPI router for the experimental web runtime."""

    if not route.startswith("/"):
        raise ValueError("route must start with '/'")
    if route == "/":
        raise ValueError("the experimental runtime may not replace the application root")

    router = APIRouter(tags=["experimental-runtime"])

    @router.get(route, response_class=HTMLResponse, include_in_schema=False)
    def get_runtime() -> HTMLResponse:
        return HTMLResponse(runtime_html())

    if include_manifest:
        manifest_route = f"{route.rstrip('/')}/manifest"

        @router.get(manifest_route)
        def get_manifest() -> dict[str, str | bool]:
            return AutoPoieticWebManifest().to_dict()

    return router


def mount_autopoietic_runtime(
    app: FastAPI,
    *,
    route: str = DEFAULT_ROUTE,
    include_manifest: bool = True,
) -> None:
    """Mount the packaged runtime without changing canonical VM execution."""

    app.include_router(
        create_autopoietic_router(route=route, include_manifest=include_manifest)
    )
