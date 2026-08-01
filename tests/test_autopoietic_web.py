from fastapi import FastAPI

from jarvisx.autopoietic_web import (
    AutoPoieticWebManifest,
    create_autopoietic_router,
    mount_autopoietic_runtime,
    runtime_html,
)


def test_runtime_asset_exposes_bounded_host_bridge() -> None:
    html = runtime_html()

    assert "window.JarvisXAutopoieticRuntime" in html
    assert "jarvisx:autopoietic-state" in html
    assert "COMMIT/ROLLBACK" in html
    assert "bounded-parameter-mutation-only" in html


def test_manifest_keeps_runtime_non_authoritative() -> None:
    manifest = AutoPoieticWebManifest().to_dict()

    assert manifest["maturity"] == "experimental"
    assert manifest["authoritative"] is False
    assert manifest["safety_boundary"] == "bounded-parameter-mutation-only"


def test_router_rejects_application_root() -> None:
    try:
        create_autopoietic_router(route="/")
    except ValueError as error:
        assert "may not replace" in str(error)
    else:
        raise AssertionError("root mounting should be rejected")


def test_runtime_mount_registers_runtime_and_manifest_routes() -> None:
    app = FastAPI()
    mount_autopoietic_runtime(app)
    paths = {route.path for route in app.routes}

    assert "/research/autopoietic-runtime" in paths
    assert "/research/autopoietic-runtime/manifest" in paths
