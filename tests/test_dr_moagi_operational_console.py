from __future__ import annotations

from importlib import resources

from jarvisx.dr_moagi_operational_console import ConsoleHub, OperationalRuntime


def test_operational_runtime_exposes_exact_sparse_contract() -> None:
    runtime = OperationalRuntime(max_active_tiles=1000)
    state = runtime.snapshot()
    telemetry = state["telemetry"]

    assert telemetry["virtual_voxels"] == 1_000_000_000
    assert telemetry["active_tiles"] == 1_000
    assert telemetry["active_voxels"] == 1_000_000
    assert telemetry["work_reduction_factor"] == 1000.0
    assert telemetry["cycle"] >= 1
    assert telemetry["total_ms"] >= 0.0
    assert telemetry["residual_mse"] >= 0.0

    assert state["contract"]["authoritative_state"] == "python-engine"
    assert state["contract"]["visualization_role"] == "render-only"
    assert state["contract"]["error_determines_computation"] is True
    assert state["contract"]["wall_clock_1000x_guaranteed"] is False


def test_operational_runtime_tiles_are_real_3d_scheduler_state() -> None:
    runtime = OperationalRuntime(max_active_tiles=64)
    state = runtime.step(max_tiles=16)

    assert len(state["tiles"]) == 16
    assert len(state["control_words_sample"]) == 8
    for tile in state["tiles"]:
        assert 0 <= tile["x"] < 100
        assert 0 <= tile["y"] < 100
        assert 0 <= tile["z"] < 100
        assert tile["residual"] >= 0.0
        assert 0.0 <= tile["residual_norm"] <= 1.0


def test_console_cadence_is_bounded() -> None:
    hub = ConsoleHub(OperationalRuntime(max_active_tiles=8))
    hub.set_cadence(1)
    assert hub.cadence_ms == 16
    hub.set_cadence(50_000)
    assert hub.cadence_ms == 1000


def test_console_html_uses_live_websocket_without_random_telemetry() -> None:
    html = (
        resources.files("jarvisx")
        .joinpath("static/dr_moagi_operational_console.html")
        .read_text(encoding="utf-8")
    )

    assert "new WebSocket" in html
    assert "applyTelemetry" in html
    assert "residual_mse" in html
    assert "work_reduction_factor" in html
    assert "Math.random" not in html
    assert "COMPUTATION GENERATES THE ANIMATION" in html
