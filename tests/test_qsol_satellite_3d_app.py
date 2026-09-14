from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "apps" / "qsol-satellite-3d" / "index.html"
README = APP.parent / "README.md"


def test_qsol_satellite_3d_surface_is_self_contained_and_bounded() -> None:
    html = APP.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    required_markers = (
        "QSOL :: SATELLITE SIGNAL TRACKER 3D",
        "Xi(tau, f_D, theta, t)",
        "DELAY_DOPPLER_MATCH",
        "CLOUD_FUSE",
        "POSTERIOR_CORRECT",
        "canvas id=\"scene\"",
        "S_(t+1) = T_cloud",
        "synthetic complex-baseband observations",
    )
    for marker in required_markers:
        assert marker in html

    assert "https://" not in html
    assert "http://" not in html
    assert "jarvisx.system_runtime" in readme
    assert "does not mutate authoritative Jarvis-X state" in readme
    for capability in ("network", "SDR", "radio", "satellite-control"):
        assert capability in readme
    assert "authorized signals" in readme
