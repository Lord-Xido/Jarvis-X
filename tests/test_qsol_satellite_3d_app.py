from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "apps" / "qsol-satellite-3d" / "index.html"
README = APP.parent / "README.md"
ADR = Path(__file__).resolve().parents[1] / "docs" / "adr" / "0019-qsol-satellite-signal-tracking-surface.md"


def test_qsol_satellite_3d_surface_is_self_contained_and_bounded() -> None:
    html = APP.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    adr = ADR.read_text(encoding="utf-8")

    required_markers = (
        "QSOL :: SATELLITE SIGNAL TRACKER 3D",
        "DR MOAGI ANN",
        "Xi(tau, f_D, theta, t)",
        "DELAY_DOPPLER_MATCH",
        "AUTO_ENCODE",
        "LATENT_PREDICT",
        "AUTO_DECODE",
        "RESIDUAL_COMPARE",
        "CLOUD_FUSE",
        "POSTERIOR_CORRECT",
        "ADAPT_WEIGHTS",
        "MEMORY_POLICY_UPDATE",
        "canvas id=\"scene\"",
        "S_(t+1) = M_Theta,Pi",
        "synthetic complex-baseband observations",
        "Adaptive ANN weights",
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
    assert "browser memory" in readme

    for architecture_marker in (
        "S_t = [X_t, Z_t, Xhat_t, E_t, Omega_t, Theta_t, Pi_t]",
        "Theta_(t+1) = clip",
        "Omega_(t+1)",
        "Authorized research only",
    ):
        assert architecture_marker in adr
