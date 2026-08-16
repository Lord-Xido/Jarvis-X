from pathlib import Path


APP = Path(__file__).resolve().parents[1] / "apps" / "qsol-kinetic-3d" / "index.html"
README = APP.parent / "README.md"


def test_qsol_kinetic_3d_surface_is_self_contained_and_bounded() -> None:
    html = APP.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    required_markers = (
        "QSOL-ASM :: KINETIC PROCESSOR 3D",
        "phase-space VM",
        "PROJ_LAMBDA",
        "Pi_Lambda",
        "Omega",
        "canvas id=\"scene\"",
        "drawOmega" if "drawOmega" in html else "torus()",
        "[-10,10]^3",
    )
    for marker in required_markers:
        assert marker in html

    assert "https://" not in html
    assert "http://" not in html
    assert "jarvisx.system_runtime" in readme
    assert "does not mutate authoritative Jarvis-X state" in readme
