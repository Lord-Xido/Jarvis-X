from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "dr-moagi-ann-cognitive-matrix"
INDEX = APP / "index.html"
ENGINE = APP / "engine.mjs"
README = APP / "README.md"
ADR = ROOT / "docs" / "adr" / "0029-dr-moagi-ann-cognitive-matrix.md"


def test_ann_cognitive_matrix_surface_is_bounded_and_repo_native() -> None:
    html = INDEX.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    adr = ADR.read_text(encoding="utf-8")

    for marker in (
        "Dr Moagi ANN Cognitive Matrix",
        "1,200,000 logical nodes",
        "32-bit VM",
        "sparse 1 GiB virtual ring",
        "fixed-point + CTR",
        'script type="module" src="./app.mjs"',
    ):
        assert marker in html

    for marker in (
        "LOGICAL_NODE_COUNT = 1_200_000",
        "VIRTUAL_BYTES = 1024 * 1024 * 1024",
        "LOAD_BITSTREAM",
        "AUTO_ENCODE",
        "DECODE_SPATIAL",
        "TENSOR_MAP",
        "INWARD_FOLD",
        "EMIT_STREAM",
        "HALT_SYNC",
        "resident page budget exceeded",
        "fixedPointResidual",
        "ctrEnergy",
        "COMMIT",
    ):
        assert marker in engine

    assert "https://" not in html
    assert "http://" not in html
    assert "1.2 million independent MLPs" in readme
    assert "does not invent FPS, Gbps, MIPS" in readme
    assert "ADR-026" in readme

    for marker in (
        "ADR-029",
        "1,200,000",
        "virtual 1 GiB",
        "CTR",
        "logical capacity",
        "measured performance",
    ):
        assert marker in adr
