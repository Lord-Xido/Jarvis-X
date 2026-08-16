from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "apps" / "windows" / "dm-vomega-xi-6400-swarm"
SRC = APP / "dm_vomega_xi_6400gib_swarm_isa.c"
ISA = APP / "ISA.md"
README = APP / "README.md"


def test_canonical_machine_words_are_preserved() -> None:
    source = SRC.read_text(encoding="utf-8")
    words = (
        "0x50A12300U",
        "0x10FA0000U",
        "0x30CF0000U",
        "0x31710000U",
        "0x32810000U",
        "0x21778000U",
        "0x2177C000U",
        "0x21447000U",
        "0x21114000U",
        "0x40BF0000U",
        "0x11AB0000U",
        "0x0F000000U",
        "0xFF000000U",
    )
    for word in words:
        assert word in source


def test_fixed_width_and_sparse_address_invariants_are_documented() -> None:
    source = SRC.read_text(encoding="utf-8")
    isa = ISA.read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")

    assert "PROGRAM_ENCODE_DT001" in source
    assert "Q16_DT_001 655" in source
    assert "RESIDENT_PAGES 2048" in source
    assert "ACTIVE_AGENTS 64" in source
    assert "hash_rs3" in source
    assert "raw 32-bit resident-page handle" in isa
    assert "not authoritative Jarvis-X commits" in readme
    assert "jarvisx.system_runtime" in readme
