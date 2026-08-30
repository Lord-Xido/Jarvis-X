from __future__ import annotations

from pathlib import Path

import pytest

from jarvisx.dr_moagi_animation_loop import canonical_animation_loop_program
from jarvisx.dr_moagi_animation_pdf import (
    build_animation_pdf_package,
    load_animation_pdf_package,
    run_animation_pdf_package,
)
from jarvisx.dr_moagi_pdf_bytecode import make_seed_volume


def test_animation_pdf_round_trip_and_execution(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    program = canonical_animation_loop_program(cycles=2, refinement_passes=2)
    path = tmp_path / "dr-moagi-animation.dm3d.pdf"

    written_manifest = build_animation_pdf_package(path, program)
    read_manifest, recovered = load_animation_pdf_package(path)

    assert recovered == program
    assert read_manifest == written_manifest
    assert read_manifest.cycles == 2
    assert read_manifest.inner_instruction_count == 9

    result = run_animation_pdf_package(path, make_seed_volume(8))
    assert result.cycles_executed == 2
    assert len(result.frames) == 10
    assert all(cycle.fixed_point_pass for cycle in result.cycles)
