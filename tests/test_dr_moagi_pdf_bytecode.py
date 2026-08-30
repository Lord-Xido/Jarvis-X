from __future__ import annotations

from pathlib import Path

import pytest

from jarvisx.dr_moagi_pdf_bytecode import (
    ProgramLimits,
    build_pdf_package,
    canonical_autoencoder_program,
    execute_program,
    load_pdf_package,
    make_seed_volume,
    parse_program,
)


def test_program_round_trip_and_integrity_gate() -> None:
    program = canonical_autoencoder_program(pool=2, refinement_passes=4)
    instructions = parse_program(program)
    assert len(instructions) == 9

    tampered = bytearray(program)
    tampered[20] ^= 0x01
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_program(bytes(tampered))


def test_end_to_end_3d_cycle_is_bounded_and_cycle_consistent() -> None:
    program = canonical_autoencoder_program(pool=2, refinement_passes=4)
    result = execute_program(program, make_seed_volume(16))

    assert (result.volumes[0].nx, result.volumes[0].ny, result.volumes[0].nz) == (16, 16, 16)
    assert (result.volumes[2].nx, result.volumes[2].ny, result.volumes[2].nz) == (8, 8, 8)
    assert (result.volumes[3].nx, result.volumes[3].ny, result.volumes[3].nz) == (16, 16, 16)
    assert result.scalars[10] > 0.0
    assert result.scalars[11] < 1e-24
    assert result.scalars[12] > 0.0
    assert result.scalars[13] == 1.0
    assert result.metrics.refinement_updates == 4 * 8**3
    assert result.metrics.neighbor_reads == 10_752
    assert result.metrics.physical_steps > 0


def test_refinement_pass_limit_is_enforced_before_execution() -> None:
    program = canonical_autoencoder_program(pool=2, refinement_passes=4)
    with pytest.raises(ValueError, match="refinement pass count"):
        parse_program(program, ProgramLimits(max_refinement_passes=2))


def test_physical_work_budget_is_enforced() -> None:
    program = canonical_autoencoder_program(pool=2, refinement_passes=4)
    limits = ProgramLimits(max_physical_steps=1_000)
    with pytest.raises(RuntimeError, match="physical-step budget"):
        execute_program(program, make_seed_volume(16), limits)


def test_pdf_package_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("fitz")
    program = canonical_autoencoder_program(pool=2, refinement_passes=4)
    path = tmp_path / "dr-moagi.dm3d.pdf"
    written_manifest = build_pdf_package(path, program)
    read_manifest, recovered = load_pdf_package(path)

    assert recovered == program
    assert read_manifest == written_manifest
    result = execute_program(recovered, make_seed_volume(16))
    assert result.scalars[13] == 1.0
