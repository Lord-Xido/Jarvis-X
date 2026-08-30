import pytest

from jarvisx.dr_moagi_animation_loop import (
    AutoLoopLimits,
    canonical_animation_loop_program,
    execute_auto_loop,
    parse_auto_loop_program,
)
from jarvisx.dr_moagi_pdf_bytecode import ProgramLimits, make_seed_volume


def test_canonical_animation_loop_closes_and_captures_frames() -> None:
    payload = canonical_animation_loop_program(cycles=3, refinement_passes=2)
    parsed = parse_auto_loop_program(payload)
    assert parsed.cycles == 3
    assert parsed.feedback_register == 3
    assert parsed.frame_registers == (0, 1, 2, 3, 4)

    result = execute_auto_loop(payload, make_seed_volume(8))
    assert result.cycles_executed == 3
    assert result.stopped_on_fixed_point is False
    assert len(result.frames) == 15
    assert result.final_volume.voxel_count == 8**3
    assert result.metrics.refinement_updates == 3 * 2 * 4**3
    assert all(item.fixed_point_pass for item in result.cycles)
    assert all(item.cycle_mse == pytest.approx(0.0, abs=1e-24) for item in result.cycles)


def test_auto_loop_stops_on_fixed_point_when_requested() -> None:
    payload = canonical_animation_loop_program(
        cycles=8,
        refinement_passes=2,
        stop_on_fixed_point=True,
    )
    result = execute_auto_loop(payload, make_seed_volume(8))
    assert result.cycles_executed == 1
    assert result.stopped_on_fixed_point is True
    assert len(result.frames) == 5


def test_auto_loop_rejects_digest_tampering() -> None:
    payload = bytearray(canonical_animation_loop_program(cycles=2))
    payload[-40] ^= 0x01
    with pytest.raises(ValueError, match="SHA-256"):
        parse_auto_loop_program(bytes(payload))


def test_auto_loop_enforces_cycle_limit() -> None:
    payload = canonical_animation_loop_program(cycles=4)
    with pytest.raises(ValueError, match="cycle count"):
        parse_auto_loop_program(payload, loop_limits=AutoLoopLimits(max_cycles=3))


def test_auto_loop_enforces_total_physical_work_budget() -> None:
    payload = canonical_animation_loop_program(cycles=3, refinement_passes=2)
    with pytest.raises(RuntimeError, match="physical-step budget"):
        execute_auto_loop(
            payload,
            make_seed_volume(8),
            loop_limits=AutoLoopLimits(max_total_physical_steps=10_000),
            vm_limits=ProgramLimits(max_physical_steps=100_000),
        )
