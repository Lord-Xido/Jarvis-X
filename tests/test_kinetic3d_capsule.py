from __future__ import annotations

import pytest

from jarvisx.kinetic3d_backend import ReferenceBackend
from jarvisx.kinetic3d_capsule import (
    CapsuleError,
    build_capsule,
    decode_capsule,
    parse_capsule,
    plan_rate_distortion,
)


def _capsule_for(
    current: list[float],
    prediction: list[float],
    shape: tuple[int, int, int],
    *,
    active_threshold: float = 0.0,
    coarse_factor: int = 2,
    refine_threshold: float = 0.0,
    tolerance: float = 0.0,
) -> bytes:
    step = ReferenceBackend().step(
        current,
        prediction,
        shape,
        active_threshold=active_threshold,
        coarse_factor=coarse_factor,
        refine_threshold=refine_threshold,
    )
    return build_capsule(
        shape=shape,
        prediction=prediction,
        active_threshold=active_threshold,
        coarse_factor=coarse_factor,
        refine_threshold=refine_threshold,
        tolerance=tolerance,
        active_indices=step.active_indices,
        coarse_values=step.coarse_values,
        fine_corrections=step.fine_corrections,
    )


def test_jxk2_capsule_is_self_describing_reversible_and_integrity_sealed() -> None:
    shape = (8, 8, 8)
    prediction = [0.0] * 512
    current = [5.0] * 512
    capsule_bytes = _capsule_for(current, prediction, shape, coarse_factor=8)

    capsule = parse_capsule(capsule_bytes)
    assert capsule.shape == shape
    assert capsule.coarse_factor == 8
    assert len(capsule.active_indices) == 512
    assert len(capsule.coarse_values) == 1
    assert len(capsule.fine_corrections) == 0
    assert capsule.decode(prediction) == tuple(current)
    assert len(capsule_bytes) < len(current) * 8

    corrupted = bytearray(capsule_bytes)
    corrupted[-33] ^= 0x01
    with pytest.raises(CapsuleError, match="checksum"):
        parse_capsule(bytes(corrupted))


def test_capsule_binds_delta_to_exact_predictor() -> None:
    shape = (2, 2, 2)
    prediction = [1.0] * 8
    current = [1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 1.0]
    capsule_bytes = _capsule_for(current, prediction, shape, coarse_factor=2)

    assert decode_capsule(capsule_bytes, prediction) == tuple(current)
    with pytest.raises(CapsuleError, match="prediction checksum"):
        decode_capsule(capsule_bytes, [0.0] * 8)


def test_irregular_volume_uses_fine_corrections_and_round_trips_exactly() -> None:
    shape = (2, 2, 2)
    prediction = [0.0] * 8
    current = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    capsule_bytes = _capsule_for(current, prediction, shape, coarse_factor=2)
    capsule = parse_capsule(capsule_bytes)

    assert capsule.decode(prediction) == tuple(current)
    assert len(capsule.coarse_values) == 1
    assert len(capsule.fine_corrections) > 0


def test_rate_distortion_planner_prefers_large_uniform_block_for_exact_change() -> None:
    shape = (8, 8, 8)
    prediction = [0.0] * 512
    current = [2.0] * 512

    plan = plan_rate_distortion(current, prediction, shape, tolerance=0.0)

    assert plan.selected.coarse_factor == 8
    assert plan.selected.active_cells == 512
    assert plan.selected.coarse_values == 1
    assert plan.selected.fine_corrections == 0
    assert plan.selected.max_abs_error == 0.0
    assert plan.selected.wire_compression_ratio > 1.0


def test_rate_distortion_planner_is_deterministic_and_respects_error_budget() -> None:
    shape = (4, 4, 4)
    prediction = [0.0] * 64
    current = [float(index % 7) / 10.0 for index in range(64)]

    first = plan_rate_distortion(current, prediction, shape, tolerance=0.2)
    second = plan_rate_distortion(current, prediction, shape, tolerance=0.2)

    assert first == second
    assert first.selected.max_abs_error <= 0.2
    assert 1 <= len(first.candidates) <= 64
    assert first.selected.capsule_bytes == min(candidate.capsule_bytes for candidate in first.candidates)


def test_capsule_validation_fails_closed() -> None:
    with pytest.raises(CapsuleError, match="truncated"):
        parse_capsule(b"JXK2")

    shape = (1, 1, 1)
    prediction = [0.0]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_capsule(
            shape=shape,
            prediction=prediction,
            active_threshold=0.0,
            coarse_factor=1,
            refine_threshold=0.0,
            tolerance=0.0,
            active_indices=[0, 0],
            coarse_values=[((0, 0, 0), 1.0)],
            fine_corrections=[],
        )
