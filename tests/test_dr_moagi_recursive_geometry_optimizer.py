from __future__ import annotations

import math

import pytest

from jarvisx.dr_moagi_recursive_geometry_optimizer import (
    ExternalBaseline,
    Geometry3D,
    GeometryCodec,
    GeometryEvaluator,
    RecursiveGeometryOptimizer,
)


pytest.importorskip("numpy")


def test_geometry_codec_stays_inside_runtime_contract() -> None:
    low = GeometryCodec.decode(Geometry3D(-1.0, -1.0, -1.0))
    high = GeometryCodec.decode(Geometry3D(1.0, 1.0, 1.0))

    assert low.max_active_tiles == 64
    assert high.max_active_tiles == 1000
    assert low.latent_dim == 2
    assert high.latent_dim == 8
    assert 0.0 <= low.omega_decay < 1.0
    assert 0.0 <= high.omega_decay < 1.0
    assert 0.0 <= low.correction_gain <= 1.0
    assert 0.0 <= high.correction_gain <= 1.0
    assert low.seed != high.seed


def test_recursive_shell_contains_cartesian_and_non_cartesian_directions() -> None:
    shell = RecursiveGeometryOptimizer.shell(Geometry3D(), 0.5)

    assert len(shell) >= 26
    assert len({tuple(round(v, 8) for v in item.array()) for item in shell}) == len(shell)
    assert all(-1.0 <= value <= 1.0 for item in shell for value in item.array())


def test_external_sota_gate_requires_all_enabled_thresholds() -> None:
    loose = ExternalBaseline(
        name="matched-loose-baseline",
        max_residual_mse=1.0,
        max_median_latency_ms=10_000.0,
        max_active_voxels=2_000_000,
        minimum_work_reduction=1.0,
    )
    evaluator = GeometryEvaluator(
        suite={"text": b"recursive geometry benchmark" * 64},
        warmup_cycles=0,
        measured_cycles=1,
        external_baseline=loose,
    )
    result = evaluator.evaluate(Geometry3D())

    assert result.beyond_external_baseline is True

    strict = ExternalBaseline(
        name="unreachable-baseline",
        max_residual_mse=0.0,
        max_median_latency_ms=0.0,
    )
    strict_result = GeometryEvaluator(
        suite={"text": b"recursive geometry benchmark" * 64},
        warmup_cycles=0,
        measured_cycles=1,
        external_baseline=strict,
    ).evaluate(Geometry3D())
    assert strict_result.beyond_external_baseline is False


def test_failed_generation_rolls_back_incumbent() -> None:
    evaluator = GeometryEvaluator(
        suite={"text": b"rollback benchmark" * 64},
        warmup_cycles=0,
        measured_cycles=1,
    )
    optimizer = RecursiveGeometryOptimizer(evaluator, commit_margin=1.0)
    before = optimizer.incumbent.geometry

    event = optimizer.step()

    assert event.accepted is False
    assert optimizer.incumbent.geometry == before
    assert "rollback" in event.reason
    assert optimizer.radius < optimizer.initial_radius


def test_report_separates_recursive_search_from_sota_claim() -> None:
    evaluator = GeometryEvaluator(
        suite={"code": b"def f(x): return x + 1\n" * 64},
        warmup_cycles=0,
        measured_cycles=1,
    )
    optimizer = RecursiveGeometryOptimizer(evaluator, commit_margin=1.0)
    optimizer.step()
    report = optimizer.report()

    assert report["claims"]["recursive_self_optimization"] is True
    assert report["claims"]["source_self_modification"] is False
    assert report["claims"]["infinite_performance_claim"] is False
    assert report["claims"]["external_sota_baseline_supplied"] is False
    assert report["claims"]["external_sota_gate_passed"] is False
    assert math.isfinite(report["baseline"]["merit"])
