from __future__ import annotations

import pytest

from jarvisx.dm_vomegaxi_fixed_point import DMvOmegaXiFixedPointConfig
from jarvisx.dr_moagi_autoexec import AutoExecPolicy
from jarvisx.reality_reckoning import (
    RealityConstrainedDMvOmegaXiEngine,
    SparseRealityAnchor,
)


def _config(**overrides):
    values = dict(
        side=16,
        max_active_cells=64,
        value_min=-2.0,
        value_max=2.0,
        policy=AutoExecPolicy(block_size=2, quantization=0.01, prune_epsilon=0.0),
        latent_bound=1.0,
        omega_memory=0.5,
        theta_gain=0.5,
        theta_max_delta=0.25,
        fixed_point_tolerance=1.0e-6,
        semantic_floor=1.0e-12,
        max_iterations=128,
    )
    values.update(overrides)
    return DMvOmegaXiFixedPointConfig(**values)


def test_glyph_converges_only_when_internal_and_external_conditions_hold():
    field = {(2, 2, 2): 0.75, (3, 2, 2): 0.75}
    anchor = SparseRealityAnchor(field, tolerance=1.0e-12, correction_gain=0.5)
    engine = RealityConstrainedDMvOmegaXiEngine(anchor, _config())
    engine.load(field)

    report = engine.step()

    assert report.committed
    assert report.internal_converged
    assert report.external_correspondence
    assert report.converged
    assert report.external_residual_after == pytest.approx(0.0)
    assert not report.reality_corrected
    assert engine.status()["glyph"] == "⟲⊙"


def test_internal_fixed_point_alone_is_not_reported_as_converged():
    initial = {(2, 2, 2): 0.75, (3, 2, 2): 0.75}
    world = {(2, 2, 2): 0.25, (3, 2, 2): 0.25}
    anchor = SparseRealityAnchor(world, tolerance=1.0e-4, correction_gain=0.5)
    engine = RealityConstrainedDMvOmegaXiEngine(anchor, _config(fixed_point_tolerance=1.0e-4))
    engine.load(initial)

    first = engine.step()

    assert first.committed
    assert first.reality_corrected
    assert first.external_residual_after < first.external_residual_before
    assert not first.converged

    reports = engine.run_until_fixed_point()
    assert reports[-1].converged
    assert reports[-1].internal_converged
    assert reports[-1].external_correspondence
    assert reports[-1].external_residual_after <= anchor.tolerance


def test_reality_correction_can_add_observed_support():
    initial = {(2, 2, 2): 0.5}
    world = {(2, 2, 2): 0.5, (4, 4, 4): 1.0}
    anchor = SparseRealityAnchor(world, tolerance=1.0e-4, correction_gain=1.0)
    engine = RealityConstrainedDMvOmegaXiEngine(anchor, _config())
    engine.load(initial)

    report = engine.step()

    assert report.committed
    assert (4, 4, 4) in engine.snapshot()
    assert engine.snapshot()[(4, 4, 4)] == pytest.approx(1.0)
    assert report.external_residual_after < report.external_residual_before


def test_theta_gate_still_has_authority_over_reality_corrected_candidate():
    initial = {(2, 2, 2): 1.0}
    anchor = SparseRealityAnchor({(2, 2, 2): 0.0}, correction_gain=0.5)
    engine = RealityConstrainedDMvOmegaXiEngine(
        anchor,
        _config(),
        theta_gate=lambda _: False,
    )
    engine.load(initial)

    report = engine.step()

    assert not report.committed
    assert not report.theta_gate_passed
    assert report.rejection_reason == "Theta policy gate rejected candidate"
    assert engine.snapshot() == initial
    assert engine.journal.verify()


def test_anchor_validation_rejects_nonfinite_world_values():
    with pytest.raises(ValueError, match="world values must be finite"):
        SparseRealityAnchor({(0, 0, 0): float("nan")})


def test_status_exposes_permeation_contract():
    field = {(1, 1, 1): 0.5}
    engine = RealityConstrainedDMvOmegaXiEngine(
        SparseRealityAnchor(field, tolerance=1.0e-8),
        _config(),
    )
    engine.load(field)
    engine.step()

    status = engine.status()

    assert status["operator"] == "Reality-Constrained Recursive Reckoning"
    assert status["reckoning_cycle"] == [
        "Generate",
        "Contrast",
        "Reckon",
        "Verify",
        "Correct",
        "Recur",
    ]
    assert status["permeation_stack"] == [
        "Psi",
        "Phi",
        "Lambda^-1",
        "Omega",
        "Theta",
        "X_world",
    ]
    assert status["external_correspondence"]
