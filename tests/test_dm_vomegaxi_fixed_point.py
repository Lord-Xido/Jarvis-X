from __future__ import annotations

import pytest

from jarvisx.dm_vomegaxi_fixed_point import (
    DMvOmegaXiFixedPointConfig,
    DMvOmegaXiFixedPointEngine,
)
from jarvisx.dr_moagi_autoexec import AutoExecPolicy


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
        fixed_point_tolerance=1.0e-8,
        semantic_floor=1.0e-12,
        max_iterations=128,
    )
    values.update(overrides)
    return DMvOmegaXiFixedPointConfig(**values)


def test_exact_internal_fixed_point_keeps_positive_semantic_floor():
    engine = DMvOmegaXiFixedPointEngine(_config())
    field = {(2, 2, 2): 0.75, (3, 2, 2): 0.75}
    engine.load(field)

    report = engine.step()

    assert report.committed
    assert report.converged
    assert report.fixed_point_residual == pytest.approx(0.0)
    assert report.semantic_gap == pytest.approx(engine.config.semantic_floor)
    assert engine.snapshot() == field
    assert engine.status()["fixed_point_equation"] == "H* = F_DM(H*)"


def test_lossy_description_folds_inward_until_self_consistent():
    engine = DMvOmegaXiFixedPointEngine(_config(fixed_point_tolerance=1.0e-5))
    engine.load({(2, 2, 2): 1.0, (3, 2, 2): 0.5})

    reports = engine.run_until_fixed_point()

    assert reports[-1].converged
    assert reports[-1].fixed_point_residual <= engine.config.fixed_point_tolerance
    state = engine.snapshot()
    assert state[(2, 2, 2)] == pytest.approx(0.75, abs=1.0e-4)
    assert state[(3, 2, 2)] == pytest.approx(0.75, abs=1.0e-4)
    assert reports[-1].semantic_gap >= engine.config.semantic_floor


def test_lambda_inverse_bounds_latent_bottleneck():
    engine = DMvOmegaXiFixedPointEngine(_config(latent_bound=0.25))
    engine.load({(2, 2, 2): 1.5, (3, 2, 2): 1.0})

    bounded = engine.lambda_inverse(engine.phi(engine.snapshot()))

    assert bounded.cells
    assert all(abs(cell.value) <= 0.25 for cell in bounded.cells)


def test_logical_hypervolume_is_metadata_not_dense_materialization():
    engine = DMvOmegaXiFixedPointEngine(
        _config(side=1_000_000, logical_hypervolume_tb=10**27)
    )
    engine.load({(1, 2, 3): 0.8, (999_999, 999_999, 999_999): 0.6})

    status = engine.status()

    assert status["logical_hypervolume_tb"] == str(10**27)
    assert status["materialization"] == "sparse-active-support-only"
    assert status["active_cells"] == 2


def test_theta_policy_gate_can_reject_without_committing():
    engine = DMvOmegaXiFixedPointEngine(_config(), theta_gate=lambda _: False)
    initial = {(2, 2, 2): 1.0, (3, 2, 2): 0.5}
    engine.load(initial)

    report = engine.step()

    assert not report.committed
    assert not report.theta_gate_passed
    assert report.rejection_reason == "Theta policy gate rejected candidate"
    assert engine.snapshot() == initial
    assert engine.journal.verify()


def test_hash_chained_fixed_point_journal_round_trips(tmp_path):
    path = tmp_path / "dm-vomegaxi-fixed-point.jsonl"
    engine = DMvOmegaXiFixedPointEngine(_config(), journal_path=path)
    engine.load({(2, 2, 2): 1.0, (3, 2, 2): 0.5})

    engine.run_until_fixed_point()

    assert path.exists()
    assert engine.journal.verify()
    assert len(engine.journal.head) == 64
