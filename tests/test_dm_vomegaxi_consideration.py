from __future__ import annotations

import pytest

from jarvisx.dm_vomegaxi_consideration import (
    DMvOmegaXiConsiderationConfig,
    DMvOmegaXiConsiderationLoop,
)
from jarvisx.dr_moagi_autoexec import AutoExecPolicy


def _config(**overrides):
    values = dict(
        side=32,
        max_active_cells=256,
        value_min=-2.0,
        value_max=2.0,
        policy=AutoExecPolicy(block_size=2, quantization=0.001, prune_epsilon=0.0),
        attention_keep_ratio=1.0,
        attention_min_cells=1,
        latent_bound=2.0,
        memory_retention=0.5,
        memory_constraint_gain=0.0,
        theta_constraint=1.0,
        state_gain=0.5,
        max_state_delta=0.5,
        semantic_hbar=1.0,
        dissipation_rate=0.1,
        fixed_point_tolerance=1.0e-6,
        equilibrium_tolerance=1.0e-8,
        max_iterations=256,
        logical_side=1_000_000,
    )
    values.update(overrides)
    return DMvOmegaXiConsiderationConfig(**values)


def test_attention_contracts_high_entropy_sparse_support():
    engine = DMvOmegaXiConsiderationLoop(_config(attention_keep_ratio=0.5))
    field = {(i, 0, 0): 1.0 for i in range(8)}
    engine.load(field)

    report = engine.step()

    assert report.active_cells_before_attention == 8
    assert report.active_cells_after_attention == 4
    assert report.attended_entropy < report.input_entropy
    assert report.entropy_contraction > 0.0


def test_gamma_term_is_executable_as_real_residual_damping():
    engine = DMvOmegaXiConsiderationLoop(
        _config(state_gain=0.25, dissipation_rate=0.5, semantic_hbar=1.0)
    )
    engine.load({(2, 2, 2): 1.0, (3, 2, 2): 0.5})

    report = engine.step()

    assert report.dissipated_rms > 0.0
    assert report.description_rms >= 0.0
    assert report.fixed_point_residual >= 0.0


def test_consideration_loop_converges_to_internal_fixed_point():
    engine = DMvOmegaXiConsiderationLoop(_config())
    engine.load({(2, 2, 2): 1.0, (3, 2, 2): 0.5})

    reports = engine.run_until_fixed_point()

    assert reports[-1].converged
    assert reports[-1].fixed_point_residual <= engine.config.fixed_point_tolerance
    assert reports[-1].h_mmm_delta <= engine.config.equilibrium_tolerance
    state = engine.snapshot()
    assert state[(2, 2, 2)] == pytest.approx(0.75, abs=2.0e-4)
    assert state[(3, 2, 2)] == pytest.approx(0.75, abs=2.0e-4)


def test_memory_constraint_uses_bounded_spatial_divergence():
    engine = DMvOmegaXiConsiderationLoop(
        _config(memory_constraint_gain=0.5, theta_constraint=2.0)
    )
    engine.load({(2, 2, 2): 1.0, (3, 2, 2): -1.0})

    correction = engine.memory_constraint(set(engine.snapshot()))

    assert correction[(2, 2, 2)] < 0.0
    assert correction[(3, 2, 2)] > 0.0
    assert all(abs(value) <= 0.25 for value in correction.values())


def test_million_cubed_domain_is_logical_not_dense_materialization():
    engine = DMvOmegaXiConsiderationLoop(
        _config(side=1_000_000, logical_side=1_000_000)
    )
    engine.load({(1, 2, 3): 0.8, (999_999, 999_999, 999_999): 0.6})

    status = engine.status()

    assert status["logical_domain"] == "1000000^3"
    assert status["logical_voxels"] == str(10**18)
    assert status["materialization"] == "sparse-active-support-only"
    assert status["active_cells"] == 2


def test_consideration_journal_is_hash_chained(tmp_path):
    path = tmp_path / "dm-vomegaxi-consideration.jsonl"
    engine = DMvOmegaXiConsiderationLoop(_config(), journal_path=path)
    engine.load({(2, 2, 2): 1.0, (3, 2, 2): 0.5})

    engine.run_until_fixed_point()

    assert path.exists()
    assert engine.journal.verify()
    assert len(engine.journal.head) == 64
