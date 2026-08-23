from __future__ import annotations

from dataclasses import replace

import pytest

from jarvisx.dr_moagi_meta_optimizer import (
    DrMoagi3DMetaOptimizer,
    MetaSearchConfig,
    MetaVector3D,
    SelfOptimizing3DSystem,
)
from jarvisx.dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field


def _search() -> MetaSearchConfig:
    return MetaSearchConfig(
        max_candidates=3,
        probe_cycles=1,
        confirm_cycles=1,
        max_eval_cells=64,
        survivors=1,
        min_relative_improvement=0.0,
    )


def test_meta_vector_is_a_bounded_3d_lattice_coordinate():
    assert MetaVector3D(-1, 0, 1).manhattan == 2
    with pytest.raises(ValueError):
        MetaVector3D(2, 0, 0)


def test_candidate_config_moves_all_three_operational_axes_without_expanding_state_dir():
    base = DrMoagiOSConfig(
        side=16,
        max_active_cells=256,
        block_size=2,
        quantization=0.01,
        prune_epsilon=0.01,
        deep_distiller_max_latent_cells=128,
        deep_distiller_learning_rate=0.05,
        contraction=0.08,
        attenuation=0.10,
        fixed_point_passes=1,
        state_dir=None,
    )
    optimizer = DrMoagi3DMetaOptimizer(_search())
    candidate = optimizer.candidate_config(base, MetaVector3D(1, 1, 1))

    assert candidate.block_size == 4
    assert candidate.quantization > base.quantization
    assert candidate.prune_epsilon > base.prune_epsilon
    assert candidate.deep_distiller_learning_rate > base.deep_distiller_learning_rate
    assert candidate.deep_distiller_max_latent_cells <= candidate.max_active_cells
    assert candidate.contraction > base.contraction
    assert candidate.attenuation > base.attenuation
    assert candidate.fixed_point_passes >= base.fixed_point_passes
    assert candidate.state_dir is None
    assert candidate.auto_optimize is False


def test_optimizer_replays_candidates_and_refuses_unverified_sota_claim():
    config = DrMoagiOSConfig(
        side=16,
        max_active_cells=512,
        fixed_point_passes=1,
        state_dir=None,
    )
    optimizer = DrMoagi3DMetaOptimizer(_search())
    report = optimizer.optimize(demo_field(16), config)

    assert report.evaluated_candidates == 5
    assert report.baseline.metrics.workloads == 3
    assert report.best.metrics.workloads == 3
    assert report.best.metrics.score >= 0.0
    assert report.claim_status == "unverified_against_external_sota"
    if report.promoted:
        assert report.promoted_config is not None
        assert report.relative_improvement >= 0.0


def test_self_optimizing_wrapper_preserves_authoritative_state_across_meta_epoch(tmp_path):
    config = DrMoagiOSConfig(
        side=16,
        max_active_cells=512,
        fixed_point_passes=1,
        state_dir=tmp_path,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    kernel.load(demo_field(16))
    kernel.step()
    before = kernel.status()
    before_hash = before["state_hash"]
    before_iteration = before["distiller"]["iteration"]
    before_cycle = kernel.cycle

    system = SelfOptimizing3DSystem(kernel, search=_search())
    report = system.turn_inward()
    after = system.status()

    assert after["state_hash"] == before_hash
    assert after["cycle"] == before_cycle
    assert after["distiller"]["iteration"] == before_iteration
    assert after["meta_optimizer"]["epoch"] == 1
    assert after["meta_optimizer"]["journal_valid"] is True
    assert after["meta_optimizer"]["external_sota_verified"] is False
    assert system.meta_journal.verify()
    if report.promoted:
        assert report.promoted_config is not None
        assert system.kernel.config.state_dir == tmp_path


def test_meta_optimizer_is_bounded_by_evaluation_sample():
    config = DrMoagiOSConfig(side=64, max_active_cells=1_000, state_dir=None)
    source = {
        (index % 64, (index // 64) % 64, (index // (64 * 64)) % 64): 0.75
        for index in range(300)
    }
    report = DrMoagi3DMetaOptimizer(
        replace(_search(), max_eval_cells=32, max_candidates=1, survivors=1)
    ).optimize(source, config)

    assert report.evaluated_candidates == 3
    assert report.baseline.metrics.workloads == 3
