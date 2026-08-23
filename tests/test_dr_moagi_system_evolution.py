from __future__ import annotations

from dataclasses import replace

import pytest

from jarvisx.dr_moagi_meta_optimizer import MetaSearchConfig, SelfOptimizing3DSystem
from jarvisx.dr_moagi_os import DrMoagiOSConfig, DrMoagiOSKernel, demo_field
from jarvisx.dr_moagi_system_evolution import (
    ArchitecturePolicy,
    ArchitectureVector3D,
    DrMoagiArchitectureOptimizer,
    REQUIRED_PIPELINE,
    SelfEvolving3DArchitecture,
)


def _search() -> MetaSearchConfig:
    return MetaSearchConfig(
        max_candidates=1,
        probe_cycles=1,
        confirm_cycles=1,
        max_eval_cells=16,
        survivors=1,
        min_relative_improvement=0.0,
    )


def _policy() -> ArchitecturePolicy:
    return ArchitecturePolicy(
        state_cycles_per_meta=1,
        meta_epochs_per_architecture_review=1,
        max_architecture_candidates=1,
        max_architecture_eval_cells=16,
        max_eval_state_cycles=1,
        min_architecture_improvement=0.0,
        meta_search=_search(),
    )


def _architecture(tmp_path=None) -> SelfEvolving3DArchitecture:
    config = DrMoagiOSConfig(
        side=8,
        max_active_cells=256,
        deep_distiller_max_latent_cells=128,
        fixed_point_passes=1,
        state_dir=tmp_path,
    )
    kernel = DrMoagiOSKernel(config)
    kernel.boot(restore=False)
    kernel.load(demo_field(8))
    system = SelfOptimizing3DSystem(kernel, search=_search())
    return SelfEvolving3DArchitecture(system, policy=_policy())


def test_architecture_vector_is_bounded_3d_coordinate():
    assert ArchitectureVector3D(-1, 0, 1).manhattan == 2
    with pytest.raises(ValueError):
        ArchitectureVector3D(0, 2, 0)


def test_architecture_policy_preserves_constitutional_pipeline():
    architecture = _architecture()
    capabilities = architecture.capabilities()

    assert tuple(capabilities["required_pipeline"]) == REQUIRED_PIPELINE
    assert capabilities["transactional_state_commit"] is True
    assert capabilities["transactional_configuration_promotion"] is True
    assert capabilities["transactional_architecture_promotion"] is True
    assert capabilities["self_rewriting_source"] is False


def test_candidate_policy_moves_cadence_search_and_resilience_axes():
    optimizer = DrMoagiArchitectureOptimizer()
    base = ArchitecturePolicy(meta_search=MetaSearchConfig(max_candidates=4, survivors=2))
    faster = optimizer.candidate_policy(base, ArchitectureVector3D(-1, 1, 1))

    assert faster.state_cycles_per_meta < base.state_cycles_per_meta
    assert faster.meta_search.max_candidates > base.meta_search.max_candidates
    assert faster.meta_search.max_metric_regression < base.meta_search.max_metric_regression
    assert faster.meta_search.rejection_penalty > base.meta_search.rejection_penalty


def test_architecture_evolution_preserves_authoritative_state(tmp_path):
    architecture = _architecture(tmp_path)
    before = architecture.status()
    before_hash = before["state_hash"]
    before_theta = before["distiller"]["theta_hash"]
    before_cycle = before["cycle"]

    report = architecture.evolve_architecture()
    after = architecture.status()

    assert after["state_hash"] == before_hash
    assert after["distiller"]["theta_hash"] == before_theta
    assert after["cycle"] == before_cycle
    assert after["architecture_evolution"]["epoch"] == 1
    assert after["architecture_evolution"]["journal_valid"] is True
    assert report.claim_status == "internal_architecture_improvement_only"


def test_architecture_lattice_has_center_and_26_neighbours():
    architecture = _architecture()
    lattice = architecture.architecture_lattice()

    assert len(lattice["nodes"]) == 27
    assert lattice["center"] == {"cadence": 0, "search": 0, "resilience": 0}
    assert lattice["axes"]["x"] == "state-to-meta cadence"


def test_autonomic_run_closes_all_nested_loops_with_bounded_policy(tmp_path):
    architecture = _architecture(tmp_path)
    report = architecture.run_autonomic(1)
    status = architecture.status()

    assert len(report.state_reports) == 1
    assert report.state_reports[0].committed
    assert len(report.meta_reports) == 1
    assert len(report.architecture_reports) == 1
    assert status["cycle"] == 1
    assert status["meta_optimizer"]["epoch"] == 1
    assert status["architecture_evolution"]["epoch"] == 1
    assert status["architecture_evolution"]["required_pipeline_mutable"] is False


def test_architecture_optimizer_is_bounded_by_candidate_budget():
    architecture = _architecture()
    source = demo_field(8)
    optimizer = DrMoagiArchitectureOptimizer()
    policy = replace(_policy(), max_architecture_candidates=1)

    report = optimizer.optimize(source, architecture.kernel.config, policy)

    assert report.evaluated_candidates == 2
    assert report.baseline.metrics.state_cycles == 1
    assert report.best.metrics.score >= 0.0 or report.best.metrics.meta_relative_improvement > 0.0


def test_system_api_exposes_four_scale_control_plane_routes():
    from jarvisx.dr_moagi_system_api import app

    paths = {route.path for route in app.routes}
    assert "/healthz" in paths
    assert "/v1/system/capabilities" in paths
    assert "/v1/system/status" in paths
    assert "/v1/system/step" in paths
    assert "/v1/system/run" in paths
    assert "/v1/system/meta/optimize" in paths
    assert "/v1/system/architecture/lattice" in paths
    assert "/v1/system/architecture/evolve" in paths
    assert "/v1/system/autonomic/run" in paths
