import math

from jarvisx.dr_moagi_inward_swarm import (
    LOGICAL_AXIS,
    LOGICAL_CELLS,
    InwardSwarmConfig,
    VirtualExavoxelInwardSwarm,
    modeled_work_reduction,
    verify_recursive_geometry,
    verify_work_model,
)


def test_logical_exavoxel_geometry():
    assert LOGICAL_AXIS == 1_000_000
    assert LOGICAL_CELLS == 10**18


def test_recursive_default_geometry_reaches_million_fold_ceiling():
    cfg = InwardSwarmConfig(proxy_agents=128)
    assert cfg.fold_edge == 10
    assert cfg.fold_levels == 2
    assert cfg.per_level_volume == 1000
    assert cfg.total_fold_edge == 100
    assert cfg.fold_volume == 1_000_000
    assert cfg.coarse_axis == 10_000
    assert cfg.coarse_cells == 10**12
    assert verify_recursive_geometry(cfg)


def test_modeled_work_reduction_bounds_for_million_fold():
    fold_volume = 1_000_000
    assert modeled_work_reduction(1.0, fold_volume) == 1.0
    assert modeled_work_reduction(0.0, fold_volume) == 1_000_000.0
    mid = modeled_work_reduction(0.1, fold_volume)
    assert 1.0 < mid < 1_000_000.0
    assert verify_work_model(0.1, fold_volume, mid)


def test_single_level_remains_compatible_with_original_thousand_fold():
    cfg = InwardSwarmConfig(proxy_agents=128, fold_edge=10, fold_levels=1)
    assert cfg.fold_volume == 1000
    assert cfg.total_fold_edge == 10
    assert verify_recursive_geometry(cfg)


def test_swarm_step_is_bounded_virtualized_and_verified():
    cfg = InwardSwarmConfig(
        proxy_agents=128,
        fold_edge=10,
        fold_levels=2,
        residual_threshold=0.12,
    )
    engine = VirtualExavoxelInwardSwarm(config=cfg)
    metrics = engine.step(virtual_ops=0, auto_optimize=False)

    assert metrics.logical_cells == 10**18
    assert metrics.proxy_agents == 128
    assert metrics.per_level_volume == 1000
    assert metrics.fold_volume == 1_000_000
    assert metrics.coarse_axis == 10_000
    assert metrics.coarse_cells == 10**12
    assert 0.0 <= metrics.active_refinement_fraction <= 1.0
    assert 1.0 <= metrics.modeled_work_reduction <= 1_000_000.0
    assert math.isfinite(metrics.proxy_residual_rms)
    assert math.isfinite(metrics.aggregate_reconstruction_mse)
    assert math.isfinite(metrics.aggregate_cycle_mse)
    assert metrics.geometry_verified
    assert metrics.work_model_verified
    assert metrics.operational_mechanics_verified


def test_auto_optimizer_keeps_threshold_in_bounds():
    cfg = InwardSwarmConfig(
        proxy_agents=128,
        fold_edge=10,
        fold_levels=2,
        residual_threshold=0.12,
        min_threshold=0.05,
        max_threshold=0.20,
    )
    engine = VirtualExavoxelInwardSwarm(config=cfg)
    engine.run(3, deterministic_virtual_stride=1_000, auto_optimize=True)
    assert cfg.min_threshold <= cfg.residual_threshold <= cfg.max_threshold


def test_deterministic_run_length_and_cycle_count():
    cfg = InwardSwarmConfig(proxy_agents=128, seed=1234)
    engine = VirtualExavoxelInwardSwarm(config=cfg)
    metrics = engine.run(3, deterministic_virtual_stride=10)
    assert [m.cycle for m in metrics] == [1, 2, 3]


def test_invalid_recursive_geometry_is_rejected():
    try:
        InwardSwarmConfig(proxy_agents=128, fold_edge=7, fold_levels=2)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected invalid recursive geometry to be rejected")
