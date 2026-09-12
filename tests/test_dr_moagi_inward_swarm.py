import math

from jarvisx.dr_moagi_inward_swarm import (
    LOGICAL_AXIS,
    LOGICAL_CELLS,
    InwardSwarmConfig,
    VirtualExavoxelInwardSwarm,
    modeled_work_reduction,
)


def test_logical_exavoxel_geometry():
    assert LOGICAL_AXIS == 1_000_000
    assert LOGICAL_CELLS == 10**18


def test_modeled_work_reduction_bounds():
    assert modeled_work_reduction(1.0, 1000) == 1.0
    assert modeled_work_reduction(0.0, 1000) == 1000.0
    mid = modeled_work_reduction(0.1, 1000)
    assert 1.0 < mid < 1000.0


def test_swarm_step_is_bounded_and_virtualized():
    cfg = InwardSwarmConfig(proxy_agents=128, fold_edge=10, residual_threshold=0.12)
    engine = VirtualExavoxelInwardSwarm(config=cfg)
    metrics = engine.step(virtual_ops=0, auto_optimize=False)

    assert metrics.logical_cells == 10**18
    assert metrics.proxy_agents == 128
    assert metrics.fold_volume == 1000
    assert 0.0 <= metrics.active_refinement_fraction <= 1.0
    assert 1.0 <= metrics.modeled_work_reduction <= 1000.0
    assert math.isfinite(metrics.proxy_residual_rms)
    assert math.isfinite(metrics.aggregate_reconstruction_mse)
    assert math.isfinite(metrics.aggregate_cycle_mse)


def test_auto_optimizer_keeps_threshold_in_bounds():
    cfg = InwardSwarmConfig(
        proxy_agents=128,
        fold_edge=10,
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
