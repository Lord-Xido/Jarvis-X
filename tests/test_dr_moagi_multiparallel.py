from __future__ import annotations

import pytest

from jarvisx.dr_moagi_multiparallel import DrMoagiMultiparallel3D, MultiparallelConfig


def test_default_topology_is_one_million_logical_loops_without_dense_allocation():
    engine = DrMoagiMultiparallel3D()
    status = engine.status()

    assert status["logical_loops"] == 1_000_000
    assert status["logical_volume_cells"] == 9_000_000
    assert status["active_loops"] == 0
    assert status["allocated_volume_cells"] == 0


def test_sparse_inward_fold_materializes_only_active_depth_cells():
    config = MultiparallelConfig(depth=4, expand_halo=False, global_coupling=0.0)
    engine = DrMoagiMultiparallel3D(config)
    engine.load({(10, 20): (1.0, 0.5), (999, 999): (0.25, -0.5)})

    assert engine.active_loop_count == 2
    assert engine.allocated_volume_cell_count == 2 * (config.depth + 1)

    reconstruction = engine.reconstruct()
    assert reconstruction[(10, 20)] == pytest.approx((1.0, 0.5), abs=1e-4)
    assert reconstruction[(999, 999)] == pytest.approx((0.25, -0.5), abs=1e-4)


def test_kinetic_permeation_activates_neighbor_halo_and_commits():
    config = MultiparallelConfig(
        depth=3,
        max_active_loops=32,
        diffusion=0.1,
        global_coupling=0.0,
        expand_halo=True,
    )
    engine = DrMoagiMultiparallel3D(config)
    engine.load({(500, 500): (1.0, 0.0)})

    report = engine.step()
    surface = engine.snapshot_surface()

    assert report.committed
    assert report.support_loops == 5
    assert engine.active_loop_count == 5
    assert surface[(499, 500)][0] > 0.0
    assert surface[(501, 500)][0] > 0.0
    assert surface[(500, 499)][0] > 0.0
    assert surface[(500, 501)][0] > 0.0


def test_global_core_couples_local_cores_without_dense_materialization():
    config = MultiparallelConfig(
        depth=2,
        expand_halo=False,
        global_coupling=0.25,
        max_reconstruction_mse=1.0,
    )
    engine = DrMoagiMultiparallel3D(config)
    engine.load({(0, 0): (1.0,), (999, 999): (-1.0,)})

    report = engine.step()

    assert report.committed
    assert report.global_core == pytest.approx((0.0,), abs=1e-6)
    assert report.allocated_volume_cells == 6
    assert report.reconstruction_mse > 0.0


def test_reconstruction_budget_rejects_and_rolls_back_surface():
    config = MultiparallelConfig(
        depth=2,
        diffusion=0.0,
        memory_gain=0.0,
        expand_halo=False,
        global_coupling=0.5,
        max_reconstruction_mse=0.0,
    )
    engine = DrMoagiMultiparallel3D(config)
    engine.load({(1, 1): (1.0,), (2, 2): (0.5,)})
    before = engine.snapshot_surface()

    report = engine.step()

    assert not report.committed
    assert report.rejection_reason == "reconstruction error budget exceeded"
    assert engine.snapshot_surface() == before


def test_input_contract_rejects_bad_coordinates_and_mixed_channel_widths():
    engine = DrMoagiMultiparallel3D()

    with pytest.raises(ValueError, match="outside logical"):
        engine.load({(1000, 0): (1.0,)})
    with pytest.raises(ValueError, match="same channel count"):
        engine.load({(0, 0): (1.0,), (1, 1): (1.0, 2.0)})
