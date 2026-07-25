import pytest

from jarvisx.swarm import SparseSwarm30D, SwarmConfig


def unit_vector(dimensions: int = 30, value: float = 1.0):
    vector = [0.0] * dimensions
    vector[0] = value
    return vector


def test_virtual_volume_is_not_materialised():
    swarm = SparseSwarm30D()
    assert swarm.virtual_voxels == 1_000_000_000
    assert swarm.active_voxels == 0


def test_single_voxel_diffuses_to_neighbours():
    config = SwarmConfig(
        diffusion=0.1,
        reaction=0.0,
        learning_rate=0.0,
        memory_rate=0.0,
        epsilon=1e-12,
    )
    swarm = SparseSwarm30D(config)
    swarm.set_voxel((500, 500, 500), unit_vector())

    metrics = swarm.step()

    assert metrics.active_before == 1
    assert metrics.active_after == 7
    assert swarm.voxels[(500, 500, 500)].theta[0] == pytest.approx(0.4)
    for coord in (
        (501, 500, 500),
        (499, 500, 500),
        (500, 501, 500),
        (500, 499, 500),
        (500, 500, 501),
        (500, 500, 499),
    ):
        assert swarm.voxels[coord].theta[0] == pytest.approx(0.1)


def test_zero_state_remains_sparse_and_converged():
    swarm = SparseSwarm30D()
    metrics = swarm.step()
    assert metrics.active_after == 0
    assert metrics.residual_l2 == 0.0
    assert metrics.motion_l2 == 0.0
    assert metrics.constraint_l2 == 0.0
    assert metrics.total_loss == 0.0


def test_periodic_boundary_wraps_diffusion():
    config = SwarmConfig(
        side_length=4,
        diffusion=0.1,
        reaction=0.0,
        learning_rate=0.0,
        memory_rate=0.0,
        epsilon=1e-12,
        boundary="periodic",
    )
    swarm = SparseSwarm30D(config)
    swarm.set_voxel((0, 0, 0), unit_vector())
    swarm.step()
    assert (3, 0, 0) in swarm.voxels
    assert (0, 3, 0) in swarm.voxels
    assert (0, 0, 3) in swarm.voxels


def test_run_stops_at_prediction_and_motion_tolerance():
    swarm = SparseSwarm30D()
    history = swarm.run(iterations=100, tolerance=0.0, motion_tolerance=0.0)
    assert len(history) == 1


def test_unstable_explicit_diffusion_is_rejected():
    with pytest.raises(ValueError, match="unstable"):
        SwarmConfig(diffusion=0.2, time_step=1.0, spacing=1.0)


def test_active_budget_bounds_frontier_expansion():
    config = SwarmConfig(
        diffusion=0.1,
        reaction=0.0,
        learning_rate=0.0,
        memory_rate=0.0,
        active_budget=3,
        epsilon=1e-12,
    )
    swarm = SparseSwarm30D(config)
    swarm.set_voxel((500, 500, 500), unit_vector())

    metrics = swarm.step()

    assert metrics.active_after == 3
    assert metrics.budget_utilization == 1.0
    assert (500, 500, 500) in swarm.voxels


def test_constraint_projection_is_measured_and_enforced():
    config = SwarmConfig(
        dimensions=2,
        diffusion=0.0,
        reaction=0.0,
        learning_rate=0.0,
        memory_rate=0.0,
        max_abs_state=0.5,
    )
    swarm = SparseSwarm30D(config)
    swarm.set_voxel((1, 1, 1), [2.0, -3.0])

    metrics = swarm.step()

    assert swarm.voxels[(1, 1, 1)].theta == [0.5, -0.5]
    assert metrics.constraint_l2 > 0.0
    assert metrics.constraint_loss > 0.0


def test_residual_updates_persistent_memory():
    config = SwarmConfig(
        dimensions=2,
        diffusion=0.0,
        reaction=0.0,
        learning_rate=0.0,
        memory_decay=1.0,
        memory_rate=0.5,
    )
    swarm = SparseSwarm30D(config, predictor_bias=[0.2, 0.0])
    swarm.set_voxel((1, 1, 1), [1.0, 0.0])

    swarm.step()

    assert swarm.voxels[(1, 1, 1)].memory[0] == pytest.approx(-0.1)


def test_identical_manifests_replay_to_identical_hashes():
    config = SwarmConfig(
        dimensions=2,
        diffusion=0.1,
        reaction=0.0,
        learning_rate=0.0,
        memory_rate=0.0,
    )
    left = SparseSwarm30D(config)
    right = SparseSwarm30D(config)
    for swarm in (left, right):
        swarm.set_voxel((5, 5, 5), [1.0, -0.5])

    left_metrics = left.step()
    right_metrics = right.step()

    assert left_metrics.journal_hash == right_metrics.journal_hash
    assert left.full_snapshot() == right.full_snapshot()


def test_journal_hash_chains_across_iterations():
    config = SwarmConfig(
        dimensions=2,
        diffusion=0.1,
        reaction=0.0,
        learning_rate=0.0,
        memory_rate=0.0,
    )
    swarm = SparseSwarm30D(config)
    swarm.set_voxel((5, 5, 5), [1.0, 0.0])

    first = swarm.step().journal_hash
    second = swarm.step().journal_hash

    assert first != "0" * 64
    assert second != first
    assert len(second) == 64
