from jarvisx.swarm import SparseSwarm30D, SwarmConfig


def unit_vector(dimensions: int = 30, value: float = 1.0):
    vector = [0.0] * dimensions
    vector[0] = value
    return vector


def test_virtual_volume_is_not_materialised():
    swarm = SparseSwarm30D()
    assert swarm.virtual_voxels == 1_000_000_000
    assert len(swarm.voxels) == 0


def test_single_voxel_diffuses_to_neighbours():
    config = SwarmConfig(diffusion=0.1, reaction=0.0, learning_rate=0.0, epsilon=1e-12)
    swarm = SparseSwarm30D(config)
    swarm.set_voxel((500, 500, 500), unit_vector())

    metrics = swarm.step()

    assert metrics.active_before == 1
    assert metrics.active_after == 7
    assert swarm.voxels[(500, 500, 500)].theta[0] == 0.4
    for coord in (
        (501, 500, 500), (499, 500, 500),
        (500, 501, 500), (500, 499, 500),
        (500, 500, 501), (500, 500, 499),
    ):
        assert swarm.voxels[coord].theta[0] == 0.1


def test_zero_state_remains_sparse_and_converged():
    swarm = SparseSwarm30D()
    metrics = swarm.step()
    assert metrics.active_after == 0
    assert metrics.residual_l2 == 0.0
    assert metrics.total_loss == 0.0


def test_periodic_boundary_wraps_diffusion():
    config = SwarmConfig(
        side_length=4,
        diffusion=0.1,
        reaction=0.0,
        learning_rate=0.0,
        epsilon=1e-12,
        boundary="periodic",
    )
    swarm = SparseSwarm30D(config)
    swarm.set_voxel((0, 0, 0), unit_vector())
    swarm.step()
    assert (3, 0, 0) in swarm.voxels
    assert (0, 3, 0) in swarm.voxels
    assert (0, 0, 3) in swarm.voxels


def test_run_stops_at_tolerance():
    swarm = SparseSwarm30D()
    history = swarm.run(iterations=100, tolerance=0.0)
    assert len(history) == 1
