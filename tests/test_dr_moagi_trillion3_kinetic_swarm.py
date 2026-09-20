import math

import pytest

from jarvisx.dr_moagi_trillion3_kinetic_swarm import (
    LOGICAL_PARAMETER_POSITIONS,
    LOGICAL_PARAMETERS_PER_SITE,
    LOGICAL_SITES,
    VIRTUAL_SIDE,
    DrMoagiTrillion3KineticSwarm,
    Trillion3KineticConfig,
    chart_coordinate,
)


def test_trillion_cubed_extent_is_virtual_metadata_not_dense_allocation():
    assert VIRTUAL_SIDE == 10**12
    assert LOGICAL_SITES == 10**36
    assert LOGICAL_PARAMETERS_PER_SITE == 10**12
    assert LOGICAL_PARAMETER_POSITIONS == 10**48

    cfg = Trillion3KineticConfig(active_nodes=27)
    engine = DrMoagiTrillion3KineticSwarm(cfg)

    assert len(engine.nodes) == 27
    assert engine.block_side == 3
    assert all(len(node.address) == 3 for node in engine.nodes)
    assert all(
        0 <= coordinate < VIRTUAL_SIDE
        for node in engine.nodes
        for coordinate in node.address
    )


def test_contiguous_materialization_has_exact_six_neighbor_cube_topology():
    cfg = Trillion3KineticConfig(active_nodes=125, neighbor_mode=6)
    engine = DrMoagiTrillion3KineticSwarm(cfg)

    # A 5x5x5 grid has 3 * (5 - 1) * 5 * 5 = 300 undirected edges.
    assert engine.block_side == 5
    assert engine.active_edges == 300
    assert max(len(items) for items in engine.neighbors) == 6


def test_chart_projection_is_bounded():
    assert chart_coordinate(0) == pytest.approx(-1.0)
    assert chart_coordinate(VIRTUAL_SIDE - 1) == pytest.approx(1.0)
    assert -1.0 <= chart_coordinate(VIRTUAL_SIDE // 2) <= 1.0

    with pytest.raises(ValueError):
        chart_coordinate(-1)
    with pytest.raises(ValueError):
        chart_coordinate(VIRTUAL_SIDE)


def test_kinetic_decomposition_closes_and_manifold_is_enforced():
    cfg = Trillion3KineticConfig(
        active_nodes=64,
        state_dim=6,
        model_sketch_dim=8,
        latent_dim=8,
        manifold_radius=0.35,
        seed=17,
    )
    engine = DrMoagiTrillion3KineticSwarm(cfg)

    receipt = engine.step()

    assert receipt.kinetic_balance_error <= 1.0e-12
    assert receipt.post_projection_violations == 0
    assert receipt.active_nodes == 64
    assert receipt.logical_sites == 10**36
    assert receipt.logical_parameter_positions == 10**48
    assert math.isfinite(receipt.mean_cognitive_speed)
    assert math.isfinite(receipt.mean_consensus_speed)
    assert math.isfinite(receipt.mean_projection_speed)
    assert math.isfinite(receipt.mean_total_speed)
    assert math.isfinite(receipt.model_sketch_mse)

    for node in engine.nodes:
        norm = math.sqrt(sum(value * value for value in node.state))
        assert norm <= cfg.manifold_radius + 1.0e-12


def test_recursive_swarm_is_deterministic_and_memory_is_bounded():
    cfg = Trillion3KineticConfig(active_nodes=32, seed=29)
    left = DrMoagiTrillion3KineticSwarm(cfg)
    right = DrMoagiTrillion3KineticSwarm(cfg)

    initial_digest = left.state_digest()
    left_receipts = left.run(3)
    right_receipts = right.run(3)

    assert left_receipts == right_receipts
    assert left.state_digest() == right.state_digest()
    assert left.state_digest() != initial_digest
    assert any(receipt.mean_memory_norm > 0.0 for receipt in left_receipts)

    for node in left.nodes:
        assert all(math.isfinite(value) for value in node.state)
        assert all(math.isfinite(value) for value in node.memory)


def test_twenty_six_neighbor_mode_is_supported_and_denser():
    six = DrMoagiTrillion3KineticSwarm(
        Trillion3KineticConfig(active_nodes=27, neighbor_mode=6)
    )
    twenty_six = DrMoagiTrillion3KineticSwarm(
        Trillion3KineticConfig(active_nodes=27, neighbor_mode=26)
    )

    assert twenty_six.active_edges > six.active_edges
    assert max(len(items) for items in twenty_six.neighbors) == 26


@pytest.mark.parametrize(
    "kwargs",
    [
        {"active_nodes": 0},
        {"neighbor_mode": 7},
        {"dt": 0.0},
        {"manifold_radius": 0.0},
        {"consensus_gain": 1.1},
        {"memory_beta": -0.1},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        Trillion3KineticConfig(**kwargs)
