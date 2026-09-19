import math

import pytest

from jarvisx.dr_moagi_septillion_swarm import (
    LOGICAL_SITES,
    VIRTUAL_SIDE,
    DrMoagiSeptillionSwarmEngine,
    SeptillionSwarmConfig,
    chart_coordinate,
    virtual_coordinate,
)


def test_septillion_cubed_virtual_extent_is_metadata_not_allocation():
    assert VIRTUAL_SIDE == 10**24
    assert LOGICAL_SITES == 10**72

    cfg = SeptillionSwarmConfig(active_agents=12, latent_dim=4)
    engine = DrMoagiSeptillionSwarmEngine(cfg)

    assert len(engine.agents) == 12
    assert all(len(agent.address) == 3 for agent in engine.agents)


def test_virtual_addressing_is_deterministic_and_bounded():
    first = [virtual_coordinate(17, axis, seed=11) for axis in range(3)]
    second = [virtual_coordinate(17, axis, seed=11) for axis in range(3)]

    assert first == second
    assert len(set(first)) == 3
    assert all(0 <= value < VIRTUAL_SIDE for value in first)
    assert all(-1.0 <= chart_coordinate(value) <= 1.0 for value in first)


def test_recursive_swarm_is_deterministic_and_physically_bounded():
    cfg = SeptillionSwarmConfig(
        active_agents=24,
        latent_dim=6,
        max_refine_steps=12,
        fixed_point_tolerance=1.0e-8,
        seed=19,
    )
    left = DrMoagiSeptillionSwarmEngine(cfg)
    right = DrMoagiSeptillionSwarmEngine(cfg)

    left_receipts = left.run(3)
    right_receipts = right.run(3)

    assert left_receipts == right_receipts
    assert left.state_digest() == right.state_digest()

    for receipt in left_receipts:
        assert receipt.logical_sites == 10**72
        assert receipt.active_agents == cfg.active_agents
        assert 1 <= receipt.physical_refine_steps <= cfg.max_refine_steps
        assert math.isfinite(receipt.fixed_point_residual)
        assert math.isfinite(receipt.swarm_radius)


def test_ctr_guard_never_commits_a_worse_decoder_candidate():
    cfg = SeptillionSwarmConfig(active_agents=32, latent_dim=6, max_refine_steps=10)
    engine = DrMoagiSeptillionSwarmEngine(cfg)

    receipt = engine.step()

    assert receipt.post_adaptation_mse <= receipt.pre_adaptation_mse + 1.0e-15
    assert receipt.corrected_mse <= receipt.pre_adaptation_mse + 1.0e-12


def test_recurrence_updates_memory_and_world_state():
    cfg = SeptillionSwarmConfig(active_agents=16, latent_dim=4, max_refine_steps=8)
    engine = DrMoagiSeptillionSwarmEngine(cfg)

    before_position = [agent.position[:] for agent in engine.agents]
    before_memory = engine.omega[:]

    engine.step()

    assert engine.omega != before_memory
    assert [agent.position for agent in engine.agents] != before_position
    assert engine.cycle == 1
    assert len(engine.receipts) == 1


def test_configuration_rejects_non_contractive_latent_gain():
    with pytest.raises(ValueError):
        SeptillionSwarmConfig(hidden_contractivity=0.90, collective_gain=0.15)
