import math

import pytest

from jarvisx.tetration_field import (
    BRICK_SIZE,
    BrickAutoencoderMoE,
    BrickState,
    FieldMechanics,
    SparseHashDirectory,
    TetrationAddress,
    TetrationFieldAutomaton,
    TetrationUniverse,
    make_brick_pulse,
)


def origin(height=2):
    return TetrationAddress(height, "origin", 0, 0, 0)


def test_tetration_descriptor_height_two():
    universe = TetrationUniverse(height=2)
    descriptor = universe.descriptor()
    assert descriptor["axis_size"] == "1000^1000"
    assert descriptor["virtual_cells"] == "(1000^1000)^3"
    assert descriptor["coordinate_bits_materialised"] == 29898


def test_higher_tower_remains_symbolic():
    universe = TetrationUniverse(height=4)
    assert universe.coordinate_bits_if_materialisable is None
    assert universe.axis_expression == "1000↑↑4"


def test_hash_directory_collision_chaining():
    directory = SparseHashDirectory(bucket_count=1)
    zero = tuple(0.0 for _ in range(BRICK_SIZE))
    a = origin()
    b = a.offset(1, 0, 0)
    directory.set(a, BrickState(zero, zero))
    directory.set(b, BrickState(tuple(1.0 for _ in range(BRICK_SIZE)), zero))
    assert len(directory) == 2
    assert directory.collision_count() == 1
    assert directory.get(a).values[0] == 0.0
    assert directory.get(b).values[0] == 1.0


def test_full_projection_and_top_one_router():
    network = BrickAutoencoderMoE(latent_dim=8, expert_count=3, seed=7)
    values = tuple(16.0 + (index % 11) for index in range(BRICK_SIZE))
    omega = tuple(0.1 for _ in range(BRICK_SIZE))
    latent = network.encode(values)
    conditioned = network.condition_with_omega(latent, omega)
    expert, gates = network.route(conditioned)
    decoded, forward_expert, _ = network.forward(values, omega)
    assert len(latent) == 8
    assert len(decoded) == BRICK_SIZE
    assert forward_expert == expert
    assert math.isclose(sum(gates), 1.0, rel_tol=1e-12)
    assert len(set(round(value, 8) for value in decoded)) > 1


def test_stability_contract_rejects_rho_one():
    with pytest.raises(ValueError):
        FieldMechanics(omega_retention=1.0).validate()


def test_cross_brick_laplacian_reads_neighbour_brick():
    engine = TetrationFieldAutomaton(seed=3)
    a = origin()
    b = a.offset(1, 0, 0)
    base_a = list(engine.procedural_brick(a))
    base_b = list(engine.procedural_brick(b))
    idx_a = engine._flat_index(0, 3, 1, 1)
    idx_b = engine._flat_index(0, 0, 1, 1)
    base_a[idx_a] = 20.0
    base_b[idx_b] = 100.0
    zero = tuple(0.0 for _ in range(BRICK_SIZE))
    states = {
        a: BrickState(tuple(base_a), zero),
        b: BrickState(tuple(base_b), zero),
    }
    lap = engine._laplacian(states, a, 0, 3, 1, 1, 20.0)
    local_other_five = (
        engine._voxel_value(states, a, 0, 2, 1, 1)
        + engine._voxel_value(states, a, 0, 3, 2, 1)
        + engine._voxel_value(states, a, 0, 3, 0, 1)
        + engine._voxel_value(states, a, 0, 3, 1, 2)
        + engine._voxel_value(states, a, 0, 3, 1, 0)
    )
    assert math.isclose(lap, 100.0 + local_other_five - 120.0)


def test_transaction_is_sparse_and_commits():
    engine = TetrationFieldAutomaton(
        mechanics=FieldMechanics(max_active_bricks=32),
        latent_dim=8,
        expert_count=3,
    )
    metrics = engine.step({origin(): make_brick_pulse(24.0)})
    assert metrics.committed
    assert metrics.materialised_bricks <= 32
    assert metrics.frontier_bricks <= 32
    assert sum(metrics.expert_histogram) == metrics.frontier_bricks
    assert engine.snapshot()["universe"]["virtual_cells"] == "(1000^1000)^3"


def test_exact_replay():
    kwargs = dict(
        mechanics=FieldMechanics(max_active_bricks=32),
        latent_dim=8,
        expert_count=3,
        seed=99,
        bucket_count=17,
    )
    left = TetrationFieldAutomaton(**kwargs)
    right = TetrationFieldAutomaton(**kwargs)
    workload = [{origin(): make_brick_pulse(20.0)}, None, None]
    for injections in workload:
        assert left.step(injections).to_dict() == right.step(injections).to_dict()
    assert left.journal_hash == right.journal_hash


def test_failed_injection_rolls_back():
    engine = TetrationFieldAutomaton()
    before = engine.snapshot()
    metrics = engine.step({origin(): [1.0, 2.0]})
    assert not metrics.committed
    assert engine.snapshot()["cycle"] == before["cycle"]
    assert engine.snapshot()["journal_hash"] == before["journal_hash"]
