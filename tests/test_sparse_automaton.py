import math

import pytest

from jarvisx.automaton import (
    ADDRESS_DEPTH,
    AXIS_SIZE,
    RADIX,
    BoundedMechanicsOptimizer,
    Coordinate3D,
    DeterministicAutoencoder,
    Mechanics,
    Sparse3DAutomaton,
    make_echo_injections,
)


def test_virtual_universe_is_exactly_described_without_dense_allocation():
    descriptor = Sparse3DAutomaton.universe_descriptor()
    assert RADIX == 1000
    assert ADDRESS_DEPTH == 1000
    assert AXIS_SIZE == 10**3000
    assert descriptor["virtual_cells"] == "10^9000"
    assert descriptor["virtual_cell_exponent"] == 9000


def test_coordinate_wraps_at_the_virtual_axis_boundary():
    origin = Coordinate3D(0, 0, 0)
    wrapped = origin.offset(-1, 0, 0)
    assert wrapped.x == AXIS_SIZE - 1
    assert wrapped.offset(1, 0, 0) == origin


def test_procedural_field_is_deterministic_and_unmaterialised():
    coordinate = Coordinate3D(123, 456, 789)
    left = Sparse3DAutomaton(seed=17)
    right = Sparse3DAutomaton(seed=17)
    assert left.materialised_cells == 0
    assert left.procedural_value(coordinate) == right.procedural_value(coordinate)
    assert left.materialised_cells == 0


def test_autoencoder_has_bounded_deterministic_dimensions():
    network = DeterministicAutoencoder(latent_dim=6, seed=9)
    neighbourhood = (1.0, 0.5, -0.5, 0.25, -0.25, 0.1, -0.1)
    latent = network.encode(neighbourhood)
    evolved = network.evolve(latent, omega=0.2)
    reconstruction = network.decode(evolved)
    assert len(latent) == 6
    assert len(evolved) == 6
    assert len(reconstruction) == 7
    assert all(-1.0 <= value <= 1.0 for value in latent + evolved + reconstruction)


def test_transaction_commits_and_advances_hash_chain():
    engine = Sparse3DAutomaton(seed=21)
    before_hash = engine.journal_hash
    pulse = make_echo_injections(side=3, amplitude=1.0)
    metrics = engine.step(pulse)
    assert metrics.committed
    assert metrics.cycle == 1
    assert metrics.materialised_cells > 0
    assert metrics.journal_hash != before_hash
    assert engine.journal_hash == metrics.journal_hash


def test_replay_is_bit_deterministic():
    pulse = make_echo_injections(side=3, amplitude=0.75)
    first = Sparse3DAutomaton(seed=1337)
    second = Sparse3DAutomaton(seed=1337)
    for index in range(5):
        inputs = pulse if index == 0 else None
        left = first.step(inputs)
        right = second.step(inputs)
        assert left == right
    assert first.state_digest() == second.state_digest()
    assert first.cells == second.cells


def test_sparse_budget_is_enforced_by_frontier_selection():
    mechanics = Mechanics(max_active_cells=32)
    engine = Sparse3DAutomaton(seed=3, mechanics=mechanics)
    pulse = make_echo_injections(side=5, amplitude=1.0)
    metrics = engine.step(pulse)
    assert metrics.committed
    assert metrics.materialised_cells <= mechanics.max_active_cells
    assert metrics.frontier_cells <= mechanics.max_active_cells


def test_failed_energy_verification_rolls_back_atomically():
    mechanics = Mechanics(max_energy=1e-12, max_active_cells=128)
    engine = Sparse3DAutomaton(seed=5, mechanics=mechanics)
    digest_before = engine.state_digest()
    metrics = engine.step({Coordinate3D(1, 1, 1): 1.0})
    assert not metrics.committed
    assert metrics.rollback_reason == "energy budget exceeded"
    assert engine.cycle == 0
    assert engine.materialised_cells == 0
    assert engine.state_digest() == digest_before


def test_bounded_optimizer_never_accepts_invalid_candidate():
    engine = Sparse3DAutomaton(seed=8)
    pulse = make_echo_injections(side=1, amplitude=0.4)
    optimizer = BoundedMechanicsOptimizer()
    invalid = Mechanics(diffusion=2.0)
    with pytest.raises(ValueError):
        optimizer.optimize(engine, [pulse], candidates=[invalid])


def test_bounded_optimizer_returns_declared_mechanics():
    engine = Sparse3DAutomaton(seed=11)
    pulse = make_echo_injections(side=1, amplitude=0.2)
    optimizer = BoundedMechanicsOptimizer()
    result = optimizer.optimize(engine, [pulse, {}])
    assert math.isfinite(result.baseline_score)
    assert math.isfinite(result.candidate_score)
    assert result.candidate_score <= result.baseline_score + 1e-15
    assert engine.mechanics == result.selected_mechanics
