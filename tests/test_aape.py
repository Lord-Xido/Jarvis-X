import itertools

import pytest

from jarvisx.aape import AAPEConfig, BitLattice, JXAAPEEngine


def small_engine(**kwargs):
    return JXAAPEEngine(AAPEConfig(side=4, max_ca_steps=4, **kwargs))


def test_packing_round_trip():
    engine = small_engine()
    lattice = engine.lattice([0, 1, 63])
    assert lattice.word_count == 1
    assert BitLattice.from_words(4, lattice.words()) == lattice


def test_majority7_boolean_network_is_exact():
    for values in itertools.product((0, 1), repeat=7):
        actual = JXAAPEEngine._majority7_exact(*values)
        assert actual == (1 if sum(values) >= 4 else 0)


def test_toroidal_neighbors_wrap_across_x_boundary():
    engine = small_engine()
    source = engine.lattice([0])
    x_plus, x_minus = engine._topology.neighbors(source.bits)[:2]
    assert x_plus & (1 << 3)
    assert x_minus & (1 << 1)


def test_intent_anchor_prevents_empty_projection_and_gates_decode():
    engine = small_engine(max_tokens=8)
    anchor = engine.lattice([7])
    state = engine.cycle([0xFFFF], intent_mask=anchor, max_tokens=8)
    assert state.projected.bits & anchor.bits
    assert state.tokens
    assert state.representation_tag == "SIMULATION_NOT_TERRITORY"
    assert state.semantic_gap > 0


def test_kappa_feedback_is_bounded():
    engine = small_engine()
    for _ in range(20):
        engine.update_kappa(1)
    assert engine.kappa == engine.config.kappa_min
    for _ in range(20):
        engine.update_kappa(0)
    assert engine.kappa == engine.config.kappa_max


def test_hash_chain_is_deterministic_for_equal_trajectories():
    config = AAPEConfig(side=4, max_ca_steps=2)
    left = JXAAPEEngine(config)
    right = JXAAPEEngine(config)
    intent = left.lattice([1, 2, 3])
    state_left = left.cycle([0x5000, 0x6000], intent_mask=intent, quality_signal=1)
    state_right = right.cycle([0x5000, 0x6000], intent_mask=intent, quality_signal=1)
    assert state_left.projected == state_right.projected
    assert state_left.tokens == state_right.tokens
    assert state_left.omega_digest == state_right.omega_digest


def test_embedding_range_is_validated():
    engine = small_engine()
    with pytest.raises(ValueError):
        engine.encode([-1])
