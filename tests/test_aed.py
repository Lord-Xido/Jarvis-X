import math

import pytest

from jarvisx.aed import AEDConfig, MM3DAEDEngine


def test_encoder_reaches_signed_three_bit_extrema():
    engine = MM3DAEDEngine()
    assert engine.encode([0, 255]) == (-4, 3)


def test_cycle_preserves_invariants_and_commits_front_buffer():
    engine = MM3DAEDEngine()
    state = engine.cycle([0, 127.5, 255])

    assert state.cycle == 1
    assert engine.front_state is state
    assert len(state.ambient_output) == 3
    assert all(0 <= value <= 255 for value in state.ambient_output)
    assert state.semantic_gap > 0
    assert math.isinf(state.reality_separation)
    assert state.representation_tag == "SIMULATION_NOT_TERRITORY"


def test_memory_and_intent_change_the_projection():
    engine = MM3DAEDEngine(AEDConfig(memory_coupling=0.5, intent_gain=0.5))
    baseline = engine.cycle([128, 128], memory=[0], intent=[0])
    shifted = engine.cycle([128, 128], memory=[3], intent=[1])

    assert shifted.latent_projected != baseline.latent_projected
    assert shifted.ambient_output != baseline.ambient_output


def test_constraints_gate_the_projected_latent():
    engine = MM3DAEDEngine()
    state = engine.cycle([255], intent=[100], constraints=[(-1.0, 1.0)])
    assert state.latent_projected == (1.0,)


def test_vector_lengths_are_validated():
    engine = MM3DAEDEngine()
    with pytest.raises(ValueError):
        engine.cycle([0, 1], memory=[0, 1, 2])
