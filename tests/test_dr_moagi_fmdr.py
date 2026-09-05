import math

import pytest

from jarvisx.dr_moagi_fmdr import (
    FMDRConfig,
    FourierMarkovDiffusionResonanceEngine,
    analyze_history,
    axis_wavevectors,
    diffusion_attenuation,
    feedback_candidate,
    spatial_fourier,
)


def wave_field(side: int, *, phase: float = 0.0, amplitude: float = 0.5):
    k = 2.0 * math.pi / side
    return {
        (x, 0, 0): amplitude * math.cos(k * x + phase)
        for x in range(side)
    }


def test_axis_wavevectors_and_sparse_fourier_identify_x_mode():
    wavevectors = axis_wavevectors(8)
    field = wave_field(8)

    assert len(wavevectors) == 3
    x_mode = spatial_fourier(field, wavevectors[0])
    y_mode = spatial_fourier(field, wavevectors[1])
    z_mode = spatial_fourier(field, wavevectors[2])

    assert abs(x_mode) == pytest.approx(0.25)
    assert abs(y_mode) < 1.0e-12
    assert abs(z_mode) < 1.0e-12


def test_diffusion_multiplier_matches_exponential_mode_decay():
    wavevector = axis_wavevectors(8)[0]
    observed = diffusion_attenuation(wavevector, (0.1, 0.2, 0.3), 0.5)
    k = 2.0 * math.pi / 8
    expected = math.exp(-0.1 * k * k * 0.5)

    assert observed == pytest.approx(expected)
    assert 0.0 < observed < 1.0


def test_history_analysis_detects_temporal_resonance_and_markov_persistence():
    x_mode, y_mode, _ = axis_wavevectors(8)
    config = FMDRConfig(
        wavevectors=(x_mode, y_mode),
        dt=1.0,
        diffusion=(0.01, 0.01, 0.01),
        damping=0.05,
    )
    history = [
        wave_field(8, phase=index * math.pi / 2)
        for index in range(8)
    ]

    report = analyze_history(history, config)
    selected = report.selected_mode

    assert report.dominant_mode_index == 0
    assert report.selected_mode_index == 0
    assert report.markov_persistence == pytest.approx(1.0)
    assert report.markov_transition_matrix[0][0] == pytest.approx(1.0)
    assert report.markov_transition_matrix[1][1] == pytest.approx(1.0)
    assert report.modes[1].self_transition_probability == pytest.approx(0.5)
    assert selected.spectral_coherence == pytest.approx(1.0)
    assert selected.dominant_omega == pytest.approx(math.pi / 2)
    assert selected.resonance_score > 0.0


def test_feedback_is_bounded_per_cell_and_within_value_projection():
    x_mode, y_mode, _ = axis_wavevectors(8)
    config = FMDRConfig(
        wavevectors=(x_mode, y_mode),
        diffusion=(0.01, 0.01, 0.01),
        damping=0.05,
        feedback_gain=0.5,
        max_feedback_delta=0.02,
        value_min=-0.75,
        value_max=0.75,
    )
    history = [wave_field(8, phase=index * math.pi / 2) for index in range(8)]
    report = analyze_history(history, config)
    source = history[-1]
    candidate = feedback_candidate(source, report, config)

    assert candidate != source
    for coordinate, value in source.items():
        updated = candidate.get(coordinate, 0.0)
        assert abs(updated - value) <= config.max_feedback_delta + 1.0e-12
        assert config.value_min <= updated <= config.value_max


def test_engine_waits_for_history_then_closes_the_inward_loop():
    x_mode, y_mode, _ = axis_wavevectors(8)
    config = FMDRConfig(
        wavevectors=(x_mode, y_mode),
        diffusion=(0.01, 0.01, 0.01),
        damping=0.05,
        feedback_gain=0.25,
        max_feedback_delta=0.02,
        min_history=4,
        max_history=6,
    )
    frames = [wave_field(8, phase=index * math.pi / 2) for index in range(7)]
    engine = FourierMarkovDiffusionResonanceEngine(frames[0], config)

    first = engine.step(frames[1])
    second = engine.step(frames[2])
    third = engine.step(frames[3])

    assert first.field == frames[1]
    assert second.field == frames[2]
    assert third.field != frames[3]
    assert third.report.sample_count == 4

    for frame in frames[4:]:
        engine.step(frame)
    assert len(engine.history) == config.max_history


def test_engine_projects_values_even_before_resonance_feedback_activates():
    x_mode, y_mode, _ = axis_wavevectors(8)
    config = FMDRConfig(
        wavevectors=(x_mode, y_mode),
        value_min=-0.25,
        value_max=0.25,
        min_history=4,
    )
    engine = FourierMarkovDiffusionResonanceEngine({(0, 0, 0): 2.0}, config)

    assert engine.state.field[(0, 0, 0)] == pytest.approx(0.25)
    state = engine.step({(0, 0, 0): -2.0})
    assert state.field[(0, 0, 0)] == pytest.approx(-0.25)


def test_validator_rejection_rolls_back_state_and_history_atomically():
    x_mode, y_mode, _ = axis_wavevectors(8)
    config = FMDRConfig(
        wavevectors=(x_mode, y_mode),
        min_history=4,
        max_history=8,
    )
    frames = [wave_field(8, phase=index * math.pi / 2) for index in range(4)]
    engine = FourierMarkovDiffusionResonanceEngine(frames[0], config)
    engine.step(frames[1])
    engine.step(frames[2])
    previous_state = engine.state
    previous_history = engine.history

    with pytest.raises(RuntimeError, match="validator"):
        engine.step(frames[3], validator=lambda _candidate, _report: False)

    assert engine.state is previous_state
    assert engine.history == previous_history


def test_configuration_rejects_zero_mode_and_invalid_history_window():
    with pytest.raises(ValueError, match="zero wavevector"):
        FMDRConfig(wavevectors=((0.0, 0.0, 0.0),))

    with pytest.raises(ValueError, match="min_history"):
        FMDRConfig(
            wavevectors=((1.0, 0.0, 0.0),),
            min_history=5,
            max_history=4,
        )
