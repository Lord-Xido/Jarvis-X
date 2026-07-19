import math

from jarvisx.geometric_memory import GeometricConfig, VisualMemoryANN, Volume3D, make_demo_volume


def test_volume_index_and_mse():
    volume = Volume3D.from_function((2, 2, 2), lambda z, y, x: z + y + x)
    assert volume.at(1, 1, 1) == 3.0
    assert volume.mse(volume) == 0.0


def test_permeation_is_deterministic_from_equal_state():
    volume = make_demo_volume(8)
    config = GeometricConfig(
        latent_shape=(2, 2, 2),
        refinement_steps=2,
        max_candidate_steps=3,
    )
    left = VisualMemoryANN(config).permeate(volume)
    right = VisualMemoryANN(config).permeate(volume)

    assert left.reconstruction.values == right.reconstruction.values
    assert left.selected == right.selected
    assert left.trace == right.trace


def test_permeation_shapes_trace_and_bounds():
    volume = make_demo_volume(8)
    config = GeometricConfig(latent_shape=(2, 2, 2), channels=6, refinement_steps=3)
    result = VisualMemoryANN(config).permeate(volume, auto_optimize=False)

    assert result.reconstruction.shape == volume.shape
    assert result.latent.shape == config.latent_shape
    assert result.latent.channels == config.channels
    assert len(result.trace) == config.refinement_steps
    assert all(0.0 <= value <= 1.0 for value in result.reconstruction.values)
    assert math.isfinite(result.selected.objective)


def test_auto_optimization_is_bounded_and_journaled():
    volume = make_demo_volume(8)
    engine = VisualMemoryANN(
        GeometricConfig(
            latent_shape=(2, 2, 2),
            refinement_steps=2,
            max_candidate_steps=3,
        )
    )
    result = engine.permeate(volume, auto_optimize=True)

    assert 1 <= result.candidate_count <= 5
    assert engine.config.refinement_steps <= engine.config.max_candidate_steps
    assert len(engine.journal) == 1
    assert result.selected.objective <= result.baseline.objective + 1e-15
