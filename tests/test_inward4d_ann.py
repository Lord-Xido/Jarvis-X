from __future__ import annotations

import math
from dataclasses import replace

import pytest

from jarvisx.inward4d_ann import (
    Inward4DANN,
    Inward4DConfig,
    coordinate_to_index,
    index_to_coordinate,
)


def signal() -> list[float]:
    return [
        0.30 * math.sin(index * 0.013) + 0.10 * math.cos(index * 0.031) for index in range(1000)
    ]


def test_default_index_mapping_is_a_complete_bijection():
    assert coordinate_to_index((2, 3, 4)) == 234
    assert index_to_coordinate(234) == (2, 3, 4)
    assert {coordinate_to_index(index_to_coordinate(index)) for index in range(1000)} == set(
        range(1000)
    )


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: coordinate_to_index((10, 0, 0)), ValueError),
        (lambda: coordinate_to_index((True, 0, 0)), TypeError),
        (lambda: index_to_coordinate(1000), ValueError),
        (lambda: index_to_coordinate(True), TypeError),
    ],
)
def test_index_mapping_rejects_invalid_addresses(call, error):
    with pytest.raises(error):
        call()


def test_full_fold_has_exact_periodic_topology_and_finite_r4_geometry():
    engine = Inward4DANN()
    summary = engine.arithmetic_summary()

    assert summary["nodes"] == 1000
    assert summary["flat_synapses"] == 2700
    assert summary["periodic_synapses"] == 3000
    assert summary["active_synapses"] == 3000
    assert summary["active_wrap_synapses"] == 300
    assert min(engine.degrees) == max(engine.degrees) == 6
    assert max(edge.distance4d for edge in engine.edge_geometries) <= 5.5
    assert all(
        len(point) == 4 and all(math.isfinite(value) for value in point)
        for point in engine.positions
    )


def test_worked_node_234_position_matches_closed_form():
    engine = Inward4DANN()
    point = engine.positions[234]

    assert point == pytest.approx(
        (1.0881867157809302, 3.3490943405317752, -2.436499466930409, 1.770220482187334)
    )


def test_zero_fold_is_a_flat_open_boundary_cube():
    engine = Inward4DANN(Inward4DConfig(fold_factor=0.0))

    assert engine.active_synapse_count == 2700
    assert engine.active_wrap_synapse_count == 0
    assert min(engine.degrees) == 3
    assert max(engine.degrees) == 6
    assert all(point[3] == pytest.approx(0.0) for point in engine.positions)


def test_proximity_radius_cannot_create_an_invalid_graph():
    with pytest.raises(ValueError, match="synapse|degree|disconnected"):
        Inward4DANN(Inward4DConfig(proximity_radius=0.01))


def test_forward_zero_is_an_exact_self_description_fixed_point():
    engine = Inward4DANN()
    zero = [0.0] * engine.node_count

    forward = engine.forward(zero)
    evaluation = engine.evaluate(zero)

    assert forward.latent == pytest.approx(zero)
    assert forward.reconstruction == pytest.approx(zero)
    assert evaluation.description_residual_rms == pytest.approx(0.0)
    assert evaluation.max_abs_residual == pytest.approx(0.0)
    assert evaluation.converged


def test_forward_validates_shape_type_and_finiteness():
    engine = Inward4DANN()

    with pytest.raises(ValueError, match="exactly 1000"):
        engine.forward([0.0] * 999)
    with pytest.raises(TypeError, match="numeric"):
        engine.forward([False] + [0.0] * 999)
    with pytest.raises(ValueError, match="finite"):
        engine.forward([math.inf] + [0.0] * 999)


def test_reverse_mode_edge_and_bias_gradients_match_finite_differences():
    engine = Inward4DANN()
    target = signal()
    gradient = engine.gradients(target)
    snapshot = engine.snapshot()
    epsilon = 1.0e-6

    edge_index = 1234
    edge_losses = []
    for direction in (1.0, -1.0):
        weights = list(snapshot.weights)
        weights[edge_index] += direction * epsilon
        engine.restore(replace(snapshot, weights=tuple(weights)))
        edge_losses.append(engine.evaluate(target).loss.total)
    edge_finite_difference = (edge_losses[0] - edge_losses[1]) / (2.0 * epsilon)
    assert gradient.edge_weights[edge_index] == pytest.approx(
        edge_finite_difference, rel=2.0e-5, abs=1.0e-11
    )

    engine.restore(snapshot)
    node_index = 234
    bias_losses = []
    for direction in (1.0, -1.0):
        decoder_bias = list(snapshot.decoder_bias)
        decoder_bias[node_index] += direction * epsilon
        engine.restore(replace(snapshot, decoder_bias=tuple(decoder_bias)))
        bias_losses.append(engine.evaluate(target).loss.total)
    bias_finite_difference = (bias_losses[0] - bias_losses[1]) / (2.0 * epsilon)
    assert gradient.decoder_bias[node_index] == pytest.approx(
        bias_finite_difference, rel=2.0e-5, abs=1.0e-11
    )
    engine.restore(snapshot)


def test_transactional_step_is_deterministic_and_non_regressing():
    target = signal()
    first = Inward4DANN()
    second = Inward4DANN()

    first_step = first.train_step(target)
    second_step = second.train_step(target)

    assert first_step == second_step
    assert first.snapshot() == second.snapshot()
    assert first_step.committed
    assert first_step.loss_after <= first_step.loss_before
    assert first_step.active_synapses == 3000
    assert first_step.pruned_synapses == 0


def test_validator_rejection_rolls_parameters_and_topology_back():
    engine = Inward4DANN()
    before = engine.snapshot()

    metrics = engine.train_step(signal(), validator=lambda candidate: False)
    after = engine.snapshot()

    assert not metrics.committed
    assert metrics.rejection_reason == "validator rejected candidate"
    assert metrics.learning_rate_used == 0.0
    assert after.epoch == before.epoch + 1
    assert after.weights == before.weights
    assert after.encoder_bias == before.encoder_bias
    assert after.decoder_bias == before.decoder_bias
    assert after.active_edges == before.active_edges


def test_pruning_is_symmetric_bounded_and_connectivity_preserving():
    engine = Inward4DANN(
        Inward4DConfig(
            base_weight=0.10,
            prune_interval=1,
            min_degree=4,
            edge_energy_weight=0.0,
            homeostasis_weight=0.0,
            bias_regularization_weight=0.0,
        )
    )

    metrics = engine.train_step([0.0] * engine.node_count)

    assert metrics.committed
    assert metrics.pruned_synapses > 0
    assert metrics.active_synapses == 3000 - metrics.pruned_synapses
    assert min(engine.degrees) >= 4


def test_restore_rejects_malformed_state_without_mutation():
    engine = Inward4DANN()
    before = engine.snapshot()
    malformed = replace(before, weights=before.weights[:-1])

    with pytest.raises(ValueError, match="weight count"):
        engine.restore(malformed)

    assert engine.snapshot() == before


def test_optimization_budget_is_finite_and_auditable():
    engine = Inward4DANN()
    report = engine.optimize(signal(), max_epochs=3)

    assert report.attempted_epochs == 3
    assert report.committed_epochs == 3
    assert len(report.history) == 3
    assert report.final.loss.total <= report.initial.loss.total
    assert report.final.description_residual_rms <= report.initial.description_residual_rms


def test_zero_signal_converges_without_claiming_an_optimization_epoch():
    engine = Inward4DANN()
    report = engine.optimize([0.0] * engine.node_count, max_epochs=10)

    assert report.converged
    assert report.attempted_epochs == 0
    assert report.committed_epochs == 0


@pytest.mark.parametrize(
    "config",
    [
        Inward4DConfig,
    ],
)
def test_config_rejects_unbounded_or_incoherent_values(config):
    with pytest.raises(ValueError):
        config(fold_factor=1.1)
    with pytest.raises(ValueError):
        config(decay=1.0)
    with pytest.raises(ValueError):
        config(prune_threshold=2.0)
    with pytest.raises(ValueError):
        config(min_degree=0)


def test_optimization_rejects_invalid_budget_and_tolerance():
    engine = Inward4DANN()
    zero = [0.0] * engine.node_count

    with pytest.raises(ValueError, match="max_epochs"):
        engine.optimize(zero, max_epochs=-1)
    with pytest.raises(ValueError, match="tolerance"):
        engine.optimize(zero, tolerance=0.0)
