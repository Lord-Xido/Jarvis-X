import math

import pytest

from jarvisx.ann30d import (
    DIMENSIONS,
    SparseField30D,
    VirtualANNProcessor30D,
    flatten_coordinate,
    latent_to_coordinate,
    quantize_q3,
    quantize_vector30,
)


def test_signed_three_bit_quantizer_and_30d_coordinate():
    assert quantize_q3(-99) == -4
    assert quantize_q3(99) == 3
    latent = quantize_vector30([0.6] * DIMENSIONS)
    coordinate = latent_to_coordinate(latent)
    assert latent == (1,) * DIMENSIONS
    assert coordinate == (5,) * DIMENSIONS
    assert flatten_coordinate(coordinate) >= 0


def test_sparse_field_does_not_allocate_dense_8_power_30_volume():
    field = SparseField30D(side=8)
    assert field.theoretical_cells == 8 ** 30
    assert field.active_cells == 0
    field.deposit((4,) * DIMENSIONS, 1.0)
    assert field.active_cells == 1
    assert field.theoretical_cells > 10 ** 27


def test_default_bytecode_cycle_updates_prediction_memory_and_output():
    processor = VirtualANNProcessor30D()
    snapshot = processor.run([0.8, -0.3, 0.5, 1.0], target=0.8)

    assert snapshot.dimensions == 30
    assert snapshot.active_cells == 1
    assert snapshot.coordinate is not None and len(snapshot.coordinate) == 30
    assert snapshot.latent is not None and len(snapshot.latent) == 30
    assert len(snapshot.output) == 4
    assert snapshot.halted is True
    assert snapshot.cycles == 10
    assert math.isfinite(snapshot.prediction)
    assert snapshot.memory == pytest.approx(0.25 * snapshot.residual)


def test_processor_is_deterministic_for_equal_initial_state():
    first = VirtualANNProcessor30D().run([1.0, 2.0, 3.0], target=0.25)
    second = VirtualANNProcessor30D().run([1.0, 2.0, 3.0], target=0.25)
    assert first.coordinate == second.coordinate
    assert first.output == pytest.approx(second.output)
    assert first.prediction == pytest.approx(second.prediction)


def test_dimension_validation_rejects_non_30d_latent_vectors():
    with pytest.raises(ValueError):
        latent_to_coordinate([0, 1, 2])
