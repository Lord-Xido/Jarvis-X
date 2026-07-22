import pytest

from jarvisx.ann30d_safe import SafeANNProcessor30D


def test_safe_processor_rejects_invalid_input():
    processor = SafeANNProcessor30D(max_input_length=2)
    with pytest.raises(ValueError):
        processor.run([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        processor.run([float("inf")])


def test_safe_processor_resets_run_state_and_preserves_memory():
    processor = SafeANNProcessor30D()
    first = processor.run([1.0, 0.0], target=1.0)
    first_hash = processor.state_hash()
    second = processor.run([1.0, 0.0], target=1.0)
    assert first.coordinate == second.coordinate
    assert second.cycles == 10
    assert processor.state_hash() != first_hash


def test_active_cell_quota_rolls_back_failed_placement():
    processor = SafeANNProcessor30D(max_active_cells=1)
    processor.run([1.0], target=0.0)
    assert processor.field.active_cells == 1
    with pytest.raises(MemoryError):
        processor.run([-100.0], target=0.0)
    assert processor.field.active_cells == 1
