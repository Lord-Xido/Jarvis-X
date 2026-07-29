"""Invariant tests for the sparse billion-address Dr Moagi field."""

import pytest

from jarvisx.dr_moagi_billion_field import BillionFieldConfig, SparseBillionField


def test_default_virtual_geometry_without_dense_allocation() -> None:
    field = SparseBillionField()

    assert field.virtual_cell_count == 1_000_000_000
    assert field.active_cell_count == 0
    assert field.padded_block_count == 32_768
    assert field.estimate_dense_state_bytes(32) == 32_000_000_000


def test_address_and_coordinate_are_exact_inverses() -> None:
    field = SparseBillionField()
    coordinates = (
        (0, 0, 0),
        (999, 0, 0),
        (0, 999, 0),
        (0, 0, 999),
        (123, 456, 789),
        (999, 999, 999),
    )

    for coordinate in coordinates:
        assert field.coordinate(field.address(coordinate)) == coordinate

    assert field.address((999, 999, 999)) == 999_999_999
    assert field.block_address((999, 999, 999)) == (31, 31, 31)


def test_sparse_transaction_is_deterministic() -> None:
    observations = {
        (500, 500, 500): 1.0,
        (501, 500, 500): 0.5,
        (500, 501, 500): -0.25,
    }
    first = SparseBillionField()
    second = SparseBillionField()

    first_metrics = first.run(4, observations)
    second_metrics = second.run(4, dict(reversed(tuple(observations.items()))))

    assert first_metrics == second_metrics
    assert tuple(first.iter_active()) == tuple(second.iter_active())
    assert first_metrics.journal_digest == second_metrics.journal_digest


def test_latents_remain_in_signed_three_bit_range() -> None:
    field = SparseBillionField()
    metrics = field.step(
        {
            (0, 0, 0): -1.0,
            (1, 0, 0): -0.5,
            (2, 0, 0): 0.0,
            (3, 0, 0): 0.5,
            (4, 0, 0): 1.0,
        }
    )

    assert metrics.active_cells == 5
    assert metrics.virtual_cells == 1_000_000_000
    assert metrics.active_ratio == pytest.approx(5 / 1_000_000_000)
    assert 0.0 < metrics.coherence <= 1.0
    assert all(-4 <= latent <= 3 for latent in field.encoded_snapshot().values())


def test_invalid_candidate_rolls_back_transactionally() -> None:
    config = BillionFieldConfig(residual_threshold=0.0)
    field = SparseBillionField(config)

    metrics = field.step({(10, 10, 10): 1.0})
    state = field.state((10, 10, 10))

    assert state.valid is False
    assert state.committed == 0.0
    assert metrics.valid_cells == 0
    assert metrics.rejected_cells == 1


def test_active_cell_budget_is_enforced_before_commit() -> None:
    field = SparseBillionField(BillionFieldConfig(max_active_cells=2))
    field.step({(0, 0, 0): 1.0, (1, 0, 0): 0.5})

    with pytest.raises(RuntimeError, match="active-cell budget exceeded"):
        field.step({(2, 0, 0): 0.25})


def test_input_validation_rejects_non_finite_values_and_bad_coordinates() -> None:
    field = SparseBillionField()

    with pytest.raises(ValueError, match="finite"):
        field.step({(0, 0, 0): float("nan")})

    with pytest.raises(ValueError, match="outside"):
        field.activate((1000, 0, 0), 1.0)

    with pytest.raises(TypeError, match="three-integer tuple"):
        field.activate((0, 0), 1.0)  # type: ignore[arg-type]


def test_journal_digest_advances_with_each_committed_cycle() -> None:
    field = SparseBillionField()
    initial = field.metrics().journal_digest
    first = field.step({(8, 8, 8): 0.75}).journal_digest
    second = field.step().journal_digest

    assert len(initial) == len(first) == len(second) == 64
    assert len({initial, first, second}) == 3
