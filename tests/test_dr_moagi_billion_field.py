"""Invariant tests for the sparse billion-address Dr Moagi field."""

import copy

import pytest

from jarvisx.dr_moagi_billion_field import BillionFieldConfig, SparseBillionField


def test_default_virtual_geometry_without_dense_allocation() -> None:
    field = SparseBillionField()

    assert field.virtual_cell_count == 1_000_000_000
    assert field.active_cell_count == 0
    assert field.padded_block_count == 32_768
    assert field.estimate_dense_state_bytes(32) == 32_000_000_000


def test_address_coordinate_and_type_contracts() -> None:
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
    with pytest.raises(TypeError, match="components must be integers"):
        field.address((True, 0, 0))
    with pytest.raises(TypeError, match="address must be an integer"):
        field.coordinate(True)


def test_sparse_transaction_is_deterministic_with_controls() -> None:
    observations = {
        (500, 500, 500): 1.0,
        (501, 500, 500): 0.5,
        (500, 501, 500): -0.25,
    }
    controls = {(500, 500, 500): -0.02}
    first = SparseBillionField()
    second = SparseBillionField()

    first_metrics = first.run(4, observations, controls)
    second_metrics = second.run(
        4,
        dict(reversed(tuple(observations.items()))),
        controls,
    )

    assert first_metrics == second_metrics
    assert tuple(first.iter_active()) == tuple(second.iter_active())
    assert first_metrics.journal_digest == second_metrics.journal_digest
    assert first_metrics.state_digest == second_metrics.state_digest


def test_q3_uses_all_eight_codes_and_decodes_distinctly() -> None:
    config = BillionFieldConfig(
        context_gain=0.0,
        reasoning_gain=1.0,
        coupling_gain=0.0,
        omega_gain=0.0,
    )
    field = SparseBillionField(config)
    values = (-1.0, -0.75, -0.5, -0.25, 0.0, 1 / 3, 2 / 3, 1.0)

    field.step({(index, 0, 0): value for index, value in enumerate(values)})

    assert list(field.encoded_snapshot().values()) == [-4, -3, -2, -1, 0, 1, 2, 3]
    assert [field.state((index, 0, 0)).decoded for index in range(8)] == pytest.approx(
        values
    )


def test_invalid_candidate_rolls_back_persistent_state_atomically() -> None:
    config = BillionFieldConfig(residual_threshold=0.0)
    field = SparseBillionField(config)

    metrics = field.step({(10, 10, 10): 1.0})
    state = field.state((10, 10, 10))

    assert state.valid is False
    assert state.committed == 0.0
    assert state.omega == 0.0
    assert state.residual != 0.0
    assert metrics.valid_cells == 0
    assert metrics.rejected_cells == 1


def test_active_cell_budget_failure_leaves_state_unchanged() -> None:
    field = SparseBillionField(BillionFieldConfig(max_active_cells=2))
    field.step({(0, 0, 0): 1.0, (1, 0, 0): 0.5})
    before = field.checkpoint()

    with pytest.raises(RuntimeError, match="active-cell budget exceeded"):
        field.step({(2, 0, 0): 0.25})

    assert field.checkpoint() == before


def test_halo_expands_bounded_support_in_six_neighbour_geometry() -> None:
    config = BillionFieldConfig(side=5, halo_depth=1, max_active_cells=20)
    field = SparseBillionField(config)

    field.step({(2, 2, 2): 1.0})

    assert field.active_cell_count == 7
    assert set(field.active_coordinates()) == {
        (2, 2, 2),
        (1, 2, 2),
        (3, 2, 2),
        (2, 1, 2),
        (2, 3, 2),
        (2, 2, 1),
        (2, 2, 3),
    }


def test_control_is_transient_while_observation_persists() -> None:
    config = BillionFieldConfig(
        context_gain=0.0,
        reasoning_steps=1,
        reasoning_gain=0.5,
        coupling_gain=0.0,
        omega_gain=0.0,
    )
    field = SparseBillionField(config)

    field.step({(1, 1, 1): 0.0}, {(1, 1, 1): 0.5})
    first = field.state((1, 1, 1)).committed
    field.step()
    second = field.state((1, 1, 1)).committed

    assert first == pytest.approx(0.5)
    assert second == pytest.approx(0.25)
    assert field.state((1, 1, 1)).observed == 0.0


def test_checkpoint_round_trip_and_tamper_detection() -> None:
    field = SparseBillionField(
        BillionFieldConfig(halo_depth=1, max_active_cells=50)
    )
    field.run(2, {(5, 5, 5): 0.75})
    checkpoint = field.checkpoint()

    restored = SparseBillionField.from_checkpoint(checkpoint)

    assert restored.config == field.config
    assert restored.metrics() == field.metrics()
    assert tuple(restored.iter_active()) == tuple(field.iter_active())

    tampered_state = copy.deepcopy(checkpoint)
    tampered_state["cells"][0]["state"]["committed"] = 0.123
    with pytest.raises(ValueError, match="digest mismatch"):
        SparseBillionField.from_checkpoint(tampered_state)

    tampered_journal = copy.deepcopy(checkpoint)
    tampered_journal["journal_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        SparseBillionField.from_checkpoint(tampered_journal)


def test_configuration_rejects_nonfinite_and_noncanonical_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        BillionFieldConfig(coupling_gain=float("nan"))
    with pytest.raises(ValueError, match="canonical signed Q3"):
        BillionFieldConfig(latent_min=-3)
    with pytest.raises(ValueError, match="canonical field range"):
        BillionFieldConfig(value_min=-2.0)
    with pytest.raises(TypeError, match="side must be an integer"):
        BillionFieldConfig(side=True)


def test_input_validation_rejects_nonfinite_values_and_bad_coordinates() -> None:
    field = SparseBillionField()

    with pytest.raises(ValueError, match="finite"):
        field.step({(0, 0, 0): float("nan")})
    with pytest.raises(ValueError, match="outside"):
        field.activate((1000, 0, 0), 1.0)
    with pytest.raises(TypeError, match="three-integer tuple"):
        field.activate((0, 0), 1.0)  # type: ignore[arg-type]


def test_pruning_removes_quiescent_halo_but_protects_explicit_inputs() -> None:
    config = BillionFieldConfig(
        side=5,
        halo_depth=1,
        prune_epsilon=1.0e-12,
        max_active_cells=20,
    )
    field = SparseBillionField(config)

    field.step({(2, 2, 2): 0.0})

    assert field.active_coordinates() == ((2, 2, 2),)


def test_journal_and_state_digests_advance_with_each_cycle() -> None:
    field = SparseBillionField()
    initial = field.metrics()
    first = field.step({(8, 8, 8): 0.75})
    second = field.step()

    assert len(initial.journal_digest) == len(first.journal_digest) == 64
    assert len(initial.state_digest) == len(first.state_digest) == 64
    assert len({initial.journal_digest, first.journal_digest, second.journal_digest}) == 3
    assert len({initial.state_digest, first.state_digest, second.state_digest}) == 3
