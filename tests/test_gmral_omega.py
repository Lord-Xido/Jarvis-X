"""Invariant, integration and adversarial tests for GMRAL-Omega."""
from __future__ import annotations

import itertools
import random

import pytest

from jarvisx.bitwise_hyperscale_ann import morton60_encode
from jarvisx.gmral_omega import (
    AffineBitOperator,
    evaluate_exponential_iterations,
    morton3d,
    parallel_prefix_add,
    permutation_order,
    rotation_permutation,
    seed_cube,
)


def _literal(op: AffineBitOperator, state: int, steps: int) -> int:
    for _ in range(steps):
        state = op.apply(state)
    return state


@pytest.mark.parametrize("side", [2, 4, 8])
def test_morton3d_agrees_with_canonical_source_order(side: int) -> None:
    for x, y, z in itertools.product(range(side), repeat=3):
        assert morton3d(x, y, z, side=side) == morton60_encode(x, y, z)
    assert sorted(rotation_permutation(side)) == list(range(side**3))


@pytest.mark.parametrize("side", [2, 4])
def test_geometric_affine_matches_literal_cycles(side: int) -> None:
    op = AffineBitOperator(rotation_permutation(side), 1)
    initial = seed_cube(side)
    assert _literal(op, initial, op.exact_period()) == initial
    for steps in range(100):
        assert op.power(steps).apply(initial) == _literal(op, initial, steps)


def test_operator_composition_order_and_noncommutativity() -> None:
    shift = AffineBitOperator((1, 2, 3, 0), 0b0001)
    flip = AffineBitOperator((0, 1, 2, 3), 0b0110)
    for state in range(16):
        assert shift.then(flip).apply(state) == flip.apply(shift.apply(state))
        assert flip.then(shift).apply(state) == shift.apply(flip.apply(state))
    assert shift.then(flip) != flip.then(shift)


def test_exact_period_is_minimal_for_all_small_masks() -> None:
    permutation = (1, 2, 3, 0)
    assert permutation_order(permutation) == 4
    identity = AffineBitOperator.identity(4)
    for mask in range(16):
        operator = AffineBitOperator(permutation, mask)
        period = operator.exact_period()
        assert period in (4, 8)
        assert operator.power(period) == identity
        assert all(operator.power(k) != identity for k in range(1, period))


def test_giant_exponent_evaluated_without_constructing_it() -> None:
    op = AffineBitOperator(rotation_permutation(4), 1)
    initial = seed_cube(4)
    receipt = evaluate_exponential_iterations(op, initial)
    assert receipt.width == 64
    assert receipt.operator_period == 24
    assert receipt.exponent_residue == 16
    assert receipt.result == _literal(op, initial, 16)
    assert receipt.result == op.power(16).apply(initial)


def test_exponential_edge_cases() -> None:
    op = AffineBitOperator((1, 0), 0b01)
    for base, exp in [(0, 0), (0, 3), (1, 10**6), (2, 0), (3, 7)]:
        expected = _literal(op, 2, base**exp if exp < 8 else 1)
        receipt = evaluate_exponential_iterations(op, 2, base=base, exponent=exp)
        if exp < 8:
            assert receipt.result == expected
        else:
            assert receipt.result == op.power(1).apply(2)


def test_prefix_adder_exhaustive_four_bit() -> None:
    for a, b in itertools.product(range(16), repeat=2):
        low, carry, rounds = parallel_prefix_add(a, b, width=4)
        assert low + (carry << 4) == a + b
        assert rounds == 2


def test_prefix_adder_randomized_and_boundaries() -> None:
    rng = random.Random(0xC0FFEE)
    for width in [1, 2, 3, 7, 8, 32, 64, 65]:
        for a, b in (
            (0, 0),
            ((1 << width) - 1, 1),
            ((1 << width) - 1, (1 << width) - 1),
            *[(rng.getrandbits(width), rng.getrandbits(width)) for _ in range(20)],
        ):
            result, carry, rounds = parallel_prefix_add(a, b, width=width)
            assert result + (carry << width) == a + b
            assert rounds == (width - 1).bit_length()


@pytest.mark.parametrize(
    "permutation,mask",
    [((0, 0), 0), ((1, 2), 0), ((), 0), ((0, 1), -1), ((0, 1), 4)],
)
def test_malformed_operator_rejected(permutation: tuple[int, ...], mask: int) -> None:
    with pytest.raises(ValueError):
        AffineBitOperator(permutation, mask)


@pytest.mark.parametrize("coordinate", [(-1, 0, 0), (0, 4, 0), (True, 0, 0)])
def test_bad_coordinates_rejected(coordinate: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        morton3d(*coordinate)


def test_malformed_inputs_rejected() -> None:
    op = AffineBitOperator((1, 0), 1)
    with pytest.raises(ValueError):
        op.power(-1)
    with pytest.raises(ValueError):
        op.apply(4)
    with pytest.raises(ValueError):
        evaluate_exponential_iterations(op, 2, base=-1)
    with pytest.raises(ValueError):
        evaluate_exponential_iterations(op, -1)
    with pytest.raises(ValueError):
        op.then(AffineBitOperator.identity(3))
    with pytest.raises(ValueError):
        parallel_prefix_add(-1, 0, width=4)
    with pytest.raises(ValueError):
        parallel_prefix_add(2, 4, width=2)
    with pytest.raises(ValueError):
        parallel_prefix_add(0, 0, width=0)
