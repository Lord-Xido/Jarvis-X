"""Tests for fail-closed arithmetic semantics and full 3D voxel witnesses."""
import random

import pytest

from jarvisx.gmral_omega import AffineBitOperator, rotation_permutation
from jarvisx.gmral_verified_3d import (
    BitCube3D, Grid3D, Operation3D, VerificationError, execute_atomic,
    execute_verified, verify_transition,
)


@pytest.mark.parametrize("side", [2, 4, 8, 16])
def test_every_address_has_exact_geometric_inverse(side):
    grid = Grid3D(side)
    addresses = set()
    for i in range(grid.width):
        xyz = grid.coordinates(i)
        assert all(0 <= coordinate < side for coordinate in xyz)
        assert grid.address(xyz) == i
        addresses.add(grid.address(xyz))
    assert addresses == set(range(side**3))


def test_truth_tables_for_all_3d_bitwise_operations():
    grid = Grid3D(2)
    for a in range(4):
        for b in range(4):
            left, right = BitCube3D(grid, a), BitCube3D(grid, b)
            for name, expected in [
                ("xor", a ^ b), ("and", a & b), ("or", a | b)
            ]:
                t = execute_verified(left, Operation3D(name, right=right))
                assert t.state.bits == expected
                assert len(t.witness) == grid.width
                assert t.receipt.voxel_count == grid.width
                assert all(t.state.at(w.xyz) == w.output_bit for w in t.witness)
            inversion = execute_verified(left, Operation3D("not"))
            assert inversion.state.bits == a ^ 0xFF


@pytest.mark.parametrize("side", [2, 4])
def test_full_3d_addition_and_carry_invariant(side):
    grid = Grid3D(side)
    rng = random.Random(891)
    samples = [(0, 0), (5, 3), (2**grid.width - 1, 1)] + [
        (rng.getrandbits(grid.width), rng.getrandbits(grid.width)) for _ in range(25)
    ]
    for a, b in samples:
        t = execute_verified(
            BitCube3D(grid, a), Operation3D("add", right=BitCube3D(grid, b))
        )
        assert t.state.bits + (t.receipt.carry_out << grid.width) == a + b
        for w in t.witness:
            assert w.right_bit is not None
            assert w.carry_in is not None
            assert w.carry_out is not None
            assert w.left_bit + w.right_bit + w.carry_in == w.output_bit + 2*w.carry_out


def test_geometric_operator_exact_composition_and_voxel_source():
    grid = Grid3D(4)
    mask = 1 << grid.address((0, 0, 0))
    op = AffineBitOperator(rotation_permutation(4), mask)
    for bits in [0, 1, (1 << grid.width) - 1, 0x76543210FEEDABCD]:
        t = execute_verified(BitCube3D(grid, bits), Operation3D("affine", operator=op))
        assert t.state.bits == op.apply(bits)
        assert all(w.source_xyz is not None for w in t.witness)
        for w in t.witness:
            source_bit = (bits >> grid.address(w.source_xyz)) & 1
            dst_mask = (mask >> grid.address(w.xyz)) & 1
            assert w.output_bit == source_bit ^ dst_mask


def test_mega_exponential_exact_with_cycle_reference():
    grid = Grid3D(4)
    op = AffineBitOperator(rotation_permutation(4), 1)
    for bits in [0, 1, 0xFFFF0000ABCDEF12]:
        a = BitCube3D(grid, bits)
        req = Operation3D("evolve", operator=op, base=10**6, exponent=10**6)
        t = execute_verified(a, req)
        assert t.receipt.exact_period == 24
        assert t.receipt.exponent_residue == 16
        assert t.state.bits == op.power(16).apply(bits)
        assert len(t.witness) == grid.width


def test_cycle_reference_matches_literal_for_nontrivial_mask_and_exponents():
    grid = Grid3D(2)
    op = AffineBitOperator(rotation_permutation(2), 0b10010101)
    for n in range(60):
        t = execute_verified(BitCube3D(grid, 0xA7), Operation3D("evolve", operator=op, base=n, exponent=1))
        expected = 0xA7
        for _ in range(n):
            expected = op.apply(expected)
        assert t.state.bits == expected


def test_digest_reproducible_and_not_a_formal_proof():
    grid = Grid3D(2)
    left, right = BitCube3D(grid, 22), BitCube3D(grid, 33)
    req = Operation3D("xor", right=right)
    a = execute_verified(left, req)
    b = execute_verified(left, req)
    assert a.receipt.witness_digest == b.receipt.witness_digest
    assert a.receipt.proof_level == "exact bounded runtime oracle + per-voxel witness"


def test_mismatched_candidate_fail_closed():
    grid = Grid3D(2)
    a = BitCube3D(grid, 5)
    b = BitCube3D(grid, 3)
    with pytest.raises(VerificationError, match="candidate differs"):
        verify_transition(a, Operation3D("add", right=b), 9, carry_out=0)
    with pytest.raises(VerificationError):
        verify_transition(a, Operation3D("add", right=b), 8, carry_out=1)


def test_refuse_unadmitted_and_malformed_operations():
    grid = Grid3D(2)
    with pytest.raises(ValueError, match="not on the admitted"):
        Operation3D("invented")
    with pytest.raises(ValueError, match="missing or extraneous"):
        Operation3D("not", right=BitCube3D(grid, 1))
    with pytest.raises(ValueError, match="same grid"):
        execute_verified(BitCube3D(grid, 1), Operation3D("xor", right=BitCube3D(Grid3D(4), 1)))
    with pytest.raises(ValueError, match="width"):
        execute_verified(BitCube3D(grid, 1), Operation3D("affine", operator=AffineBitOperator(rotation_permutation(4), 0)))


def test_atomic_transaction_commits_only_verified_chain():
    grid = Grid3D(2)
    original = BitCube3D(grid, 5)
    steps = (Operation3D("add", right=BitCube3D(grid, 3)), Operation3D("xor", right=BitCube3D(grid, 2)))
    final, receipts = execute_atomic(original, steps)
    assert final.bits == 10
    assert original.bits == 5
    assert len(receipts) == 2
    with pytest.raises(ValueError):
        execute_atomic(original, steps + (Operation3D("xor", right=BitCube3D(Grid3D(4), 1)),))
