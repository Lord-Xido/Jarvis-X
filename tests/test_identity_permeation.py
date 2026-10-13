from __future__ import annotations

import math

import pytest

from jarvisx.identity_permeation import (
    CANONICAL_CELLS,
    IdentityPermeationVerifier,
    TorusLattice3D,
    canonical_field,
    identity_permeation,
    verify_canonical_identity,
)


def test_canonical_lattice_has_exact_11x6x4_cardinality() -> None:
    lattice = TorusLattice3D()
    assert lattice.shape == (11, 6, 4)
    assert lattice.cells == CANONICAL_CELLS == 264


def test_modular_wrapping_closes_all_three_torus_axes() -> None:
    lattice = TorusLattice3D()
    assert lattice.wrap((11, 0, 0)) == (0, 0, 0)
    assert lattice.wrap((0, 6, 0)) == (0, 0, 0)
    assert lattice.wrap((0, 0, 4)) == (0, 0, 0)
    assert lattice.wrap((-1, -1, -1)) == (10, 5, 3)
    assert lattice.periodic_closure_holds()


def test_identity_operator_is_idempotent_and_pointwise_fixed() -> None:
    receipt = verify_canonical_identity()
    assert receipt.permeates
    assert receipt.cells_checked == 264
    assert receipt.idempotence_residual == pytest.approx(0.0)
    assert receipt.fixed_point_residual == pytest.approx(0.0)
    assert receipt.identity_defect_gradient_residual == pytest.approx(0.0)
    assert receipt.circulation_residual == pytest.approx(0.0)
    assert receipt.periodic_closure


def test_idempotence_alone_does_not_imply_identity_permeation() -> None:
    verifier = IdentityPermeationVerifier()
    field = canonical_field()

    def zero_projection(source):
        return {coordinate: 0.0 for coordinate in source}

    receipt = verifier.verify(zero_projection, field)

    assert receipt.idempotence_residual == pytest.approx(0.0)
    assert receipt.fixed_point_residual > 0.0
    assert not receipt.permeates


def test_all_fundamental_cycle_sums_are_preserved_by_identity() -> None:
    lattice = TorusLattice3D()
    field = canonical_field()
    after = identity_permeation(field)

    for y in range(6):
        for z in range(4):
            base = (0, y, z)
            assert lattice.fundamental_cycle_sum(after, 0, base) == pytest.approx(
                lattice.fundamental_cycle_sum(field, 0, base)
            )
    for x in range(11):
        for z in range(4):
            base = (x, 0, z)
            assert lattice.fundamental_cycle_sum(after, 1, base) == pytest.approx(
                lattice.fundamental_cycle_sum(field, 1, base)
            )
    for x in range(11):
        for y in range(6):
            base = (x, y, 0)
            assert lattice.fundamental_cycle_sum(after, 2, base) == pytest.approx(
                lattice.fundamental_cycle_sum(field, 2, base)
            )


def test_verifier_rejects_incomplete_and_nonfinite_fields() -> None:
    verifier = IdentityPermeationVerifier()
    field = canonical_field()

    incomplete = dict(field)
    incomplete.pop((0, 0, 0))
    with pytest.raises(ValueError, match="264 coordinates"):
        verifier.verify(identity_permeation, incomplete)

    nonfinite = dict(field)
    nonfinite[(0, 0, 0)] = math.inf
    with pytest.raises(ValueError, match="finite"):
        verifier.verify(identity_permeation, nonfinite)


def test_geometry_is_locked_to_user_supplied_11x6x4_contract() -> None:
    with pytest.raises(ValueError, match="locked"):
        TorusLattice3D((10, 6, 4))
