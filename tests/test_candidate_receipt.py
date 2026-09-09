from __future__ import annotations

import pytest

from jarvisx.candidate_receipt import (
    VerificationVector,
    build_candidate_receipt,
    hash_sparse_field,
)


def test_sparse_field_hash_is_coordinate_order_independent():
    left = {(2, 1): (3.0, 4.0), (0, 0): (1.0, 2.0)}
    right = {(0, 0): (1.0, 2.0), (2, 1): (3.0, 4.0)}

    assert hash_sparse_field(left) == hash_sparse_field(right)


def test_commit_receipt_requires_candidate_to_be_published():
    parent = hash_sparse_field({(0, 0): (1.0,)})
    candidate = hash_sparse_field({(0, 0): (2.0,)})

    with pytest.raises(ValueError, match="publish the candidate"):
        build_candidate_receipt(
            subsystem="test",
            operator="candidate",
            state_scope="surface",
            parent_state_hash=parent,
            candidate_state_hash=candidate,
            after_state_hash=parent,
            objective_before=1.0,
            candidate_objective=0.5,
            objective_after=0.5,
            metrics={},
            hard_constraints={"finite": True},
            verification=VerificationVector("PASS", "PASS", "NOT_APPLICABLE", "PASS"),
            committed=True,
        )


def test_rollback_receipt_is_deterministic_and_preserves_parent():
    parent = hash_sparse_field({(0, 0): (1.0,)})
    candidate = hash_sparse_field({(0, 0): (2.0,)})
    kwargs = dict(
        subsystem="test",
        operator="candidate",
        state_scope="surface",
        parent_state_hash=parent,
        candidate_state_hash=candidate,
        after_state_hash=parent,
        objective_before=0.0,
        candidate_objective=1.0,
        objective_after=0.0,
        metrics={"mse": 1.0},
        hard_constraints={"finite": True, "quality": False},
        verification=VerificationVector("PASS", "FAIL", "NOT_APPLICABLE", "FAIL"),
        committed=False,
        rejection_reason="quality gate",
    )

    first = build_candidate_receipt(**kwargs)
    second = build_candidate_receipt(**kwargs)

    assert first.receipt_hash == second.receipt_hash
    assert first.after_state_hash == parent
    assert not first.committed
