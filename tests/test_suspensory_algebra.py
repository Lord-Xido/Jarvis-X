"""Contract tests: consideration is not authority."""

import math

import pytest

from jarvisx.suspensory_algebra import AffineHypothesis, SuspensoryAlgebra


P = (1, 2, 3)
Q = (9, 8, 7)


def test_two_competing_ideas_stay_suspended():
    algebra = SuspensoryAlgebra({P: 4.0})
    a = algebra.consider("linear", AffineHypothesis("2x+3", 2, 3))
    b = algebra.consider("quadratic_at_4", AffineHypothesis("x^2-3 at x=4", 4, -3))
    assert a.snapshot() == {P: 11.0}
    assert b.snapshot() == {P: 13.0}
    assert algebra.snapshot() == {P: 4.0}
    assert algebra.revision == 0
    assert algebra.commit("linear", expected_hash=a.candidate_hash).reason == "execution_gate_closed"
    assert algebra.snapshot() == {P: 4.0}


def test_arithmetic_branching_never_mutates_real_state():
    algebra = SuspensoryAlgebra({P: 4.0})
    first = algebra.consider("a", AffineHypothesis("f", 2, 3))
    second = algebra.consider("a_then_g", AffineHypothesis("g", 3, -1), parent_id="a")
    assert second.snapshot() == {P: 32.0}
    assert second.parent_id == first.idea_id
    assert second.base_state_hash == algebra.state_hash
    assert algebra.snapshot() == {P: 4.0}


def test_explicit_matching_commit_only():
    algebra = SuspensoryAlgebra({P: 4.0})
    idea = algebra.consider("linear", AffineHypothesis("f", 2, 3))
    assert not algebra.commit("linear", authorization=True, expected_hash="incorrect").committed
    assert algebra.snapshot() == {P: 4.0}
    decision = algebra.commit("linear", authorization=True, expected_hash=idea.candidate_hash)
    assert decision.committed
    assert decision.previous_revision == 0
    assert decision.current_revision == 1
    assert algebra.snapshot() == {P: 11.0}
    assert decision.state_hash == algebra.state_hash


def test_previously_valid_proposals_become_stale():
    algebra = SuspensoryAlgebra({P: 4.0})
    a = algebra.consider("a", AffineHypothesis("f", 2, 3))
    b = algebra.consider("b", AffineHypothesis("g", 1, 1))
    assert algebra.commit("a", authorization=True, expected_hash=a.candidate_hash).committed
    d = algebra.commit("b", authorization=True, expected_hash=b.candidate_hash)
    assert not d.committed
    assert d.reason == "stale_base_state"
    with pytest.raises(ValueError, match="stale"):
        algebra.consider("from_stale", AffineHypothesis("f", 1, 1), parent_id="b")
    assert algebra.snapshot() == {P: 11.0}


def test_candidate_snapshot_is_immutable_in_engine_and_copy_safe():
    initial = {P: 4.0}
    algebra = SuspensoryAlgebra(initial)
    idea = algebra.consider("a", AffineHypothesis("f", 2, 3))
    initial[P] = -100
    copy = idea.snapshot()
    copy[P] = -100
    current = algebra.snapshot()
    current[P] = -100
    assert idea.snapshot() == {P: 11.0}
    assert algebra.snapshot() == {P: 4.0}


def test_3d_sparse_support_remains_sparse_and_digest_order_is_stable():
    a = SuspensoryAlgebra({P: 4.0, Q: 1.5}, side=1000000)
    b = SuspensoryAlgebra({Q: 1.5, P: 4.0}, side=1000000)
    x = a.consider("x", AffineHypothesis("identity", 1, 0))
    y = b.consider("x", AffineHypothesis("identity", 1, 0))
    assert x.candidate_hash == y.candidate_hash
    assert x.delta_l2 == 0
    assert len(x.candidate) == 2
    assert a.snapshot() == {P: 4.0, Q: 1.5}


@pytest.mark.parametrize("coordinate", [(-1, 0, 0), (100, 0, 0), (0, 1), (1, True, 1)])
def test_bad_coordinates_rejected(coordinate):
    with pytest.raises(ValueError, match="coordinates"):
        SuspensoryAlgebra({coordinate: 1.0}, side=100)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_invalid_values_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        SuspensoryAlgebra({P: value})


def test_invalid_candidate_and_budget_do_not_pollute_ledger():
    algebra = SuspensoryAlgebra({P: 4.0}, max_abs_value=10.0, max_candidates=1)
    with pytest.raises(ValueError, match="bound"):
        algebra.consider("overflow", AffineHypothesis("large", 10, 10))
    assert algebra.snapshot() == {P: 4.0}
    good = algebra.consider("fits", AffineHypothesis("small", 1, 1))
    with pytest.raises(ValueError, match="budget"):
        algebra.consider("extra", AffineHypothesis("small", 1, 1))
    assert good.snapshot() == {P: 5.0}


def test_reject_duplicate_identifier_and_missing_parent():
    algebra = SuspensoryAlgebra({P: 1.0})
    algebra.consider("idea", AffineHypothesis("identity", 1, 0))
    with pytest.raises(ValueError, match="duplicate"):
        algebra.consider("idea", AffineHypothesis("identity", 1, 0))
    with pytest.raises(KeyError):
        algebra.consider("new", AffineHypothesis("identity", 1, 0), parent_id="missing")


def test_zero_support_and_exact_state_preservation():
    algebra = SuspensoryAlgebra({}, side=10)
    idea = algebra.consider("empty", AffineHypothesis("affine", 2, 100))
    assert idea.snapshot() == {}
    assert idea.delta_l2 == 0
    assert algebra.snapshot() == {}
