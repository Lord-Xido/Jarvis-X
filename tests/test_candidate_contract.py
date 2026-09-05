from jarvisx.candidate_contract import (
    AdmissionPolicy,
    CandidateDecision,
    CandidateProposal,
    ConstraintResult,
    ResourceEnvelope,
    ResourceUsage,
    admit_candidate,
    canonical_state_hash,
    normalize_metrics,
)


def _proposal(**overrides):
    parent = canonical_state_hash({"state": [1, 2, 3]})
    candidate = canonical_state_hash({"state": [1, 2, 4]})
    values = dict(
        subsystem="fixture",
        candidate_id="candidate-001",
        operator_version="fixture-v1",
        parent_state_hash=parent,
        candidate_state_hash=candidate,
        objective_before=1.0,
        objective_after=0.75,
        metrics=normalize_metrics({"reconstruction": 0.2, "stability": 0.1}),
        constraints=(ConstraintResult("finite", True),),
        resource_envelope=ResourceEnvelope(100, 20, 1024),
        resource_usage=ResourceUsage(10, 5, 128),
    )
    values.update(overrides)
    return CandidateProposal(**values)


def test_improving_admissible_candidate_commits_with_verifiable_receipt():
    receipt = admit_candidate(_proposal())

    assert receipt.decision is CandidateDecision.COMMIT
    assert receipt.improvement == 0.25
    assert receipt.rejection_reasons == ()
    assert receipt.verify()
    assert receipt.to_dict()["proposal"]["metrics"] == {
        "reconstruction": 0.2,
        "stability": 0.1,
    }


def test_hard_constraint_failure_cannot_be_bought_by_better_objective():
    proposal = _proposal(
        objective_after=0.0,
        constraints=(ConstraintResult("anchor_drift", False, observed=0.2, limit=0.1),),
    )
    receipt = admit_candidate(proposal)

    assert receipt.decision is CandidateDecision.ROLLBACK
    assert "constraint:anchor_drift" in receipt.rejection_reasons
    assert receipt.verify()


def test_resource_overrun_rolls_back_even_when_objective_improves():
    proposal = _proposal(resource_usage=ResourceUsage(101, 5, 128))
    receipt = admit_candidate(proposal)

    assert receipt.decision is CandidateDecision.ROLLBACK
    assert receipt.rejection_reasons == ("resource:max_work_units",)


def test_non_improving_candidate_rolls_back_deterministically():
    proposal = _proposal(objective_after=1.0)
    first = admit_candidate(proposal)
    second = admit_candidate(proposal)

    assert first.decision is CandidateDecision.ROLLBACK
    assert first.receipt_hash == second.receipt_hash
    assert first.rejection_reasons == ("objective:no_material_improvement",)


def test_minimum_improvement_policy_is_enforced():
    proposal = _proposal(objective_before=1.0, objective_after=0.95)
    receipt = admit_candidate(proposal, policy=AdmissionPolicy(min_improvement=0.1))

    assert receipt.decision is CandidateDecision.ROLLBACK


def test_metric_normalization_is_order_invariant():
    assert normalize_metrics({"b": 2.0, "a": 1.0}) == normalize_metrics(
        [("a", 1.0), ("b", 2.0)]
    )
