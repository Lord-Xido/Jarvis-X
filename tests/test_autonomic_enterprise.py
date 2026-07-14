from dataclasses import replace

import pytest

from jarvisx.saas.autonomy import (
    AutonomicEnterpriseController,
    CausalEventLedger,
    CommitPolicyEngine,
    CommitRequest,
    DigitalTwin,
    DurableOrchestrator,
    EnterpriseState,
    Scenario,
    TemporalSemanticCache,
    WorkflowDefinition,
    WorkflowStep,
)

SIGNING_KEY = "test-authority-signing-key-32-bytes-minimum"


def enterprise_state():
    return EnterpriseState(
        cash_minor=1_000_000,
        monthly_revenue_minor=500_000,
        monthly_cost_minor=350_000,
        receivables_minor=300_000,
        pipeline_minor=2_000_000,
        delivery_health=0.85,
        finance_health=0.75,
        governance_health=0.9,
        churn_rate=0.03,
        collection_rate=0.8,
    )


def test_causal_ledger_integrity_and_optimistic_concurrency():
    ledger = CausalEventLedger()
    first = ledger.append(
        tenant_id="t1",
        stream="engagement:e1",
        event_type="engagement.created",
        payload={"name": "Alpha"},
        actor="u1",
        expected_version=0,
    )
    second = ledger.append(
        tenant_id="t1",
        stream="engagement:e1",
        event_type="engagement.updated",
        payload={"progress": 0.5},
        actor="u1",
        causation_id=first.event_id,
        expected_version=1,
    )
    assert ledger.verify()
    assert second.sequence == 2
    assert ledger.version("t1") == 2
    assert ledger.merkle_root("t1") != ledger.GENESIS
    with pytest.raises(RuntimeError):
        ledger.append(
            tenant_id="t1",
            stream="engagement:e1",
            event_type="engagement.updated",
            payload={},
            actor="u1",
            expected_version=1,
        )


def authorization_request(required_approvals=0):
    return CommitRequest(
        tenant_id="t1",
        subject="u1",
        action="invoice.pay",
        resource="invoice:i1",
        state_version=4,
        approval_epoch=2,
        bindings={"invoice_id": "i1", "amount": 500},
        estimated_cost_minor=500,
        risk=0.1,
        required_roles=frozenset({"finance_admin"}),
        required_approvals=required_approvals,
    )


def test_commit_time_authorization_rejects_stale_rebound_or_forged_witness():
    policy = CommitPolicyEngine(
        max_risk=0.4,
        max_cost_minor=1000,
        signing_key=SIGNING_KEY,
    )
    request = authorization_request()
    witness = policy.issue_witness(
        request,
        roles={"finance_admin"},
        ttl_seconds=60,
        now_ns=1_000,
    )
    assert policy.decide(request, witness, now_ns=2_000).allowed
    rebound = replace(request, bindings={"invoice_id": "i2", "amount": 500})
    assert not policy.decide(rebound, witness, now_ns=2_000).allowed
    assert not policy.decide(
        request,
        witness,
        now_ns=60_000_001_001,
    ).allowed

    body, signature = witness.split(".")
    tampered = ("A" if body[0] != "A" else "B") + body[1:] + "." + signature
    assert policy.decide(request, tampered, now_ns=2_000).reason == (
        "failed:witness_signature"
    )
    foreign = CommitPolicyEngine(
        signing_key="foreign-signing-key-that-is-32-bytes-long"
    )
    assert not foreign.decide(request, witness, now_ns=2_000).allowed


def test_approval_gate_requires_signed_scoped_distinct_approvers():
    policy = CommitPolicyEngine(
        max_risk=0.4,
        max_cost_minor=1000,
        signing_key=SIGNING_KEY,
    )
    request = authorization_request(required_approvals=2)
    witness = policy.issue_witness(
        request,
        roles={"finance_admin"},
        now_ns=1_000,
    )

    unsigned = replace(request, approval_tokens=("anything", "anything-else"))
    assert not policy.decide(unsigned, witness, now_ns=2_000).allowed

    approval_a = policy.issue_approval(
        request,
        approver="director-a",
        now_ns=1_000,
    )
    self_approval = policy.issue_approval(
        request,
        approver=request.subject,
        now_ns=1_000,
    )
    duplicate = replace(
        request,
        approval_tokens=(approval_a, approval_a, self_approval),
    )
    assert not policy.decide(duplicate, witness, now_ns=2_000).allowed

    approval_b = policy.issue_approval(
        request,
        approver="director-b",
        now_ns=1_000,
    )
    approved = replace(request, approval_tokens=(approval_a, approval_b))
    decision = policy.decide(approved, witness, now_ns=2_000)
    assert decision.allowed
    assert len(decision.approval_ids) == 2

    rebound = replace(
        request,
        bindings={"invoice_id": "i2", "amount": 500},
        approval_tokens=(approval_a, approval_b),
    )
    assert not policy.decide(rebound, witness, now_ns=2_000).allowed


def test_temporal_cache_is_bound_to_scope_version_and_time():
    cache = TemporalSemanticCache()
    cache.put(
        tenant_id="t1",
        semantic_key={"question": "forecast"},
        scope={"engagement": "e1"},
        value={"risk": 0.2},
        state_version=3,
        ttl_seconds=10,
        now_ns=1_000,
    )
    assert cache.get(
        tenant_id="t1",
        semantic_key={"question": "forecast"},
        scope={"engagement": "e1"},
        state_version=3,
        now_ns=2_000,
    ) == {"risk": 0.2}
    assert cache.get(
        tenant_id="t1",
        semantic_key={"question": "forecast"},
        scope={"engagement": "e2"},
        state_version=3,
        now_ns=2_000,
    ) is None
    assert cache.get(
        tenant_id="t1",
        semantic_key={"question": "forecast"},
        scope={"engagement": "e1"},
        state_version=4,
        now_ns=2_000,
    ) is None


def test_digital_twin_is_deterministic_and_downside_aware():
    twin = DigitalTwin()
    scenarios = [
        Scenario(name="conservative", revenue_growth=0.02, cost_growth=0.01),
        Scenario(
            name="expansion",
            revenue_growth=0.08,
            cost_growth=0.05,
            one_off_cost_minor=400_000,
            pipeline_conversion=0.1,
        ),
        Scenario(name="stress", revenue_growth=-0.08, churn_delta=0.1),
    ]
    first = twin.rank(enterprise_state(), scenarios, risk_aversion=2.0, paths=32)
    second = twin.rank(enterprise_state(), scenarios, risk_aversion=2.0, paths=32)
    assert first == second
    assert first[-1].scenario == "stress"
    assert all(0 <= item.survival_probability <= 1 for item in first)


def test_durable_workflow_is_idempotent_and_compensates():
    ledger = CausalEventLedger()
    policy = CommitPolicyEngine(
        max_risk=0.5,
        max_cost_minor=10_000,
        signing_key=SIGNING_KEY,
    )
    orchestrator = DurableOrchestrator(ledger, policy)
    definition = WorkflowDefinition(
        name="client-onboarding",
        version=1,
        steps=(
            WorkflowStep(
                "create_client",
                "client.create",
                "clients",
                required_roles=frozenset({"operations_manager"}),
                compensate_action="client.delete",
            ),
            WorkflowStep(
                "open_engagement",
                "engagement.create",
                "engagements",
                dependencies=frozenset({"create_client"}),
                required_roles=frozenset({"operations_manager"}),
                compensate_action="engagement.close",
            ),
        ),
    )
    run = orchestrator.start("t1", "u1", definition)
    bindings = {"client": "Acme"}
    request = CommitRequest(
        tenant_id=run.tenant_id,
        subject=run.subject,
        action="client.create",
        resource="clients",
        state_version=run.state_version,
        approval_epoch=run.approval_epoch,
        bindings=bindings,
        estimated_cost_minor=0,
        risk=0.0,
        required_roles=frozenset({"operations_manager"}),
    )
    witness = policy.issue_witness(request, roles={"operations_manager"})
    calls = []

    def handler(step, data):
        calls.append(step.step_id)
        return {"id": step.step_id + "-1"}

    output = orchestrator.execute_step(
        run.run_id,
        "create_client",
        witness_token=witness,
        bindings=bindings,
        approvals=(),
        handler=handler,
        idempotency_key="k1",
    )
    assert (
        orchestrator.execute_step(
            run.run_id,
            "create_client",
            witness_token=witness,
            bindings=bindings,
            approvals=(),
            handler=handler,
            idempotency_key="k1",
        )
        == output
    )
    assert calls == ["create_client"]
    compensated = []
    orchestrator.compensate(
        run.run_id,
        lambda step, result: compensated.append((step.step_id, result["id"])),
    )
    assert compensated == [("create_client", "create_client-1")]
    assert ledger.verify()


def test_workflow_idempotency_keys_are_scoped_to_each_step():
    ledger = CausalEventLedger()
    policy = CommitPolicyEngine(
        max_risk=1.0,
        max_cost_minor=10_000,
        signing_key=SIGNING_KEY,
    )
    orchestrator = DurableOrchestrator(ledger, policy)
    definition = WorkflowDefinition(
        name="parallel",
        version=1,
        steps=(
            WorkflowStep(
                "step_a",
                "a.execute",
                "resource:a",
                required_roles=frozenset({"operator"}),
            ),
            WorkflowStep(
                "step_b",
                "b.execute",
                "resource:b",
                required_roles=frozenset({"operator"}),
            ),
        ),
    )
    run = orchestrator.start("t1", "operator-1", definition)
    calls = []
    for step in definition.steps:
        bindings = {"step": step.step_id}
        request = CommitRequest(
            tenant_id=run.tenant_id,
            subject=run.subject,
            action=step.action,
            resource=step.resource,
            state_version=run.state_version,
            approval_epoch=run.approval_epoch,
            bindings=bindings,
            estimated_cost_minor=0,
            risk=0.0,
            required_roles=step.required_roles,
        )
        witness = policy.issue_witness(request, roles={"operator"})
        result = orchestrator.execute_step(
            run.run_id,
            step.step_id,
            witness_token=witness,
            bindings=bindings,
            approvals=(),
            handler=lambda current, data: calls.append(current.step_id)
            or current.step_id,
            idempotency_key="shared-key",
        )
        assert result == step.step_id
    assert calls == ["step_a", "step_b"]


def test_autonomic_controller_requires_signed_fresh_proof_before_commit():
    controller = AutonomicEnterpriseController(
        policy=CommitPolicyEngine(
            max_risk=1.0,
            max_cost_minor=1_000_000,
            signing_key=SIGNING_KEY,
        )
    )
    proposal = controller.propose(
        tenant_id="t1",
        subject="u1",
        state=enterprise_state(),
        scenarios=[Scenario(name="steady", revenue_growth=0.02)],
    )[0]
    with pytest.raises(PermissionError):
        controller.commit(proposal, "client-forged-witness")

    witness = controller.policy.issue_witness(
        proposal.commit_request,
        roles={"tenant_owner"},
    )
    event_id = controller.commit(proposal, witness)
    assert event_id
    assert controller.ledger.verify()
    with pytest.raises(PermissionError):
        controller.commit(proposal, witness)
