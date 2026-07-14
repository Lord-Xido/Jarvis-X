from dataclasses import replace

import pytest

from jarvisx.saas.autonomy import (
    AuthorityWitness,
    AutonomicEnterpriseController,
    CausalEventLedger,
    CommitPolicyEngine,
    DigitalTwin,
    DurableOrchestrator,
    EnterpriseState,
    Scenario,
    TemporalSemanticCache,
    WorkflowDefinition,
    WorkflowStep,
)


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


def test_commit_time_authorization_rejects_stale_or_rebound_witness():
    from jarvisx.saas.autonomy import CommitRequest

    policy = CommitPolicyEngine(max_risk=0.4, max_cost_minor=1000)
    bindings = {"invoice_id": "i1", "amount": 500}
    witness = AuthorityWitness.issue(
        tenant_id="t1",
        subject="u1",
        action="invoice.pay",
        resource="invoice:i1",
        state_version=4,
        approval_epoch=2,
        bindings=bindings,
        roles={"finance_admin"},
        ttl_seconds=60,
        now_ns=1_000,
    )
    request = CommitRequest(
        tenant_id="t1",
        subject="u1",
        action="invoice.pay",
        resource="invoice:i1",
        state_version=4,
        approval_epoch=2,
        bindings=bindings,
        estimated_cost_minor=500,
        risk=0.1,
        required_roles=frozenset({"finance_admin"}),
    )
    assert policy.decide(request, witness, now_ns=2_000).allowed
    rebound = replace(request, bindings={"invoice_id": "i2", "amount": 500})
    assert not policy.decide(rebound, witness, now_ns=2_000).allowed
    assert not policy.decide(request, witness, now_ns=witness.expires_at_ns + 1).allowed


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
    orchestrator = DurableOrchestrator(
        ledger,
        CommitPolicyEngine(max_risk=0.5, max_cost_minor=10_000),
    )
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
    witness = AuthorityWitness.issue(
        tenant_id="t1",
        subject="u1",
        action="client.create",
        resource="clients",
        state_version=run.state_version,
        approval_epoch=0,
        bindings=bindings,
        roles={"operations_manager"},
    )
    calls = []

    def handler(step, data):
        calls.append(step.step_id)
        return {"id": step.step_id + "-1"}

    output = orchestrator.execute_step(
        run.run_id,
        "create_client",
        witness=witness,
        bindings=bindings,
        approvals=(),
        handler=handler,
        idempotency_key="k1",
    )
    assert (
        orchestrator.execute_step(
            run.run_id,
            "create_client",
            witness=witness,
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


def test_autonomic_controller_requires_fresh_proof_before_commit():
    controller = AutonomicEnterpriseController(
        policy=CommitPolicyEngine(max_risk=1.0, max_cost_minor=1_000_000)
    )
    proposal = controller.propose(
        tenant_id="t1",
        subject="u1",
        state=enterprise_state(),
        scenarios=[Scenario(name="steady", revenue_growth=0.02)],
    )[0]
    request = proposal.commit_request
    witness = AuthorityWitness.issue(
        tenant_id=request.tenant_id,
        subject=request.subject,
        action=request.action,
        resource=request.resource,
        state_version=request.state_version,
        approval_epoch=request.approval_epoch,
        bindings=request.bindings,
        roles={"tenant_owner"},
    )
    event_id = controller.commit(proposal, witness)
    assert event_id
    assert controller.ledger.verify()
    with pytest.raises(PermissionError):
        controller.commit(proposal, replace(witness, state_version=99))
