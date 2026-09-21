import pytest

from jarvisx.adaptive_orchestrator import (
    HashChainedMemory,
    OrchestrationRuntime,
    PolicyEngine,
    SecurityState,
    ToolRegistry,
    WorkflowStatus,
    WorkPacket,
)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    def reserve(payload, state):
        quantity = int(payload["quantity"])
        available = int(state.get("inventory", 0))
        if quantity > available:
            raise RuntimeError("insufficient inventory")
        state["inventory"] = available - quantity
        state["reserved"] = int(state.get("reserved", 0)) + quantity
        return {"reserved": quantity}

    def release(payload, state):
        quantity = int(payload["quantity"])
        state["reserved"] = int(state.get("reserved", 0)) - quantity
        state["inventory"] = int(state.get("inventory", 0)) + quantity
        return {"released": quantity}

    def create_order(payload, state):
        quantity = int(payload["quantity"])
        if int(state.get("reserved", 0)) < quantity:
            raise RuntimeError("reservation missing")
        order_id = str(payload["order_id"])
        state["production_order"] = order_id
        return {"order_id": order_id, "quantity": quantity}

    def quality_gate(payload, state):
        return {"quality": "FAIL"}

    registry.register("reserve_inventory", reserve)
    registry.register("release_inventory", release)
    registry.register("create_production_order", create_order)
    registry.register("quality_gate", quality_gate)
    return registry


def test_orchestrator_executes_verified_dependency_graph() -> None:
    runtime = OrchestrationRuntime(registry=_registry())
    receipt = runtime.run(
        workflow_id="wf-001",
        state={"inventory": 20},
        principal_scopes={"inventory:write", "production:write"},
        packets=(
            WorkPacket(
                task_id="reserve",
                action="reserve_inventory",
                payload={"quantity": 10},
                expected={"reserved": 10},
                required_scopes=("inventory:write",),
                idempotency_key="wf-001:reserve",
                compensate_action="release_inventory",
                compensate_payload={"quantity": 10},
            ),
            WorkPacket(
                task_id="production",
                action="create_production_order",
                payload={"quantity": 10, "order_id": "PO-10"},
                depends_on=("reserve",),
                expected={"order_id": "PO-10"},
                required_scopes=("production:write",),
                idempotency_key="wf-001:production",
            ),
        ),
    )
    assert receipt.status is WorkflowStatus.COMPLETED
    assert receipt.state["inventory"] == 10
    assert receipt.state["reserved"] == 10
    assert receipt.state["production_order"] == "PO-10"
    assert runtime.memory.verify()


def test_intrusion_signal_contracts_authority_and_blocks_mutation() -> None:
    runtime = OrchestrationRuntime(registry=_registry())
    receipt = runtime.run(
        workflow_id="wf-intrusion",
        state={"inventory": 20},
        principal_scopes={"inventory:write"},
        security=SecurityState(
            confidence=0.2,
            intrusion_detected=True,
            reason="intrusion signal active",
        ),
        packets=(
            WorkPacket(
                task_id="reserve",
                action="reserve_inventory",
                payload={"quantity": 10},
                expected={"reserved": 10},
                required_scopes=("inventory:write",),
                idempotency_key="wf-intrusion:reserve",
            ),
        ),
    )
    assert receipt.status is WorkflowStatus.HALTED
    assert receipt.state["inventory"] == 20
    assert any(
        event.event_type == "POLICY_DECISION"
        and event.payload["allowed"] is False
        for event in runtime.memory.events
    )


def test_verification_failure_compensates_prior_mutation() -> None:
    runtime = OrchestrationRuntime(registry=_registry())
    receipt = runtime.run(
        workflow_id="wf-compensate",
        state={"inventory": 20},
        principal_scopes={"inventory:write", "quality:read"},
        packets=(
            WorkPacket(
                task_id="reserve",
                action="reserve_inventory",
                payload={"quantity": 10},
                expected={"reserved": 10},
                required_scopes=("inventory:write",),
                idempotency_key="wf-compensate:reserve",
                compensate_action="release_inventory",
                compensate_payload={"quantity": 10},
            ),
            WorkPacket(
                task_id="quality",
                action="quality_gate",
                depends_on=("reserve",),
                expected={"quality": "PASS"},
                required_scopes=("quality:read",),
                mutating=False,
            ),
        ),
    )
    assert receipt.status is WorkflowStatus.HALTED
    assert receipt.state["inventory"] == 20
    assert receipt.state["reserved"] == 0
    assert any(event.event_type == "TASK_COMPENSATED" for event in runtime.memory.events)


def test_mutating_task_requires_idempotency_key() -> None:
    runtime = OrchestrationRuntime(registry=_registry())
    receipt = runtime.run(
        workflow_id="wf-idempotency",
        state={"inventory": 20},
        principal_scopes={"inventory:write"},
        packets=(
            WorkPacket(
                task_id="reserve",
                action="reserve_inventory",
                payload={"quantity": 1},
                required_scopes=("inventory:write",),
            ),
        ),
    )
    assert receipt.status is WorkflowStatus.HALTED
    assert receipt.state["inventory"] == 20


def test_graph_rejects_unknown_dependencies_and_cycles() -> None:
    runtime = OrchestrationRuntime(registry=_registry())
    with pytest.raises(ValueError, match="unknown tasks"):
        runtime.run(
            workflow_id="bad-missing",
            packets=(
                WorkPacket(
                    task_id="a",
                    action="quality_gate",
                    mutating=False,
                    depends_on=("missing",),
                ),
            ),
        )
    with pytest.raises(ValueError, match="cycle"):
        runtime.run(
            workflow_id="bad-cycle",
            packets=(
                WorkPacket(task_id="a", action="quality_gate", mutating=False, depends_on=("b",)),
                WorkPacket(task_id="b", action="quality_gate", mutating=False, depends_on=("a",)),
            ),
        )


def test_policy_threshold_and_hash_chain_integrity() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        PolicyEngine(mutation_security_threshold=1.5)

    memory = HashChainedMemory()
    memory.append(event_type="A", workflow_id="wf", task_id=None, payload={"x": 1})
    assert memory.verify()
    memory.events[0].payload["x"] = 2
    assert not memory.verify()
