import pytest

from jarvisx.assembler import Assembler
from jarvisx.parser import Parser
from jarvisx.system_runtime import (
    CAP_VM_EXECUTE,
    CAP_VM_REFLEX,
    DeterministicPlanner,
    ExecutionRequest,
    ExecutionStatus,
    PlanCandidate,
    RequestCollisionError,
    ResourceBudget,
    SystemRuntime,
)


def assemble(source: str) -> tuple[int, ...]:
    return tuple(Assembler().assemble(Parser().parse(source)))


def test_successful_request_commits_only_after_verification() -> None:
    runtime = SystemRuntime()
    request = ExecutionRequest(
        request_id="task-1",
        program=assemble("SET A 10\nSET B 20\nADD Ψ A B\nHALT"),
        granted_capabilities=frozenset({CAP_VM_EXECUTE}),
    )

    receipt = runtime.execute(request)

    assert receipt.status is ExecutionStatus.COMMITTED
    assert receipt.committed is True
    assert receipt.state_dict()["Ψ"] == 30
    assert runtime.committed_state("task-1") == receipt.state_dict()
    assert receipt.state_hash is not None
    assert receipt.vm_ledger_head is not None
    assert runtime.audit.verify()
    assert runtime.verify()


def test_missing_capability_is_rejected_without_authoritative_state() -> None:
    runtime = SystemRuntime()
    request = ExecutionRequest(
        request_id="task-2",
        program=assemble("HALT"),
        granted_capabilities=frozenset(),
    )

    receipt = runtime.execute(request)

    assert receipt.status is ExecutionStatus.REJECTED
    assert receipt.committed is False
    assert receipt.error_type == "PolicyRejection"
    assert runtime.committed_state("task-2") is None
    assert len(runtime.audit.chain) == 1
    assert runtime.verify()


def test_reflex_requires_explicit_adaptation_capability() -> None:
    runtime = SystemRuntime()
    request = ExecutionRequest(
        request_id="task-reflex",
        program=assemble("SET Ψ 10\nSET Φ 20\nHALT"),
        granted_capabilities=frozenset({CAP_VM_EXECUTE}),
        enable_reflex=True,
    )

    receipt = runtime.execute(request)

    assert receipt.status is ExecutionStatus.REJECTED
    assert CAP_VM_REFLEX in request.required_capabilities
    assert runtime.committed_state("task-reflex") is None


def test_cycle_budget_failure_discards_isolated_vm_state() -> None:
    runtime = SystemRuntime()
    request = ExecutionRequest(
        request_id="task-budget",
        program=assemble("SET A 1\nSET A 2\nHALT"),
        granted_capabilities=frozenset({CAP_VM_EXECUTE}),
        budget=ResourceBudget(max_cycles=1),
    )

    receipt = runtime.execute(request)

    assert receipt.status is ExecutionStatus.FAILED
    assert receipt.committed is False
    assert receipt.state == ()
    assert runtime.committed_state("task-budget") is None
    assert runtime.verify()


def test_request_id_is_idempotent_for_identical_contents() -> None:
    runtime = SystemRuntime()
    request = ExecutionRequest(
        request_id="task-idempotent",
        program=assemble("SET A 4\nHALT"),
        granted_capabilities=frozenset({CAP_VM_EXECUTE}),
    )

    first = runtime.execute(request)
    audit_length = len(runtime.audit.chain)
    second = runtime.execute(request)

    assert second == first
    assert len(runtime.audit.chain) == audit_length


def test_request_id_collision_fails_closed() -> None:
    runtime = SystemRuntime()
    runtime.execute(
        ExecutionRequest(
            request_id="task-collision",
            program=assemble("SET A 1\nHALT"),
            granted_capabilities=frozenset({CAP_VM_EXECUTE}),
        )
    )

    with pytest.raises(RequestCollisionError, match="reused with different contents"):
        runtime.execute(
            ExecutionRequest(
                request_id="task-collision",
                program=assemble("SET A 2\nHALT"),
                granted_capabilities=frozenset({CAP_VM_EXECUTE}),
            )
        )

    assert runtime.committed_state("task-collision")["A"] == 1


def test_planner_filters_unauthorized_candidate_and_is_deterministic() -> None:
    planner = DeterministicPlanner()
    budget = ResourceBudget(max_candidates=4)
    candidates = (
        PlanCandidate(
            "unauthorized",
            assemble("HALT"),
            quality=100.0,
            cost=0.0,
            latency=0.0,
            risk=0.0,
            required_capabilities=frozenset({CAP_VM_EXECUTE, CAP_VM_REFLEX}),
        ),
        PlanCandidate(
            "plan-b",
            assemble("SET A 2\nHALT"),
            quality=5.0,
            cost=1.0,
            latency=1.0,
            risk=1.0,
        ),
        PlanCandidate(
            "plan-a",
            assemble("SET A 1\nHALT"),
            quality=5.0,
            cost=1.0,
            latency=1.0,
            risk=1.0,
        ),
    )

    selected = planner.select(
        candidates,
        granted_capabilities={CAP_VM_EXECUTE},
        budget=budget,
    )

    assert selected.plan_id == "plan-a"


def test_execute_plans_preserves_selected_plan_provenance() -> None:
    runtime = SystemRuntime()
    receipt = runtime.execute_plans(
        request_id="task-planned",
        candidates=(
            PlanCandidate(
                "low",
                assemble("SET A 1\nHALT"),
                quality=1.0,
                cost=1.0,
                latency=1.0,
                risk=1.0,
            ),
            PlanCandidate(
                "high",
                assemble("SET A 9\nHALT"),
                quality=10.0,
                cost=1.0,
                latency=1.0,
                risk=1.0,
            ),
        ),
        granted_capabilities={CAP_VM_EXECUTE},
    )

    assert receipt.status is ExecutionStatus.COMMITTED
    assert receipt.plan_id == "high"
    assert receipt.state_dict()["A"] == 9


def test_system_audit_write_failure_cannot_become_committed_state(monkeypatch) -> None:
    runtime = SystemRuntime()
    request = ExecutionRequest(
        request_id="task-audit-failure",
        program=assemble("SET A 7\nHALT"),
        granted_capabilities=frozenset({CAP_VM_EXECUTE}),
    )

    def fail_log(state: object, opcode: int) -> None:
        raise OSError("system audit unavailable")

    monkeypatch.setattr(runtime.audit, "log", fail_log)

    with pytest.raises(OSError, match="system audit unavailable"):
        runtime.execute(request)

    assert runtime.committed_state("task-audit-failure") is None
    assert runtime.receipt("task-audit-failure") is None
    assert len(runtime.audit.chain) == 0
