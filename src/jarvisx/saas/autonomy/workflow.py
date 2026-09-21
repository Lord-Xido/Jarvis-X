"""Durable, idempotent, compensating enterprise workflow orchestration."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .events import CausalEventLedger
from .policy import CommitPolicyEngine, CommitRequest


def _fingerprint(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkflowStep:
    step_id: str
    action: str
    resource: str
    dependencies: FrozenSet[str] = field(default_factory=frozenset)
    estimated_cost_minor: int = 0
    risk: float = 0.0
    required_roles: FrozenSet[str] = field(default_factory=frozenset)
    required_approvals: int = 0
    compensate_action: Optional[str] = None


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    version: int
    steps: Tuple[WorkflowStep, ...]

    def validate(self) -> None:
        ids = {step.step_id for step in self.steps}
        if len(ids) != len(self.steps):
            raise ValueError("workflow step ids must be unique")
        for step in self.steps:
            if not step.dependencies.issubset(ids):
                raise ValueError("step has an unknown dependency")
        remaining = {step.step_id: set(step.dependencies) for step in self.steps}
        resolved: Set[str] = set()
        while remaining:
            ready = sorted(key for key, deps in remaining.items() if deps <= resolved)
            if not ready:
                raise ValueError("workflow contains a dependency cycle")
            for key in ready:
                resolved.add(key)
                remaining.pop(key)


@dataclass
class WorkflowRun:
    run_id: str
    tenant_id: str
    subject: str
    definition: WorkflowDefinition
    state_version: int
    approval_epoch: int
    status: str = "running"
    completed: List[str] = field(default_factory=list)
    compensated: List[str] = field(default_factory=list)
    failed_step: Optional[str] = None
    outputs: Dict[str, object] = field(default_factory=dict)
    idempotency: Dict[Tuple[str, str], object] = field(default_factory=dict)


class DurableOrchestrator:
    def __init__(self, ledger: CausalEventLedger, policy: CommitPolicyEngine) -> None:
        self.ledger = ledger
        self.policy = policy
        self._runs: Dict[str, WorkflowRun] = {}

    def start(
        self,
        tenant_id: str,
        subject: str,
        definition: WorkflowDefinition,
        *,
        approval_epoch: int = 0,
    ) -> WorkflowRun:
        definition.validate()
        run = WorkflowRun(
            run_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            subject=subject,
            definition=definition,
            state_version=self.ledger.version(tenant_id),
            approval_epoch=approval_epoch,
        )
        self._runs[run.run_id] = run
        self.ledger.append(
            tenant_id=tenant_id,
            stream="workflow:" + run.run_id,
            event_type="workflow.started",
            payload={"name": definition.name, "version": definition.version},
            actor=subject,
            expected_version=run.state_version,
        )
        run.state_version = self.ledger.version(tenant_id)
        return run

    def ready_steps(self, run: WorkflowRun) -> Tuple[WorkflowStep, ...]:
        completed = set(run.completed)
        return tuple(
            step
            for step in run.definition.steps
            if step.step_id not in completed
            and step.step_id != run.failed_step
            and step.dependencies.issubset(completed)
        )

    def execute_step(
        self,
        run_id: str,
        step_id: str,
        *,
        witness_token: str,
        bindings: Dict[str, object],
        approvals: Iterable[str],
        handler: Callable[[WorkflowStep, Dict[str, object]], object],
        idempotency_key: str,
    ) -> object:
        run = self._runs[run_id]
        if run.status != "running":
            raise RuntimeError("workflow is not running")
        scoped_key = (step_id, idempotency_key)
        if scoped_key in run.idempotency:
            return run.idempotency[scoped_key]
        step = next(
            (item for item in run.definition.steps if item.step_id == step_id),
            None,
        )
        if step is None or step not in self.ready_steps(run):
            raise RuntimeError("step is not ready")
        request = CommitRequest(
            tenant_id=run.tenant_id,
            subject=run.subject,
            action=step.action,
            resource=step.resource,
            state_version=run.state_version,
            approval_epoch=run.approval_epoch,
            bindings=bindings,
            estimated_cost_minor=step.estimated_cost_minor,
            risk=step.risk,
            required_roles=step.required_roles,
            required_approvals=step.required_approvals,
            approval_tokens=tuple(approvals),
        )
        decision = self.policy.decide(request, witness_token)
        if not decision.allowed:
            self.ledger.append(
                tenant_id=run.tenant_id,
                stream="workflow:" + run.run_id,
                event_type="workflow.commit_rejected",
                payload={
                    "step_id": step.step_id,
                    "reason": decision.reason,
                    "proof_hash": decision.proof_hash,
                },
                actor=run.subject,
                causation_id=decision.witness_id,
                expected_version=run.state_version,
            )
            run.state_version = self.ledger.version(run.tenant_id)
            raise PermissionError(decision.reason)
        try:
            output = handler(step, bindings)
        except Exception as exc:
            run.failed_step = step.step_id
            run.status = "failed"
            self.ledger.append(
                tenant_id=run.tenant_id,
                stream="workflow:" + run.run_id,
                event_type="workflow.step_failed",
                payload={"step_id": step.step_id, "error": type(exc).__name__},
                actor=run.subject,
                causation_id=decision.witness_id,
                expected_version=run.state_version,
            )
            run.state_version = self.ledger.version(run.tenant_id)
            raise
        run.completed.append(step.step_id)
        run.outputs[step.step_id] = output
        run.idempotency[scoped_key] = output
        self.ledger.append(
            tenant_id=run.tenant_id,
            stream="workflow:" + run.run_id,
            event_type="workflow.step_committed",
            payload={
                "step_id": step.step_id,
                "output_hash": _fingerprint(output),
                "proof_hash": decision.proof_hash,
                "approval_ids": list(decision.approval_ids),
                "idempotency_key": idempotency_key,
            },
            actor=run.subject,
            causation_id=decision.witness_id,
            expected_version=run.state_version,
        )
        run.state_version = self.ledger.version(run.tenant_id)
        if len(run.completed) == len(run.definition.steps):
            run.status = "completed"
        return output

    def compensate(
        self,
        run_id: str,
        handler: Callable[[WorkflowStep, object], None],
    ) -> Tuple[str, ...]:
        run = self._runs[run_id]
        steps = {step.step_id: step for step in run.definition.steps}
        for step_id in reversed(run.completed):
            step = steps[step_id]
            if not step.compensate_action or step_id in run.compensated:
                continue
            handler(step, run.outputs.get(step_id))
            run.compensated.append(step_id)
            self.ledger.append(
                tenant_id=run.tenant_id,
                stream="workflow:" + run.run_id,
                event_type="workflow.step_compensated",
                payload={"step_id": step_id, "action": step.compensate_action},
                actor=run.subject,
                expected_version=run.state_version,
            )
            run.state_version = self.ledger.version(run.tenant_id)
        run.status = "compensated"
        return tuple(run.compensated)
