from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence


class WorkflowStatus(str, Enum):
    NEW = "NEW"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECONCILING = "RECONCILING"
    COMPLETED = "COMPLETED"
    HALTED = "HALTED"


@dataclass(frozen=True)
class SecurityState:
    confidence: float = 1.0
    intrusion_detected: bool = False
    reason: str = ""

    def permits_mutation(self, threshold: float = 0.75) -> bool:
        return (
            not self.intrusion_detected
            and 0.0 <= self.confidence <= 1.0
            and self.confidence >= threshold
        )


@dataclass(frozen=True)
class WorkPacket:
    task_id: str
    action: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    expected: Mapping[str, Any] = field(default_factory=dict)
    required_scopes: tuple[str, ...] = ()
    mutating: bool = True
    idempotency_key: str | None = None
    retries: int = 0
    compensate_action: str | None = None
    compensate_payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class TaskReceipt:
    task_id: str
    action: str
    success: bool
    attempts: int
    output: Mapping[str, Any]
    verified: bool
    error: str | None = None


@dataclass(frozen=True)
class Event:
    sequence: int
    event_type: str
    workflow_id: str
    task_id: str | None
    payload: Mapping[str, Any]
    previous_hash: str
    event_hash: str


@dataclass(frozen=True)
class WorkflowReceipt:
    workflow_id: str
    status: WorkflowStatus
    task_receipts: tuple[TaskReceipt, ...]
    event_count: int
    ledger_head: str
    state: Mapping[str, Any]


Tool = Callable[[Mapping[str, Any], MutableMapping[str, Any]], Mapping[str, Any]]


class HashChainedMemory:
    """Append-only, tamper-evident operational memory."""

    def __init__(self) -> None:
        self._events: list[Event] = []
        self._head = "0" * 64

    @property
    def head(self) -> str:
        return self._head

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    def append(
        self,
        *,
        event_type: str,
        workflow_id: str,
        task_id: str | None,
        payload: Mapping[str, Any],
    ) -> Event:
        body = {
            "sequence": len(self._events),
            "event_type": event_type,
            "workflow_id": workflow_id,
            "task_id": task_id,
            "payload": _json_native(payload),
            "previous_hash": self._head,
        }
        digest = sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        event = Event(
            sequence=body["sequence"],
            event_type=event_type,
            workflow_id=workflow_id,
            task_id=task_id,
            payload=dict(payload),
            previous_hash=self._head,
            event_hash=digest,
        )
        self._events.append(event)
        self._head = digest
        return event

    def verify(self) -> bool:
        previous = "0" * 64
        for sequence, event in enumerate(self._events):
            body = {
                "sequence": sequence,
                "event_type": event.event_type,
                "workflow_id": event.workflow_id,
                "task_id": event.task_id,
                "payload": _json_native(event.payload),
                "previous_hash": previous,
            }
            digest = sha256(
                json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if (
                event.sequence != sequence
                or event.previous_hash != previous
                or event.event_hash != digest
            ):
                return False
            previous = digest
        return previous == self._head


class ToolRegistry:
    """Narrow named capabilities exposed to the orchestration runtime."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, tool: Tool) -> None:
        if not name or name in self._tools:
            raise ValueError(f"tool {name!r} is empty or already registered")
        self._tools[name] = tool

    def execute(
        self,
        name: str,
        payload: Mapping[str, Any],
        state: MutableMapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise RuntimeError(f"unregistered capability: {name}") from exc
        result = tool(payload, state)
        if not isinstance(result, Mapping):
            raise TypeError(f"tool {name!r} must return a mapping")
        return dict(result)


class PolicyEngine:
    """Fail-closed business/security projection Pi_Lambda."""

    def __init__(self, mutation_security_threshold: float = 0.75) -> None:
        if not 0.0 <= mutation_security_threshold <= 1.0:
            raise ValueError("mutation security threshold must be in [0, 1]")
        self.mutation_security_threshold = mutation_security_threshold

    def authorize(
        self,
        packet: WorkPacket,
        *,
        principal_scopes: Iterable[str],
        security: SecurityState,
    ) -> PolicyDecision:
        missing = sorted(set(packet.required_scopes) - set(principal_scopes))
        if missing:
            return PolicyDecision(False, f"missing scopes: {', '.join(missing)}")
        if packet.mutating and not packet.idempotency_key:
            return PolicyDecision(False, "mutating tasks require an idempotency key")
        if packet.mutating and not security.permits_mutation(
            self.mutation_security_threshold
        ):
            return PolicyDecision(
                False,
                security.reason or "security confidence below mutation threshold",
            )
        return PolicyDecision(True, "authorized")


class OrchestrationRuntime:
    """Reference runtime: govern -> execute -> verify -> recover -> remember."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy: PolicyEngine | None = None,
        memory: HashChainedMemory | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or PolicyEngine()
        self.memory = memory or HashChainedMemory()
        self._idempotent_receipts: dict[str, TaskReceipt] = {}

    def run(
        self,
        *,
        workflow_id: str,
        packets: Sequence[WorkPacket],
        state: MutableMapping[str, Any] | None = None,
        principal_scopes: Iterable[str] = (),
        security: SecurityState | None = None,
    ) -> WorkflowReceipt:
        if not workflow_id:
            raise ValueError("workflow_id is required")
        workflow_state = state if state is not None else {}
        security_state = security or SecurityState()
        packet_map = self._validate_graph(packets)
        receipts: dict[str, TaskReceipt] = {}
        completed: list[WorkPacket] = []
        pending = set(packet_map)
        self._event("WORKFLOW_STARTED", workflow_id, None, {"tasks": len(packet_map)})

        while pending:
            ready = sorted(
                task_id
                for task_id in pending
                if set(packet_map[task_id].depends_on).issubset(receipts)
            )
            if not ready:
                return self._halted(workflow_id, receipts, workflow_state, "dependency deadlock")
            self._event("WAVEFRONT_READY", workflow_id, None, {"tasks": ready})

            for task_id in ready:
                packet = packet_map[task_id]
                decision = self.policy.authorize(
                    packet,
                    principal_scopes=principal_scopes,
                    security=security_state,
                )
                self._event(
                    "POLICY_DECISION",
                    workflow_id,
                    task_id,
                    {"allowed": decision.allowed, "reason": decision.reason},
                )
                if not decision.allowed:
                    self._compensate(workflow_id, completed, workflow_state)
                    return self._halted(
                        workflow_id,
                        receipts,
                        workflow_state,
                        f"{task_id}: {decision.reason}",
                    )

                receipt = self._execute(workflow_id, packet, workflow_state)
                receipts[task_id] = receipt
                pending.remove(task_id)
                if not receipt.success or not receipt.verified:
                    self._compensate(workflow_id, completed, workflow_state)
                    return self._halted(
                        workflow_id,
                        receipts,
                        workflow_state,
                        receipt.error or f"{task_id}: verification failed",
                    )
                completed.append(packet)

        self._event(
            "RECONCILIATION_PASSED",
            workflow_id,
            None,
            {"completed_tasks": len(receipts)},
        )
        if not self.memory.verify():
            raise RuntimeError("operational memory integrity failure")
        self._event(
            "WORKFLOW_COMPLETED",
            workflow_id,
            None,
            {"completed_tasks": len(receipts)},
        )
        return WorkflowReceipt(
            workflow_id=workflow_id,
            status=WorkflowStatus.COMPLETED,
            task_receipts=tuple(receipts[key] for key in sorted(receipts)),
            event_count=len(self.memory.events),
            ledger_head=self.memory.head,
            state=dict(workflow_state),
        )

    def _execute(
        self,
        workflow_id: str,
        packet: WorkPacket,
        state: MutableMapping[str, Any],
    ) -> TaskReceipt:
        if packet.idempotency_key in self._idempotent_receipts:
            self._event(
                "IDEMPOTENT_REPLAY",
                workflow_id,
                packet.task_id,
                {"idempotency_key": packet.idempotency_key},
            )
            return self._idempotent_receipts[packet.idempotency_key]

        attempts = 0
        last_error: str | None = None
        while attempts <= packet.retries:
            attempts += 1
            self._event(
                "TASK_ATTEMPTED",
                workflow_id,
                packet.task_id,
                {"action": packet.action, "attempt": attempts},
            )
            try:
                output = self.registry.execute(packet.action, packet.payload, state)
                verified = all(output.get(k) == v for k, v in packet.expected.items())
                self._event(
                    "TASK_VERIFIED" if verified else "TASK_VERIFICATION_FAILED",
                    workflow_id,
                    packet.task_id,
                    {"output": output, "expected": packet.expected},
                )
                if verified:
                    receipt = TaskReceipt(
                        packet.task_id,
                        packet.action,
                        True,
                        attempts,
                        output,
                        True,
                    )
                    if packet.idempotency_key:
                        self._idempotent_receipts[packet.idempotency_key] = receipt
                    return receipt
                last_error = "verification mismatch"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self._event(
                    "TASK_ERROR",
                    workflow_id,
                    packet.task_id,
                    {"error": last_error, "attempt": attempts},
                )

        return TaskReceipt(
            packet.task_id,
            packet.action,
            False,
            attempts,
            {},
            False,
            last_error,
        )

    def _compensate(
        self,
        workflow_id: str,
        completed: Sequence[WorkPacket],
        state: MutableMapping[str, Any],
    ) -> None:
        for packet in reversed(completed):
            if not packet.compensate_action:
                continue
            try:
                output = self.registry.execute(
                    packet.compensate_action,
                    packet.compensate_payload,
                    state,
                )
                self._event(
                    "TASK_COMPENSATED",
                    workflow_id,
                    packet.task_id,
                    {"action": packet.compensate_action, "output": output},
                )
            except Exception as exc:
                self._event(
                    "COMPENSATION_FAILED",
                    workflow_id,
                    packet.task_id,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )

    def _halted(
        self,
        workflow_id: str,
        receipts: Mapping[str, TaskReceipt],
        state: Mapping[str, Any],
        reason: str,
    ) -> WorkflowReceipt:
        self._event("WORKFLOW_HALTED", workflow_id, None, {"reason": reason})
        return WorkflowReceipt(
            workflow_id,
            WorkflowStatus.HALTED,
            tuple(receipts[key] for key in sorted(receipts)),
            len(self.memory.events),
            self.memory.head,
            dict(state),
        )

    def _event(
        self,
        event_type: str,
        workflow_id: str,
        task_id: str | None,
        payload: Mapping[str, Any],
    ) -> Event:
        return self.memory.append(
            event_type=event_type,
            workflow_id=workflow_id,
            task_id=task_id,
            payload=payload,
        )

    @staticmethod
    def _validate_graph(packets: Sequence[WorkPacket]) -> dict[str, WorkPacket]:
        packet_map: dict[str, WorkPacket] = {}
        for packet in packets:
            if not packet.task_id:
                raise ValueError("task_id is required")
            if packet.task_id in packet_map:
                raise ValueError(f"duplicate task_id: {packet.task_id}")
            if packet.retries < 0:
                raise ValueError("retries must be non-negative")
            packet_map[packet.task_id] = packet

        known = set(packet_map)
        for packet in packets:
            missing = set(packet.depends_on) - known
            if missing:
                raise ValueError(
                    f"{packet.task_id} depends on unknown tasks: {sorted(missing)}"
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("workflow graph contains a cycle")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in packet_map[task_id].depends_on:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in sorted(packet_map):
            visit(task_id)
        return packet_map


def _json_native(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)
