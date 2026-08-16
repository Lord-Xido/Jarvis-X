from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

SYSTEM_COMMIT_OPCODE = 0x1000
SYSTEM_REJECT_OPCODE = 0x1001
SYSTEM_FAILURE_OPCODE = 0x1002
CAP_VM_EXECUTE = "vm.execute"
CAP_VM_REFLEX = "vm.reflex"
DEFAULT_RUNTIME_CAPABILITIES = frozenset({CAP_VM_EXECUTE, CAP_VM_REFLEX})


class RuntimeControlError(RuntimeError):
    """Base class for bounded control-plane failures."""


class PolicyRejection(RuntimeControlError):
    """Raised when a request is outside the Lambda capability projection."""


class VerificationError(RuntimeControlError):
    """Raised when tentative execution cannot be verified for commit."""


class RequestCollisionError(RuntimeControlError):
    """Raised when a request id is reused for different request contents."""


class NoAdmissiblePlan(RuntimeControlError):
    """Raised when no candidate plan survives policy and resource projection."""


class ExecutionStatus(str, Enum):
    COMMITTED = "committed"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass(frozen=True)
class ResourceBudget:
    """Deterministic task budget enforced before and during VM execution."""

    max_cycles: int = 10_000
    max_program_words: int = 4_096
    max_candidates: int = 32

    def __post_init__(self) -> None:
        for name, value in (
            ("max_cycles", self.max_cycles),
            ("max_program_words", self.max_program_words),
            ("max_candidates", self.max_candidates),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ExecutionRequest:
    """Immutable request crossing from the control plane into execution."""

    request_id: str
    program: tuple[int, ...]
    granted_capabilities: frozenset[str]
    required_capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({CAP_VM_EXECUTE})
    )
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    enable_reflex: bool = False
    plan_id: str | None = None

    def __post_init__(self) -> None:
        request_id = str(self.request_id).strip()
        if not request_id:
            raise ValueError("request_id cannot be empty")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "program", tuple(self.program))
        object.__setattr__(self, "granted_capabilities", frozenset(self.granted_capabilities))
        required = frozenset(self.required_capabilities)
        if self.enable_reflex:
            required = required | {CAP_VM_REFLEX}
        object.__setattr__(self, "required_capabilities", required)
        if self.plan_id is not None:
            plan_id = str(self.plan_id).strip()
            if not plan_id:
                raise ValueError("plan_id cannot be blank")
            object.__setattr__(self, "plan_id", plan_id)


@dataclass(frozen=True)
class ExecutionReceipt:
    """Immutable, audit-linked result of one system request."""

    request_id: str
    request_fingerprint: str
    status: ExecutionStatus
    committed: bool
    cycles: int
    state: tuple[tuple[str, int], ...]
    state_hash: str | None
    vm_ledger_head: str | None
    audit_head: str
    plan_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def state_dict(self) -> dict[str, int]:
        return dict(self.state)


@dataclass(frozen=True)
class UtilityWeights:
    quality: float = 1.0
    cost: float = 1.0
    latency: float = 1.0
    risk: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("quality", self.quality),
            ("cost", self.cost),
            ("latency", self.latency),
            ("risk", self.risk),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} weight must be finite and non-negative")


@dataclass(frozen=True)
class PlanCandidate:
    """Pure proposal. A plan has no authority to create side effects itself."""

    plan_id: str
    program: tuple[int, ...]
    quality: float
    cost: float
    latency: float
    risk: float
    required_capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({CAP_VM_EXECUTE})
    )
    enable_reflex: bool = False

    def __post_init__(self) -> None:
        plan_id = str(self.plan_id).strip()
        if not plan_id:
            raise ValueError("plan_id cannot be empty")
        object.__setattr__(self, "plan_id", plan_id)
        object.__setattr__(self, "program", tuple(self.program))
        required = frozenset(self.required_capabilities)
        if self.enable_reflex:
            required = required | {CAP_VM_REFLEX}
        object.__setattr__(self, "required_capabilities", required)
        for name, value in (
            ("quality", self.quality),
            ("cost", self.cost),
            ("latency", self.latency),
            ("risk", self.risk),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in (
            ("cost", self.cost),
            ("latency", self.latency),
            ("risk", self.risk),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")


class CapabilityPolicy:
    """Fail-closed Lambda projection for task-level capabilities."""

    def __init__(self, allowed_capabilities: Iterable[str] = DEFAULT_RUNTIME_CAPABILITIES) -> None:
        allowed = frozenset(str(capability) for capability in allowed_capabilities)
        if not allowed:
            raise ValueError("allowed_capabilities cannot be empty")
        self.allowed_capabilities = allowed

    def project(self, required: Iterable[str], granted: Iterable[str]) -> frozenset[str]:
        required_set = frozenset(str(capability) for capability in required)
        granted_set = frozenset(str(capability) for capability in granted)
        unknown = required_set - self.allowed_capabilities
        if unknown:
            raise PolicyRejection(
                "request requires unsupported capabilities: " + ", ".join(sorted(unknown))
            )
        missing = required_set - granted_set
        if missing:
            raise PolicyRejection(
                "request exceeds granted capabilities: " + ", ".join(sorted(missing))
            )
        return required_set


class DeterministicPlanner:
    """Bounded, side-effect-free candidate selector.

    J(P_i) = w_q Q_i - w_c C_i - w_l L_i - w_r R_i.
    Ties are resolved lexicographically by plan id to keep selection reproducible.
    """

    def __init__(
        self,
        *,
        policy: CapabilityPolicy | None = None,
        weights: UtilityWeights | None = None,
    ) -> None:
        self.policy = policy or CapabilityPolicy()
        self.weights = weights or UtilityWeights()

    def utility(self, candidate: PlanCandidate) -> float:
        return (
            self.weights.quality * candidate.quality
            - self.weights.cost * candidate.cost
            - self.weights.latency * candidate.latency
            - self.weights.risk * candidate.risk
        )

    def select(
        self,
        candidates: Iterable[PlanCandidate],
        *,
        granted_capabilities: Iterable[str],
        budget: ResourceBudget,
    ) -> PlanCandidate:
        materialized = tuple(candidates)
        if not materialized:
            raise NoAdmissiblePlan("no candidate plans were supplied")
        if len(materialized) > budget.max_candidates:
            raise NoAdmissiblePlan(
                f"candidate count {len(materialized)} exceeds budget {budget.max_candidates}"
            )

        admissible: list[PlanCandidate] = []
        for candidate in materialized:
            if not candidate.program or len(candidate.program) > budget.max_program_words:
                continue
            try:
                self.policy.project(candidate.required_capabilities, granted_capabilities)
            except PolicyRejection:
                continue
            admissible.append(candidate)

        if not admissible:
            raise NoAdmissiblePlan("no candidate survived capability and resource projection")

        return min(admissible, key=lambda candidate: (-self.utility(candidate), candidate.plan_id))


class _LedgerProtocol(Protocol):
    chain: list[dict[str, Any]]

    def checkpoint(self) -> int: ...

    def restore(self, checkpoint: int) -> None: ...

    def log(self, state: Mapping[str, Any], opcode: int) -> dict[str, Any]: ...

    def verify(self) -> bool: ...


class _VMProtocol(Protocol):
    cycles: int
    ledger: _LedgerProtocol

    def load(self, bytecode: Iterable[int]) -> None: ...

    def run(self) -> dict[str, int]: ...


VMFactory = Callable[..., _VMProtocol]


def _default_vm_factory(*, enable_reflex: bool, max_cycles: int) -> _VMProtocol:
    from .core import CodexVM

    return CodexVM(enable_reflex=enable_reflex, max_cycles=max_cycles)


def _default_audit_ledger() -> _LedgerProtocol:
    from .ledger import OmegaLedger

    return OmegaLedger()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class SystemRuntime:
    """Task-level transactional boundary around the canonical CodexVM.

    Planning remains outside this class. Execution happens in an isolated VM.
    The VM result becomes authoritative only after capability projection,
    bounded execution, ledger verification, result hashing and system-audit
    append all succeed. Consequently Plan != Execute and Execute != Commit.
    """

    def __init__(
        self,
        *,
        policy: CapabilityPolicy | None = None,
        vm_factory: VMFactory = _default_vm_factory,
        audit_ledger: _LedgerProtocol | None = None,
    ) -> None:
        self.policy = policy or CapabilityPolicy()
        self._vm_factory = vm_factory
        self.audit = audit_ledger or _default_audit_ledger()
        self._receipts: dict[str, ExecutionReceipt] = {}
        self._fingerprints: dict[str, str] = {}
        self._committed_states: dict[str, tuple[tuple[str, int], ...]] = {}

    @staticmethod
    def request_fingerprint(request: ExecutionRequest) -> str:
        return _sha256(
            {
                "request_id": request.request_id,
                "program": list(request.program),
                "granted_capabilities": sorted(request.granted_capabilities),
                "required_capabilities": sorted(request.required_capabilities),
                "budget": {
                    "max_cycles": request.budget.max_cycles,
                    "max_program_words": request.budget.max_program_words,
                    "max_candidates": request.budget.max_candidates,
                },
                "enable_reflex": request.enable_reflex,
                "plan_id": request.plan_id,
            }
        )

    def committed_state(self, request_id: str) -> dict[str, int] | None:
        state = self._committed_states.get(request_id)
        return None if state is None else dict(state)

    def receipt(self, request_id: str) -> ExecutionReceipt | None:
        return self._receipts.get(request_id)

    def verify(self) -> bool:
        if not self.audit.verify():
            return False
        for request_id, state in self._committed_states.items():
            receipt = self._receipts.get(request_id)
            if receipt is None or not receipt.committed:
                return False
            if receipt.state_hash != _sha256(dict(state)):
                return False
        return True

    def execute(self, request: ExecutionRequest) -> ExecutionReceipt:
        fingerprint = self.request_fingerprint(request)
        cached = self._receipts.get(request.request_id)
        if cached is not None:
            if cached.request_fingerprint != fingerprint:
                raise RequestCollisionError(
                    f"request_id {request.request_id!r} was reused with different contents"
                )
            return cached

        known_fingerprint = self._fingerprints.get(request.request_id)
        if known_fingerprint is not None and known_fingerprint != fingerprint:
            raise RequestCollisionError(
                f"request_id {request.request_id!r} was reused with different contents"
            )
        self._fingerprints[request.request_id] = fingerprint

        try:
            self._validate_request(request)
            self.policy.project(request.required_capabilities, request.granted_capabilities)
            vm = self._vm_factory(
                enable_reflex=request.enable_reflex,
                max_cycles=request.budget.max_cycles,
            )
            vm.load(request.program)
            tentative_state = vm.run()
            self._verify_tentative_result(request, vm, tentative_state)
            return self._commit(request, fingerprint, vm, tentative_state)
        except PolicyRejection as exc:
            return self._record_noncommit(
                request,
                fingerprint,
                status=ExecutionStatus.REJECTED,
                error=exc,
            )
        except Exception as exc:
            return self._record_noncommit(
                request,
                fingerprint,
                status=ExecutionStatus.FAILED,
                error=exc,
            )

    def execute_plans(
        self,
        *,
        request_id: str,
        candidates: Iterable[PlanCandidate],
        granted_capabilities: Iterable[str],
        budget: ResourceBudget | None = None,
        planner: DeterministicPlanner | None = None,
    ) -> ExecutionReceipt:
        resolved_budget = budget or ResourceBudget()
        resolved_grants = frozenset(granted_capabilities)
        selected = (planner or DeterministicPlanner(policy=self.policy)).select(
            candidates,
            granted_capabilities=resolved_grants,
            budget=resolved_budget,
        )
        request = ExecutionRequest(
            request_id=request_id,
            program=selected.program,
            granted_capabilities=resolved_grants,
            required_capabilities=selected.required_capabilities,
            budget=resolved_budget,
            enable_reflex=selected.enable_reflex,
            plan_id=selected.plan_id,
        )
        return self.execute(request)

    @staticmethod
    def _validate_request(request: ExecutionRequest) -> None:
        if not request.program:
            raise ValueError("program cannot be empty")
        if len(request.program) > request.budget.max_program_words:
            raise PolicyRejection(
                f"program length {len(request.program)} exceeds budget "
                f"{request.budget.max_program_words}"
            )
        for index, word in enumerate(request.program):
            if not isinstance(word, int) or isinstance(word, bool):
                raise TypeError(f"program word {index} is not an integer")
            if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"program word {index} is outside unsigned 64-bit range")

    @staticmethod
    def _verify_tentative_result(
        request: ExecutionRequest,
        vm: _VMProtocol,
        state: Mapping[str, Any],
    ) -> None:
        if vm.cycles > request.budget.max_cycles:
            raise VerificationError("VM exceeded the requested cycle budget")
        if not isinstance(state, Mapping):
            raise VerificationError("VM returned a non-mapping authoritative state")
        if not vm.ledger.verify():
            raise VerificationError("VM ledger verification failed")

    def _commit(
        self,
        request: ExecutionRequest,
        fingerprint: str,
        vm: _VMProtocol,
        state: Mapping[str, int],
    ) -> ExecutionReceipt:
        state_items = tuple(sorted((str(key), int(value)) for key, value in state.items()))
        state_hash = _sha256(dict(state_items))
        vm_ledger_head = vm.ledger.chain[-1]["hash"] if vm.ledger.chain else None
        audit_state = {
            "request_id": request.request_id,
            "request_fingerprint": fingerprint,
            "status": ExecutionStatus.COMMITTED.value,
            "committed": True,
            "cycles": int(vm.cycles),
            "state_hash": state_hash,
            "vm_ledger_head": vm_ledger_head,
            "plan_id": request.plan_id,
        }

        checkpoint = self.audit.checkpoint()
        try:
            entry = self.audit.log(audit_state, SYSTEM_COMMIT_OPCODE)
            if not self.audit.verify():
                raise VerificationError("system audit verification failed after commit append")
            receipt = ExecutionReceipt(
                request_id=request.request_id,
                request_fingerprint=fingerprint,
                status=ExecutionStatus.COMMITTED,
                committed=True,
                cycles=int(vm.cycles),
                state=state_items,
                state_hash=state_hash,
                vm_ledger_head=vm_ledger_head,
                audit_head=str(entry["hash"]),
                plan_id=request.plan_id,
            )
            self._committed_states[request.request_id] = state_items
            self._receipts[request.request_id] = receipt
            return receipt
        except Exception:
            self.audit.restore(checkpoint)
            self._committed_states.pop(request.request_id, None)
            self._receipts.pop(request.request_id, None)
            raise

    def _record_noncommit(
        self,
        request: ExecutionRequest,
        fingerprint: str,
        *,
        status: ExecutionStatus,
        error: Exception,
    ) -> ExecutionReceipt:
        opcode = SYSTEM_REJECT_OPCODE if status is ExecutionStatus.REJECTED else SYSTEM_FAILURE_OPCODE
        audit_state = {
            "request_id": request.request_id,
            "request_fingerprint": fingerprint,
            "status": status.value,
            "committed": False,
            "cycles": 0,
            "state_hash": None,
            "vm_ledger_head": None,
            "plan_id": request.plan_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        checkpoint = self.audit.checkpoint()
        try:
            entry = self.audit.log(audit_state, opcode)
            if not self.audit.verify():
                raise VerificationError("system audit verification failed after non-commit append")
            receipt = ExecutionReceipt(
                request_id=request.request_id,
                request_fingerprint=fingerprint,
                status=status,
                committed=False,
                cycles=0,
                state=(),
                state_hash=None,
                vm_ledger_head=None,
                audit_head=str(entry["hash"]),
                plan_id=request.plan_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
            self._receipts[request.request_id] = receipt
            return receipt
        except Exception:
            self.audit.restore(checkpoint)
            self._receipts.pop(request.request_id, None)
            raise
