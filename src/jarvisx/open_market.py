from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, cast

from .ledger import OmegaLedger
from .system_runtime import (
    CAP_VM_EXECUTE,
    DeterministicPlanner,
    ExecutionRequest,
    ExecutionStatus,
    PlanCandidate,
    ResourceBudget,
    SystemRuntime,
    UtilityWeights,
)

MARKET_PROVIDER_OPCODE = 0x2000
MARKET_TASK_OPCODE = 0x2001
MARKET_BID_OPCODE = 0x2002
MARKET_AWARD_OPCODE = 0x2003
MARKET_SETTLEMENT_OPCODE = 0x2004
MARKET_FAILURE_OPCODE = 0x2005


class MarketError(RuntimeError):
    """Base class for open-market failures."""


class DuplicateResource(MarketError):
    """Raised when an immutable market identifier is reused."""


class UnknownResource(MarketError):
    """Raised when a provider, task, bid or settlement cannot be found."""


class MarketStateError(MarketError):
    """Raised when an operation is invalid for the current market state."""


class TaskStatus(str, Enum):
    OPEN = "open"
    AWARDED = "awarded"
    SETTLED = "settled"
    FAILED = "failed"


class SettlementStatus(str, Enum):
    SETTLED = "settled"
    EXECUTION_FAILED = "execution_failed"


def _nonempty(name: str, value: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} cannot be empty")
    return normalized


def _finite_nonnegative(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


@dataclass(frozen=True)
class Provider:
    provider_id: str
    display_name: str
    capabilities: frozenset[str]
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _nonempty("provider_id", self.provider_id))
        object.__setattr__(self, "display_name", _nonempty("display_name", self.display_name))
        capabilities = frozenset(_nonempty("capability", value) for value in self.capabilities)
        if not capabilities:
            raise ValueError("capabilities cannot be empty")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "metadata",
            tuple(sorted((_nonempty("metadata key", k), str(v)) for k, v in self.metadata)),
        )


@dataclass
class MarketTask:
    task_id: str
    buyer_id: str
    capability: str
    max_price_units: int
    granted_capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({CAP_VM_EXECUTE})
    )
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    weights: UtilityWeights = field(default_factory=UtilityWeights)
    status: TaskStatus = TaskStatus.OPEN
    awarded_bid_id: str | None = None

    def __post_init__(self) -> None:
        self.task_id = _nonempty("task_id", self.task_id)
        self.buyer_id = _nonempty("buyer_id", self.buyer_id)
        self.capability = _nonempty("capability", self.capability)
        if (
            not isinstance(self.max_price_units, int)
            or isinstance(self.max_price_units, bool)
            or self.max_price_units < 1
        ):
            raise ValueError("max_price_units must be a positive integer")
        self.granted_capabilities = frozenset(self.granted_capabilities)
        if CAP_VM_EXECUTE not in self.granted_capabilities:
            raise ValueError("tasks must grant vm.execute")


@dataclass(frozen=True)
class MarketBid:
    bid_id: str
    task_id: str
    provider_id: str
    program: tuple[int, ...]
    price_units: int
    quality: float
    latency: float
    risk: float
    required_capabilities: frozenset[str] = field(
        default_factory=lambda: frozenset({CAP_VM_EXECUTE})
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "bid_id", _nonempty("bid_id", self.bid_id))
        object.__setattr__(self, "task_id", _nonempty("task_id", self.task_id))
        object.__setattr__(self, "provider_id", _nonempty("provider_id", self.provider_id))
        program = tuple(self.program)
        if not program:
            raise ValueError("program cannot be empty")
        for index, word in enumerate(program):
            if not isinstance(word, int) or isinstance(word, bool):
                raise TypeError(f"program word {index} is not an integer")
            if not 0 <= word <= 0xFFFFFFFFFFFFFFFF:
                raise ValueError(f"program word {index} is outside unsigned 64-bit range")
        object.__setattr__(self, "program", program)
        if (
            not isinstance(self.price_units, int)
            or isinstance(self.price_units, bool)
            or self.price_units < 0
        ):
            raise ValueError("price_units must be a non-negative integer")
        object.__setattr__(self, "quality", _finite_nonnegative("quality", self.quality))
        object.__setattr__(self, "latency", _finite_nonnegative("latency", self.latency))
        object.__setattr__(self, "risk", _finite_nonnegative("risk", self.risk))
        required = frozenset(self.required_capabilities)
        if CAP_VM_EXECUTE not in required:
            required = required | {CAP_VM_EXECUTE}
        object.__setattr__(self, "required_capabilities", required)


@dataclass(frozen=True)
class SettlementReceipt:
    settlement_id: str
    task_id: str
    bid_id: str
    buyer_id: str
    provider_id: str
    status: SettlementStatus
    gross_units: int
    platform_fee_units: int
    provider_units: int
    execution_status: str
    execution_request_id: str
    execution_state: tuple[tuple[str, int], ...]
    execution_state_hash: str | None
    vm_ledger_head: str | None
    system_audit_head: str
    market_ledger_head: str
    error_type: str | None = None
    error_message: str | None = None

    def state_dict(self) -> dict[str, int]:
        return dict(self.execution_state)


class OpenMarketEngine:
    """Open capability market settled only by verified Jarvis-X execution.

    The engine intentionally does not move external money. ``*_units`` are
    integer accounting units that can later be mapped to a regulated payment
    rail by an adapter. A successful market settlement requires a committed
    ``SystemRuntime`` receipt; failed or rejected execution settles zero units.
    """

    def __init__(
        self,
        *,
        runtime: SystemRuntime | None = None,
        fee_bps: int = 250,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(fee_bps, int) or isinstance(fee_bps, bool) or not 0 <= fee_bps <= 10_000:
            raise ValueError("fee_bps must be an integer between 0 and 10000")
        self.runtime = runtime or SystemRuntime()
        self.fee_bps = fee_bps
        self.ledger = OmegaLedger(clock_ns=clock_ns)
        self.providers: dict[str, Provider] = {}
        self.tasks: dict[str, MarketTask] = {}
        self.bids: dict[str, MarketBid] = {}
        self.task_bids: dict[str, list[str]] = {}
        self.settlements: dict[str, SettlementReceipt] = {}
        self._lock = threading.RLock()

    def register_provider(self, provider: Provider) -> Provider:
        with self._lock:
            if provider.provider_id in self.providers:
                raise DuplicateResource(f"provider {provider.provider_id!r} already exists")
            self.providers[provider.provider_id] = provider
            self._log(
                MARKET_PROVIDER_OPCODE,
                {
                    "event": "provider.registered",
                    "provider_id": provider.provider_id,
                    "display_name": provider.display_name,
                    "capabilities": sorted(provider.capabilities),
                },
            )
            return provider

    def create_task(self, task: MarketTask) -> MarketTask:
        with self._lock:
            if task.task_id in self.tasks:
                raise DuplicateResource(f"task {task.task_id!r} already exists")
            self.tasks[task.task_id] = task
            self.task_bids[task.task_id] = []
            self._log(
                MARKET_TASK_OPCODE,
                {
                    "event": "task.created",
                    "task_id": task.task_id,
                    "buyer_id": task.buyer_id,
                    "capability": task.capability,
                    "max_price_units": task.max_price_units,
                    "max_cycles": task.budget.max_cycles,
                    "max_program_words": task.budget.max_program_words,
                },
            )
            return task

    def submit_bid(self, bid: MarketBid) -> MarketBid:
        with self._lock:
            if bid.bid_id in self.bids:
                raise DuplicateResource(f"bid {bid.bid_id!r} already exists")
            task = self._task(bid.task_id)
            if task.status is not TaskStatus.OPEN:
                raise MarketStateError(f"task {task.task_id!r} is not open")
            provider = self._provider(bid.provider_id)
            if task.capability not in provider.capabilities:
                raise MarketStateError(
                    f"provider {provider.provider_id!r} does not advertise capability {task.capability!r}"
                )
            if bid.price_units > task.max_price_units:
                raise MarketStateError(
                    f"bid price {bid.price_units} exceeds task maximum {task.max_price_units}"
                )
            if len(bid.program) > task.budget.max_program_words:
                raise MarketStateError(
                    f"bid program length {len(bid.program)} exceeds task budget "
                    f"{task.budget.max_program_words}"
                )
            if not bid.required_capabilities.issubset(task.granted_capabilities):
                missing = sorted(bid.required_capabilities - task.granted_capabilities)
                raise MarketStateError(
                    "bid requires capabilities not granted by task: " + ", ".join(missing)
                )
            self.bids[bid.bid_id] = bid
            self.task_bids[task.task_id].append(bid.bid_id)
            self._log(
                MARKET_BID_OPCODE,
                {
                    "event": "bid.submitted",
                    "bid_id": bid.bid_id,
                    "task_id": bid.task_id,
                    "provider_id": bid.provider_id,
                    "price_units": bid.price_units,
                    "quality": bid.quality,
                    "latency": bid.latency,
                    "risk": bid.risk,
                    "program_words": len(bid.program),
                },
            )
            return bid

    def award_task(self, task_id: str) -> MarketBid:
        with self._lock:
            task = self._task(task_id)
            if task.status is TaskStatus.SETTLED:
                return self._bid(task.awarded_bid_id)
            if task.status is TaskStatus.FAILED:
                raise MarketStateError(f"task {task.task_id!r} has failed")
            if task.status is TaskStatus.AWARDED:
                return self._bid(task.awarded_bid_id)
            bid_ids = tuple(self.task_bids.get(task.task_id, ()))
            if not bid_ids:
                raise MarketStateError(f"task {task.task_id!r} has no bids")

            candidates = tuple(self._plan_candidate(self.bids[bid_id]) for bid_id in bid_ids)
            planner = DeterministicPlanner(weights=task.weights)
            selected = planner.select(
                candidates,
                granted_capabilities=task.granted_capabilities,
                budget=task.budget,
            )
            winning_bid = self._bid(selected.plan_id)
            task.awarded_bid_id = winning_bid.bid_id
            task.status = TaskStatus.AWARDED
            self._log(
                MARKET_AWARD_OPCODE,
                {
                    "event": "task.awarded",
                    "task_id": task.task_id,
                    "bid_id": winning_bid.bid_id,
                    "provider_id": winning_bid.provider_id,
                    "price_units": winning_bid.price_units,
                    "utility": planner.utility(selected),
                },
            )
            return winning_bid

    def execute_task(self, task_id: str) -> SettlementReceipt:
        with self._lock:
            existing = self.settlements.get(task_id)
            if existing is not None:
                return existing
            task = self._task(task_id)
            if task.status is TaskStatus.FAILED:
                raise MarketStateError(f"task {task.task_id!r} has failed")
            bid = self.award_task(task.task_id)
            request_id = f"market:{task.task_id}:{bid.bid_id}"
            receipt = self.runtime.execute(
                ExecutionRequest(
                    request_id=request_id,
                    program=bid.program,
                    granted_capabilities=task.granted_capabilities,
                    required_capabilities=bid.required_capabilities,
                    budget=task.budget,
                    plan_id=bid.bid_id,
                )
            )

            if receipt.status is ExecutionStatus.COMMITTED and receipt.committed:
                gross = bid.price_units
                fee = (gross * self.fee_bps) // 10_000
                provider_units = gross - fee
                status = SettlementStatus.SETTLED
                task.status = TaskStatus.SETTLED
                opcode = MARKET_SETTLEMENT_OPCODE
            else:
                gross = 0
                fee = 0
                provider_units = 0
                status = SettlementStatus.EXECUTION_FAILED
                task.status = TaskStatus.FAILED
                opcode = MARKET_FAILURE_OPCODE

            event = self._log(
                opcode,
                {
                    "event": "task.settled" if status is SettlementStatus.SETTLED else "task.failed",
                    "task_id": task.task_id,
                    "bid_id": bid.bid_id,
                    "provider_id": bid.provider_id,
                    "buyer_id": task.buyer_id,
                    "gross_units": gross,
                    "platform_fee_units": fee,
                    "provider_units": provider_units,
                    "execution_status": receipt.status.value,
                    "execution_request_id": receipt.request_id,
                    "execution_state_hash": receipt.state_hash,
                    "system_audit_head": receipt.audit_head,
                    "error_type": receipt.error_type,
                },
            )
            settlement = SettlementReceipt(
                settlement_id=f"settlement:{task.task_id}",
                task_id=task.task_id,
                bid_id=bid.bid_id,
                buyer_id=task.buyer_id,
                provider_id=bid.provider_id,
                status=status,
                gross_units=gross,
                platform_fee_units=fee,
                provider_units=provider_units,
                execution_status=receipt.status.value,
                execution_request_id=receipt.request_id,
                execution_state=receipt.state,
                execution_state_hash=receipt.state_hash,
                vm_ledger_head=receipt.vm_ledger_head,
                system_audit_head=receipt.audit_head,
                market_ledger_head=str(event["hash"]),
                error_type=receipt.error_type,
                error_message=receipt.error_message,
            )
            self.settlements[task.task_id] = settlement
            return settlement

    def bids_for_task(self, task_id: str) -> tuple[MarketBid, ...]:
        with self._lock:
            self._task(task_id)
            return tuple(self.bids[bid_id] for bid_id in self.task_bids.get(task_id, ()))

    def settlement_for_task(self, task_id: str) -> SettlementReceipt | None:
        with self._lock:
            self._task(task_id)
            return self.settlements.get(task_id)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            status_counts = {status.value: 0 for status in TaskStatus}
            for task in self.tasks.values():
                status_counts[task.status.value] += 1
            gross = sum(item.gross_units for item in self.settlements.values())
            fees = sum(item.platform_fee_units for item in self.settlements.values())
            return {
                "providers": len(self.providers),
                "tasks": len(self.tasks),
                "bids": len(self.bids),
                "settlements": len(self.settlements),
                "task_status": status_counts,
                "gross_verified_units": gross,
                "platform_fee_units": fees,
                "market_ledger_entries": len(self.ledger.chain),
                "market_ledger_valid": self.ledger.verify(),
                "runtime_valid": self.runtime.verify(),
            }

    def verify(self) -> bool:
        with self._lock:
            if not self.ledger.verify() or not self.runtime.verify():
                return False
            for task_id, settlement in self.settlements.items():
                task = self.tasks.get(task_id)
                bid = self.bids.get(settlement.bid_id)
                if task is None or bid is None or task.awarded_bid_id != bid.bid_id:
                    return False
                if settlement.status is SettlementStatus.SETTLED:
                    if task.status is not TaskStatus.SETTLED:
                        return False
                    if settlement.gross_units != bid.price_units:
                        return False
                    expected_fee = (bid.price_units * self.fee_bps) // 10_000
                    if settlement.platform_fee_units != expected_fee:
                        return False
                    if settlement.provider_units != bid.price_units - expected_fee:
                        return False
                    if settlement.execution_state_hash is None:
                        return False
                else:
                    if task.status is not TaskStatus.FAILED:
                        return False
                    if any(
                        (settlement.gross_units, settlement.platform_fee_units, settlement.provider_units)
                    ):
                        return False
            return True

    def _plan_candidate(self, bid: MarketBid) -> PlanCandidate:
        return PlanCandidate(
            plan_id=bid.bid_id,
            program=bid.program,
            quality=bid.quality,
            cost=float(bid.price_units),
            latency=bid.latency,
            risk=bid.risk,
            required_capabilities=bid.required_capabilities,
        )

    def _provider(self, provider_id: str) -> Provider:
        try:
            return self.providers[provider_id]
        except KeyError as exc:
            raise UnknownResource(f"provider {provider_id!r} does not exist") from exc

    def _task(self, task_id: str) -> MarketTask:
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise UnknownResource(f"task {task_id!r} does not exist") from exc

    def _bid(self, bid_id: str | None) -> MarketBid:
        if bid_id is None:
            raise UnknownResource("bid is not assigned")
        try:
            return self.bids[bid_id]
        except KeyError as exc:
            raise UnknownResource(f"bid {bid_id!r} does not exist") from exc

    def _log(self, opcode: int, state: dict[str, object]) -> dict[str, object]:
        return cast(dict[str, object], self.ledger.log(state, opcode))


__all__ = [
    "DuplicateResource",
    "MarketBid",
    "MarketError",
    "MarketStateError",
    "MarketTask",
    "OpenMarketEngine",
    "Provider",
    "SettlementReceipt",
    "SettlementStatus",
    "TaskStatus",
    "UnknownResource",
]
