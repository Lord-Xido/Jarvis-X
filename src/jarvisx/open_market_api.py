"""FastAPI surface for the Jarvis-X open verified-execution market."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .assembler import Assembler
from .open_market import (
    DuplicateResource,
    MarketBid,
    MarketError,
    MarketStateError,
    MarketTask,
    OpenMarketEngine,
    Provider,
    SettlementReceipt,
    UnknownResource,
)
from .parser import Parser
from .system_runtime import CAP_VM_EXECUTE, ResourceBudget, UtilityWeights

app = FastAPI(
    title="Jarvis-X Open Market",
    version="0.1.0",
    description=(
        "Open capability market for deterministic Jarvis-X plans. Providers advertise "
        "capabilities, buyers post bounded tasks, bids compete on quality/cost/latency/risk, "
        "and accounting settlement occurs only after verified committed execution."
    ),
)
market = OpenMarketEngine()


class ProviderCreate(BaseModel):
    provider_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=200)
    capabilities: list[str] = Field(min_length=1, max_length=64)
    metadata: dict[str, str] = Field(default_factory=dict)


class TaskCreate(BaseModel):
    task_id: str = Field(min_length=1, max_length=128)
    buyer_id: str = Field(min_length=1, max_length=128)
    capability: str = Field(min_length=1, max_length=128)
    max_price_units: int = Field(ge=1, le=10**15)
    max_cycles: int = Field(default=10_000, ge=1, le=1_000_000)
    max_program_words: int = Field(default=4_096, ge=1, le=100_000)
    max_candidates: int = Field(default=32, ge=1, le=1_000)
    quality_weight: float = Field(default=1.0, ge=0.0, le=1_000_000.0)
    cost_weight: float = Field(default=1.0, ge=0.0, le=1_000_000.0)
    latency_weight: float = Field(default=1.0, ge=0.0, le=1_000_000.0)
    risk_weight: float = Field(default=1.0, ge=0.0, le=1_000_000.0)


class BidCreate(BaseModel):
    bid_id: str = Field(min_length=1, max_length=128)
    provider_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=262_144)
    price_units: int = Field(ge=0, le=10**15)
    quality: float = Field(ge=0.0, le=1_000_000.0)
    latency: float = Field(ge=0.0, le=1_000_000_000.0)
    risk: float = Field(ge=0.0, le=1_000_000.0)
    required_capabilities: list[str] = Field(default_factory=lambda: [CAP_VM_EXECUTE])


def _assemble(source: str) -> tuple[int, ...]:
    return tuple(Assembler().assemble(Parser().parse(source)))


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, UnknownResource):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, DuplicateResource):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (MarketStateError, MarketError)):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, (TypeError, ValueError)):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="internal market failure")


def _provider_dict(provider: Provider) -> dict[str, Any]:
    return {
        "provider_id": provider.provider_id,
        "display_name": provider.display_name,
        "capabilities": sorted(provider.capabilities),
        "metadata": dict(provider.metadata),
    }


def _task_dict(task: MarketTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "buyer_id": task.buyer_id,
        "capability": task.capability,
        "max_price_units": task.max_price_units,
        "granted_capabilities": sorted(task.granted_capabilities),
        "budget": {
            "max_cycles": task.budget.max_cycles,
            "max_program_words": task.budget.max_program_words,
            "max_candidates": task.budget.max_candidates,
        },
        "weights": asdict(task.weights),
        "status": task.status.value,
        "awarded_bid_id": task.awarded_bid_id,
    }


def _bid_dict(bid: MarketBid, *, include_program: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "bid_id": bid.bid_id,
        "task_id": bid.task_id,
        "provider_id": bid.provider_id,
        "price_units": bid.price_units,
        "quality": bid.quality,
        "latency": bid.latency,
        "risk": bid.risk,
        "required_capabilities": sorted(bid.required_capabilities),
        "program_words": len(bid.program),
    }
    if include_program:
        result["program"] = list(bid.program)
    return result


def _settlement_dict(settlement: SettlementReceipt) -> dict[str, Any]:
    return {
        "settlement_id": settlement.settlement_id,
        "task_id": settlement.task_id,
        "bid_id": settlement.bid_id,
        "buyer_id": settlement.buyer_id,
        "provider_id": settlement.provider_id,
        "status": settlement.status.value,
        "gross_units": settlement.gross_units,
        "platform_fee_units": settlement.platform_fee_units,
        "provider_units": settlement.provider_units,
        "execution_status": settlement.execution_status,
        "execution_request_id": settlement.execution_request_id,
        "execution_state": settlement.state_dict(),
        "execution_state_hash": settlement.execution_state_hash,
        "vm_ledger_head": settlement.vm_ledger_head,
        "system_audit_head": settlement.system_audit_head,
        "market_ledger_head": settlement.market_ledger_head,
        "error_type": settlement.error_type,
        "error_message": settlement.error_message,
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    snapshot = market.snapshot()
    return {
        "status": "ok" if market.verify() else "degraded",
        "service": "jarvisx-open-market",
        "version": app.version,
        **snapshot,
    }


@app.get("/v1/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "market": {
            "provider_registry": True,
            "task_market": True,
            "competitive_bids": True,
            "deterministic_award": True,
            "verified_execution": True,
            "settlement_mode": "internal integer accounting units only",
            "external_payment_rail": False,
        },
        "runtime": {
            "engine": "SystemRuntime/CodexVM",
            "plan_execute_commit_separation": True,
            "cycle_bounded": True,
            "audit_linked_receipts": True,
        },
        "security_boundary": (
            "Reference integration only. The current VM cycle sandbox is not hostile-code "
            "process isolation; do not accept untrusted public bytecode in production."
        ),
    }


@app.post("/v1/providers")
def register_provider(request: ProviderCreate) -> dict[str, Any]:
    try:
        provider = market.register_provider(
            Provider(
                provider_id=request.provider_id,
                display_name=request.display_name,
                capabilities=frozenset(request.capabilities),
                metadata=tuple(request.metadata.items()),
            )
        )
        return _provider_dict(provider)
    except Exception as exc:
        raise _fail(exc) from exc


@app.get("/v1/providers")
def list_providers(limit: int = Query(default=100, ge=1, le=1_000)) -> dict[str, Any]:
    providers = sorted(market.providers.values(), key=lambda item: item.provider_id)[:limit]
    return {"providers": [_provider_dict(provider) for provider in providers]}


@app.post("/v1/tasks")
def create_task(request: TaskCreate) -> dict[str, Any]:
    try:
        task = market.create_task(
            MarketTask(
                task_id=request.task_id,
                buyer_id=request.buyer_id,
                capability=request.capability,
                max_price_units=request.max_price_units,
                budget=ResourceBudget(
                    max_cycles=request.max_cycles,
                    max_program_words=request.max_program_words,
                    max_candidates=request.max_candidates,
                ),
                weights=UtilityWeights(
                    quality=request.quality_weight,
                    cost=request.cost_weight,
                    latency=request.latency_weight,
                    risk=request.risk_weight,
                ),
            )
        )
        return _task_dict(task)
    except Exception as exc:
        raise _fail(exc) from exc


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    try:
        return _task_dict(market.tasks[task_id])
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"task {task_id!r} does not exist") from exc


@app.post("/v1/tasks/{task_id}/bids")
def submit_bid(task_id: str, request: BidCreate) -> dict[str, Any]:
    try:
        bid = market.submit_bid(
            MarketBid(
                bid_id=request.bid_id,
                task_id=task_id,
                provider_id=request.provider_id,
                program=_assemble(request.source),
                price_units=request.price_units,
                quality=request.quality,
                latency=request.latency,
                risk=request.risk,
                required_capabilities=frozenset(request.required_capabilities),
            )
        )
        return _bid_dict(bid)
    except Exception as exc:
        raise _fail(exc) from exc


@app.get("/v1/tasks/{task_id}/bids")
def list_bids(task_id: str) -> dict[str, Any]:
    try:
        bids = market.bids_for_task(task_id)
        return {"bids": [_bid_dict(bid) for bid in bids]}
    except Exception as exc:
        raise _fail(exc) from exc


@app.post("/v1/tasks/{task_id}/award")
def award_task(task_id: str) -> dict[str, Any]:
    try:
        return _bid_dict(market.award_task(task_id))
    except Exception as exc:
        raise _fail(exc) from exc


@app.post("/v1/tasks/{task_id}/execute")
def execute_task(task_id: str) -> dict[str, Any]:
    try:
        return _settlement_dict(market.execute_task(task_id))
    except Exception as exc:
        raise _fail(exc) from exc


@app.get("/v1/tasks/{task_id}/settlement")
def get_settlement(task_id: str) -> dict[str, Any]:
    try:
        settlement = market.settlement_for_task(task_id)
        if settlement is None:
            raise HTTPException(status_code=404, detail="task has not settled")
        return _settlement_dict(settlement)
    except HTTPException:
        raise
    except Exception as exc:
        raise _fail(exc) from exc


@app.get("/v1/market")
def market_status() -> dict[str, Any]:
    return market.snapshot()


@app.get("/v1/ledger")
def market_ledger(limit: int = Query(default=100, ge=1, le=10_000)) -> dict[str, Any]:
    chain = market.ledger.chain[-limit:]
    return {
        "valid": market.ledger.verify(),
        "entries": chain,
        "head": chain[-1]["hash"] if chain else None,
    }
