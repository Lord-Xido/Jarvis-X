import pytest

from jarvisx.assembler import Assembler
from jarvisx.open_market import (
    MarketBid,
    MarketStateError,
    MarketTask,
    OpenMarketEngine,
    Provider,
    SettlementStatus,
    TaskStatus,
)
from jarvisx.parser import Parser
from jarvisx.system_runtime import ResourceBudget, UtilityWeights


def assemble(source: str) -> tuple[int, ...]:
    return tuple(Assembler().assemble(Parser().parse(source)))


def provider(provider_id: str, *capabilities: str) -> Provider:
    return Provider(
        provider_id=provider_id,
        display_name=provider_id.title(),
        capabilities=frozenset(capabilities),
    )


def test_open_market_end_to_end_awards_executes_and_settles_verified_bid() -> None:
    market = OpenMarketEngine(fee_bps=250, clock_ns=lambda: 1)
    market.register_provider(provider("alpha", "optimization"))
    market.register_provider(provider("beta", "optimization"))
    market.create_task(
        MarketTask(
            task_id="task-1",
            buyer_id="factory-1",
            capability="optimization",
            max_price_units=2_000,
            weights=UtilityWeights(quality=100.0, cost=0.01, latency=1.0, risk=10.0),
        )
    )
    market.submit_bid(
        MarketBid(
            bid_id="bid-alpha",
            task_id="task-1",
            provider_id="alpha",
            program=assemble("SET A 10\nHALT"),
            price_units=1_000,
            quality=0.80,
            latency=2.0,
            risk=0.20,
        )
    )
    market.submit_bid(
        MarketBid(
            bid_id="bid-beta",
            task_id="task-1",
            provider_id="beta",
            program=assemble("SET A 42\nHALT"),
            price_units=1_200,
            quality=0.98,
            latency=1.0,
            risk=0.05,
        )
    )

    settlement = market.execute_task("task-1")

    assert settlement.status is SettlementStatus.SETTLED
    assert settlement.bid_id == "bid-beta"
    assert settlement.provider_id == "beta"
    assert settlement.gross_units == 1_200
    assert settlement.platform_fee_units == 30
    assert settlement.provider_units == 1_170
    assert settlement.state_dict()["A"] == 42
    assert settlement.execution_state_hash is not None
    assert settlement.vm_ledger_head is not None
    assert market.tasks["task-1"].status is TaskStatus.SETTLED
    assert market.snapshot()["gross_verified_units"] == 1_200
    assert market.verify()


def test_provider_must_advertise_requested_capability() -> None:
    market = OpenMarketEngine(clock_ns=lambda: 1)
    market.register_provider(provider("render-node", "rendering"))
    market.create_task(
        MarketTask(
            task_id="task-2",
            buyer_id="factory-1",
            capability="optimization",
            max_price_units=500,
        )
    )

    with pytest.raises(MarketStateError, match="does not advertise capability"):
        market.submit_bid(
            MarketBid(
                bid_id="wrong-capability",
                task_id="task-2",
                provider_id="render-node",
                program=assemble("HALT"),
                price_units=10,
                quality=1.0,
                latency=1.0,
                risk=0.0,
            )
        )


def test_bid_over_buyer_price_ceiling_is_rejected() -> None:
    market = OpenMarketEngine(clock_ns=lambda: 1)
    market.register_provider(provider("optimizer", "optimization"))
    market.create_task(
        MarketTask(
            task_id="task-3",
            buyer_id="factory-1",
            capability="optimization",
            max_price_units=100,
        )
    )

    with pytest.raises(MarketStateError, match="exceeds task maximum"):
        market.submit_bid(
            MarketBid(
                bid_id="expensive",
                task_id="task-3",
                provider_id="optimizer",
                program=assemble("HALT"),
                price_units=101,
                quality=1.0,
                latency=1.0,
                risk=0.0,
            )
        )


def test_failed_execution_settles_zero_and_cannot_pay_provider() -> None:
    market = OpenMarketEngine(fee_bps=500, clock_ns=lambda: 1)
    market.register_provider(provider("optimizer", "optimization"))
    market.create_task(
        MarketTask(
            task_id="task-fail",
            buyer_id="factory-1",
            capability="optimization",
            max_price_units=1_000,
            budget=ResourceBudget(max_cycles=1, max_program_words=32, max_candidates=4),
        )
    )
    market.submit_bid(
        MarketBid(
            bid_id="will-fail",
            task_id="task-fail",
            provider_id="optimizer",
            program=assemble("SET A 1\nSET A 2\nHALT"),
            price_units=500,
            quality=1.0,
            latency=1.0,
            risk=0.0,
        )
    )

    settlement = market.execute_task("task-fail")

    assert settlement.status is SettlementStatus.EXECUTION_FAILED
    assert settlement.gross_units == 0
    assert settlement.platform_fee_units == 0
    assert settlement.provider_units == 0
    assert settlement.execution_state == ()
    assert market.tasks["task-fail"].status is TaskStatus.FAILED
    assert market.verify()


def test_execute_is_idempotent_after_settlement() -> None:
    market = OpenMarketEngine(clock_ns=lambda: 1)
    market.register_provider(provider("optimizer", "optimization"))
    market.create_task(
        MarketTask(
            task_id="task-idempotent",
            buyer_id="factory-1",
            capability="optimization",
            max_price_units=100,
        )
    )
    market.submit_bid(
        MarketBid(
            bid_id="bid-idempotent",
            task_id="task-idempotent",
            provider_id="optimizer",
            program=assemble("SET A 9\nHALT"),
            price_units=100,
            quality=1.0,
            latency=1.0,
            risk=0.0,
        )
    )

    first = market.execute_task("task-idempotent")
    ledger_length = len(market.ledger.chain)
    second = market.execute_task("task-idempotent")

    assert second == first
    assert len(market.ledger.chain) == ledger_length
    assert market.verify()


def test_market_ledger_detects_tampering() -> None:
    market = OpenMarketEngine(clock_ns=lambda: 1)
    market.register_provider(provider("optimizer", "optimization"))

    assert market.verify()
    market.ledger.chain[0]["state"]["provider_id"] = "attacker"
    assert market.verify() is False
