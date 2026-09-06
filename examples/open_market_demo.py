from jarvisx.assembler import Assembler
from jarvisx.open_market import MarketBid, MarketTask, OpenMarketEngine, Provider
from jarvisx.parser import Parser
from jarvisx.system_runtime import UtilityWeights


def assemble(source: str) -> tuple[int, ...]:
    return tuple(Assembler().assemble(Parser().parse(source)))


def main() -> None:
    market = OpenMarketEngine(fee_bps=250)

    market.register_provider(
        Provider(
            provider_id="alpha-optimizer",
            display_name="Alpha Optimizer",
            capabilities=frozenset({"optimization"}),
        )
    )
    market.register_provider(
        Provider(
            provider_id="beta-optimizer",
            display_name="Beta Optimizer",
            capabilities=frozenset({"optimization"}),
        )
    )

    market.create_task(
        MarketTask(
            task_id="factory-throughput-001",
            buyer_id="factory-demo",
            capability="optimization",
            max_price_units=2_000,
            weights=UtilityWeights(quality=100.0, cost=0.01, latency=1.0, risk=10.0),
        )
    )

    market.submit_bid(
        MarketBid(
            bid_id="alpha-plan",
            task_id="factory-throughput-001",
            provider_id="alpha-optimizer",
            program=assemble("SET A 10\nHALT"),
            price_units=1_000,
            quality=0.80,
            latency=2.0,
            risk=0.20,
        )
    )
    market.submit_bid(
        MarketBid(
            bid_id="beta-plan",
            task_id="factory-throughput-001",
            provider_id="beta-optimizer",
            program=assemble("SET A 42\nHALT"),
            price_units=1_200,
            quality=0.98,
            latency=1.0,
            risk=0.05,
        )
    )

    settlement = market.execute_task("factory-throughput-001")

    print("winner:", settlement.provider_id)
    print("bid:", settlement.bid_id)
    print("execution:", settlement.execution_status)
    print("state:", settlement.state_dict())
    print("gross units:", settlement.gross_units)
    print("platform fee:", settlement.platform_fee_units)
    print("provider units:", settlement.provider_units)
    print("market ledger head:", settlement.market_ledger_head)
    print("market valid:", market.verify())


if __name__ == "__main__":
    main()
