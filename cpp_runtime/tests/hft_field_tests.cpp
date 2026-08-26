#include "jarvisx/hft_field.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

using namespace jarvisx::hft;

static void q16_arithmetic_is_deterministic() {
    constexpr Q16 a = Q16::from_ratio(3, 2);
    constexpr Q16 b = Q16::from_ratio(-5, 4);
    constexpr Q16 c = a * b;
    static_assert(c.raw() == -122880, "Q16 multiply regression");
    assert((a + b).raw() == 16384);
}

static void replay_is_bit_exact() {
    HftFieldEngine<> left;
    HftFieldEngine<> right;
    for (std::uint64_t i = 0; i < 4096U; ++i) {
        const MarketEvent event{
            static_cast<std::int32_t>(42000 + (i & 15U)),
            static_cast<std::uint16_t>((i >> 4U) & 3U),
            (i & 1U) == 0U ? Side::Bid : Side::Ask,
            Q16::from_ratio(static_cast<std::int32_t>((i % 5U) + 1U), 16),
            i
        };
        const auto a = left.process(event);
        const auto b = right.process(event);
        assert(a.action == b.action);
        assert(a.quantity == b.quantity);
        assert(a.score == b.score);
        assert(a.risk_accepted == b.risk_accepted);
        if (a.action != Action::None && a.risk_accepted) {
            left.on_fill(a.action, a.quantity);
            right.on_fill(b.action, b.quantity);
        }
    }
    assert(left.digest() == right.digest());
}

static void positive_and_negative_impulses_separate() {
    HftFieldEngine<> bid_engine;
    HftFieldEngine<> ask_engine;
    const MarketEvent bid{100, 0, Side::Bid, Q16::from_int(1), 1};
    const MarketEvent ask{100, 0, Side::Ask, Q16::from_int(1), 1};
    const auto bid_intent = bid_engine.process(bid);
    const auto ask_intent = ask_engine.process(ask);
    assert(bid_intent.score.raw() > 0);
    assert(ask_intent.score.raw() < 0);
    assert(bid_engine.digest() != ask_engine.digest());
}

static void risk_gate_fails_closed() {
    HftFieldConfig config;
    config.max_inventory = Q16::from_int(1);
    config.max_order_quantity = Q16::from_int(1);
    config.decision_threshold = Q16::from_ratio(1, 1024);
    HftFieldEngine<> engine(config);

    MarketEvent bid{100, 0, Side::Bid, Q16::from_int(1), 1};
    auto first = engine.process(bid);
    if (first.action == Action::Buy && first.risk_accepted) {
        engine.on_fill(first.action, first.quantity);
    }
    auto second = engine.process({100, 0, Side::Bid, Q16::from_int(1), 2});
    assert(second.action == Action::None || !second.risk_accepted);
}

static void pipeline_budget_is_inside_design_target() {
    constexpr PipelineBudget budget{};
    static_assert(budget.total_cycles() == 48, "pipeline budget changed");
    assert(budget.target_latency_ns(500.0) == 96.0);
}

int main() {
    q16_arithmetic_is_deterministic();
    replay_is_bit_exact();
    positive_and_negative_impulses_separate();
    risk_gate_fails_closed();
    pipeline_budget_is_inside_design_target();
    std::cout << "hft field regressions passed\n";
    return 0;
}
