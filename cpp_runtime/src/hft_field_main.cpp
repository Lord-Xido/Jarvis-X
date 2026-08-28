#include "jarvisx/hft_field.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    using namespace jarvisx::hft;
    std::uint64_t events = 1'000'000ULL;
    if (argc > 1) {
        events = static_cast<std::uint64_t>(std::strtoull(argv[1], nullptr, 10));
        if (events == 0U) events = 1U;
    }

    HftFieldEngine<> engine;
    std::uint64_t intents = 0U;
    const auto start = std::chrono::steady_clock::now();
    for (std::uint64_t i = 0; i < events; ++i) {
        const bool bid = (i & 1ULL) == 0ULL;
        MarketEvent event{
            static_cast<std::int32_t>(100000 + (i & 31ULL)),
            static_cast<std::uint16_t>((i >> 5U) & 3ULL),
            bid ? Side::Bid : Side::Ask,
            Q16::from_ratio(static_cast<std::int32_t>((i % 7ULL) + 1ULL), 8),
            i + 1ULL
        };
        const auto intent = engine.process(event);
        if (intent.action != Action::None && intent.risk_accepted) {
            ++intents;
            // Functional simulator only: assume immediate fills to exercise risk state.
            engine.on_fill(intent.action, intent.quantity);
        }
    }
    const auto stop = std::chrono::steady_clock::now();
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

    const PipelineBudget budget{};
    std::cout << "Jarvis-X HFT sparse field functional simulator\n"
              << "events=" << events << "\n"
              << "order_intents=" << intents << "\n"
              << "digest=0x" << std::hex << engine.digest() << std::dec << "\n"
              << "software_ns_per_event=" << std::fixed << std::setprecision(2)
              << static_cast<double>(elapsed_ns) / static_cast<double>(events) << "\n"
              << "fpga_target_cycles=" << budget.total_cycles() << "\n"
              << "fpga_target_latency_ns_at_500MHz=" << budget.target_latency_ns(500.0) << "\n"
              << "NOTE: software timing is not an FPGA latency measurement.\n";
    return 0;
}
