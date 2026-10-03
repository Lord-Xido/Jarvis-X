#include "jarvisx/volumetric_80k_fastpath.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " [--ticks N] [--candidates N] [--select-ratio F]"
           " [--active-bricks N]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::uint32_t ticks = 4u;
        jarvisx::volumetric_80k_fast::FastPolicy fast{};
        jarvisx::volumetric_80k::RuntimePolicy runtime{};

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(arg + " requires a value");
                }
                return argv[++i];
            };

            if (arg == "--ticks") {
                const auto value = std::stoul(next());
                if (value == 0ul || value > 1000ul) {
                    throw std::invalid_argument("--ticks must be in [1,1000]");
                }
                ticks = static_cast<std::uint32_t>(value);
            } else if (arg == "--candidates") {
                const auto value = std::stoul(next());
                if (value == 0ul || value > 4096ul) {
                    throw std::invalid_argument(
                        "--candidates must be in [1,4096]");
                }
                fast.candidates_per_tick = static_cast<std::uint32_t>(value);
            } else if (arg == "--select-ratio") {
                fast.selection_ratio = std::stof(next());
            } else if (arg == "--active-bricks") {
                const auto value = std::stoul(next());
                if (value == 0ul || value > 4096ul) {
                    throw std::invalid_argument(
                        "--active-bricks must be in [1,4096]");
                }
                runtime.max_active_bricks = static_cast<std::uint32_t>(value);
            } else if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        jarvisx::volumetric_80k_fast::Scheduler workload(fast, runtime);
        jarvisx::volumetric_80k::Engine baseline(runtime);

        const auto baseline_start = std::chrono::steady_clock::now();
        for (std::uint32_t tick = 0; tick < ticks; ++tick) {
            const auto candidates = workload.candidate_voxels(tick);
            for (const auto& voxel : candidates) {
                (void)baseline.step_at(tick, voxel);
            }
        }
        const auto baseline_stop = std::chrono::steady_clock::now();

        jarvisx::volumetric_80k_fast::Scheduler optimized(fast, runtime);
        const auto fast_start = std::chrono::steady_clock::now();
        for (std::uint32_t tick = 0; tick < ticks; ++tick) {
            (void)optimized.tick(tick);
        }
        const auto fast_stop = std::chrono::steady_clock::now();

        const double baseline_ms =
            std::chrono::duration<double, std::milli>(
                baseline_stop - baseline_start).count();
        const double fast_ms =
            std::chrono::duration<double, std::milli>(
                fast_stop - fast_start).count();
        const double speedup =
            fast_ms > 0.0 ? baseline_ms / fast_ms : 0.0;

        const auto& stats = optimized.stats();
        const double selection_rate =
            stats.candidates_seen == 0u
                ? 0.0
                : static_cast<double>(stats.selected) /
                      static_cast<double>(stats.candidates_seen);
        const double cache_hit_rate =
            stats.selected == 0u
                ? 0.0
                : static_cast<double>(stats.cache_hits) /
                      static_cast<double>(stats.selected);
        const double executed_rate =
            stats.candidates_seen == 0u
                ? 0.0
                : static_cast<double>(stats.processed) /
                      static_cast<double>(stats.candidates_seen);

        std::cout << std::fixed << std::setprecision(3)
                  << "DM80K fast-path benchmark\n"
                  << "ticks              : " << ticks << '\n'
                  << "candidates/tick    : " << fast.candidates_per_tick << '\n'
                  << "selection ratio    : " << selection_rate << '\n'
                  << "cache hit rate     : " << cache_hit_rate << '\n'
                  << "executed work rate : " << executed_rate << '\n'
                  << "baseline ms        : " << baseline_ms << '\n'
                  << "fast-path ms       : " << fast_ms << '\n'
                  << "measured speedup   : " << speedup << "x\n"
                  << "\n"
                  << "The measured ratio covers portable CPU selection/cache effects only.\n"
                  << "It does not include CUDA kernel fusion, FP8 tensor cores, TMA,\n"
                  << "warp specialization, or hardware video encoding.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
