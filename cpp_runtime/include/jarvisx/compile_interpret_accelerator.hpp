#pragma once

#include "jarvisx/recursive_cube_interpreter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jarvisx::cube::acceleration {

// A septillion is 10^24. Therefore septillion^septillion is
// (10^24)^(10^24) = 10^(24 * 10^24).  The target is kept in log10 space;
// the runtime never attempts to materialize this astronomically large scalar.
constexpr long double kSeptillion = 1.0e24L;
constexpr long double kSeptillionSquaredExponentLog10 = 24.0L * kSeptillion;

struct AccelerationTarget {
    long double log10_speedup{kSeptillionSquaredExponentLog10};

    void validate() const {
        if (!std::isfinite(log10_speedup) || log10_speedup < 0.0L) {
            throw std::invalid_argument("acceleration target must be finite and non-negative in log10 space");
        }
    }
};

struct AccelerationFactors {
    long double parallel{1.0L};
    long double sparse{1.0L};
    long double memoization{1.0L};
    long double fusion{1.0L};
    long double vectorization{1.0L};
    long double pipeline{1.0L};
    long double cache{1.0L};
    long double speculative{1.0L};

    void validate() const {
        const long double factors[] = {
            parallel, sparse, memoization, fusion,
            vectorization, pipeline, cache, speculative,
        };
        for (const long double value : factors) {
            if (!std::isfinite(value) || value < 1.0L) {
                throw std::invalid_argument("measured acceleration factors must be finite and >= 1");
            }
        }
    }

    long double log10_product() const {
        validate();
        return std::log10(parallel) + std::log10(sparse) +
               std::log10(memoization) + std::log10(fusion) +
               std::log10(vectorization) + std::log10(pipeline) +
               std::log10(cache) + std::log10(speculative);
    }
};

struct RecursiveFoldProfile {
    // rho_l is the fraction of work retained after fold l; 0 < rho_l <= 1.
    std::vector<long double> retention;

    long double log10_speedup() const {
        long double result = 0.0L;
        for (const long double rho : retention) {
            if (!std::isfinite(rho) || rho <= 0.0L || rho > 1.0L) {
                throw std::invalid_argument("recursive fold retention must lie in (0,1]");
            }
            result -= std::log10(rho);
        }
        return result;
    }

    long double retained_work_fraction() const {
        long double result = 1.0L;
        for (const long double rho : retention) {
            if (!std::isfinite(rho) || rho <= 0.0L || rho > 1.0L) {
                throw std::invalid_argument("recursive fold retention must lie in (0,1]");
            }
            result *= rho;
        }
        return result;
    }
};

struct CompileInterpretTiming {
    long double compile_ns{0.0L};
    long double interpret_ns{0.0L};
    long double commit_ns{0.0L};

    void validate() const {
        if (!std::isfinite(compile_ns) || !std::isfinite(interpret_ns) ||
            !std::isfinite(commit_ns) || compile_ns < 0.0L || interpret_ns < 0.0L ||
            commit_ns < 0.0L) {
            throw std::invalid_argument("compile/interpret timing must be finite and non-negative");
        }
    }

    // Sequential baseline: compile then interpret then commit.
    long double sequential_ns() const {
        validate();
        return compile_ns + interpret_ns + commit_ns;
    }

    // Ideal fused schedule: compilation/preparation overlaps interpretation;
    // commit remains a serialized correctness boundary.
    long double fused_ns() const {
        validate();
        return std::max(compile_ns, interpret_ns) + commit_ns;
    }

    long double ideal_fusion_speedup() const {
        const long double fused = fused_ns();
        if (fused <= 0.0L) return 1.0L;
        return std::max(1.0L, sequential_ns() / fused);
    }
};

struct AccelerationContract {
    AccelerationTarget target{};
    AccelerationFactors measured_factors{};
    RecursiveFoldProfile fold{};
    CompileInterpretTiming fusion_timing{};

    void validate() const {
        target.validate();
        measured_factors.validate();
        (void)fold.log10_speedup();
        fusion_timing.validate();
    }

    long double modeled_log10_speedup() const {
        validate();
        const long double fusion_gain = std::log10(fusion_timing.ideal_fusion_speedup());
        return measured_factors.log10_product() + fold.log10_speedup() + fusion_gain;
    }

    long double target_gap_log10() const {
        return target.log10_speedup - modeled_log10_speedup();
    }

    // Map an unbounded acceleration score into a bounded kinetic controller.
    // This is deliberately not used as a physical clock multiplier.
    long double bounded_kinetic_gain() const {
        const long double score = std::max(0.0L, modeled_log10_speedup());
        return std::tanh(std::log1p(score));
    }
};

struct HarmonicCompileInterpretPhase {
    long double phase{0.0L};

    // Compile and interpret are quadrature components of one bounded phase.
    // The phase scheduler remains numerically stable regardless of target size.
    void advance(long double base_omega, long double dt, const AccelerationContract& contract) {
        if (!std::isfinite(base_omega) || !std::isfinite(dt) || base_omega < 0.0L || dt < 0.0L) {
            throw std::invalid_argument("harmonic scheduler requires finite non-negative omega and dt");
        }
        const long double gain = 1.0L + contract.bounded_kinetic_gain();
        phase = std::fmod(phase + base_omega * gain * dt, 2.0L * 3.1415926535897932384626433832795L);
    }

    long double compile_component(long double radius = 1.0L) const { return radius * std::cos(phase); }
    long double interpret_component(long double radius = 1.0L) const { return radius * std::sin(phase); }
};

struct AcceleratorReport {
    CubeRunMetrics cube{};
    std::uint64_t elapsed_ns{};
    std::uint64_t maximum_tile_pass_budget{};
    long double convergence_work_reduction{1.0L};
    long double modeled_log10_speedup{};
    long double target_log10_speedup{};
    long double target_gap_log10{};
    long double bounded_kinetic_gain{};
};

class ParallelCompileInterpretAccelerator {
public:
    explicit ParallelCompileInterpretAccelerator(RecursiveCubeInterpreter& interpreter,
                                                  CubeInterpreterConfig config = {})
        : interpreter_(interpreter), config_(config) {
        config_.validate();
    }

    AcceleratorReport run(const std::vector<std::uint8_t>& execution_buffer,
                          const AccelerationContract& contract = {}) {
        contract.validate();

        // Parse once here to calculate the constitutional work budget. The
        // interpreter re-validates the same buffer before execution, preserving
        // its existing transaction boundary.
        const auto commands = parse_execution_buffer(execution_buffer, config_);
        std::uint64_t pass_budget = 0ULL;
        for (const auto& command : commands) {
            if (command.opcode == CubeOpcode::EncodeRefine) {
                const std::uint64_t tiles = static_cast<std::uint64_t>(command.tile_count);
                const std::uint64_t passes = static_cast<std::uint64_t>(command.max_passes);
                if (passes != 0ULL && tiles > std::numeric_limits<std::uint64_t>::max() / passes) {
                    throw std::overflow_error("compile-interpret acceleration pass budget overflow");
                }
                pass_budget += tiles * passes;
            } else if (command.opcode == CubeOpcode::Decode) {
                pass_budget += static_cast<std::uint64_t>(command.tile_count);
            }
        }

        const auto start = std::chrono::steady_clock::now();
        CubeRunMetrics metrics = interpreter_.run(execution_buffer);
        const auto stop = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();

        const std::uint64_t actual_ops = std::max<std::uint64_t>(1ULL, metrics.total_passes +
            (metrics.commands_executed > 0ULL ? metrics.decoded_output_bytes / kCubeTileBytes : 0ULL));
        const long double convergence = pass_budget == 0ULL ? 1.0L :
            std::max(1.0L, static_cast<long double>(pass_budget) / static_cast<long double>(actual_ops));

        AcceleratorReport report;
        report.cube = std::move(metrics);
        report.elapsed_ns = elapsed < 0 ? 0ULL : static_cast<std::uint64_t>(elapsed);
        report.maximum_tile_pass_budget = pass_budget;
        report.convergence_work_reduction = convergence;
        report.modeled_log10_speedup = contract.modeled_log10_speedup() + std::log10(convergence);
        report.target_log10_speedup = contract.target.log10_speedup;
        report.target_gap_log10 = report.target_log10_speedup - report.modeled_log10_speedup;
        report.bounded_kinetic_gain = contract.bounded_kinetic_gain();
        return report;
    }

private:
    RecursiveCubeInterpreter& interpreter_;
    CubeInterpreterConfig config_;
};

} // namespace jarvisx::cube::acceleration
