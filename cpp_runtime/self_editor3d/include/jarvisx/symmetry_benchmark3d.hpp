#pragma once

#include "jarvisx/symmetry_loop3d.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace jarvisx {
namespace symmetry3d {
namespace bench {

enum class NoiseDomain {
    InputCommon,
    LatentIndependent,
};

struct BenchmarkConfig {
    std::vector<std::size_t> sizes{8, 16, 32, 64, 128, 256};
    std::vector<double> noise_levels{0.0, 0.05, 0.10, 0.20, 0.30, 0.40};
    std::size_t repeats{5};
    std::uint64_t seed{0x4d4f414749334455ULL};
    LoopConfig loop{};
};

struct BenchmarkRow {
    std::string model;
    std::string noise_domain;
    std::size_t n{};
    double noise_probability{};
    std::size_t repeat{};
    double mse{};
    double bit_accuracy{};
    double theoretical_bit_error{};
    std::size_t feedback_steps{};
    double fixed_point_mse{};
    double initial_objective{};
    double final_objective{};
    double objective_reduction_fraction{};
    std::size_t optimization_sweeps{};
    std::size_t accepted_moves{};
    double optimization_latency_us{};
    double latency_us{};
    double throughput_mpix_s{};
    std::size_t estimated_working_set_bytes{};
    double latent_expansion_ratio{};
};

struct BenchmarkSummary {
    std::size_t row_count{};
    double mean_mse{};
    double mean_bit_accuracy{};
    double mean_latency_us{};
    double mean_throughput_mpix_s{};
};

Grid make_reference_pattern(std::size_t n, std::uint64_t seed);
Grid corrupt_grid(const Grid& source, double probability, std::uint64_t seed);
Tensor3 corrupt_latent_independently(const Tensor3& latent,
                                     double probability,
                                     std::uint64_t seed);

double grid_mse(const Grid& a, const Grid& b);
double bit_accuracy(const Grid& predicted, const Grid& reference, double threshold = 0.5);
double majority_vote_bit_error(double independent_flip_probability);

class BenchmarkRunner {
public:
    explicit BenchmarkRunner(BenchmarkConfig config = {});

    std::vector<BenchmarkRow> run() const;
    BenchmarkSummary summarize(const std::vector<BenchmarkRow>& rows) const;
    void write_csv(const std::filesystem::path& path,
                   const std::vector<BenchmarkRow>& rows) const;

private:
    BenchmarkConfig config_;

    std::vector<BenchmarkRow> run_size(std::size_t n) const;
};

std::string noise_domain_name(NoiseDomain domain);

}  // namespace bench
}  // namespace symmetry3d
}  // namespace jarvisx
