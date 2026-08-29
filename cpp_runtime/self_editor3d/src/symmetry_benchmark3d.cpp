#include "jarvisx/symmetry_benchmark3d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace jarvisx {
namespace symmetry3d {
namespace bench {
namespace {

using Clock = std::chrono::steady_clock;

std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

double uniform01(std::uint64_t& state) {
    const std::uint64_t bits = splitmix64(state) >> 11U;
    return static_cast<double>(bits) * (1.0 / 9007199254740992.0);
}

std::uint64_t mix_seed(std::uint64_t seed,
                       std::size_t n,
                       std::size_t noise_index,
                       std::size_t repeat,
                       std::uint64_t tag) {
    std::uint64_t state = seed ^ tag;
    state ^= static_cast<std::uint64_t>(n) * 0x9e3779b185ebca87ULL;
    state ^= static_cast<std::uint64_t>(noise_index + 1U) * 0xc2b2ae3d27d4eb4fULL;
    state ^= static_cast<std::uint64_t>(repeat + 1U) * 0x165667b19e3779f9ULL;
    return splitmix64(state);
}

double elapsed_us(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double throughput_mpix(std::size_t pixels, double latency_us) {
    if (!(latency_us > 0.0)) {
        return 0.0;
    }
    return static_cast<double>(pixels) / latency_us;
}

double objective_reduction(double initial, double final_value) {
    if (!(initial > 0.0)) {
        return 0.0;
    }
    return (initial - final_value) / initial;
}

BenchmarkRow base_row(const std::string& model,
                      NoiseDomain domain,
                      std::size_t n,
                      double noise_probability,
                      std::size_t repeat,
                      double theoretical_error) {
    BenchmarkRow row;
    row.model = model;
    row.noise_domain = noise_domain_name(domain);
    row.n = n;
    row.noise_probability = noise_probability;
    row.repeat = repeat;
    row.theoretical_bit_error = theoretical_error;
    return row;
}

void fill_quality(BenchmarkRow& row,
                  const Grid& predicted,
                  const Grid& reference,
                  double latency_us) {
    row.mse = grid_mse(predicted, reference);
    row.bit_accuracy = bit_accuracy(predicted, reference);
    row.latency_us = latency_us;
    row.throughput_mpix_s = throughput_mpix(reference.size(), latency_us);
}

}  // namespace

Grid make_reference_pattern(std::size_t n, std::uint64_t seed) {
    if (n == 0) {
        throw std::invalid_argument("reference pattern side must be positive");
    }
    Grid grid(n * n, 0.0);
    const std::size_t phase = static_cast<std::size_t>(seed % 11ULL);
    const double center = (static_cast<double>(n) - 1.0) * 0.5;
    const double radius = std::max(1.0, static_cast<double>(n) * 0.28);
    const double radius2 = radius * radius;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const bool checker = (((i / 2U) + (j / 2U) + phase) & 1U) != 0U;
            const bool diagonal = ((i + 2U * j + phase) % 7U) < 3U;
            const double di = static_cast<double>(i) - center;
            const double dj = static_cast<double>(j) - center;
            const double d2 = di * di + dj * dj;
            const bool disk = d2 <= radius2;
            const bool value = checker ^ diagonal ^ disk;
            grid[i * n + j] = value ? 1.0 : 0.0;
        }
    }
    return grid;
}

Grid corrupt_grid(const Grid& source, double probability, std::uint64_t seed) {
    if (source.empty()) {
        throw std::invalid_argument("cannot corrupt an empty grid");
    }
    if (probability < 0.0 || probability > 1.0 || !std::isfinite(probability)) {
        throw std::invalid_argument("noise probability must be in [0,1]");
    }
    Grid out = source;
    std::uint64_t state = seed;
    for (double& value : out) {
        if (uniform01(state) < probability) {
            value = value >= 0.5 ? 0.0 : 1.0;
        }
    }
    return out;
}

Tensor3 corrupt_latent_independently(const Tensor3& latent,
                                     double probability,
                                     std::uint64_t seed) {
    if (latent.n == 0) {
        throw std::invalid_argument("cannot corrupt an empty latent tensor");
    }
    Tensor3 out = latent;
    for (std::size_t k = 0; k < 3; ++k) {
        out.layer[k] = corrupt_grid(
            latent.layer[k],
            probability,
            seed ^ (0xd6e8feb86659fd93ULL * static_cast<std::uint64_t>(k + 1U)));
    }
    return out;
}

double grid_mse(const Grid& a, const Grid& b) {
    if (a.empty() || a.size() != b.size()) {
        throw std::invalid_argument("grid_mse requires equal non-empty grids");
    }
    double total = 0.0;
    for (std::size_t p = 0; p < a.size(); ++p) {
        const double d = a[p] - b[p];
        total += d * d;
    }
    return total / static_cast<double>(a.size());
}

double bit_accuracy(const Grid& predicted, const Grid& reference, double threshold) {
    if (predicted.empty() || predicted.size() != reference.size()) {
        throw std::invalid_argument("bit_accuracy requires equal non-empty grids");
    }
    std::size_t correct = 0;
    for (std::size_t p = 0; p < predicted.size(); ++p) {
        const bool a = predicted[p] >= threshold;
        const bool b = reference[p] >= threshold;
        if (a == b) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(predicted.size());
}

double majority_vote_bit_error(double independent_flip_probability) {
    const double p = independent_flip_probability;
    if (p < 0.0 || p > 1.0 || !std::isfinite(p)) {
        throw std::invalid_argument("flip probability must be in [0,1]");
    }
    return 3.0 * p * p - 2.0 * p * p * p;
}

std::string noise_domain_name(NoiseDomain domain) {
    switch (domain) {
        case NoiseDomain::InputCommon:
            return "input-common";
        case NoiseDomain::LatentIndependent:
            return "latent-independent";
    }
    throw std::invalid_argument("unknown noise domain");
}

BenchmarkRunner::BenchmarkRunner(BenchmarkConfig config) : config_(std::move(config)) {
    if (config_.sizes.empty() || config_.noise_levels.empty() || config_.repeats == 0) {
        throw std::invalid_argument("benchmark requires sizes, noise levels and repeats");
    }
    for (std::size_t n : config_.sizes) {
        if (n == 0) {
            throw std::invalid_argument("benchmark grid sides must be positive");
        }
    }
    for (double p : config_.noise_levels) {
        if (p < 0.0 || p > 1.0 || !std::isfinite(p)) {
            throw std::invalid_argument("benchmark noise levels must be in [0,1]");
        }
    }
}

std::vector<BenchmarkRow> BenchmarkRunner::run_size(std::size_t n) const {
    const std::size_t pixels = n * n;
    const std::size_t scalar_bytes = pixels * sizeof(double);
    const Grid reference = make_reference_pattern(n, config_.seed ^ static_cast<std::uint64_t>(n));
    SymmetryCodec codec(n);
    ClosedLoopOptimizer optimizer(n, config_.loop);

    const auto optimize_begin = Clock::now();
    const OptimizationReport optimized = optimizer.optimize(reference, Transport3::cyclic_inward());
    const auto optimize_end = Clock::now();
    const double optimization_latency = elapsed_us(optimize_begin, optimize_end);
    const double reduction = objective_reduction(
        optimized.initial_loss.objective,
        optimized.final_loss.objective);
    const Tensor3 clean_latent = codec.encode(reference);

    std::vector<BenchmarkRow> rows;
    rows.reserve(config_.noise_levels.size() * config_.repeats * 6U);

    for (std::size_t noise_index = 0; noise_index < config_.noise_levels.size(); ++noise_index) {
        const double p = config_.noise_levels[noise_index];
        for (std::size_t repeat = 0; repeat < config_.repeats; ++repeat) {
            const std::uint64_t common_seed = mix_seed(
                config_.seed, n, noise_index, repeat, 0x434f4d4d4f4eULL);
            const Grid common_noisy = corrupt_grid(reference, p, common_seed);

            {
                BenchmarkRow row = base_row("identity", NoiseDomain::InputCommon, n, p, repeat, p);
                const auto begin = Clock::now();
                const Grid predicted = common_noisy;
                const auto end = Clock::now();
                fill_quality(row, predicted, reference, elapsed_us(begin, end));
                row.estimated_working_set_bytes = 2U * scalar_bytes;
                row.latent_expansion_ratio = 1.0;
                rows.push_back(row);
            }

            {
                BenchmarkRow row = base_row("symmetry-exact", NoiseDomain::InputCommon, n, p, repeat, p);
                const auto begin = Clock::now();
                const Grid predicted = codec.decode_majority(codec.encode(common_noisy));
                const auto end = Clock::now();
                fill_quality(row, predicted, reference, elapsed_us(begin, end));
                row.estimated_working_set_bytes = 4U * scalar_bytes;
                row.latent_expansion_ratio = 3.0;
                rows.push_back(row);
            }

            {
                BenchmarkRow row = base_row(
                    "symmetry-closed-loop", NoiseDomain::InputCommon, n, p, repeat, -1.0);
                const auto begin = Clock::now();
                const FeedbackReport feedback = optimizer.close_feedback(
                    common_noisy, reference, optimized.transport);
                const auto end = Clock::now();
                fill_quality(row, feedback.final_binary_state, reference, elapsed_us(begin, end));
                row.feedback_steps = feedback.steps;
                row.fixed_point_mse = feedback.fixed_point_mse;
                row.initial_objective = optimized.initial_loss.objective;
                row.final_objective = optimized.final_loss.objective;
                row.objective_reduction_fraction = reduction;
                row.optimization_sweeps = optimized.sweeps;
                row.accepted_moves = optimized.accepted_moves;
                row.optimization_latency_us = optimization_latency;
                row.estimated_working_set_bytes = 8U * scalar_bytes + 9U * sizeof(double);
                row.latent_expansion_ratio = 3.0;
                rows.push_back(row);
            }

            const std::uint64_t latent_seed = mix_seed(
                config_.seed, n, noise_index, repeat, 0x4c4154454e54ULL);
            const Tensor3 latent_noisy = corrupt_latent_independently(clean_latent, p, latent_seed);

            {
                BenchmarkRow row = base_row(
                    "single-copy", NoiseDomain::LatentIndependent, n, p, repeat, p);
                const auto begin = Clock::now();
                const Grid predicted = latent_noisy.layer[0];
                const auto end = Clock::now();
                fill_quality(row, predicted, reference, elapsed_us(begin, end));
                row.estimated_working_set_bytes = 2U * scalar_bytes;
                row.latent_expansion_ratio = 1.0;
                rows.push_back(row);
            }

            {
                BenchmarkRow row = base_row(
                    "symmetry-majority", NoiseDomain::LatentIndependent, n, p, repeat,
                    majority_vote_bit_error(p));
                const auto begin = Clock::now();
                const Grid predicted = codec.decode_majority(latent_noisy);
                const auto end = Clock::now();
                fill_quality(row, predicted, reference, elapsed_us(begin, end));
                row.estimated_working_set_bytes = 4U * scalar_bytes;
                row.latent_expansion_ratio = 3.0;
                rows.push_back(row);
            }

            {
                BenchmarkRow row = base_row(
                    "symmetry-learned-transport", NoiseDomain::LatentIndependent, n, p, repeat, -1.0);
                const auto begin = Clock::now();
                const Tensor3 transported = codec.apply_transport(latent_noisy, optimized.transport);
                const Grid soft = codec.decode_soft(transported);
                const Grid predicted = codec.hard_threshold(soft, config_.loop.hard_threshold);
                const auto end = Clock::now();
                fill_quality(row, predicted, reference, elapsed_us(begin, end));
                row.initial_objective = optimized.initial_loss.objective;
                row.final_objective = optimized.final_loss.objective;
                row.objective_reduction_fraction = reduction;
                row.optimization_sweeps = optimized.sweeps;
                row.accepted_moves = optimized.accepted_moves;
                row.optimization_latency_us = optimization_latency;
                row.estimated_working_set_bytes = 7U * scalar_bytes + 9U * sizeof(double);
                row.latent_expansion_ratio = 3.0;
                rows.push_back(row);
            }
        }
    }
    return rows;
}

std::vector<BenchmarkRow> BenchmarkRunner::run() const {
    std::vector<BenchmarkRow> rows;
    for (std::size_t n : config_.sizes) {
        std::vector<BenchmarkRow> size_rows = run_size(n);
        rows.insert(rows.end(), size_rows.begin(), size_rows.end());
    }
    return rows;
}

BenchmarkSummary BenchmarkRunner::summarize(const std::vector<BenchmarkRow>& rows) const {
    BenchmarkSummary summary;
    summary.row_count = rows.size();
    if (rows.empty()) {
        return summary;
    }
    for (const BenchmarkRow& row : rows) {
        summary.mean_mse += row.mse;
        summary.mean_bit_accuracy += row.bit_accuracy;
        summary.mean_latency_us += row.latency_us;
        summary.mean_throughput_mpix_s += row.throughput_mpix_s;
    }
    const double count = static_cast<double>(rows.size());
    summary.mean_mse /= count;
    summary.mean_bit_accuracy /= count;
    summary.mean_latency_us /= count;
    summary.mean_throughput_mpix_s /= count;
    return summary;
}

void BenchmarkRunner::write_csv(const std::filesystem::path& path,
                                const std::vector<BenchmarkRow>& rows) const {
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open benchmark CSV for writing");
    }
    out << "model,noise_domain,n,noise_probability,repeat,mse,bit_accuracy,"
           "theoretical_bit_error,feedback_steps,fixed_point_mse,initial_objective,"
           "final_objective,objective_reduction_fraction,optimization_sweeps,accepted_moves,"
           "optimization_latency_us,latency_us,throughput_mpix_s,estimated_working_set_bytes,"
           "latent_expansion_ratio\n";
    out << std::setprecision(17);
    for (const BenchmarkRow& row : rows) {
        out << row.model << ','
            << row.noise_domain << ','
            << row.n << ','
            << row.noise_probability << ','
            << row.repeat << ','
            << row.mse << ','
            << row.bit_accuracy << ','
            << row.theoretical_bit_error << ','
            << row.feedback_steps << ','
            << row.fixed_point_mse << ','
            << row.initial_objective << ','
            << row.final_objective << ','
            << row.objective_reduction_fraction << ','
            << row.optimization_sweeps << ','
            << row.accepted_moves << ','
            << row.optimization_latency_us << ','
            << row.latency_us << ','
            << row.throughput_mpix_s << ','
            << row.estimated_working_set_bytes << ','
            << row.latent_expansion_ratio << '\n';
    }
}

}  // namespace bench
}  // namespace symmetry3d
}  // namespace jarvisx
