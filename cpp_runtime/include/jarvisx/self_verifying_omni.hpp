#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace jarvisx::omni {

constexpr std::size_t channels = 8;
constexpr double pi = 3.14159265358979323846;
constexpr double norm_tolerance = 1.0e-10;
constexpr double mean_norm_tolerance = 1.0e-11;
constexpr std::size_t maximum_pathways = 10000000;

// C++17 aligned allocation works on x86, ARM and MSVC, without ISA intrinsics.
template <typename T> struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() noexcept = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    T* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{64}));
    }
    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete(pointer, std::align_val_t{64});
    }
};
template <typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept { return true; }
template <typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) noexcept { return false; }
template <typename T> using AlignedVector = std::vector<T, AlignedAllocator<T>>;

struct Config {
    std::size_t pathways = 1000000;
    std::uint32_t depth = 16;
    int requested_threads = 64;
    std::uint64_t seed = 0;
    double residual_mse_tolerance = 0.001;
    double spatial_tolerance = 0.001;
};

inline Config validated(Config config) {
    if (config.pathways == 0 || config.pathways > maximum_pathways ||
        config.depth == 0 || config.depth > 4096 ||
        config.requested_threads < 1 || config.requested_threads > 1024 ||
        !std::isfinite(config.residual_mse_tolerance) ||
        config.residual_mse_tolerance <= 0.0 || config.residual_mse_tolerance > 1.0 ||
        !std::isfinite(config.spatial_tolerance) ||
        config.spatial_tolerance <= 0.0 || config.spatial_tolerance > 1.0) {
        throw std::invalid_argument("invalid pathway, depth, thread or tolerance bound");
    }
    return config;
}

struct PathwayManifold {
    std::array<AlignedVector<float>, 3> position;
    std::array<AlignedVector<float>, channels> state; // channel-major true SoA
    std::array<AlignedVector<double>, 2> token;
    AlignedVector<float> prediction_error;
    AlignedVector<float> compute_budget;
    AlignedVector<std::uint32_t> active_mask;
    AlignedVector<double> initial_radius_sq;

    explicit PathwayManifold(std::size_t count) {
        if (count == 0 || count > maximum_pathways) {
            throw std::invalid_argument("pathway allocation outside resident bound");
        }
        for (auto& axis : position) axis.resize(count);
        for (auto& channel : state) channel.resize(count);
        for (auto& component : token) component.resize(count);
        prediction_error.resize(count);
        compute_budget.resize(count);
        active_mask.resize(count);
        initial_radius_sq.resize(count);
    }

    bool has_shape(std::size_t count) const noexcept {
        for (const auto& axis : position) if (axis.size() != count) return false;
        for (const auto& channel : state) if (channel.size() != count) return false;
        for (const auto& component : token) if (component.size() != count) return false;
        return prediction_error.size() == count && compute_budget.size() == count &&
               active_mask.size() == count && initial_radius_sq.size() == count;
    }
};

struct SpatialAttractorGrid {
    alignas(64) std::array<float, 4096> field_density{};

    SpatialAttractorGrid() {
        for (std::size_t z = 0; z < 16; ++z) {
            for (std::size_t y = 0; y < 16; ++y) {
                for (std::size_t x = 0; x < 16; ++x) {
                    const double dx = static_cast<double>(x) / 15.0 - 0.5;
                    const double dy = static_cast<double>(y) / 15.0 - 0.5;
                    const double dz = static_cast<double>(z) / 15.0 - 0.5;
                    field_density[(z << 8U) | (y << 4U) | x] =
                        static_cast<float>(std::exp(-4.0 * std::sqrt(dx*dx + dy*dy + dz*dz)));
                }
            }
        }
    }

    float sample_unchecked(float x, float y, float z) const noexcept {
        const auto index = [](float v) {
            return static_cast<std::size_t>(std::clamp(v, 0.0F, 1.0F) * 15.0F);
        };
        return field_density[(index(z) << 8U) | (index(y) << 4U) | index(x)];
    }

    float query(float x, float y, float z) const {
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
            throw std::invalid_argument("non-finite attractor coordinate");
        }
        return sample_unchecked(x, y, z);
    }
};

inline double radius_sq(const std::array<float, 3>& position) noexcept {
    double result = 0.0;
    for (const float value : position) {
        const double delta = static_cast<double>(value) - 0.5;
        result += delta * delta;
    }
    return result;
}

inline void rotate_token(double& x, double& y, double theta) noexcept {
    const double s = std::sin(theta);
    const double c = std::cos(theta);
    const double previous_x = x;
    const double previous_y = y;
    x = std::fma(previous_x, c, -previous_y * s);
    y = std::fma(previous_x, s, previous_y * c);
    // No normalization: the independent audit must observe numerical drift.
}

struct RunStats {
    std::uint32_t depth = 0;
    int worker_threads = 1;
    std::uint64_t executed_updates = 0;
    std::uint64_t rejected_updates = 0;
    double elapsed_ms = 0.0;
};

struct VerificationReport {
    bool storage_valid = true;
    bool execution_valid = true;
    std::size_t nonfinite_pathways = 0;
    std::size_t invalid_pathways = 0;
    std::size_t active = 0;
    std::size_t pruned = 0;
    std::size_t converged = 0;
    std::size_t premature_prunes = 0;
    double total_norm = 0.0;
    double norm_sum_error = 0.0;
    double max_norm_error = 0.0;
    double mean_absolute_norm_error = 0.0;
    double initial_mean_radius = 0.0;
    double final_mean_radius = 0.0;
    double max_residual_mse = 0.0;
    double state_checksum = 0.0;
    double total_budget = 0.0;

    bool passed() const noexcept {
        return storage_valid && execution_valid && nonfinite_pathways == 0 &&
               invalid_pathways == 0 && premature_prunes == 0 &&
               max_norm_error <= norm_tolerance &&
               mean_absolute_norm_error <= mean_norm_tolerance;
    }
};

// Audit authoritative arrays independently, recomputing norms and residuals.
// Counts validate mask values; they are not a proof of race freedom or leak freedom.
inline VerificationReport verify_manifold(const PathwayManifold& manifold,
                                           const Config& config,
                                           const RunStats& stats) {
    validated(config);
    VerificationReport report;
    report.storage_valid = manifold.has_shape(config.pathways);
    report.execution_valid = stats.rejected_updates == 0 && stats.worker_threads >= 1 &&
        stats.worker_threads <= config.requested_threads &&
        stats.depth <= config.depth && std::isfinite(stats.elapsed_ms) && stats.elapsed_ms >= 0.0 &&
        stats.executed_updates <= static_cast<std::uint64_t>(config.pathways) * stats.depth;
    if (!report.storage_valid) return report;
    const SpatialAttractorGrid grid;
    for (std::size_t i = 0; i < config.pathways; ++i) {
        const std::array<float, 3> position{
            manifold.position[0][i], manifold.position[1][i], manifold.position[2][i]};
        bool finite = std::isfinite(manifold.token[0][i]) && std::isfinite(manifold.token[1][i]) &&
            std::isfinite(manifold.prediction_error[i]) && std::isfinite(manifold.compute_budget[i]) &&
            std::isfinite(manifold.initial_radius_sq[i]);
        for (const float value : position) finite = finite && std::isfinite(value);
        for (const auto& channel : manifold.state) finite = finite && std::isfinite(channel[i]);
        if (!finite) { ++report.nonfinite_pathways; continue; }

        const double radius = radius_sq(position);
        const double density = grid.query(position[0], position[1], position[2]);
        double residual = 0.0;
        bool bounded = true;
        for (const float value : position) bounded = bounded && value >= 0.0F && value <= 1.0F;
        for (const auto& channel : manifold.state) {
            const double delta = static_cast<double>(channel[i]) - density;
            residual += delta * delta;
            report.state_checksum += channel[i];
            bounded = bounded && channel[i] >= 0.0F && channel[i] <= 1.0F;
        }
        residual /= static_cast<double>(channels);
        const bool converged = residual <= config.residual_mse_tolerance &&
            radius <= config.spatial_tolerance * config.spatial_tolerance;
        if (converged) ++report.converged;
        const auto mask = manifold.active_mask[i];
        const double budget = manifold.compute_budget[i];
        if (mask == 0) {
            ++report.pruned;
            if (!converged) ++report.premature_prunes;
        } else if (mask == 1) {
            ++report.active;
        }
        if (!bounded || mask > 1 || budget < 0.0 || budget > 1.0 ||
            (mask == 0 && budget != 0.0) || (mask == 1 && budget <= 0.0) ||
            manifold.prediction_error[i] < 0.0F ||
            std::abs(static_cast<double>(manifold.prediction_error[i]) - residual) > 5.0e-7 ||
            manifold.initial_radius_sq[i] < 0.0 || manifold.initial_radius_sq[i] > 0.75 ||
            radius > manifold.initial_radius_sq[i] + 1.0e-12) ++report.invalid_pathways;

        const double x = manifold.token[0][i];
        const double y = manifold.token[1][i];
        const double norm = std::fma(x, x, y * y);
        if (!std::isfinite(norm)) { ++report.nonfinite_pathways; continue; }
        const double error = std::abs(norm - 1.0);
        report.total_norm += norm;
        report.max_norm_error = std::max(report.max_norm_error, error);
        report.mean_absolute_norm_error += error;
        report.initial_mean_radius += std::sqrt(manifold.initial_radius_sq[i]);
        report.final_mean_radius += std::sqrt(radius);
        report.max_residual_mse = std::max(report.max_residual_mse, residual);
        report.total_budget += budget;
    }
    const double count = static_cast<double>(config.pathways);
    report.norm_sum_error = std::abs(report.total_norm - count);
    report.mean_absolute_norm_error /= count;
    report.initial_mean_radius /= count;
    report.final_mean_radius /= count;
    report.execution_valid = report.execution_valid && report.norm_sum_error <= count * mean_norm_tolerance;
    return report;
}

class SelfVerifyingOmniEngine {
    Config config_;
    PathwayManifold pathways_;
    SpatialAttractorGrid grid_;
    RunStats stats_;
    std::uint64_t rejected_total_ = 0;

    static std::uint64_t mix(std::uint64_t value) noexcept {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    bool update(std::size_t i) noexcept {
        const std::array<float, 3> previous{
            pathways_.position[0][i], pathways_.position[1][i], pathways_.position[2][i]};
        const float budget = pathways_.compute_budget[i];
        const float density = grid_.sample_unchecked(previous[0], previous[1], previous[2]);
        const float step = 0.05F * density * budget;
        std::array<float, 3> candidate{};
        for (std::size_t axis = 0; axis < 3; ++axis) {
            candidate[axis] = std::fma(0.5F - previous[axis], step, previous[axis]);
        }
        std::array<float, channels> state{};
        const double new_density = grid_.sample_unchecked(candidate[0], candidate[1], candidate[2]);
        double residual = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel) {
            state[channel] = std::fma(density, 0.1F, pathways_.state[channel][i] * 0.9F);
            const double delta = static_cast<double>(state[channel]) - new_density;
            residual += delta * delta;
        }
        residual /= static_cast<double>(channels);
        double token_x = pathways_.token[0][i];
        double token_y = pathways_.token[1][i];
        rotate_token(token_x, token_y, static_cast<double>(density) * pi * budget);
        const double norm = std::fma(token_x, token_x, token_y * token_y);
        const double candidate_radius = radius_sq(candidate);
        bool admissible = std::isfinite(norm) && std::abs(norm - 1.0) <= norm_tolerance &&
            std::isfinite(residual) && candidate_radius <= radius_sq(previous) + 1.0e-12;
        for (const float value : candidate) admissible = admissible && std::isfinite(value) && value >= 0.0F && value <= 1.0F;
        for (const float value : state) admissible = admissible && std::isfinite(value) && value >= 0.0F && value <= 1.0F;
        if (!admissible) return false; // leave the complete pathway untouched

        const bool converged = residual <= config_.residual_mse_tolerance &&
            candidate_radius <= config_.spatial_tolerance * config_.spatial_tolerance;
        const float next_budget = std::fma(0.8F, budget, converged ? 0.0F : 0.2F);
        const bool prune = converged && next_budget < 0.01F;
        for (std::size_t axis = 0; axis < 3; ++axis) pathways_.position[axis][i] = candidate[axis];
        for (std::size_t channel = 0; channel < channels; ++channel) pathways_.state[channel][i] = state[channel];
        pathways_.token[0][i] = token_x;
        pathways_.token[1][i] = token_y;
        pathways_.prediction_error[i] = static_cast<float>(residual);
        pathways_.compute_budget[i] = prune ? 0.0F : next_budget;
        pathways_.active_mask[i] = prune ? 0U : 1U;
        return true;
    }

public:
    explicit SelfVerifyingOmniEngine(Config config = {})
        : config_(validated(config)), pathways_(config_.pathways) {
        for (std::size_t i = 0; i < config_.pathways; ++i) {
            std::array<float, 3> position{};
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto random = mix(config_.seed + static_cast<std::uint64_t>(i) * 3U + axis);
                position[axis] = static_cast<float>(random >> 40U) / 16777216.0F;
                pathways_.position[axis][i] = position[axis];
            }
            pathways_.token[0][i] = 1.0;
            pathways_.token[1][i] = 0.0;
            pathways_.compute_budget[i] = 1.0F;
            pathways_.active_mask[i] = 1;
            pathways_.initial_radius_sq[i] = radius_sq(position);
            const double density = grid_.query(position[0], position[1], position[2]);
            pathways_.prediction_error[i] = static_cast<float>(density * density);
        }
    }

    const PathwayManifold& manifold() const noexcept { return pathways_; }
    const Config& config() const noexcept { return config_; }

    std::size_t resident_payload_bytes() const noexcept {
        return config_.pathways * (13U * sizeof(float) + 3U * sizeof(double) + sizeof(std::uint32_t)) +
            grid_.field_density.size() * sizeof(float);
    }

    RunStats execute_verified_inward_loop() {
        const auto start = std::chrono::steady_clock::now();
        std::uint64_t executed = 0;
        std::uint64_t rejected = 0;
        const auto count = static_cast<std::int64_t>(config_.pathways);
#ifdef _OPENMP
        const int workers = std::min({config_.requested_threads, omp_get_max_threads(),
            omp_get_thread_limit(), omp_get_num_procs(), static_cast<int>(config_.pathways)});
#pragma omp parallel num_threads(workers) reduction(+:executed,rejected)
#endif
        {
#ifdef _OPENMP
#pragma omp single
            { stats_.worker_threads = omp_get_num_threads(); }
#else
            stats_.worker_threads = 1;
#endif
            for (std::uint32_t depth = 0; depth < config_.depth; ++depth) {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                for (std::int64_t index = 0; index < count; ++index) {
                    const auto i = static_cast<std::size_t>(index);
                    if (pathways_.active_mask[i] == 0) continue;
                    ++executed;
                    if (!update(i)) ++rejected;
                }
            }
        }
        rejected_total_ += rejected;
        stats_.depth = config_.depth;
        stats_.executed_updates = executed;
        stats_.rejected_updates = rejected_total_;
        stats_.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return stats_;
    }

    VerificationReport verify_end_to_end_mechanics() const {
        return verify_manifold(pathways_, config_, stats_);
    }
};

} // namespace jarvisx::omni
