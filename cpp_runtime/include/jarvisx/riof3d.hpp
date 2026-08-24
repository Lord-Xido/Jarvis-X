#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace jarvisx {

struct Riof3DConfig {
    std::size_t edge{16U};
    double smoothness_weight{0.18};
    double bending_weight{0.015};
    double prediction_weight{0.30};
    double refinement_gain{0.08};
    double enhancement_gain{0.015};
    double memory_gain{0.025};
    double base_damping{0.20};
    double cfl{0.12};
    double min_timestep{0.001};
    double max_timestep{0.035};
    double projection_limit{1.5};
    std::size_t adjoint_iterations{8U};
    std::uint64_t seed{0x52494F463344ULL};

    void validate() const {
        if (edge < 4U || edge > 128U) {
            throw std::invalid_argument("RIOF edge must be in [4, 128]");
        }
        const double weights[] = {smoothness_weight, bending_weight, prediction_weight,
                                  refinement_gain, enhancement_gain, memory_gain,
                                  base_damping, cfl, min_timestep, max_timestep,
                                  projection_limit};
        for (double value : weights) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("RIOF configuration values must be finite and non-negative");
            }
        }
        if (smoothness_weight == 0.0 && bending_weight == 0.0 &&
            prediction_weight == 0.0) {
            throw std::invalid_argument("RIOF requires at least one objective term");
        }
        if (cfl <= 0.0 || min_timestep <= 0.0 || max_timestep <= 0.0 ||
            min_timestep > max_timestep) {
            throw std::invalid_argument("RIOF timestep bounds are invalid");
        }
        if (projection_limit <= 0.0 || projection_limit > 100.0) {
            throw std::invalid_argument("RIOF projection limit must be in (0, 100]");
        }
        if (adjoint_iterations == 0U || adjoint_iterations > 256U) {
            throw std::invalid_argument("RIOF adjoint iterations must be in [1, 256]");
        }
    }
};

struct Riof3DMetrics {
    std::uint64_t step{};
    double total_energy{};
    double gradient_energy{};
    double bending_energy{};
    double prediction_energy{};
    double kinetic_energy{};
    double mean_abs_residual{};
    double max_abs_value{};
    double timestep{};
    double damping{};
    double enhancement{};
};

class Riof3D {
public:
    explicit Riof3D(Riof3DConfig config)
        : config_(config),
          field_(checked_volume(config.edge), 0.0),
          momentum_(field_.size(), 0.0),
          adjoint_(field_.size(), 0.0),
          memory_(field_.size(), 0.0),
          residual_(field_.size(), 0.0),
          laplacian_(field_.size(), 0.0),
          bilaplacian_(field_.size(), 0.0),
          force_(field_.size(), 0.0) {
        config_.validate();
    }

    const Riof3DConfig& config() const noexcept { return config_; }
    const std::vector<double>& field() const noexcept { return field_; }
    const std::vector<double>& momentum() const noexcept { return momentum_; }
    const std::vector<double>& adjoint() const noexcept { return adjoint_; }
    const std::vector<double>& memory() const noexcept { return memory_; }
    std::uint64_t steps() const noexcept { return step_; }

    void initialize(const std::string& pattern = "sphere") {
        const double n = static_cast<double>(config_.edge);
        const double center = 0.5 * (n - 1.0);
        for (std::size_t z = 0; z < config_.edge; ++z) {
            for (std::size_t y = 0; y < config_.edge; ++y) {
                for (std::size_t x = 0; x < config_.edge; ++x) {
                    const double dx = (static_cast<double>(x) - center) / n;
                    const double dy = (static_cast<double>(y) - center) / n;
                    const double dz = (static_cast<double>(z) - center) / n;
                    const double radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                    double value = 0.0;
                    if (pattern == "sphere") {
                        value = std::tanh((0.28 - radius) * 12.0);
                    } else if (pattern == "shell") {
                        value = std::exp(-180.0 * (radius - 0.27) * (radius - 0.27)) * 2.0 - 1.0;
                    } else if (pattern == "wave") {
                        value = 0.55 * std::sin(12.0 * dx) * std::cos(10.0 * dy) * std::sin(8.0 * dz);
                    } else if (pattern == "noise") {
                        value = signed_noise(index(x, y, z), 0U);
                    } else {
                        throw std::invalid_argument("unknown RIOF pattern: " + pattern);
                    }
                    field_[index(x, y, z)] = value;
                }
            }
        }
        std::fill(momentum_.begin(), momentum_.end(), 0.0);
        std::fill(adjoint_.begin(), adjoint_.end(), 0.0);
        std::fill(memory_.begin(), memory_.end(), 0.0);
        std::fill(residual_.begin(), residual_.end(), 0.0);
        step_ = 0U;
        last_timestep_ = config_.max_timestep;
        update_geometry();
        solve_adjoint();
    }

    Riof3DMetrics step() {
        update_geometry();
        solve_adjoint();
        const Dynamics first = compute_force();
        const double dt = adaptive_timestep(first.max_force);

        for (std::size_t i = 0; i < field_.size(); ++i) {
            momentum_[i] += 0.5 * dt * force_[i];
            field_[i] += dt * momentum_[i];
            project(field_[i], momentum_[i]);
        }

        update_geometry();
        solve_adjoint();
        const Dynamics second = compute_force();
        for (std::size_t i = 0; i < field_.size(); ++i) {
            momentum_[i] += 0.5 * dt * force_[i];
            project(field_[i], momentum_[i]);
            memory_[i] = 0.97 * memory_[i] + 0.03 * residual_[i];
        }

        ++step_;
        last_timestep_ = dt;
        last_damping_ = second.damping;
        last_enhancement_ = second.enhancement;
        return metrics();
    }

    Riof3DMetrics metrics() {
        update_geometry();
        double gradient_sum = 0.0;
        double bending_sum = 0.0;
        double prediction_sum = 0.0;
        double kinetic_sum = 0.0;
        double residual_sum = 0.0;
        double max_abs = 0.0;

        for (std::size_t z = 0; z < config_.edge; ++z) {
            for (std::size_t y = 0; y < config_.edge; ++y) {
                for (std::size_t x = 0; x < config_.edge; ++x) {
                    const std::size_t i = index(x, y, z);
                    const double dx = field_[index(wrap_plus(x), y, z)] - field_[i];
                    const double dy = field_[index(x, wrap_plus(y), z)] - field_[i];
                    const double dz = field_[index(x, y, wrap_plus(z))] - field_[i];
                    gradient_sum += dx * dx + dy * dy + dz * dz;
                    bending_sum += laplacian_[i] * laplacian_[i];
                    prediction_sum += residual_[i] * residual_[i];
                    kinetic_sum += momentum_[i] * momentum_[i];
                    residual_sum += std::abs(residual_[i]);
                    max_abs = std::max(max_abs, std::abs(field_[i]));
                }
            }
        }

        const double inv = 1.0 / static_cast<double>(field_.size());
        Riof3DMetrics result;
        result.step = step_;
        result.gradient_energy = 0.5 * config_.smoothness_weight * gradient_sum * inv;
        result.bending_energy = 0.5 * config_.bending_weight * bending_sum * inv;
        result.prediction_energy = 0.5 * config_.prediction_weight * prediction_sum * inv;
        result.kinetic_energy = 0.5 * kinetic_sum * inv;
        result.total_energy = result.gradient_energy + result.bending_energy +
                              result.prediction_energy + result.kinetic_energy;
        result.mean_abs_residual = residual_sum * inv;
        result.max_abs_value = max_abs;
        result.timestep = last_timestep_;
        result.damping = last_damping_;
        result.enhancement = last_enhancement_;
        return result;
    }

private:
    struct Dynamics {
        double max_force{};
        double damping{};
        double enhancement{};
    };

    Riof3DConfig config_;
    std::vector<double> field_;
    std::vector<double> momentum_;
    std::vector<double> adjoint_;
    std::vector<double> memory_;
    std::vector<double> residual_;
    std::vector<double> laplacian_;
    std::vector<double> bilaplacian_;
    std::vector<double> force_;
    std::uint64_t step_{0U};
    double last_timestep_{0.0};
    double last_damping_{0.0};
    double last_enhancement_{0.0};

    static std::size_t checked_volume(std::size_t edge) {
        if (edge == 0U) return 0U;
        const std::size_t max = std::numeric_limits<std::size_t>::max();
        if (edge > max / edge || edge * edge > max / edge) {
            throw std::overflow_error("RIOF volume size overflow");
        }
        return edge * edge * edge;
    }

    std::size_t index(std::size_t x, std::size_t y, std::size_t z) const noexcept {
        return x + config_.edge * (y + config_.edge * z);
    }

    std::size_t wrap_plus(std::size_t value) const noexcept {
        return (value + 1U == config_.edge) ? 0U : value + 1U;
    }

    std::size_t wrap_minus(std::size_t value) const noexcept {
        return (value == 0U) ? config_.edge - 1U : value - 1U;
    }

    double neighbor_mean(const std::vector<double>& values,
                         std::size_t x, std::size_t y, std::size_t z) const noexcept {
        return (values[index(wrap_plus(x), y, z)] +
                values[index(wrap_minus(x), y, z)] +
                values[index(x, wrap_plus(y), z)] +
                values[index(x, wrap_minus(y), z)] +
                values[index(x, y, wrap_plus(z))] +
                values[index(x, y, wrap_minus(z))]) / 6.0;
    }

    double laplacian_at(const std::vector<double>& values,
                        std::size_t x, std::size_t y, std::size_t z) const noexcept {
        const std::size_t i = index(x, y, z);
        return 6.0 * (neighbor_mean(values, x, y, z) - values[i]);
    }

    void update_geometry() {
        for (std::size_t z = 0; z < config_.edge; ++z) {
            for (std::size_t y = 0; y < config_.edge; ++y) {
                for (std::size_t x = 0; x < config_.edge; ++x) {
                    const std::size_t i = index(x, y, z);
                    const double prediction = neighbor_mean(field_, x, y, z);
                    residual_[i] = field_[i] - prediction;
                    laplacian_[i] = laplacian_at(field_, x, y, z);
                }
            }
        }
        for (std::size_t z = 0; z < config_.edge; ++z) {
            for (std::size_t y = 0; y < config_.edge; ++y) {
                for (std::size_t x = 0; x < config_.edge; ++x) {
                    bilaplacian_[index(x, y, z)] = laplacian_at(laplacian_, x, y, z);
                }
            }
        }
    }

    void solve_adjoint() {
        std::vector<double> next(adjoint_.size(), 0.0);
        for (std::size_t iteration = 0; iteration < config_.adjoint_iterations; ++iteration) {
            for (std::size_t z = 0; z < config_.edge; ++z) {
                for (std::size_t y = 0; y < config_.edge; ++y) {
                    for (std::size_t x = 0; x < config_.edge; ++x) {
                        const std::size_t i = index(x, y, z);
                        next[i] = neighbor_mean(adjoint_, x, y, z) - residual_[i] / 6.0;
                    }
                }
            }
            adjoint_.swap(next);
        }
        double mean = 0.0;
        for (double value : adjoint_) mean += value;
        mean /= static_cast<double>(adjoint_.size());
        for (double& value : adjoint_) value -= mean;
    }

    Dynamics compute_force() {
        double residual_mean = 0.0;
        double curvature_mean = 0.0;
        for (std::size_t i = 0; i < field_.size(); ++i) {
            residual_mean += std::abs(residual_[i]);
            curvature_mean += std::abs(laplacian_[i]);
        }
        const double inv = 1.0 / static_cast<double>(field_.size());
        residual_mean *= inv;
        curvature_mean *= inv;

        const double disagreement = residual_mean / (curvature_mean + 1.0e-9);
        const double enhancement = config_.enhancement_gain *
            std::clamp(disagreement, 0.0, 2.0);
        const double damping = config_.base_damping *
            (1.0 + std::clamp(residual_mean, 0.0, 2.0));

        double max_force = 0.0;
        for (std::size_t i = 0; i < field_.size(); ++i) {
            const double curvature_scale = std::abs(laplacian_[i]) /
                (curvature_mean + 1.0e-9);
            const double high_frequency = signed_noise(i, step_ + 1U) *
                std::clamp(curvature_scale, 0.0, 3.0);
            const double conservative =
                config_.smoothness_weight * laplacian_[i] -
                config_.bending_weight * bilaplacian_[i] -
                config_.prediction_weight * residual_[i];
            const double refinement = -config_.refinement_gain * adjoint_[i];
            const double history = -config_.memory_gain * memory_[i];
            const double spectral_surrogate = enhancement * high_frequency;
            force_[i] = conservative + refinement + history +
                        spectral_surrogate - damping * momentum_[i];
            max_force = std::max(max_force, std::abs(force_[i]));
        }
        return Dynamics{max_force, damping, enhancement};
    }

    double adaptive_timestep(double max_force) const noexcept {
        double max_momentum = 0.0;
        for (double value : momentum_) max_momentum = std::max(max_momentum, std::abs(value));
        const double scale = 1.0 + std::sqrt(std::max(0.0, max_force)) + max_momentum;
        return std::clamp(config_.cfl / scale, config_.min_timestep, config_.max_timestep);
    }

    void project(double& value, double& momentum) const noexcept {
        value = std::clamp(value, -config_.projection_limit, config_.projection_limit);
        const double momentum_limit = 4.0 * config_.projection_limit;
        momentum = std::clamp(momentum, -momentum_limit, momentum_limit);
    }

    double signed_noise(std::size_t cell, std::uint64_t epoch) const noexcept {
        std::uint64_t x = config_.seed;
        x ^= static_cast<std::uint64_t>(cell) + 0x9E3779B97F4A7C15ULL + (x << 6U) + (x >> 2U);
        x ^= epoch + 0xBF58476D1CE4E5B9ULL + (x << 6U) + (x >> 2U);
        x ^= x >> 30U;
        x *= 0xBF58476D1CE4E5B9ULL;
        x ^= x >> 27U;
        x *= 0x94D049BB133111EBULL;
        x ^= x >> 31U;
        const double unit = static_cast<double>(x >> 11U) * (1.0 / 9007199254740992.0);
        return unit * 2.0 - 1.0;
    }
};

} // namespace jarvisx
