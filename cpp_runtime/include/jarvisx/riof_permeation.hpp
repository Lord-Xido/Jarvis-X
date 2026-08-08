#pragma once

#include "jarvisx/autoencoder3d.hpp"
#include "jarvisx/riof3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace jarvisx {

struct RiofPermeationConfig {
    std::size_t cycles{4U};
    Riof3DConfig dynamics{};

    void validate() const {
        if (cycles > 4096U) {
            throw std::invalid_argument("RIOF permeation cycles must be <= 4096");
        }
        if (dynamics.edge < 4U || dynamics.edge > 128U) {
            throw std::invalid_argument("RIOF permeation edge must be in [4, 128]");
        }
        if (dynamics.adjoint_iterations == 0U || dynamics.adjoint_iterations > 256U) {
            throw std::invalid_argument("RIOF permeation adjoint iterations must be in [1, 256]");
        }
        const double scalars[] = {
            dynamics.smoothness_weight, dynamics.bending_weight,
            dynamics.prediction_weight, dynamics.refinement_gain,
            dynamics.enhancement_gain, dynamics.memory_gain,
            dynamics.base_damping, dynamics.cfl, dynamics.min_timestep,
            dynamics.max_timestep, dynamics.projection_limit
        };
        for (double value : scalars) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument("RIOF permeation parameters must be finite and non-negative");
            }
        }
        if (dynamics.cfl <= 0.0 || dynamics.min_timestep <= 0.0 ||
            dynamics.max_timestep <= 0.0 ||
            dynamics.min_timestep > dynamics.max_timestep) {
            throw std::invalid_argument("RIOF permeation timestep bounds are invalid");
        }
        if (dynamics.projection_limit <= 0.0 || dynamics.projection_limit > 100.0) {
            throw std::invalid_argument("RIOF permeation projection limit must be in (0, 100]");
        }
    }
};

struct RiofPermeationMetrics {
    std::size_t cycles{};
    double initial_mean_abs_residual{};
    double final_mean_abs_residual{};
    double mean_abs_delta{};
    double max_abs_delta{};
    double last_timestep{};
    double last_damping{};
    double last_enhancement{};
};

struct RiofPermeationResult {
    Tensor4D latent;
    RiofPermeationMetrics metrics;
};

struct RiofAutoencodingResult {
    Tensor4D latent;
    Tensor4D reconstruction;
    RiofPermeationMetrics metrics;
};

class RiofLatentPermeator {
public:
    explicit RiofLatentPermeator(RiofPermeationConfig config)
        : config_(config) {
        config_.validate();
    }

    const RiofPermeationConfig& config() const noexcept { return config_; }

    RiofPermeationResult permeate(const Tensor4D& latent) const {
        validate_latent(latent);
        Tensor4D state = latent;
        Tensor4D velocity(latent.shape(), 0.0F);
        Tensor4D memory(latent.shape(), 0.0F);
        Tensor4D residual(latent.shape(), 0.0F);
        Tensor4D laplacian(latent.shape(), 0.0F);
        Tensor4D bilaplacian(latent.shape(), 0.0F);
        Tensor4D adjoint(latent.shape(), 0.0F);
        Tensor4D force(latent.shape(), 0.0F);

        RiofPermeationMetrics metrics;
        metrics.cycles = config_.cycles;
        update_geometry(state, residual, laplacian, bilaplacian);
        metrics.initial_mean_abs_residual = mean_abs(residual);

        for (std::size_t cycle = 0; cycle < config_.cycles; ++cycle) {
            update_geometry(state, residual, laplacian, bilaplacian);
            solve_adjoint(residual, adjoint);
            const Control first = compute_force(
                velocity, memory, residual, laplacian,
                bilaplacian, adjoint, force, cycle);
            const double dt = adaptive_timestep(first.max_force, velocity);

            integrate_half_step(state, velocity, force, dt, true);

            update_geometry(state, residual, laplacian, bilaplacian);
            solve_adjoint(residual, adjoint);
            const Control second = compute_force(
                velocity, memory, residual, laplacian,
                bilaplacian, adjoint, force, cycle + 1U);
            integrate_half_step(state, velocity, force, dt, false);
            update_memory(memory, residual);

            metrics.last_timestep = dt;
            metrics.last_damping = second.damping;
            metrics.last_enhancement = second.enhancement;
        }

        update_geometry(state, residual, laplacian, bilaplacian);
        metrics.final_mean_abs_residual = mean_abs(residual);
        measure_delta(latent, state, metrics);
        return RiofPermeationResult{state, metrics};
    }

private:
    struct Control {
        double max_force{};
        double damping{};
        double enhancement{};
    };

    RiofPermeationConfig config_;

    static std::size_t wrap_plus(std::size_t value, std::size_t edge) noexcept {
        return value + 1U == edge ? 0U : value + 1U;
    }

    static std::size_t wrap_minus(std::size_t value, std::size_t edge) noexcept {
        return value == 0U ? edge - 1U : value - 1U;
    }

    void validate_latent(const Tensor4D& latent) const {
        const TensorShape4D& shape = latent.shape();
        if (shape.channels == 0U || shape.depth != shape.height ||
            shape.depth != shape.width) {
            throw std::invalid_argument("RIOF permeation requires non-empty cubic latent channels");
        }
        if (shape.width != config_.dynamics.edge) {
            throw std::invalid_argument("RIOF permeation edge must match latent spatial edge");
        }
    }

    static double neighbor_mean(const Tensor4D& field, std::size_t channel,
                                std::size_t z, std::size_t y, std::size_t x) {
        const std::size_t edge = field.shape().width;
        return (static_cast<double>(field(channel, z, y, wrap_plus(x, edge))) +
                static_cast<double>(field(channel, z, y, wrap_minus(x, edge))) +
                static_cast<double>(field(channel, z, wrap_plus(y, edge), x)) +
                static_cast<double>(field(channel, z, wrap_minus(y, edge), x)) +
                static_cast<double>(field(channel, wrap_plus(z, edge), y, x)) +
                static_cast<double>(field(channel, wrap_minus(z, edge), y, x))) / 6.0;
    }

    static double laplacian_at(const Tensor4D& field, std::size_t channel,
                               std::size_t z, std::size_t y, std::size_t x) {
        const double center = static_cast<double>(field(channel, z, y, x));
        return 6.0 * (neighbor_mean(field, channel, z, y, x) - center);
    }

    static void update_geometry(const Tensor4D& state, Tensor4D& residual,
                                Tensor4D& laplacian, Tensor4D& bilaplacian) {
        const TensorShape4D& shape = state.shape();
        for (std::size_t c = 0; c < shape.channels; ++c) {
            for (std::size_t z = 0; z < shape.depth; ++z) {
                for (std::size_t y = 0; y < shape.height; ++y) {
                    for (std::size_t x = 0; x < shape.width; ++x) {
                        const double center = static_cast<double>(state(c, z, y, x));
                        const double prediction = neighbor_mean(state, c, z, y, x);
                        residual(c, z, y, x) = static_cast<float>(center - prediction);
                        laplacian(c, z, y, x) = static_cast<float>(
                            6.0 * (prediction - center));
                    }
                }
            }
        }
        for (std::size_t c = 0; c < shape.channels; ++c) {
            for (std::size_t z = 0; z < shape.depth; ++z) {
                for (std::size_t y = 0; y < shape.height; ++y) {
                    for (std::size_t x = 0; x < shape.width; ++x) {
                        bilaplacian(c, z, y, x) = static_cast<float>(
                            laplacian_at(laplacian, c, z, y, x));
                    }
                }
            }
        }
    }

    void solve_adjoint(const Tensor4D& residual, Tensor4D& adjoint) const {
        Tensor4D next(adjoint.shape(), 0.0F);
        const TensorShape4D& shape = adjoint.shape();
        for (std::size_t iteration = 0;
             iteration < config_.dynamics.adjoint_iterations; ++iteration) {
            for (std::size_t c = 0; c < shape.channels; ++c) {
                for (std::size_t z = 0; z < shape.depth; ++z) {
                    for (std::size_t y = 0; y < shape.height; ++y) {
                        for (std::size_t x = 0; x < shape.width; ++x) {
                            const double relaxed = neighbor_mean(adjoint, c, z, y, x) -
                                static_cast<double>(residual(c, z, y, x)) / 6.0;
                            next(c, z, y, x) = static_cast<float>(relaxed);
                        }
                    }
                }
            }
            adjoint = next;
        }

        for (std::size_t c = 0; c < shape.channels; ++c) {
            double mean = 0.0;
            for (std::size_t z = 0; z < shape.depth; ++z)
                for (std::size_t y = 0; y < shape.height; ++y)
                    for (std::size_t x = 0; x < shape.width; ++x)
                        mean += static_cast<double>(adjoint(c, z, y, x));
            const double count = static_cast<double>(shape.depth * shape.height * shape.width);
            mean /= count;
            for (std::size_t z = 0; z < shape.depth; ++z)
                for (std::size_t y = 0; y < shape.height; ++y)
                    for (std::size_t x = 0; x < shape.width; ++x)
                        adjoint(c, z, y, x) = static_cast<float>(
                            static_cast<double>(adjoint(c, z, y, x)) - mean);
        }
    }

    Control compute_force(const Tensor4D& velocity,
                          const Tensor4D& memory, const Tensor4D& residual,
                          const Tensor4D& laplacian, const Tensor4D& bilaplacian,
                          const Tensor4D& adjoint, Tensor4D& force,
                          std::size_t epoch) const {
        const double residual_mean = mean_abs(residual);
        const double curvature_mean = mean_abs(laplacian);
        const double disagreement = residual_mean / (curvature_mean + 1.0e-9);
        const double enhancement = config_.dynamics.enhancement_gain *
            std::clamp(disagreement, 0.0, 2.0);
        const double damping = config_.dynamics.base_damping *
            (1.0 + std::clamp(residual_mean, 0.0, 2.0));

        const TensorShape4D& shape = force.shape();
        double max_force = 0.0;
        for (std::size_t c = 0; c < shape.channels; ++c) {
            for (std::size_t z = 0; z < shape.depth; ++z) {
                for (std::size_t y = 0; y < shape.height; ++y) {
                    for (std::size_t x = 0; x < shape.width; ++x) {
                        const double curvature = std::abs(
                            static_cast<double>(laplacian(c, z, y, x)));
                        const double curvature_scale = curvature /
                            (curvature_mean + 1.0e-9);
                        const double high_frequency = signed_noise(c, z, y, x, epoch) *
                            std::clamp(curvature_scale, 0.0, 3.0);
                        const double value =
                            config_.dynamics.smoothness_weight *
                                static_cast<double>(laplacian(c, z, y, x)) -
                            config_.dynamics.bending_weight *
                                static_cast<double>(bilaplacian(c, z, y, x)) -
                            config_.dynamics.prediction_weight *
                                static_cast<double>(residual(c, z, y, x)) -
                            config_.dynamics.refinement_gain *
                                static_cast<double>(adjoint(c, z, y, x)) -
                            config_.dynamics.memory_gain *
                                static_cast<double>(memory(c, z, y, x)) +
                            enhancement * high_frequency -
                            damping * static_cast<double>(velocity(c, z, y, x));
                        force(c, z, y, x) = static_cast<float>(value);
                        max_force = std::max(max_force, std::abs(value));
                    }
                }
            }
        }
        return Control{max_force, damping, enhancement};
    }

    double adaptive_timestep(double max_force, const Tensor4D& velocity) const {
        double max_velocity = 0.0;
        for (float value : velocity.values()) {
            max_velocity = std::max(max_velocity, std::abs(static_cast<double>(value)));
        }
        const double scale = 1.0 + std::sqrt(std::max(0.0, max_force)) + max_velocity;
        return std::clamp(config_.dynamics.cfl / scale,
                          config_.dynamics.min_timestep,
                          config_.dynamics.max_timestep);
    }

    void integrate_half_step(Tensor4D& state, Tensor4D& velocity,
                             const Tensor4D& force, double dt,
                             bool advance_position) const {
        const TensorShape4D& shape = state.shape();
        const double value_limit = config_.dynamics.projection_limit;
        const double velocity_limit = 4.0 * value_limit;
        for (std::size_t c = 0; c < shape.channels; ++c) {
            for (std::size_t z = 0; z < shape.depth; ++z) {
                for (std::size_t y = 0; y < shape.height; ++y) {
                    for (std::size_t x = 0; x < shape.width; ++x) {
                        double v = static_cast<double>(velocity(c, z, y, x)) +
                            0.5 * dt * static_cast<double>(force(c, z, y, x));
                        v = std::clamp(v, -velocity_limit, velocity_limit);
                        double q = static_cast<double>(state(c, z, y, x));
                        if (advance_position) q += dt * v;
                        q = std::clamp(q, -value_limit, value_limit);
                        velocity(c, z, y, x) = static_cast<float>(v);
                        state(c, z, y, x) = static_cast<float>(q);
                    }
                }
            }
        }
    }

    static void update_memory(Tensor4D& memory, const Tensor4D& residual) {
        const TensorShape4D& shape = memory.shape();
        for (std::size_t c = 0; c < shape.channels; ++c)
            for (std::size_t z = 0; z < shape.depth; ++z)
                for (std::size_t y = 0; y < shape.height; ++y)
                    for (std::size_t x = 0; x < shape.width; ++x)
                        memory(c, z, y, x) = static_cast<float>(
                            0.97 * static_cast<double>(memory(c, z, y, x)) +
                            0.03 * static_cast<double>(residual(c, z, y, x)));
    }

    static double mean_abs(const Tensor4D& field) {
        if (field.size() == 0U) return 0.0;
        double sum = 0.0;
        for (float value : field.values()) sum += std::abs(static_cast<double>(value));
        return sum / static_cast<double>(field.size());
    }

    static void measure_delta(const Tensor4D& before, const Tensor4D& after,
                              RiofPermeationMetrics& metrics) {
        double sum = 0.0;
        double max_delta = 0.0;
        for (std::size_t i = 0; i < before.size(); ++i) {
            const double delta = std::abs(
                static_cast<double>(after.values().at(i)) -
                static_cast<double>(before.values().at(i)));
            sum += delta;
            max_delta = std::max(max_delta, delta);
        }
        metrics.mean_abs_delta = before.size() == 0U ? 0.0 :
            sum / static_cast<double>(before.size());
        metrics.max_abs_delta = max_delta;
    }

    double signed_noise(std::size_t channel, std::size_t z,
                        std::size_t y, std::size_t x,
                        std::size_t epoch) const noexcept {
        const std::size_t edge = config_.dynamics.edge;
        const std::size_t cell = x + edge * (y + edge * (z + edge * channel));
        std::uint64_t hash = config_.dynamics.seed;
        hash ^= static_cast<std::uint64_t>(cell) + 0x9E3779B97F4A7C15ULL +
                (hash << 6U) + (hash >> 2U);
        hash ^= static_cast<std::uint64_t>(epoch) + 0xBF58476D1CE4E5B9ULL +
                (hash << 6U) + (hash >> 2U);
        hash ^= hash >> 30U;
        hash *= 0xBF58476D1CE4E5B9ULL;
        hash ^= hash >> 27U;
        hash *= 0x94D049BB133111EBULL;
        hash ^= hash >> 31U;
        const double unit = static_cast<double>(hash >> 11U) *
            (1.0 / 9007199254740992.0);
        return unit * 2.0 - 1.0;
    }
};

inline RiofAutoencodingResult reconstruct_with_riof(
    const Autoencoder3D& model, const Tensor4D& input,
    RiofPermeationConfig config, bool quantize = false) {
    Tensor4D latent = model.encode(input, quantize);
    config.dynamics.edge = latent.shape().width;
    RiofLatentPermeator permeator(config);
    RiofPermeationResult refined = permeator.permeate(latent);
    Tensor4D reconstruction = model.decode(refined.latent);
    return RiofAutoencodingResult{
        refined.latent, reconstruction, refined.metrics
    };
}

} // namespace jarvisx
