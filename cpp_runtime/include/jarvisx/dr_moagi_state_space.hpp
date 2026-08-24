#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace jarvisx {

struct MultimodalIngress {
    std::uint64_t timestamp_ns{};
    std::vector<double> spatial_3d_mesh;
    std::vector<double> spatial_audio;
    std::vector<double> contextual_text;
};

struct OledTile {
    std::uint32_t tile_id{};
    std::uint32_t x_offset{};
    std::uint32_t y_offset{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<double> latent_slice;
    std::array<std::uint8_t, 3> preview_rgb{0U, 0U, 0U};
    double active_power_mw{};
};

// Single-producer/single-consumer ring buffer. The memory-ordering contract is
// intentionally narrow; this type must not be used as an MPMC queue.
template <typename T, std::size_t Capacity>
class SpscIngressRingBuffer {
    static_assert(Capacity >= 2U, "ring capacity must be at least two");

public:
    bool push(const T& item) {
        const std::size_t current_tail = tail_.load(std::memory_order_relaxed);
        const std::size_t next_tail = (current_tail + 1U) % Capacity;
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;
        }
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const std::size_t current_head = head_.load(std::memory_order_relaxed);
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }
        item = buffer_[current_head];
        head_.store((current_head + 1U) % Capacity, std::memory_order_release);
        return true;
    }

private:
    std::array<T, Capacity> buffer_{};
    std::atomic<std::size_t> head_{0U};
    std::atomic<std::size_t> tail_{0U};
};

class LinearStateGovernor {
public:
    static double infinity_norm(const std::vector<double>& matrix, std::size_t dim) {
        validate_square(matrix, dim);
        double max_row_sum = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
            double row_sum = 0.0;
            for (std::size_t j = 0; j < dim; ++j) {
                row_sum += std::abs(matrix[i * dim + j]);
            }
            max_row_sum = std::max(max_row_sum, row_sum);
        }
        return max_row_sum;
    }

    // Power-iteration estimate of the dominant linear growth factor. This is
    // telemetry, not a proof of the full nonlinear closed-loop Jacobian radius.
    static double dominant_growth_estimate(const std::vector<double>& matrix,
                                           std::size_t dim,
                                           std::size_t max_iter = 30U) {
        validate_square(matrix, dim);
        if (dim == 0U) {
            return 0.0;
        }

        std::vector<double> v(dim, 1.0 / std::sqrt(static_cast<double>(dim)));
        double estimate = 0.0;
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            std::vector<double> w(dim, 0.0);
            for (std::size_t i = 0; i < dim; ++i) {
                for (std::size_t j = 0; j < dim; ++j) {
                    w[i] += matrix[i * dim + j] * v[j];
                }
            }

            double norm_sq = 0.0;
            for (const double value : w) {
                norm_sq += value * value;
            }
            const double norm = std::sqrt(norm_sq);
            if (norm < 1.0e-14) {
                return 0.0;
            }
            estimate = norm;
            for (std::size_t i = 0; i < dim; ++i) {
                v[i] = w[i] / norm;
            }
        }
        return estimate;
    }

    // Enforces ||Phi||_inf <= target. Since rho(Phi) <= ||Phi||_inf, this is
    // a conservative sufficient bound for the linear transition operator.
    static double enforce_contractive_bound(std::vector<double>& matrix,
                                            std::size_t dim,
                                            double target = 0.94) {
        if (!(target > 0.0 && target < 1.0)) {
            throw std::invalid_argument("target contraction bound must lie in (0, 1)");
        }
        const double current = infinity_norm(matrix, dim);
        if (current <= target || current == 0.0) {
            return 1.0;
        }
        const double scale = target / current;
        for (double& value : matrix) {
            value *= scale;
        }
        return scale;
    }

private:
    static void validate_square(const std::vector<double>& matrix, std::size_t dim) {
        if (dim == 0U || matrix.size() != dim * dim) {
            throw std::invalid_argument("matrix dimensions do not match latent dimension");
        }
    }
};

struct DrMoagiStateSpaceConfig {
    std::size_t latent_dim{256U};
    std::size_t ingress_dim{32U};
    std::size_t tiles_x{8U};
    std::size_t tiles_y{8U};
    std::uint32_t logical_width{1'000'000U};
    std::uint32_t logical_height{1'000'000U};
    double transition_bound{0.94};
    double state_abs_limit{1.0e6};

    void validate() const {
        if (latent_dim == 0U || ingress_dim == 0U || tiles_x == 0U || tiles_y == 0U) {
            throw std::invalid_argument("state-space dimensions must be non-zero");
        }
        if (logical_width == 0U || logical_height == 0U) {
            throw std::invalid_argument("logical display dimensions must be non-zero");
        }
        if (!(transition_bound > 0.0 && transition_bound < 1.0)) {
            throw std::invalid_argument("transition bound must lie in (0, 1)");
        }
        if (!(state_abs_limit > 0.0) || !std::isfinite(state_abs_limit)) {
            throw std::invalid_argument("state absolute limit must be finite and positive");
        }
    }

    std::size_t total_tiles() const noexcept { return tiles_x * tiles_y; }
};

class DrMoagiStateSpaceEngine {
public:
    explicit DrMoagiStateSpaceEngine(DrMoagiStateSpaceConfig config)
        : config_(std::move(config)),
          z_state_(config_.latent_dim, 0.05),
          phi_(config_.latent_dim * config_.latent_dim, 0.0),
          psi_(config_.latent_dim * config_.ingress_dim, 0.02) {
        config_.validate();
        for (std::size_t i = 0; i < config_.latent_dim; ++i) {
            phi_[i * config_.latent_dim + i] = 0.85;
        }
    }

    std::vector<double> encode_ingress(const MultimodalIngress& ingress) const {
        std::vector<double> u(config_.ingress_dim, 0.0);
        accumulate_weighted(u, ingress.spatial_3d_mesh, 0.5);
        accumulate_weighted(u, ingress.spatial_audio, 0.3);
        accumulate_weighted(u, ingress.contextual_text, 0.2);
        return u;
    }

    void step_state_space(const std::vector<double>& u,
                          const std::vector<double>& delta_context) {
        if (u.size() != config_.ingress_dim) {
            throw std::invalid_argument("ingress vector dimension mismatch");
        }
        if (delta_context.size() != config_.latent_dim) {
            throw std::invalid_argument("context vector dimension mismatch");
        }

        LinearStateGovernor::enforce_contractive_bound(
            phi_, config_.latent_dim, config_.transition_bound);

        std::vector<double> next(config_.latent_dim, 0.0);
        for (std::size_t i = 0; i < config_.latent_dim; ++i) {
            double state_sum = 0.0;
            for (std::size_t j = 0; j < config_.latent_dim; ++j) {
                state_sum += phi_[i * config_.latent_dim + j] * z_state_[j];
            }

            double input_sum = 0.0;
            for (std::size_t j = 0; j < config_.ingress_dim; ++j) {
                input_sum += psi_[i * config_.ingress_dim + j] * u[j];
            }

            const double candidate = state_sum + input_sum + delta_context[i];
            if (!std::isfinite(candidate)) {
                throw std::runtime_error("non-finite latent state rejected by admissibility gate");
            }
            next[i] = std::clamp(candidate, -config_.state_abs_limit, config_.state_abs_limit);
        }
        z_state_.swap(next);
    }

    std::vector<OledTile> decode_and_dispatch_tiles() const {
        const std::size_t total = config_.total_tiles();
        std::vector<OledTile> tiles(total);
        const unsigned int hw = std::thread::hardware_concurrency();
        const std::size_t available = hw == 0U ? 1U : static_cast<std::size_t>(hw);
        const std::size_t worker_count = std::max<std::size_t>(1U, std::min(available, total));
        const std::size_t batch = (total + worker_count - 1U) / worker_count;

        auto decode_work = [this, total, &tiles](std::size_t begin_tile, std::size_t end_tile) {
            for (std::size_t tile_index = begin_tile; tile_index < end_tile; ++tile_index) {
                const std::size_t latent_begin = tile_index * config_.latent_dim / total;
                const std::size_t latent_end = (tile_index + 1U) * config_.latent_dim / total;

                OledTile tile;
                tile.tile_id = static_cast<std::uint32_t>(tile_index);
                const std::size_t tx = tile_index % config_.tiles_x;
                const std::size_t ty = tile_index / config_.tiles_x;
                tile.x_offset = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(config_.logical_width) * tx) / config_.tiles_x);
                tile.y_offset = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(config_.logical_height) * ty) / config_.tiles_y);
                const std::uint32_t x_end = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(config_.logical_width) * (tx + 1U)) / config_.tiles_x);
                const std::uint32_t y_end = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(config_.logical_height) * (ty + 1U)) / config_.tiles_y);
                tile.width = x_end - tile.x_offset;
                tile.height = y_end - tile.y_offset;
                tile.latent_slice.assign(z_state_.begin() + static_cast<std::ptrdiff_t>(latent_begin),
                                         z_state_.begin() + static_cast<std::ptrdiff_t>(latent_end));

                double energy = 0.0;
                for (const double value : tile.latent_slice) {
                    energy += std::abs(value);
                }
                if (energy >= 0.001) {
                    tile.preview_rgb = {
                        saturating_byte(energy * 100.0),
                        saturating_byte(energy * 150.0),
                        saturating_byte(energy * 200.0)};
                    tile.active_power_mw = energy * 12.5;
                }
                tiles[tile_index] = std::move(tile);
            }
        };

        std::vector<std::future<void>> futures;
        futures.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const std::size_t begin_tile = worker * batch;
            const std::size_t end_tile = std::min(begin_tile + batch, total);
            if (begin_tile < end_tile) {
                futures.emplace_back(std::async(std::launch::async, decode_work, begin_tile, end_tile));
            }
        }
        for (auto& future : futures) {
            future.get();
        }
        return tiles;
    }

    double transition_infinity_norm() const {
        return LinearStateGovernor::infinity_norm(phi_, config_.latent_dim);
    }

    double transition_growth_estimate() const {
        return LinearStateGovernor::dominant_growth_estimate(phi_, config_.latent_dim);
    }

    double latent_norm() const noexcept {
        double norm_sq = 0.0;
        for (const double value : z_state_) {
            norm_sq += value * value;
        }
        return std::sqrt(norm_sq);
    }

    const std::vector<double>& state() const noexcept { return z_state_; }
    const DrMoagiStateSpaceConfig& config() const noexcept { return config_; }

private:
    static void accumulate_weighted(std::vector<double>& target,
                                    const std::vector<double>& source,
                                    double weight) {
        const std::size_t count = std::min(target.size(), source.size());
        for (std::size_t i = 0; i < count; ++i) {
            target[i] += source[i] * weight;
        }
    }

    static std::uint8_t saturating_byte(double value) noexcept {
        const double bounded = std::clamp(value, 0.0, 255.0);
        return static_cast<std::uint8_t>(bounded);
    }

    DrMoagiStateSpaceConfig config_;
    std::vector<double> z_state_;
    std::vector<double> phi_;
    std::vector<double> psi_;
};

} // namespace jarvisx
