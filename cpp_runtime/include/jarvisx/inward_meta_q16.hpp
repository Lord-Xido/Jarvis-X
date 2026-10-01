#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jarvisx::dmimte {

constexpr int WX = 64;
constexpr int WY = 64;
constexpr int WZ = 64;
constexpr int WORLD_SIZE = WX * WY * WZ;
constexpr int BX = 16;
constexpr int BY = 16;
constexpr int BZ = 16;
constexpr int LATENT_SIZE = BX * BY * BZ;
constexpr int BLK = WX / BX;
constexpr int CX = 4;
constexpr int CY = 4;
constexpr int CZ = 4;
constexpr int CORE_SIZE = CX * CY * CZ;
constexpr std::int32_t Q_ONE = 1 << 16;
constexpr std::int32_t Q_HALF = Q_ONE / 2;

inline std::int32_t clamp_q16(std::int64_t value) noexcept {
    if (value > std::numeric_limits<std::int32_t>::max()) {
        return std::numeric_limits<std::int32_t>::max();
    }
    if (value < std::numeric_limits<std::int32_t>::min()) {
        return std::numeric_limits<std::int32_t>::min();
    }
    return static_cast<std::int32_t>(value);
}

inline std::int32_t q16_mul(std::int32_t a, std::int32_t b) noexcept {
    const std::int64_t product = static_cast<std::int64_t>(a) * static_cast<std::int64_t>(b);
    return clamp_q16(product >> 16);
}

inline std::int32_t q16_add(std::int32_t a, std::int32_t b) noexcept {
    return clamp_q16(static_cast<std::int64_t>(a) + static_cast<std::int64_t>(b));
}

inline double q16_to_double(std::int32_t value) noexcept {
    return static_cast<double>(value) / static_cast<double>(Q_ONE);
}

inline std::int32_t q16_from_double(double value) noexcept {
    const double scaled = value * static_cast<double>(Q_ONE);
    if (scaled >= static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
        return std::numeric_limits<std::int32_t>::max();
    }
    if (scaled <= static_cast<double>(std::numeric_limits<std::int32_t>::min())) {
        return std::numeric_limits<std::int32_t>::min();
    }
    return static_cast<std::int32_t>(std::llround(scaled));
}

struct HierarchicalBlockEncQ16 {
    static constexpr int latent_index(int bx, int by, int bz) noexcept {
        return bx + by * BX + bz * BX * BY;
    }

    static constexpr int world_index(int x, int y, int z) noexcept {
        return x + y * WX + z * WX * WY;
    }

    static void apply(const std::vector<std::int32_t>& x,
                      std::vector<std::int32_t>& z) {
        if (x.size() != static_cast<std::size_t>(WORLD_SIZE)) {
            throw std::invalid_argument("W input must contain 64^3 Q16 voxels");
        }
        z.assign(static_cast<std::size_t>(LATENT_SIZE), 0);
        for (int bz = 0; bz < BZ; ++bz) {
            for (int by = 0; by < BY; ++by) {
                for (int bx = 0; bx < BX; ++bx) {
                    std::int64_t sum = 0;
                    for (int dz = 0; dz < BLK; ++dz) {
                        for (int dy = 0; dy < BLK; ++dy) {
                            for (int dx = 0; dx < BLK; ++dx) {
                                const int xw = bx * BLK + dx;
                                const int yw = by * BLK + dy;
                                const int zw = bz * BLK + dz;
                                sum += x[static_cast<std::size_t>(world_index(xw, yw, zw))];
                            }
                        }
                    }
                    // 1/sqrt(64) = 1/8. Division is used instead of signed shifts.
                    z[static_cast<std::size_t>(latent_index(bx, by, bz))] =
                        clamp_q16(sum / 8);
                }
            }
        }
    }

    static void applyT(const std::vector<std::int32_t>& z,
                       std::vector<std::int32_t>& x) {
        if (z.size() != static_cast<std::size_t>(LATENT_SIZE)) {
            throw std::invalid_argument("W^T input must contain 16^3 Q16 latents");
        }
        x.assign(static_cast<std::size_t>(WORLD_SIZE), 0);
        for (int bz = 0; bz < BZ; ++bz) {
            for (int by = 0; by < BY; ++by) {
                for (int bx = 0; bx < BX; ++bx) {
                    const auto v = z[static_cast<std::size_t>(latent_index(bx, by, bz))] / 8;
                    for (int dz = 0; dz < BLK; ++dz) {
                        for (int dy = 0; dy < BLK; ++dy) {
                            for (int dx = 0; dx < BLK; ++dx) {
                                const int xw = bx * BLK + dx;
                                const int yw = by * BLK + dy;
                                const int zw = bz * BLK + dz;
                                x[static_cast<std::size_t>(world_index(xw, yw, zw))] = v;
                            }
                        }
                    }
                }
            }
        }
    }
};

struct EngineConfig {
    std::int32_t local_lr = q16_from_double(0.50);
    std::int32_t gate_eta = q16_from_double(0.25);
    std::int32_t core_gain = q16_from_double(0.25);
    std::int32_t core_rho = q16_from_double(0.75);
    std::int32_t min_gate = q16_from_double(0.50);
    std::int32_t max_gate = q16_from_double(1.50);
    std::int32_t initial_noise_amp = q16_from_double(0.0015);
    std::int32_t noise_decay = q16_from_double(std::exp(-1.0 / 50.0));
    bool enable_noise = true;
    std::uint32_t seed = 0x7FFF0116U;

    void validate() const {
        if (local_lr <= 0 || local_lr >= Q_ONE) {
            throw std::invalid_argument("local_lr must be in (0,1)");
        }
        if (gate_eta < 0 || core_gain < 0) {
            throw std::invalid_argument("gate_eta and core_gain must be non-negative");
        }
        if (core_rho < 0 || core_rho >= Q_ONE) {
            throw std::invalid_argument("core_rho must be in [0,1)");
        }
        if (min_gate <= 0 || max_gate < min_gate) {
            throw std::invalid_argument("invalid multiplicative gate bounds");
        }
        if (initial_noise_amp < 0 || noise_decay < 0 || noise_decay > Q_ONE) {
            throw std::invalid_argument("invalid annealing configuration");
        }
    }
};

struct StepMetrics {
    int step = 0;
    double rmse = 0.0;
    double projection_floor_rmse = 0.0;
    double latent_max_error = 0.0;
    double core_max_abs = 0.0;
    double noise_amplitude = 0.0;
};

class InwardMetaEngineQ16 {
public:
    explicit InwardMetaEngineQ16(EngineConfig config = {})
        : config_(std::move(config)), rng_state_(config_.seed) {
        config_.validate();
        x_.assign(static_cast<std::size_t>(WORLD_SIZE), 0);
        xhat_.assign(static_cast<std::size_t>(WORLD_SIZE), 0);
        target_latent_.assign(static_cast<std::size_t>(LATENT_SIZE), 0);
        z_.assign(static_cast<std::size_t>(LATENT_SIZE), 0);
        warm_z_.assign(static_cast<std::size_t>(LATENT_SIZE), 0);
        dz_.assign(static_cast<std::size_t>(LATENT_SIZE), 0);
        core_residual_.fill(0);
        core_memory_.fill(0);
        current_noise_amp_ = config_.initial_noise_amp;
        build_complex_target();
        HierarchicalBlockEncQ16::apply(x_, target_latent_);
        projection_floor_rmse_ = compute_projection_floor();
        HierarchicalBlockEncQ16::applyT(z_, xhat_);
    }

    const std::vector<std::int32_t>& target() const noexcept { return x_; }
    const std::vector<std::int32_t>& reconstruction() const noexcept { return xhat_; }
    const std::vector<std::int32_t>& latent() const noexcept { return z_; }
    const std::vector<std::int32_t>& target_latent() const noexcept { return target_latent_; }
    const std::array<std::int32_t, CORE_SIZE>& core_memory() const noexcept { return core_memory_; }
    double projection_floor_rmse() const noexcept { return projection_floor_rmse_; }
    int step_index() const noexcept { return step_; }

    double rmse() const {
        return rmse_between(x_, xhat_);
    }

    StepMetrics step() {
        // W*x is precomputed once. The hot path is 4096 latent nodes, not 64^3 world voxels.
        for (int i = 0; i < LATENT_SIZE; ++i) {
            dz_[static_cast<std::size_t>(i)] = clamp_q16(
                static_cast<std::int64_t>(target_latent_[static_cast<std::size_t>(i)])
                - static_cast<std::int64_t>(warm_z_[static_cast<std::size_t>(i)]));
        }

        restrict_residual_to_core();
        update_core_memory();
        outward_permeate();

        HierarchicalBlockEncQ16::applyT(z_, xhat_);

        StepMetrics metrics;
        metrics.step = step_;
        metrics.rmse = rmse();
        metrics.projection_floor_rmse = projection_floor_rmse_;
        metrics.latent_max_error = latent_max_error();
        metrics.core_max_abs = core_max_abs();
        metrics.noise_amplitude = q16_to_double(current_noise_amp_);

        // Diffusion perturbs only the next warm start. It never contaminates reported z* or xhat.
        warm_z_ = z_;
        anneal_diffuse_warm_start();
        ++step_;
        return metrics;
    }

private:
    EngineConfig config_;
    std::vector<std::int32_t> x_;
    std::vector<std::int32_t> xhat_;
    std::vector<std::int32_t> target_latent_;
    std::vector<std::int32_t> z_;
    std::vector<std::int32_t> warm_z_;
    std::vector<std::int32_t> dz_;
    std::array<std::int32_t, CORE_SIZE> core_residual_{};
    std::array<std::int32_t, CORE_SIZE> core_memory_{};
    std::uint32_t rng_state_ = 0;
    std::int32_t current_noise_amp_ = 0;
    int step_ = 0;
    double projection_floor_rmse_ = 0.0;

    static constexpr int core_index_from_latent(int latent_index) noexcept {
        const int x = latent_index % BX;
        const int y = (latent_index / BX) % BY;
        const int z = latent_index / (BX * BY);
        return (x >> 2) + (y >> 2) * CX + (z >> 2) * CX * CY;
    }

    void build_complex_target() {
        for (int z = 0; z < WZ; ++z) {
            for (int y = 0; y < WY; ++y) {
                for (int x = 0; x < WX; ++x) {
                    const std::int32_t dx = (x << 1) - 63;
                    const std::int32_t dy = (y << 1) - 63;
                    const std::int32_t dz = (z << 1) - 63;
                    const std::int32_t distance_squared = dx * dx + dy * dy + dz * dz;
                    const bool in_sphere = distance_squared < 1600;
                    const bool in_core = distance_squared < 256;
                    const bool lattice = ((x % 8 == 0) || (y % 8 == 0) || (z % 8 == 0))
                                      && distance_squared < 2704;
                    const auto index = static_cast<std::size_t>(
                        HierarchicalBlockEncQ16::world_index(x, y, z));
                    x_[index] = ((in_sphere && !in_core) || lattice)
                        ? Q_ONE
                        : (in_core ? Q_HALF : 0);
                }
            }
        }
    }

    void restrict_residual_to_core() {
        // Reset every step; the submitted prototype accumulated core_dz across outer steps.
        std::array<std::int64_t, CORE_SIZE> accum{};
        std::array<int, CORE_SIZE> counts{};
        core_residual_.fill(0);

        for (int i = 0; i < LATENT_SIZE; ++i) {
            const int core = core_index_from_latent(i);
            accum[static_cast<std::size_t>(core)] += dz_[static_cast<std::size_t>(i)];
            ++counts[static_cast<std::size_t>(core)];
        }
        for (int i = 0; i < CORE_SIZE; ++i) {
            const int count = counts[static_cast<std::size_t>(i)];
            core_residual_[static_cast<std::size_t>(i)] = count == 0
                ? 0
                : clamp_q16(accum[static_cast<std::size_t>(i)] / count);
        }
    }

    void update_core_memory() {
        const std::int32_t one_minus_rho = Q_ONE - config_.core_rho;
        for (int i = 0; i < CORE_SIZE; ++i) {
            const auto previous = q16_mul(
                config_.core_rho, core_memory_[static_cast<std::size_t>(i)]);
            const auto incoming = q16_mul(
                one_minus_rho, core_residual_[static_cast<std::size_t>(i)]);
            core_memory_[static_cast<std::size_t>(i)] = q16_add(previous, incoming);
        }
    }

    void outward_permeate() {
        for (int i = 0; i < LATENT_SIZE; ++i) {
            const int core = core_index_from_latent(i);
            const auto coarse = core_memory_[static_cast<std::size_t>(core)];
            auto gate = q16_add(Q_ONE, q16_mul(config_.gate_eta, coarse));
            gate = std::clamp(gate, config_.min_gate, config_.max_gate);

            // Multiplicative permeation modulates the correction, not z itself.
            // This avoids the zero-state deadlock z=0 => z remains 0.
            const auto local = q16_mul(
                config_.local_lr, q16_mul(gate, dz_[static_cast<std::size_t>(i)]));
            const auto global = q16_mul(config_.core_gain, coarse);
            const auto correction = q16_add(local, global);
            z_[static_cast<std::size_t>(i)] =
                q16_add(warm_z_[static_cast<std::size_t>(i)], correction);
        }
    }

    void anneal_diffuse_warm_start() {
        if (!config_.enable_noise || current_noise_amp_ <= 0) {
            return;
        }
        for (auto& value : warm_z_) {
            rng_state_ = rng_state_ * 1664525U + 1013904223U;
            const std::int32_t raw = static_cast<std::int32_t>(rng_state_ >> 16U) - 32768;
            // raw/32768 is approximately U[-1,1); multiplication avoids signed shift UB.
            const std::int32_t normalized =
                clamp_q16(static_cast<std::int64_t>(raw) * 2);
            value = q16_add(value, q16_mul(current_noise_amp_, normalized));
        }
        current_noise_amp_ = q16_mul(current_noise_amp_, config_.noise_decay);
    }

    double latent_max_error() const {
        double maximum = 0.0;
        for (int i = 0; i < LATENT_SIZE; ++i) {
            const auto delta =
                static_cast<std::int64_t>(target_latent_[static_cast<std::size_t>(i)])
                - static_cast<std::int64_t>(z_[static_cast<std::size_t>(i)]);
            maximum = std::max(
                maximum,
                std::abs(static_cast<double>(delta)) / static_cast<double>(Q_ONE));
        }
        return maximum;
    }

    double core_max_abs() const {
        double maximum = 0.0;
        for (const auto value : core_memory_) {
            maximum = std::max(maximum, std::abs(q16_to_double(value)));
        }
        return maximum;
    }

    static double rmse_between(const std::vector<std::int32_t>& a,
                               const std::vector<std::int32_t>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("RMSE vectors must have identical sizes");
        }
        long double sum = 0.0L;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const long double delta =
                static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
            const long double normalized = delta / static_cast<long double>(Q_ONE);
            sum += normalized * normalized;
        }
        return std::sqrt(static_cast<double>(sum / static_cast<long double>(a.size())));
    }

    double compute_projection_floor() const {
        std::vector<std::int32_t> projection;
        HierarchicalBlockEncQ16::applyT(target_latent_, projection);
        return rmse_between(x_, projection);
    }
};

} // namespace jarvisx::dmimte
