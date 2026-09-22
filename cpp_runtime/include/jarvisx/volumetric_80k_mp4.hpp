#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jarvisx::volumetric_80k {

constexpr std::uint32_t kAxis = 80000u;
constexpr std::uint64_t kLogicalVoxels =
    static_cast<std::uint64_t>(kAxis) * kAxis * kAxis;
constexpr std::uint32_t kCoordBits = 17u;
constexpr std::uint32_t kBrickEdge = 32u;
constexpr std::uint32_t kBrickVoxels =
    kBrickEdge * kBrickEdge * kBrickEdge;
constexpr std::uint32_t kBrickAxis = (kAxis + kBrickEdge - 1u) / kBrickEdge;
constexpr std::uint32_t kBrickBits = 12u;
constexpr std::size_t kLatentDim = 8u;
constexpr float kEpsilon = 1.0e-6F;

static_assert(3u * kCoordBits <= 64u, "80K voxel address must fit in uint64_t");
static_assert(3u * kBrickBits <= 64u, "80K brick address must fit in uint64_t");

struct Vec3u {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
};

inline bool operator==(const Vec3u& a, const Vec3u& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

struct RuntimePolicy {
    std::uint32_t max_active_bricks = 32u;
    std::uint32_t fixed_point_steps = 12u;
    float fixed_point_tolerance = 1.0e-4F;
    float omega_decay = 0.92F;
    float learning_rate = 1.0e-3F;
    float residual_gain = 0.50F;
    float verify_mse_ceiling = 1.0F;
};

struct PyramidLevel {
    std::uint32_t edge{};
    std::vector<float> values;
};

struct CTRReceipt {
    float mse_before{};
    float mse_after{};
    float fixed_point_relative{};
    bool finite{};
    bool non_worsening{};
    bool within_ceiling{};
    bool committed{};
};

struct StepReceipt {
    std::uint64_t cycle{};
    Vec3u voxel{};
    std::uint64_t voxel_address{};
    std::uint64_t brick_key{};
    CTRReceipt ctr{};
};

struct EngineStats {
    std::uint64_t cycles{};
    std::uint64_t commits{};
    std::uint64_t rollbacks{};
    std::size_t active_bricks{};
    std::uint64_t resident_bytes{};
    float last_mse{};
    float last_fixed_point_relative{};
};

struct Brick {
    std::vector<float> signal;
    std::vector<float> reconstruction;
    std::vector<float> residual;
    std::uint64_t touches = 0u;
    float last_error = 0.0F;

    Brick()
        : signal(kBrickVoxels, 0.0F),
          reconstruction(kBrickVoxels, 0.0F),
          residual(kBrickVoxels, 0.0F) {}
};

class SparseBrickStore {
public:
    Brick& fetch_or_allocate(std::uint64_t key, const RuntimePolicy& policy) {
        if (policy.max_active_bricks == 0u) {
            throw std::invalid_argument("max_active_bricks must be > 0");
        }

        auto it = bricks_.find(key);
        if (it != bricks_.end()) {
            ++it->second.touches;
            return it->second;
        }

        if (bricks_.size() >= policy.max_active_bricks) {
            evict();
        }

        auto [inserted, ok] = bricks_.emplace(key, Brick{});
        if (!ok) {
            throw std::runtime_error("failed to allocate sparse 80K brick");
        }
        inserted->second.touches = 1u;
        return inserted->second;
    }

    [[nodiscard]] std::size_t size() const noexcept { return bricks_.size(); }

    [[nodiscard]] std::uint64_t resident_bytes() const noexcept {
        return static_cast<std::uint64_t>(bricks_.size()) *
               static_cast<std::uint64_t>(kBrickVoxels) *
               sizeof(float) * 3ull;
    }

private:
    std::unordered_map<std::uint64_t, Brick> bricks_;

    void evict() {
        if (bricks_.empty()) {
            return;
        }
        auto victim = bricks_.begin();
        for (auto it = bricks_.begin(); it != bricks_.end(); ++it) {
            const double score = static_cast<double>(it->second.touches) +
                                 100.0 * static_cast<double>(it->second.last_error);
            const double victim_score =
                static_cast<double>(victim->second.touches) +
                100.0 * static_cast<double>(victim->second.last_error);
            if (score < victim_score) {
                victim = it;
            }
        }
        bricks_.erase(victim);
    }
};

class Engine {
public:
    explicit Engine(RuntimePolicy policy = {}) : policy_(policy) {
        if (policy_.max_active_bricks == 0u) {
            throw std::invalid_argument("max_active_bricks must be > 0");
        }
        if (!(policy_.residual_gain >= 0.0F && policy_.residual_gain <= 1.0F)) {
            throw std::invalid_argument("residual_gain must be in [0,1]");
        }
        for (std::size_t i = 0; i < kLatentDim; ++i) {
            theta_[i] = 0.12F + 0.025F * static_cast<float>(i);
        }
    }

    static std::uint64_t pack_voxel(const Vec3u& p) {
        if (p.x >= kAxis || p.y >= kAxis || p.z >= kAxis) {
            throw std::out_of_range("80K coordinate out of range");
        }
        return (static_cast<std::uint64_t>(p.x) << (2u * kCoordBits)) |
               (static_cast<std::uint64_t>(p.y) << kCoordBits) |
               static_cast<std::uint64_t>(p.z);
    }

    static Vec3u unpack_voxel(std::uint64_t address) noexcept {
        constexpr std::uint64_t mask = (1ull << kCoordBits) - 1ull;
        return {
            static_cast<std::uint32_t>((address >> (2u * kCoordBits)) & mask),
            static_cast<std::uint32_t>((address >> kCoordBits) & mask),
            static_cast<std::uint32_t>(address & mask)
        };
    }

    static std::uint64_t pack_brick(const Vec3u& b) {
        if (b.x >= kBrickAxis || b.y >= kBrickAxis || b.z >= kBrickAxis) {
            throw std::out_of_range("80K brick coordinate out of range");
        }
        return (static_cast<std::uint64_t>(b.x) << (2u * kBrickBits)) |
               (static_cast<std::uint64_t>(b.y) << kBrickBits) |
               static_cast<std::uint64_t>(b.z);
    }

    static std::vector<PyramidLevel> build_pyramid(const std::vector<float>& input) {
        if (input.size() != kBrickVoxels) {
            throw std::invalid_argument("input must be exactly one 32^3 brick");
        }

        std::vector<PyramidLevel> levels;
        levels.push_back({kBrickEdge, input});
        while (levels.back().edge > 1u) {
            const auto& prev = levels.back();
            const std::uint32_t next_edge = prev.edge / 2u;
            PyramidLevel next{next_edge,
                std::vector<float>(
                    static_cast<std::size_t>(next_edge) * next_edge * next_edge, 0.0F)};

            for (std::uint32_t z = 0; z < next_edge; ++z) {
                for (std::uint32_t y = 0; y < next_edge; ++y) {
                    for (std::uint32_t x = 0; x < next_edge; ++x) {
                        double sum = 0.0;
                        for (std::uint32_t dz = 0; dz < 2u; ++dz) {
                            for (std::uint32_t dy = 0; dy < 2u; ++dy) {
                                for (std::uint32_t dx = 0; dx < 2u; ++dx) {
                                    sum += prev.values[index3d(
                                        prev.edge,
                                        2u * x + dx,
                                        2u * y + dy,
                                        2u * z + dz)];
                                }
                            }
                        }
                        next.values[index3d(next_edge, x, y, z)] =
                            static_cast<float>(sum / 8.0);
                    }
                }
            }
            levels.push_back(std::move(next));
        }
        return levels;
    }

    StepReceipt step(std::uint64_t cycle) {
        const Vec3u voxel = resolve(cycle);
        const Vec3u brick_coord{
            voxel.x / kBrickEdge,
            voxel.y / kBrickEdge,
            voxel.z / kBrickEdge
        };
        const auto brick_key = pack_brick(brick_coord);
        current_ = &store_.fetch_or_allocate(brick_key, policy_);

        materialize(brick_coord, cycle);
        pyramid_ = build_pyramid(current_->signal);
        encode_latent();
        contract_latent();
        const float fp = fixed_point();
        decode();

        const float mse_before = compare();
        correct_residual();
        const float mse_after = compare();

        const bool finite =
            std::isfinite(mse_before) && std::isfinite(mse_after) && std::isfinite(fp);
        const bool non_worsening = mse_after <= mse_before + 1.0e-7F;
        const bool within_ceiling = mse_after <= policy_.verify_mse_ceiling;
        const bool committed = finite && non_worsening && within_ceiling;

        if (committed) {
            commit_adaptation();
            ++stats_.commits;
        } else {
            ++stats_.rollbacks;
        }

        current_->last_error = mse_after;
        ++stats_.cycles;
        stats_.active_bricks = store_.size();
        stats_.resident_bytes = store_.resident_bytes();
        stats_.last_mse = mse_after;
        stats_.last_fixed_point_relative = fp;

        return {
            cycle,
            voxel,
            pack_voxel(voxel),
            brick_key,
            {mse_before, mse_after, fp, finite, non_worsening, within_ceiling, committed}
        };
    }

    [[nodiscard]] const EngineStats& stats() const noexcept { return stats_; }

    [[nodiscard]] std::vector<std::uint8_t> render_rgb(
        std::uint32_t width,
        std::uint32_t height) const {
        if (current_ == nullptr) {
            throw std::runtime_error("render requested before first step");
        }
        if (width == 0u || height == 0u || width > 4096u || height > 4096u) {
            throw std::invalid_argument("render dimensions must be in [1,4096]");
        }

        std::vector<std::uint8_t> rgb(
            static_cast<std::size_t>(width) * height * 3u, 0u);

        for (std::uint32_t py = 0; py < height; ++py) {
            for (std::uint32_t px = 0; px < width; ++px) {
                const std::uint32_t x =
                    std::min(kBrickEdge - 1u,
                             static_cast<std::uint32_t>(
                                 (static_cast<std::uint64_t>(px) * kBrickEdge) / width));
                const std::uint32_t y =
                    std::min(kBrickEdge - 1u,
                             static_cast<std::uint32_t>(
                                 (static_cast<std::uint64_t>(py) * kBrickEdge) / height));

                float accumulator = 0.0F;
                for (std::uint32_t z = 0; z < kBrickEdge; ++z) {
                    accumulator += current_->reconstruction[index3d(kBrickEdge, x, y, z)];
                }
                const float value = std::clamp(
                    0.5F + 0.5F * accumulator / static_cast<float>(kBrickEdge),
                    0.0F,
                    1.0F);
                const auto q = static_cast<std::uint8_t>(std::lround(value * 255.0F));
                const std::size_t out =
                    (static_cast<std::size_t>(py) * width + px) * 3u;
                rgb[out + 0u] = q;
                rgb[out + 1u] = static_cast<std::uint8_t>((q * 3u) / 4u);
                rgb[out + 2u] = static_cast<std::uint8_t>(255u - q / 2u);
            }
        }
        return rgb;
    }

private:
    RuntimePolicy policy_;
    SparseBrickStore store_;
    Brick* current_ = nullptr;
    std::vector<PyramidLevel> pyramid_;
    std::array<float, kLatentDim> latent_{};
    std::array<float, kLatentDim> omega_{};
    std::array<float, kLatentDim> theta_{};
    EngineStats stats_{};

    static std::size_t index3d(
        std::uint32_t edge,
        std::uint32_t x,
        std::uint32_t y,
        std::uint32_t z) {
        return (static_cast<std::size_t>(z) * edge + y) * edge + x;
    }

    static Vec3u resolve(std::uint64_t cycle) noexcept {
        const std::uint64_t a = 6364136223846793005ull;
        const std::uint64_t c = 1442695040888963407ull;
        const std::uint64_t s0 = a * (cycle + 1ull) + c;
        const std::uint64_t s1 = a * (s0 + 0x9E3779B97F4A7C15ull) + c;
        const std::uint64_t s2 = a * (s1 + 0xD1B54A32D192ED03ull) + c;
        return {
            static_cast<std::uint32_t>(s0 % kAxis),
            static_cast<std::uint32_t>(s1 % kAxis),
            static_cast<std::uint32_t>(s2 % kAxis)
        };
    }

    void materialize(const Vec3u& brick_coord, std::uint64_t cycle) {
        const float phase = 0.013F * static_cast<float>(cycle % 10000ull);
        for (std::uint32_t z = 0; z < kBrickEdge; ++z) {
            for (std::uint32_t y = 0; y < kBrickEdge; ++y) {
                for (std::uint32_t x = 0; x < kBrickEdge; ++x) {
                    const std::uint32_t gx = brick_coord.x * kBrickEdge + x;
                    const std::uint32_t gy = brick_coord.y * kBrickEdge + y;
                    const std::uint32_t gz = brick_coord.z * kBrickEdge + z;
                    const std::size_t i = index3d(kBrickEdge, x, y, z);

                    const float wave =
                        std::sin(0.0031F * static_cast<float>(gx) + phase) +
                        std::cos(0.0027F * static_cast<float>(gy) - 0.5F * phase) +
                        std::sin(0.0023F * static_cast<float>(gz) + 0.25F * phase);
                    current_->signal[i] = wave / 3.0F;
                    current_->reconstruction[i] = 0.0F;
                    current_->residual[i] = 0.0F;
                }
            }
        }
    }

    void encode_latent() {
        const auto& coarse = pyramid_[3u];
        double mean = 0.0;
        double energy = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();

        for (float v : coarse.values) {
            mean += v;
            energy += static_cast<double>(v) * v;
            minimum = std::min(minimum, v);
            maximum = std::max(maximum, v);
        }

        const double n = static_cast<double>(coarse.values.size());
        latent_[0] = static_cast<float>(mean / n);
        latent_[1] = static_cast<float>(energy / n);
        latent_[2] = minimum;
        latent_[3] = maximum;
        latent_[4] = pyramid_[1u].values.front();
        latent_[5] = pyramid_[2u].values.back();
        latent_[6] = pyramid_[4u].values.front();
        latent_[7] = pyramid_.back().values.front();
    }

    void contract_latent() {
        std::array<float, kLatentDim> next{};
        for (std::size_t i = 0; i < kLatentDim; ++i) {
            const float l = latent_[(i + kLatentDim - 1u) % kLatentDim];
            const float c = latent_[i];
            const float r = latent_[(i + 1u) % kLatentDim];
            next[i] = 0.2F * l + 0.6F * c + 0.2F * r;
        }
        latent_ = next;
    }

    float fixed_point() {
        float relative = std::numeric_limits<float>::infinity();
        for (std::uint32_t step = 0; step < policy_.fixed_point_steps; ++step) {
            std::array<float, kLatentDim> next{};
            float num = 0.0F;
            float den = 0.0F;
            for (std::size_t i = 0; i < kLatentDim; ++i) {
                const float drive =
                    0.55F * latent_[i] +
                    0.20F * latent_[(i + 1u) % kLatentDim] +
                    0.15F * omega_[i] +
                    0.10F * theta_[i];
                next[i] = std::tanh(drive);
                const float d = next[i] - latent_[i];
                num += d * d;
                den += latent_[i] * latent_[i];
            }
            latent_ = next;
            relative = std::sqrt(num) / (std::sqrt(den) + kEpsilon);
            if (relative < policy_.fixed_point_tolerance) {
                break;
            }
        }
        return relative;
    }

    void decode() {
        const auto& coarse = pyramid_[3u];
        const std::uint32_t scale = kBrickEdge / coarse.edge;
        for (std::uint32_t z = 0; z < kBrickEdge; ++z) {
            for (std::uint32_t y = 0; y < kBrickEdge; ++y) {
                for (std::uint32_t x = 0; x < kBrickEdge; ++x) {
                    const std::size_t i = index3d(kBrickEdge, x, y, z);
                    const float base = coarse.values[index3d(
                        coarse.edge, x / scale, y / scale, z / scale)];
                    const float u = static_cast<float>(i) /
                                    static_cast<float>(kBrickVoxels);
                    const float harmonic =
                        latent_[1] * std::sin(6.28318530718F * u + latent_[0]);
                    current_->reconstruction[i] = 0.90F * base + 0.10F * harmonic;
                }
            }
        }
    }

    float compare() {
        double mse = 0.0;
        for (std::uint32_t i = 0; i < kBrickVoxels; ++i) {
            const float e = current_->signal[i] - current_->reconstruction[i];
            current_->residual[i] = e;
            mse += static_cast<double>(e) * e;
        }
        return static_cast<float>(mse / static_cast<double>(kBrickVoxels));
    }

    void correct_residual() {
        for (std::uint32_t i = 0; i < kBrickVoxels; ++i) {
            current_->reconstruction[i] += policy_.residual_gain * current_->residual[i];
        }
    }

    void commit_adaptation() {
        const float rho = policy_.omega_decay;
        for (std::size_t i = 0; i < kLatentDim; ++i) {
            const float omega_candidate =
                rho * omega_[i] + (1.0F - rho) * latent_[i];
            const float gradient = stats_.last_mse * latent_[i];
            const float theta_candidate =
                std::clamp(theta_[i] - policy_.learning_rate * gradient, -2.0F, 2.0F);
            omega_[i] = omega_candidate;
            theta_[i] = theta_candidate;
        }
    }
};

} // namespace jarvisx::volumetric_80k
