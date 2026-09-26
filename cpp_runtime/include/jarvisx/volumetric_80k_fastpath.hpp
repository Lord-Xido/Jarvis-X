#pragma once

#include "jarvisx/volumetric_80k_mp4.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jarvisx::volumetric_80k_fast {

using jarvisx::volumetric_80k::Engine;
using jarvisx::volumetric_80k::EngineStats;
using jarvisx::volumetric_80k::RuntimePolicy;
using jarvisx::volumetric_80k::Vec3u;
using jarvisx::volumetric_80k::kAxis;
using jarvisx::volumetric_80k::kBrickAxis;
using jarvisx::volumetric_80k::kBrickEdge;

struct FastPolicy {
    std::uint32_t candidates_per_tick = 64u;
    float selection_ratio = 0.10F;
    std::uint32_t temporal_cache_window = 2u;
    float temporal_phase_epsilon = 0.020F;
    float cache_mse_ceiling = 0.25F;
    float error_weight = 1.0F;
    float temporal_weight = 0.25F;
    std::uint32_t max_cached_bricks = 512u;
};

struct TickReceipt {
    std::uint64_t tick{};
    std::uint32_t candidates{};
    std::uint32_t selected{};
    std::uint32_t cache_hits{};
    std::uint32_t processed{};
    float mean_mse{};
    double work_ratio{};
};

struct FastStats {
    std::uint64_t ticks{};
    std::uint64_t candidates_seen{};
    std::uint64_t selected{};
    std::uint64_t cache_hits{};
    std::uint64_t processed{};
    double accumulated_mse{};
};

struct Candidate {
    Vec3u voxel{};
    std::uint64_t brick_key{};
    float priority{};
};

class Scheduler {
public:
    explicit Scheduler(
        FastPolicy fast_policy = {},
        RuntimePolicy runtime_policy = {})
        : fast_policy_(fast_policy), engine_(runtime_policy) {
        validate();
    }

    [[nodiscard]] std::vector<Vec3u> candidate_voxels(std::uint64_t tick) const {
        std::vector<Vec3u> out;
        out.reserve(fast_policy_.candidates_per_tick);

        // Hold the anchor for four ticks so adjacent frames exhibit temporal
        // locality. This makes cache behavior measurable rather than accidental.
        const std::uint64_t epoch = tick / 4ull;
        const std::uint32_t ax =
            static_cast<std::uint32_t>((epoch * 7919ull + 17ull) % kBrickAxis);
        const std::uint32_t ay =
            static_cast<std::uint32_t>((epoch * 1543ull + 29ull) % kBrickAxis);
        const std::uint32_t az =
            static_cast<std::uint32_t>((epoch * 3571ull + 43ull) % kBrickAxis);

        for (std::uint32_t i = 0; i < fast_policy_.candidates_per_tick; ++i) {
            const std::uint32_t bx = (ax + (17u * i) % 53u) % kBrickAxis;
            const std::uint32_t by = (ay + (29u * i) % 47u) % kBrickAxis;
            const std::uint32_t bz = (az + (37u * i) % 43u) % kBrickAxis;

            out.push_back({
                std::min(kAxis - 1u, bx * kBrickEdge + kBrickEdge / 2u),
                std::min(kAxis - 1u, by * kBrickEdge + kBrickEdge / 2u),
                std::min(kAxis - 1u, bz * kBrickEdge + kBrickEdge / 2u)
            });
        }
        return out;
    }

    TickReceipt tick(std::uint64_t tick_index) {
        auto candidates = rank_candidates(tick_index);
        const std::uint32_t selected_count = selection_count(candidates.size());

        std::uint32_t cache_hits = 0u;
        std::uint32_t processed = 0u;
        double mse_sum = 0.0;

        for (std::uint32_t i = 0; i < selected_count; ++i) {
            const Candidate& candidate = candidates[i];
            auto cache_it = cache_.find(candidate.brick_key);

            if (cache_it != cache_.end() &&
                coherent(cache_it->second, tick_index) &&
                engine_.activate_resident_brick(candidate.brick_key)) {
                ++cache_hits;
                ++cache_it->second.touches;
                continue;
            }

            const auto receipt = engine_.step_at(tick_index, candidate.voxel);
            ++processed;
            mse_sum += static_cast<double>(receipt.ctr.mse_after);

            CacheEntry& entry = cache_[candidate.brick_key];
            entry.last_tick = tick_index;
            entry.last_mse = receipt.ctr.mse_after;
            ++entry.touches;
            trim_cache();
        }

        ++stats_.ticks;
        stats_.candidates_seen += candidates.size();
        stats_.selected += selected_count;
        stats_.cache_hits += cache_hits;
        stats_.processed += processed;
        stats_.accumulated_mse += mse_sum;

        const float mean_mse =
            processed == 0u ? 0.0F
                            : static_cast<float>(mse_sum / static_cast<double>(processed));
        const double work_ratio =
            candidates.empty()
                ? 0.0
                : static_cast<double>(processed) /
                      static_cast<double>(candidates.size());

        return {
            tick_index,
            static_cast<std::uint32_t>(candidates.size()),
            selected_count,
            cache_hits,
            processed,
            mean_mse,
            work_ratio
        };
    }

    [[nodiscard]] const FastStats& stats() const noexcept { return stats_; }
    [[nodiscard]] const EngineStats& engine_stats() const noexcept {
        return engine_.stats();
    }

    [[nodiscard]] std::vector<std::uint8_t> render_rgb(
        std::uint32_t width,
        std::uint32_t height) const {
        return engine_.render_rgb(width, height);
    }

private:
    struct CacheEntry {
        std::uint64_t last_tick{};
        float last_mse{};
        std::uint64_t touches{};
    };

    FastPolicy fast_policy_;
    Engine engine_;
    std::unordered_map<std::uint64_t, CacheEntry> cache_;
    FastStats stats_{};

    void validate() const {
        if (fast_policy_.candidates_per_tick == 0u ||
            fast_policy_.candidates_per_tick > 4096u) {
            throw std::invalid_argument(
                "candidates_per_tick must be in [1,4096]");
        }
        if (!(fast_policy_.selection_ratio > 0.0F &&
              fast_policy_.selection_ratio <= 1.0F)) {
            throw std::invalid_argument("selection_ratio must be in (0,1]");
        }
        if (fast_policy_.max_cached_bricks == 0u) {
            throw std::invalid_argument("max_cached_bricks must be > 0");
        }
        if (fast_policy_.temporal_phase_epsilon < 0.0F) {
            throw std::invalid_argument("temporal_phase_epsilon must be >= 0");
        }
    }

    static std::uint64_t brick_key_for(const Vec3u& voxel) {
        const Vec3u brick{
            voxel.x / kBrickEdge,
            voxel.y / kBrickEdge,
            voxel.z / kBrickEdge
        };
        return Engine::pack_brick(brick);
    }

    float priority_for(
        const Vec3u& voxel,
        std::uint64_t key,
        std::uint64_t tick_index) const {
        const float bx = static_cast<float>(voxel.x / kBrickEdge);
        const float by = static_cast<float>(voxel.y / kBrickEdge);
        const float bz = static_cast<float>(voxel.z / kBrickEdge);

        const float gradient_proxy =
            std::fabs(std::sin(0.021F * (bx + 1.0F))) +
            std::fabs(std::cos(0.017F * (by + 1.0F))) +
            std::fabs(std::sin(0.013F * (bz + 1.0F)));

        float error_proxy = 0.0F;
        float temporal_proxy = 1.0F;

        const auto it = cache_.find(key);
        if (it != cache_.end()) {
            error_proxy = it->second.last_mse;
            const std::uint64_t dt =
                tick_index >= it->second.last_tick
                    ? tick_index - it->second.last_tick
                    : 0ull;
            temporal_proxy = std::min(
                1.0F,
                0.013F * static_cast<float>(dt) /
                    std::max(fast_policy_.temporal_phase_epsilon, 1.0e-6F));
        }

        return gradient_proxy +
               fast_policy_.error_weight * error_proxy +
               fast_policy_.temporal_weight * temporal_proxy;
    }

    std::vector<Candidate> rank_candidates(std::uint64_t tick_index) const {
        const auto voxels = candidate_voxels(tick_index);
        std::vector<Candidate> ranked;
        ranked.reserve(voxels.size());

        for (const auto& voxel : voxels) {
            const auto key = brick_key_for(voxel);
            ranked.push_back({
                voxel,
                key,
                priority_for(voxel, key, tick_index)
            });
        }

        std::sort(
            ranked.begin(),
            ranked.end(),
            [](const Candidate& a, const Candidate& b) {
                if (a.priority != b.priority) {
                    return a.priority > b.priority;
                }
                return a.brick_key < b.brick_key;
            });

        return ranked;
    }

    std::uint32_t selection_count(std::size_t candidate_count) const {
        if (candidate_count == 0u) {
            return 0u;
        }
        const auto raw = static_cast<std::uint32_t>(
            std::ceil(
                static_cast<double>(candidate_count) *
                static_cast<double>(fast_policy_.selection_ratio)));
        return std::max<std::uint32_t>(
            1u,
            std::min<std::uint32_t>(
                static_cast<std::uint32_t>(candidate_count),
                raw));
    }

    bool coherent(
        const CacheEntry& entry,
        std::uint64_t tick_index) const noexcept {
        if (tick_index <= entry.last_tick) {
            return false;
        }
        const std::uint64_t dt = tick_index - entry.last_tick;
        if (dt > fast_policy_.temporal_cache_window) {
            return false;
        }

        const float phase_delta = 0.013F * static_cast<float>(dt);
        return phase_delta <= fast_policy_.temporal_phase_epsilon &&
               entry.last_mse <= fast_policy_.cache_mse_ceiling;
    }

    void trim_cache() {
        while (cache_.size() > fast_policy_.max_cached_bricks) {
            auto victim = cache_.begin();
            for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                if (it->second.last_tick < victim->second.last_tick ||
                    (it->second.last_tick == victim->second.last_tick &&
                     it->second.touches < victim->second.touches)) {
                    victim = it;
                }
            }
            cache_.erase(victim);
        }
    }
};

} // namespace jarvisx::volumetric_80k_fast
