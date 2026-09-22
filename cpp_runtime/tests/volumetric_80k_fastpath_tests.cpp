#include "jarvisx/volumetric_80k_fastpath.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
    using namespace jarvisx::volumetric_80k;
    using namespace jarvisx::volumetric_80k_fast;

    RuntimePolicy runtime{};
    runtime.max_active_bricks = 16u;

    Engine direct(runtime);
    const Vec3u p{160u, 320u, 480u};
    const auto explicit_receipt = direct.step_at(7u, p);
    assert(explicit_receipt.voxel == p);
    assert(std::isfinite(explicit_receipt.ctr.mse_after));

    FastPolicy sparse{};
    sparse.candidates_per_tick = 20u;
    sparse.selection_ratio = 0.10F;
    sparse.temporal_phase_epsilon = 0.0F;
    Scheduler sparse_scheduler(sparse, runtime);
    const auto sparse_tick = sparse_scheduler.tick(0u);
    assert(sparse_tick.candidates == 20u);
    assert(sparse_tick.selected == 2u);
    assert(sparse_tick.processed == 2u);
    assert(sparse_tick.work_ratio <= 0.10 + 1.0e-9);

    FastPolicy cached{};
    cached.candidates_per_tick = 4u;
    cached.selection_ratio = 1.0F;
    cached.temporal_cache_window = 2u;
    cached.temporal_phase_epsilon = 0.020F;
    cached.cache_mse_ceiling = 1.0F;
    Scheduler cache_scheduler(cached, runtime);

    const auto first = cache_scheduler.tick(0u);
    const auto second = cache_scheduler.tick(1u);
    assert(first.processed == 4u);
    assert(second.selected == 4u);
    assert(second.cache_hits == 4u);
    assert(second.processed == 0u);

    const auto& stats = cache_scheduler.stats();
    assert(stats.ticks == 2u);
    assert(stats.candidates_seen == 8u);
    assert(stats.cache_hits == 4u);

    std::cout << "80K^3 fast-path regressions passed\n";
    return 0;
}
