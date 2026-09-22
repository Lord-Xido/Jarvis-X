#include "jarvisx/volumetric_80k_mp4.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    using namespace jarvisx::volumetric_80k;

    static_assert(kLogicalVoxels == 512000000000000ull);
    static_assert(kBrickAxis == 2500u);

    const Vec3u p{79999u, 54321u, 12345u};
    const auto packed = Engine::pack_voxel(p);
    assert(Engine::unpack_voxel(packed) == p);

    std::vector<float> constant(kBrickVoxels, 0.25F);
    const auto pyramid = Engine::build_pyramid(constant);
    assert(pyramid.size() == 6u);
    assert(pyramid.front().edge == 32u);
    assert(pyramid.back().edge == 1u);
    assert(std::fabs(pyramid.back().values.front() - 0.25F) < 1.0e-6F);

    RuntimePolicy policy{};
    policy.max_active_bricks = 4u;
    Engine engine(policy);

    for (std::uint32_t i = 0; i < 6u; ++i) {
        const auto receipt = engine.step(i);
        assert(receipt.voxel.x < kAxis);
        assert(receipt.voxel.y < kAxis);
        assert(receipt.voxel.z < kAxis);
        assert(std::isfinite(receipt.ctr.mse_before));
        assert(std::isfinite(receipt.ctr.mse_after));
        assert(receipt.ctr.mse_after <= receipt.ctr.mse_before + 1.0e-6F);
    }

    const auto& stats = engine.stats();
    assert(stats.cycles == 6u);
    assert(stats.active_bricks > 0u);
    assert(stats.active_bricks <= policy.max_active_bricks);
    assert(stats.resident_bytes > 0u);
    assert(stats.commits + stats.rollbacks == stats.cycles);

    const auto frame = engine.render_rgb(64u, 48u);
    assert(frame.size() == 64u * 48u * 3u);

    std::cout << "80K^3 volumetric AE/AD regressions passed\n";
    return 0;
}
