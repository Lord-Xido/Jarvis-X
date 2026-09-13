#include "jarvisx/volumetric_rom_ann.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace jarvisx::volumetric_rom;

    const Vec3u probe{1u, 54321u, kAxisPositions - 1u};
    const auto packed = Engine::pack_voxel(probe);
    assert(Engine::unpack_voxel(packed) == probe);

    std::vector<float> constant(kTileVoxels, 0.25F);
    const auto pyramid = Engine::build_pyramid(constant);
    assert(pyramid.size() == 6u);
    assert(pyramid.front().edge == 32u);
    assert(pyramid.back().edge == 1u);
    assert(pyramid.back().values.size() == 1u);
    assert(std::fabs(pyramid.back().values.front() - 0.25F) < 1.0e-6F);

    RuntimePolicy policy{};
    policy.max_active_tiles = 8u;
    Engine engine(policy);
    const auto stats = engine.run(4u, true);

    assert(stats.cycles == 4u);
    assert(stats.active_tiles > 0u);
    assert(stats.active_tiles <= policy.max_active_tiles);
    assert(stats.physical_bytes > 0u);
    assert(stats.physical_bytes < (16ull * 1024ull * 1024ull));
    assert(stats.pyramid_levels == 6u);
    assert(std::isfinite(stats.last_error));
    assert(stats.last_error >= 0.0F);
    assert(std::isfinite(stats.last_fixed_point_relative));
    assert(stats.fixed_point_steps >= 4u && stats.fixed_point_steps <= 24u);
    assert(stats.sparse_residual_voxels <= kTileVoxels);

    std::cout << "volumetric ROM ANN regressions passed\n";
    return 0;
}
