#include "jarvisx/inward_meta_q16.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    using namespace jarvisx::dmimte;

    require(WORLD_SIZE == 262144, "world size must be 64^3");
    require(LATENT_SIZE == 4096, "latent size must be 16^3");
    require(CORE_SIZE == 64, "core size must be 4^3");
    require(q16_mul(Q_ONE, Q_HALF) == Q_HALF, "Q16 multiply identity failed");
    require(
        q16_add(std::numeric_limits<std::int32_t>::max(), 1)
            == std::numeric_limits<std::int32_t>::max(),
        "Q16 addition must saturate");

    // Exact WW^T round-trip for coefficients divisible by 8 Q16 units.
    std::vector<std::int32_t> latent(static_cast<std::size_t>(LATENT_SIZE), 0);
    for (int i = 0; i < LATENT_SIZE; ++i) {
        latent[static_cast<std::size_t>(i)] = (i % 9) * 8 * 1024;
    }
    std::vector<std::int32_t> world;
    std::vector<std::int32_t> roundtrip;
    HierarchicalBlockEncQ16::applyT(latent, world);
    HierarchicalBlockEncQ16::apply(world, roundtrip);
    require(roundtrip == latent, "W W^T must preserve compatible Q16 coefficients");

    EngineConfig deterministic;
    deterministic.enable_noise = false;

    InwardMetaEngineQ16 first(deterministic);
    InwardMetaEngineQ16 second(deterministic);

    const double initial = first.rmse();
    require(
        initial > first.projection_floor_rmse(),
        "zero latent must begin above the representation floor");

    double previous = initial;
    for (int i = 0; i < 15; ++i) {
        const auto a = first.step();
        const auto b = second.step();

        require(std::isfinite(a.rmse), "RMSE must remain finite");
        require(
            a.rmse <= previous + 1.0e-9,
            "noise-free reconstruction RMSE must be non-increasing");
        require(
            std::abs(a.rmse - b.rmse) < 1.0e-12,
            "noise-free runs must be deterministic");

        previous = a.rmse;
    }

    require(
        previous - first.projection_floor_rmse() < 1.0e-6,
        "15-step solver must approach the block projection floor");

    for (const auto value : first.latent()) {
        require(
            value != std::numeric_limits<std::int32_t>::min(),
            "latent state must not wrap to INT32_MIN");
    }

    std::cout << "inward-meta-q16 tests passed\n";
    return 0;
}
