#include "jarvisx/dr_moagi_state_space.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void test_spsc_ring() {
    jarvisx::SpscIngressRingBuffer<int, 4U> ring;
    assert(ring.push(1));
    assert(ring.push(2));
    assert(ring.push(3));
    assert(!ring.push(4));
    int value = 0;
    assert(ring.pop(value) && value == 1);
    assert(ring.pop(value) && value == 2);
    assert(ring.pop(value) && value == 3);
    assert(!ring.pop(value));
}

void test_linear_bound() {
    std::vector<double> matrix{
        1.2, 0.2,
        0.1, 1.1};
    const double scale = jarvisx::LinearStateGovernor::enforce_contractive_bound(matrix, 2U, 0.94);
    assert(scale < 1.0);
    const double norm = jarvisx::LinearStateGovernor::infinity_norm(matrix, 2U);
    assert(norm <= 0.9400000001);
}

void test_engine_dimensions_and_tiles() {
    jarvisx::DrMoagiStateSpaceConfig config;
    config.latent_dim = 10U;
    config.ingress_dim = 4U;
    config.tiles_x = 3U;
    config.tiles_y = 2U;
    config.logical_width = 10U;
    config.logical_height = 8U;

    jarvisx::DrMoagiStateSpaceEngine engine(config);
    jarvisx::MultimodalIngress ingress;
    ingress.spatial_3d_mesh = {1.0, 2.0, 3.0, 4.0};
    const auto u = engine.encode_ingress(ingress);
    std::vector<double> context(config.latent_dim, 0.0);
    engine.step_state_space(u, context);

    const auto tiles = engine.decode_and_dispatch_tiles();
    assert(tiles.size() == 6U);
    std::size_t latent_count = 0U;
    for (const auto& tile : tiles) {
        latent_count += tile.latent_slice.size();
    }
    assert(latent_count == config.latent_dim);
    assert(engine.transition_infinity_norm() < 1.0);
    assert(std::isfinite(engine.latent_norm()));
}

void test_dimension_rejection() {
    jarvisx::DrMoagiStateSpaceConfig config;
    config.latent_dim = 8U;
    config.ingress_dim = 4U;
    jarvisx::DrMoagiStateSpaceEngine engine(config);
    bool threw = false;
    try {
        engine.step_state_space(std::vector<double>(3U, 0.0),
                                std::vector<double>(8U, 0.0));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main() {
    test_spsc_ring();
    test_linear_bound();
    test_engine_dimensions_and_tiles();
    test_dimension_rejection();
    std::cout << "dr_moagi_state_space_tests: PASS\n";
    return 0;
}
