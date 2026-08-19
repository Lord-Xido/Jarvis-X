#include "jarvisx/dr_moagi_state_space.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using namespace jarvisx;

    DrMoagiStateSpaceConfig config;
    SpscIngressRingBuffer<MultimodalIngress, 16U> ingress_ring;
    DrMoagiStateSpaceEngine engine(config);

    MultimodalIngress payload;
    payload.timestamp_ns = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    payload.spatial_3d_mesh = {0.8, 0.4, -0.2, 0.9, 0.1};
    payload.spatial_audio = {0.1, 0.85, 0.44};
    payload.contextual_text = {1.0, 0.0, 0.5, 0.25};

    if (!ingress_ring.push(payload)) {
        std::cerr << "failed to enqueue synthetic ingress\n";
        return 1;
    }

    std::cout << "=========================================================\n"
              << "  DR MOAGI STATE-SPACE C++ REFERENCE CORE               \n"
              << "=========================================================\n\n"
              << std::fixed << std::setprecision(4);

    for (int frame = 1; frame <= 5; ++frame) {
        const auto start = std::chrono::high_resolution_clock::now();

        MultimodalIngress active_ingress;
        std::vector<double> u(config.ingress_dim, 0.05);
        if (ingress_ring.pop(active_ingress)) {
            u = engine.encode_ingress(active_ingress);
        }

        std::vector<double> delta_context(config.latent_dim, 0.005);
        engine.step_state_space(u, delta_context);
        const auto tiles = engine.decode_and_dispatch_tiles();

        double total_power_mw = 0.0;
        for (const auto& tile : tiles) {
            total_power_mw += tile.active_power_mw;
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "[FRAME " << frame << "]"
                  << " LatentNorm=" << engine.latent_norm()
                  << " | ||Phi||_inf=" << engine.transition_infinity_norm()
                  << " | growth_est(Phi)=" << engine.transition_growth_estimate()
                  << " | tile_power_model=" << total_power_mw << " mW"
                  << " | latency=" << frame_ms << " ms\n";
    }

    std::cout << "\n[STATE] Linear transition bound enforced. "
              << "Full nonlinear closed-loop Jacobian stability is not asserted by this harness.\n";
    return 0;
}
