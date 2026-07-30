#include "jarvisx/dm_geom.hpp"

#include <chrono>
#include <exception>
#include <iostream>

int main() {
    using namespace jarvisx::dmgeom;

    try {
        std::cout << "=== DM-GEOM v0.3 PERMEATED BOOT ===\n";
        const auto started = std::chrono::steady_clock::now();

        constexpr int segments = 32;
        const Mesh input = make_uv_sphere(2.0F, segments);
        std::cout << "[1] INPUT: " << input.vertices.size() << " verts, "
                  << input.triangles.size() << " tris\n";

        const Latent3D encoded = Encoder::encode(input);
        std::cout << "[2] ENCODED: z=(" << encoded.x << ',' << encoded.y << ',' << encoded.z << ")\n";

        const Latent3D disturbed{encoded.x + 0.35F, encoded.y + 0.40F, encoded.z + 0.20F};
        const Mesh reconstruction = Decoder::decode(disturbed, segments);
        export_binary_stl(reconstruction, "recon.stl");
        const float initial_loss = reconstruction_loss(input, reconstruction).total;
        std::cout << "[3] DECODED: " << reconstruction.vertices.size()
                  << " verts -> recon.stl, loss=" << initial_loss << '\n';

        EvolutionConfig config{};
        config.steps = 120;
        config.segments = segments;
        config.learning_rate = 0.08F;
        std::cout << "[4] EVOLVING...\n";
        const Latent3D optimized = evolve(input, disturbed, config);

        const Mesh output = Decoder::decode(optimized, segments);
        export_binary_stl(output, "optimized.stl");
        const float final_loss = reconstruction_loss(input, output).total;
        std::cout << "[5] OPTIMIZED: " << output.vertices.size()
                  << " verts -> optimized.stl, loss=" << final_loss << '\n';

        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
        std::cout << "TOTAL TIME: " << elapsed.count() << "s\n";

        if (!(final_loss < initial_loss)) {
            std::cerr << "Evolution failed to improve reconstruction loss\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DM-GEOM fatal error: " << error.what() << '\n';
        return 1;
    }
}
