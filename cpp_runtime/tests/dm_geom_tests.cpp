#include "jarvisx/dm_geom.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_mesh_is_valid_and_non_degenerate() {
    using namespace jarvisx::dmgeom;
    const Mesh mesh = make_uv_sphere(2.0F, 16);
    std::string error;
    require(validate_mesh(mesh, &error), "generated sphere is invalid: " + error);
    require(mesh.vertices.size() == 482U, "unexpected unique-pole vertex count");
    require(mesh.triangles.size() == 960U, "unexpected triangle count");

    for (const Triangle& triangle : mesh.triangles) {
        const Vec3 normal = (mesh.vertices[triangle.i1] - mesh.vertices[triangle.i0])
                                .cross(mesh.vertices[triangle.i2] - mesh.vertices[triangle.i0]);
        require(normal.length() > 1.0e-8F, "generated sphere contains a degenerate triangle");
    }
}

void test_encoder_uses_nine_finite_features() {
    using namespace jarvisx::dmgeom;
    const FeatureVector features = Encoder::extract_features(make_uv_sphere(2.0F, 16));
    for (float value : features) {
        require(std::isfinite(value), "feature vector contains a non-finite value");
    }
    require(features[3] > 1.9F && features[3] < 2.1F, "mean-radius feature is incorrect");
    require(features[5] > 3.9F && features[6] > 3.9F && features[7] > 3.9F,
            "bounding-box features were not populated");
}

void test_evolution_reduces_continuous_loss() {
    using namespace jarvisx::dmgeom;
    constexpr int segments = 16;
    const Mesh target = make_uv_sphere(2.0F, segments);
    const Latent3D encoded = Encoder::encode(target);
    const Latent3D start{encoded.x + 0.30F, encoded.y + 0.35F, encoded.z + 0.25F};
    const float before = reconstruction_loss(target, Decoder::decode(start, segments)).total;

    EvolutionConfig config{};
    config.steps = 100;
    config.segments = segments;
    config.learning_rate = 0.08F;
    config.verbose = false;
    const Latent3D optimized = evolve(target, start, config);
    const float after = reconstruction_loss(target, Decoder::decode(optimized, segments)).total;

    require(after < before * 0.10F, "evolution did not materially reduce reconstruction loss");
}

void test_singularity_is_removed() {
    using namespace jarvisx::dmgeom;
    const Mesh mesh = Decoder::decode({-2.0F, 0.0F, 0.0F}, 8);
    std::string error;
    require(validate_mesh(mesh, &error), "latent x=-2 produced an invalid mesh: " + error);
    require(mean_radius(mesh) > 0.0F && std::isfinite(mean_radius(mesh)),
            "latent x=-2 produced a singular radius");
}

}  // namespace

int main() {
    try {
        test_mesh_is_valid_and_non_degenerate();
        test_encoder_uses_nine_finite_features();
        test_evolution_reduces_continuous_loss();
        test_singularity_is_removed();
        std::cout << "dm-geom regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dm-geom regression failure: " << error.what() << '\n';
        return 1;
    }
}
