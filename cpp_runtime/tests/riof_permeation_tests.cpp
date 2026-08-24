#include "jarvisx/riof_permeation.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

jarvisx::RiofPermeationConfig config_for(const jarvisx::Tensor4D& latent,
                                         std::size_t cycles = 4U) {
    jarvisx::RiofPermeationConfig config;
    config.cycles = cycles;
    config.dynamics.edge = latent.shape().width;
    config.dynamics.seed = 42U;
    config.dynamics.projection_limit = 1.25;
    return config;
}

void deterministic_replay() {
    jarvisx::Autoencoder3D model({8U, 3U, 0.02F, 1.0e-4F, 1.0F, 42U});
    const auto input = jarvisx::make_volume(8U, "sphere", 42U);
    const auto latent = model.encode(input);
    const auto config = config_for(latent, 6U);

    const auto first = jarvisx::RiofLatentPermeator(config).permeate(latent);
    const auto second = jarvisx::RiofLatentPermeator(config).permeate(latent);

    require(first.latent.values() == second.latent.values(),
            "same latent/config must replay exactly");
    require(first.metrics.mean_abs_delta == second.metrics.mean_abs_delta,
            "permeation metrics must replay exactly");
}

void zero_cycles_are_identity() {
    jarvisx::Autoencoder3D model({8U, 2U, 0.02F, 0.0F, 1.0F, 9U});
    const auto input = jarvisx::make_volume(8U, "wave", 9U);
    const auto latent = model.encode(input);
    const auto result = jarvisx::RiofLatentPermeator(
        config_for(latent, 0U)).permeate(latent);

    require(result.latent.values() == latent.values(),
            "zero-cycle permeation must be an exact identity");
    require(result.metrics.mean_abs_delta == 0.0,
            "zero-cycle permeation delta must be zero");
}

void projection_and_shape_are_preserved() {
    jarvisx::Autoencoder3D model({8U, 4U, 0.02F, 1.0e-4F, 1.0F, 17U});
    const auto input = jarvisx::make_volume(8U, "shell", 17U);
    const auto latent = model.encode(input);
    const auto config = config_for(latent, 8U);
    const auto result = jarvisx::RiofLatentPermeator(config).permeate(latent);

    require(result.latent.shape().channels == latent.shape().channels,
            "permeation must preserve channel count");
    require(result.latent.shape().depth == latent.shape().depth &&
            result.latent.shape().height == latent.shape().height &&
            result.latent.shape().width == latent.shape().width,
            "permeation must preserve spatial shape");

    for (float value : result.latent.values()) {
        require(std::isfinite(value), "permeated latent must remain finite");
        require(std::abs(static_cast<double>(value)) <=
                    config.dynamics.projection_limit + 1.0e-6,
                "Lambda projection must bound the latent field");
    }
    require(result.metrics.mean_abs_delta > 0.0,
            "non-zero permeation cycles must evolve the latent field");
}

void autoencoder_bridge_decodes() {
    jarvisx::Autoencoder3D model({8U, 3U, 0.02F, 1.0e-4F, 1.0F, 123U});
    const auto input = jarvisx::make_volume(8U, "sphere", 123U);
    jarvisx::RiofPermeationConfig config;
    config.cycles = 3U;
    config.dynamics.seed = 123U;

    const auto result = jarvisx::reconstruct_with_riof(model, input, config);

    require(result.reconstruction.shape().channels == input.shape().channels &&
            result.reconstruction.shape().depth == input.shape().depth &&
            result.reconstruction.shape().height == input.shape().height &&
            result.reconstruction.shape().width == input.shape().width,
            "RIOF bridge reconstruction must match input shape");
    require(std::isfinite(result.metrics.final_mean_abs_residual),
            "RIOF bridge telemetry must remain finite");
}

} // namespace

int main() {
    try {
        deterministic_replay();
        zero_cycles_are_identity();
        projection_and_shape_are_preserved();
        autoencoder_bridge_decodes();
        std::cout << "RIOF latent permeation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RIOF latent permeation test failure: " << error.what() << '\n';
        return 1;
    }
}
