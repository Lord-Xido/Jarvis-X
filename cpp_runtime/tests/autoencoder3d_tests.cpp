#include "jarvisx/autoencoder3d.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void deterministic_initialization() {
    const jarvisx::Autoencoder3DConfig config{8U, 3U, 0.02F, 1.0e-4F, 1.0F, 42U};
    jarvisx::Autoencoder3D first(config);
    jarvisx::Autoencoder3D second(config);
    const auto input = jarvisx::make_volume(8U, "sphere", 42U);
    const auto a = first.reconstruct(input);
    const auto b = second.reconstruct(input);
    require(a.values() == b.values(), "same seed must produce identical output");
}

void training_reduces_error() {
    jarvisx::Autoencoder3D model({8U, 4U, 0.03F, 1.0e-4F, 1.0F, 7U});
    const auto input = jarvisx::make_volume(8U, "sphere", 7U);
    const auto before_latent = model.encode(input);
    const auto before_output = model.decode(before_latent);
    const float before = jarvisx::measure_reconstruction(
        input, before_latent, before_output, 0U).mse;
    for (std::size_t step = 0; step < 180U; ++step) {
        const auto metrics = model.train_step(input);
        require(std::isfinite(metrics.mse), "training MSE must stay finite");
    }
    const auto after_latent = model.encode(input);
    const auto after_output = model.decode(after_latent);
    const float after = jarvisx::measure_reconstruction(
        input, after_latent, after_output, model.steps()).mse;
    require(after < before * 0.85F, "training must materially reduce reconstruction error");
}

void q3_latent_is_bounded() {
    jarvisx::Autoencoder3D model({8U, 2U, 0.02F, 0.0F, 1.0F, 99U});
    const auto input = jarvisx::make_volume(8U, "wave", 99U);
    const auto latent = model.encode(input, true);
    for (const float value : latent.values()) {
        require(value >= -1.0F && value <= 1.0F, "Q3 latent must remain bounded");
        const std::int8_t q = jarvisx::quantize_q3(value);
        require(std::fabs(value - jarvisx::dequantize_q3(q)) < 1.0e-6F,
                "quantized latent must lie on Q3 reconstruction levels");
    }
}

void model_round_trip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "jarvisx-autoencoder3d-test.jx3d";
    jarvisx::Autoencoder3D model({8U, 3U, 0.02F, 1.0e-4F, 1.0F, 123U});
    const auto input = jarvisx::make_volume(8U, "shell", 123U);
    for (std::size_t step = 0; step < 5U; ++step) model.train_step(input);
    const auto expected = model.reconstruct(input);
    model.save(path);
    jarvisx::Autoencoder3D loaded = jarvisx::Autoencoder3D::load(path);
    const auto actual = loaded.reconstruct(input);
    std::filesystem::remove(path);
    require(expected.values() == actual.values(), "saved model must replay exactly");
    require(loaded.steps() == model.steps(), "saved step count must round-trip");
}

} // namespace

int main() {
    try {
        deterministic_initialization();
        training_reduces_error();
        q3_latent_is_bounded();
        model_round_trip();
        std::cout << "3D autoencoder tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D autoencoder test failure: " << error.what() << '\n';
        return 1;
    }
}
