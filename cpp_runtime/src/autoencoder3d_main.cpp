#include "jarvisx/autoencoder3d.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

struct Options {
    std::size_t edge{8U};
    std::size_t channels{4U};
    std::uint64_t epochs{250U};
    float learning_rate{0.03F};
    float l2_penalty{1.0e-4F};
    float gradient_clip{1.0F};
    std::uint64_t seed{0x4A415256495358ULL};
    std::string pattern{"sphere"};
    std::filesystem::path export_dir{".jarvisx-autoencoder3d"};
    std::filesystem::path save_model;
    std::filesystem::path load_model;
    bool quantized{};
    bool quiet{};
};

std::uint64_t parse_u64(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const std::uint64_t parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid integer after " + flag);
    }
    return parsed;
}

float parse_float(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("invalid floating-point value after " + flag);
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    auto value = [&](int& index, const std::string& flag) -> std::string {
        if (index + 1 >= argc) throw std::invalid_argument("missing value after " + flag);
        ++index;
        return argv[index];
    };

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--edge") {
            options.edge = static_cast<std::size_t>(parse_u64(value(index, argument), argument));
        } else if (argument == "--channels") {
            options.channels = static_cast<std::size_t>(parse_u64(value(index, argument), argument));
        } else if (argument == "--epochs") {
            options.epochs = std::max<std::uint64_t>(1U, parse_u64(value(index, argument), argument));
        } else if (argument == "--learning-rate") {
            options.learning_rate = parse_float(value(index, argument), argument);
        } else if (argument == "--l2") {
            options.l2_penalty = parse_float(value(index, argument), argument);
        } else if (argument == "--gradient-clip") {
            options.gradient_clip = parse_float(value(index, argument), argument);
        } else if (argument == "--seed") {
            options.seed = parse_u64(value(index, argument), argument);
        } else if (argument == "--pattern") {
            options.pattern = value(index, argument);
        } else if (argument == "--export-dir") {
            options.export_dir = value(index, argument);
        } else if (argument == "--save-model") {
            options.save_model = value(index, argument);
        } else if (argument == "--load-model") {
            options.load_model = value(index, argument);
        } else if (argument == "--quantized") {
            options.quantized = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: jarvisx-autoencoder3d [options]\n"
                << "  --edge N              even input edge in [4,64]\n"
                << "  --channels N          latent channels in [1,32]\n"
                << "  --epochs N            SGD training steps\n"
                << "  --learning-rate X     learning rate in (0,1]\n"
                << "  --l2 X                L2 regularization in [0,1]\n"
                << "  --gradient-clip X     elementwise gradient clip\n"
                << "  --seed N              deterministic initialization seed\n"
                << "  --pattern NAME        sphere|shell|checker|wave|noise\n"
                << "  --quantized           use signed 3-bit latent for final inference\n"
                << "  --export-dir PATH     write CSV and OBJ artifacts\n"
                << "  --save-model PATH     persist trained model\n"
                << "  --load-model PATH     resume a saved model\n"
                << "  --quiet                suppress progress output\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        jarvisx::Autoencoder3D model = options.load_model.empty()
            ? jarvisx::Autoencoder3D({options.edge, options.channels,
                                     options.learning_rate, options.l2_penalty,
                                     options.gradient_clip, options.seed})
            : jarvisx::Autoencoder3D::load(options.load_model);

        const auto input = jarvisx::make_volume(
            model.config().input_edge, options.pattern, model.config().seed);
        std::filesystem::create_directories(options.export_dir);
        std::ofstream metrics_file(options.export_dir / "metrics.csv", std::ios::trunc);
        if (!metrics_file) throw std::runtime_error("cannot write metrics.csv");
        metrics_file << "step,mse,mae,max_abs_error,latent_energy,gradient_l2\n";

        const std::uint64_t report_interval = std::max<std::uint64_t>(1U, options.epochs / 20U);
        jarvisx::Autoencoder3DMetrics metrics{};
        for (std::uint64_t epoch = 0; epoch < options.epochs; ++epoch) {
            metrics = model.train_step(input);
            metrics_file << metrics.step << ',' << metrics.mse << ',' << metrics.mae
                         << ',' << metrics.max_abs_error << ',' << metrics.latent_energy
                         << ',' << metrics.gradient_l2 << '\n';
            if (!options.quiet &&
                (epoch == 0U || epoch + 1U == options.epochs ||
                 (epoch + 1U) % report_interval == 0U)) {
                std::cout << "step=" << metrics.step
                          << " mse=" << metrics.mse
                          << " mae=" << metrics.mae
                          << " latent_energy=" << metrics.latent_energy
                          << " gradient_l2=" << metrics.gradient_l2 << '\n';
            }
        }

        const auto latent = model.encode(input, options.quantized);
        const auto reconstruction = model.decode(latent);
        const auto final_metrics = jarvisx::measure_reconstruction(
            input, latent, reconstruction, model.steps());

        jarvisx::export_obj(input, options.export_dir / "input.obj");
        jarvisx::export_obj(latent, options.export_dir / "latent.obj");
        jarvisx::export_obj(reconstruction,
                            options.export_dir / "reconstruction.obj");

        const std::filesystem::path model_path = options.save_model.empty()
            ? options.export_dir / "model.jx3d"
            : options.save_model;
        model.save(model_path);

        if (!options.quiet) {
            std::cout << "final_mse=" << final_metrics.mse
                      << " final_mae=" << final_metrics.mae
                      << " quantized=" << (options.quantized ? "true" : "false")
                      << " model=" << model_path.string()
                      << " artifacts=" << options.export_dir.string() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Jarvis-X 3D autoencoder failure: " << error.what() << '\n';
        return 1;
    }
}
