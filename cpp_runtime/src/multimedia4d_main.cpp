#include "jarvisx/multimedia4d.hpp"

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
    std::size_t temporal_depth{8U};
    std::uint16_t proposal_steps{2U};
    std::uint64_t cycles{120U};
    float learning_rate{0.03F};
    float temporal_decay{0.72F};
    float accept_tolerance{1.0e-6F};
    std::uint64_t seed{0x4A415256495358ULL};
    std::filesystem::path output_dir{".jarvisx-multimedia4d"};
    bool quantized{};
    bool quiet{};
};

std::uint64_t parse_u64(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid integer after " + flag);
    }
    return parsed;
}

float parse_float(const std::string& value, const std::string& flag) {
    std::size_t consumed = 0U;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument("invalid float after " + flag);
    }
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    auto value = [&](int& index, const std::string& flag) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value after " + flag);
        }
        return std::string(argv[++index]);
    };

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--edge") {
            options.edge = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--channels") {
            options.channels = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--temporal-depth") {
            options.temporal_depth = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--proposal-steps") {
            options.proposal_steps = static_cast<std::uint16_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--cycles") {
            options.cycles = std::max<std::uint64_t>(
                1U, parse_u64(value(index, argument), argument));
        } else if (argument == "--learning-rate") {
            options.learning_rate = parse_float(value(index, argument), argument);
        } else if (argument == "--temporal-decay") {
            options.temporal_decay = parse_float(value(index, argument), argument);
        } else if (argument == "--accept-tolerance") {
            options.accept_tolerance = parse_float(value(index, argument), argument);
        } else if (argument == "--seed") {
            options.seed = parse_u64(value(index, argument), argument);
        } else if (argument == "--output-dir") {
            options.output_dir = value(index, argument);
        } else if (argument == "--quantized") {
            options.quantized = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: jarvisx-multimedia4d [options]\n"
                << "  --edge N              even input edge in [4,64]\n"
                << "  --channels N          latent channels in [1,32]\n"
                << "  --temporal-depth N    latent history depth in [1,64]\n"
                << "  --proposal-steps N    candidate SGD steps in [1,64]\n"
                << "  --cycles N            adaptive transactions\n"
                << "  --learning-rate X     base SGD learning rate\n"
                << "  --temporal-decay X    history weight decay in [0,1)\n"
                << "  --accept-tolerance X  maximum admitted MSE regression\n"
                << "  --seed N              deterministic seed\n"
                << "  --quantized           signed 3-bit temporal inference\n"
                << "  --output-dir PATH     telemetry, OBJ and checkpoint path\n"
                << "  --quiet               suppress progress output\n";
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
        jarvisx::Multimedia4DConfig config;
        config.model.input_edge = options.edge;
        config.model.latent_channels = options.channels;
        config.model.learning_rate = options.learning_rate;
        config.model.seed = options.seed;
        config.temporal_depth = options.temporal_depth;
        config.temporal_decay = options.temporal_decay;
        config.proposal_steps = options.proposal_steps;
        config.accept_tolerance = options.accept_tolerance;
        config.quantized_inference = options.quantized;

        jarvisx::MultimediaAutoencoder4D engine(config);
        std::filesystem::create_directories(options.output_dir);
        std::ofstream telemetry(options.output_dir / "multimedia4d.csv",
                                std::ios::trunc);
        if (!telemetry) throw std::runtime_error("cannot write multimedia4d.csv");
        telemetry
            << "cycle,selected,aggregate_mse,temporal_coherence,accepted,rejected,"
            << "visual_mse,audio_mse,text_mse,generic_mse\n";

        const std::uint64_t report_interval =
            std::max<std::uint64_t>(1U, options.cycles / 20U);
        for (std::uint64_t cycle = 0U; cycle < options.cycles; ++cycle) {
            const auto metrics = engine.step();
            telemetry << metrics.cycle << ','
                      << jarvisx::media_name(metrics.selected) << ','
                      << metrics.aggregate_mse << ','
                      << metrics.aggregate_temporal_coherence << ','
                      << metrics.accepted << ',' << metrics.rejected;
            for (const auto media : jarvisx::media_types()) {
                telemetry << ',' << metrics.modalities[
                    jarvisx::media_index(media)].temporal_mse;
            }
            telemetry << '\n';

            if (!options.quiet &&
                (cycle == 0U || cycle + 1U == options.cycles ||
                 (cycle + 1U) % report_interval == 0U)) {
                std::cout << "cycle=" << metrics.cycle
                          << " selected=" << jarvisx::media_name(metrics.selected)
                          << " aggregate_mse=" << metrics.aggregate_mse
                          << " temporal_coherence="
                          << metrics.aggregate_temporal_coherence
                          << " commits=" << metrics.accepted
                          << " rollbacks=" << metrics.rejected << '\n';
            }
        }

        jarvisx::export_multimedia4d_snapshot(engine, options.output_dir);
        engine.save_checkpoint(options.output_dir / "checkpoint");
        if (!options.quiet) {
            std::cout << "artifacts=" << options.output_dir.string() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Jarvis-X multimedia4d failure: " << error.what() << '\n';
        return 1;
    }
}
