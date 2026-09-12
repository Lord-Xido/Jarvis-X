#include "jarvisx/moagi3d_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

namespace {

struct Options {
    std::size_t edge{8U};
    std::size_t channels{3U};
    std::size_t latent_channels{4U};
    std::size_t max_iterations{32U};
    float epsilon{1.0e-5F};
    float feedback_gain{0.5F};
    float temperature{8.0F};
    std::uint64_t seed{0x4D4F4147493344ULL};
    std::string pattern{"sphere"};
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
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value after " + flag);
        }
        ++index;
        return argv[index];
    };

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--edge") {
            options.edge = static_cast<std::size_t>(parse_u64(value(index, argument), argument));
        } else if (argument == "--channels") {
            options.channels = static_cast<std::size_t>(parse_u64(value(index, argument), argument));
        } else if (argument == "--latent-channels") {
            options.latent_channels = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--max-iterations") {
            options.max_iterations = static_cast<std::size_t>(
                parse_u64(value(index, argument), argument));
        } else if (argument == "--epsilon") {
            options.epsilon = parse_float(value(index, argument), argument);
        } else if (argument == "--feedback-gain") {
            options.feedback_gain = parse_float(value(index, argument), argument);
        } else if (argument == "--temperature") {
            options.temperature = parse_float(value(index, argument), argument);
        } else if (argument == "--seed") {
            options.seed = parse_u64(value(index, argument), argument);
        } else if (argument == "--pattern") {
            options.pattern = value(index, argument);
        } else if (argument == "--quantized") {
            options.quantized = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: DrMoagi-3D-Engine [options]\n"
                << "  --edge N              even 3D input edge in [4,64]\n"
                << "  --channels K          parallel Moagi manifold channels in [1,32]\n"
                << "  --latent-channels N   latent channels per encoder/decoder\n"
                << "  --max-iterations N    RAC hard iteration ceiling\n"
                << "  --epsilon X           RAC absolute L2 convergence threshold\n"
                << "  --feedback-gain X     recursive under-relaxation gain in (0,1]\n"
                << "  --temperature X       FRM softmax temperature\n"
                << "  --seed N              deterministic channel seed base\n"
                << "  --pattern NAME        sphere|shell|checker|wave|noise\n"
                << "  --quantized           use Q3 latent reconstruction\n"
                << "  --quiet               suppress telemetry\n";
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
        const jarvisx::Moagi3DConfig config{
            options.edge,
            options.channels,
            options.latent_channels,
            options.max_iterations,
            options.epsilon,
            options.feedback_gain,
            options.temperature,
            options.seed,
            options.quantized
        };

        const jarvisx::Tensor4D input = jarvisx::make_volume(
            options.edge, options.pattern, options.seed);
        const jarvisx::Moagi3DEngine engine(config);
        const jarvisx::Moagi3DResult result = engine.run(input);

        if (!options.quiet) {
            std::cout << std::setprecision(8)
                      << "engine=Moagi-3D"
                      << " channels=" << engine.channel_count()
                      << " iterations=" << result.iterations
                      << " converged=" << (result.converged ? "true" : "false")
                      << " final_delta_l2=" << result.final_delta_l2
                      << " anchor_l2=" << result.final_anchor_l2
                      << " termination="
                      << (result.converged ? "epsilon" : "max_iterations")
                      << '\n';

            for (const auto& step : result.history) {
                std::cout << "iteration=" << step.iteration
                          << " delta_l2=" << step.delta_l2
                          << " anchor_l2=" << step.anchor_l2
                          << " alpha=[";
                for (std::size_t index = 0U; index < step.fusion_weights.size(); ++index) {
                    if (index != 0U) std::cout << ',';
                    std::cout << step.fusion_weights[index];
                }
                std::cout << "] mse=[";
                for (std::size_t index = 0U; index < step.channel_mse.size(); ++index) {
                    if (index != 0U) std::cout << ',';
                    std::cout << step.channel_mse[index];
                }
                std::cout << "]\n";
            }
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Moagi-3D engine failure: " << error.what() << '\n';
        return 1;
    }
}
