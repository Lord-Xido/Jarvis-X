#include "jarvisx/symmetry_benchmark3d.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::size_t> parse_sizes(const std::string& text) {
    std::vector<std::size_t> out;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            throw std::invalid_argument("empty size token");
        }
        out.push_back(static_cast<std::size_t>(std::stoull(token)));
    }
    if (out.empty()) {
        throw std::invalid_argument("sizes list is empty");
    }
    return out;
}

std::vector<double> parse_noise(const std::string& text) {
    std::vector<double> out;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            throw std::invalid_argument("empty noise token");
        }
        out.push_back(std::stod(token));
    }
    if (out.empty()) {
        throw std::invalid_argument("noise list is empty");
    }
    return out;
}

void usage() {
    std::cout
        << "Jarvis-X 3D symmetry performance envelope\n\n"
        << "Usage:\n"
        << "  jarvisx-symmetry-benchmark3d [options]\n\n"
        << "Options:\n"
        << "  --quick                  Small deterministic smoke benchmark\n"
        << "  --sizes CSV              Grid sides, default 8,16,32,64,128,256\n"
        << "  --noise CSV              Flip probabilities, default 0,.05,.1,.2,.3,.4\n"
        << "  --repeats N              Repetitions per point, default 5\n"
        << "  --output PATH            CSV output, default symmetry3d-benchmark.csv\n"
        << "  --help                    Show this message\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        jarvisx::symmetry3d::bench::BenchmarkConfig config;
        std::string output = "symmetry3d-benchmark.csv";

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                usage();
                return EXIT_SUCCESS;
            }
            if (arg == "--quick") {
                config.sizes = {8, 16, 32};
                config.noise_levels = {0.0, 0.10, 0.30};
                config.repeats = 2;
                config.loop.max_optimization_sweeps = 32;
                continue;
            }
            if (arg == "--sizes" || arg == "--noise" || arg == "--repeats" || arg == "--output") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("missing value for " + arg);
                }
                const std::string value = argv[++i];
                if (arg == "--sizes") {
                    config.sizes = parse_sizes(value);
                } else if (arg == "--noise") {
                    config.noise_levels = parse_noise(value);
                } else if (arg == "--repeats") {
                    config.repeats = static_cast<std::size_t>(std::stoull(value));
                } else {
                    output = value;
                }
                continue;
            }
            throw std::invalid_argument("unknown argument: " + arg);
        }

        jarvisx::symmetry3d::bench::BenchmarkRunner runner(config);
        const auto rows = runner.run();
        runner.write_csv(output, rows);
        const auto summary = runner.summarize(rows);

        std::cout << "rows=" << summary.row_count << '\n'
                  << "mean_mse=" << summary.mean_mse << '\n'
                  << "mean_bit_accuracy=" << summary.mean_bit_accuracy << '\n'
                  << "mean_latency_us=" << summary.mean_latency_us << '\n'
                  << "mean_throughput_mpix_s=" << summary.mean_throughput_mpix_s << '\n'
                  << "csv=" << output << '\n'
                  << "note=timing is machine-dependent; correctness metrics are deterministic\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
