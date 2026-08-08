#include "jarvisx/riof3d.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::size_t parse_size(const char* text, const char* name) {
    try {
        const unsigned long long value = std::stoull(text);
        if (value == 0ULL) throw std::invalid_argument("zero");
        return static_cast<std::size_t>(value);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
}

std::uint64_t parse_u64(const char* text, const char* name) {
    try {
        return static_cast<std::uint64_t>(std::stoull(text));
    } catch (...) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
}

void print_usage() {
    std::cout << "jarvisx-riof3d [--cycles N] [--edge N] [--pattern sphere|shell|wave|noise] "
                 "[--seed N] [--quiet]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        jarvisx::Riof3DConfig config;
        std::size_t cycles = 64U;
        std::string pattern = "sphere";
        bool quiet = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cycles" && i + 1 < argc) {
                cycles = parse_size(argv[++i], "cycle count");
            } else if (arg == "--edge" && i + 1 < argc) {
                config.edge = parse_size(argv[++i], "edge");
            } else if (arg == "--pattern" && i + 1 < argc) {
                pattern = argv[++i];
            } else if (arg == "--seed" && i + 1 < argc) {
                config.seed = parse_u64(argv[++i], "seed");
            } else if (arg == "--quiet") {
                quiet = true;
            } else if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else {
                throw std::invalid_argument("unknown or incomplete argument: " + arg);
            }
        }

        jarvisx::Riof3D engine(config);
        engine.initialize(pattern);
        const jarvisx::Riof3DMetrics initial = engine.metrics();

        if (!quiet) {
            std::cout << "Jarvis X RIOF-3D | edge=" << config.edge
                      << " | voxels=" << engine.field().size()
                      << " | pattern=" << pattern << '\n';
            std::cout << "step,total_energy,residual,dt,damping,enhancement,max_abs\n";
            std::cout << std::setprecision(9) << initial.step << ',' << initial.total_energy
                      << ',' << initial.mean_abs_residual << ',' << initial.timestep
                      << ',' << initial.damping << ',' << initial.enhancement
                      << ',' << initial.max_abs_value << '\n';
        }

        jarvisx::Riof3DMetrics current = initial;
        for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
            current = engine.step();
            if (!quiet && (cycle + 1U == cycles || (cycle + 1U) % 8U == 0U)) {
                std::cout << current.step << ',' << current.total_energy
                          << ',' << current.mean_abs_residual << ',' << current.timestep
                          << ',' << current.damping << ',' << current.enhancement
                          << ',' << current.max_abs_value << '\n';
            }
        }

        if (!std::isfinite(current.total_energy) || !std::isfinite(current.max_abs_value)) {
            std::cerr << "RIOF-3D became non-finite\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RIOF-3D error: " << error.what() << '\n';
        return 1;
    }
}
