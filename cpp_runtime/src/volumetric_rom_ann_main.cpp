#include "jarvisx/volumetric_rom_ann.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [--cycles N] [--active-tiles N] [--quiet]\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::uint32_t cycles = 32u;
        bool quiet = false;
        jarvisx::volumetric_rom::RuntimePolicy policy{};

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--cycles") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("--cycles requires a value");
                }
                const auto value = std::stoul(argv[++i]);
                if (value == 0ul || value > 100000ul) {
                    throw std::invalid_argument("--cycles must be in [1, 100000]");
                }
                cycles = static_cast<std::uint32_t>(value);
            } else if (arg == "--active-tiles") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("--active-tiles requires a value");
                }
                const auto value = std::stoul(argv[++i]);
                if (value == 0ul || value > 4096ul) {
                    throw std::invalid_argument("--active-tiles must be in [1, 4096]");
                }
                policy.max_active_tiles = static_cast<std::uint32_t>(value);
            } else if (arg == "--quiet") {
                quiet = true;
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        jarvisx::volumetric_rom::Engine engine(policy);
        const auto stats = engine.run(cycles, quiet);

        if (!quiet) {
            std::cout << "\n=== Runtime State ===\n"
                      << "cycles            : " << stats.cycles << '\n'
                      << "active tiles      : " << stats.active_tiles << '\n'
                      << "physical bytes    : " << stats.physical_bytes << '\n'
                      << "virtual capacity  : 2^60 bytes = 1 EiB\n"
                      << "pyramid levels    : " << stats.pyramid_levels << '\n'
                      << "last MSE          : " << stats.last_error << '\n'
                      << "last FP relative  : " << stats.last_fixed_point_relative << '\n'
                      << "FP step budget    : " << stats.fixed_point_steps << '\n'
                      << "sparse residuals  : " << stats.sparse_residual_voxels << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
