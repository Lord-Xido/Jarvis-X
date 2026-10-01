#include "jarvisx/inward_meta_q16.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    int steps = 15;
    bool quiet = false;
    jarvisx::dmimte::EngineConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value after " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--steps") {
            steps = std::stoi(require_value("--steps"));
        } else if (arg == "--no-noise") {
            config.enable_noise = false;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "DM-IMTE-Q16 bounded inward meta-evolution runtime\n\n"
                << "Usage:\n"
                << "  jarvisx-inward-meta-q16 [--steps N] [--no-noise] [--quiet]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }

    if (steps <= 0) {
        std::cerr << "--steps must be positive\n";
        return 2;
    }

    jarvisx::dmimte::InwardMetaEngineQ16 engine(config);
    std::cout << std::fixed << std::setprecision(8);

    if (!quiet) {
        std::cout << "DM-IMTE-Q16 deterministic inward meta-evolution runtime\n";
        std::cout << "world=64^3 latent=16^3 core=4^3\n";
        std::cout << "initial_rmse=" << engine.rmse() << "\n";
        std::cout << "projection_floor=" << engine.projection_floor_rmse() << "\n\n";
    }

    jarvisx::dmimte::StepMetrics last;
    for (int i = 0; i < steps; ++i) {
        last = engine.step();
        if (!quiet) {
            std::cout
                << "step=" << std::setw(2) << last.step
                << " rmse=" << last.rmse
                << " latent_max_error=" << last.latent_max_error
                << " core_max=" << last.core_max_abs
                << " noise=" << last.noise_amplitude
                << "\n";
        }
    }

    if (quiet) {
        std::cout
            << "step=" << last.step
            << " rmse=" << last.rmse
            << " projection_floor=" << last.projection_floor_rmse
            << " latent_max_error=" << last.latent_max_error
            << "\n";
    }

    return 0;
}
