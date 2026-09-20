#include "jarvisx/dr_moagi_6400x1000.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

struct CliOptions {
    jarvisx::dm6400::EngineConfig config{};
    bool quiet = false;
};

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after " << name << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--cycles") options.config.cycles = std::stoi(require_value("--cycles"));
        else if (arg == "--workers") options.config.physical_workers = std::stoi(require_value("--workers"));
        else if (arg == "--iterations") options.config.programmable_iterations = std::stoi(require_value("--iterations"));
        else if (arg == "--seed") options.config.seed = std::stoull(require_value("--seed"));
        else if (arg == "--fp-tol") options.config.fixed_point_tolerance = std::stod(require_value("--fp-tol"));
        else if (arg == "--active-tol") options.config.active_tolerance = std::stod(require_value("--active-tol"));
        else if (arg == "--quiet") options.quiet = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Dr Moagi 6400x1000 3D Geometric Coordinating AE/AD Engine\n\n"
                << "Usage:\n"
                << "  DrMoagi-6400x1000-3D [options]\n\n"
                << "Options:\n"
                << "  --cycles N       macrocycles (default 4)\n"
                << "  --workers N      bounded physical worker threads\n"
                << "  --iterations N   programmable slots/panel (default 1000)\n"
                << "  --seed N         deterministic initialization seed\n"
                << "  --fp-tol X       fixed-point tolerance\n"
                << "  --active-tol X   per-panel activity threshold\n"
                << "  --quiet          only print final receipt\n";
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(2);
        }
    }
    options.config.validate();
    return options;
}

void print_receipt(const jarvisx::dm6400::CycleReceipt& r) {
    std::cout
        << "cycle=" << r.cycle
        << " logical=" << r.logical_program_slots
        << " executed=" << r.executed_program_ops
        << " active=" << r.active_panels
        << " exec_fraction=" << r.execution_fraction
        << " mse=" << r.reconstruction_mse
        << " fp=" << r.fixed_point_residual
        << " corrections=" << r.corrections
        << " relax=" << r.relaxation
        << " rho=" << r.memory_rho
        << " checksum=" << r.checksum
        << " ms=" << r.cycle_ms
        << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const CliOptions options = parse_args(argc, argv);
    jarvisx::dm6400::DrMoagi6400x1000Engine engine(options.config);

    if (!options.quiet) {
        const auto sizes = jarvisx::dm6400::Geometry::multigrid_sizes();
        std::cout << "Dr Moagi 3D Geometric Coordinating 6400x1000 AE/AD Engine\n";
        std::cout << "geometry      : 20x20x16 = 6400 panels\n";
        std::cout << "clusters      : 5x5x4 = 100 clusters x 64 panels\n";
        std::cout << "program slots : 6400 x " << options.config.programmable_iterations
                  << " = " << static_cast<unsigned long long>(
                        6400ULL * static_cast<unsigned long long>(options.config.programmable_iterations))
                  << " logical slots/cycle\n";
        std::cout << "inward path   : ";
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            if (i != 0U) std::cout << " -> ";
            std::cout << sizes[i];
        }
        std::cout << " -> 4 -> 18 -> 100 -> 800 -> 6400\n";
        std::cout << "workers       : " << engine.physical_workers() << "\n\n";
    }

    jarvisx::dm6400::CycleReceipt last;
    for (int cycle = 0; cycle < options.config.cycles; ++cycle) {
        last = engine.step();
        if (!options.quiet) print_receipt(last);
    }

    if (options.quiet) print_receipt(last);
    return 0;
}
