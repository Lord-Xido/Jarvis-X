#include "jarvisx/compile_interpret_accelerator.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path state_dir{"jarvisx-compile-interpret-acceleration-state"};
    std::uint32_t tiles{32U};
    std::uint8_t levels{2U};
    bool quiet{};
};

std::uint32_t parse_u32(const std::string& text, const char* name) {
    std::size_t used = 0U;
    const unsigned long value = std::stoul(text, &used, 10);
    if (used != text.size() || value > 0xffffffffUL) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return static_cast<std::uint32_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--state-dir") {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after --state-dir");
            options.state_dir = argv[++i];
        } else if (arg == "--tiles") {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after --tiles");
            options.tiles = parse_u32(argv[++i], "tile count");
        } else if (arg == "--levels") {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after --levels");
            const std::uint32_t value = parse_u32(argv[++i], "level count");
            if (value > 255U) throw std::invalid_argument("level count exceeds uint8");
            options.levels = static_cast<std::uint8_t>(value);
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: DrMoagi-Compile-Interpret-Accelerator "
                         "[--tiles N] [--levels N] [--state-dir PATH] [--quiet]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.tiles == 0U || options.levels == 0U || options.levels > 8U) {
        throw std::invalid_argument("tiles must be positive and levels must be in [1,8]");
    }
    return options;
}

std::uint8_t source_byte(std::uint64_t index) noexcept {
    return static_cast<std::uint8_t>((index * 37ULL + (index >> 3U) + (index >> 11U)) & 0xffULL);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::filesystem::remove_all(options.state_dir);

        jarvisx::intelligence3d::VirtualVolume3D volume({
            jarvisx::world::kVmadCoordExtent,
            32U,
            8ULL * 1024ULL * 1024ULL,
            options.state_dir / "pages",
        });

        const auto plan = jarvisx::cube::make_demo_plan(options.tiles, options.levels);
        const std::uint64_t source_bytes =
            static_cast<std::uint64_t>(options.tiles) * jarvisx::cube::kCubeTileBytes;

        for (std::uint64_t i = 0ULL; i < source_bytes; ++i) {
            const auto address = jarvisx::world::vmad_advance_linear(plan.source, i);
            volume.write(address.coord(), source_byte(i));
        }

        jarvisx::cube::RecursiveCubeInterpreter interpreter(volume);
        jarvisx::cube::acceleration::ParallelCompileInterpretAccelerator accelerator(interpreter);

        jarvisx::cube::acceleration::AccelerationContract contract;
        contract.fold.retention.assign(static_cast<std::size_t>(options.levels), 1.0L / 32.0L);

        const auto report = accelerator.run(plan.execution_buffer, contract);
        volume.flush();

        jarvisx::cube::acceleration::HarmonicCompileInterpretPhase phase;
        phase.advance(1.0L, 1.0L, contract);

        if (!options.quiet) {
            std::cout << std::setprecision(12)
                      << "DM-vOmegaXi+ parallel compile-interpret acceleration contract\n"
                      << "target_definition=(10^24)^(10^24)\n"
                      << "target_log10_speedup=" << report.target_log10_speedup << '\n'
                      << "modeled_log10_speedup=" << report.modeled_log10_speedup << '\n'
                      << "target_gap_log10=" << report.target_gap_log10 << '\n'
                      << "elapsed_ns=" << report.elapsed_ns << '\n'
                      << "maximum_tile_pass_budget=" << report.maximum_tile_pass_budget << '\n'
                      << "actual_passes=" << report.cube.total_passes << '\n'
                      << "convergence_work_reduction=" << report.convergence_work_reduction << '\n'
                      << "bounded_kinetic_gain=" << report.bounded_kinetic_gain << '\n'
                      << "compile_phase_component=" << phase.compile_component() << '\n'
                      << "interpret_phase_component=" << phase.interpret_component() << '\n'
                      << "commands_executed=" << report.cube.commands_executed << '\n'
                      << "tiles_processed=" << report.cube.tiles_processed << '\n'
                      << "claim_boundary=hyper_exponential_target_is_symbolic_log_space_only;"
                         "reported_runtime_and_work_reduction_are_empirical_for_this_run\n";
        }

        std::filesystem::remove_all(options.state_dir);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DrMoagi Compile-Interpret Accelerator failure: " << error.what() << '\n';
        return 1;
    }
}
