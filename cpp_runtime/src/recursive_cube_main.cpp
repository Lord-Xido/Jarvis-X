#include "jarvisx/recursive_cube_interpreter.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path state_dir{"jarvisx-recursive-cube-state"};
    std::uint32_t tiles{32U};
    std::uint8_t levels{2U};
    bool quiet{};
};

std::uint32_t parse_u32(const std::string& text, const char* name) {
    std::size_t used = 0U;
    const unsigned long value = std::stoul(text, &used, 10);
    if (used != text.size() || value > 0xffffffffUL) throw std::invalid_argument(std::string("invalid ") + name);
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
            std::cout << "Usage: DrMoagi-Recursive-Cube [--tiles N] [--levels N] [--state-dir PATH] [--quiet]\n";
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
        const std::uint64_t source_bytes = static_cast<std::uint64_t>(options.tiles) * jarvisx::cube::kCubeTileBytes;
        for (std::uint64_t i = 0ULL; i < source_bytes; ++i) {
            const auto address = jarvisx::world::vmad_advance_linear(plan.source, i);
            volume.write(address.coord(), source_byte(i));
        }

        jarvisx::cube::RecursiveCubeInterpreter interpreter(volume);
        const auto metrics = interpreter.run(plan.execution_buffer);
        volume.flush();

        long double absolute_error = 0.0L;
        for (std::uint64_t i = 0ULL; i < source_bytes; ++i) {
            const auto address = jarvisx::world::vmad_advance_linear(plan.final_output, i);
            const int source = static_cast<int>(source_byte(i));
            const int reconstructed = static_cast<int>(volume.read(address.coord()));
            const int delta = source - reconstructed;
            absolute_error += static_cast<long double>(delta < 0 ? -delta : delta);
        }
        const double mean_absolute_error = source_bytes == 0ULL ? 0.0 :
            static_cast<double>(absolute_error / static_cast<long double>(source_bytes));

        if (!options.quiet) {
            std::cout << "DM-vOmegaXi+ recursive cube interpreter\n"
                      << "execution_buffer=DMCUBE1 validated=" << (metrics.execution_buffer_validated ? 1 : 0) << '\n'
                      << "logical_cube_extent=1000000000 sparse=1 vmad_bits=128\n"
                      << "base_tiles=" << options.tiles << " levels=" << static_cast<unsigned>(options.levels)
                      << " source_bytes=" << source_bytes << '\n'
                      << "commands_executed=" << metrics.commands_executed
                      << " tiles_processed=" << metrics.tiles_processed
                      << " passes=" << metrics.total_passes << '\n'
                      << "accepted_passes=" << metrics.accepted_passes
                      << " rejected_passes=" << metrics.rejected_passes
                      << " converged_tiles=" << metrics.converged_tiles << '\n'
                      << "aggregate_command_input_bytes=" << metrics.encoded_input_bytes
                      << " latent_bytes_committed=" << metrics.latent_bytes_committed
                      << " total_world_output_bytes=" << metrics.decoded_output_bytes << '\n'
                      << "recursive_output_mae=" << mean_absolute_error << '\n'
                      << "world_commits=" << interpreter.engine().stats().commits
                      << " world_rollbacks=" << interpreter.engine().stats().rollbacks << '\n'
                      << "claim_boundary=sparse_software_reference_no_sota_or_physical_timing_claim\n";
        }

        std::filesystem::remove_all(options.state_dir);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DrMoagi Recursive Cube failure: " << error.what() << '\n';
        return 1;
    }
}
