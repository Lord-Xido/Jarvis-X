#include "jarvisx/world_engine_vmad.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::filesystem::path state_dir{"jarvisx-world-state"};
    bool quiet{};
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--state-dir") {
            if (i + 1 >= argc) throw std::invalid_argument("missing value after --state-dir");
            options.state_dir = argv[++i];
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: DrMoagi-World-Engine [--state-dir PATH] [--quiet]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::filesystem::remove_all(options.state_dir);

        jarvisx::intelligence3d::VirtualVolume3D volume({
            jarvisx::world::kVmadCoordExtent,
            32U,
            2ULL * 1024ULL * 1024ULL,
            options.state_dir / "pages",
        });

        const auto source = jarvisx::world::Vmad128::pack(1U, 1U, 0U, 12U, 34U, 56U);
        const auto output = jarvisx::world::Vmad128::pack(2U, 1U, 0U, 20U, 40U, 60U);

        for (std::size_t i = 0U; i < jarvisx::world::kDefaultTileBytes; ++i) {
            const auto address = jarvisx::world::vmad_advance_linear(source, static_cast<std::uint64_t>(i));
            const std::uint8_t value = static_cast<std::uint8_t>((i * 37U + (i >> 3U)) & 0xffU);
            volume.write(address.coord(), value);
        }

        jarvisx::world::WorldEngine128 engine(volume);
        const auto program = jarvisx::world::make_world_demo_program(source, output);
        engine.run(program, 1000ULL);
        volume.flush();

        if (!options.quiet) {
            const auto& stats = engine.stats();
            std::cout << "DM-vOmegaXi+ VMAD128 kinetic world-engine reference\n"
                      << "vmad_bits=128 coord_bits=33 axis_extent=" << jarvisx::world::kVmadCoordExtent << '\n'
                      << "vector_registers=" << jarvisx::world::kVectorRegisterCount
                      << " vector_bytes=" << jarvisx::world::kVectorBytes
                      << " scalar_registers=" << jarvisx::world::kScalarRegisterCount
                      << " vmad_registers=" << jarvisx::world::kVmadRegisterCount << '\n'
                      << "micro_ops=" << stats.issued_micro_ops
                      << " issue_cycles=" << stats.logical_issue_cycles
                      << " estimated_pipeline_latency_cycles=" << stats.estimated_pipeline_latency_cycles << '\n'
                      << "bytes_ingested=" << stats.bytes_ingested
                      << " bytes_stored=" << stats.bytes_stored
                      << " commits=" << stats.commits
                      << " rollbacks=" << stats.rollbacks
                      << " delta_mean=" << engine.last_delta_mean() << '\n'
                      << "stage_ingest=" << stats.stage_issues[0]
                      << " stage_reduce=" << stats.stage_issues[1]
                      << " stage_fuse=" << stats.stage_issues[2]
                      << " stage_reconstruct=" << stats.stage_issues[3]
                      << " stage_feedback=" << stats.stage_issues[4] << '\n'
                      << "claim_boundary=software_reference_no_photonic_or_subnanosecond_timing_claim\n";
        }

        std::filesystem::remove_all(options.state_dir);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DrMoagi World Engine failure: " << error.what() << '\n';
        return 1;
    }
}
