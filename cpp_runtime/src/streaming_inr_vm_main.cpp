#include "jarvisx/streaming_inr_vm.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using jarvisx::inrvm::MemoryPlan;
using jarvisx::inrvm::MiB;
using jarvisx::inrvm::Query3D;
using jarvisx::inrvm::StreamingInrVm;
using jarvisx::inrvm::VmConfig;

int main(int argc, char** argv) {
    std::size_t queries = 64;
    bool quiet = false;
    VmConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--queries" && i + 1 < argc) queries = static_cast<std::size_t>(std::stoull(argv[++i]));
        else if (arg == "--axis" && i + 1 < argc) cfg.virtual_axis = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--cutoff" && i + 1 < argc) cfg.entropy_cutoff = std::stof(argv[++i]);
        else if (arg == "--quiet") quiet = true;
        else if (arg == "--help") {
            std::cout << "Usage: jarvisx-streaming-inr-vm [--queries N] [--axis N] [--cutoff X] [--quiet]\n";
            return EXIT_SUCCESS;
        }
    }

    try {
        StreamingInrVm vm(cfg);
        std::vector<Query3D> batch;
        batch.reserve(queries);
        const double axis = static_cast<double>(cfg.virtual_axis);
        for (std::size_t i = 0; i < queries; ++i) {
            const double u = queries > 1 ? static_cast<double>(i) / static_cast<double>(queries - 1) : 0.5;
            batch.push_back({u * axis, (1.0 - u) * axis, std::fmod(0.61803398875 * static_cast<double>(i), 1.0) * axis, static_cast<double>(i)});
        }

        const auto results = vm.execute_batch(batch);
        std::size_t emitted = 0;
        double checksum = 0.0;
        for (const auto& r : results) {
            if (r.emitted) {
                ++emitted;
                for (float v : r.feature) checksum += static_cast<double>(v);
            }
        }

        if (!quiet) {
            std::cout << "Jarvis-X Streaming INR VM\n"
                      << "virtual axis: " << cfg.virtual_axis << " (volume remains virtual; no voxel array allocated)\n"
                      << "queries: " << queries << ", emitted: " << emitted << "\n"
                      << "managed ceiling: " << MemoryPlan::managed_limit / MiB << " MiB\n"
                      << "host headroom: " << MemoryPlan::host_headroom / MiB << " MiB\n"
                      << "actual managed bytes claimed: " << vm.budget().total_used() << "\n"
                      << std::setprecision(12) << "checksum: " << checksum << "\n";
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "streaming INR VM error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
