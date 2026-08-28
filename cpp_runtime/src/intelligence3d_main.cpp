#include "jarvisx/intelligence_vm3d.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace jarvisx::intelligence3d;

namespace {
struct Options {
    std::uint64_t axis_gb{1000U};
    std::uint64_t resident_gb{10U};
    std::uint32_t page_edge{32U};
    std::uint64_t cycles{1U};
    std::uint64_t max_steps{100000U};
    std::filesystem::path state_dir{"jarvisx-3d-state"};
    std::filesystem::path bytecode{};
    std::filesystem::path generate_demo{};
    bool quiet{false};
};

std::uint64_t parse_u64(const std::string& value, const char* name) {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) {
        throw std::invalid_argument(std::string(name) + " must be an integer");
    }
    return parsed;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "--axis-gb") options.axis_gb = parse_u64(next("--axis-gb"), "axis-gb");
        else if (arg == "--resident-gb") options.resident_gb = parse_u64(next("--resident-gb"), "resident-gb");
        else if (arg == "--page-edge") options.page_edge = static_cast<std::uint32_t>(parse_u64(next("--page-edge"), "page-edge"));
        else if (arg == "--cycles") options.cycles = parse_u64(next("--cycles"), "cycles");
        else if (arg == "--max-steps") options.max_steps = parse_u64(next("--max-steps"), "max-steps");
        else if (arg == "--state-dir") options.state_dir = next("--state-dir");
        else if (arg == "--bytecode") options.bytecode = next("--bytecode");
        else if (arg == "--generate-demo") options.generate_demo = next("--generate-demo");
        else if (arg == "--quiet") options.quiet = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Jarvis-X 3D Intelligence VM\n"
                      << "  --axis-gb N        virtual extent per X/Y/Z axis (default 1000)\n"
                      << "  --resident-gb N    maximum resident page cache (default 10)\n"
                      << "  --page-edge N      cubic page edge in bytes (default 32)\n"
                      << "  --cycles N         auto-execution cycles (default 1)\n"
                      << "  --max-steps N      safety ceiling per cycle\n"
                      << "  --state-dir PATH   persistent sparse page store\n"
                      << "  --bytecode PATH    load JX3DVM1 bytecode\n"
                      << "  --generate-demo P  emit demo bytecode to P before running\n"
                      << "  --quiet             compact output\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.axis_gb == 0U || options.resident_gb == 0U || options.cycles == 0U) {
        throw std::invalid_argument("axis-gb, resident-gb and cycles must be positive");
    }
    if (options.axis_gb > std::numeric_limits<std::uint64_t>::max() / kDecimalGB ||
        options.resident_gb > std::numeric_limits<std::uint64_t>::max() / kDecimalGB) {
        throw std::invalid_argument("GB value is too large");
    }
    return options;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const auto axis_extent = options.axis_gb * kDecimalGB;
        const auto resident_limit = options.resident_gb * kDecimalGB;

        VirtualVolume3D volume({axis_extent, options.page_edge, resident_limit,
                                options.state_dir / "pages"});
        PsiIntelligenceCore psi(volume, {});
        IntelligenceVm3D vm(volume, psi);

        BytecodeProgram program = make_demo_program(1U, 1U, 1U, 173U);
        if (!options.generate_demo.empty()) {
            std::filesystem::create_directories(options.generate_demo.parent_path().empty()
                                                    ? std::filesystem::path(".")
                                                    : options.generate_demo.parent_path());
            program.save(options.generate_demo);
        }
        if (!options.bytecode.empty()) {
            program = BytecodeProgram::load(options.bytecode);
        }

        vm.run(program, options.cycles, options.max_steps);
        volume.flush();

        const auto& vs = volume.stats();
        const auto& ms = vm.stats();
        const auto& trace = psi.last_trace();
        if (!options.quiet) {
            std::cout << "JARVIS-X 3D INTELLIGENCE VM\n"
                      << "virtual-axis-bytes=" << volume.axis_extent() << "\n"
                      << "conceptual-volume=" << volume.conceptual_capacity() << "\n"
                      << "resident-cap-bytes=" << volume.resident_limit_bytes() << "\n"
                      << "resident-bytes=" << volume.resident_bytes() << "\n"
                      << "page-bytes=" << volume.page_bytes() << "\n"
                      << "vm-cycles=" << ms.cycles << " vm-steps=" << ms.steps << "\n"
                      << "psi-inferences=" << ms.psi_inferences
                      << " psi-learning=" << ms.psi_learning_steps << "\n"
                      << "page-faults=" << vs.page_faults << " evictions=" << vs.evictions
                      << " disk-loads=" << vs.disk_loads << " disk-stores=" << vs.disk_stores << "\n"
                      << "last-mask=" << trace.mask << " last-decoded="
                      << static_cast<unsigned>(trace.decoded) << " last-error="
                      << trace.prediction_error << "\n";
        } else {
            std::cout << "JX3D OK cycles=" << ms.cycles << " steps=" << ms.steps
                      << " resident=" << volume.resident_bytes() << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jarvisx-intelligence3d: " << error.what() << '\n';
        return 2;
    }
}
