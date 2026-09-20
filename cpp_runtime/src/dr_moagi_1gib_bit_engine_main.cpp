#include "jarvisx/bitmatrix1gib.hpp"
#include "jarvisx/dr_moagi_6400x1000.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    jarvisx::dm6400::EngineConfig engine{};
    jarvisx::Stream1GiBConfig bitstream{};
    bool quiet{false};

    Options() {
        engine.cycles = 1;
        engine.programmable_iterations = 1000;
        bitstream.target_bytes = jarvisx::kOneGiBBytes;
        bitstream.chunk_bytes = 8U * 1024U * 1024U;
        bitstream.pattern = jarvisx::StreamPattern::Sparse3D;
        bitstream.window_ms = 100U;
    }
};

std::uint64_t parse_u64(const std::string& text, const char* name) {
    std::size_t consumed = 0U;
    const auto value = std::stoull(text, &consumed, 10);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    return value;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(name) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--cycles") {
            options.engine.cycles = std::stoi(require_value("--cycles"));
        } else if (arg == "--iterations") {
            options.engine.programmable_iterations = std::stoi(require_value("--iterations"));
        } else if (arg == "--workers") {
            options.engine.physical_workers = std::stoi(require_value("--workers"));
        } else if (arg == "--bytes") {
            options.bitstream.target_bytes = parse_u64(require_value("--bytes"), "byte count");
        } else if (arg == "--chunk-mib") {
            const auto mib = parse_u64(require_value("--chunk-mib"), "chunk MiB");
            if (mib == 0U || mib > 256U) {
                throw std::invalid_argument("chunk MiB must be in [1, 256]");
            }
            options.bitstream.chunk_bytes = static_cast<std::size_t>(mib * 1024U * 1024U);
        } else if (arg == "--pattern") {
            options.bitstream.pattern = jarvisx::parse_stream_pattern(require_value("--pattern"));
        } else if (arg == "--seed") {
            const auto seed = parse_u64(require_value("--seed"), "seed");
            options.engine.seed = seed;
            options.bitstream.seed = seed;
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Dr Moagi 1 GiB 3D AE/AD Bit Iterations Processor\n\n"
                << "No arguments: process the canonical 1 GiB / 2048^3-bit volume\n"
                << "and execute one 6400-panel x 1000-slot geometric AE/AD macrocycle.\n\n"
                << "Options:\n"
                << "  --cycles N       6400-panel AE/AD macrocycles (default 1)\n"
                << "  --iterations N   programmable iterations/panel (default 1000)\n"
                << "  --workers N      bounded physical worker threads\n"
                << "  --bytes N        bitstream bytes, <= 1073741824\n"
                << "  --chunk-mib N    reusable stream chunk MiB (default 8)\n"
                << "  --pattern P      sparse3d|checker3d|zero|random\n"
                << "  --seed N         deterministic seed\n"
                << "  --quiet          compact output\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    options.engine.validate();
    options.bitstream.validate();
    return options;
}

void print_banner(const Options& options) {
    const auto logical_slots =
        static_cast<std::uint64_t>(jarvisx::dm6400::Geometry::panels) *
        static_cast<std::uint64_t>(options.engine.programmable_iterations);

    std::cout
        << "Dr Moagi 1 GiB 3D Auto-Encoding/Decoding Bit Iterations Processor\n"
        << "control lattice : 20x20x16 = 6400 logical panels\n"
        << "clusters        : 100 x 64 panels\n"
        << "program depth   : " << options.engine.programmable_iterations << " slots/panel\n"
        << "logical slots   : " << logical_slots << " per macrocycle\n"
        << "bit volume      : " << options.bitstream.target_bytes << " bytes\n"
        << "canonical full  : 2048x2048x2048 bits = 1 GiB\n"
        << "inward loop     : 6400->800->100->18->4->1->4->18->100->800->6400\n"
        << "pipeline        : Encode->Coordinate->Refine->Decode->Verify->Correct->Remember->Recur\n\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        if (!options.quiet) {
            print_banner(options);
        }

        jarvisx::dm6400::DrMoagi6400x1000Engine engine(options.engine);
        jarvisx::dm6400::CycleReceipt final_receipt{};
        for (int cycle = 0; cycle < options.engine.cycles; ++cycle) {
            final_receipt = engine.step();
        }

        const auto bit_metrics = jarvisx::run_stream_1gib(options.bitstream);

        const bool pass =
            bit_metrics.exact_round_trip &&
            bit_metrics.codec_fixed_point &&
            std::isfinite(final_receipt.reconstruction_mse) &&
            std::isfinite(final_receipt.fixed_point_residual);

        std::cout << std::fixed << std::setprecision(6);
        std::cout
            << "AEAD cycle        : " << final_receipt.cycle << '\n'
            << "logical slots     : " << final_receipt.logical_program_slots << '\n'
            << "executed ops      : " << final_receipt.executed_program_ops << '\n'
            << "active panels     : " << final_receipt.active_panels << '\n'
            << "reconstruction MSE: " << final_receipt.reconstruction_mse << '\n'
            << "fixed-point resid.: " << final_receipt.fixed_point_residual << '\n'
            << "bit bytes         : " << bit_metrics.processed_bytes << '\n'
            << "logical 512b vecs : " << bit_metrics.logical_vectors_512 << '\n'
            << "compression ratio : " << bit_metrics.compression_ratio << '\n'
            << "bit throughput    : " << bit_metrics.throughput_gbps << " Gbps\n"
            << "exact round-trip  : " << (bit_metrics.exact_round_trip ? "PASS" : "FAIL") << '\n'
            << "codec fixed point : " << (bit_metrics.codec_fixed_point ? "PASS" : "FAIL") << '\n'
            << "system status     : " << (pass ? "PASS" : "FAIL") << '\n';

        return pass ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "DrMoagi 1 GiB bit engine failure: " << error.what() << '\n';
        return 1;
    }
}
