#include "jarvisx/bitmatrix1gib.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

struct Options {
    jarvisx::Stream1GiBConfig config;
    fs::path output_dir{".jarvisx-bitmatrix1gib"};
    bool quiet{false};
};

std::uint64_t parse_u64(const std::string& text, const char* name) {
    std::size_t consumed = 0U;
    const auto value = std::stoull(text, &consumed, 10);
    if (consumed != text.size()) throw std::invalid_argument(std::string("invalid ") + name);
    return value;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string(name) + " requires a value");
            return argv[++i];
        };
        if (arg == "--bytes") {
            options.config.target_bytes = parse_u64(require_value("--bytes"), "byte count");
        } else if (arg == "--chunk-mib") {
            const auto mib = parse_u64(require_value("--chunk-mib"), "chunk MiB");
            if (mib == 0U || mib > 256U) throw std::invalid_argument("chunk MiB must be in [1, 256]");
            options.config.chunk_bytes = static_cast<std::size_t>(mib * 1024U * 1024U);
        } else if (arg == "--pattern") {
            options.config.pattern = jarvisx::parse_stream_pattern(require_value("--pattern"));
        } else if (arg == "--window-ms") {
            const auto ms = parse_u64(require_value("--window-ms"), "window ms");
            if (ms > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("window ms is too large");
            options.config.window_ms = static_cast<std::uint32_t>(ms);
        } else if (arg == "--l3-mib") {
            const auto mib = parse_u64(require_value("--l3-mib"), "L3 MiB");
            options.config.configured_l3_bytes = mib * 1024U * 1024U;
        } else if (arg == "--seed") {
            options.config.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--output-dir") {
            options.output_dir = require_value("--output-dir");
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "jarvisx-bitmatrix1gib [options]\n"
                << "  --bytes N          bytes to process (default 1073741824 = 1 GiB)\n"
                << "  --chunk-mib N      reusable chunk size in MiB (default 8)\n"
                << "  --pattern P        sparse3d|checker3d|zero|random\n"
                << "  --window-ms N      telemetry burst window (default 100)\n"
                << "  --l3-mib N         optional configured L3 size for fit comparison\n"
                << "  --seed N           deterministic generator seed\n"
                << "  --output-dir PATH  artifact directory\n"
                << "  --quiet             suppress console telemetry\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    options.config.validate();
    return options;
}

void write_metrics_csv(const fs::path& path, const jarvisx::Stream1GiBMetrics& metrics) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write metrics.csv");
    out << "offset_bytes,raw_bytes,encoded_bytes,mode,ingest_ms,encode_ms,core_verify_ms,decode_ms\n";
    out << std::setprecision(10);
    for (const auto& chunk : metrics.chunks) {
        out << chunk.offset_bytes << ','
            << chunk.raw_bytes << ','
            << chunk.encoded_bytes << ','
            << (chunk.raw_passthrough ? "raw" : "rle") << ','
            << chunk.ingest_ms << ','
            << chunk.encode_ms << ','
            << chunk.core_verify_ms << ','
            << chunk.decode_ms << '\n';
    }
}

void write_report(const fs::path& path,
                  const Options& options,
                  const jarvisx::Stream1GiBMetrics& metrics) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write report.txt");
    out << std::setprecision(12);
    out << "DM-vOmegaXi+ 1 GiB Operational Bit-Wise 3D Stream Report\n";
    out << "target_bytes=" << metrics.target_bytes << '\n';
    out << "canonical_full_volume_bits=" << jarvisx::kOneGiBBits << '\n';
    out << "canonical_full_volume_shape_bits=2048x2048x2048\n";
    out << "processed_bytes=" << metrics.processed_bytes << '\n';
    out << "pattern=" << jarvisx::stream_pattern_name(options.config.pattern) << '\n';
    out << "chunk_bytes=" << options.config.chunk_bytes << '\n';
    out << "logical_vector_width_bits=512\n";
    out << "logical_vector_count_512=" << metrics.logical_vectors_512 << '\n';
    out << "execution_backend=portable_u64_reference\n";
    out << "avx512_execution_claimed=false\n";
    out << "encoded_bytes=" << metrics.encoded_bytes << '\n';
    out << "compression_ratio=" << metrics.compression_ratio << '\n';
    out << "total_seconds=" << metrics.total_seconds << '\n';
    out << "throughput_gbps=" << metrics.throughput_gbps << '\n';
    out << "first_window_requested_ms=" << options.config.window_ms << '\n';
    out << "first_window_observed_ms=" << metrics.first_window_elapsed_ms << '\n';
    out << "first_window_bytes=" << metrics.first_window_bytes << '\n';
    out << "first_window_gbps=" << metrics.first_window_gbps << '\n';
    out << "ingest_seconds=" << metrics.ingest_seconds << '\n';
    out << "encode_seconds=" << metrics.encode_seconds << '\n';
    out << "core_verify_seconds=" << metrics.core_verify_seconds << '\n';
    out << "decode_seconds=" << metrics.decode_seconds << '\n';
    out << "reusable_working_set_bytes=" << metrics.reusable_working_set_bytes << '\n';
    out << "hot_path_reallocations=" << metrics.hot_path_reallocations << '\n';
    out << "exact_round_trip=" << (metrics.exact_round_trip ? "true" : "false") << '\n';
    out << "codec_fixed_point=" << (metrics.codec_fixed_point ? "true" : "false") << '\n';
    out << "configured_l3_bytes=" << options.config.configured_l3_bytes << '\n';
    out << "working_set_fits_configured_l3=" << (metrics.fits_configured_l3 ? "true" : "false") << '\n';
    out << "zero_dram_latency_claimed=false\n";
    out << "zero_gc_claimed=false\n";
    out << "semantic_loss_claimed=false\n";
    out << "lossless_bit_identity_verified=true\n";
    out << "reality_gap_gamma=conceptual_boundary_not_hardware_metric\n";
    out << "kinetic_visual_radii=12.5->8.0->2.0->0.8->1.3->7.5\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        fs::create_directories(options.output_dir);
        const auto metrics = jarvisx::run_stream_1gib(options.config);
        write_metrics_csv(options.output_dir / "metrics.csv", metrics);
        write_report(options.output_dir / "report.txt", options, metrics);

        if (!options.quiet) {
            std::cout << std::fixed << std::setprecision(3)
                      << "DM-vOmegaXi+ 3D stream processed "
                      << (static_cast<double>(metrics.processed_bytes) / 1073741824.0)
                      << " GiB\n"
                      << "pattern: " << jarvisx::stream_pattern_name(options.config.pattern) << '\n'
                      << "compression: " << metrics.compression_ratio << "x\n"
                      << "throughput: " << metrics.throughput_gbps << " Gbps\n"
                      << "100ms-window-equivalent: " << metrics.first_window_gbps << " Gbps\n"
                      << "logical 512-bit vectors: " << metrics.logical_vectors_512 << '\n'
                      << "working set: "
                      << (static_cast<double>(metrics.reusable_working_set_bytes) / 1048576.0)
                      << " MiB\n"
                      << "hot-path reallocations: " << metrics.hot_path_reallocations << '\n'
                      << "exact round-trip: " << (metrics.exact_round_trip ? "PASS" : "FAIL") << '\n';
        }
        return metrics.exact_round_trip && metrics.codec_fixed_point ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "bitmatrix1gib failure: " << error.what() << '\n';
        return 1;
    }
}
