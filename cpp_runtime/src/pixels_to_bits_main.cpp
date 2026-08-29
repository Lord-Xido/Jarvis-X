#include "jarvisx/pixels_to_bits_codec.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::filesystem::path input{};
    std::filesystem::path encoded{"dr-moagi-pixels-to-bits.dmpb"};
    std::filesystem::path decoded{"dr-moagi-pixels-to-bits-decoded.ppm"};
    std::uint32_t width{64U};
    std::uint32_t height{64U};
    std::uint32_t fps{120U};
    std::uint8_t max_shift{4U};
    double max_mse{20.0};
    double min_psnr{47.0};
    bool lossless{};
    bool quiet{};
    bool report_8k120{true};
};

std::uint64_t parse_u64(const std::string& value, const char* flag) {
    std::size_t consumed = 0U;
    const std::uint64_t result = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) throw std::invalid_argument(std::string("invalid integer for ") + flag);
    return result;
}

double parse_double(const std::string& value, const char* flag) {
    std::size_t consumed = 0U;
    const double result = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(result)) {
        throw std::invalid_argument(std::string("invalid number for ") + flag);
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    auto next = [&](int& index, const char* flag) {
        if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value after ") + flag);
        return std::string(argv[++index]);
    };

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--input") {
            options.input = next(index, "--input");
        } else if (argument == "--encoded") {
            options.encoded = next(index, "--encoded");
        } else if (argument == "--decoded") {
            options.decoded = next(index, "--decoded");
        } else if (argument == "--width") {
            const auto value = parse_u64(next(index, "--width"), "--width");
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::out_of_range("width too large");
            options.width = static_cast<std::uint32_t>(value);
        } else if (argument == "--height") {
            const auto value = parse_u64(next(index, "--height"), "--height");
            if (value > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::out_of_range("height too large");
            options.height = static_cast<std::uint32_t>(value);
        } else if (argument == "--fps") {
            const auto value = parse_u64(next(index, "--fps"), "--fps");
            if (value == 0ULL || value > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::out_of_range("fps out of range");
            options.fps = static_cast<std::uint32_t>(value);
        } else if (argument == "--max-shift") {
            const auto value = parse_u64(next(index, "--max-shift"), "--max-shift");
            if (value > 8ULL) throw std::out_of_range("max-shift must be <= 8");
            options.max_shift = static_cast<std::uint8_t>(value);
        } else if (argument == "--max-mse") {
            options.max_mse = parse_double(next(index, "--max-mse"), "--max-mse");
        } else if (argument == "--min-psnr") {
            options.min_psnr = parse_double(next(index, "--min-psnr"), "--min-psnr");
        } else if (argument == "--lossless") {
            options.lossless = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--no-8k120-report") {
            options.report_8k120 = false;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: DrMoagi-Pixels-to-Bits [options]\n"
                << "  --input FILE       P6 PPM source (8-bit or <=10-bit maxval)\n"
                << "  --encoded FILE     output DM-PXBT v1 bitstream\n"
                << "  --decoded FILE     reconstructed 10-bit PPM\n"
                << "  --width N          generated gradient width (default 64)\n"
                << "  --height N         generated gradient height (default 64)\n"
                << "  --fps N            target frame rate for budget telemetry\n"
                << "  --lossless         exact 10-bit round-trip mode\n"
                << "  --max-shift N      maximum predictive quantization shift [0,8]\n"
                << "  --max-mse X        per-tile Lambda MSE ceiling\n"
                << "  --min-psnr X       per-tile Lambda PSNR floor in dB\n"
                << "  --no-8k120-report  suppress canonical 7680x4320@120 profile\n"
                << "  --quiet            suppress telemetry\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

std::string ppm_token(std::istream& input) {
    std::string token;
    char ch = '\0';
    while (input.get(ch)) {
        if (ch == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            token.push_back(ch);
            break;
        }
    }
    while (input.get(ch)) {
        if (ch == '#') {
            input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            break;
        }
        if (std::isspace(static_cast<unsigned char>(ch))) break;
        token.push_back(ch);
    }
    if (token.empty()) throw std::runtime_error("truncated PPM header");
    return token;
}

jarvisx::pixels::Frame10 read_ppm10(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input PPM: " + path.string());
    if (ppm_token(input) != "P6") throw std::runtime_error("only binary P6 PPM is supported");
    const std::uint64_t width64 = parse_u64(ppm_token(input), "PPM width");
    const std::uint64_t height64 = parse_u64(ppm_token(input), "PPM height");
    const std::uint64_t maxval64 = parse_u64(ppm_token(input), "PPM maxval");
    if (width64 == 0ULL || height64 == 0ULL ||
        width64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) ||
        height64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("PPM dimensions out of range");
    }
    if (maxval64 == 0ULL || maxval64 > 1023ULL) throw std::runtime_error("PPM maxval must be in [1,1023]");

    jarvisx::pixels::Frame10 frame;
    frame.width = static_cast<std::uint32_t>(width64);
    frame.height = static_cast<std::uint32_t>(height64);
    const std::uint64_t samples64 = width64 * height64 * 3ULL;
    if (samples64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::overflow_error("PPM too large");
    frame.rgb.resize(static_cast<std::size_t>(samples64));

    const bool wide = maxval64 > 255ULL;
    for (std::size_t i = 0U; i < frame.rgb.size(); ++i) {
        std::uint16_t source = 0U;
        if (wide) {
            const int high = input.get();
            const int low = input.get();
            if (high < 0 || low < 0) throw std::runtime_error("truncated PPM pixel data");
            source = static_cast<std::uint16_t>((static_cast<std::uint16_t>(high) << 8U) |
                                                static_cast<std::uint16_t>(low));
        } else {
            const int byte = input.get();
            if (byte < 0) throw std::runtime_error("truncated PPM pixel data");
            source = static_cast<std::uint16_t>(byte);
        }
        if (source > maxval64) throw std::runtime_error("PPM sample exceeds maxval");
        frame.rgb[i] = static_cast<std::uint16_t>(
            (static_cast<std::uint64_t>(source) * 1023ULL + maxval64 / 2ULL) / maxval64);
    }
    frame.validate();
    return frame;
}

void write_ppm10(const std::filesystem::path& path, const jarvisx::pixels::Frame10& frame) {
    frame.validate();
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create decoded PPM: " + path.string());
    output << "P6\n" << frame.width << ' ' << frame.height << "\n1023\n";
    for (const std::uint16_t value : frame.rgb) {
        const char high = static_cast<char>((value >> 8U) & 0xFFU);
        const char low = static_cast<char>(value & 0xFFU);
        output.put(high);
        output.put(low);
    }
    if (!output) throw std::runtime_error("failed writing decoded PPM");
}

void write_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create encoded stream: " + path.string());
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!output) throw std::runtime_error("failed writing encoded stream");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const jarvisx::pixels::Frame10 frame = options.input.empty()
            ? jarvisx::pixels::make_gradient_frame(options.width, options.height)
            : read_ppm10(options.input);

        jarvisx::pixels::CodecConfig config;
        config.lossless = options.lossless;
        config.max_shift = options.max_shift;
        config.max_tile_mse = options.lossless ? 0.0 : options.max_mse;
        config.min_tile_psnr_db = options.lossless ? 0.0 : options.min_psnr;
        config.fps = options.fps;

        jarvisx::pixels::PixelsToBitsCodec codec(config);
        const auto begin = std::chrono::steady_clock::now();
        const auto result = codec.encode(frame);
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();

        write_bytes(options.encoded, result.bitstream);
        write_ppm10(options.decoded, result.reconstructed);

        if (!options.quiet) {
            std::cout << "DM-vOmegaXi+ Pixels-to-Bits reference\n"
                      << "frame=" << frame.width << 'x' << frame.height
                      << " RGB10 fps_target=" << options.fps << '\n'
                      << "tiles=" << result.metrics.tiles
                      << " raw_bits=" << result.metrics.raw_bits
                      << " encoded_bits=" << result.metrics.encoded_bits
                      << " ratio=" << result.metrics.compression_ratio << "x\n"
                      << "mse=" << result.metrics.mse
                      << " psnr_db=" << result.metrics.psnr_db
                      << " hbar_visual=" << result.metrics.hbar_semantic_visual << '\n'
                      << "avg_shift=" << result.metrics.average_tile_shift
                      << " avg_vcl_entropy=" << result.metrics.average_vcl_entropy
                      << " avg_active_nodes=" << result.metrics.average_active_nodes
                      << " avg_container_delta=" << result.metrics.average_container_delta << '\n'
                      << "elapsed_ms=" << elapsed_ms
                      << " target_budget_ms=" << result.metrics.frame_budget_ms
                      << " software_meets_target=" << (elapsed_ms <= result.metrics.frame_budget_ms ? "yes" : "no") << '\n'
                      << "container_bytes=" << jarvisx::dm8kb::kContainerBytes
                      << " epoch=" << result.final_container.epoch()
                      << " accepted_receipt=" << (result.final_container.receipt().accepted ? "yes" : "no") << '\n'
                      << "encoded=" << options.encoded.string() << '\n'
                      << "decoded=" << options.decoded.string() << '\n';
            if (options.report_8k120) {
                const auto profile = jarvisx::pixels::uhd8k120_profile();
                std::cout << "8k120_raw_pixels=" << profile.pixels
                          << " raw_bits_per_frame=" << profile.raw_bits_per_frame
                          << " packed_bytes_per_frame=" << profile.packed_bytes_per_frame
                          << " packed_GBps=" << profile.packed_gbytes_per_second
                          << " frame_budget_ms=" << profile.frame_budget_ms << '\n';
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DrMoagi Pixels-to-Bits failure: " << error.what() << '\n';
        return 1;
    }
}
