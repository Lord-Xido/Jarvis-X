#include "jarvisx/intelligence_media_processor.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    std::filesystem::path input{};
    std::filesystem::path output{"dr-moagi-intelligence-media-output.bin"};
    std::filesystem::path bytecode{};
    jarvisx::media8::MediaModality modality{jarvisx::media8::MediaModality::Generic};
    std::size_t demo_bytes{4096U};
    std::uint64_t passes{1U};
    double max_mse{4096.0};
    bool fallback{true};
    bool quiet{};
};

std::uint64_t parse_u64(const std::string& value, const char* name) {
    std::size_t consumed = 0U;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) throw std::invalid_argument(std::string(name) + " must be an integer");
    return parsed;
}

double parse_double(const std::string& value, const char* name) {
    std::size_t consumed = 0U;
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
    return parsed;
}

jarvisx::media8::MediaModality parse_modality(const std::string& value) {
    using jarvisx::media8::MediaModality;
    if (value == "visual") return MediaModality::Visual;
    if (value == "audio") return MediaModality::Audio;
    if (value == "text") return MediaModality::Text;
    if (value == "generic") return MediaModality::Generic;
    throw std::invalid_argument("modality must be visual, audio, text, or generic");
}

Options parse_options(int argc, char** argv) {
    Options options;
    auto next = [&](int& index, const char* flag) -> std::string {
        if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + flag);
        return argv[++index];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") options.input = next(i, "--input");
        else if (arg == "--output") options.output = next(i, "--output");
        else if (arg == "--bytecode") options.bytecode = next(i, "--bytecode");
        else if (arg == "--modality") options.modality = parse_modality(next(i, "--modality"));
        else if (arg == "--demo-bytes") options.demo_bytes = static_cast<std::size_t>(parse_u64(next(i, "--demo-bytes"), "demo-bytes"));
        else if (arg == "--passes") options.passes = parse_u64(next(i, "--passes"), "passes");
        else if (arg == "--max-mse") options.max_mse = parse_double(next(i, "--max-mse"), "max-mse");
        else if (arg == "--no-fallback") options.fallback = false;
        else if (arg == "--quiet") options.quiet = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Dr Moagi Intelligence Media Processor\n"
                << "  --input PATH        arbitrary binary/media input\n"
                << "  --output PATH       processed output path\n"
                << "  --bytecode PATH     VCL-BVM-8 program (raw bytes)\n"
                << "  --modality NAME     visual|audio|text|generic\n"
                << "  --demo-bytes N      deterministic demo size when --input is absent\n"
                << "  --passes N          inward processing passes\n"
                << "  --max-mse X         Lambda output-quality ceiling\n"
                << "  --no-fallback       emit rejected candidate instead of original tile\n"
                << "  --quiet             compact output\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.passes == 0U) throw std::invalid_argument("passes must be positive");
    if (options.demo_bytes == 0U && options.input.empty()) {
        throw std::invalid_argument("demo-bytes must be positive without --input");
    }
    return options;
}

std::vector<std::uint8_t> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input: " + path.string());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>());
}

void write_binary(const std::filesystem::path& path,
                  const std::vector<std::uint8_t>& bytes) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output: " + path.string());
    if (!bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    if (!output) throw std::runtime_error("cannot write output: " + path.string());
}

std::vector<std::uint8_t> make_demo(std::size_t count,
                                    jarvisx::media8::MediaModality modality) {
    std::vector<std::uint8_t> bytes(count);
    for (std::size_t i = 0U; i < count; ++i) {
        const std::uint32_t x = static_cast<std::uint32_t>(i & 7U);
        const std::uint32_t y = static_cast<std::uint32_t>((i >> 3U) & 7U);
        const std::uint32_t z = static_cast<std::uint32_t>((i >> 6U) & 7U);
        std::uint32_t value = 0U;
        switch (modality) {
        case jarvisx::media8::MediaModality::Visual:
            value = (31U * x + 17U * y + 11U * z + static_cast<std::uint32_t>(i / 512U)) & 0xFFU;
            break;
        case jarvisx::media8::MediaModality::Audio:
            value = (128U + ((37U * static_cast<std::uint32_t>(i)) & 0x7FU)) & 0xFFU;
            break;
        case jarvisx::media8::MediaModality::Text: {
            static constexpr char text[] = "I AM = I DESCRIBE | DR MOAGI INTELLIGENCE MEDIA PROCESSOR | ";
            value = static_cast<std::uint8_t>(text[i % (sizeof(text) - 1U)]);
            break;
        }
        case jarvisx::media8::MediaModality::Generic:
            value = static_cast<std::uint32_t>((i * 73U + 19U) ^ (i >> 3U)) & 0xFFU;
            break;
        }
        bytes[i] = static_cast<std::uint8_t>(value);
    }
    return bytes;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::vector<std::uint8_t> bytes = options.input.empty()
            ? make_demo(options.demo_bytes, options.modality)
            : read_binary(options.input);

        jarvisx::media8::ProcessorConfig config;
        config.max_output_mse = options.max_mse;
        config.fallback_on_regression = options.fallback;

        std::vector<std::uint8_t> program = options.bytecode.empty()
            ? jarvisx::media8::default_media_program()
            : read_binary(options.bytecode);

        jarvisx::media8::DrMoagiIntelligenceMediaProcessor processor(config, program);
        jarvisx::media8::ProcessorMetrics last{};
        for (std::uint64_t pass = 0U; pass < options.passes; ++pass) {
            auto result = processor.process(bytes, options.modality);
            bytes = std::move(result.bytes);
            last = result.metrics;
            if (!options.quiet) {
                std::cout << "pass=" << (pass + 1U)
                          << " modality=" << jarvisx::media8::modality_name(last.modality)
                          << " tiles=" << last.tiles
                          << " accepted=" << last.accepted_tiles
                          << " rejected=" << last.rejected_tiles
                          << " candidate_mse=" << last.average_candidate_mse
                          << " output_mse=" << last.average_output_mse
                          << " hbar_semantic=" << last.hbar_semantic
                          << " entropy=" << last.average_entropy
                          << " active_nodes=" << last.average_active_nodes
                          << " theta_commits=" << last.evolution_commits
                          << " theta_rollbacks=" << last.evolution_rollbacks
                          << '\n';
            }
        }

        write_binary(options.output, bytes);
        if (options.quiet) {
            std::cout << "DMIMP OK bytes=" << bytes.size()
                      << " tiles=" << last.tiles
                      << " accepted=" << last.accepted_tiles
                      << " rejected=" << last.rejected_tiles
                      << " hbar=" << last.hbar_semantic << '\n';
        } else {
            std::cout << "output=" << options.output.string() << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dr-moagi-intelligence-media: " << error.what() << '\n';
        return 2;
    }
}
