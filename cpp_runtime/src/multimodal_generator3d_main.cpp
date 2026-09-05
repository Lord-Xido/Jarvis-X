#include "jarvisx/multimodal_generator3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    jarvisx::mm3d::Modality source{jarvisx::mm3d::Modality::Text};
    jarvisx::mm3d::Modality target{jarvisx::mm3d::Modality::Image};
    std::string text{"Jarvis X 3D multimodal generation"};
    std::string prompt{"translate through a shared three-dimensional latent field"};
    std::filesystem::path input_path;
    std::filesystem::path output_dir{".jarvisx-mm3d"};
    std::size_t edge{8U};
    std::size_t channels{4U};
    std::size_t width{0U};
    std::size_t height{0U};
    std::size_t frames{1U};
    std::size_t train_steps{0U};
    std::uint32_t sample_rate{44100U};
    std::uint64_t seed{0x4D4F4147493344ULL};
    float conditioning_mix{0.85F};
    bool quantized{false};
    bool unconditional{false};
    bool quiet{false};
};

std::size_t parse_size(const std::string& value, const char* name) {
    std::size_t consumed = 0U;
    const unsigned long long parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) throw std::invalid_argument(std::string(name) + " must be an integer");
    return static_cast<std::size_t>(parsed);
}

float parse_float(const std::string& value, const char* name) {
    std::size_t consumed = 0U;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
    return parsed;
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
            ++i;
            return argv[i];
        };
        if (arg == "--source") options.source = jarvisx::mm3d::parse_modality(require_value("--source"));
        else if (arg == "--target") options.target = jarvisx::mm3d::parse_modality(require_value("--target"));
        else if (arg == "--text") options.text = require_value("--text");
        else if (arg == "--prompt") options.prompt = require_value("--prompt");
        else if (arg == "--input") options.input_path = require_value("--input");
        else if (arg == "--output-dir") options.output_dir = require_value("--output-dir");
        else if (arg == "--edge") options.edge = parse_size(require_value("--edge"), "--edge");
        else if (arg == "--channels") options.channels = parse_size(require_value("--channels"), "--channels");
        else if (arg == "--width") options.width = parse_size(require_value("--width"), "--width");
        else if (arg == "--height") options.height = parse_size(require_value("--height"), "--height");
        else if (arg == "--frames") options.frames = parse_size(require_value("--frames"), "--frames");
        else if (arg == "--train-steps") options.train_steps = parse_size(require_value("--train-steps"), "--train-steps");
        else if (arg == "--sample-rate") options.sample_rate = static_cast<std::uint32_t>(parse_size(require_value("--sample-rate"), "--sample-rate"));
        else if (arg == "--seed") options.seed = static_cast<std::uint64_t>(parse_size(require_value("--seed"), "--seed"));
        else if (arg == "--mix") options.conditioning_mix = parse_float(require_value("--mix"), "--mix");
        else if (arg == "--quantized") options.quantized = true;
        else if (arg == "--unconditional") options.unconditional = true;
        else if (arg == "--quiet") options.quiet = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Jarvis-X 3D multimodal generator\n"
                << "  --source text|image|audio|video|volume3d|generic\n"
                << "  --target text|image|audio|video|volume3d|generic\n"
                << "  --text TEXT | --input FILE\n"
                << "  --prompt TEXT --mix 0..1 --seed N\n"
                << "  --edge N --channels N --train-steps N --quantized\n"
                << "  --width N --height N --frames N --sample-rate N\n"
                << "  --unconditional --output-dir PATH --quiet\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return options;
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("unable to open input: " + path.string());
    stream.seekg(0, std::ios::end);
    const std::streamoff length = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (length < 0) throw std::runtime_error("unable to measure input file");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("failed while reading input file");
    }
    return bytes;
}

jarvisx::mm3d::MediaPacket build_packet(const Options& options) {
    jarvisx::mm3d::MediaPacket packet;
    packet.modality = options.source;
    packet.width = options.width;
    packet.height = options.height;
    packet.frames = options.frames;
    packet.sample_rate = options.sample_rate;

    if (options.source == jarvisx::mm3d::Modality::Text) {
        if (!options.input_path.empty()) {
            const auto bytes = read_bytes(options.input_path);
            packet.text.assign(bytes.begin(), bytes.end());
        } else {
            packet.text = options.text;
        }
        return packet;
    }

    std::vector<std::uint8_t> bytes;
    if (!options.input_path.empty()) {
        bytes = read_bytes(options.input_path);
    } else {
        const std::size_t demo_count = 4096U;
        bytes.resize(demo_count);
        std::uint64_t state = options.seed == 0ULL ? 1ULL : options.seed;
        for (std::size_t i = 0U; i < demo_count; ++i) {
            state ^= state << 13U;
            state ^= state >> 7U;
            state ^= state << 17U;
            const double phase = static_cast<double>(i) * 0.03125;
            const int wave = static_cast<int>(std::lround(127.5 + 86.0 * std::sin(phase)));
            const int noise = static_cast<int>(state & 0x1FULL) - 16;
            bytes[i] = static_cast<std::uint8_t>(std::clamp(wave + noise, 0, 255));
        }
    }
    packet.samples.reserve(bytes.size());
    for (std::uint8_t byte : bytes) {
        packet.samples.push_back(static_cast<float>(byte) / 127.5F - 1.0F);
    }

    if (options.source == jarvisx::mm3d::Modality::Image) {
        if (packet.width == 0U || packet.height == 0U) {
            const std::size_t side = static_cast<std::size_t>(std::sqrt(static_cast<double>(packet.samples.size())));
            packet.width = std::max<std::size_t>(1U, side);
            packet.height = std::max<std::size_t>(1U, side);
            packet.samples.resize(packet.width * packet.height);
        }
    } else if (options.source == jarvisx::mm3d::Modality::Video) {
        if (packet.width == 0U) packet.width = 32U;
        if (packet.height == 0U) packet.height = 32U;
        if (packet.frames == 0U) packet.frames = 1U;
        const std::size_t needed = packet.width * packet.height * packet.frames;
        if (packet.samples.size() < needed) {
            const std::vector<float> original = packet.samples;
            packet.samples.resize(needed);
            for (std::size_t i = 0U; i < needed; ++i) {
                packet.samples[i] = original.empty() ? 0.0F : original[i % original.size()];
            }
        } else {
            packet.samples.resize(needed);
        }
    } else if (options.source == jarvisx::mm3d::Modality::Volume3D) {
        std::size_t side = options.width;
        if (side == 0U) {
            side = static_cast<std::size_t>(std::cbrt(static_cast<double>(packet.samples.size())));
            side = std::max<std::size_t>(1U, side);
        }
        packet.width = side;
        packet.height = options.height == 0U ? side : options.height;
        packet.depth = side;
        const std::size_t needed = packet.width * packet.height * packet.depth;
        packet.samples.resize(needed, 0.0F);
    }
    return packet;
}

std::uint8_t to_u8(float value) noexcept {
    const float normalized = std::clamp((value + 1.0F) * 127.5F, 0.0F, 255.0F);
    return static_cast<std::uint8_t>(std::lround(normalized));
}

void write_pgm(const std::filesystem::path& path, const std::vector<float>& samples,
               std::size_t width, std::size_t height, std::size_t offset = 0U) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("unable to create " + path.string());
    stream << "P5\n" << width << ' ' << height << "\n255\n";
    const std::size_t count = width * height;
    for (std::size_t i = 0U; i < count; ++i) {
        const std::size_t index = offset + i;
        const std::uint8_t byte = index < samples.size() ? to_u8(samples[index]) : 0U;
        stream.write(reinterpret_cast<const char*>(&byte), 1);
    }
}

void write_u16_le(std::ofstream& stream, std::uint16_t value) {
    const std::array<std::uint8_t, 2U> bytes{
        static_cast<std::uint8_t>(value & 0xFFU),
        static_cast<std::uint8_t>((value >> 8U) & 0xFFU)};
    stream.write(reinterpret_cast<const char*>(bytes.data()), 2);
}

void write_u32_le(std::ofstream& stream, std::uint32_t value) {
    const std::array<std::uint8_t, 4U> bytes{
        static_cast<std::uint8_t>(value & 0xFFU),
        static_cast<std::uint8_t>((value >> 8U) & 0xFFU),
        static_cast<std::uint8_t>((value >> 16U) & 0xFFU),
        static_cast<std::uint8_t>((value >> 24U) & 0xFFU)};
    stream.write(reinterpret_cast<const char*>(bytes.data()), 4);
}

void write_wav(const std::filesystem::path& path, const std::vector<float>& samples,
               std::uint32_t sample_rate) {
    if (samples.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max() / 2U)) {
        throw std::runtime_error("audio output exceeds WAV32 size limit");
    }
    const std::uint32_t data_bytes = static_cast<std::uint32_t>(samples.size() * 2U);
    std::ofstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("unable to create " + path.string());
    stream.write("RIFF", 4);
    write_u32_le(stream, 36U + data_bytes);
    stream.write("WAVEfmt ", 8);
    write_u32_le(stream, 16U);
    write_u16_le(stream, 1U);
    write_u16_le(stream, 1U);
    write_u32_le(stream, sample_rate);
    write_u32_le(stream, sample_rate * 2U);
    write_u16_le(stream, 2U);
    write_u16_le(stream, 16U);
    stream.write("data", 4);
    write_u32_le(stream, data_bytes);
    for (float value : samples) {
        const float clipped = std::clamp(value, -1.0F, 1.0F);
        const std::int16_t pcm = static_cast<std::int16_t>(std::lround(clipped * 32767.0F));
        write_u16_le(stream, static_cast<std::uint16_t>(pcm));
    }
}

void write_volume_obj(const std::filesystem::path& path,
                      const jarvisx::mm3d::GeneratedMedia& media) {
    std::ofstream stream(path);
    if (!stream) throw std::runtime_error("unable to create " + path.string());
    stream << "# Jarvis-X MM3D generated volume point cloud\n";
    for (std::size_t z = 0U; z < media.depth; ++z) {
        for (std::size_t y = 0U; y < media.height; ++y) {
            for (std::size_t x = 0U; x < media.width; ++x) {
                const std::size_t index = (z * media.height + y) * media.width + x;
                if (index >= media.samples.size() || media.samples[index] <= 0.0F) continue;
                stream << "v " << x << ' ' << y << ' ' << z << '\n';
            }
        }
    }
}

void write_output(const std::filesystem::path& directory,
                  const jarvisx::mm3d::GeneratedMedia& media) {
    std::filesystem::create_directories(directory);
    using jarvisx::mm3d::Modality;
    if (media.modality == Modality::Text) {
        std::ofstream stream(directory / "generated.txt");
        stream << media.text << '\n';
    } else if (media.modality == Modality::Image) {
        write_pgm(directory / "generated.pgm", media.samples, media.width, media.height);
    } else if (media.modality == Modality::Audio) {
        write_wav(directory / "generated.wav", media.samples, media.sample_rate);
    } else if (media.modality == Modality::Video) {
        const std::filesystem::path frames_dir = directory / "frames";
        std::filesystem::create_directories(frames_dir);
        const std::size_t stride = media.width * media.height;
        for (std::size_t frame = 0U; frame < media.frames; ++frame) {
            std::ostringstream name;
            name << "frame-" << std::setw(4) << std::setfill('0') << frame << ".pgm";
            write_pgm(frames_dir / name.str(), media.samples, media.width, media.height,
                      frame * stride);
        }
        std::ofstream manifest(directory / "generated-video.txt");
        manifest << "format=pgm-sequence\nframes=" << media.frames
                 << "\nwidth=" << media.width << "\nheight=" << media.height << '\n';
    } else if (media.modality == Modality::Volume3D) {
        write_volume_obj(directory / "generated.obj", media);
    } else {
        std::ofstream stream(directory / "generated.f32", std::ios::binary);
        stream.write(reinterpret_cast<const char*>(media.samples.data()),
                     static_cast<std::streamsize>(media.samples.size() * sizeof(float)));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        jarvisx::mm3d::GeneratorConfig config;
        config.autoencoder.input_edge = options.edge;
        config.autoencoder.latent_channels = options.channels;
        config.autoencoder.seed = options.seed;
        config.validate();

        jarvisx::mm3d::MultimodalGenerator3D engine(config);
        const jarvisx::mm3d::MediaPacket source = build_packet(options);

        jarvisx::Autoencoder3DMetrics last_train{};
        for (std::size_t step = 0U; step < options.train_steps; ++step) {
            last_train = engine.train_step(source);
        }

        jarvisx::mm3d::GenerationMetrics metrics;
        jarvisx::mm3d::GeneratedMedia output;
        if (options.unconditional) {
            output = engine.generate(options.target, options.prompt, options.seed);
            metrics.source = source.modality;
            metrics.target = options.target;
            metrics.output_elements = output.samples.size();
        } else {
            output = engine.translate(source, options.target, options.prompt,
                                      options.conditioning_mix, options.quantized,
                                      options.seed, &metrics);
        }
        write_output(options.output_dir, output);

        std::ofstream telemetry(options.output_dir / "generation.csv");
        telemetry << "source,target,latent_elements,output_elements,latent_energy,output_energy,mix,train_steps,last_train_mse\n"
                  << jarvisx::mm3d::modality_name(metrics.source) << ','
                  << jarvisx::mm3d::modality_name(metrics.target) << ','
                  << metrics.latent_elements << ',' << metrics.output_elements << ','
                  << metrics.latent_energy << ',' << metrics.output_energy << ','
                  << metrics.conditioning_mix << ',' << options.train_steps << ','
                  << last_train.mse << '\n';

        if (!options.quiet) {
            std::cout << "Jarvis-X MM3D generated "
                      << jarvisx::mm3d::modality_name(options.target)
                      << " from " << jarvisx::mm3d::modality_name(options.source)
                      << " through a " << options.channels << "x"
                      << options.edge / 2U << "^3 latent field\n"
                      << "output: " << options.output_dir << '\n'
                      << "latent energy: " << metrics.latent_energy
                      << " output energy: " << metrics.output_energy << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "jarvisx-mm3d: " << error.what() << '\n';
        return 1;
    }
}
