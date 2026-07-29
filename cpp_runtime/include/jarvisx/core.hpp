#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jarvisx {

namespace fs = std::filesystem;

constexpr std::uint32_t kWorldEdge = 8192;
constexpr std::uint32_t kTileEdge = 8;
constexpr std::uint32_t kTileVolume = kTileEdge * kTileEdge * kTileEdge;
constexpr std::uint32_t kTileAxis = kWorldEdge / kTileEdge;
constexpr std::size_t kMaxPacketBytes = 1U << 20U;
constexpr float kEpsilon = 1.0e-6F;

static_assert(kWorldEdge % kTileEdge == 0, "invalid lattice geometry");
static_assert((kWorldEdge & (kWorldEdge - 1U)) == 0, "edge must be power of two");

float clampf(float value, float low, float high) noexcept {
    return std::max(low, std::min(value, high));
}

std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

float signed_unit(std::uint64_t key) noexcept {
    const std::uint64_t bits = mix64(key);
    const double unit = static_cast<double>(bits >> 11U) /
                        static_cast<double>(1ULL << 53U);
    return static_cast<float>(unit * 2.0 - 1.0);
}

std::uint32_t crc32(const std::vector<std::uint8_t>& bytes) noexcept {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (const std::uint8_t byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = static_cast<std::uint32_t>(
                -static_cast<std::int32_t>(crc & 1U));
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

struct Vec3u {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
};

struct Cell {
    std::int8_t latent{};
    std::int8_t prediction{};
    std::int8_t residual{};
    std::uint8_t modality{};
    std::uint16_t generation{};
    std::uint16_t flags{};
};

struct Tile {
    std::array<Cell, kTileVolume> cells{};
};

class SparseLattice {
public:
    Cell read(const Vec3u& point) const {
        const auto [key, index] = locate(point);
        const auto found = tiles_.find(key);
        return found == tiles_.end() ? Cell{} : found->second.cells[index];
    }

    void write(const Vec3u& point, const Cell& cell) {
        const auto [key, index] = locate(point);
        tiles_[key].cells[index] = cell;
    }

    std::size_t tile_count() const noexcept { return tiles_.size(); }

    std::uint64_t estimated_bytes() const noexcept {
        return static_cast<std::uint64_t>(tiles_.size()) * sizeof(Tile);
    }

private:
    std::unordered_map<std::uint64_t, Tile> tiles_;

    static std::pair<std::uint64_t, std::uint32_t> locate(const Vec3u& p) {
        if (p.x >= kWorldEdge || p.y >= kWorldEdge || p.z >= kWorldEdge) {
            throw std::out_of_range("coordinate outside 8192^3 lattice");
        }
        const std::uint32_t tx = p.x / kTileEdge;
        const std::uint32_t ty = p.y / kTileEdge;
        const std::uint32_t tz = p.z / kTileEdge;
        const std::uint64_t key =
            static_cast<std::uint64_t>(tx) +
            static_cast<std::uint64_t>(kTileAxis) *
                (static_cast<std::uint64_t>(ty) +
                 static_cast<std::uint64_t>(kTileAxis) * tz);
        const std::uint32_t lx = p.x % kTileEdge;
        const std::uint32_t ly = p.y % kTileEdge;
        const std::uint32_t lz = p.z % kTileEdge;
        return {key, lx + kTileEdge * (ly + kTileEdge * lz)};
    }
};

enum class Modality : std::uint8_t {
    Text = 1,
    Image = 2,
    Audio = 3,
    Video = 4,
    Binary = 5,
    Sensor = 6
};

struct Packet {
    Modality modality{Modality::Binary};
    std::vector<std::uint8_t> bytes;
    std::uint32_t checksum{};

    void seal() noexcept { checksum = crc32(bytes); }
    bool valid() const noexcept { return checksum == crc32(bytes); }
};

Packet text_packet(const std::string& text) {
    Packet packet;
    packet.modality = Modality::Text;
    packet.bytes.assign(text.begin(), text.end());
    packet.seal();
    return packet;
}

std::optional<Packet> file_packet(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;

    input.seekg(0, std::ios::end);
    const std::streamoff end = input.tellg();
    if (end < 0) return std::nullopt;
    input.seekg(0, std::ios::beg);

    Packet packet;
    packet.modality = Modality::Binary;
    packet.bytes.resize(std::min<std::size_t>(
        static_cast<std::size_t>(end), kMaxPacketBytes));
    input.read(reinterpret_cast<char*>(packet.bytes.data()),
               static_cast<std::streamsize>(packet.bytes.size()));
    packet.bytes.resize(static_cast<std::size_t>(input.gcount()));
    packet.seal();
    return packet;
}

std::vector<float> extract_features(const Packet& packet, std::size_t width) {
    if (!packet.valid()) throw std::runtime_error("packet CRC mismatch");
    if (width < 8) throw std::invalid_argument("feature width below 8");

    std::vector<float> features(width, 0.0F);
    if (packet.bytes.empty()) return features;

    double mean = 0.0;
    double square = 0.0;
    for (std::size_t i = 0; i < packet.bytes.size(); ++i) {
        const float x = static_cast<float>(packet.bytes[i]) / 127.5F - 1.0F;
        const std::uint64_t h = mix64(
            (static_cast<std::uint64_t>(i) << 8U) ^
            packet.bytes[i] ^
            (static_cast<std::uint64_t>(packet.modality) << 56U));
        features[h % width] += x;
        features[(h >> 19U) % width] += x * x - 0.333333F;
        mean += x;
        square += static_cast<double>(x) * x;
    }

    const double count = static_cast<double>(packet.bytes.size());
    features[0] += static_cast<float>(mean / count);
    features[1] += static_cast<float>(square / count);
    features[2] += static_cast<float>(std::log1p(count) / 16.0);
    features[3] += static_cast<float>(packet.modality) / 8.0F;

    double norm2 = 0.0;
    for (const float value : features) norm2 += value * value;
    const float inverse = 1.0F / std::sqrt(static_cast<float>(norm2) + kEpsilon);
    for (float& value : features) value *= inverse;
    return features;
}

std::int8_t quantize_q3(float value) noexcept {
    const int q = static_cast<int>(std::lround(clampf(value, -1.0F, 1.0F) * 3.5F));
    return static_cast<std::int8_t>(std::max(-4, std::min(3, q)));
}

float dequantize_q3(std::int8_t value) noexcept {
    return clampf(static_cast<float>(value) / 3.5F, -1.0F, 1.0F);
}

struct Genome {
    std::uint32_t version{1};
    std::uint64_t generation{};
    std::size_t feature_dim{128};
    std::size_t latent_dim{64};
    std::uint16_t iterations{12};
    std::uint16_t diffusion_radius{1};
    std::uint16_t learning_units{80};
    std::uint16_t omega_units{40};
    std::uint16_t max_mse_units{2500};
    std::uint16_t max_energy_units{8500};
    std::uint16_t min_coherence_units{800};
    std::uint64_t seed{0x4A415256495358ULL};

    void clamp() noexcept {
        feature_dim = std::max<std::size_t>(32, std::min<std::size_t>(512, feature_dim));
        latent_dim = std::max<std::size_t>(8, std::min<std::size_t>(256, latent_dim));
        latent_dim = std::min(latent_dim, feature_dim);
        iterations = static_cast<std::uint16_t>(std::max(2, std::min(128, int(iterations))));
        diffusion_radius = static_cast<std::uint16_t>(
            std::max(1, std::min(32, int(diffusion_radius))));
        learning_units = static_cast<std::uint16_t>(
            std::max(1, std::min(400, int(learning_units))));
        omega_units = static_cast<std::uint16_t>(
            std::max(1, std::min(300, int(omega_units))));
        max_mse_units = static_cast<std::uint16_t>(
            std::max(100, std::min(10000, int(max_mse_units))));
        max_energy_units = static_cast<std::uint16_t>(
            std::max(1000, std::min(10000, int(max_energy_units))));
        min_coherence_units = static_cast<std::uint16_t>(
            std::max(0, std::min(9000, int(min_coherence_units))));
    }

    std::string fingerprint() const {
        std::ostringstream out;
        out << std::hex << std::uppercase << mix64(
            seed ^ generation ^
            (static_cast<std::uint64_t>(feature_dim) << 32U) ^
            (static_cast<std::uint64_t>(latent_dim) << 16U) ^
            iterations ^ diffusion_radius ^ learning_units ^ omega_units);
        return out.str();
    }
};

enum class Op : std::uint8_t {
    Extract = 1,
    Encode = 2,
    Scatter = 3,
    Diffuse = 4,
    Decode = 5,
    Learn = 6,
    Project = 7,
    Commit = 8,
    Loop = 9,
    Halt = 0xFF
};

struct Instruction {
    std::uint64_t word{};

    static Instruction make(Op op, std::uint16_t a = 0,
                            std::uint16_t b = 0, std::uint16_t c = 0) noexcept {
        return { (static_cast<std::uint64_t>(op) << 56U) |
                 (static_cast<std::uint64_t>(a) << 32U) |
                 (static_cast<std::uint64_t>(b) << 16U) |
                 c };
    }

    Op op() const noexcept { return static_cast<Op>((word >> 56U) & 0xFFU); }
    std::uint16_t a() const noexcept { return (word >> 32U) & 0xFFFFU; }
    std::uint16_t b() const noexcept { return (word >> 16U) & 0xFFFFU; }
    std::uint16_t c() const noexcept { return word & 0xFFFFU; }
};

std::vector<Instruction> synthesize_rom(const Genome& g) {
    return {
        Instruction::make(Op::Extract),
        Instruction::make(Op::Encode),
        Instruction::make(Op::Scatter),
        Instruction::make(Op::Diffuse, g.diffusion_radius),
        Instruction::make(Op::Decode),
        Instruction::make(Op::Learn, g.learning_units, g.omega_units),
        Instruction::make(Op::Project, g.max_mse_units, g.max_energy_units,
                          g.min_coherence_units),
        Instruction::make(Op::Commit),
        Instruction::make(Op::Loop, g.iterations, 1),
        Instruction::make(Op::Halt)
    };
}

} // namespace jarvisx
