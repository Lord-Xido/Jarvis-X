#pragma once

#include "jarvisx/dm8kb_container.hpp"
#include "jarvisx/intelligence_media_processor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jarvisx::pixels {

constexpr std::uint16_t kMaxPixel10 = 1023U;
constexpr std::size_t kChannels = 3U;
constexpr std::size_t kPixelTileEdge = 8U;
constexpr std::size_t kPixelsPerTile = kPixelTileEdge * kPixelTileEdge;

struct Frame10 {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint16_t> rgb;

    void validate() const {
        if (width == 0U || height == 0U) throw std::invalid_argument("pixel frame dimensions must be positive");
        const std::uint64_t samples = static_cast<std::uint64_t>(width) *
                                      static_cast<std::uint64_t>(height) *
                                      static_cast<std::uint64_t>(kChannels);
        if (samples > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("pixel frame exceeds host addressable size");
        }
        if (rgb.size() != static_cast<std::size_t>(samples)) {
            throw std::invalid_argument("pixel frame RGB sample count mismatch");
        }
        for (const std::uint16_t value : rgb) {
            if (value > kMaxPixel10) throw std::invalid_argument("pixel sample exceeds 10-bit range");
        }
    }

    std::uint16_t at(std::uint32_t x, std::uint32_t y, std::size_t channel) const {
        if (x >= width || y >= height || channel >= kChannels) {
            throw std::out_of_range("pixel coordinate outside frame");
        }
        const std::size_t index =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)) * kChannels + channel;
        return rgb[index];
    }

    void set(std::uint32_t x, std::uint32_t y, std::size_t channel, std::uint16_t value) {
        if (x >= width || y >= height || channel >= kChannels || value > kMaxPixel10) {
            throw std::out_of_range("invalid 10-bit pixel write");
        }
        const std::size_t index =
            (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
             static_cast<std::size_t>(x)) * kChannels + channel;
        rgb[index] = value;
    }
};

struct RawProfile {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t fps{};
    std::uint64_t pixels{};
    std::uint64_t raw_bits_per_frame{};
    std::uint64_t packed_bytes_per_frame{};
    double packed_gbytes_per_second{};
    double frame_budget_ms{};
};

inline RawProfile raw_profile(std::uint32_t width, std::uint32_t height, std::uint32_t fps) {
    if (width == 0U || height == 0U || fps == 0U) throw std::invalid_argument("profile dimensions/fps must be positive");
    RawProfile profile;
    profile.width = width;
    profile.height = height;
    profile.fps = fps;
    profile.pixels = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    profile.raw_bits_per_frame = profile.pixels * 30ULL;
    profile.packed_bytes_per_frame = (profile.raw_bits_per_frame + 7ULL) / 8ULL;
    profile.packed_gbytes_per_second =
        static_cast<double>(profile.packed_bytes_per_frame) * static_cast<double>(fps) / 1.0e9;
    profile.frame_budget_ms = 1000.0 / static_cast<double>(fps);
    return profile;
}

inline RawProfile uhd8k120_profile() { return raw_profile(7680U, 4320U, 120U); }

class ByteWriter {
public:
    void u8(std::uint8_t value) { bytes_.push_back(value); }

    void u16(std::uint16_t value) {
        u8(static_cast<std::uint8_t>(value & 0xFFU));
        u8(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    }

    void u32(std::uint32_t value) {
        for (std::size_t i = 0U; i < 4U; ++i) {
            u8(static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFU));
        }
    }

    void u64(std::uint64_t value) {
        for (std::size_t i = 0U; i < 8U; ++i) {
            u8(static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFULL));
        }
    }

    void append(const std::vector<std::uint8_t>& values) {
        bytes_.insert(bytes_.end(), values.begin(), values.end());
    }

    void append(const std::uint8_t* values, std::size_t size) {
        if (size != 0U) bytes_.insert(bytes_.end(), values, values + size);
    }

    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }
    std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
    std::vector<std::uint8_t> bytes_;
};

class ByteReader {
public:
    explicit ByteReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    std::uint8_t u8() {
        require(1U);
        return bytes_[offset_++];
    }

    std::uint16_t u16() {
        const std::uint16_t a = static_cast<std::uint16_t>(u8());
        const std::uint16_t b = static_cast<std::uint16_t>(u8());
        return static_cast<std::uint16_t>(a | static_cast<std::uint16_t>(b << 8U));
    }

    std::uint32_t u32() {
        std::uint32_t value = 0U;
        for (std::size_t i = 0U; i < 4U; ++i) value |= static_cast<std::uint32_t>(u8()) << (8U * i);
        return value;
    }

    std::uint64_t u64() {
        std::uint64_t value = 0ULL;
        for (std::size_t i = 0U; i < 8U; ++i) value |= static_cast<std::uint64_t>(u8()) << (8U * i);
        return value;
    }

    std::vector<std::uint8_t> bytes(std::size_t count) {
        require(count);
        std::vector<std::uint8_t> result(
            bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
            bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return result;
    }

    std::size_t offset() const noexcept { return offset_; }
    std::size_t remaining() const noexcept { return bytes_.size() - offset_; }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_{};

    void require(std::size_t count) const {
        if (count > bytes_.size() - offset_) throw std::runtime_error("truncated pixels-to-bits stream");
    }
};

class BitWriter {
public:
    void write(std::uint32_t value, std::uint8_t bits) {
        if (bits > 32U) throw std::invalid_argument("bit width above 32");
        for (std::uint8_t bit = 0U; bit < bits; ++bit) {
            if ((used_bits_ & 7U) == 0U) bytes_.push_back(0U);
            const std::uint8_t one = static_cast<std::uint8_t>((value >> bit) & 1U);
            bytes_.back() = static_cast<std::uint8_t>(
                bytes_.back() | static_cast<std::uint8_t>(one << (used_bits_ & 7U)));
            ++used_bits_;
        }
    }

    const std::vector<std::uint8_t>& bytes() const noexcept { return bytes_; }

private:
    std::vector<std::uint8_t> bytes_;
    std::uint32_t used_bits_{};
};

class BitReader {
public:
    explicit BitReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

    std::uint32_t read(std::uint8_t bits) {
        if (bits > 32U) throw std::invalid_argument("bit width above 32");
        if (static_cast<std::uint64_t>(bit_offset_) + static_cast<std::uint64_t>(bits) >
            static_cast<std::uint64_t>(bytes_.size()) * 8ULL) {
            throw std::runtime_error("truncated packed residual bits");
        }
        std::uint32_t value = 0U;
        for (std::uint8_t bit = 0U; bit < bits; ++bit) {
            const std::size_t byte_index = bit_offset_ >> 3U;
            const std::uint8_t bit_index = static_cast<std::uint8_t>(bit_offset_ & 7U);
            const std::uint32_t one = static_cast<std::uint32_t>((bytes_[byte_index] >> bit_index) & 1U);
            value |= one << bit;
            ++bit_offset_;
        }
        return value;
    }

private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t bit_offset_{};
};

inline std::uint32_t zigzag_encode(int value) noexcept {
    if (value >= 0) return static_cast<std::uint32_t>(value) * 2U;
    return static_cast<std::uint32_t>(-value) * 2U - 1U;
}

inline int zigzag_decode(std::uint32_t value) noexcept {
    if ((value & 1U) == 0U) return static_cast<int>(value >> 1U);
    return -static_cast<int>((value + 1U) >> 1U);
}

inline std::uint8_t required_bits(std::uint32_t value) noexcept {
    std::uint8_t bits = 0U;
    while (value != 0U) {
        ++bits;
        value >>= 1U;
    }
    return bits;
}

inline int quantize_difference(int difference, int step) noexcept {
    if (step <= 1) return difference;
    const int half = step / 2;
    if (difference >= 0) return (difference + half) / step;
    return -((-difference + half) / step);
}

inline std::uint16_t clamp_pixel10(int value) noexcept {
    return static_cast<std::uint16_t>(std::clamp(value, 0, static_cast<int>(kMaxPixel10)));
}

inline std::pair<std::uint8_t, std::uint8_t> morton_xy(std::uint8_t code) noexcept {
    std::uint8_t x = 0U;
    std::uint8_t y = 0U;
    for (std::uint8_t bit = 0U; bit < 3U; ++bit) {
        x = static_cast<std::uint8_t>(x | static_cast<std::uint8_t>(((code >> (2U * bit)) & 1U) << bit));
        y = static_cast<std::uint8_t>(y | static_cast<std::uint8_t>(((code >> (2U * bit + 1U)) & 1U) << bit));
    }
    return {x, y};
}

struct TileCandidate {
    std::uint16_t tile_x{};
    std::uint16_t tile_y{};
    std::uint8_t valid_w{};
    std::uint8_t valid_h{};
    std::uint8_t shift{};
    std::array<std::uint16_t, kChannels> bases{};
    std::array<std::uint8_t, kChannels> widths{};
    std::vector<std::uint8_t> residual_payload;
    std::vector<std::uint8_t> record;
    double mse{};
    double psnr_db{std::numeric_limits<double>::infinity()};
};

struct CodecConfig {
    bool lossless{};
    std::uint8_t max_shift{4U};
    double max_tile_mse{20.0};
    double min_tile_psnr_db{47.0};
    std::uint32_t fps{120U};
    bool persistent_adaptation{};
    std::size_t instruction_limit{4096U};

    void validate() const {
        if (max_shift > 8U) throw std::invalid_argument("max_shift must be <= 8");
        if (!std::isfinite(max_tile_mse) || max_tile_mse < 0.0) {
            throw std::invalid_argument("max_tile_mse must be finite and non-negative");
        }
        if (!std::isfinite(min_tile_psnr_db) || min_tile_psnr_db < 0.0) {
            throw std::invalid_argument("min_tile_psnr_db must be finite and non-negative");
        }
        if (fps == 0U) throw std::invalid_argument("fps must be positive");
        if (instruction_limit == 0U) throw std::invalid_argument("instruction_limit must be positive");
    }
};

struct CodecMetrics {
    std::uint64_t tiles{};
    std::uint64_t raw_bits{};
    std::uint64_t encoded_bits{};
    double compression_ratio{};
    double mse{};
    double psnr_db{std::numeric_limits<double>::infinity()};
    double hbar_semantic_visual{};
    double average_tile_shift{};
    double average_vcl_entropy{};
    double average_active_nodes{};
    double average_container_delta{};
    std::uint64_t evolution_commits{};
    std::uint64_t evolution_rollbacks{};
    double frame_budget_ms{};
};

struct EncodeResult {
    std::vector<std::uint8_t> bitstream;
    Frame10 reconstructed;
    CodecMetrics metrics;
    dm8kb::Container final_container;
};

class PixelsToBitsCodec {
public:
    explicit PixelsToBitsCodec(CodecConfig config = {}) : config_(config) {
        config_.validate();
        reset_runtime();
    }

    void reset_runtime() {
        container_ = dm8kb::Container{};
        control_tile_ = media8::VCLTile8{};
        container_.set_microcode(media8::default_media_program());
    }

    EncodeResult encode(const Frame10& frame) {
        frame.validate();
        if (!config_.persistent_adaptation) reset_runtime();

        ByteWriter output;
        constexpr std::array<std::uint8_t, 8U> magic{{'D', 'M', 'P', 'X', 'B', 'T', '1', 0U}};
        output.append(magic.data(), magic.size());
        output.u16(1U);
        output.u32(frame.width);
        output.u32(frame.height);
        output.u16(static_cast<std::uint16_t>(std::min<std::uint32_t>(config_.fps, 65535U)));
        output.u8(static_cast<std::uint8_t>(kChannels));
        output.u8(10U);
        output.u8(static_cast<std::uint8_t>(kPixelTileEdge));
        output.u8(config_.lossless ? 1U : 0U);
        output.u8(config_.max_shift);
        output.u8(0U);

        const std::uint32_t tiles_x = (frame.width + 7U) / 8U;
        const std::uint32_t tiles_y = (frame.height + 7U) / 8U;
        const std::uint64_t tile_count64 = static_cast<std::uint64_t>(tiles_x) * static_cast<std::uint64_t>(tiles_y);
        if (tile_count64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::overflow_error("too many pixel tiles for v1 stream");
        }
        output.u32(static_cast<std::uint32_t>(tile_count64));
        output.u64(static_cast<std::uint64_t>(frame.width) * static_cast<std::uint64_t>(frame.height) * 30ULL);

        CodecMetrics metrics;
        metrics.tiles = tile_count64;
        metrics.raw_bits = static_cast<std::uint64_t>(frame.width) * static_cast<std::uint64_t>(frame.height) * 30ULL;
        metrics.frame_budget_ms = 1000.0 / static_cast<double>(config_.fps);
        double shift_sum = 0.0;
        double entropy_sum = 0.0;
        double active_sum = 0.0;
        double container_delta_sum = 0.0;
        std::uint64_t evolution_commits = 0ULL;
        std::uint64_t evolution_rollbacks = 0ULL;

        for (std::uint32_t ty = 0U; ty < tiles_y; ++ty) {
            for (std::uint32_t tx = 0U; tx < tiles_x; ++tx) {
                const std::uint32_t x0 = tx * 8U;
                const std::uint32_t y0 = ty * 8U;
                const std::uint8_t valid_w = static_cast<std::uint8_t>(std::min<std::uint32_t>(8U, frame.width - x0));
                const std::uint8_t valid_h = static_cast<std::uint8_t>(std::min<std::uint32_t>(8U, frame.height - y0));

                const auto container_before = container_.snapshot();
                const media8::AdaptiveSnapshot adaptive_before = control_tile_.adaptive_snapshot();
                const media8::TileMetrics control_metrics = execute_control_tile(frame, x0, y0, valid_w, valid_h);

                TileCandidate candidate = select_candidate(frame, tx, ty, valid_w, valid_h);
                const bool quality_ok = candidate.mse <= config_.max_tile_mse + 1.0e-12 &&
                                        candidate.psnr_db + 1.0e-12 >= config_.min_tile_psnr_db;
                if (!quality_ok) {
                    control_tile_.restore_adaptive(adaptive_before);
                    container_.restore(container_before);
                    throw std::runtime_error("lossless fallback failed quality gate");
                }

                synchronize_container(frame, x0, y0, valid_w, valid_h, candidate, control_metrics,
                                      container_before, true);

                if (candidate.record.size() > static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
                    throw std::overflow_error("tile record exceeds v1 length field");
                }
                output.u16(static_cast<std::uint16_t>(candidate.record.size()));
                output.append(candidate.record);

                shift_sum += static_cast<double>(candidate.shift);
                entropy_sum += control_metrics.normalized_entropy;
                active_sum += static_cast<double>(control_metrics.active_nodes);
                container_delta_sum += dm8kb::Container::normalized_delta(container_before, container_.snapshot());
                evolution_commits += control_metrics.evolution_commits;
                evolution_rollbacks += control_metrics.evolution_rollbacks;
            }
        }

        std::vector<std::uint8_t> stream = output.take();
        const std::uint64_t stream_digest = dm8kb::fnv1a64(stream.data(), stream.size());
        ByteWriter with_digest;
        with_digest.append(stream);
        with_digest.u64(stream_digest);
        stream = with_digest.take();

        Frame10 reconstructed = decode(stream);
        metrics.encoded_bits = static_cast<std::uint64_t>(stream.size()) * 8ULL;
        metrics.compression_ratio = metrics.encoded_bits == 0ULL ? 0.0 :
            static_cast<double>(metrics.raw_bits) / static_cast<double>(metrics.encoded_bits);
        metrics.mse = frame_mse(frame, reconstructed);
        metrics.psnr_db = psnr(metrics.mse);
        metrics.hbar_semantic_visual = std::sqrt(metrics.mse) / static_cast<double>(kMaxPixel10);
        const double tiles = static_cast<double>(metrics.tiles);
        metrics.average_tile_shift = shift_sum / tiles;
        metrics.average_vcl_entropy = entropy_sum / tiles;
        metrics.average_active_nodes = active_sum / tiles;
        metrics.average_container_delta = container_delta_sum / tiles;
        metrics.evolution_commits = evolution_commits;
        metrics.evolution_rollbacks = evolution_rollbacks;

        EncodeResult result;
        result.bitstream = std::move(stream);
        result.reconstructed = std::move(reconstructed);
        result.metrics = metrics;
        result.final_container = container_;
        return result;
    }

    static Frame10 decode(const std::vector<std::uint8_t>& bitstream) {
        if (bitstream.size() < 8U + 2U + 4U + 4U + 2U + 6U + 4U + 8U + 8U) {
            throw std::runtime_error("pixels-to-bits stream too small");
        }
        const std::size_t payload_size = bitstream.size() - 8U;
        const std::uint64_t expected_digest = read_tail_u64(bitstream);
        const std::uint64_t actual_digest = dm8kb::fnv1a64(bitstream.data(), payload_size);
        if (expected_digest != actual_digest) throw std::runtime_error("pixels-to-bits stream digest mismatch");

        std::vector<std::uint8_t> payload(bitstream.begin(), bitstream.end() - 8);
        ByteReader input(payload);
        constexpr std::array<std::uint8_t, 8U> magic{{'D', 'M', 'P', 'X', 'B', 'T', '1', 0U}};
        for (const std::uint8_t expected : magic) {
            if (input.u8() != expected) throw std::runtime_error("unsupported pixels-to-bits magic");
        }
        if (input.u16() != 1U) throw std::runtime_error("unsupported pixels-to-bits version");
        const std::uint32_t width = input.u32();
        const std::uint32_t height = input.u32();
        (void)input.u16();
        if (input.u8() != static_cast<std::uint8_t>(kChannels)) throw std::runtime_error("unsupported channel count");
        if (input.u8() != 10U) throw std::runtime_error("unsupported bit depth");
        if (input.u8() != static_cast<std::uint8_t>(kPixelTileEdge)) throw std::runtime_error("unsupported tile edge");
        (void)input.u8();
        (void)input.u8();
        (void)input.u8();
        const std::uint32_t tile_count = input.u32();
        const std::uint64_t raw_bits = input.u64();
        const std::uint64_t expected_raw_bits = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 30ULL;
        if (raw_bits != expected_raw_bits) throw std::runtime_error("pixels-to-bits raw size metadata mismatch");

        const std::uint64_t sample_count64 = static_cast<std::uint64_t>(width) *
                                             static_cast<std::uint64_t>(height) *
                                             static_cast<std::uint64_t>(kChannels);
        if (sample_count64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("decoded frame exceeds host addressable size");
        }
        Frame10 frame;
        frame.width = width;
        frame.height = height;
        frame.rgb.assign(static_cast<std::size_t>(sample_count64), 0U);

        const std::uint32_t tiles_x = (width + 7U) / 8U;
        const std::uint32_t tiles_y = (height + 7U) / 8U;
        if (static_cast<std::uint64_t>(tile_count) !=
            static_cast<std::uint64_t>(tiles_x) * static_cast<std::uint64_t>(tiles_y)) {
            throw std::runtime_error("pixels-to-bits tile count mismatch");
        }

        for (std::uint32_t tile = 0U; tile < tile_count; ++tile) {
            const std::size_t record_size = static_cast<std::size_t>(input.u16());
            const std::vector<std::uint8_t> record = input.bytes(record_size);
            decode_tile_record(record, frame);
        }
        if (input.remaining() != 0U) throw std::runtime_error("unexpected trailing pixels-to-bits payload");
        frame.validate();
        return frame;
    }

    const dm8kb::Container& container() const noexcept { return container_; }

private:
    CodecConfig config_;
    dm8kb::Container container_;
    media8::VCLTile8 control_tile_;

    static std::uint64_t read_tail_u64(const std::vector<std::uint8_t>& values) noexcept {
        const std::size_t base = values.size() - 8U;
        std::uint64_t value = 0ULL;
        for (std::size_t i = 0U; i < 8U; ++i) {
            value |= static_cast<std::uint64_t>(values[base + i]) << (8U * i);
        }
        return value;
    }

    static std::vector<std::pair<std::uint8_t, std::uint8_t>> valid_morton_positions(
        std::uint8_t valid_w, std::uint8_t valid_h) {
        std::vector<std::pair<std::uint8_t, std::uint8_t>> positions;
        positions.reserve(static_cast<std::size_t>(valid_w) * static_cast<std::size_t>(valid_h));
        for (std::uint8_t code = 0U; code < 64U; ++code) {
            const auto xy = morton_xy(code);
            if (xy.first < valid_w && xy.second < valid_h) positions.push_back(xy);
        }
        return positions;
    }

    TileCandidate build_candidate(const Frame10& frame, std::uint32_t tx, std::uint32_t ty,
                                  std::uint8_t valid_w, std::uint8_t valid_h, std::uint8_t shift) const {
        TileCandidate candidate;
        if (tx > 65535U || ty > 65535U) throw std::overflow_error("tile coordinate exceeds v1 record");
        candidate.tile_x = static_cast<std::uint16_t>(tx);
        candidate.tile_y = static_cast<std::uint16_t>(ty);
        candidate.valid_w = valid_w;
        candidate.valid_h = valid_h;
        candidate.shift = shift;

        const std::uint32_t x0 = tx * 8U;
        const std::uint32_t y0 = ty * 8U;
        const auto positions = valid_morton_positions(valid_w, valid_h);
        if (positions.empty()) throw std::runtime_error("empty pixel tile");

        std::array<std::vector<int>, kChannels> residuals;
        std::array<std::uint32_t, kChannels> max_zigzag{};
        double squared_error = 0.0;
        const int step = 1 << shift;

        for (std::size_t channel = 0U; channel < kChannels; ++channel) {
            const auto first = positions.front();
            const std::uint16_t base = frame.at(x0 + first.first, y0 + first.second, channel);
            candidate.bases[channel] = base;
            int reconstructed_previous = static_cast<int>(base);
            residuals[channel].reserve(positions.size() - 1U);
            for (std::size_t index = 1U; index < positions.size(); ++index) {
                const auto xy = positions[index];
                const int source = static_cast<int>(frame.at(x0 + xy.first, y0 + xy.second, channel));
                const int difference = source - reconstructed_previous;
                const int q = quantize_difference(difference, step);
                const int reconstructed = static_cast<int>(clamp_pixel10(reconstructed_previous + q * step));
                residuals[channel].push_back(q);
                max_zigzag[channel] = std::max(max_zigzag[channel], zigzag_encode(q));
                const double error = static_cast<double>(source - reconstructed);
                squared_error += error * error;
                reconstructed_previous = reconstructed;
            }
            candidate.widths[channel] = required_bits(max_zigzag[channel]);
        }

        const std::size_t sample_count = positions.size() * kChannels;
        candidate.mse = sample_count == 0U ? 0.0 : squared_error / static_cast<double>(sample_count);
        candidate.psnr_db = psnr(candidate.mse);

        BitWriter packed;
        for (std::size_t channel = 0U; channel < kChannels; ++channel) {
            const std::uint8_t width = candidate.widths[channel];
            for (const int residual : residuals[channel]) {
                if (width != 0U) packed.write(zigzag_encode(residual), width);
            }
        }
        candidate.residual_payload = packed.bytes();

        ByteWriter record;
        record.u16(candidate.tile_x);
        record.u16(candidate.tile_y);
        record.u8(candidate.valid_w);
        record.u8(candidate.valid_h);
        record.u8(candidate.shift);
        for (const std::uint8_t width : candidate.widths) record.u8(width);
        for (const std::uint16_t base : candidate.bases) record.u16(base);
        if (candidate.residual_payload.size() > static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
            throw std::overflow_error("packed tile residual payload exceeds v1 size field");
        }
        record.u16(static_cast<std::uint16_t>(candidate.residual_payload.size()));
        record.append(candidate.residual_payload);
        std::vector<std::uint8_t> without_digest = record.take();
        const std::uint64_t digest = dm8kb::fnv1a64(without_digest.data(), without_digest.size());
        ByteWriter complete;
        complete.append(without_digest);
        complete.u64(digest);
        candidate.record = complete.take();
        return candidate;
    }

    TileCandidate select_candidate(const Frame10& frame, std::uint32_t tx, std::uint32_t ty,
                                   std::uint8_t valid_w, std::uint8_t valid_h) const {
        TileCandidate best = build_candidate(frame, tx, ty, valid_w, valid_h, 0U);
        if (config_.lossless) return best;

        for (std::uint8_t shift = 1U; shift <= config_.max_shift; ++shift) {
            TileCandidate candidate = build_candidate(frame, tx, ty, valid_w, valid_h, shift);
            const bool quality_ok = candidate.mse <= config_.max_tile_mse + 1.0e-12 &&
                                    candidate.psnr_db + 1.0e-12 >= config_.min_tile_psnr_db;
            if (!quality_ok) continue;
            if (candidate.record.size() < best.record.size() ||
                (candidate.record.size() == best.record.size() && candidate.mse < best.mse)) {
                best = std::move(candidate);
            }
        }
        return best;
    }

    media8::TileMetrics execute_control_tile(const Frame10& frame, std::uint32_t x0, std::uint32_t y0,
                                             std::uint8_t valid_w, std::uint8_t valid_h) {
        std::array<std::uint8_t, media8::kTileNodes> projection{};
        projection.fill(128U);
        std::size_t cursor = 0U;
        for (std::uint8_t y = 0U; y < valid_h; ++y) {
            for (std::uint8_t x = 0U; x < valid_w; ++x) {
                for (std::size_t channel = 0U; channel < kChannels; ++channel) {
                    if (cursor >= projection.size()) break;
                    const std::uint16_t sample = frame.at(x0 + x, y0 + y, channel);
                    projection[cursor++] = static_cast<std::uint8_t>(sample >> 2U);
                }
            }
        }
        control_tile_.ingest_tile(projection);
        media8::VCLBVM8 vm(control_tile_);
        return vm.execute(container_.microcode(), config_.instruction_limit);
    }

    void synchronize_container(const Frame10& frame, std::uint32_t x0, std::uint32_t y0,
                               std::uint8_t valid_w, std::uint8_t valid_h,
                               const TileCandidate& candidate, const media8::TileMetrics& control_metrics,
                               const std::array<std::uint8_t, dm8kb::kContainerBytes>& before,
                               bool accepted) {
        const std::uint64_t prior_digest = dm8kb::fnv1a64(before.data(), dm8kb::region_offset(dm8kb::Region::Integrity));

        std::array<std::uint8_t, media8::kTileNodes> states{};
        const auto& nodes = control_tile_.nodes();
        for (std::size_t i = 0U; i < nodes.size(); ++i) states[i] = media8::centered_to_byte(nodes[i].state);
        container_.write_region(dm8kb::Region::VclState, states.data(), states.size());

        const media8::AdaptiveSnapshot adaptive = control_tile_.adaptive_snapshot();
        std::array<std::uint8_t, media8::kTileNodes> theta{};
        for (std::size_t i = 0U; i < theta.size(); ++i) theta[i] = static_cast<std::uint8_t>(adaptive.weights[i]);
        container_.write_region(dm8kb::Region::Theta, theta.data(), theta.size());

        std::array<std::uint8_t, 128U> masks{};
        for (std::size_t i = 0U; i < nodes.size(); ++i) {
            const std::size_t byte = i >> 3U;
            const std::uint8_t bit = static_cast<std::uint8_t>(1U << (i & 7U));
            if (nodes[i].logic_mask != 0U) masks[byte] = static_cast<std::uint8_t>(masks[byte] | bit);
            if (nodes[i].control_flag != 0U) masks[64U + byte] = static_cast<std::uint8_t>(masks[64U + byte] | bit);
        }
        container_.write_region(dm8kb::Region::Masks, masks.data(), masks.size());

        std::array<std::uint8_t, 512U> omega{};
        for (std::size_t i = 0U; i < adaptive.omega.size(); ++i) {
            const std::uint16_t value = static_cast<std::uint16_t>(adaptive.omega[i]);
            omega[i * 2U] = static_cast<std::uint8_t>(value & 0xFFU);
            omega[i * 2U + 1U] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
        }
        container_.write_region(dm8kb::Region::Omega, omega.data(), omega.size());

        ByteWriter residual;
        const auto positions = valid_morton_positions(valid_w, valid_h);
        const int step = 1 << candidate.shift;
        for (std::size_t channel = 0U; channel < kChannels; ++channel) {
            int previous = static_cast<int>(candidate.bases[channel]);
            const auto first = positions.front();
            (void)first;
            for (std::size_t index = 1U; index < positions.size(); ++index) {
                const auto xy = positions[index];
                const int source = static_cast<int>(frame.at(x0 + xy.first, y0 + xy.second, channel));
                const int q = quantize_difference(source - previous, step);
                const int reconstructed = static_cast<int>(clamp_pixel10(previous + q * step));
                const int error = source - reconstructed;
                residual.u16(static_cast<std::uint16_t>(static_cast<std::int16_t>(error)));
                previous = reconstructed;
            }
        }
        container_.write_region(dm8kb::Region::Residual, residual.bytes());

        ByteWriter features;
        features.u32(x0);
        features.u32(y0);
        features.u8(valid_w);
        features.u8(valid_h);
        features.u8(candidate.shift);
        features.u8(accepted ? 1U : 0U);
        features.u32(dm8kb::q24(candidate.mse));
        features.u32(dm8kb::q24(candidate.mse == 0.0 ? 0.0 : std::sqrt(candidate.mse) / 1023.0));
        features.u32(dm8kb::q24(control_metrics.normalized_entropy));
        features.u16(static_cast<std::uint16_t>(std::min<std::size_t>(control_metrics.active_nodes, 65535U)));
        container_.write_region(dm8kb::Region::Features, features.bytes());
        container_.stage_shadow(candidate.record);

        const std::uint64_t candidate_digest = container_.digest();
        if (accepted) container_.increment_epoch();
        const auto after = container_.snapshot();
        const double state_delta = dm8kb::Container::normalized_delta(before, after);

        dm8kb::CommitReceipt receipt;
        receipt.prior_digest = prior_digest;
        receipt.candidate_digest = candidate_digest;
        receipt.result_digest = container_.digest();
        receipt.microcode_digest = container_.microcode_digest();
        receipt.epoch = container_.epoch();
        receipt.changed_regions = dm8kb::Container::changed_region_bitmap(before, after);
        receipt.semantic_gap_q24 = dm8kb::q24(candidate.mse == 0.0 ? 0.0 : std::sqrt(candidate.mse) / 1023.0);
        receipt.state_delta_q24 = dm8kb::q24(state_delta);
        receipt.accepted = accepted;
        container_.write_receipt(receipt);
    }

    static void decode_tile_record(const std::vector<std::uint8_t>& record, Frame10& frame) {
        if (record.size() < 2U + 2U + 1U + 1U + 1U + 3U + 6U + 2U + 8U) {
            throw std::runtime_error("pixel tile record too small");
        }
        const std::size_t digest_offset = record.size() - 8U;
        std::uint64_t expected_digest = 0ULL;
        for (std::size_t i = 0U; i < 8U; ++i) {
            expected_digest |= static_cast<std::uint64_t>(record[digest_offset + i]) << (8U * i);
        }
        const std::uint64_t actual_digest = dm8kb::fnv1a64(record.data(), digest_offset);
        if (expected_digest != actual_digest) throw std::runtime_error("pixel tile digest mismatch");

        std::vector<std::uint8_t> body(record.begin(), record.begin() + static_cast<std::ptrdiff_t>(digest_offset));
        ByteReader input(body);
        const std::uint16_t tx = input.u16();
        const std::uint16_t ty = input.u16();
        const std::uint8_t valid_w = input.u8();
        const std::uint8_t valid_h = input.u8();
        const std::uint8_t shift = input.u8();
        if (valid_w == 0U || valid_w > 8U || valid_h == 0U || valid_h > 8U || shift > 8U) {
            throw std::runtime_error("invalid pixel tile geometry/shift");
        }
        std::array<std::uint8_t, kChannels> widths{};
        for (std::uint8_t& width : widths) {
            width = input.u8();
            if (width > 11U) throw std::runtime_error("invalid residual bit width");
        }
        std::array<std::uint16_t, kChannels> bases{};
        for (std::uint16_t& base : bases) {
            base = input.u16();
            if (base > kMaxPixel10) throw std::runtime_error("invalid 10-bit tile base");
        }
        const std::size_t payload_size = static_cast<std::size_t>(input.u16());
        const std::vector<std::uint8_t> residual_payload = input.bytes(payload_size);
        if (input.remaining() != 0U) throw std::runtime_error("unexpected tile record bytes");

        const std::uint32_t x0 = static_cast<std::uint32_t>(tx) * 8U;
        const std::uint32_t y0 = static_cast<std::uint32_t>(ty) * 8U;
        if (x0 >= frame.width || y0 >= frame.height ||
            x0 + valid_w > frame.width || y0 + valid_h > frame.height) {
            throw std::runtime_error("tile record outside decoded frame");
        }

        const auto positions = valid_morton_positions(valid_w, valid_h);
        const int step = 1 << shift;
        BitReader packed(residual_payload);
        for (std::size_t channel = 0U; channel < kChannels; ++channel) {
            int previous = static_cast<int>(bases[channel]);
            const auto first = positions.front();
            frame.set(x0 + first.first, y0 + first.second, channel, bases[channel]);
            const std::uint8_t width = widths[channel];
            for (std::size_t index = 1U; index < positions.size(); ++index) {
                const int q = width == 0U ? 0 : zigzag_decode(packed.read(width));
                const std::uint16_t reconstructed = clamp_pixel10(previous + q * step);
                const auto xy = positions[index];
                frame.set(x0 + xy.first, y0 + xy.second, channel, reconstructed);
                previous = static_cast<int>(reconstructed);
            }
        }
    }

    static double frame_mse(const Frame10& left, const Frame10& right) {
        left.validate();
        right.validate();
        if (left.width != right.width || left.height != right.height || left.rgb.size() != right.rgb.size()) {
            throw std::invalid_argument("frame MSE geometry mismatch");
        }
        if (left.rgb.empty()) return 0.0;
        long double squared = 0.0L;
        for (std::size_t i = 0U; i < left.rgb.size(); ++i) {
            const long double delta = static_cast<long double>(left.rgb[i]) - static_cast<long double>(right.rgb[i]);
            squared += delta * delta;
        }
        return static_cast<double>(squared / static_cast<long double>(left.rgb.size()));
    }

    static double psnr(double mse) noexcept {
        if (mse <= 0.0) return std::numeric_limits<double>::infinity();
        const double peak = static_cast<double>(kMaxPixel10);
        return 10.0 * std::log10((peak * peak) / mse);
    }
};

inline Frame10 make_gradient_frame(std::uint32_t width, std::uint32_t height) {
    if (width == 0U || height == 0U) throw std::invalid_argument("gradient dimensions must be positive");
    const std::uint64_t count64 = static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 3ULL;
    if (count64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("gradient frame too large");
    }
    Frame10 frame;
    frame.width = width;
    frame.height = height;
    frame.rgb.assign(static_cast<std::size_t>(count64), 0U);
    for (std::uint32_t y = 0U; y < height; ++y) {
        for (std::uint32_t x = 0U; x < width; ++x) {
            const std::uint16_t r = width <= 1U ? 0U : static_cast<std::uint16_t>((static_cast<std::uint64_t>(x) * 1023ULL) / static_cast<std::uint64_t>(width - 1U));
            const std::uint16_t g = height <= 1U ? 0U : static_cast<std::uint16_t>((static_cast<std::uint64_t>(y) * 1023ULL) / static_cast<std::uint64_t>(height - 1U));
            const std::uint16_t b = static_cast<std::uint16_t>((static_cast<std::uint32_t>(r) + static_cast<std::uint32_t>(g)) / 2U);
            frame.set(x, y, 0U, r);
            frame.set(x, y, 1U, g);
            frame.set(x, y, 2U, b);
        }
    }
    return frame;
}

} // namespace jarvisx::pixels
