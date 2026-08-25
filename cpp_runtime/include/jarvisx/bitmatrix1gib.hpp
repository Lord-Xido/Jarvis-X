#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace jarvisx {

constexpr std::uint64_t kOneGiBBytes = std::uint64_t{1} << 30U;
constexpr std::uint64_t kOneGiBBits = kOneGiBBytes * 8U;
constexpr std::uint32_t kCanonicalBitCubeEdge = 2048U;
constexpr std::uint64_t kLogicalVectorBits = 512U;

struct BitCoordinate3D {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
};

inline BitCoordinate3D bit_coordinate_2048(std::uint64_t bit_index) {
    if (bit_index >= kOneGiBBits) {
        throw std::out_of_range("bit index is outside the canonical 2048^3-bit volume");
    }
    constexpr std::uint64_t plane =
        static_cast<std::uint64_t>(kCanonicalBitCubeEdge) *
        static_cast<std::uint64_t>(kCanonicalBitCubeEdge);
    const std::uint64_t z = bit_index / plane;
    const std::uint64_t rem = bit_index % plane;
    const std::uint64_t y = rem / kCanonicalBitCubeEdge;
    const std::uint64_t x = rem % kCanonicalBitCubeEdge;
    return {
        static_cast<std::uint32_t>(x),
        static_cast<std::uint32_t>(y),
        static_cast<std::uint32_t>(z)
    };
}

inline std::uint64_t bit_index_2048(BitCoordinate3D coordinate) {
    if (coordinate.x >= kCanonicalBitCubeEdge ||
        coordinate.y >= kCanonicalBitCubeEdge ||
        coordinate.z >= kCanonicalBitCubeEdge) {
        throw std::out_of_range("3D coordinate is outside the canonical 2048^3-bit volume");
    }
    return static_cast<std::uint64_t>(coordinate.z) *
               static_cast<std::uint64_t>(kCanonicalBitCubeEdge) *
               static_cast<std::uint64_t>(kCanonicalBitCubeEdge) +
           static_cast<std::uint64_t>(coordinate.y) *
               static_cast<std::uint64_t>(kCanonicalBitCubeEdge) +
           static_cast<std::uint64_t>(coordinate.x);
}

enum class StreamPattern {
    Sparse3D,
    Checker3D,
    Zero,
    Random
};

inline const char* stream_pattern_name(StreamPattern pattern) noexcept {
    switch (pattern) {
        case StreamPattern::Sparse3D: return "sparse3d";
        case StreamPattern::Checker3D: return "checker3d";
        case StreamPattern::Zero: return "zero";
        case StreamPattern::Random: return "random";
    }
    return "unknown";
}

inline StreamPattern parse_stream_pattern(const std::string& value) {
    if (value == "sparse3d") return StreamPattern::Sparse3D;
    if (value == "checker3d") return StreamPattern::Checker3D;
    if (value == "zero") return StreamPattern::Zero;
    if (value == "random") return StreamPattern::Random;
    throw std::invalid_argument("pattern must be sparse3d, checker3d, zero or random");
}

struct Stream1GiBConfig {
    std::uint64_t target_bytes{kOneGiBBytes};
    std::size_t chunk_bytes{8U * 1024U * 1024U};
    StreamPattern pattern{StreamPattern::Sparse3D};
    std::uint64_t seed{0x444D315342535452ULL};
    std::uint32_t window_ms{100U};
    std::uint64_t configured_l3_bytes{0U};

    void validate() const {
        if (target_bytes == 0U || target_bytes > kOneGiBBytes) {
            throw std::invalid_argument("target bytes must be in (0, 1 GiB]");
        }
        if ((target_bytes & 7U) != 0U) {
            throw std::invalid_argument("target bytes must be divisible by 8");
        }
        if (chunk_bytes < 64U || (chunk_bytes & 7U) != 0U) {
            throw std::invalid_argument("chunk bytes must be >=64 and divisible by 8");
        }
        if (static_cast<std::uint64_t>(chunk_bytes) > target_bytes) {
            throw std::invalid_argument("chunk bytes cannot exceed target bytes");
        }
        if (window_ms == 0U || window_ms > 60000U) {
            throw std::invalid_argument("window ms must be in [1, 60000]");
        }
    }
};

struct StreamChunkMetrics {
    std::uint64_t offset_bytes{};
    std::size_t raw_bytes{};
    std::size_t encoded_bytes{};
    bool raw_passthrough{};
    double ingest_ms{};
    double encode_ms{};
    double core_verify_ms{};
    double decode_ms{};
};

struct Stream1GiBMetrics {
    std::uint64_t target_bytes{};
    std::uint64_t processed_bytes{};
    std::uint64_t encoded_bytes{};
    std::uint64_t logical_vectors_512{};
    std::uint64_t first_window_bytes{};
    double first_window_elapsed_ms{};
    double ingest_seconds{};
    double encode_seconds{};
    double core_verify_seconds{};
    double decode_seconds{};
    double total_seconds{};
    double compression_ratio{};
    double throughput_gbps{};
    double first_window_gbps{};
    std::size_t reusable_working_set_bytes{};
    std::size_t hot_path_reallocations{};
    bool exact_round_trip{};
    bool codec_fixed_point{};
    bool fits_configured_l3{};
    std::vector<StreamChunkMetrics> chunks;
};

inline std::uint64_t splitmix64(std::uint64_t value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

inline void generate_stream_words(std::vector<std::uint64_t>& words,
                                  std::uint64_t global_word_offset,
                                  StreamPattern pattern,
                                  std::uint64_t seed) {
    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::uint64_t global_word =
            global_word_offset + static_cast<std::uint64_t>(i);
        switch (pattern) {
            case StreamPattern::Zero:
                words[i] = 0U;
                break;
            case StreamPattern::Random:
                words[i] = splitmix64(seed ^ global_word);
                break;
            case StreamPattern::Checker3D: {
                const auto c = bit_coordinate_2048((global_word * 64U) % kOneGiBBits);
                const std::uint32_t cell =
                    (c.x / 64U) + (c.y / 32U) + (c.z / 16U);
                words[i] = (cell & 1U) == 0U
                    ? std::uint64_t{0}
                    : std::numeric_limits<std::uint64_t>::max();
                break;
            }
            case StreamPattern::Sparse3D: {
                const auto c = bit_coordinate_2048((global_word * 64U) % kOneGiBBits);
                const std::uint32_t gate =
                    ((c.x / 64U) + (c.y / 64U) * 3U + (c.z / 64U) * 5U) & 63U;
                if (gate == 0U) {
                    const std::uint64_t mixed = splitmix64(seed ^ global_word);
                    words[i] = std::uint64_t{1} << (mixed & 63U);
                } else {
                    words[i] = 0U;
                }
                break;
            }
        }
    }
}

inline void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    output.push_back(static_cast<std::uint8_t>(value & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<std::uint8_t>((value >> 24U) & 0xFFU));
}

inline void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
    }
}

inline std::uint32_t read_u32_le(const std::vector<std::uint8_t>& input,
                                 std::size_t& cursor) {
    if (cursor + 4U > input.size()) throw std::runtime_error("truncated RLE count");
    std::uint32_t value = 0U;
    for (unsigned shift = 0U; shift < 32U; shift += 8U) {
        value |= static_cast<std::uint32_t>(input[cursor++]) << shift;
    }
    return value;
}

inline std::uint64_t read_u64_le(const std::vector<std::uint8_t>& input,
                                 std::size_t& cursor) {
    if (cursor + 8U > input.size()) throw std::runtime_error("truncated literal word");
    std::uint64_t value = 0U;
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
        value |= static_cast<std::uint64_t>(input[cursor++]) << shift;
    }
    return value;
}

class AdaptiveWordCodec {
public:
    static bool encode(const std::vector<std::uint64_t>& input,
                       std::vector<std::uint8_t>& output) {
        if (input.empty()) throw std::invalid_argument("codec input cannot be empty");
        output.clear();
        output.push_back(1U); // RLE mode
        std::size_t i = 0U;
        while (i < input.size()) {
            if (input[i] == 0U || input[i] == std::numeric_limits<std::uint64_t>::max()) {
                const std::uint64_t repeated = input[i];
                std::size_t end = i + 1U;
                while (end < input.size() && input[end] == repeated &&
                       (end - i) < static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                    ++end;
                }
                output.push_back(repeated == 0U ? 0U : 1U);
                append_u32_le(output, static_cast<std::uint32_t>(end - i));
                i = end;
            } else {
                const std::size_t begin = i;
                ++i;
                while (i < input.size() && input[i] != 0U &&
                       input[i] != std::numeric_limits<std::uint64_t>::max() &&
                       (i - begin) < static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                    ++i;
                }
                output.push_back(2U);
                append_u32_le(output, static_cast<std::uint32_t>(i - begin));
                for (std::size_t j = begin; j < i; ++j) append_u64_le(output, input[j]);
            }
        }

        const std::size_t raw_size = 1U + input.size() * sizeof(std::uint64_t);
        if (output.size() < raw_size) return false;

        output.clear();
        output.reserve(raw_size);
        output.push_back(0U); // raw mode
        for (const auto word : input) append_u64_le(output, word);
        return true;
    }

    static void decode(const std::vector<std::uint8_t>& encoded,
                       std::size_t expected_words,
                       std::vector<std::uint64_t>& output) {
        if (encoded.empty()) throw std::runtime_error("encoded chunk is empty");
        output.clear();
        if (output.capacity() < expected_words) output.reserve(expected_words);
        std::size_t cursor = 1U;
        const std::uint8_t mode = encoded[0];
        if (mode == 0U) {
            const std::size_t expected_bytes = 1U + expected_words * sizeof(std::uint64_t);
            if (encoded.size() != expected_bytes) throw std::runtime_error("raw chunk size mismatch");
            for (std::size_t i = 0U; i < expected_words; ++i) {
                output.push_back(read_u64_le(encoded, cursor));
            }
        } else if (mode == 1U) {
            while (cursor < encoded.size() && output.size() < expected_words) {
                const std::uint8_t tag = encoded[cursor++];
                const std::uint32_t count = read_u32_le(encoded, cursor);
                if (count == 0U ||
                    static_cast<std::uint64_t>(output.size()) + count > expected_words) {
                    throw std::runtime_error("invalid RLE run length");
                }
                if (tag == 0U || tag == 1U) {
                    const std::uint64_t word = tag == 0U
                        ? std::uint64_t{0}
                        : std::numeric_limits<std::uint64_t>::max();
                    output.insert(output.end(), count, word);
                } else if (tag == 2U) {
                    for (std::uint32_t i = 0U; i < count; ++i) {
                        output.push_back(read_u64_le(encoded, cursor));
                    }
                } else {
                    throw std::runtime_error("unknown RLE token");
                }
            }
            if (output.size() != expected_words || cursor != encoded.size()) {
                throw std::runtime_error("RLE chunk did not decode canonically");
            }
        } else {
            throw std::runtime_error("unknown chunk codec mode");
        }
    }
};

inline double milliseconds_since(const std::chrono::steady_clock::time_point& begin,
                                 const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline Stream1GiBMetrics run_stream_1gib(const Stream1GiBConfig& config) {
    config.validate();
    const std::size_t max_words = config.chunk_bytes / sizeof(std::uint64_t);
    std::vector<std::uint64_t> input(max_words, 0U);
    std::vector<std::uint64_t> decoded;
    decoded.reserve(max_words);
    std::vector<std::uint8_t> encoded;
    const std::size_t reserve_encoded =
        config.chunk_bytes > (std::numeric_limits<std::size_t>::max() - 64U) / 2U
            ? throw std::overflow_error("chunk reserve overflow")
            : config.chunk_bytes * 2U + 64U;
    encoded.reserve(reserve_encoded);

    const std::size_t input_capacity0 = input.capacity();
    const std::size_t decoded_capacity0 = decoded.capacity();
    const std::size_t encoded_capacity0 = encoded.capacity();

    Stream1GiBMetrics metrics;
    metrics.target_bytes = config.target_bytes;
    metrics.logical_vectors_512 =
        (config.target_bytes * 8U + kLogicalVectorBits - 1U) / kLogicalVectorBits;
    metrics.exact_round_trip = true;
    metrics.codec_fixed_point = true;
    metrics.chunks.reserve(static_cast<std::size_t>(
        (config.target_bytes + static_cast<std::uint64_t>(config.chunk_bytes) - 1U) /
        static_cast<std::uint64_t>(config.chunk_bytes)));

    const auto all_begin = std::chrono::steady_clock::now();
    bool window_closed = false;
    std::uint64_t offset_bytes = 0U;
    while (offset_bytes < config.target_bytes) {
        const std::uint64_t remaining = config.target_bytes - offset_bytes;
        const std::size_t raw_bytes = static_cast<std::size_t>(std::min<std::uint64_t>(
            remaining, static_cast<std::uint64_t>(config.chunk_bytes)));
        const std::size_t words = raw_bytes / sizeof(std::uint64_t);
        input.resize(words);

        const auto ingest_begin = std::chrono::steady_clock::now();
        generate_stream_words(input, offset_bytes / sizeof(std::uint64_t),
                              config.pattern, config.seed);
        const auto ingest_end = std::chrono::steady_clock::now();

        const auto encode_begin = ingest_end;
        const bool raw_mode = AdaptiveWordCodec::encode(input, encoded);
        const auto encode_end = std::chrono::steady_clock::now();

        const auto decode_begin = encode_end;
        AdaptiveWordCodec::decode(encoded, words, decoded);
        const auto decode_end = std::chrono::steady_clock::now();

        const auto verify_begin = decode_end;
        const bool equal = input == decoded;
        const auto verify_end = std::chrono::steady_clock::now();
        if (!equal) {
            metrics.exact_round_trip = false;
            metrics.codec_fixed_point = false;
            throw std::runtime_error("stream codec round-trip mismatch");
        }

        StreamChunkMetrics chunk;
        chunk.offset_bytes = offset_bytes;
        chunk.raw_bytes = raw_bytes;
        chunk.encoded_bytes = encoded.size();
        chunk.raw_passthrough = raw_mode;
        chunk.ingest_ms = milliseconds_since(ingest_begin, ingest_end);
        chunk.encode_ms = milliseconds_since(encode_begin, encode_end);
        chunk.decode_ms = milliseconds_since(decode_begin, decode_end);
        chunk.core_verify_ms = milliseconds_since(verify_begin, verify_end);
        metrics.chunks.push_back(chunk);

        metrics.ingest_seconds += chunk.ingest_ms / 1000.0;
        metrics.encode_seconds += chunk.encode_ms / 1000.0;
        metrics.decode_seconds += chunk.decode_ms / 1000.0;
        metrics.core_verify_seconds += chunk.core_verify_ms / 1000.0;
        metrics.processed_bytes += raw_bytes;
        metrics.encoded_bytes += encoded.size();
        offset_bytes += raw_bytes;

        if (input.capacity() != input_capacity0) {
            ++metrics.hot_path_reallocations;
        }
        if (decoded.capacity() != decoded_capacity0) {
            ++metrics.hot_path_reallocations;
        }
        if (encoded.capacity() != encoded_capacity0) {
            ++metrics.hot_path_reallocations;
        }

        const auto now = std::chrono::steady_clock::now();
        const double elapsed_ms = milliseconds_since(all_begin, now);
        if (!window_closed) {
            metrics.first_window_bytes = metrics.processed_bytes;
            metrics.first_window_elapsed_ms = elapsed_ms;
            if (elapsed_ms >= static_cast<double>(config.window_ms)) window_closed = true;
        }
    }
    const auto all_end = std::chrono::steady_clock::now();
    metrics.total_seconds =
        std::chrono::duration<double>(all_end - all_begin).count();
    metrics.compression_ratio = metrics.encoded_bytes == 0U
        ? 0.0
        : static_cast<double>(metrics.processed_bytes) /
          static_cast<double>(metrics.encoded_bytes);
    metrics.throughput_gbps = metrics.total_seconds <= 0.0
        ? 0.0
        : static_cast<double>(metrics.processed_bytes) * 8.0 /
          metrics.total_seconds / 1.0e9;
    metrics.first_window_gbps = metrics.first_window_elapsed_ms <= 0.0
        ? 0.0
        : static_cast<double>(metrics.first_window_bytes) * 8.0 /
          (metrics.first_window_elapsed_ms / 1000.0) / 1.0e9;
    metrics.reusable_working_set_bytes =
        input_capacity0 * sizeof(std::uint64_t) +
        decoded_capacity0 * sizeof(std::uint64_t) +
        encoded_capacity0 * sizeof(std::uint8_t);
    metrics.fits_configured_l3 =
        config.configured_l3_bytes != 0U &&
        static_cast<std::uint64_t>(metrics.reusable_working_set_bytes) <=
            config.configured_l3_bytes;
    return metrics;
}

} // namespace jarvisx
