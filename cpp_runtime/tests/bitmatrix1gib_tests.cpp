#include "jarvisx/bitmatrix1gib.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void canonical_volume_mapping() {
    require(jarvisx::kOneGiBBits ==
                static_cast<std::uint64_t>(jarvisx::kCanonicalBitCubeEdge) *
                jarvisx::kCanonicalBitCubeEdge * jarvisx::kCanonicalBitCubeEdge,
            "1 GiB must equal the 2048^3-bit canonical volume");
    const auto origin = jarvisx::bit_coordinate_2048(0U);
    require(origin.x == 0U && origin.y == 0U && origin.z == 0U,
            "origin coordinate mismatch");
    const auto last = jarvisx::bit_coordinate_2048(jarvisx::kOneGiBBits - 1U);
    require(last.x == 2047U && last.y == 2047U && last.z == 2047U,
            "last coordinate mismatch");
    require(jarvisx::bit_index_2048(last) == jarvisx::kOneGiBBits - 1U,
            "coordinate inverse mismatch");
    require(jarvisx::kOneGiBBytes / 64U == 16777216U,
            "1 GiB must contain exactly 16,777,216 logical 512-bit vectors");
}

void codec_round_trip_and_fallback() {
    std::vector<std::uint64_t> sparse(4096U, 0U);
    sparse[0] = 1U;
    sparse[2048] = 2U;
    std::vector<std::uint8_t> encoded;
    encoded.reserve(16384U);
    const bool sparse_raw = jarvisx::AdaptiveWordCodec::encode(sparse, encoded);
    require(!sparse_raw, "sparse data should select RLE mode");
    require(encoded.size() < sparse.size() * sizeof(std::uint64_t),
            "sparse encoding should be smaller than raw words");
    std::vector<std::uint64_t> decoded;
    jarvisx::AdaptiveWordCodec::decode(encoded, sparse.size(), decoded);
    require(decoded == sparse, "sparse codec round-trip failed");

    std::vector<std::uint64_t> random(4096U, 0U);
    jarvisx::generate_stream_words(random, 0U, jarvisx::StreamPattern::Random, 42U);
    const bool random_raw = jarvisx::AdaptiveWordCodec::encode(random, encoded);
    require(random_raw, "incompressible deterministic random data should fall back to raw");
    jarvisx::AdaptiveWordCodec::decode(encoded, random.size(), decoded);
    require(decoded == random, "raw fallback round-trip failed");
}

void bounded_stream_smoke() {
    jarvisx::Stream1GiBConfig config;
    config.target_bytes = 8U * 1024U * 1024U;
    config.chunk_bytes = 1U * 1024U * 1024U;
    config.pattern = jarvisx::StreamPattern::Sparse3D;
    config.window_ms = 20U;
    config.configured_l3_bytes = 16U * 1024U * 1024U;
    const auto metrics = jarvisx::run_stream_1gib(config);
    require(metrics.processed_bytes == config.target_bytes, "smoke target byte count mismatch");
    require(metrics.exact_round_trip, "smoke exact round-trip failed");
    require(metrics.codec_fixed_point, "smoke codec fixed point failed");
    require(metrics.compression_ratio > 1.0, "sparse smoke stream should compress");
    require(metrics.logical_vectors_512 == config.target_bytes / 64U,
            "logical 512-bit vector count mismatch");
    require(metrics.hot_path_reallocations == 0U,
            "reusable hot path should not reallocate after warm-up reservation");
    require(metrics.reusable_working_set_bytes > 0U, "working set telemetry missing");
    require(metrics.throughput_gbps > 0.0 && std::isfinite(metrics.throughput_gbps),
            "throughput telemetry must be finite and positive");
}

void zero_stream_is_highly_compressible() {
    jarvisx::Stream1GiBConfig config;
    config.target_bytes = 1U * 1024U * 1024U;
    config.chunk_bytes = 1U * 1024U * 1024U;
    config.pattern = jarvisx::StreamPattern::Zero;
    config.window_ms = 10U;
    const auto metrics = jarvisx::run_stream_1gib(config);
    require(metrics.compression_ratio > 1000.0,
            "all-zero stream should compress by more than three orders of magnitude");
    require(metrics.exact_round_trip, "zero stream round-trip failed");
}

} // namespace

int main() {
    try {
        canonical_volume_mapping();
        codec_round_trip_and_fallback();
        bounded_stream_smoke();
        zero_stream_is_highly_compressible();
        std::cout << "1 GiB 3D stream tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "1 GiB 3D stream test failure: " << error.what() << '\n';
        return 1;
    }
}
