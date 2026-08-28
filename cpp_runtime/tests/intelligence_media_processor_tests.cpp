#include "jarvisx/intelligence_media_processor.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace jarvisx::media8;

namespace {

std::array<std::uint8_t, kTileNodes> ramp_tile() {
    std::array<std::uint8_t, kTileNodes> bytes{};
    for (std::size_t i = 0U; i < bytes.size(); ++i) {
        bytes[i] = static_cast<std::uint8_t>((i * 37U + (i >> 2U)) & 0xFFU);
    }
    return bytes;
}

void test_centered_byte_round_trip() {
    for (const std::uint8_t value : {0U, 1U, 64U, 127U, 128U, 200U, 254U, 255U}) {
        const std::uint8_t recovered = centered_to_byte(byte_to_centered(value));
        assert(recovered == value || (value == 0U && recovered == 0U));
    }
}

void test_boundary_convolution_is_defined() {
    VCLTile8 tile;
    std::array<std::uint8_t, kTileNodes> neutral{};
    neutral.fill(128U);
    tile.ingest_tile(neutral);
    tile.ingest_plane(0U, 200U);
    tile.conv3d(1U, 0U);
    const auto& nodes = tile.nodes();
    const int corner = static_cast<int>(nodes[tile_index(0U, 0U, 0U)].state);
    const int adjacent = static_cast<int>(nodes[tile_index(0U, 0U, 1U)].state);
    assert(corner != 0);
    assert(adjacent != 0);
}

void test_true_entropy_distinguishes_uniform_and_varied_tiles() {
    VCLTile8 tile;
    std::array<std::uint8_t, kTileNodes> uniform{};
    uniform.fill(77U);
    tile.ingest_tile(uniform);
    const double uniform_entropy = tile.eval_entropy(0U, 1U);
    assert(std::fabs(uniform_entropy) < 1.0e-12);

    auto varied = ramp_tile();
    tile.ingest_tile(varied);
    const double varied_entropy = tile.eval_entropy(0U, 1U);
    assert(varied_entropy > 0.5);
    assert(varied_entropy <= 1.0);
}

void test_vcl_program_executes_to_sync_lock() {
    VCLTile8 tile;
    const auto bytes = ramp_tile();
    tile.ingest_tile(bytes);
    VCLBVM8 vm(tile);
    const TileMetrics metrics = vm.execute(default_media_program());
    assert(metrics.synchronized);
    assert(metrics.active_nodes <= kTileNodes);
    assert(std::isfinite(metrics.reconstruction_mse));
    assert(std::isfinite(metrics.normalized_entropy));
    assert(metrics.evolution_commits + metrics.evolution_rollbacks == 1U);
}

void test_documented_vcl_cycle_executes() {
    VCLTile8 tile;
    std::array<std::uint8_t, kTileNodes> initial{};
    initial.fill(128U);
    tile.ingest_tile(initial);
    const std::vector<std::uint8_t> program = {
        0x10U, 0x00U, 0xC8U,
        0x40U, 0x01U, 0x04U,
        0x30U, 0x01U, 0x40U,
        0x20U, 0x00U, 0x02U,
        0x50U, 0x00U, 0x10U,
        0x60U, 0x01U,
        0x80U, 0x02U,
        0x70U, 0x05U, 0x07U,
        0xFFU,
    };
    VCLBVM8 vm(tile);
    const TileMetrics metrics = vm.execute(program);
    assert(metrics.synchronized);
    assert(std::isfinite(metrics.reconstruction_mse));
}

void test_truncated_bytecode_is_rejected() {
    VCLTile8 tile;
    const std::vector<std::uint8_t> truncated = {0x40U, 0x01U};
    VCLBVM8 vm(tile);
    bool threw = false;
    try {
        (void)vm.execute(truncated);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_lambda_fallback_preserves_media_bytes_and_adaptive_state() {
    ProcessorConfig config;
    config.max_output_mse = 0.0;
    config.fallback_on_regression = true;
    DrMoagiIntelligenceMediaProcessor processor(config);

    std::vector<std::uint8_t> input(1500U);
    for (std::size_t i = 0U; i < input.size(); ++i) {
        input[i] = static_cast<std::uint8_t>((i * 73U + 29U) & 0xFFU);
    }
    const AdaptiveSnapshot before = processor.tile().adaptive_snapshot();
    const ProcessorResult result = processor.process(input, MediaModality::Visual);
    const AdaptiveSnapshot after = processor.tile().adaptive_snapshot();

    assert(result.bytes == input);
    assert(result.metrics.tiles == 3U);
    assert(result.metrics.accepted_tiles + result.metrics.rejected_tiles == 3U);
    assert(result.metrics.rejected_tiles > 0U);
    assert(result.metrics.average_output_mse == 0.0);
    assert(result.metrics.hbar_semantic == 0.0);
    assert(before.weights == after.weights);
    assert(before.omega == after.omega);
}

void test_processor_is_deterministic() {
    std::vector<std::uint8_t> input(2048U);
    for (std::size_t i = 0U; i < input.size(); ++i) {
        input[i] = static_cast<std::uint8_t>((i * 17U + (i >> 1U)) & 0xFFU);
    }
    DrMoagiIntelligenceMediaProcessor left;
    DrMoagiIntelligenceMediaProcessor right;
    const auto left_result = left.process(input, MediaModality::Audio);
    const auto right_result = right.process(input, MediaModality::Audio);
    assert(left_result.bytes == right_result.bytes);
    assert(left_result.metrics.tiles == right_result.metrics.tiles);
    assert(left_result.metrics.accepted_tiles == right_result.metrics.accepted_tiles);
    assert(left_result.metrics.rejected_tiles == right_result.metrics.rejected_tiles);
    assert(left_result.metrics.average_candidate_mse == right_result.metrics.average_candidate_mse);
    assert(left_result.metrics.hbar_semantic == right_result.metrics.hbar_semantic);
}

} // namespace

int main() {
    test_centered_byte_round_trip();
    test_boundary_convolution_is_defined();
    test_true_entropy_distinguishes_uniform_and_varied_tiles();
    test_vcl_program_executes_to_sync_lock();
    test_documented_vcl_cycle_executes();
    test_truncated_bytecode_is_rejected();
    test_lambda_fallback_preserves_media_bytes_and_adaptive_state();
    test_processor_is_deterministic();
    std::cout << "intelligence_media_processor_tests: OK\n";
    return 0;
}
