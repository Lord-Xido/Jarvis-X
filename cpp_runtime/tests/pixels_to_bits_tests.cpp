#include "jarvisx/pixels_to_bits_codec.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void test_exact_8kb_layout_and_bounds() {
    static_assert(jarvisx::dm8kb::layout_is_exact(), "layout must be exact");
    jarvisx::dm8kb::Container container;
    assert(container.bytes().size() == 8192U);
    assert(jarvisx::dm8kb::region_offset(jarvisx::dm8kb::Region::Integrity) == 0x1F00U);
    assert(jarvisx::dm8kb::region_size(jarvisx::dm8kb::Region::Integrity) == 256U);

    bool rejected = false;
    try {
        container.write(8192U, 1U);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);
}

void test_toroidal_indexing() {
    using jarvisx::dm8kb::Container;
    assert(Container::torus_index(0, 0, 0) == 0U);
    assert(Container::torus_index(8, 0, 0) == 0U);
    assert(Container::torus_index(-1, 0, 0) == 7U);
    assert(Container::torus_index(0, -1, 0) == 56U);
    assert(Container::torus_index(0, 0, -1) == 448U);
    assert(Container::torus_index(8, 8, 8) == 0U);
}

void test_8k120_profile_arithmetic() {
    const auto profile = jarvisx::pixels::uhd8k120_profile();
    assert(profile.pixels == 33177600ULL);
    assert(profile.raw_bits_per_frame == 995328000ULL);
    assert(profile.packed_bytes_per_frame == 124416000ULL);
    assert(std::fabs(profile.packed_gbytes_per_second - 14.92992) < 1.0e-9);
    assert(std::fabs(profile.frame_budget_ms - (1000.0 / 120.0)) < 1.0e-12);
}

void test_lossless_roundtrip_partial_tiles() {
    const auto frame = jarvisx::pixels::make_gradient_frame(13U, 10U);
    jarvisx::pixels::CodecConfig config;
    config.lossless = true;
    config.max_tile_mse = 0.0;
    config.min_tile_psnr_db = 0.0;
    jarvisx::pixels::PixelsToBitsCodec codec(config);
    const auto result = codec.encode(frame);

    assert(result.reconstructed.width == frame.width);
    assert(result.reconstructed.height == frame.height);
    assert(result.reconstructed.rgb == frame.rgb);
    assert(result.metrics.mse == 0.0);
    assert(std::isinf(result.metrics.psnr_db));
    assert(result.metrics.hbar_semantic_visual == 0.0);
    assert(result.final_container.bytes().size() == 8192U);
    assert(result.final_container.receipt().accepted);
    assert(result.final_container.epoch() == result.metrics.tiles);

    const auto decoded_again = jarvisx::pixels::PixelsToBitsCodec::decode(result.bitstream);
    assert(decoded_again.rgb == frame.rgb);
}

void test_near_lossless_quality_gate_and_compression() {
    const auto frame = jarvisx::pixels::make_gradient_frame(32U, 24U);
    jarvisx::pixels::CodecConfig config;
    config.lossless = false;
    config.max_shift = 4U;
    config.max_tile_mse = 100.0;
    config.min_tile_psnr_db = 40.0;
    jarvisx::pixels::PixelsToBitsCodec codec(config);
    const auto result = codec.encode(frame);

    assert(result.metrics.mse <= 100.0 + 1.0e-9);
    assert(result.metrics.psnr_db >= 40.0 - 1.0e-9);
    assert(result.metrics.compression_ratio > 1.0);
    assert(result.metrics.average_tile_shift > 0.0);
    assert(result.metrics.hbar_semantic_visual >= 0.0);
    assert(result.metrics.hbar_semantic_visual < 0.02);
}

void test_deterministic_stream() {
    const auto frame = jarvisx::pixels::make_gradient_frame(16U, 16U);
    jarvisx::pixels::CodecConfig config;
    config.max_shift = 3U;
    config.max_tile_mse = 80.0;
    config.min_tile_psnr_db = 40.0;
    jarvisx::pixels::PixelsToBitsCodec left(config);
    jarvisx::pixels::PixelsToBitsCodec right(config);
    const auto a = left.encode(frame);
    const auto b = right.encode(frame);
    assert(a.bitstream == b.bitstream);
    assert(a.reconstructed.rgb == b.reconstructed.rgb);
    assert(a.final_container.bytes() == b.final_container.bytes());
}

void test_corruption_fails_closed() {
    const auto frame = jarvisx::pixels::make_gradient_frame(8U, 8U);
    jarvisx::pixels::CodecConfig config;
    config.lossless = true;
    config.max_tile_mse = 0.0;
    config.min_tile_psnr_db = 0.0;
    jarvisx::pixels::PixelsToBitsCodec codec(config);
    auto result = codec.encode(frame);
    assert(result.bitstream.size() > 24U);
    result.bitstream[20U] = static_cast<std::uint8_t>(result.bitstream[20U] ^ 0x01U);

    bool rejected = false;
    try {
        (void)jarvisx::pixels::PixelsToBitsCodec::decode(result.bitstream);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}

void test_shadow_and_receipt_are_bounded() {
    jarvisx::dm8kb::Container container;
    container.set_microcode(jarvisx::media8::default_media_program());
    std::vector<std::uint8_t> candidate(1024U, 0xA5U);
    container.stage_shadow(candidate);
    assert(container.shadow_size() == 1024U);

    bool rejected = false;
    try {
        container.stage_shadow(std::vector<std::uint8_t>(1025U, 0U));
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    test_exact_8kb_layout_and_bounds();
    test_toroidal_indexing();
    test_8k120_profile_arithmetic();
    test_lossless_roundtrip_partial_tiles();
    test_near_lossless_quality_gate_and_compression();
    test_deterministic_stream();
    test_corruption_fails_closed();
    test_shadow_and_receipt_are_bounded();
    std::cout << "DM-vOmegaXi+ pixels-to-bits tests passed\n";
    return 0;
}
