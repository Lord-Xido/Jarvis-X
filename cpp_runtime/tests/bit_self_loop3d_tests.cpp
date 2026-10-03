#include "jarvisx/bit_self_loop3d.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

using jarvisx::bit_self_loop3d::BitFixedPointResidual;
using jarvisx::bit_self_loop3d::CandidateMetrics;
using jarvisx::bit_self_loop3d::VerificationBitmap1M;
using jarvisx::bit_self_loop3d::VoxelFields;
using jarvisx::bit_self_loop3d::Word;
using jarvisx::bit_self_loop3d::accept_candidate;
using jarvisx::bit_self_loop3d::commit_or_rollback;
using jarvisx::bit_self_loop3d::dyadic_axis_bits;
using jarvisx::bit_self_loop3d::fixed_point_residual;
using jarvisx::bit_self_loop3d::fold_octet;
using jarvisx::bit_self_loop3d::fold_refine_reconstruct;
using jarvisx::bit_self_loop3d::hamming_distance;
using jarvisx::bit_self_loop3d::hysteretic_activity_gate;
using jarvisx::bit_self_loop3d::is_exact_fixed_point;
using jarvisx::bit_self_loop3d::kMillionPathways;
using jarvisx::bit_self_loop3d::masked_overwrite;
using jarvisx::bit_self_loop3d::pack;
using jarvisx::bit_self_loop3d::reconstruct_from_xor_residuals;
using jarvisx::bit_self_loop3d::refine_toward;
using jarvisx::bit_self_loop3d::saturating_half_extent;
using jarvisx::bit_self_loop3d::unpack;
using jarvisx::bit_self_loop3d::verified_mux;

namespace {

void test_pack_round_trip() {
    const VoxelFields fields{
        0xABu,
        0x0ABCu,
        0x7Fu,
        0x12u,
        0x0789u,
        0xBEEFu
    };

    const Word word = pack(fields);
    const VoxelFields decoded = unpack(word);
    assert(decoded == fields);
    assert(pack(decoded) == word);
}

void test_pack_rejects_wide_fields() {
    bool residual_failed = false;
    try {
        (void)pack(VoxelFields{0u, 0x1000u, 0u, 0u, 0u, 0u});
    } catch (const std::out_of_range&) {
        residual_failed = true;
    }
    assert(residual_failed);

    bool feature_failed = false;
    try {
        (void)pack(VoxelFields{0u, 0u, 0u, 0u, 0x1000u, 0u});
    } catch (const std::out_of_range&) {
        feature_failed = true;
    }
    assert(feature_failed);
}

void test_identical_octet_contracts_to_itself() {
    constexpr Word value = 0xD15EA5E5CAFEBEEFull;
    const std::array<Word, 8> children{
        value, value, value, value, value, value, value, value
    };

    const auto folded = fold_octet(children);
    assert(folded.latent == value);

    const auto reconstructed =
        reconstruct_from_xor_residuals(folded.latent, folded.residuals);
    assert(reconstructed == children);
}

void test_residual_shell_is_exact() {
    const std::array<Word, 8> children{
        0x0000000000000000ull,
        0xFFFFFFFFFFFFFFFFull,
        0x0123456789ABCDEFull,
        0xFEDCBA9876543210ull,
        0xAAAAAAAAAAAAAAAAull,
        0x5555555555555555ull,
        0x0F0F0F0F0F0F0F0Full,
        0xF0F0F0F0F0F0F0F0ull
    };

    const auto folded = fold_octet(children);
    const auto reconstructed =
        reconstruct_from_xor_residuals(folded.latent, folded.residuals);
    assert(reconstructed == children);
}

void test_hamming_refinement_is_monotone() {
    constexpr Word initial = 0x0000000000000000ull;
    constexpr Word target = 0xF0F0F0F0F0F0F0F0ull;

    const auto refined = refine_toward(initial, target, 8u, 8u);
    assert(refined.monotonic);
    assert(refined.initial_distance == 32u);
    assert(refined.final_distance == 0u);
    assert(refined.value == target);
    assert(hamming_distance(refined.value, target) == 0u);
}

void test_fold_refine_keeps_lossless_shell() {
    const std::array<Word, 8> children{
        0x0101010101010101ull,
        0x0303030303030303ull,
        0x0707070707070707ull,
        0x0F0F0F0F0F0F0F0Full,
        0x1F1F1F1F1F1F1F1Full,
        0x3F3F3F3F3F3F3F3Full,
        0x7F7F7F7F7F7F7F7Full,
        0xFFFFFFFFFFFFFFFFull
    };
    constexpr Word target = 0xAAAAAAAAAAAAAAAAull;

    const auto cycle = fold_refine_reconstruct(children, target, 16u, 4u);
    assert(cycle.monotonic);
    assert(cycle.target_distance_after <= cycle.target_distance_before);
    assert(cycle.reconstructed == children);
}

void test_candidate_gate_and_rollback() {
    const CandidateMetrics baseline{10.0, true, true};
    const CandidateMetrics improved{8.5, true, true};
    const CandidateMetrics weak{9.95, true, true};
    const CandidateMetrics invalid{1.0, true, false};

    assert(accept_candidate(baseline, improved, 1.0));
    assert(!accept_candidate(baseline, weak, 0.1));
    assert(!accept_candidate(baseline, invalid, 0.0));

    const CandidateMetrics nonfinite{
        std::numeric_limits<double>::infinity(), true, true
    };
    assert(!accept_candidate(baseline, nonfinite, 0.0));

    constexpr Word old_state = 0x1111111111111111ull;
    constexpr Word candidate = 0x2222222222222222ull;
    assert(commit_or_rollback(old_state, candidate, true) == candidate);
    assert(commit_or_rollback(old_state, candidate, false) == old_state);
}

void test_exact_fixed_point_residual_is_xor_popcount() {
    constexpr Word current = 0b10101100u;
    constexpr Word next = 0b10100101u;

    const BitFixedPointResidual residual = fixed_point_residual(current, next, 8u);
    assert(residual.xor_delta == 0b00001001u);
    assert(residual.changed_bits == 2u);
    assert(residual.normalized_fraction == 0.25);
    assert(!residual.exact);
    assert(!is_exact_fixed_point(current, next));

    const auto exact = fixed_point_residual(current, current, 8u);
    assert(exact.xor_delta == 0u);
    assert(exact.changed_bits == 0u);
    assert(exact.normalized_fraction == 0.0);
    assert(exact.exact);
    assert(is_exact_fixed_point(current, current));
}

void test_fixed_point_residual_rejects_invalid_width() {
    bool failed = false;
    try {
        (void)fixed_point_residual(0u, 0u, 0u);
    } catch (const std::out_of_range&) {
        failed = true;
    }
    assert(failed);
}

void test_masked_memory_update_and_verified_mux() {
    constexpr Word old_state = 0b10101010u;
    constexpr Word new_info = 0b11001100u;
    constexpr Word write_mask = 0b11110000u;

    assert(masked_overwrite(old_state, new_info, write_mask) == 0b11001010u);

    constexpr Word baseline = 0x1111111111111111ull;
    constexpr Word candidate = 0xAAAAAAAAAAAAAAAAull;
    assert(verified_mux(false, baseline, candidate) == baseline);
    assert(verified_mux(true, baseline, candidate) == candidate);
}

void test_dyadic_inward_extent_saturates_at_one() {
    std::uint64_t extent = 32u;
    const std::array<std::uint64_t, 6> expected{32u, 16u, 8u, 4u, 2u, 1u};
    const std::array<std::uint32_t, 6> bits{5u, 4u, 3u, 2u, 1u, 0u};

    for (std::size_t i = 0u; i < expected.size(); ++i) {
        assert(extent == expected[i]);
        assert(dyadic_axis_bits(extent) == bits[i]);
        extent = saturating_half_extent(extent);
    }

    assert(extent == 1u);
    assert(saturating_half_extent(1u) == 1u);
}

void test_hysteretic_pruning_prevents_threshold_chatter() {
    assert(hysteretic_activity_gate(false, 11u, 4u, 10u));
    assert(hysteretic_activity_gate(true, 5u, 4u, 10u));
    assert(!hysteretic_activity_gate(true, 3u, 4u, 10u));
    assert(!hysteretic_activity_gate(false, 8u, 4u, 10u));

    bool failed = false;
    try {
        (void)hysteretic_activity_gate(false, 1u, 10u, 10u);
    } catch (const std::out_of_range&) {
        failed = true;
    }
    assert(failed);
}

void test_million_pathway_verification_bitmap() {
    VerificationBitmap1M bitmap;
    assert(bitmap.pass_count() == 0u);
    assert(!bitmap.all_pass());

    bitmap.set_all(true);
    assert(bitmap.pass_count() == kMillionPathways);
    assert(bitmap.all_pass());

    bitmap.set(123456u, false);
    assert(!bitmap.get(123456u));
    assert(bitmap.pass_count() == kMillionPathways - 1u);
    assert(!bitmap.all_pass());

    bitmap.set(123456u, true);
    assert(bitmap.get(123456u));
    assert(bitmap.pass_count() == kMillionPathways);
    assert(bitmap.all_pass());
}

} // namespace

int main() {
    test_pack_round_trip();
    test_pack_rejects_wide_fields();
    test_identical_octet_contracts_to_itself();
    test_residual_shell_is_exact();
    test_hamming_refinement_is_monotone();
    test_fold_refine_keeps_lossless_shell();
    test_candidate_gate_and_rollback();
    test_exact_fixed_point_residual_is_xor_popcount();
    test_fixed_point_residual_rejects_invalid_width();
    test_masked_memory_update_and_verified_mux();
    test_dyadic_inward_extent_saturates_at_one();
    test_hysteretic_pruning_prevents_threshold_chatter();
    test_million_pathway_verification_bitmap();

    std::cout << "bit_self_loop3d regressions: ok\n";
    return 0;
}
