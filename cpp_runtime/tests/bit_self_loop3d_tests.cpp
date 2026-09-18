#include "jarvisx/bit_self_loop3d.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

using jarvisx::bit_self_loop3d::CandidateMetrics;
using jarvisx::bit_self_loop3d::VoxelFields;
using jarvisx::bit_self_loop3d::Word;
using jarvisx::bit_self_loop3d::accept_candidate;
using jarvisx::bit_self_loop3d::commit_or_rollback;
using jarvisx::bit_self_loop3d::fold_octet;
using jarvisx::bit_self_loop3d::fold_refine_reconstruct;
using jarvisx::bit_self_loop3d::hamming_distance;
using jarvisx::bit_self_loop3d::pack;
using jarvisx::bit_self_loop3d::reconstruct_from_xor_residuals;
using jarvisx::bit_self_loop3d::refine_toward;
using jarvisx::bit_self_loop3d::unpack;

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

} // namespace

int main() {
    test_pack_round_trip();
    test_pack_rejects_wide_fields();
    test_identical_octet_contracts_to_itself();
    test_residual_shell_is_exact();
    test_hamming_refinement_is_monotone();
    test_fold_refine_keeps_lossless_shell();
    test_candidate_gate_and_rollback();

    std::cout << "bit_self_loop3d regressions: ok\n";
    return 0;
}
