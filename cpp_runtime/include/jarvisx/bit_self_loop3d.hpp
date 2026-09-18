#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace jarvisx::bit_self_loop3d {

using Word = std::uint64_t;

constexpr std::uint16_t kPayloadMask = 0xFFFFu;
constexpr std::uint16_t kFeatureMask = 0x0FFFu;
constexpr std::uint8_t kByteMask = 0xFFu;
constexpr std::uint16_t kResidualMask = 0x0FFFu;

constexpr std::uint32_t kPayloadShift = 0u;
constexpr std::uint32_t kFeatureShift = 16u;
constexpr std::uint32_t kMemoryShift = 28u;
constexpr std::uint32_t kActivationShift = 36u;
constexpr std::uint32_t kResidualShift = 44u;
constexpr std::uint32_t kOpcodeShift = 56u;

struct VoxelFields {
    std::uint8_t opcode{};
    std::uint16_t residual{};
    std::uint8_t activation{};
    std::uint8_t memory{};
    std::uint16_t feature{};
    std::uint16_t payload{};
};

inline bool operator==(const VoxelFields& lhs, const VoxelFields& rhs) noexcept {
    return lhs.opcode == rhs.opcode &&
           lhs.residual == rhs.residual &&
           lhs.activation == rhs.activation &&
           lhs.memory == rhs.memory &&
           lhs.feature == rhs.feature &&
           lhs.payload == rhs.payload;
}

inline Word pack(const VoxelFields& fields) {
    if (fields.residual > kResidualMask) {
        throw std::out_of_range("residual exceeds 12-bit field");
    }
    if (fields.feature > kFeatureMask) {
        throw std::out_of_range("feature exceeds 12-bit field");
    }

    return (static_cast<Word>(fields.payload & kPayloadMask) << kPayloadShift) |
           (static_cast<Word>(fields.feature & kFeatureMask) << kFeatureShift) |
           (static_cast<Word>(fields.memory & kByteMask) << kMemoryShift) |
           (static_cast<Word>(fields.activation & kByteMask) << kActivationShift) |
           (static_cast<Word>(fields.residual & kResidualMask) << kResidualShift) |
           (static_cast<Word>(fields.opcode & kByteMask) << kOpcodeShift);
}

inline VoxelFields unpack(Word word) noexcept {
    return {
        static_cast<std::uint8_t>((word >> kOpcodeShift) & kByteMask),
        static_cast<std::uint16_t>((word >> kResidualShift) & kResidualMask),
        static_cast<std::uint8_t>((word >> kActivationShift) & kByteMask),
        static_cast<std::uint8_t>((word >> kMemoryShift) & kByteMask),
        static_cast<std::uint16_t>((word >> kFeatureShift) & kFeatureMask),
        static_cast<std::uint16_t>((word >> kPayloadShift) & kPayloadMask)
    };
}

inline std::uint32_t popcount(Word value) noexcept {
    std::uint32_t count = 0u;
    while (value != 0u) {
        value &= (value - 1u);
        ++count;
    }
    return count;
}

inline std::uint32_t hamming_distance(Word lhs, Word rhs) noexcept {
    return popcount(lhs ^ rhs);
}

inline Word majority_contract(const std::array<Word, 8>& children) noexcept {
    Word latent = 0u;
    for (std::uint32_t bit = 0u; bit < 64u; ++bit) {
        const Word mask = Word{1} << bit;
        std::uint32_t ones = 0u;
        for (Word child : children) {
            if ((child & mask) != 0u) {
                ++ones;
            }
        }
        if (ones >= 4u) {
            latent |= mask;
        }
    }
    return latent;
}

inline std::array<Word, 8> xor_residuals(
    const std::array<Word, 8>& children,
    Word latent) noexcept {
    std::array<Word, 8> residuals{};
    for (std::size_t i = 0u; i < children.size(); ++i) {
        residuals[i] = children[i] ^ latent;
    }
    return residuals;
}

inline std::array<Word, 8> reconstruct_from_xor_residuals(
    Word latent,
    const std::array<Word, 8>& residuals) noexcept {
    std::array<Word, 8> reconstructed{};
    for (std::size_t i = 0u; i < residuals.size(); ++i) {
        reconstructed[i] = latent ^ residuals[i];
    }
    return reconstructed;
}

struct FoldResult {
    Word latent{};
    std::array<Word, 8> residuals{};
};

inline FoldResult fold_octet(const std::array<Word, 8>& children) noexcept {
    const Word latent = majority_contract(children);
    return {latent, xor_residuals(children, latent)};
}

struct RefinementResult {
    Word value{};
    std::uint32_t steps{};
    std::uint32_t initial_distance{};
    std::uint32_t final_distance{};
    bool monotonic{true};
};

inline RefinementResult refine_toward(
    Word initial,
    Word target,
    std::uint32_t max_steps,
    std::uint32_t flips_per_step = 8u) noexcept {
    RefinementResult result{};
    result.value = initial;
    result.initial_distance = hamming_distance(initial, target);
    result.final_distance = result.initial_distance;

    if (flips_per_step == 0u) {
        return result;
    }

    for (std::uint32_t step = 0u; step < max_steps; ++step) {
        Word diff = result.value ^ target;
        if (diff == 0u) {
            break;
        }

        const std::uint32_t before = hamming_distance(result.value, target);
        for (std::uint32_t i = 0u; i < flips_per_step && diff != 0u; ++i) {
            const Word lowest = diff & (~diff + Word{1});
            result.value ^= lowest;
            diff ^= lowest;
        }

        const std::uint32_t after = hamming_distance(result.value, target);
        if (after > before) {
            result.monotonic = false;
            break;
        }

        result.final_distance = after;
        ++result.steps;
    }

    result.final_distance = hamming_distance(result.value, target);
    return result;
}

struct CandidateMetrics {
    double objective{};
    bool finite{true};
    bool invariant_ok{true};
};

inline bool accept_candidate(
    const CandidateMetrics& baseline,
    const CandidateMetrics& candidate,
    double minimum_improvement = 0.0) noexcept {
    if (!baseline.finite || !candidate.finite || !candidate.invariant_ok) {
        return false;
    }
    if (!std::isfinite(baseline.objective) || !std::isfinite(candidate.objective)) {
        return false;
    }
    return candidate.objective <= baseline.objective - minimum_improvement;
}

inline Word commit_or_rollback(
    Word baseline,
    Word candidate,
    bool verified) noexcept {
    return verified ? candidate : baseline;
}

struct SelfFoldCycle {
    Word latent_before{};
    Word latent_after{};
    std::array<Word, 8> residuals{};
    std::array<Word, 8> reconstructed{};
    std::uint32_t target_distance_before{};
    std::uint32_t target_distance_after{};
    bool monotonic{true};
};

inline SelfFoldCycle fold_refine_reconstruct(
    const std::array<Word, 8>& children,
    Word refinement_target,
    std::uint32_t max_steps,
    std::uint32_t flips_per_step = 8u) noexcept {
    const FoldResult folded = fold_octet(children);
    const RefinementResult refined =
        refine_toward(folded.latent, refinement_target, max_steps, flips_per_step);

    // The residual shell belongs to latent_before. Reconstruct against that
    // reference so the lossless shell invariant remains exact even when the
    // inner latent is subsequently refined.
    const auto reconstructed =
        reconstruct_from_xor_residuals(folded.latent, folded.residuals);

    return {
        folded.latent,
        refined.value,
        folded.residuals,
        reconstructed,
        refined.initial_distance,
        refined.final_distance,
        refined.monotonic
    };
}

} // namespace jarvisx::bit_self_loop3d
