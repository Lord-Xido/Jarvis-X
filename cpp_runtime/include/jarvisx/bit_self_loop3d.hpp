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

constexpr std::size_t kMillionPathways = 1'000'000u;
constexpr std::size_t kBitsPerWord = 64u;
constexpr std::size_t kMillionBitmapWords = kMillionPathways / kBitsPerWord;
static_assert(kMillionPathways % kBitsPerWord == 0u);

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

struct BitFixedPointResidual {
    Word current{};
    Word next{};
    Word xor_delta{};
    std::uint32_t changed_bits{};
    double normalized_fraction{};
    bool exact{};
};

inline BitFixedPointResidual fixed_point_residual(
    Word current,
    Word next,
    std::uint32_t logical_bits = 64u) {
    if (logical_bits == 0u || logical_bits > 64u) {
        throw std::out_of_range("logical_bits must be in [1,64]");
    }

    Word mask = ~Word{0};
    if (logical_bits < 64u) {
        mask = (Word{1} << logical_bits) - Word{1};
    }

    const Word delta = (current ^ next) & mask;
    const std::uint32_t changed = popcount(delta);
    return {
        current,
        next,
        delta,
        changed,
        static_cast<double>(changed) / static_cast<double>(logical_bits),
        changed == 0u
    };
}

inline bool is_exact_fixed_point(Word current, Word next) noexcept {
    return (current ^ next) == 0u;
}

inline Word masked_overwrite(
    Word old_state,
    Word new_information,
    Word write_mask) noexcept {
    return (old_state & ~write_mask) | (new_information & write_mask);
}

inline Word verified_mux(
    bool verified,
    Word baseline,
    Word candidate) noexcept {
    const Word select = Word{0} - static_cast<Word>(verified);
    return (baseline & ~select) | (candidate & select);
}

inline std::uint64_t saturating_half_extent(std::uint64_t extent) {
    if (extent == 0u) {
        throw std::out_of_range("extent must be positive");
    }
    return extent == 1u ? 1u : (extent >> 1u);
}

inline std::uint32_t dyadic_axis_bits(std::uint64_t extent) {
    if (extent == 0u || (extent & (extent - 1u)) != 0u) {
        throw std::out_of_range("extent must be a positive power of two");
    }

    std::uint32_t bits = 0u;
    while (extent > 1u) {
        extent >>= 1u;
        ++bits;
    }
    return bits;
}

inline bool hysteretic_activity_gate(
    bool currently_active,
    std::uint32_t error,
    std::uint32_t threshold_off,
    std::uint32_t threshold_on) {
    if (threshold_on <= threshold_off) {
        throw std::out_of_range("threshold_on must exceed threshold_off");
    }

    if (currently_active) {
        return error >= threshold_off;
    }
    return error > threshold_on;
}

class VerificationBitmap1M {
public:
    void clear() noexcept {
        words_.fill(Word{0});
    }

    void set_all(bool passed) noexcept {
        words_.fill(passed ? ~Word{0} : Word{0});
    }

    void set(std::size_t index, bool passed) {
        if (index >= kMillionPathways) {
            throw std::out_of_range("verification pathway index out of range");
        }
        const std::size_t word_index = index / kBitsPerWord;
        const std::size_t bit_index = index % kBitsPerWord;
        const Word mask = Word{1} << bit_index;
        if (passed) {
            words_[word_index] |= mask;
        } else {
            words_[word_index] &= ~mask;
        }
    }

    bool get(std::size_t index) const {
        if (index >= kMillionPathways) {
            throw std::out_of_range("verification pathway index out of range");
        }
        const std::size_t word_index = index / kBitsPerWord;
        const std::size_t bit_index = index % kBitsPerWord;
        return (words_[word_index] & (Word{1} << bit_index)) != 0u;
    }

    std::size_t pass_count() const noexcept {
        std::size_t total = 0u;
        for (Word word : words_) {
            total += popcount(word);
        }
        return total;
    }

    bool all_pass() const noexcept {
        for (Word word : words_) {
            if (word != ~Word{0}) {
                return false;
            }
        }
        return true;
    }

    const std::array<Word, kMillionBitmapWords>& words() const noexcept {
        return words_;
    }

private:
    std::array<Word, kMillionBitmapWords> words_{};
};

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
    return verified_mux(verified, baseline, candidate);
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
