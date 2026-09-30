#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jarvisx::world::bitwise500 {

constexpr std::uint32_t kWorldEdge = 500U;
constexpr std::uint64_t kWorldAgentCount =
    static_cast<std::uint64_t>(kWorldEdge) * kWorldEdge * kWorldEdge;
constexpr std::uint32_t kLatentDimensions = 16U;
constexpr std::uint32_t kBitsPerFp32 = 32U;
constexpr std::uint32_t kBitsPerAgent = kLatentDimensions * kBitsPerFp32;
constexpr std::uint64_t kWorldLatentBits = kWorldAgentCount * kBitsPerAgent;
constexpr std::uint64_t kWorldLatentBytes = kWorldLatentBits / 8ULL;
constexpr long double kWorldLatentGiB =
    static_cast<long double>(kWorldLatentBytes) / 1073741824.0L;

struct Coord500 {
    std::uint16_t x{};
    std::uint16_t y{};
    std::uint16_t z{};

    bool operator==(const Coord500& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

inline std::uint64_t linear_address(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    if (x >= kWorldEdge || y >= kWorldEdge || z >= kWorldEdge) {
        throw std::out_of_range("500^3 world coordinate is outside [0,499]");
    }
    return static_cast<std::uint64_t>(x) +
           static_cast<std::uint64_t>(kWorldEdge) * static_cast<std::uint64_t>(y) +
           static_cast<std::uint64_t>(kWorldEdge) * static_cast<std::uint64_t>(kWorldEdge) *
               static_cast<std::uint64_t>(z);
}

inline Coord500 coordinate_from_address(std::uint64_t address) {
    if (address >= kWorldAgentCount) {
        throw std::out_of_range("500^3 world address exceeds the agent domain");
    }
    const std::uint64_t plane = static_cast<std::uint64_t>(kWorldEdge) * kWorldEdge;
    const auto z = static_cast<std::uint16_t>(address / plane);
    const std::uint64_t remainder = address % plane;
    const auto y = static_cast<std::uint16_t>(remainder / kWorldEdge);
    const auto x = static_cast<std::uint16_t>(remainder % kWorldEdge);
    return {x, y, z};
}

inline std::vector<std::uint8_t> pack_bits_lsb_first(const std::vector<std::uint8_t>& bits) {
    std::vector<std::uint8_t> bytes((bits.size() + 7U) / 8U, 0U);
    for (std::size_t i = 0U; i < bits.size(); ++i) {
        if (bits[i] > 1U) throw std::invalid_argument("bit stream contains a value other than 0 or 1");
        bytes[i / 8U] |= static_cast<std::uint8_t>(bits[i] << (i % 8U));
    }
    return bytes;
}

inline std::vector<std::uint8_t> unpack_bits_lsb_first(
    const std::vector<std::uint8_t>& bytes, std::size_t bit_count) {
    if (bit_count > bytes.size() * 8U) {
        throw std::invalid_argument("requested bit count exceeds packed byte capacity");
    }
    std::vector<std::uint8_t> bits(bit_count, 0U);
    for (std::size_t i = 0U; i < bit_count; ++i) {
        bits[i] = static_cast<std::uint8_t>((bytes[i / 8U] >> (i % 8U)) & 1U);
    }
    return bits;
}

inline float normalize_byte(std::uint8_t value) noexcept {
    return static_cast<float>(value) / 255.0F;
}

inline std::uint8_t xor_residual(std::uint8_t source, std::uint8_t reconstructed) noexcept {
    return static_cast<std::uint8_t>(source ^ reconstructed);
}

inline std::uint32_t popcount32(std::uint32_t value) noexcept {
    std::uint32_t count = 0U;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

inline std::uint32_t fp32_bits(float value) noexcept {
    static_assert(sizeof(float) == sizeof(std::uint32_t), "World500 requires IEEE-like 32-bit float storage");
    std::uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline std::uint32_t fp32_xor_residual(float source, float reconstructed) noexcept {
    return fp32_bits(source) ^ fp32_bits(reconstructed);
}

inline float byte_xor_error_rate(std::uint8_t source, std::uint8_t reconstructed) noexcept {
    return static_cast<float>(popcount32(xor_residual(source, reconstructed))) / 8.0F;
}

inline float numeric_abs_error(float source, float reconstructed) noexcept {
    return std::fabs(source - reconstructed);
}

inline float permeate_channel(float self, float neighbourhood_mean, float self_weight = 0.82F) {
    if (!std::isfinite(self) || !std::isfinite(neighbourhood_mean) || !std::isfinite(self_weight)) {
        throw std::invalid_argument("permeation inputs must be finite");
    }
    if (self_weight < 0.0F || self_weight > 1.0F) {
        throw std::invalid_argument("permeation self weight must lie in [0,1]");
    }
    const float neighbour_weight = 1.0F - self_weight;
    return self_weight * self + neighbour_weight * neighbourhood_mean;
}

inline std::vector<float> softmax(const std::vector<float>& logits) {
    if (logits.empty()) return {};
    for (const float value : logits) {
        if (!std::isfinite(value)) throw std::invalid_argument("attention logits must be finite");
    }
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> weights(logits.size(), 0.0F);
    double denominator = 0.0;
    for (std::size_t i = 0U; i < logits.size(); ++i) {
        weights[i] = static_cast<float>(std::exp(static_cast<double>(logits[i] - maximum)));
        denominator += static_cast<double>(weights[i]);
    }
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        throw std::runtime_error("attention softmax denominator is invalid");
    }
    for (float& value : weights) value = static_cast<float>(static_cast<double>(value) / denominator);
    return weights;
}

inline float scaled_dot_logit(
    const std::array<float, kLatentDimensions>& query,
    const std::array<float, kLatentDimensions>& key) {
    double dot = 0.0;
    for (std::size_t i = 0U; i < query.size(); ++i) {
        if (!std::isfinite(query[i]) || !std::isfinite(key[i])) {
            throw std::invalid_argument("attention vectors must be finite");
        }
        dot += static_cast<double>(query[i]) * static_cast<double>(key[i]);
    }
    return static_cast<float>(dot / std::sqrt(static_cast<double>(kLatentDimensions)));
}

inline float memory_update(float omega, float residual_signal, float rho) {
    if (!std::isfinite(omega) || !std::isfinite(residual_signal) || !std::isfinite(rho)) {
        throw std::invalid_argument("memory update inputs must be finite");
    }
    if (rho < 0.0F || rho > 1.0F) throw std::invalid_argument("memory rho must lie in [0,1]");
    return rho * omega + (1.0F - rho) * residual_signal;
}

inline float latent_velocity(
    float reconstruction_gradient,
    float attention_pressure,
    float memory_pressure,
    float latent_state,
    float alpha,
    float beta,
    float gamma,
    float delta) {
    const std::array<float, 8U> values{
        reconstruction_gradient, attention_pressure, memory_pressure, latent_state,
        alpha, beta, gamma, delta,
    };
    for (const float value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("latent velocity inputs must be finite");
    }
    if (alpha < 0.0F || beta < 0.0F || gamma < 0.0F || delta < 0.0F) {
        throw std::invalid_argument("latent velocity coefficients must be non-negative");
    }
    return -alpha * reconstruction_gradient + beta * attention_pressure +
           gamma * memory_pressure - delta * latent_state;
}

inline bool should_commit(double current_error, double candidate_error) {
    if (!std::isfinite(current_error) || !std::isfinite(candidate_error) ||
        current_error < 0.0 || candidate_error < 0.0) {
        throw std::invalid_argument("commit errors must be finite and non-negative");
    }
    return candidate_error < current_error;
}

inline bool byte_fixed_point(std::uint8_t input, std::uint8_t output) noexcept {
    return xor_residual(input, output) == 0U;
}

} // namespace jarvisx::world::bitwise500
