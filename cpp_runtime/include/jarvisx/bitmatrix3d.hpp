#pragma once

#include "jarvisx/autoencoder3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jarvisx {

struct BitMatrix3DConfig {
    std::size_t input_edge{8U};
    std::size_t latent_channels{4U};
    float learning_rate{0.01F};
    float l2_penalty{1.0e-4F};
    float gradient_clip{1.0F};
    float ternary_threshold{0.35F};
    float ste_clip{1.0F};
    float omega_decay{0.9F};
    float omega_gain{0.15F};
    float fixed_point_tolerance{1.0e-3F};
    std::uint64_t seed{0x444D564249544D58ULL};

    void validate() const {
        if (input_edge < 4U || input_edge > 64U || input_edge % 2U != 0U) {
            throw std::invalid_argument("input edge must be even and in [4, 64]");
        }
        if (latent_channels < 1U || latent_channels > 32U) {
            throw std::invalid_argument("latent channels must be in [1, 32]");
        }
        auto finite_between = [](float value, float lo, float hi) {
            return std::isfinite(value) && value >= lo && value <= hi;
        };
        if (!finite_between(learning_rate, 1.0e-7F, 1.0F))
            throw std::invalid_argument("learning rate must be finite and in [1e-7, 1]");
        if (!finite_between(l2_penalty, 0.0F, 1.0F))
            throw std::invalid_argument("L2 penalty must be finite and in [0, 1]");
        if (!finite_between(gradient_clip, 1.0e-6F, 100.0F))
            throw std::invalid_argument("gradient clip must be finite and in [1e-6, 100]");
        if (!finite_between(ternary_threshold, 0.0F, 1.0F))
            throw std::invalid_argument("ternary threshold must be finite and in [0, 1]");
        if (!finite_between(ste_clip, 1.0e-6F, 8.0F))
            throw std::invalid_argument("STE clip must be finite and in [1e-6, 8]");
        if (!finite_between(omega_decay, 0.0F, 1.0F))
            throw std::invalid_argument("omega decay must be finite and in [0, 1]");
        if (!finite_between(omega_gain, 0.0F, 4.0F))
            throw std::invalid_argument("omega gain must be finite and in [0, 4]");
        if (!finite_between(fixed_point_tolerance, 0.0F, 1.0F))
            throw std::invalid_argument("fixed point tolerance must be finite and in [0, 1]");
    }
};

inline std::uint32_t popcount64_portable(std::uint64_t value) noexcept {
    std::uint32_t count = 0U;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

class PackedTernary {
public:
    PackedTernary() = default;

    static PackedTernary pack(const std::vector<std::int8_t>& values) {
        PackedTernary packed;
        packed.count_ = values.size();
        const std::size_t words = (values.size() + 63U) / 64U;
        packed.sign_.assign(words, 0U);
        packed.nonzero_.assign(words, 0U);
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::int8_t value = values[i];
            if (value < -1 || value > 1)
                throw std::invalid_argument("ternary value must be -1, 0 or +1");
            if (value == 0) continue;
            const std::size_t word = i / 64U;
            const std::uint64_t bit = std::uint64_t{1} << (i % 64U);
            packed.nonzero_[word] |= bit;
            if (value > 0) packed.sign_[word] |= bit;
        }
        packed.validate_padding();
        return packed;
    }

    std::vector<std::int8_t> unpack() const {
        validate_padding();
        std::vector<std::int8_t> values(count_, 0);
        for (std::size_t i = 0; i < count_; ++i) {
            const std::size_t word = i / 64U;
            const std::uint64_t bit = std::uint64_t{1} << (i % 64U);
            if ((nonzero_[word] & bit) == 0U) continue;
            values[i] = (sign_[word] & bit) != 0U ? std::int8_t{1} : std::int8_t{-1};
        }
        return values;
    }

    std::size_t size() const noexcept { return count_; }
    std::size_t physical_bytes() const noexcept {
        return (sign_.size() + nonzero_.size()) * sizeof(std::uint64_t);
    }
    std::size_t nonzero_count() const noexcept {
        std::size_t count = 0U;
        for (const auto word : nonzero_) count += popcount64_portable(word);
        return count;
    }
    double density() const noexcept {
        return count_ == 0U ? 0.0 : static_cast<double>(nonzero_count()) / static_cast<double>(count_);
    }
    const std::vector<std::uint64_t>& sign_words() const noexcept { return sign_; }
    const std::vector<std::uint64_t>& nonzero_words() const noexcept { return nonzero_; }

    static int binary_dot(std::uint64_t lhs, std::uint64_t rhs, std::size_t valid_bits = 64U) {
        if (valid_bits > 64U) throw std::invalid_argument("valid_bits must be <= 64");
        if (valid_bits == 0U) return 0;
        const std::uint64_t mask = valid_bits == 64U
            ? std::numeric_limits<std::uint64_t>::max()
            : ((std::uint64_t{1} << valid_bits) - 1U);
        const auto mismatches = popcount64_portable((lhs ^ rhs) & mask);
        return static_cast<int>(valid_bits) - 2 * static_cast<int>(mismatches);
    }

    static int ternary_binary_dot(std::uint64_t binary_sign,
                                  std::uint64_t ternary_sign,
                                  std::uint64_t ternary_nonzero,
                                  std::size_t valid_bits = 64U) {
        if (valid_bits > 64U) throw std::invalid_argument("valid_bits must be <= 64");
        if (valid_bits == 0U) return 0;
        const std::uint64_t valid_mask = valid_bits == 64U
            ? std::numeric_limits<std::uint64_t>::max()
            : ((std::uint64_t{1} << valid_bits) - 1U);
        const std::uint64_t mask = ternary_nonzero & valid_mask;
        const auto active = popcount64_portable(mask);
        const auto mismatches = popcount64_portable((binary_sign ^ ternary_sign) & mask);
        return static_cast<int>(active) - 2 * static_cast<int>(mismatches);
    }

private:
    std::size_t count_{};
    std::vector<std::uint64_t> sign_;
    std::vector<std::uint64_t> nonzero_;

    void validate_padding() const {
        if (sign_.size() != nonzero_.size())
            throw std::logic_error("ternary planes have different word counts");
        if (count_ == 0U) {
            if (!sign_.empty() || !nonzero_.empty())
                throw std::logic_error("empty ternary field has storage");
            return;
        }
        const std::size_t expected = (count_ + 63U) / 64U;
        if (sign_.size() != expected)
            throw std::logic_error("ternary plane word count is not canonical");
        if ((count_ & 63U) != 0U) {
            const std::size_t used = count_ & 63U;
            const std::uint64_t valid = (std::uint64_t{1} << used) - 1U;
            if ((sign_.back() & ~valid) != 0U || (nonzero_.back() & ~valid) != 0U)
                throw std::logic_error("ternary plane has non-zero padding bits");
        }
        for (std::size_t i = 0; i < sign_.size(); ++i) {
            if ((sign_[i] & ~nonzero_[i]) != 0U)
                throw std::logic_error("ternary sign bits must be zero where mask is zero");
        }
    }
};

struct QuantizedTernaryWeights {
    float scale{1.0F};
    std::vector<std::int8_t> values;
    PackedTernary packed;
};

inline QuantizedTernaryWeights ternary_quantize_weights(const std::vector<float>& shadow) {
    if (shadow.empty()) throw std::invalid_argument("shadow weights cannot be empty");
    double abs_sum = 0.0;
    for (const float value : shadow) {
        if (!std::isfinite(value)) throw std::invalid_argument("shadow weights must be finite");
        abs_sum += std::fabs(static_cast<double>(value));
    }
    float scale = static_cast<float>(abs_sum / static_cast<double>(shadow.size()));
    if (scale < 1.0e-12F) scale = 1.0F;
    std::vector<std::int8_t> values(shadow.size(), 0);
    for (std::size_t i = 0; i < shadow.size(); ++i) {
        const float normalized = shadow[i] / scale;
        const float rounded = std::round(std::max(-1.0F, std::min(1.0F, normalized)));
        values[i] = static_cast<std::int8_t>(rounded);
    }
    return {scale, values, PackedTernary::pack(values)};
}

struct QuantizedInt8 {
    float scale{1.0F};
    std::vector<std::int8_t> values;
};

inline QuantizedInt8 quantize_symmetric_int8(const std::vector<float>& input) {
    if (input.empty()) throw std::invalid_argument("activation tensor cannot be empty");
    float max_abs = 0.0F;
    for (const float value : input) {
        if (!std::isfinite(value)) throw std::invalid_argument("activation tensor must be finite");
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const float scale = max_abs < 1.0e-12F ? 1.0F : max_abs / 127.0F;
    std::vector<std::int8_t> values(input.size(), 0);
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float scaled = input[i] / scale;
        const float clipped = std::max(-127.0F, std::min(127.0F, scaled));
        values[i] = static_cast<std::int8_t>(std::lround(clipped));
    }
    return {scale, values};
}

struct BitMatrix3DMetrics {
    std::uint64_t step{};
    float mse{};
    float mae{};
    float max_abs_error{};
    float fixed_point_residual{};
    float encoder_weight_density{};
    float decoder_weight_density{};
    float latent_density{};
    float gradient_l2{};
    std::size_t shadow_weight_bytes{};
    std::size_t packed_weight_bytes{};
    std::size_t packed_latent_bytes{};
    bool self_description_valid{};
    bool fixed_point_converged{};
};

struct BitMatrix3DForward {
    Tensor4D latent_shadow;
    Tensor4D latent_ternary;
    PackedTernary latent_packed;
    Tensor4D reconstruction;
    QuantizedTernaryWeights encoder_weights;
    QuantizedTernaryWeights decoder_weights;
};

class BitMatrix3DEngine {
public:
    static constexpr std::size_t kKernelEdge = 3U;
    static constexpr std::size_t kKernelVolume = 27U;

    explicit BitMatrix3DEngine(BitMatrix3DConfig config)
        : config_(config),
          encoder_shadow_(config.latent_channels * kKernelVolume),
          encoder_bias_(config.latent_channels, 0.0F),
          decoder_shadow_(config.latent_channels * kKernelVolume),
          omega_({1U, config.input_edge, config.input_edge, config.input_edge}) {
        config_.validate();
        initialize_weights();
    }

    const BitMatrix3DConfig& config() const noexcept { return config_; }
    std::uint64_t steps() const noexcept { return steps_; }
    const std::vector<float>& encoder_shadow_weights() const noexcept { return encoder_shadow_; }
    const std::vector<float>& decoder_shadow_weights() const noexcept { return decoder_shadow_; }

    BitMatrix3DForward forward(const Tensor4D& input) const {
        validate_input(input);
        const auto encoder_q = ternary_quantize_weights(encoder_shadow_);
        const auto decoder_q = ternary_quantize_weights(decoder_shadow_);
        const auto input_q = quantize_symmetric_int8(input.values());
        const std::size_t latent_edge = config_.input_edge / 2U;
        Tensor4D latent_shadow({config_.latent_channels, latent_edge, latent_edge, latent_edge});
        Tensor4D latent_ternary({config_.latent_channels, latent_edge, latent_edge, latent_edge});
        std::vector<std::int8_t> latent_codes(latent_ternary.size(), 0);

        for (std::size_t channel = 0; channel < config_.latent_channels; ++channel) {
            for (std::size_t z = 0; z < latent_edge; ++z) {
                for (std::size_t y = 0; y < latent_edge; ++y) {
                    for (std::size_t x = 0; x < latent_edge; ++x) {
                        std::int32_t integer_sum = 0;
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t iz = wrap_index(static_cast<long long>(2U * z + kz) - 1LL, config_.input_edge);
                                    const std::size_t iy = wrap_index(static_cast<long long>(2U * y + ky) - 1LL, config_.input_edge);
                                    const std::size_t ix = wrap_index(static_cast<long long>(2U * x + kx) - 1LL, config_.input_edge);
                                    const auto qx = input_q.values[input_index(iz, iy, ix)];
                                    const auto qw = encoder_q.values[weight_index(channel, kernel)];
                                    integer_sum += static_cast<std::int32_t>(qx) * static_cast<std::int32_t>(qw);
                                }
                            }
                        }
                        const float pre = encoder_bias_[channel] +
                            static_cast<float>(integer_sum) * input_q.scale * encoder_q.scale;
                        const float soft = std::tanh(pre);
                        const std::int8_t code = ternary_activation(soft);
                        latent_shadow(channel, z, y, x) = soft;
                        latent_ternary(channel, z, y, x) = static_cast<float>(code);
                        latent_codes[latent_index(channel, z, y, x)] = code;
                    }
                }
            }
        }

        Tensor4D reconstruction({1U, config_.input_edge, config_.input_edge, config_.input_edge});
        for (std::size_t z = 0; z < config_.input_edge; ++z) {
            for (std::size_t y = 0; y < config_.input_edge; ++y) {
                for (std::size_t x = 0; x < config_.input_edge; ++x) {
                    std::int32_t integer_sum = 0;
                    const std::size_t lz0 = z / 2U;
                    const std::size_t ly0 = y / 2U;
                    const std::size_t lx0 = x / 2U;
                    for (std::size_t channel = 0; channel < config_.latent_channels; ++channel) {
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t lz = wrap_index(static_cast<long long>(lz0 + kz) - 1LL, latent_edge);
                                    const std::size_t ly = wrap_index(static_cast<long long>(ly0 + ky) - 1LL, latent_edge);
                                    const std::size_t lx = wrap_index(static_cast<long long>(lx0 + kx) - 1LL, latent_edge);
                                    const auto qx = latent_codes[latent_index(channel, lz, ly, lx)];
                                    const auto qw = decoder_q.values[weight_index(channel, kernel)];
                                    integer_sum += static_cast<std::int32_t>(qx) * static_cast<std::int32_t>(qw);
                                }
                            }
                        }
                    }
                    const float pre = decoder_bias_ + static_cast<float>(integer_sum) * decoder_q.scale;
                    reconstruction(0U, z, y, x) = std::tanh(pre);
                }
            }
        }

        return {std::move(latent_shadow), std::move(latent_ternary),
                PackedTernary::pack(latent_codes), std::move(reconstruction),
                encoder_q, decoder_q};
    }

    Tensor4D reconstruct(const Tensor4D& input) const { return forward(input).reconstruction; }

    BitMatrix3DMetrics train_step(const Tensor4D& input) {
        validate_input(input);
        const auto fwd = forward(input);
        const std::size_t latent_edge = config_.input_edge / 2U;
        Tensor4D output_delta(fwd.reconstruction.shape());
        Tensor4D latent_gradient(fwd.latent_shadow.shape());
        std::vector<float> decoder_gradient(decoder_shadow_.size(), 0.0F);
        std::vector<float> encoder_gradient(encoder_shadow_.size(), 0.0F);
        std::vector<float> encoder_bias_gradient(encoder_bias_.size(), 0.0F);
        float decoder_bias_gradient = 0.0F;
        double mse_sum = 0.0;
        double mae_sum = 0.0;
        float max_abs = 0.0F;
        const float inverse_count = 1.0F / static_cast<float>(fwd.reconstruction.size());

        for (std::size_t z = 0; z < config_.input_edge; ++z) {
            for (std::size_t y = 0; y < config_.input_edge; ++y) {
                for (std::size_t x = 0; x < config_.input_edge; ++x) {
                    const float predicted = fwd.reconstruction(0U, z, y, x);
                    const float error = predicted - input(0U, z, y, x);
                    const float absolute = std::fabs(error);
                    mse_sum += static_cast<double>(error) * error;
                    mae_sum += absolute;
                    max_abs = std::max(max_abs, absolute);
                    const float delta = 2.0F * inverse_count * error * (1.0F - predicted * predicted);
                    output_delta(0U, z, y, x) = delta;
                    decoder_bias_gradient += delta;
                    const std::size_t lz0 = z / 2U;
                    const std::size_t ly0 = y / 2U;
                    const std::size_t lx0 = x / 2U;
                    for (std::size_t channel = 0; channel < config_.latent_channels; ++channel) {
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t lz = wrap_index(static_cast<long long>(lz0 + kz) - 1LL, latent_edge);
                                    const std::size_t ly = wrap_index(static_cast<long long>(ly0 + ky) - 1LL, latent_edge);
                                    const std::size_t lx = wrap_index(static_cast<long long>(lx0 + kx) - 1LL, latent_edge);
                                    const std::size_t wi = weight_index(channel, kernel);
                                    const float latent_value = fwd.latent_ternary(channel, lz, ly, lx);
                                    decoder_gradient[wi] += delta * latent_value;
                                    const float physical_weight = fwd.decoder_weights.scale *
                                        static_cast<float>(fwd.decoder_weights.values[wi]);
                                    latent_gradient(channel, lz, ly, lx) += delta * physical_weight;
                                }
                            }
                        }
                    }
                }
            }
        }

        for (std::size_t channel = 0; channel < config_.latent_channels; ++channel) {
            for (std::size_t z = 0; z < latent_edge; ++z) {
                for (std::size_t y = 0; y < latent_edge; ++y) {
                    for (std::size_t x = 0; x < latent_edge; ++x) {
                        const float soft = fwd.latent_shadow(channel, z, y, x);
                        const float ste = std::fabs(soft) <= config_.ste_clip ? 1.0F : 0.0F;
                        const float delta = latent_gradient(channel, z, y, x) * ste * (1.0F - soft * soft);
                        encoder_bias_gradient[channel] += delta;
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t iz = wrap_index(static_cast<long long>(2U * z + kz) - 1LL, config_.input_edge);
                                    const std::size_t iy = wrap_index(static_cast<long long>(2U * y + ky) - 1LL, config_.input_edge);
                                    const std::size_t ix = wrap_index(static_cast<long long>(2U * x + kx) - 1LL, config_.input_edge);
                                    encoder_gradient[weight_index(channel, kernel)] += delta * input(0U, iz, iy, ix);
                                }
                            }
                        }
                    }
                }
            }
        }

        double gradient_square_sum = 0.0;
        apply_update(encoder_shadow_, encoder_gradient, gradient_square_sum);
        apply_update(decoder_shadow_, decoder_gradient, gradient_square_sum);
        for (std::size_t i = 0; i < encoder_bias_.size(); ++i) {
            const float gradient = clip(encoder_bias_gradient[i]);
            gradient_square_sum += static_cast<double>(gradient) * gradient;
            encoder_bias_[i] -= config_.learning_rate * gradient;
        }
        decoder_bias_gradient = clip(decoder_bias_gradient);
        gradient_square_sum += static_cast<double>(decoder_bias_gradient) * decoder_bias_gradient;
        decoder_bias_ -= config_.learning_rate * decoder_bias_gradient;
        ++steps_;

        const auto inward = inward_metrics(input, fwd);
        const double count = static_cast<double>(fwd.reconstruction.size());
        return {
            steps_,
            static_cast<float>(mse_sum / count),
            static_cast<float>(mae_sum / count),
            max_abs,
            inward.first,
            static_cast<float>(fwd.encoder_weights.packed.density()),
            static_cast<float>(fwd.decoder_weights.packed.density()),
            static_cast<float>(fwd.latent_packed.density()),
            static_cast<float>(std::sqrt(gradient_square_sum)),
            (encoder_shadow_.size() + decoder_shadow_.size()) * sizeof(float),
            fwd.encoder_weights.packed.physical_bytes() + fwd.decoder_weights.packed.physical_bytes(),
            fwd.latent_packed.physical_bytes(),
            verify_self_description(fwd),
            inward.first <= config_.fixed_point_tolerance
        };
    }

    BitMatrix3DMetrics evaluate_inward(const Tensor4D& input) {
        const auto fwd = forward(input);
        const auto inward = inward_metrics(input, fwd);
        double mse_sum = 0.0;
        double mae_sum = 0.0;
        float max_abs = 0.0F;
        for (std::size_t i = 0; i < input.size(); ++i) {
            const float error = fwd.reconstruction.values()[i] - input.values()[i];
            mse_sum += static_cast<double>(error) * error;
            mae_sum += std::fabs(error);
            max_abs = std::max(max_abs, std::fabs(error));
        }
        const double count = static_cast<double>(input.size());
        return {
            steps_, static_cast<float>(mse_sum / count), static_cast<float>(mae_sum / count), max_abs,
            inward.first,
            static_cast<float>(fwd.encoder_weights.packed.density()),
            static_cast<float>(fwd.decoder_weights.packed.density()),
            static_cast<float>(fwd.latent_packed.density()),
            0.0F,
            (encoder_shadow_.size() + decoder_shadow_.size()) * sizeof(float),
            fwd.encoder_weights.packed.physical_bytes() + fwd.decoder_weights.packed.physical_bytes(),
            fwd.latent_packed.physical_bytes(),
            verify_self_description(fwd),
            inward.first <= config_.fixed_point_tolerance
        };
    }

    bool verify_self_description(const BitMatrix3DForward& fwd) const {
        return fwd.encoder_weights.values == fwd.encoder_weights.packed.unpack() &&
               fwd.decoder_weights.values == fwd.decoder_weights.packed.unpack() &&
               latent_values(fwd.latent_ternary) == fwd.latent_packed.unpack();
    }

private:
    BitMatrix3DConfig config_;
    std::vector<float> encoder_shadow_;
    std::vector<float> encoder_bias_;
    std::vector<float> decoder_shadow_;
    float decoder_bias_{0.0F};
    Tensor4D omega_;
    std::uint64_t steps_{};

    static std::size_t kernel_index(std::size_t z, std::size_t y, std::size_t x) noexcept {
        return x + kKernelEdge * (y + kKernelEdge * z);
    }
    std::size_t weight_index(std::size_t channel, std::size_t kernel) const noexcept {
        return channel * kKernelVolume + kernel;
    }
    std::size_t input_index(std::size_t z, std::size_t y, std::size_t x) const noexcept {
        return x + config_.input_edge * (y + config_.input_edge * z);
    }
    std::size_t latent_index(std::size_t channel, std::size_t z, std::size_t y, std::size_t x) const noexcept {
        const std::size_t edge = config_.input_edge / 2U;
        return x + edge * (y + edge * (z + edge * channel));
    }
    static std::size_t wrap_index(long long value, std::size_t extent) noexcept {
        const long long size = static_cast<long long>(extent);
        long long wrapped = value % size;
        if (wrapped < 0) wrapped += size;
        return static_cast<std::size_t>(wrapped);
    }
    std::int8_t ternary_activation(float value) const noexcept {
        if (value > config_.ternary_threshold) return 1;
        if (value < -config_.ternary_threshold) return -1;
        return 0;
    }
    static std::vector<std::int8_t> latent_values(const Tensor4D& latent) {
        std::vector<std::int8_t> values(latent.size(), 0);
        for (std::size_t i = 0; i < latent.size(); ++i) {
            const float value = latent.values()[i];
            if (value > 0.5F) values[i] = 1;
            else if (value < -0.5F) values[i] = -1;
        }
        return values;
    }
    void validate_input(const Tensor4D& input) const {
        const auto& s = input.shape();
        if (s.channels != 1U || s.depth != config_.input_edge ||
            s.height != config_.input_edge || s.width != config_.input_edge)
            throw std::invalid_argument("bit-matrix input shape mismatch");
        for (const float value : input.values()) {
            if (!std::isfinite(value)) throw std::invalid_argument("bit-matrix input must be finite");
        }
    }
    void initialize_weights() {
        std::mt19937_64 rng(config_.seed);
        std::uniform_real_distribution<float> distribution(-0.35F, 0.35F);
        for (auto& value : encoder_shadow_) value = distribution(rng);
        for (auto& value : decoder_shadow_) value = distribution(rng);
    }
    float clip(float gradient) const noexcept {
        return std::max(-config_.gradient_clip, std::min(config_.gradient_clip, gradient));
    }
    void apply_update(std::vector<float>& shadow, const std::vector<float>& gradient,
                      double& gradient_square_sum) {
        for (std::size_t i = 0; i < shadow.size(); ++i) {
            float g = gradient[i] + config_.l2_penalty * shadow[i];
            g = clip(g);
            gradient_square_sum += static_cast<double>(g) * g;
            shadow[i] -= config_.learning_rate * g;
        }
    }
    std::pair<float, Tensor4D> inward_metrics(const Tensor4D& input, const BitMatrix3DForward& fwd) {
        Tensor4D candidate(input.shape());
        double numerator = 0.0;
        double denominator = 0.0;
        for (std::size_t i = 0; i < input.size(); ++i) {
            const float residual = input.values()[i] - fwd.reconstruction.values()[i];
            omega_.values()[i] = config_.omega_decay * omega_.values()[i] +
                (1.0F - config_.omega_decay) * residual;
            const float projected = std::max(-1.0F, std::min(1.0F,
                fwd.reconstruction.values()[i] + config_.omega_gain * omega_.values()[i]));
            candidate.values()[i] = projected;
            const double difference = static_cast<double>(projected) - input.values()[i];
            numerator += difference * difference;
            denominator += static_cast<double>(input.values()[i]) * input.values()[i];
        }
        const float residual = static_cast<float>(std::sqrt(numerator) /
            (std::sqrt(denominator) + 1.0e-12));
        return {residual, std::move(candidate)};
    }
};

} // namespace jarvisx
