#pragma once

#include "jarvisx/autoencoder3d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jarvisx {

struct Moagi3DConfig {
    std::size_t input_edge{8U};
    std::size_t channels{3U};
    std::size_t latent_channels{4U};
    std::size_t max_iterations{32U};
    float epsilon{1.0e-5F};
    float feedback_gain{0.5F};
    float softmax_temperature{8.0F};
    std::uint64_t seed{0x4D4F4147493344ULL};
    bool quantized_latent{false};

    void validate() const {
        if (input_edge < 4U || input_edge > 64U || input_edge % 2U != 0U) {
            throw std::invalid_argument("Moagi-3D input edge must be even and in [4, 64]");
        }
        if (channels < 1U || channels > 32U) {
            throw std::invalid_argument("Moagi-3D channel count must be in [1, 32]");
        }
        if (latent_channels < 1U || latent_channels > 32U) {
            throw std::invalid_argument("Moagi-3D latent channels must be in [1, 32]");
        }
        if (max_iterations < 1U || max_iterations > 100000U) {
            throw std::invalid_argument("Moagi-3D max iterations must be in [1, 100000]");
        }
        if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
            throw std::invalid_argument("Moagi-3D epsilon must be finite and positive");
        }
        if (!std::isfinite(feedback_gain) || feedback_gain <= 0.0F ||
            feedback_gain > 1.0F) {
            throw std::invalid_argument("Moagi-3D feedback gain must be in (0, 1]");
        }
        if (!std::isfinite(softmax_temperature) || softmax_temperature <= 0.0F ||
            softmax_temperature > 1000.0F) {
            throw std::invalid_argument("Moagi-3D softmax temperature must be in (0, 1000]");
        }
    }
};

struct Moagi3DIterationTelemetry {
    std::size_t iteration{};
    float delta_l2{};
    float anchor_l2{};
    std::vector<float> channel_mse;
    std::vector<float> fusion_weights;
};

struct Moagi3DResult {
    Tensor4D output;
    std::size_t iterations{};
    bool converged{};
    float final_delta_l2{};
    float final_anchor_l2{};
    std::vector<float> fusion_weights;
    std::vector<Moagi3DIterationTelemetry> history;
};

class Moagi3DEngine {
public:
    explicit Moagi3DEngine(Moagi3DConfig config)
        : config_(config) {
        config_.validate();
        channels_.reserve(config_.channels);
        constexpr std::uint64_t kSeedStride = 0x9E3779B97F4A7C15ULL;
        for (std::size_t channel = 0U; channel < config_.channels; ++channel) {
            const std::uint64_t channel_seed = config_.seed +
                kSeedStride * static_cast<std::uint64_t>(channel + 1U);
            channels_.emplace_back(Autoencoder3DConfig{
                config_.input_edge,
                config_.latent_channels,
                0.03F,
                1.0e-4F,
                1.0F,
                channel_seed
            });
        }
    }

    const Moagi3DConfig& config() const noexcept { return config_; }
    std::size_t channel_count() const noexcept { return channels_.size(); }

    Moagi3DResult run(const Tensor4D& input) const {
        validate_input(input);
        ensure_finite(input, "input");

        const Tensor4D anchor = input;
        Tensor4D current = input;
        std::vector<Moagi3DIterationTelemetry> history;
        history.reserve(config_.max_iterations);
        std::vector<float> final_weights(config_.channels,
                                         1.0F / static_cast<float>(config_.channels));
        float final_delta = std::numeric_limits<float>::infinity();
        float final_anchor = 0.0F;

        for (std::size_t iteration = 1U; iteration <= config_.max_iterations;
             ++iteration) {
            std::vector<Tensor4D> reconstructions;
            reconstructions.reserve(config_.channels);
            std::vector<float> channel_mse;
            channel_mse.reserve(config_.channels);

            for (const Autoencoder3D& channel : channels_) {
                const Tensor4D latent = channel.encode(current, config_.quantized_latent);
                const Tensor4D reconstruction = channel.decode(latent);
                ensure_finite(latent, "latent");
                ensure_finite(reconstruction, "channel reconstruction");
                channel_mse.push_back(mean_squared_error(current, reconstruction));
                reconstructions.push_back(reconstruction);
            }

            final_weights = attention_weights(channel_mse);
            const Tensor4D fused = fuse(reconstructions, final_weights, current.shape());
            Tensor4D next(current.shape());
            for (std::size_t index = 0U; index < current.size(); ++index) {
                next.values()[index] =
                    (1.0F - config_.feedback_gain) * current.values()[index] +
                    config_.feedback_gain * fused.values()[index];
            }
            ensure_finite(next, "recursive state");

            final_delta = l2_distance(current, next);
            final_anchor = l2_distance(anchor, next);
            history.push_back(Moagi3DIterationTelemetry{
                iteration,
                final_delta,
                final_anchor,
                channel_mse,
                final_weights
            });

            current = next;
            if (final_delta <= config_.epsilon) {
                return Moagi3DResult{
                    current,
                    iteration,
                    true,
                    final_delta,
                    final_anchor,
                    final_weights,
                    history
                };
            }
        }

        return Moagi3DResult{
            current,
            config_.max_iterations,
            false,
            final_delta,
            final_anchor,
            final_weights,
            history
        };
    }

    static float l2_distance(const Tensor4D& lhs, const Tensor4D& rhs) {
        if (lhs.shape().channels != rhs.shape().channels ||
            lhs.shape().depth != rhs.shape().depth ||
            lhs.shape().height != rhs.shape().height ||
            lhs.shape().width != rhs.shape().width) {
            throw std::invalid_argument("L2 distance requires identical tensor shapes");
        }
        double sum = 0.0;
        for (std::size_t index = 0U; index < lhs.size(); ++index) {
            const double difference = static_cast<double>(lhs.values()[index]) -
                                      static_cast<double>(rhs.values()[index]);
            sum += difference * difference;
        }
        return static_cast<float>(std::sqrt(sum));
    }

private:
    Moagi3DConfig config_;
    std::vector<Autoencoder3D> channels_;

    void validate_input(const Tensor4D& input) const {
        const TensorShape4D& shape = input.shape();
        if (shape.channels != 1U || shape.depth != config_.input_edge ||
            shape.height != config_.input_edge || shape.width != config_.input_edge) {
            throw std::invalid_argument(
                "Moagi-3D input must have shape [1, edge, edge, edge]");
        }
    }

    static void ensure_finite(const Tensor4D& tensor, const char* stage) {
        for (const float value : tensor.values()) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string("non-finite value at Moagi-3D ") + stage);
            }
        }
    }

    static float mean_squared_error(const Tensor4D& target,
                                    const Tensor4D& predicted) {
        if (target.size() != predicted.size()) {
            throw std::invalid_argument("MSE requires tensors with equal element counts");
        }
        double sum = 0.0;
        for (std::size_t index = 0U; index < target.size(); ++index) {
            const double error = static_cast<double>(predicted.values()[index]) -
                                 static_cast<double>(target.values()[index]);
            sum += error * error;
        }
        return static_cast<float>(sum / static_cast<double>(target.size()));
    }

    std::vector<float> attention_weights(const std::vector<float>& mse) const {
        if (mse.size() != config_.channels) {
            throw std::logic_error("Moagi-3D channel metric count mismatch");
        }

        std::vector<double> logits(mse.size(), 0.0);
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t index = 0U; index < mse.size(); ++index) {
            logits[index] = -static_cast<double>(config_.softmax_temperature) *
                            static_cast<double>(mse[index]);
            maximum = std::max(maximum, logits[index]);
        }

        double denominator = 0.0;
        for (double& logit : logits) {
            logit = std::exp(logit - maximum);
            denominator += logit;
        }
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            throw std::runtime_error("Moagi-3D fusion softmax became invalid");
        }

        std::vector<float> weights(logits.size(), 0.0F);
        for (std::size_t index = 0U; index < logits.size(); ++index) {
            weights[index] = static_cast<float>(logits[index] / denominator);
        }
        return weights;
    }

    static Tensor4D fuse(const std::vector<Tensor4D>& reconstructions,
                         const std::vector<float>& weights,
                         TensorShape4D shape) {
        if (reconstructions.empty() || reconstructions.size() != weights.size()) {
            throw std::invalid_argument("Moagi-3D fusion requires one weight per channel");
        }

        Tensor4D fused(shape);
        for (std::size_t channel = 0U; channel < reconstructions.size(); ++channel) {
            if (reconstructions[channel].size() != fused.size()) {
                throw std::invalid_argument("Moagi-3D channel reconstruction shape mismatch");
            }
            for (std::size_t index = 0U; index < fused.size(); ++index) {
                fused.values()[index] += weights[channel] *
                                         reconstructions[channel].values()[index];
            }
        }
        return fused;
    }
};

} // namespace jarvisx
