#pragma once

#include "jarvisx/core.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace jarvisx {

struct TensorShape4D {
    std::size_t channels{};
    std::size_t depth{};
    std::size_t height{};
    std::size_t width{};

    std::size_t elements() const {
        if (channels == 0U || depth == 0U || height == 0U || width == 0U) {
            throw std::invalid_argument("tensor dimensions must be non-zero");
        }
        const std::size_t max = std::numeric_limits<std::size_t>::max();
        if (channels > max / depth || channels * depth > max / height ||
            channels * depth * height > max / width) {
            throw std::overflow_error("tensor size overflow");
        }
        return channels * depth * height * width;
    }
};

class Tensor4D {
public:
    explicit Tensor4D(TensorShape4D shape, float fill = 0.0F)
        : shape_(shape), values_(shape.elements(), fill) {}

    const TensorShape4D& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return values_.size(); }

    float& operator()(std::size_t channel, std::size_t z,
                      std::size_t y, std::size_t x) {
        return values_.at(index(channel, z, y, x));
    }

    float operator()(std::size_t channel, std::size_t z,
                     std::size_t y, std::size_t x) const {
        return values_.at(index(channel, z, y, x));
    }

    const std::vector<float>& values() const noexcept { return values_; }
    std::vector<float>& values() noexcept { return values_; }

private:
    TensorShape4D shape_;
    std::vector<float> values_;

    std::size_t index(std::size_t channel, std::size_t z,
                      std::size_t y, std::size_t x) const {
        if (channel >= shape_.channels || z >= shape_.depth ||
            y >= shape_.height || x >= shape_.width) {
            throw std::out_of_range("tensor coordinate out of range");
        }
        return x + shape_.width *
            (y + shape_.height * (z + shape_.depth * channel));
    }
};

struct Autoencoder3DConfig {
    std::size_t input_edge{8U};
    std::size_t latent_channels{4U};
    float learning_rate{0.03F};
    float l2_penalty{1.0e-4F};
    float gradient_clip{1.0F};
    std::uint64_t seed{0x4A415256495358ULL};

    void validate() const {
        if (input_edge < 4U || input_edge > 64U || input_edge % 2U != 0U) {
            throw std::invalid_argument("input edge must be even and in [4, 64]");
        }
        if (latent_channels < 1U || latent_channels > 32U) {
            throw std::invalid_argument("latent channels must be in [1, 32]");
        }
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0F ||
            learning_rate > 1.0F) {
            throw std::invalid_argument("learning rate must be in (0, 1]");
        }
        if (!std::isfinite(l2_penalty) || l2_penalty < 0.0F ||
            l2_penalty > 1.0F) {
            throw std::invalid_argument("L2 penalty must be in [0, 1]");
        }
        if (!std::isfinite(gradient_clip) || gradient_clip <= 0.0F ||
            gradient_clip > 100.0F) {
            throw std::invalid_argument("gradient clip must be in (0, 100]");
        }
    }
};

struct Autoencoder3DMetrics {
    std::uint64_t step{};
    float mse{};
    float mae{};
    float max_abs_error{};
    float latent_energy{};
    float gradient_l2{};
};

class Autoencoder3D {
public:
    static constexpr std::size_t kKernelEdge = 3U;
    static constexpr std::size_t kKernelVolume = 27U;

    explicit Autoencoder3D(Autoencoder3DConfig config)
        : config_(config),
          encoder_weights_(config.latent_channels * kKernelVolume),
          encoder_bias_(config.latent_channels, 0.0F),
          decoder_weights_(config.latent_channels * kKernelVolume),
          decoder_bias_(0.0F) {
        config_.validate();
        initialize_weights();
    }

    const Autoencoder3DConfig& config() const noexcept { return config_; }
    std::uint64_t steps() const noexcept { return steps_; }

    Tensor4D encode(const Tensor4D& input, bool quantize = false) const {
        validate_input(input);
        const std::size_t latent_edge = config_.input_edge / 2U;
        Tensor4D latent({config_.latent_channels, latent_edge, latent_edge,
                         latent_edge});

        for (std::size_t channel = 0; channel < config_.latent_channels;
             ++channel) {
            for (std::size_t z = 0; z < latent_edge; ++z) {
                for (std::size_t y = 0; y < latent_edge; ++y) {
                    for (std::size_t x = 0; x < latent_edge; ++x) {
                        float sum = encoder_bias_[channel];
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t iz = wrap_index(
                                        static_cast<long long>(2U * z + kz) - 1LL,
                                        config_.input_edge);
                                    const std::size_t iy = wrap_index(
                                        static_cast<long long>(2U * y + ky) - 1LL,
                                        config_.input_edge);
                                    const std::size_t ix = wrap_index(
                                        static_cast<long long>(2U * x + kx) - 1LL,
                                        config_.input_edge);
                                    sum += encoder_weight(channel, kernel) *
                                           input(0U, iz, iy, ix);
                                }
                            }
                        }
                        float value = std::tanh(sum);
                        if (quantize) {
                            value = dequantize_q3(quantize_q3(value));
                        }
                        latent(channel, z, y, x) = value;
                    }
                }
            }
        }
        return latent;
    }

    Tensor4D decode(const Tensor4D& latent) const {
        validate_latent(latent);
        Tensor4D output({1U, config_.input_edge, config_.input_edge,
                         config_.input_edge});
        const std::size_t latent_edge = config_.input_edge / 2U;

        for (std::size_t z = 0; z < config_.input_edge; ++z) {
            for (std::size_t y = 0; y < config_.input_edge; ++y) {
                for (std::size_t x = 0; x < config_.input_edge; ++x) {
                    float sum = decoder_bias_;
                    const std::size_t lz0 = z / 2U;
                    const std::size_t ly0 = y / 2U;
                    const std::size_t lx0 = x / 2U;
                    for (std::size_t channel = 0;
                         channel < config_.latent_channels; ++channel) {
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t lz = wrap_index(
                                        static_cast<long long>(lz0 + kz) - 1LL,
                                        latent_edge);
                                    const std::size_t ly = wrap_index(
                                        static_cast<long long>(ly0 + ky) - 1LL,
                                        latent_edge);
                                    const std::size_t lx = wrap_index(
                                        static_cast<long long>(lx0 + kx) - 1LL,
                                        latent_edge);
                                    sum += decoder_weight(channel, kernel) *
                                           latent(channel, lz, ly, lx);
                                }
                            }
                        }
                    }
                    output(0U, z, y, x) = std::tanh(sum);
                }
            }
        }
        return output;
    }

    Tensor4D reconstruct(const Tensor4D& input, bool quantize = false) const {
        return decode(encode(input, quantize));
    }

    Autoencoder3DMetrics train_step(const Tensor4D& input) {
        validate_input(input);
        const std::size_t latent_edge = config_.input_edge / 2U;
        const Tensor4D latent = encode(input, false);
        const Tensor4D reconstruction = decode(latent);
        Tensor4D output_delta(reconstruction.shape());
        Tensor4D latent_gradient(latent.shape());
        std::vector<float> decoder_gradient(decoder_weights_.size(), 0.0F);
        std::vector<float> encoder_gradient(encoder_weights_.size(), 0.0F);
        std::vector<float> encoder_bias_gradient(encoder_bias_.size(), 0.0F);
        float decoder_bias_gradient = 0.0F;

        double mse_sum = 0.0;
        double mae_sum = 0.0;
        float max_abs = 0.0F;
        const float inverse_count = 1.0F /
            static_cast<float>(reconstruction.size());

        for (std::size_t z = 0; z < config_.input_edge; ++z) {
            for (std::size_t y = 0; y < config_.input_edge; ++y) {
                for (std::size_t x = 0; x < config_.input_edge; ++x) {
                    const float predicted = reconstruction(0U, z, y, x);
                    const float error = predicted - input(0U, z, y, x);
                    const float absolute = std::fabs(error);
                    mse_sum += static_cast<double>(error) * error;
                    mae_sum += absolute;
                    max_abs = std::max(max_abs, absolute);
                    const float delta = 2.0F * inverse_count * error *
                                        (1.0F - predicted * predicted);
                    output_delta(0U, z, y, x) = delta;
                    decoder_bias_gradient += delta;

                    const std::size_t lz0 = z / 2U;
                    const std::size_t ly0 = y / 2U;
                    const std::size_t lx0 = x / 2U;
                    for (std::size_t channel = 0;
                         channel < config_.latent_channels; ++channel) {
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t lz = wrap_index(
                                        static_cast<long long>(lz0 + kz) - 1LL,
                                        latent_edge);
                                    const std::size_t ly = wrap_index(
                                        static_cast<long long>(ly0 + ky) - 1LL,
                                        latent_edge);
                                    const std::size_t lx = wrap_index(
                                        static_cast<long long>(lx0 + kx) - 1LL,
                                        latent_edge);
                                    const std::size_t weight = weight_index(channel, kernel);
                                    decoder_gradient[weight] +=
                                        delta * latent(channel, lz, ly, lx);
                                    latent_gradient(channel, lz, ly, lx) +=
                                        delta * decoder_weights_[weight];
                                }
                            }
                        }
                    }
                }
            }
        }

        double latent_energy_sum = 0.0;
        for (std::size_t channel = 0; channel < config_.latent_channels;
             ++channel) {
            for (std::size_t z = 0; z < latent_edge; ++z) {
                for (std::size_t y = 0; y < latent_edge; ++y) {
                    for (std::size_t x = 0; x < latent_edge; ++x) {
                        const float activation = latent(channel, z, y, x);
                        latent_energy_sum += static_cast<double>(activation) * activation;
                        const float delta = latent_gradient(channel, z, y, x) *
                                            (1.0F - activation * activation);
                        encoder_bias_gradient[channel] += delta;
                        for (std::size_t kz = 0; kz < kKernelEdge; ++kz) {
                            for (std::size_t ky = 0; ky < kKernelEdge; ++ky) {
                                for (std::size_t kx = 0; kx < kKernelEdge; ++kx) {
                                    const std::size_t kernel = kernel_index(kz, ky, kx);
                                    const std::size_t iz = wrap_index(
                                        static_cast<long long>(2U * z + kz) - 1LL,
                                        config_.input_edge);
                                    const std::size_t iy = wrap_index(
                                        static_cast<long long>(2U * y + ky) - 1LL,
                                        config_.input_edge);
                                    const std::size_t ix = wrap_index(
                                        static_cast<long long>(2U * x + kx) - 1LL,
                                        config_.input_edge);
                                    encoder_gradient[weight_index(channel, kernel)] +=
                                        delta * input(0U, iz, iy, ix);
                                }
                            }
                        }
                    }
                }
            }
        }

        double gradient_square_sum = 0.0;
        apply_update(encoder_weights_, encoder_gradient, gradient_square_sum);
        apply_update(decoder_weights_, decoder_gradient, gradient_square_sum);
        for (std::size_t channel = 0; channel < encoder_bias_.size(); ++channel) {
            const float gradient = clip(encoder_bias_gradient[channel]);
            gradient_square_sum += static_cast<double>(gradient) * gradient;
            encoder_bias_[channel] -= config_.learning_rate * gradient;
        }
        decoder_bias_gradient = clip(decoder_bias_gradient);
        gradient_square_sum += static_cast<double>(decoder_bias_gradient) *
                               decoder_bias_gradient;
        decoder_bias_ -= config_.learning_rate * decoder_bias_gradient;
        ++steps_;

        const double count = static_cast<double>(reconstruction.size());
        return {
            steps_,
            static_cast<float>(mse_sum / count),
            static_cast<float>(mae_sum / count),
            max_abs,
            static_cast<float>(latent_energy_sum /
                               static_cast<double>(latent.size())),
            static_cast<float>(std::sqrt(gradient_square_sum))
        };
    }

    void save(const fs::path& path) const {
        if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
        std::ofstream output(path, std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write model: " + path.string());
        output << "JARVISX_AUTOENCODER3D_V1\n"
               << config_.input_edge << ' ' << config_.latent_channels << ' '
               << std::setprecision(std::numeric_limits<float>::max_digits10)
               << config_.learning_rate << ' ' << config_.l2_penalty << ' '
               << config_.gradient_clip << ' ' << config_.seed << ' ' << steps_ << '\n';
        write_vector(output, encoder_weights_);
        write_vector(output, encoder_bias_);
        write_vector(output, decoder_weights_);
        output << decoder_bias_ << '\n';
        if (!output) throw std::runtime_error("cannot flush model: " + path.string());
    }

    static Autoencoder3D load(const fs::path& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot read model: " + path.string());
        std::string magic;
        std::getline(input, magic);
        if (magic != "JARVISX_AUTOENCODER3D_V1") {
            throw std::runtime_error("unsupported autoencoder model format");
        }
        Autoencoder3DConfig config;
        std::uint64_t steps = 0U;
        input >> config.input_edge >> config.latent_channels >> config.learning_rate
              >> config.l2_penalty >> config.gradient_clip >> config.seed >> steps;
        config.validate();
        Autoencoder3D model(config);
        read_vector(input, model.encoder_weights_);
        read_vector(input, model.encoder_bias_);
        read_vector(input, model.decoder_weights_);
        input >> model.decoder_bias_;
        model.steps_ = steps;
        if (!input) throw std::runtime_error("truncated autoencoder model");
        return model;
    }

private:
    Autoencoder3DConfig config_;
    std::vector<float> encoder_weights_;
    std::vector<float> encoder_bias_;
    std::vector<float> decoder_weights_;
    float decoder_bias_{};
    std::uint64_t steps_{};

    static std::size_t kernel_index(std::size_t z, std::size_t y,
                                    std::size_t x) noexcept {
        return x + kKernelEdge * (y + kKernelEdge * z);
    }

    static std::size_t weight_index(std::size_t channel,
                                    std::size_t kernel) noexcept {
        return channel * kKernelVolume + kernel;
    }

    static std::size_t wrap_index(long long value, std::size_t size) noexcept {
        const long long signed_size = static_cast<long long>(size);
        long long wrapped = value % signed_size;
        if (wrapped < 0) wrapped += signed_size;
        return static_cast<std::size_t>(wrapped);
    }

    float encoder_weight(std::size_t channel, std::size_t kernel) const noexcept {
        return encoder_weights_[weight_index(channel, kernel)];
    }

    float decoder_weight(std::size_t channel, std::size_t kernel) const noexcept {
        return decoder_weights_[weight_index(channel, kernel)];
    }

    void initialize_weights() {
        const float encoder_scale = std::sqrt(
            6.0F / static_cast<float>(kKernelVolume +
                                      config_.latent_channels * kKernelVolume));
        const float decoder_scale = std::sqrt(
            6.0F / static_cast<float>(config_.latent_channels * kKernelVolume +
                                      kKernelVolume));
        for (std::size_t i = 0; i < encoder_weights_.size(); ++i) {
            encoder_weights_[i] = encoder_scale *
                signed_unit(config_.seed ^ static_cast<std::uint64_t>(i));
        }
        for (std::size_t i = 0; i < decoder_weights_.size(); ++i) {
            decoder_weights_[i] = decoder_scale * signed_unit(
                (config_.seed ^ 0xDEC0DEULL) + static_cast<std::uint64_t>(i));
        }
    }

    void validate_input(const Tensor4D& input) const {
        const TensorShape4D shape = input.shape();
        if (shape.channels != 1U || shape.depth != config_.input_edge ||
            shape.height != config_.input_edge || shape.width != config_.input_edge) {
            throw std::invalid_argument("input must be a single-channel cubic tensor matching input_edge");
        }
        for (const float value : input.values()) {
            if (!std::isfinite(value) || value < -1.0F || value > 1.0F) {
                throw std::invalid_argument("input values must be finite and in [-1, 1]");
            }
        }
    }

    void validate_latent(const Tensor4D& latent) const {
        const std::size_t edge = config_.input_edge / 2U;
        const TensorShape4D shape = latent.shape();
        if (shape.channels != config_.latent_channels || shape.depth != edge ||
            shape.height != edge || shape.width != edge) {
            throw std::invalid_argument("latent tensor shape mismatch");
        }
        for (const float value : latent.values()) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("latent values must be finite");
            }
        }
    }

    float clip(float gradient) const noexcept {
        return clampf(gradient, -config_.gradient_clip, config_.gradient_clip);
    }

    void apply_update(std::vector<float>& weights,
                      const std::vector<float>& gradients,
                      double& gradient_square_sum) {
        if (weights.size() != gradients.size()) {
            throw std::runtime_error("gradient dimension mismatch");
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            const float gradient = clip(gradients[i] + config_.l2_penalty * weights[i]);
            gradient_square_sum += static_cast<double>(gradient) * gradient;
            weights[i] -= config_.learning_rate * gradient;
        }
    }

    static void write_vector(std::ostream& output,
                             const std::vector<float>& values) {
        output << values.size();
        for (const float value : values) output << ' ' << value;
        output << '\n';
    }

    static void read_vector(std::istream& input, std::vector<float>& values) {
        std::size_t size = 0U;
        input >> size;
        if (size != values.size()) {
            throw std::runtime_error("model tensor dimension mismatch");
        }
        for (float& value : values) input >> value;
    }
};

inline Tensor4D make_volume(std::size_t edge, const std::string& pattern,
                            std::uint64_t seed) {
    if (edge < 2U) throw std::invalid_argument("volume edge below 2");
    Tensor4D volume({1U, edge, edge, edge});
    const float center = (static_cast<float>(edge) - 1.0F) * 0.5F;
    const float radius = std::max(1.0F, static_cast<float>(edge) * 0.28F);

    for (std::size_t z = 0; z < edge; ++z) {
        for (std::size_t y = 0; y < edge; ++y) {
            for (std::size_t x = 0; x < edge; ++x) {
                const float fx = (static_cast<float>(x) - center) / radius;
                const float fy = (static_cast<float>(y) - center) / radius;
                const float fz = (static_cast<float>(z) - center) / radius;
                float value = 0.0F;
                if (pattern == "sphere") {
                    value = std::sqrt(fx * fx + fy * fy + fz * fz) <= 1.0F
                        ? 1.0F : -1.0F;
                } else if (pattern == "shell") {
                    const float distance = std::sqrt(fx * fx + fy * fy + fz * fz);
                    value = std::fabs(distance - 1.0F) < 0.22F ? 1.0F : -1.0F;
                } else if (pattern == "checker") {
                    value = ((x / 2U + y / 2U + z / 2U) % 2U == 0U)
                        ? 1.0F : -1.0F;
                } else if (pattern == "wave") {
                    const float phase = 0.7F * static_cast<float>(x) +
                                        0.5F * static_cast<float>(y) +
                                        0.9F * static_cast<float>(z);
                    value = std::sin(phase);
                } else if (pattern == "noise") {
                    const std::uint64_t key = seed ^
                        (static_cast<std::uint64_t>(x) << 42U) ^
                        (static_cast<std::uint64_t>(y) << 21U) ^
                        static_cast<std::uint64_t>(z);
                    value = signed_unit(key);
                } else {
                    throw std::invalid_argument("unknown volume pattern: " + pattern);
                }
                volume(0U, z, y, x) = clampf(value, -1.0F, 1.0F);
            }
        }
    }
    return volume;
}

inline Autoencoder3DMetrics measure_reconstruction(const Tensor4D& target,
                                                    const Tensor4D& latent,
                                                    const Tensor4D& output,
                                                    std::uint64_t step) {
    if (target.shape().elements() != output.shape().elements()) {
        throw std::invalid_argument("measurement tensor shape mismatch");
    }
    double mse = 0.0;
    double mae = 0.0;
    double energy = 0.0;
    float max_abs = 0.0F;
    for (std::size_t i = 0; i < target.size(); ++i) {
        const float error = output.values()[i] - target.values()[i];
        const float absolute = std::fabs(error);
        mse += static_cast<double>(error) * error;
        mae += absolute;
        max_abs = std::max(max_abs, absolute);
    }
    for (const float value : latent.values()) {
        energy += static_cast<double>(value) * value;
    }
    return {
        step,
        static_cast<float>(mse / static_cast<double>(target.size())),
        static_cast<float>(mae / static_cast<double>(target.size())),
        max_abs,
        static_cast<float>(energy / static_cast<double>(latent.size())),
        0.0F
    };
}

inline void export_obj(const Tensor4D& tensor, const fs::path& path,
                       float threshold = 0.0F, float channel_gap = 2.0F) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write OBJ: " + path.string());
    output << "# Jarvis-X 3D autoencoder voxel point cloud\n";
    const TensorShape4D shape = tensor.shape();
    for (std::size_t channel = 0; channel < shape.channels; ++channel) {
        const float z_offset = static_cast<float>(channel) *
            (static_cast<float>(shape.depth) + channel_gap);
        for (std::size_t z = 0; z < shape.depth; ++z) {
            for (std::size_t y = 0; y < shape.height; ++y) {
                for (std::size_t x = 0; x < shape.width; ++x) {
                    if (tensor(channel, z, y, x) > threshold) {
                        output << "v " << x << ' ' << y << ' '
                               << (static_cast<float>(z) + z_offset) << '\n';
                    }
                }
            }
        }
    }
    if (!output) throw std::runtime_error("cannot flush OBJ: " + path.string());
}

} // namespace jarvisx
