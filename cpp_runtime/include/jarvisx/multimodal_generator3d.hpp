#pragma once

#include "jarvisx/autoencoder3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jarvisx::mm3d {

enum class Modality : std::uint8_t {
    Text = 0U,
    Image = 1U,
    Audio = 2U,
    Video = 3U,
    Volume3D = 4U,
    Generic = 5U,
};

inline const char* modality_name(Modality modality) noexcept {
    switch (modality) {
    case Modality::Text: return "text";
    case Modality::Image: return "image";
    case Modality::Audio: return "audio";
    case Modality::Video: return "video";
    case Modality::Volume3D: return "volume3d";
    case Modality::Generic: return "generic";
    }
    return "unknown";
}

inline Modality parse_modality(const std::string& value) {
    if (value == "text") return Modality::Text;
    if (value == "image" || value == "visual") return Modality::Image;
    if (value == "audio") return Modality::Audio;
    if (value == "video" || value == "animation") return Modality::Video;
    if (value == "volume" || value == "volume3d" || value == "3d") {
        return Modality::Volume3D;
    }
    if (value == "generic" || value == "binary") return Modality::Generic;
    throw std::invalid_argument("unsupported modality: " + value);
}

struct MediaPacket {
    Modality modality{Modality::Generic};
    std::vector<float> samples;
    std::string text;
    std::size_t width{0U};
    std::size_t height{0U};
    std::size_t depth{0U};
    std::size_t channels{1U};
    std::size_t frames{1U};
    std::uint32_t sample_rate{44100U};
};

struct GeneratedMedia {
    Modality modality{Modality::Generic};
    std::vector<float> samples;
    std::string text;
    std::size_t width{0U};
    std::size_t height{0U};
    std::size_t depth{0U};
    std::size_t channels{1U};
    std::size_t frames{1U};
    std::uint32_t sample_rate{44100U};
};

struct GeneratorConfig {
    Autoencoder3DConfig autoencoder{};
    std::size_t default_text_length{192U};
    std::size_t default_image_edge{64U};
    std::size_t default_audio_samples{4096U};
    std::size_t default_video_frames{8U};
    std::size_t default_video_edge{32U};
    float prompt_strength{0.18F};

    void validate() const {
        autoencoder.validate();
        if (default_text_length == 0U || default_text_length > 65536U) {
            throw std::invalid_argument("default text length must be in [1, 65536]");
        }
        if (default_image_edge == 0U || default_image_edge > 4096U) {
            throw std::invalid_argument("default image edge must be in [1, 4096]");
        }
        if (default_audio_samples == 0U || default_audio_samples > 16777216U) {
            throw std::invalid_argument("default audio samples out of range");
        }
        if (default_video_frames == 0U || default_video_frames > 1024U ||
            default_video_edge == 0U || default_video_edge > 1024U) {
            throw std::invalid_argument("default video geometry out of range");
        }
        if (!std::isfinite(prompt_strength) || prompt_strength < 0.0F ||
            prompt_strength > 1.0F) {
            throw std::invalid_argument("prompt strength must be in [0,1]");
        }
    }
};

struct GenerationMetrics {
    Modality source{Modality::Generic};
    Modality target{Modality::Generic};
    float latent_energy{};
    float output_energy{};
    float conditioning_mix{};
    std::size_t latent_elements{};
    std::size_t output_elements{};
};

class MultimodalGenerator3D {
public:
    explicit MultimodalGenerator3D(GeneratorConfig config)
        : config_(std::move(config)) {
        config_.validate();
        models_.reserve(kModalityCount);
        for (std::size_t i = 0U; i < kModalityCount; ++i) {
            Autoencoder3DConfig model_config = config_.autoencoder;
            model_config.seed ^= mix64(static_cast<std::uint64_t>(i) + 1ULL);
            models_.emplace_back(model_config);
        }
    }

    const GeneratorConfig& config() const noexcept { return config_; }

    Tensor4D tensorize(const MediaPacket& packet) const {
        const std::size_t edge = config_.autoencoder.input_edge;
        Tensor4D tensor({1U, edge, edge, edge});
        if (packet.modality == Modality::Text) {
            tensorize_text(packet, tensor);
        } else if (packet.modality == Modality::Image) {
            tensorize_image(packet, tensor);
        } else if (packet.modality == Modality::Audio) {
            tensorize_audio(packet, tensor);
        } else if (packet.modality == Modality::Video) {
            tensorize_video(packet, tensor);
        } else if (packet.modality == Modality::Volume3D) {
            tensorize_volume(packet, tensor);
        } else {
            tensorize_generic(packet, tensor);
        }
        return tensor;
    }

    Tensor4D encode(const MediaPacket& packet, bool quantized = false) const {
        return model(packet.modality).encode(tensorize(packet), quantized);
    }

    GeneratedMedia reconstruct(const MediaPacket& packet,
                               bool quantized = false) const {
        const Tensor4D decoded = model(packet.modality).reconstruct(
            tensorize(packet), quantized);
        return materialize(packet.modality, decoded, &packet);
    }

    Autoencoder3DMetrics train_step(const MediaPacket& packet) {
        return mutable_model(packet.modality).train_step(tensorize(packet));
    }

    GeneratedMedia generate(Modality target, const std::string& prompt,
                            std::uint64_t seed = 0ULL) const {
        Tensor4D latent = prompt_latent(prompt, seed);
        const Tensor4D decoded = model(target).decode(latent);
        return materialize(target, decoded, nullptr);
    }

    GeneratedMedia translate(const MediaPacket& source, Modality target,
                             const std::string& prompt = std::string(),
                             float conditioning_mix = 0.85F,
                             bool quantized = false,
                             std::uint64_t seed = 0ULL,
                             GenerationMetrics* metrics = nullptr) const {
        if (!std::isfinite(conditioning_mix) || conditioning_mix < 0.0F ||
            conditioning_mix > 1.0F) {
            throw std::invalid_argument("conditioning mix must be in [0,1]");
        }
        Tensor4D conditioned = encode(source, quantized);
        Tensor4D prompted = prompt_latent(prompt, seed);
        if (conditioned.shape().elements() != prompted.shape().elements()) {
            throw std::logic_error("latent geometry mismatch between modality models");
        }
        for (std::size_t i = 0U; i < conditioned.size(); ++i) {
            const float mixed = conditioning_mix * conditioned.values()[i] +
                (1.0F - conditioning_mix) * prompted.values()[i];
            conditioned.values()[i] = std::clamp(mixed, -1.0F, 1.0F);
        }
        const Tensor4D decoded = model(target).decode(conditioned);
        GeneratedMedia output = materialize(target, decoded, nullptr);
        if (metrics != nullptr) {
            metrics->source = source.modality;
            metrics->target = target;
            metrics->latent_energy = rms(conditioned.values());
            metrics->output_energy = rms(output.samples);
            metrics->conditioning_mix = conditioning_mix;
            metrics->latent_elements = conditioned.size();
            metrics->output_elements = output.samples.size();
        }
        return output;
    }

    Tensor4D prompt_latent(const std::string& prompt,
                           std::uint64_t seed = 0ULL) const {
        const std::size_t latent_edge = config_.autoencoder.input_edge / 2U;
        Tensor4D latent({config_.autoencoder.latent_channels,
                         latent_edge, latent_edge, latent_edge});
        std::uint64_t state = seed ^ hash_text(prompt) ^ config_.autoencoder.seed;
        if (state == 0ULL) state = 0x9E3779B97F4A7C15ULL;
        const float semantic = prompt.empty() ? 0.0F :
            static_cast<float>((hash_text(prompt) >> 11U) & 0xFFFFULL) / 32767.5F - 1.0F;
        for (std::size_t i = 0U; i < latent.size(); ++i) {
            state = xorshift64(state);
            const std::uint32_t low = static_cast<std::uint32_t>(state & 0xFFFFFFULL);
            const float noise = static_cast<float>(low) / 8388607.5F - 1.0F;
            const float phase = std::sin(static_cast<float>(i + 1U) * 0.17320508F +
                                         semantic * 1.6180339F);
            const float value = 0.70F * noise + config_.prompt_strength * phase +
                                0.12F * semantic;
            latent.values()[i] = std::clamp(value, -1.0F, 1.0F);
        }
        return latent;
    }

private:
    static constexpr std::size_t kModalityCount = 6U;
    GeneratorConfig config_;
    std::vector<Autoencoder3D> models_;

    static std::size_t modality_index(Modality modality) noexcept {
        return static_cast<std::size_t>(modality);
    }

    const Autoencoder3D& model(Modality modality) const {
        return models_.at(modality_index(modality));
    }

    Autoencoder3D& mutable_model(Modality modality) {
        return models_.at(modality_index(modality));
    }

    static float clamp_sample(float value) noexcept {
        if (!std::isfinite(value)) return 0.0F;
        return std::clamp(value, -1.0F, 1.0F);
    }

    static std::uint64_t mix64(std::uint64_t value) noexcept {
        value += 0x9E3779B97F4A7C15ULL;
        value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
        return value ^ (value >> 31U);
    }

    static std::uint64_t xorshift64(std::uint64_t value) noexcept {
        value ^= value << 13U;
        value ^= value >> 7U;
        value ^= value << 17U;
        return value;
    }

    static std::uint64_t hash_text(const std::string& text) noexcept {
        std::uint64_t hash = 1469598103934665603ULL;
        for (unsigned char byte : text) {
            hash ^= static_cast<std::uint64_t>(byte);
            hash *= 1099511628211ULL;
        }
        return hash;
    }

    static float rms(const std::vector<float>& values) noexcept {
        if (values.empty()) return 0.0F;
        double sum = 0.0;
        for (float value : values) sum += static_cast<double>(value) * value;
        return static_cast<float>(std::sqrt(sum / static_cast<double>(values.size())));
    }

    static std::size_t nearest(std::size_t coordinate, std::size_t dst_extent,
                               std::size_t src_extent) noexcept {
        if (src_extent <= 1U || dst_extent <= 1U) return 0U;
        const double scaled = static_cast<double>(coordinate) *
            static_cast<double>(src_extent - 1U) /
            static_cast<double>(dst_extent - 1U);
        return static_cast<std::size_t>(std::llround(scaled));
    }

    static float sample_channel_mean(const MediaPacket& packet,
                                     std::size_t base_index) noexcept {
        if (packet.samples.empty()) return 0.0F;
        const std::size_t channels = std::max<std::size_t>(1U, packet.channels);
        double sum = 0.0;
        std::size_t count = 0U;
        for (std::size_t channel = 0U; channel < channels; ++channel) {
            const std::size_t index = base_index + channel;
            if (index >= packet.samples.size()) break;
            sum += static_cast<double>(clamp_sample(packet.samples[index]));
            ++count;
        }
        return count == 0U ? 0.0F :
            static_cast<float>(sum / static_cast<double>(count));
    }

    void tensorize_text(const MediaPacket& packet, Tensor4D& tensor) const {
        const std::string& text = packet.text;
        if (text.empty()) return;
        const std::size_t edge = config_.autoencoder.input_edge;
        for (std::size_t i = 0U; i < tensor.size(); ++i) {
            const unsigned char byte = static_cast<unsigned char>(text[i % text.size()]);
            const float lexical = static_cast<float>(byte) / 127.5F - 1.0F;
            const std::size_t z = i / (edge * edge);
            const float depth_phase = std::sin(static_cast<float>(z + 1U) * 0.6180339F);
            tensor.values()[i] = std::clamp(0.88F * lexical + 0.12F * depth_phase,
                                            -1.0F, 1.0F);
        }
    }

    void tensorize_image(const MediaPacket& packet, Tensor4D& tensor) const {
        if (packet.samples.empty() || packet.width == 0U || packet.height == 0U) return;
        const std::size_t edge = config_.autoencoder.input_edge;
        const std::size_t channels = std::max<std::size_t>(1U, packet.channels);
        for (std::size_t z = 0U; z < edge; ++z) {
            for (std::size_t y = 0U; y < edge; ++y) {
                for (std::size_t x = 0U; x < edge; ++x) {
                    const std::size_t sx = nearest(x, edge, packet.width);
                    const std::size_t sy = nearest(y, edge, packet.height);
                    const std::size_t base = (sy * packet.width + sx) * channels;
                    const float value = sample_channel_mean(packet, base);
                    tensor(0U, z, y, x) = value;
                }
            }
        }
    }

    void tensorize_audio(const MediaPacket& packet, Tensor4D& tensor) const {
        if (packet.samples.empty()) return;
        for (std::size_t i = 0U; i < tensor.size(); ++i) {
            const std::size_t source = nearest(i, tensor.size(), packet.samples.size());
            tensor.values()[i] = clamp_sample(packet.samples[source]);
        }
    }

    void tensorize_video(const MediaPacket& packet, Tensor4D& tensor) const {
        if (packet.samples.empty() || packet.width == 0U || packet.height == 0U ||
            packet.frames == 0U) return;
        const std::size_t edge = config_.autoencoder.input_edge;
        const std::size_t channels = std::max<std::size_t>(1U, packet.channels);
        const std::size_t frame_stride = packet.width * packet.height * channels;
        for (std::size_t z = 0U; z < edge; ++z) {
            const std::size_t frame = nearest(z, edge, packet.frames);
            for (std::size_t y = 0U; y < edge; ++y) {
                for (std::size_t x = 0U; x < edge; ++x) {
                    const std::size_t sx = nearest(x, edge, packet.width);
                    const std::size_t sy = nearest(y, edge, packet.height);
                    const std::size_t base = frame * frame_stride +
                                             (sy * packet.width + sx) * channels;
                    tensor(0U, z, y, x) = sample_channel_mean(packet, base);
                }
            }
        }
    }

    void tensorize_volume(const MediaPacket& packet, Tensor4D& tensor) const {
        if (packet.samples.empty() || packet.width == 0U || packet.height == 0U ||
            packet.depth == 0U) return;
        const std::size_t edge = config_.autoencoder.input_edge;
        const std::size_t channels = std::max<std::size_t>(1U, packet.channels);
        for (std::size_t z = 0U; z < edge; ++z) {
            const std::size_t sz = nearest(z, edge, packet.depth);
            for (std::size_t y = 0U; y < edge; ++y) {
                const std::size_t sy = nearest(y, edge, packet.height);
                for (std::size_t x = 0U; x < edge; ++x) {
                    const std::size_t sx = nearest(x, edge, packet.width);
                    const std::size_t base =
                        ((sz * packet.height + sy) * packet.width + sx) * channels;
                    tensor(0U, z, y, x) = sample_channel_mean(packet, base);
                }
            }
        }
    }

    void tensorize_generic(const MediaPacket& packet, Tensor4D& tensor) const {
        if (packet.samples.empty()) return;
        for (std::size_t i = 0U; i < tensor.size(); ++i) {
            tensor.values()[i] = clamp_sample(packet.samples[i % packet.samples.size()]);
        }
    }

    float decoded_sample(const Tensor4D& decoded,
                         std::size_t linear_index,
                         std::size_t output_count) const noexcept {
        if (output_count <= 1U) return clamp_sample(decoded.values().front());
        const std::size_t index = nearest(linear_index, output_count, decoded.size());
        return clamp_sample(decoded.values()[index]);
    }

    GeneratedMedia materialize(Modality target, const Tensor4D& decoded,
                               const MediaPacket* shape_hint) const {
        GeneratedMedia output;
        output.modality = target;
        if (target == Modality::Text) {
            const std::size_t count = shape_hint != nullptr && !shape_hint->text.empty()
                ? shape_hint->text.size() : config_.default_text_length;
            output.text.reserve(count);
            output.samples.reserve(count);
            for (std::size_t i = 0U; i < count; ++i) {
                const float value = decoded_sample(decoded, i, count);
                output.samples.push_back(value);
                const int code = 32 + static_cast<int>(std::lround((value + 1.0F) * 47.0F));
                output.text.push_back(static_cast<char>(std::clamp(code, 32, 126)));
            }
            return output;
        }
        if (target == Modality::Image) {
            output.width = shape_hint != nullptr && shape_hint->width != 0U
                ? shape_hint->width : config_.default_image_edge;
            output.height = shape_hint != nullptr && shape_hint->height != 0U
                ? shape_hint->height : config_.default_image_edge;
            output.channels = 1U;
            const std::size_t count = output.width * output.height;
            output.samples.resize(count);
            const std::size_t edge = config_.autoencoder.input_edge;
            const std::size_t z = edge / 2U;
            for (std::size_t y = 0U; y < output.height; ++y) {
                for (std::size_t x = 0U; x < output.width; ++x) {
                    const std::size_t sx = nearest(x, output.width, edge);
                    const std::size_t sy = nearest(y, output.height, edge);
                    output.samples[y * output.width + x] = decoded(0U, z, sy, sx);
                }
            }
            return output;
        }
        if (target == Modality::Audio) {
            output.sample_rate = shape_hint != nullptr ? shape_hint->sample_rate : 44100U;
            const std::size_t count = shape_hint != nullptr && !shape_hint->samples.empty()
                ? shape_hint->samples.size() : config_.default_audio_samples;
            output.samples.resize(count);
            for (std::size_t i = 0U; i < count; ++i) {
                output.samples[i] = decoded_sample(decoded, i, count);
            }
            return output;
        }
        if (target == Modality::Video) {
            output.width = shape_hint != nullptr && shape_hint->width != 0U
                ? shape_hint->width : config_.default_video_edge;
            output.height = shape_hint != nullptr && shape_hint->height != 0U
                ? shape_hint->height : config_.default_video_edge;
            output.frames = shape_hint != nullptr && shape_hint->frames != 0U
                ? shape_hint->frames : config_.default_video_frames;
            output.channels = 1U;
            const std::size_t count = output.width * output.height * output.frames;
            output.samples.resize(count);
            const std::size_t edge = config_.autoencoder.input_edge;
            for (std::size_t frame = 0U; frame < output.frames; ++frame) {
                const std::size_t z = nearest(frame, output.frames, edge);
                for (std::size_t y = 0U; y < output.height; ++y) {
                    const std::size_t sy = nearest(y, output.height, edge);
                    for (std::size_t x = 0U; x < output.width; ++x) {
                        const std::size_t sx = nearest(x, output.width, edge);
                        const std::size_t index =
                            (frame * output.height + y) * output.width + x;
                        output.samples[index] = decoded(0U, z, sy, sx);
                    }
                }
            }
            return output;
        }
        if (target == Modality::Volume3D) {
            const std::size_t edge = shape_hint != nullptr && shape_hint->width != 0U
                ? shape_hint->width : config_.autoencoder.input_edge;
            output.width = edge;
            output.height = shape_hint != nullptr && shape_hint->height != 0U
                ? shape_hint->height : edge;
            output.depth = shape_hint != nullptr && shape_hint->depth != 0U
                ? shape_hint->depth : edge;
            output.channels = 1U;
            const std::size_t count = output.width * output.height * output.depth;
            output.samples.resize(count);
            const std::size_t model_edge = config_.autoencoder.input_edge;
            for (std::size_t z = 0U; z < output.depth; ++z) {
                const std::size_t sz = nearest(z, output.depth, model_edge);
                for (std::size_t y = 0U; y < output.height; ++y) {
                    const std::size_t sy = nearest(y, output.height, model_edge);
                    for (std::size_t x = 0U; x < output.width; ++x) {
                        const std::size_t sx = nearest(x, output.width, model_edge);
                        output.samples[(z * output.height + y) * output.width + x] =
                            decoded(0U, sz, sy, sx);
                    }
                }
            }
            return output;
        }
        output.samples = decoded.values();
        output.width = decoded.shape().width;
        output.height = decoded.shape().height;
        output.depth = decoded.shape().depth;
        output.channels = decoded.shape().channels;
        return output;
    }
};

} // namespace jarvisx::mm3d
