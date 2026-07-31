#pragma once

#include "jarvisx/autoencoder3d.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace jarvisx {

enum class MediaType : std::uint8_t {
    Visual = 0,
    Audio = 1,
    Text = 2,
    Generic = 3
};

constexpr std::size_t kMediaTypeCount = 4U;

inline std::size_t media_index(MediaType media) noexcept {
    return static_cast<std::size_t>(media);
}

inline const char* media_name(MediaType media) noexcept {
    switch (media) {
    case MediaType::Visual: return "visual";
    case MediaType::Audio: return "audio";
    case MediaType::Text: return "text";
    case MediaType::Generic: return "generic";
    }
    return "unknown";
}

inline std::array<MediaType, kMediaTypeCount> media_types() noexcept {
    return {MediaType::Visual, MediaType::Audio,
            MediaType::Text, MediaType::Generic};
}

struct Multimedia4DConfig {
    Autoencoder3DConfig model{};
    std::size_t temporal_depth{8U};
    float temporal_decay{0.72F};
    std::uint16_t proposal_steps{2U};
    float accept_tolerance{1.0e-6F};
    bool quantized_inference{};

    void validate() const {
        model.validate();
        if (temporal_depth < 1U || temporal_depth > 64U) {
            throw std::invalid_argument("temporal depth must be in [1, 64]");
        }
        if (!std::isfinite(temporal_decay) || temporal_decay < 0.0F ||
            temporal_decay >= 1.0F) {
            throw std::invalid_argument("temporal decay must be in [0, 1)");
        }
        if (proposal_steps < 1U || proposal_steps > 64U) {
            throw std::invalid_argument("proposal steps must be in [1, 64]");
        }
        if (!std::isfinite(accept_tolerance) || accept_tolerance < 0.0F ||
            accept_tolerance > 1.0F) {
            throw std::invalid_argument("accept tolerance must be in [0, 1]");
        }
    }
};

struct ModalityMetrics {
    float instantaneous_mse{std::numeric_limits<float>::infinity()};
    float temporal_mse{std::numeric_limits<float>::infinity()};
    float mae{};
    float latent_energy{};
    float temporal_coherence{};
    float gradient_l2{};
    std::uint64_t accepted{};
    std::uint64_t rejected{};
    std::uint64_t model_steps{};
};

struct Multimedia4DMetrics {
    std::uint64_t cycle{};
    MediaType selected{MediaType::Visual};
    float aggregate_mse{};
    float aggregate_temporal_coherence{};
    std::uint64_t accepted{};
    std::uint64_t rejected{};
    std::array<ModalityMetrics, kMediaTypeCount> modalities{};
};

inline Tensor4D make_media_volume(std::size_t edge, MediaType media,
                                  std::uint64_t seed) {
    if (edge < 2U) throw std::invalid_argument("media volume edge below 2");
    if (media == MediaType::Visual) {
        return make_volume(edge, "sphere", seed ^ 0x56495355414CULL);
    }
    if (media == MediaType::Generic) {
        return make_volume(edge, "noise", seed ^ 0x47454E45524943ULL);
    }

    Tensor4D volume({1U, edge, edge, edge});
    constexpr const char* text =
        "JARVIS X MULTIMODAL AUTOENCODING DECODING RUNTIME";
    constexpr std::size_t text_size =
        sizeof("JARVIS X MULTIMODAL AUTOENCODING DECODING RUNTIME") - 1U;
    const float center = 0.5F * static_cast<float>(edge - 1U);
    const float inverse_edge = 1.0F / static_cast<float>(edge);

    for (std::size_t z = 0; z < edge; ++z) {
        for (std::size_t y = 0; y < edge; ++y) {
            for (std::size_t x = 0; x < edge; ++x) {
                float value = 0.0F;
                if (media == MediaType::Audio) {
                    const float fx = (static_cast<float>(x) - center) * inverse_edge;
                    const float fy = (static_cast<float>(y) - center) * inverse_edge;
                    const float fz = (static_cast<float>(z) - center) * inverse_edge;
                    const float carrier = std::sin(
                        18.0F * fx + 11.0F * fy + 7.0F * fz);
                    const float envelope = std::exp(
                        -4.0F * (fx * fx + fy * fy + fz * fz));
                    value = carrier * envelope;
                } else {
                    const std::size_t token_index =
                        (x + 3U * y + 7U * z) % text_size;
                    const std::uint8_t token =
                        static_cast<std::uint8_t>(text[token_index]);
                    const std::uint8_t bit = static_cast<std::uint8_t>(
                        1U << ((x + y + z) % 7U));
                    const float lexical = (token & bit) != 0U ? 1.0F : -1.0F;
                    const float positional = 0.25F * std::sin(
                        0.7F * static_cast<float>(x) +
                        0.5F * static_cast<float>(y) +
                        0.3F * static_cast<float>(z));
                    value = clampf(0.8F * lexical + positional, -1.0F, 1.0F);
                }
                volume(0U, z, y, x) = value;
            }
        }
    }
    return volume;
}

inline float tensor_rms_difference(const Tensor4D& left,
                                   const Tensor4D& right) {
    if (left.shape().elements() != right.shape().elements()) {
        throw std::invalid_argument("tensor RMS shape mismatch");
    }
    double square_sum = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double delta = static_cast<double>(left.values()[index]) -
                             static_cast<double>(right.values()[index]);
        square_sum += delta * delta;
    }
    return static_cast<float>(std::sqrt(
        square_sum / static_cast<double>(left.size())));
}

inline Tensor4D fuse_temporal_latents(const std::deque<Tensor4D>& history,
                                      float decay) {
    if (history.empty()) {
        throw std::invalid_argument("cannot fuse empty temporal history");
    }
    Tensor4D fused(history.front().shape(), 0.0F);
    double weight_sum = 0.0;
    double weight = 1.0;
    for (const Tensor4D& frame : history) {
        if (frame.shape().elements() != fused.shape().elements()) {
            throw std::invalid_argument("temporal history shape mismatch");
        }
        for (std::size_t index = 0; index < fused.size(); ++index) {
            fused.values()[index] +=
                static_cast<float>(weight * frame.values()[index]);
        }
        weight_sum += weight;
        weight *= static_cast<double>(decay);
    }
    const float inverse = static_cast<float>(1.0 / weight_sum);
    for (float& value : fused.values()) value *= inverse;
    return fused;
}

class MultimediaAutoencoder4D {
public:
    explicit MultimediaAutoencoder4D(Multimedia4DConfig config)
        : config_(std::move(config)), quantized_(config_.quantized_inference) {
        config_.validate();
        initialize_states({});
        refresh_metrics(MediaType::Visual);
    }

    const Multimedia4DConfig& config() const noexcept { return config_; }
    const Multimedia4DMetrics& metrics() const noexcept { return metrics_; }
    bool quantized() const noexcept { return quantized_; }

    const Tensor4D& input(MediaType media) const {
        return state(media).input;
    }

    const Tensor4D& latent(MediaType media) const {
        return state(media).temporal_latent;
    }

    const Tensor4D& current_latent(MediaType media) const {
        return state(media).current_latent;
    }

    const Tensor4D& reconstruction(MediaType media) const {
        return state(media).reconstruction;
    }

    const std::deque<Tensor4D>& temporal_history(MediaType media) const {
        return state(media).history;
    }

    void set_quantized(bool enabled) {
        quantized_ = enabled;
        for (const MediaType media : media_types()) {
            refresh_state(media, false, 0.0F);
        }
        refresh_metrics(metrics_.selected);
    }

    Multimedia4DMetrics step() {
        const MediaType selected = select_modality();
        ModalityState& target = state(selected);

        const float baseline = instantaneous_mse(target.model, target.input);
        Autoencoder3D candidate = target.model;
        Autoencoder3DMetrics candidate_training{};
        for (std::uint16_t step_index = 0U;
             step_index < config_.proposal_steps; ++step_index) {
            candidate_training = candidate.train_step(target.input);
        }
        const float proposal = instantaneous_mse(candidate, target.input);
        const bool accept = std::isfinite(proposal) &&
            proposal <= baseline + config_.accept_tolerance;

        if (accept) {
            target.model = std::move(candidate);
            ++target.accepted;
        } else {
            ++target.rejected;
        }

        ++cycle_;
        refresh_state(selected, true,
                      accept ? candidate_training.gradient_l2 : 0.0F);
        refresh_metrics(selected);
        return metrics_;
    }

    void save_checkpoint(const fs::path& directory) const {
        fs::create_directories(directory);
        std::ofstream manifest(directory / "multimedia4d.manifest",
                               std::ios::trunc);
        if (!manifest) {
            throw std::runtime_error("cannot write multimedia4d manifest");
        }
        manifest << std::setprecision(std::numeric_limits<float>::max_digits10)
                 << "JARVISX_MULTIMEDIA4D_V1\n"
                 << config_.temporal_depth << ' '
                 << config_.temporal_decay << ' '
                 << config_.proposal_steps << ' '
                 << config_.accept_tolerance << ' '
                 << cycle_ << ' ' << (quantized_ ? 1 : 0) << '\n';
        for (const MediaType media : media_types()) {
            const ModalityState& item = state(media);
            manifest << media_name(media) << ' '
                     << item.accepted << ' ' << item.rejected << ' '
                     << item.gradient_l2 << '\n';
            const std::string prefix = media_name(media);
            item.model.save(directory / (prefix + ".jx3d"));
            write_history(directory / (prefix + ".history"), item.history);
        }
        if (!manifest) {
            throw std::runtime_error("cannot flush multimedia4d manifest");
        }
    }

    void load_checkpoint(const fs::path& directory) {
        std::ifstream manifest(directory / "multimedia4d.manifest");
        if (!manifest) {
            throw std::runtime_error("cannot read multimedia4d manifest");
        }
        std::string magic;
        std::getline(manifest, magic);
        if (magic != "JARVISX_MULTIMEDIA4D_V1") {
            throw std::runtime_error("unsupported multimedia4d checkpoint");
        }

        std::size_t temporal_depth = 0U;
        float temporal_decay = 0.0F;
        std::uint16_t proposal_steps = 0U;
        float accept_tolerance = 0.0F;
        std::uint64_t cycle = 0U;
        int quantized_flag = 0;
        manifest >> temporal_depth >> temporal_decay >> proposal_steps
                 >> accept_tolerance >> cycle >> quantized_flag;
        if (!manifest || temporal_depth != config_.temporal_depth ||
            std::fabs(temporal_decay - config_.temporal_decay) > 1.0e-6F ||
            proposal_steps != config_.proposal_steps ||
            std::fabs(accept_tolerance - config_.accept_tolerance) > 1.0e-6F) {
            throw std::runtime_error("multimedia4d checkpoint config mismatch");
        }

        std::array<std::uint64_t, kMediaTypeCount> accepted{};
        std::array<std::uint64_t, kMediaTypeCount> rejected{};
        std::array<float, kMediaTypeCount> gradients{};
        for (const MediaType media : media_types()) {
            std::string name;
            manifest >> name >> accepted[media_index(media)]
                     >> rejected[media_index(media)]
                     >> gradients[media_index(media)];
            if (!manifest || name != media_name(media)) {
                throw std::runtime_error("multimedia4d modality manifest mismatch");
            }
        }

        std::vector<Autoencoder3D> models;
        models.reserve(kMediaTypeCount);
        for (const MediaType media : media_types()) {
            models.push_back(Autoencoder3D::load(
                directory / (std::string(media_name(media)) + ".jx3d")));
            validate_model_compatibility(models.back().config());
        }

        cycle_ = cycle;
        quantized_ = quantized_flag != 0;
        initialize_states(std::move(models));
        for (const MediaType media : media_types()) {
            state(media).accepted = accepted[media_index(media)];
            state(media).rejected = rejected[media_index(media)];
            state(media).gradient_l2 = gradients[media_index(media)];
            const std::string prefix = media_name(media);
            state(media).history = read_history(
                directory / (prefix + ".history"),
                state(media).current_latent.shape());
            state(media).current_latent = state(media).history.front();
            state(media).temporal_latent = fuse_temporal_latents(
                state(media).history, config_.temporal_decay);
            state(media).reconstruction =
                state(media).model.decode(state(media).temporal_latent);
        }
        refresh_metrics(metrics_.selected);
    }

private:
    struct ModalityState {
        MediaType media;
        Autoencoder3D model;
        Tensor4D input;
        Tensor4D current_latent;
        Tensor4D temporal_latent;
        Tensor4D reconstruction;
        std::deque<Tensor4D> history;
        float gradient_l2{};
        std::uint64_t accepted{};
        std::uint64_t rejected{};

        ModalityState(MediaType media_value, Autoencoder3D model_value,
                      Tensor4D input_value, bool quantized)
            : media(media_value), model(std::move(model_value)),
              input(std::move(input_value)),
              current_latent(model.encode(input, quantized)),
              temporal_latent(current_latent),
              reconstruction(model.decode(temporal_latent)) {
            history.push_front(current_latent);
        }
    };

    Multimedia4DConfig config_;
    std::vector<ModalityState> states_;
    Multimedia4DMetrics metrics_{};
    std::uint64_t cycle_{};
    bool quantized_{};

    ModalityState& state(MediaType media) {
        return states_.at(media_index(media));
    }

    const ModalityState& state(MediaType media) const {
        return states_.at(media_index(media));
    }

    static void write_history(const fs::path& path,
                              const std::deque<Tensor4D>& history) {
        std::ofstream output(path, std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write temporal history");
        output << history.size() << '\n'
               << std::setprecision(std::numeric_limits<float>::max_digits10);
        for (const Tensor4D& frame : history) {
            output << frame.size();
            for (const float value : frame.values()) output << ' ' << value;
            output << '\n';
        }
        if (!output) throw std::runtime_error("cannot flush temporal history");
    }

    static std::deque<Tensor4D> read_history(
        const fs::path& path, const TensorShape4D& shape) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot read temporal history");
        std::size_t count = 0U;
        input >> count;
        if (count == 0U || count > 64U) {
            throw std::runtime_error("invalid temporal history length");
        }
        std::deque<Tensor4D> history;
        for (std::size_t frame_index = 0U; frame_index < count; ++frame_index) {
            std::size_t size = 0U;
            input >> size;
            if (size != shape.elements()) {
                throw std::runtime_error("temporal history tensor mismatch");
            }
            Tensor4D frame(shape);
            for (float& value : frame.values()) input >> value;
            if (!input) throw std::runtime_error("truncated temporal history");
            history.push_back(std::move(frame));
        }
        return history;
    }

    void initialize_states(std::vector<Autoencoder3D> models) {
        states_.clear();
        states_.reserve(kMediaTypeCount);
        const bool use_supplied = models.size() == kMediaTypeCount;
        for (const MediaType media : media_types()) {
            Autoencoder3D model = use_supplied
                ? std::move(models[media_index(media)])
                : Autoencoder3D(modality_config(media));
            Tensor4D input_tensor = make_media_volume(
                model.config().input_edge, media, model.config().seed);
            states_.emplace_back(media, std::move(model),
                                 std::move(input_tensor), quantized_);
        }
        for (const MediaType media : media_types()) {
            refresh_state(media, false, 0.0F);
        }
    }

    Autoencoder3DConfig modality_config(MediaType media) const {
        Autoencoder3DConfig model = config_.model;
        model.seed = mix64(model.seed ^
            (0x9E3779B97F4A7C15ULL *
             static_cast<std::uint64_t>(media_index(media) + 1U)));
        return model;
    }

    void validate_model_compatibility(const Autoencoder3DConfig& model) const {
        if (model.input_edge != config_.model.input_edge ||
            model.latent_channels != config_.model.latent_channels) {
            throw std::runtime_error("multimedia4d model geometry mismatch");
        }
    }

    static float instantaneous_mse(const Autoencoder3D& model,
                                   const Tensor4D& input) {
        const Tensor4D latent = model.encode(input, false);
        const Tensor4D reconstruction = model.decode(latent);
        return measure_reconstruction(input, latent, reconstruction,
                                      model.steps()).mse;
    }

    MediaType select_modality() const {
        MediaType selected = MediaType::Visual;
        float selected_score = -std::numeric_limits<float>::infinity();
        for (std::size_t offset = 0U; offset < kMediaTypeCount; ++offset) {
            const std::size_t index =
                (static_cast<std::size_t>(cycle_) + offset) % kMediaTypeCount;
            const MediaType media = static_cast<MediaType>(index);
            const ModalityMetrics& metric = metrics_.modalities[index];
            const float rejection_pressure = 0.001F * static_cast<float>(
                metric.rejected) / static_cast<float>(1U + metric.accepted);
            const float score = metric.instantaneous_mse + rejection_pressure;
            if (score > selected_score) {
                selected_score = score;
                selected = media;
            }
        }
        return selected;
    }

    void refresh_state(MediaType media, bool append_history,
                       float gradient_l2) {
        ModalityState& item = state(media);
        item.current_latent = item.model.encode(item.input, quantized_);
        if (append_history || item.history.empty()) {
            item.history.push_front(item.current_latent);
            while (item.history.size() > config_.temporal_depth) {
                item.history.pop_back();
            }
        } else {
            item.history.front() = item.current_latent;
        }
        item.temporal_latent = fuse_temporal_latents(
            item.history, config_.temporal_decay);
        item.reconstruction = item.model.decode(item.temporal_latent);
        item.gradient_l2 = gradient_l2;
    }

    void refresh_metrics(MediaType selected) {
        metrics_ = {};
        metrics_.cycle = cycle_;
        metrics_.selected = selected;
        double mse_sum = 0.0;
        double coherence_sum = 0.0;

        for (const MediaType media : media_types()) {
            const ModalityState& item = state(media);
            const Tensor4D immediate_reconstruction =
                item.model.decode(item.current_latent);
            const Autoencoder3DMetrics immediate = measure_reconstruction(
                item.input, item.current_latent, immediate_reconstruction,
                item.model.steps());
            const Autoencoder3DMetrics temporal = measure_reconstruction(
                item.input, item.temporal_latent, item.reconstruction,
                item.model.steps());
            float coherence = 1.0F;
            if (item.history.size() > 1U) {
                coherence = 1.0F - clampf(
                    tensor_rms_difference(item.history[0], item.history[1]),
                    0.0F, 1.0F);
            }
            ModalityMetrics metric;
            metric.instantaneous_mse = immediate.mse;
            metric.temporal_mse = temporal.mse;
            metric.mae = temporal.mae;
            metric.latent_energy = temporal.latent_energy;
            metric.temporal_coherence = coherence;
            metric.gradient_l2 = item.gradient_l2;
            metric.accepted = item.accepted;
            metric.rejected = item.rejected;
            metric.model_steps = item.model.steps();
            metrics_.modalities[media_index(media)] = metric;
            metrics_.accepted += item.accepted;
            metrics_.rejected += item.rejected;
            mse_sum += metric.temporal_mse;
            coherence_sum += metric.temporal_coherence;
        }
        metrics_.aggregate_mse = static_cast<float>(
            mse_sum / static_cast<double>(kMediaTypeCount));
        metrics_.aggregate_temporal_coherence = static_cast<float>(
            coherence_sum / static_cast<double>(kMediaTypeCount));
    }
};

inline void export_multimedia4d_snapshot(
    const MultimediaAutoencoder4D& engine, const fs::path& directory) {
    fs::create_directories(directory);
    for (const MediaType media : media_types()) {
        const std::string prefix = media_name(media);
        export_obj(engine.input(media), directory / (prefix + "-input.obj"));
        export_obj(engine.latent(media), directory / (prefix + "-latent.obj"));
        export_obj(engine.reconstruction(media),
                   directory / (prefix + "-reconstruction.obj"));
    }
}

} // namespace jarvisx
