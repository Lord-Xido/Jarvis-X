#include "jarvisx/multimodal_generator3d.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

namespace {

jarvisx::mm3d::GeneratorConfig test_config() {
    jarvisx::mm3d::GeneratorConfig config;
    config.autoencoder.input_edge = 8U;
    config.autoencoder.latent_channels = 3U;
    config.autoencoder.seed = 0x12345678ULL;
    config.default_text_length = 64U;
    config.default_image_edge = 16U;
    config.default_audio_samples = 128U;
    config.default_video_frames = 4U;
    config.default_video_edge = 8U;
    return config;
}

void assert_bounded(const std::vector<float>& values) {
    for (float value : values) {
        assert(std::isfinite(value));
        assert(value >= -1.000001F);
        assert(value <= 1.000001F);
    }
}

void test_text_tensorization_and_latent_shape() {
    jarvisx::mm3d::MultimodalGenerator3D engine(test_config());
    jarvisx::mm3d::MediaPacket packet;
    packet.modality = jarvisx::mm3d::Modality::Text;
    packet.text = "identity preserving multimodal latent geometry";

    const jarvisx::Tensor4D tensor = engine.tensorize(packet);
    assert(tensor.shape().channels == 1U);
    assert(tensor.shape().depth == 8U);
    assert(tensor.size() == 512U);

    const jarvisx::Tensor4D latent = engine.encode(packet);
    assert(latent.shape().channels == 3U);
    assert(latent.shape().depth == 4U);
    assert(latent.shape().height == 4U);
    assert(latent.shape().width == 4U);
    assert(latent.size() == 192U);
}

void test_prompt_generation_is_deterministic() {
    jarvisx::mm3d::MultimodalGenerator3D first(test_config());
    jarvisx::mm3d::MultimodalGenerator3D second(test_config());

    const auto a = first.generate(jarvisx::mm3d::Modality::Audio,
                                  "same prompt same latent", 99ULL);
    const auto b = second.generate(jarvisx::mm3d::Modality::Audio,
                                   "same prompt same latent", 99ULL);
    assert(a.samples == b.samples);
    assert(a.samples.size() == 128U);
    assert_bounded(a.samples);
}

void test_cross_modal_outputs() {
    jarvisx::mm3d::MultimodalGenerator3D engine(test_config());
    jarvisx::mm3d::MediaPacket source;
    source.modality = jarvisx::mm3d::Modality::Text;
    source.text = "cross modal conditioning fixture";

    jarvisx::mm3d::GenerationMetrics metrics;
    const auto image = engine.translate(source, jarvisx::mm3d::Modality::Image,
                                        "render geometry", 0.75F, false, 7ULL,
                                        &metrics);
    assert(image.width == 16U);
    assert(image.height == 16U);
    assert(image.samples.size() == 256U);
    assert(metrics.latent_elements == 192U);
    assert(metrics.output_elements == image.samples.size());
    assert_bounded(image.samples);

    const auto video = engine.translate(source, jarvisx::mm3d::Modality::Video,
                                        "animate geometry", 0.65F);
    assert(video.frames == 4U);
    assert(video.width == 8U);
    assert(video.height == 8U);
    assert(video.samples.size() == 256U);
    assert_bounded(video.samples);

    const auto volume = engine.translate(source, jarvisx::mm3d::Modality::Volume3D,
                                         "volumetric geometry", 0.9F);
    assert(volume.width == 8U);
    assert(volume.height == 8U);
    assert(volume.depth == 8U);
    assert(volume.samples.size() == 512U);
    assert_bounded(volume.samples);
}

void test_training_path_remains_available() {
    jarvisx::mm3d::MultimodalGenerator3D engine(test_config());
    jarvisx::mm3d::MediaPacket packet;
    packet.modality = jarvisx::mm3d::Modality::Audio;
    packet.samples.resize(256U);
    for (std::size_t i = 0U; i < packet.samples.size(); ++i) {
        packet.samples[i] = std::sin(static_cast<float>(i) * 0.07F);
    }
    const auto before = engine.reconstruct(packet);
    const auto metrics = engine.train_step(packet);
    const auto after = engine.reconstruct(packet);
    assert(std::isfinite(metrics.mse));
    assert(before.samples.size() == after.samples.size());
    assert_bounded(after.samples);
}

} // namespace

int main() {
    test_text_tensorization_and_latent_shape();
    test_prompt_generation_is_deterministic();
    test_cross_modal_outputs();
    test_training_path_remains_available();
    std::cout << "multimodal generator 3D regressions passed\n";
    return 0;
}
