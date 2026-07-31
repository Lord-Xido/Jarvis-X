#include "jarvisx/multimedia4d.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

jarvisx::Multimedia4DConfig test_config() {
    jarvisx::Multimedia4DConfig config;
    config.model.input_edge = 8U;
    config.model.latent_channels = 3U;
    config.model.learning_rate = 0.02F;
    config.model.seed = 0x4D554C54493444ULL;
    config.temporal_depth = 4U;
    config.temporal_decay = 0.65F;
    config.proposal_steps = 2U;
    config.accept_tolerance = 1.0e-6F;
    return config;
}

void test_media_fixtures_are_distinct() {
    const auto visual = jarvisx::make_media_volume(
        8U, jarvisx::MediaType::Visual, 7U);
    const auto audio = jarvisx::make_media_volume(
        8U, jarvisx::MediaType::Audio, 7U);
    const auto text = jarvisx::make_media_volume(
        8U, jarvisx::MediaType::Text, 7U);
    double visual_audio = 0.0;
    double visual_text = 0.0;
    for (std::size_t index = 0U; index < visual.size(); ++index) {
        visual_audio += std::fabs(
            static_cast<double>(visual.values()[index] - audio.values()[index]));
        visual_text += std::fabs(
            static_cast<double>(visual.values()[index] - text.values()[index]));
    }
    assert(visual_audio > 1.0);
    assert(visual_text > 1.0);
}

void test_deterministic_transaction_schedule() {
    const auto config = test_config();
    jarvisx::MultimediaAutoencoder4D left(config);
    jarvisx::MultimediaAutoencoder4D right(config);

    for (int cycle = 0; cycle < 12; ++cycle) {
        const auto before = left.metrics();
        const auto left_metrics = left.step();
        const auto right_metrics = right.step();
        assert(left_metrics.selected == right_metrics.selected);
        assert(left_metrics.aggregate_mse == right_metrics.aggregate_mse);
        assert(left_metrics.accepted == right_metrics.accepted);
        assert(left_metrics.rejected == right_metrics.rejected);
        const std::size_t selected = jarvisx::media_index(left_metrics.selected);
        assert(left_metrics.modalities[selected].instantaneous_mse <=
               before.modalities[selected].instantaneous_mse +
                   config.accept_tolerance + 1.0e-6F);
    }

    assert(left.metrics().accepted + left.metrics().rejected == 12U);
    for (const auto media : jarvisx::media_types()) {
        assert(left.temporal_history(media).size() <= config.temporal_depth);
    }
}

void test_signed_three_bit_inference() {
    const auto config = test_config();
    jarvisx::MultimediaAutoencoder4D engine(config);
    engine.set_quantized(true);
    for (const auto media : jarvisx::media_types()) {
        for (const float value : engine.current_latent(media).values()) {
            const float level = jarvisx::dequantize_q3(
                jarvisx::quantize_q3(value));
            assert(std::fabs(value - level) < 1.0e-6F);
        }
    }
}

void test_checkpoint_replay() {
    const auto config = test_config();
    jarvisx::MultimediaAutoencoder4D source(config);
    for (int cycle = 0; cycle < 9; ++cycle) source.step();
    source.set_quantized(true);

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "jarvisx-multimedia4d-tests";
    std::filesystem::remove_all(directory);
    source.save_checkpoint(directory);

    jarvisx::MultimediaAutoencoder4D restored(config);
    restored.load_checkpoint(directory);
    assert(restored.metrics().cycle == source.metrics().cycle);
    assert(restored.metrics().accepted == source.metrics().accepted);
    assert(restored.metrics().rejected == source.metrics().rejected);
    assert(restored.metrics().aggregate_mse ==
           source.metrics().aggregate_mse);
    assert(restored.quantized() == source.quantized());
    for (const auto media : jarvisx::media_types()) {
        assert(restored.temporal_history(media).size() ==
               source.temporal_history(media).size());
    }
    std::filesystem::remove_all(directory);
}

} // namespace

int main() {
    test_media_fixtures_are_distinct();
    test_deterministic_transaction_schedule();
    test_signed_three_bit_inference();
    test_checkpoint_replay();
    std::cout << "4D multimedia runtime tests passed\n";
    return 0;
}
