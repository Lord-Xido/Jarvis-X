#include "jarvisx/bitmatrix3d.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

jarvisx::Tensor4D fixture(std::size_t edge) {
    jarvisx::Tensor4D tensor({1U, edge, edge, edge});
    const float center = (static_cast<float>(edge) - 1.0F) * 0.5F;
    const float denom = std::max(center, 1.0F);
    for (std::size_t z = 0; z < edge; ++z) {
        for (std::size_t y = 0; y < edge; ++y) {
            for (std::size_t x = 0; x < edge; ++x) {
                const float dx = (static_cast<float>(x) - center) / denom;
                const float dy = (static_cast<float>(y) - center) / denom;
                const float dz = (static_cast<float>(z) - center) / denom;
                const float radius = std::sqrt(dx * dx + dy * dy + dz * dz);
                tensor(0U, z, y, x) = radius < 0.7F ? 1.0F : -0.2F;
            }
        }
    }
    return tensor;
}

void ternary_round_trip() {
    const std::vector<std::int8_t> values{1, 0, -1, 1, -1, 0, 0, 1, -1};
    const auto packed = jarvisx::PackedTernary::pack(values);
    require(packed.unpack() == values, "packed ternary planes must round-trip exactly");
    require(packed.nonzero_count() == 6U, "packed ternary nonzero count mismatch");
}

void bitwise_dot_primitives() {
    const std::uint64_t a = 0b1011U;
    const std::uint64_t b = 0b1001U;
    require(jarvisx::PackedTernary::binary_dot(a, b, 4U) == 2,
            "binary XOR/popcount dot product mismatch");
    const std::uint64_t ternary_sign = 0b1001U;
    const std::uint64_t ternary_mask = 0b1101U;
    require(jarvisx::PackedTernary::ternary_binary_dot(a, ternary_sign, ternary_mask, 4U) == 3,
            "ternary masked dot product mismatch");
}

void deterministic_forward() {
    jarvisx::BitMatrix3DConfig config;
    config.seed = 99U;
    config.ternary_threshold = 0.25F;
    jarvisx::BitMatrix3DEngine first(config);
    jarvisx::BitMatrix3DEngine second(config);
    const auto input = fixture(config.input_edge);
    const auto a = first.forward(input);
    const auto b = second.forward(input);
    require(a.reconstruction.values() == b.reconstruction.values(),
            "same seed must produce deterministic reconstruction");
    require(a.latent_packed.unpack() == b.latent_packed.unpack(),
            "same seed must produce deterministic packed latent state");
    require(first.verify_self_description(a), "first forward self-description failed");
    require(second.verify_self_description(b), "second forward self-description failed");
}

void training_updates_shadow_and_reduces_error() {
    jarvisx::BitMatrix3DConfig config;
    config.learning_rate = 0.01F;
    config.ternary_threshold = 0.25F;
    config.seed = 42U;
    jarvisx::BitMatrix3DEngine engine(config);
    const auto input = fixture(config.input_edge);
    const auto before_weights = engine.encoder_shadow_weights();
    const auto before = engine.evaluate_inward(input).mse;
    for (std::size_t step = 0; step < 220U; ++step) {
        const auto metrics = engine.train_step(input);
        require(std::isfinite(metrics.mse), "training MSE must remain finite");
        require(std::isfinite(metrics.gradient_l2), "gradient norm must remain finite");
        require(metrics.self_description_valid,
                "packed self-description must remain valid during training");
    }
    const auto after = engine.evaluate_inward(input);
    require(engine.encoder_shadow_weights() != before_weights,
            "STE training must update continuous shadow weights");
    require(after.mse < before * 0.85F,
            "bit-matrix STE training must materially reduce fixture reconstruction error");
    require(after.packed_weight_bytes < after.shadow_weight_bytes,
            "packed ternary weight representation must be smaller than FP32 shadow weights");
}

void int8_activation_quantization_is_bounded() {
    const std::vector<float> values{-2.0F, -1.0F, 0.0F, 0.5F, 2.0F};
    const auto quantized = jarvisx::quantize_symmetric_int8(values);
    require(quantized.values.size() == values.size(),
            "INT8 quantized activation size mismatch");
    require(quantized.scale > 0.0F && std::isfinite(quantized.scale),
            "INT8 activation scale must be finite and positive");
    for (const auto q : quantized.values)
        require(q >= -127, "INT8 activation code entered the reserved -128 code");
}

} // namespace

int main() {
    try {
        ternary_round_trip();
        bitwise_dot_primitives();
        deterministic_forward();
        training_updates_shadow_and_reduces_error();
        int8_activation_quantization_is_bounded();
        std::cout << "DM-vOmegaXi+ bit-matrix 3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "bit-matrix 3D test failure: " << error.what() << '\n';
        return 1;
    }
}
