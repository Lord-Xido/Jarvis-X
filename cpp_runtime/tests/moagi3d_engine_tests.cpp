#include "jarvisx/moagi3d_engine.hpp"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

jarvisx::Moagi3DConfig base_config() {
    return jarvisx::Moagi3DConfig{
        8U,
        3U,
        3U,
        8U,
        1.0e-5F,
        0.5F,
        8.0F,
        42U,
        false
    };
}

void deterministic_replay() {
    const auto config = base_config();
    const auto input = jarvisx::make_volume(8U, "sphere", 42U);
    const jarvisx::Moagi3DEngine first(config);
    const jarvisx::Moagi3DEngine second(config);
    const auto a = first.run(input);
    const auto b = second.run(input);

    require(a.output.values() == b.output.values(),
            "same Moagi-3D config must replay exactly");
    require(a.iterations == b.iterations,
            "same Moagi-3D config must use the same iteration count");
    require(a.fusion_weights == b.fusion_weights,
            "same Moagi-3D config must reproduce fusion weights");
}

void zero_is_fixed_point() {
    auto config = base_config();
    config.max_iterations = 4U;
    const jarvisx::Tensor4D zero({1U, 8U, 8U, 8U}, 0.0F);
    const jarvisx::Moagi3DEngine engine(config);
    const auto result = engine.run(zero);

    require(result.converged, "zero volume must be a fixed point");
    require(result.iterations == 1U,
            "RAC must halt immediately when the first update is stationary");
    require(result.final_delta_l2 <= config.epsilon,
            "fixed point delta must satisfy epsilon threshold");
}

void fusion_weights_are_probabilities() {
    const auto config = base_config();
    const auto input = jarvisx::make_volume(8U, "wave", 7U);
    const jarvisx::Moagi3DEngine engine(config);
    const auto result = engine.run(input);

    require(result.fusion_weights.size() == config.channels,
            "FRM must emit one alpha weight per channel");
    const float sum = std::accumulate(result.fusion_weights.begin(),
                                      result.fusion_weights.end(), 0.0F);
    require(std::fabs(sum - 1.0F) < 1.0e-5F,
            "FRM softmax weights must sum to one");
    for (const float weight : result.fusion_weights) {
        require(std::isfinite(weight) && weight >= 0.0F && weight <= 1.0F,
                "FRM softmax weights must be finite probabilities");
    }
}

void rac_enforces_hard_iteration_bound() {
    auto config = base_config();
    config.max_iterations = 3U;
    config.epsilon = 1.0e-12F;
    const auto input = jarvisx::make_volume(8U, "checker", 9U);
    const jarvisx::Moagi3DEngine engine(config);
    const auto result = engine.run(input);

    require(result.iterations <= config.max_iterations,
            "RAC must never exceed the configured iteration ceiling");
    require(result.history.size() == result.iterations,
            "RAC telemetry must contain exactly one record per executed iteration");
    require(std::isfinite(result.final_delta_l2),
            "RAC final delta must remain finite");
    require(std::isfinite(result.final_anchor_l2),
            "anchor-drift telemetry must remain finite");
}

void invalid_configuration_fails_closed() {
    auto config = base_config();
    config.channels = 0U;
    bool rejected = false;
    try {
        const jarvisx::Moagi3DEngine engine(config);
        (void)engine;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid Moagi-3D configuration must fail closed");
}

} // namespace

int main() {
    try {
        deterministic_replay();
        zero_is_fixed_point();
        fusion_weights_are_probabilities();
        rac_enforces_hard_iteration_bound();
        invalid_configuration_fails_closed();
        std::cout << "Moagi-3D engine tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Moagi-3D engine test failure: " << error.what() << '\n';
        return 1;
    }
}
