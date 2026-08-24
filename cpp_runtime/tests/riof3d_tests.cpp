#include "jarvisx/riof3d.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void deterministic_replay() {
    jarvisx::Riof3DConfig config;
    config.edge = 8U;
    config.seed = 42U;
    jarvisx::Riof3D first(config);
    jarvisx::Riof3D second(config);
    first.initialize("wave");
    second.initialize("wave");
    for (std::size_t i = 0; i < 12U; ++i) {
        first.step();
        second.step();
    }
    require(first.field() == second.field(), "same seed and state must replay bit-identically");
    require(first.momentum() == second.momentum(), "momentum replay must be deterministic");
}

void projection_and_finiteness() {
    jarvisx::Riof3DConfig config;
    config.edge = 8U;
    config.seed = 99U;
    jarvisx::Riof3D engine(config);
    engine.initialize("noise");
    for (std::size_t i = 0; i < 80U; ++i) {
        const auto metrics = engine.step();
        require(std::isfinite(metrics.total_energy), "RIOF energy must remain finite");
        require(std::isfinite(metrics.mean_abs_residual), "RIOF residual must remain finite");
        require(metrics.timestep >= config.min_timestep && metrics.timestep <= config.max_timestep,
                "intrinsic timestep must remain inside stability bounds");
        require(metrics.max_abs_value <= config.projection_limit + 1.0e-12,
                "Lambda projection must bound the volumetric field");
    }
}

void damped_objective_relaxes() {
    jarvisx::Riof3DConfig config;
    config.edge = 10U;
    config.seed = 7U;
    config.enhancement_gain = 0.0;
    config.refinement_gain = 0.0;
    config.memory_gain = 0.0;
    config.base_damping = 0.35;
    jarvisx::Riof3D engine(config);
    engine.initialize("wave");
    const double before = engine.metrics().total_energy;
    for (std::size_t i = 0; i < 160U; ++i) engine.step();
    const double after = engine.metrics().total_energy;
    require(after < before, "damped conservative RIOF must reduce total objective energy");
}

void intrinsic_controls_respond() {
    jarvisx::Riof3DConfig config;
    config.edge = 8U;
    config.seed = 123U;
    jarvisx::Riof3D engine(config);
    engine.initialize("noise");
    const auto metrics = engine.step();
    require(metrics.damping >= config.base_damping,
            "damping must be derived from volumetric disagreement");
    require(metrics.enhancement >= 0.0 &&
            metrics.enhancement <= config.enhancement_gain * 2.0 + 1.0e-12,
            "enhancement must be self-bounded by local disagreement");
}

} // namespace

int main() {
    try {
        deterministic_replay();
        projection_and_finiteness();
        damped_objective_relaxes();
        intrinsic_controls_respond();
        std::cout << "RIOF-3D tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "RIOF-3D test failure: " << error.what() << '\n';
        return 1;
    }
}
