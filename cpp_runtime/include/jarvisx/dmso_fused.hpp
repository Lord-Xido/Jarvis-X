#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace jarvisx::dmso {

enum class Opcode : std::uint8_t {
    load_self,
    aggregate_26,
    decode_front,
    load_input,
    affine,
    tanh,
    relax,
};

inline constexpr std::array<Opcode, 7> canonical_program{
    Opcode::load_self,
    Opcode::aggregate_26,
    Opcode::decode_front,
    Opcode::load_input,
    Opcode::affine,
    Opcode::tanh,
    Opcode::relax,
};

struct Parameters {
    double self_gain{0.75};
    double neighbour_gain{0.20};
    double projection_gain{0.05};
    double input_gain{0.50};
    double bias{0.0};
};

struct Context {
    std::vector<double> current;
    std::vector<double> neighbour_mean;
    std::vector<double> projected;
    std::vector<double> stimulus;
    double alpha{0.25};
};

struct ExecutionResult {
    std::vector<double> value;
    std::uint64_t dispatches{0};
};

inline void validate(const Context& context) {
    const auto channels = context.current.size();
    if (channels == 0 || context.neighbour_mean.size() != channels ||
        context.projected.size() != channels || context.stimulus.size() != channels) {
        throw std::invalid_argument("DMSO context vectors must have one equal non-zero size");
    }
    if (!std::isfinite(context.alpha) || context.alpha <= 0.0 || context.alpha > 1.0) {
        throw std::invalid_argument("DMSO alpha must be finite and in (0, 1]");
    }
    const auto check = [](const std::vector<double>& values) {
        for (const double value : values) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("DMSO context contains a non-finite value");
            }
        }
    };
    check(context.current);
    check(context.neighbour_mean);
    check(context.projected);
    check(context.stimulus);
}

inline ExecutionResult execute_primitive(const Context& context, const Parameters& parameters) {
    validate(context);
    const auto channels = context.current.size();
    std::vector<double> current(channels, 0.0);
    std::vector<double> neighbours(channels, 0.0);
    std::vector<double> projected(channels, 0.0);
    std::vector<double> stimulus(channels, 0.0);
    std::vector<double> affine(channels, 0.0);
    std::vector<double> mapped(channels, 0.0);
    std::vector<double> output = context.current;
    std::uint64_t dispatches = 0;

    for (const Opcode opcode : canonical_program) {
        ++dispatches;
        switch (opcode) {
            case Opcode::load_self:
                current = context.current;
                break;
            case Opcode::aggregate_26:
                neighbours = context.neighbour_mean;
                break;
            case Opcode::decode_front:
                projected = context.projected;
                break;
            case Opcode::load_input:
                stimulus = context.stimulus;
                break;
            case Opcode::affine:
                for (std::size_t channel = 0; channel < channels; ++channel) {
                    affine[channel] = parameters.self_gain * current[channel] +
                                      parameters.neighbour_gain * neighbours[channel] +
                                      parameters.projection_gain * projected[channel] +
                                      parameters.input_gain * stimulus[channel] + parameters.bias;
                }
                break;
            case Opcode::tanh:
                for (std::size_t channel = 0; channel < channels; ++channel) {
                    mapped[channel] = std::tanh(affine[channel]);
                }
                break;
            case Opcode::relax:
                for (std::size_t channel = 0; channel < channels; ++channel) {
                    output[channel] = current[channel] +
                                      context.alpha * (mapped[channel] - current[channel]);
                }
                break;
        }
    }
    return ExecutionResult{output, dispatches};
}

inline ExecutionResult execute_fused(const Context& context, const Parameters& parameters) {
    validate(context);
    const auto channels = context.current.size();
    std::vector<double> output(channels, 0.0);
    for (std::size_t channel = 0; channel < channels; ++channel) {
        const double activation = parameters.self_gain * context.current[channel] +
                                  parameters.neighbour_gain * context.neighbour_mean[channel] +
                                  parameters.projection_gain * context.projected[channel] +
                                  parameters.input_gain * context.stimulus[channel] + parameters.bias;
        const double mapped = std::tanh(activation);
        output[channel] = context.current[channel] +
                          context.alpha * (mapped - context.current[channel]);
    }
    return ExecutionResult{output, 1};
}

inline double max_abs_error(const ExecutionResult& left, const ExecutionResult& right) {
    if (left.value.size() != right.value.size()) {
        throw std::invalid_argument("DMSO result channel counts differ");
    }
    double error = 0.0;
    for (std::size_t channel = 0; channel < left.value.size(); ++channel) {
        error = std::max(error, std::abs(left.value[channel] - right.value[channel]));
    }
    return error;
}

}  // namespace jarvisx::dmso
