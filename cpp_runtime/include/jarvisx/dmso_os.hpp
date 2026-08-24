#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace jarvisx::dmso {

struct Instruction {
    std::uint8_t opcode{};
    std::uint16_t alpha{};
    std::uint16_t beta{};
};

class BinaryBytecodeCompiler {
public:
    static constexpr std::uint8_t RECURSE_SPACE = 0x80U;
    static constexpr std::uint8_t EVAL_FIELD_GEO = 0x90U;
    static constexpr std::uint8_t EVAL_FIELD_TEX = 0xA0U;

    static std::uint64_t compile_instruction(
        const std::uint8_t opcode,
        const std::uint16_t alpha,
        const std::uint16_t beta
    ) noexcept {
        return (static_cast<std::uint64_t>(opcode) << 56U)
            | (static_cast<std::uint64_t>(alpha) << 40U)
            | (static_cast<std::uint64_t>(beta) << 24U);
    }

    static Instruction decompile_raw_chunk(const std::uint64_t word) noexcept {
        Instruction result{};
        result.opcode = static_cast<std::uint8_t>((word >> 56U) & 0xFFU);
        result.alpha = static_cast<std::uint16_t>((word >> 40U) & 0xFFFFU);
        result.beta = static_cast<std::uint16_t>((word >> 24U) & 0xFFFFU);
        return result;
    }

    static bool is_supported(const std::uint8_t opcode) noexcept {
        return opcode == RECURSE_SPACE || opcode == EVAL_FIELD_GEO || opcode == EVAL_FIELD_TEX;
    }
};

struct VirtualVRAMRegisters {
    explicit VirtualVRAMRegisters(const std::size_t resolution)
        : res(validate_resolution(resolution)),
          reg_f(res * res * 3U, 0.0),
          z_buf(res * res, 10.0) {}

    std::size_t res;
    std::size_t reg_pc{0U};
    std::vector<double> reg_f;
    std::vector<double> z_buf;
    std::array<double, 3U> theta_g{{0.55, 4.0, 0.25}};
    std::array<double, 3U> theta_t{{0.60, 2.5, 0.10}};

private:
    static std::size_t validate_resolution(const std::size_t value) {
        if (value == 0U || value > 512U) {
            throw std::invalid_argument("resolution must be in [1, 512]");
        }
        return value;
    }
};

struct OptimizationStep {
    double baseline_loss{0.0};
    double candidate_loss{0.0};
    double committed_loss{0.0};
    bool accepted{false};
};

struct LifecycleReport {
    std::size_t epochs_executed{0U};
    std::size_t accepted_updates{0U};
    std::size_t rejected_updates{0U};
    double initial_loss{0.0};
    double final_loss{0.0};
    double elapsed_seconds{0.0};
    bool converged{false};
    std::array<double, 3U> theta_g{};
    std::array<double, 3U> theta_t{};
};

class ExecutionEngineKernel {
public:
    explicit ExecutionEngineKernel(VirtualVRAMRegisters& registers)
        : vram_(registers),
          x_coords_(registers.res * registers.res),
          y_coords_(registers.res * registers.res) {
        const double denom = registers.res > 1U ? static_cast<double>(registers.res - 1U) : 1.0;
        for (std::size_t row = 0U; row < registers.res; ++row) {
            for (std::size_t col = 0U; col < registers.res; ++col) {
                const std::size_t index = row * registers.res + col;
                x_coords_[index] = -1.0 + (2.0 * static_cast<double>(col) / denom);
                y_coords_[index] = -1.0 + (2.0 * static_cast<double>(row) / denom);
            }
        }
    }

    const std::vector<double>& dispatch_forward_pass(const std::vector<std::uint64_t>& program) {
        validate_program(program);
        std::fill(vram_.reg_f.begin(), vram_.reg_f.end(), 0.0);
        std::fill(vram_.z_buf.begin(), vram_.z_buf.end(), 10.0);
        vram_.reg_pc = 0U;
        double spatial_scale = 1.0;

        while (vram_.reg_pc < program.size()) {
            const Instruction instr = BinaryBytecodeCompiler::decompile_raw_chunk(program[vram_.reg_pc]);
            switch (instr.opcode) {
            case BinaryBytecodeCompiler::RECURSE_SPACE:
                spatial_scale = (static_cast<double>(instr.alpha) / 65535.0)
                    * static_cast<double>(instr.beta);
                break;
            case BinaryBytecodeCompiler::EVAL_FIELD_GEO:
                execute_geometry(spatial_scale);
                break;
            case BinaryBytecodeCompiler::EVAL_FIELD_TEX:
                execute_shading(spatial_scale);
                break;
            default:
                throw std::invalid_argument("unsupported opcode");
            }
            ++vram_.reg_pc;
        }
        return vram_.reg_f;
    }

    OptimizationStep dispatch_backward_optimization(
        const std::vector<std::uint64_t>& program,
        const std::vector<double>& target_pixels,
        const double learning_rate = 0.12
    ) {
        validate_target(target_pixels);
        if (!std::isfinite(learning_rate) || learning_rate < 0.0 || learning_rate > 1.0) {
            throw std::invalid_argument("learning_rate must be finite and in [0, 1]");
        }

        const auto original_g = vram_.theta_g;
        const auto original_t = vram_.theta_t;
        const double base_loss = mse(dispatch_forward_pass(program), target_pixels);
        constexpr double eps = 1.0e-5;
        std::array<double, 3U> grad_g{};
        std::array<double, 3U> grad_t{};

        for (std::size_t i = 0U; i < vram_.theta_g.size(); ++i) {
            const double original = vram_.theta_g[i];
            vram_.theta_g[i] = original + eps;
            const double high = mse(dispatch_forward_pass(program), target_pixels);
            vram_.theta_g[i] = original - eps;
            const double low = mse(dispatch_forward_pass(program), target_pixels);
            vram_.theta_g[i] = original;
            grad_g[i] = (high - low) / (2.0 * eps);
        }

        for (std::size_t i = 0U; i < vram_.theta_t.size(); ++i) {
            const double original = vram_.theta_t[i];
            vram_.theta_t[i] = original + eps;
            const double high = mse(dispatch_forward_pass(program), target_pixels);
            vram_.theta_t[i] = original - eps;
            const double low = mse(dispatch_forward_pass(program), target_pixels);
            vram_.theta_t[i] = original;
            grad_t[i] = (high - low) / (2.0 * eps);
        }

        for (std::size_t i = 0U; i < vram_.theta_g.size(); ++i) {
            vram_.theta_g[i] -= learning_rate * std::clamp(grad_g[i], -1.0, 1.0);
            vram_.theta_t[i] -= learning_rate * std::clamp(grad_t[i], -1.0, 1.0);
        }
        sanitize_parameters();

        const double candidate_loss = mse(dispatch_forward_pass(program), target_pixels);
        const bool accepted = std::isfinite(candidate_loss) && candidate_loss <= base_loss;
        if (!accepted) {
            vram_.theta_g = original_g;
            vram_.theta_t = original_t;
            dispatch_forward_pass(program);
        }
        return OptimizationStep{
            base_loss,
            candidate_loss,
            accepted ? candidate_loss : base_loss,
            accepted,
        };
    }

    static double mse(const std::vector<double>& lhs, const std::vector<double>& rhs) {
        if (lhs.size() != rhs.size() || lhs.empty()) {
            throw std::invalid_argument("mse vectors must be non-empty and equal length");
        }
        double total = 0.0;
        for (std::size_t i = 0U; i < lhs.size(); ++i) {
            const double diff = lhs[i] - rhs[i];
            total += diff * diff;
        }
        return total / static_cast<double>(lhs.size());
    }

private:
    VirtualVRAMRegisters& vram_;
    std::vector<double> x_coords_;
    std::vector<double> y_coords_;

    double query_geometry_field(const double x, const double y, const double z) const noexcept {
        const double r = std::sqrt((x * x) + (y * y) + (z * z));
        const double base = r - vram_.theta_g[0U];
        const double freq = vram_.theta_g[1U];
        const double amplitude = vram_.theta_g[2U];
        const double noise = std::sin(x * freq) * std::cos(y * freq)
            * std::sin(z * freq) * amplitude;
        return base + noise;
    }

    std::array<double, 3U> query_shading_field(
        const double x,
        const double y,
        const double z
    ) const noexcept {
        const double r = vram_.theta_t[0U] + 0.3 * std::sin(x * vram_.theta_t[1U]);
        const double g = 0.5 * (std::cos((y * vram_.theta_t[1U]) + vram_.theta_t[2U]) + 1.0);
        const double b = (0.4 * std::sin(z + 2.0)) + 0.5;
        return {{
            std::clamp(r, 0.0, 1.0),
            std::clamp(g, 0.0, 1.0),
            std::clamp(b, 0.0, 1.0),
        }};
    }

    void execute_geometry(const double spatial_scale) {
        constexpr double ray_origin_z = -2.0;
        constexpr std::size_t depth_steps = 16U;
        constexpr double step = 0.25;
        for (std::size_t depth_index = 0U; depth_index < depth_steps; ++depth_index) {
            const double z = ray_origin_z + (static_cast<double>(depth_index) * step);
            for (std::size_t pixel = 0U; pixel < vram_.z_buf.size(); ++pixel) {
                const double x = x_coords_[pixel] * spatial_scale;
                const double y = y_coords_[pixel] * spatial_scale;
                const double distance = query_geometry_field(x, y, z);
                if (distance <= 0.0 && z < vram_.z_buf[pixel]) {
                    vram_.z_buf[pixel] = z;
                }
            }
        }
    }

    void execute_shading(const double spatial_scale) {
        for (std::size_t pixel = 0U; pixel < vram_.z_buf.size(); ++pixel) {
            const double z = vram_.z_buf[pixel];
            if (z >= 10.0) {
                continue;
            }
            const double x = x_coords_[pixel] * spatial_scale;
            const double y = y_coords_[pixel] * spatial_scale;
            const auto colour = query_shading_field(x, y, z);
            const std::size_t base = pixel * 3U;
            vram_.reg_f[base] = colour[0U];
            vram_.reg_f[base + 1U] = colour[1U];
            vram_.reg_f[base + 2U] = colour[2U];
        }
    }

    static void validate_program(const std::vector<std::uint64_t>& program) {
        if (program.empty() || program.size() > 1024U) {
            throw std::invalid_argument("program must contain between 1 and 1024 instructions");
        }
        for (const auto word : program) {
            const auto instr = BinaryBytecodeCompiler::decompile_raw_chunk(word);
            if (!BinaryBytecodeCompiler::is_supported(instr.opcode)) {
                throw std::invalid_argument("program contains unsupported opcode");
            }
        }
    }

    void validate_target(const std::vector<double>& target) const {
        if (target.size() != vram_.reg_f.size()) {
            throw std::invalid_argument("target buffer size mismatch");
        }
        for (const double value : target) {
            if (!std::isfinite(value) || value < 0.0 || value > 1.0) {
                throw std::invalid_argument("target values must be finite and in [0, 1]");
            }
        }
    }

    void sanitize_parameters() {
        for (double& value : vram_.theta_g) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite geometry parameter");
            }
            value = std::clamp(value, -8.0, 8.0);
        }
        for (double& value : vram_.theta_t) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite shading parameter");
            }
            value = std::clamp(value, -8.0, 8.0);
        }
    }
};

class AutonomousSystemOS {
public:
    explicit AutonomousSystemOS(const std::size_t surface_resolution = 32U)
        : registers_(surface_resolution),
          kernel_(registers_),
          target_(surface_resolution * surface_resolution * 3U, 0.0) {
        program_ = {
            BinaryBytecodeCompiler::compile_instruction(
                BinaryBytecodeCompiler::RECURSE_SPACE,
                45000U,
                1U
            ),
            BinaryBytecodeCompiler::compile_instruction(
                BinaryBytecodeCompiler::EVAL_FIELD_GEO,
                0U,
                0U
            ),
            BinaryBytecodeCompiler::compile_instruction(
                BinaryBytecodeCompiler::EVAL_FIELD_TEX,
                0U,
                0U
            ),
        };
        seed_default_target();
    }

    LifecycleReport execute_system_lifecycle(
        const std::size_t operational_epochs = 20U,
        const double learning_rate = 0.15,
        const double convergence_threshold = 1.0e-6
    ) {
        const std::lock_guard<std::mutex> lifecycle_guard(lifecycle_mutex_);
        if (operational_epochs == 0U || operational_epochs > 10000U) {
            throw std::invalid_argument("operational_epochs must be in [1, 10000]");
        }
        if (!std::isfinite(convergence_threshold) || convergence_threshold < 0.0) {
            throw std::invalid_argument("convergence_threshold must be finite and non-negative");
        }

        const auto started = std::chrono::steady_clock::now();
        LifecycleReport report{};
        double loss = ExecutionEngineKernel::mse(kernel_.dispatch_forward_pass(program_), target_);
        report.initial_loss = loss;

        for (std::size_t epoch = 0U; epoch < operational_epochs; ++epoch) {
            const OptimizationStep step = kernel_.dispatch_backward_optimization(
                program_,
                target_,
                learning_rate
            );
            report.epochs_executed = epoch + 1U;
            if (step.accepted) {
                ++report.accepted_updates;
            } else {
                ++report.rejected_updates;
            }
            loss = step.committed_loss;
            if (loss <= convergence_threshold) {
                report.converged = true;
                break;
            }
        }

        const auto ended = std::chrono::steady_clock::now();
        report.elapsed_seconds = std::chrono::duration<double>(ended - started).count();
        report.final_loss = loss;
        report.theta_g = registers_.theta_g;
        report.theta_t = registers_.theta_t;
        return report;
    }

    std::future<LifecycleReport> execute_system_lifecycle_async(
        const std::size_t operational_epochs = 20U,
        const double learning_rate = 0.15,
        const double convergence_threshold = 1.0e-6
    ) {
        return std::async(
            std::launch::async,
            [this, operational_epochs, learning_rate, convergence_threshold]() {
                return execute_system_lifecycle(
                    operational_epochs,
                    learning_rate,
                    convergence_threshold
                );
            }
        );
    }

    const VirtualVRAMRegisters& registers() const noexcept { return registers_; }
    const std::vector<std::uint64_t>& program() const noexcept { return program_; }
    const std::vector<double>& target() const noexcept { return target_; }

private:
    VirtualVRAMRegisters registers_;
    ExecutionEngineKernel kernel_;
    std::vector<std::uint64_t> program_;
    std::vector<double> target_;
    std::mutex lifecycle_mutex_;

    void seed_default_target() {
        const std::size_t side = registers_.res;
        const std::size_t margin = side / 5U;
        const std::size_t end = side - margin;
        for (std::size_t row = margin; row < end; ++row) {
            for (std::size_t col = margin; col < end; ++col) {
                const std::size_t base = ((row * side) + col) * 3U;
                target_[base] = 0.90;
                target_[base + 1U] = 0.15;
                target_[base + 2U] = 0.35;
            }
        }
    }
};

} // namespace jarvisx::dmso
