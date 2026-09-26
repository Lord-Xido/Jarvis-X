#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jarvisx::inrvm {

constexpr std::size_t MiB = 1024ull * 1024ull;

struct MemoryPlan {
    static constexpr std::size_t process_limit = 1024ull * MiB;
    static constexpr std::size_t weights_limit = 384ull * MiB;
    static constexpr std::size_t working_limit = 352ull * MiB;
    static constexpr std::size_t omega_limit = 192ull * MiB;
    static constexpr std::size_t vm_limit = 32ull * MiB;
    static constexpr std::size_t managed_limit = weights_limit + working_limit + omega_limit + vm_limit;
    static constexpr std::size_t host_headroom = process_limit - managed_limit;
};

static_assert(MemoryPlan::managed_limit == 960ull * MiB, "managed arena must be 960 MiB");
static_assert(MemoryPlan::host_headroom == 64ull * MiB, "64 MiB host headroom required");
static_assert(MemoryPlan::managed_limit < MemoryPlan::process_limit, "managed memory must fit under process ceiling");

enum class Arena : std::uint8_t { Weights, Working, Omega, Vm };

class BudgetTracker {
public:
    void claim(Arena arena, std::size_t bytes) {
        const auto idx = static_cast<std::size_t>(arena);
        const auto cap = capacity(arena);
        if (bytes > cap - used_[idx]) {
            throw std::runtime_error("streaming INR VM arena budget exceeded");
        }
        if (bytes > MemoryPlan::managed_limit - total_used_) {
            throw std::runtime_error("streaming INR VM managed memory budget exceeded");
        }
        used_[idx] += bytes;
        total_used_ += bytes;
    }

    [[nodiscard]] std::size_t used(Arena arena) const noexcept {
        return used_[static_cast<std::size_t>(arena)];
    }

    [[nodiscard]] std::size_t total_used() const noexcept { return total_used_; }

    [[nodiscard]] static constexpr std::size_t capacity(Arena arena) noexcept {
        switch (arena) {
            case Arena::Weights: return MemoryPlan::weights_limit;
            case Arena::Working: return MemoryPlan::working_limit;
            case Arena::Omega: return MemoryPlan::omega_limit;
            case Arena::Vm: return MemoryPlan::vm_limit;
        }
        return 0;
    }

private:
    std::array<std::size_t, 4> used_{};
    std::size_t total_used_ = 0;
};

enum class Opcode : std::uint8_t {
    LD_COORD = 0x01,
    FF_TRANS = 0x02,
    SIN_ACT = 0x03,
    MAT_MUL = 0x04,
    HOLO_FUSE = 0x05,
    BND_CHECK = 0x06,
    RET_VAL = 0x07,
    HALT = 0xFF,
};

struct Instruction {
    Opcode opcode{};
    std::uint8_t dst = 0;
    std::uint8_t src = 0;
    std::uint8_t aux = 0;
    std::uint8_t reserved = 0;
    std::uint32_t imm = 0;
    float scalar = 0.0f;
};

static_assert(sizeof(Instruction) <= 16, "instruction should remain compact");

struct Query3D {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double t = 0.0;
};

struct VmConfig {
    std::uint32_t virtual_axis = 8'000'000u;
    std::size_t feature_width = 32;
    std::size_t fourier_width = 32;
    std::size_t omega_slots = 64;
    float entropy_cutoff = 0.05f;
    float omega_mix = 0.15f;
};

struct VmResult {
    bool emitted = false;
    bool skipped = false;
    float entropy = 0.0f;
    std::vector<float> feature;
};

class DenseMatrix {
public:
    DenseMatrix() = default;

    DenseMatrix(std::size_t rows, std::size_t cols, float phase)
        : rows_(rows), cols_(cols), data_(rows * cols), bias_(rows) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("matrix dimensions must be non-zero");
        }
        for (std::size_t r = 0; r < rows_; ++r) {
            bias_[r] = 0.01f * std::sin(phase + static_cast<float>(r) * 0.17f);
            for (std::size_t c = 0; c < cols_; ++c) {
                const float u = phase + static_cast<float>((r + 1) * (c + 3));
                data_[r * cols_ + c] = 0.125f * std::sin(u * 0.013f);
            }
        }
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        return (data_.size() + bias_.size()) * sizeof(float);
    }

    [[nodiscard]] std::vector<float> multiply(const std::vector<float>& x) const {
        if (x.size() != cols_) {
            throw std::runtime_error("MAT_MUL source width mismatch");
        }
        std::vector<float> y(rows_);
        for (std::size_t r = 0; r < rows_; ++r) {
            float sum = bias_[r];
            const auto base = r * cols_;
            for (std::size_t c = 0; c < cols_; ++c) {
                sum += data_[base + c] * x[c];
            }
            y[r] = sum;
        }
        return y;
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<float> data_;
    std::vector<float> bias_;
};

class StreamingInrVm {
public:
    explicit StreamingInrVm(VmConfig config = {})
        : cfg_(config),
          fourier_(cfg_.fourier_width, 3, 0.3f),
          layer1_(cfg_.feature_width, cfg_.fourier_width, 1.1f),
          layer2_(cfg_.feature_width, cfg_.feature_width, 2.7f),
          omega_(cfg_.omega_slots * cfg_.feature_width, 0.0f) {
        if (cfg_.virtual_axis == 0 || cfg_.feature_width == 0 || cfg_.fourier_width == 0 || cfg_.omega_slots == 0) {
            throw std::invalid_argument("invalid streaming INR VM configuration");
        }
        if (!(cfg_.omega_mix >= 0.0f && cfg_.omega_mix <= 1.0f)) {
            throw std::invalid_argument("omega_mix must be in [0,1]");
        }

        budget_.claim(Arena::Weights, fourier_.bytes() + layer1_.bytes() + layer2_.bytes());
        budget_.claim(Arena::Omega, omega_.size() * sizeof(float));
        budget_.claim(Arena::Working, 8ull * cfg_.feature_width * sizeof(float) + cfg_.fourier_width * sizeof(float));
        budget_.claim(Arena::Vm, default_program().size() * sizeof(Instruction));
    }

    [[nodiscard]] const BudgetTracker& budget() const noexcept { return budget_; }
    [[nodiscard]] const VmConfig& config() const noexcept { return cfg_; }

    [[nodiscard]] static std::vector<Instruction> default_program(float cutoff = 0.05f) {
        return {
            {Opcode::LD_COORD, 1, 0, 0, 0, 0, 0.0f},
            {Opcode::FF_TRANS, 2, 1, 0, 0, 0, 0.0f},
            {Opcode::SIN_ACT, 2, 2, 0, 0, 0, 0.0f},
            {Opcode::MAT_MUL, 3, 2, 0, 0, 0, 0.0f},
            {Opcode::SIN_ACT, 3, 3, 0, 0, 0, 0.0f},
            {Opcode::MAT_MUL, 4, 3, 0, 0, 1, 0.0f},
            {Opcode::HOLO_FUSE, 5, 4, 0, 0, 0, 0.0f},
            {Opcode::BND_CHECK, 5, 5, 0, 0, 0, cutoff},
            {Opcode::RET_VAL, 0, 5, 0, 0, 0, 0.0f},
            {Opcode::HALT, 0, 0, 0, 0, 0, 0.0f},
        };
    }

    [[nodiscard]] VmResult execute(const Query3D& query, const std::vector<Instruction>& program = {}) {
        const auto code = program.empty() ? default_program(cfg_.entropy_cutoff) : program;
        std::array<std::vector<float>, 8> regs;
        VmResult result;
        bool gated = false;

        for (std::size_t ip = 0; ip < code.size(); ++ip) {
            const auto& ins = code[ip];
            if (ins.dst >= regs.size() || ins.src >= regs.size()) {
                throw std::runtime_error("register index out of range");
            }
            switch (ins.opcode) {
                case Opcode::LD_COORD:
                    regs[ins.dst] = normalized_coord(query);
                    break;
                case Opcode::FF_TRANS:
                    regs[ins.dst] = fourier_.multiply(regs[ins.src]);
                    break;
                case Opcode::SIN_ACT:
                    regs[ins.dst] = regs[ins.src];
                    for (auto& v : regs[ins.dst]) v = std::sin(v);
                    break;
                case Opcode::MAT_MUL:
                    if (ins.imm == 0) regs[ins.dst] = layer1_.multiply(regs[ins.src]);
                    else if (ins.imm == 1) regs[ins.dst] = layer2_.multiply(regs[ins.src]);
                    else throw std::runtime_error("unknown weight matrix id");
                    break;
                case Opcode::HOLO_FUSE:
                    regs[ins.dst] = fuse_omega(regs[ins.src]);
                    break;
                case Opcode::BND_CHECK:
                    result.entropy = information_entropy(regs[ins.src]);
                    gated = result.entropy < ins.scalar;
                    result.skipped = gated;
                    break;
                case Opcode::RET_VAL:
                    if (!gated) {
                        result.feature = regs[ins.src];
                        result.emitted = true;
                    }
                    break;
                case Opcode::HALT:
                    return result;
            }
        }
        return result;
    }

    [[nodiscard]] std::vector<VmResult> execute_batch(const std::vector<Query3D>& queries) {
        std::vector<VmResult> out;
        out.reserve(queries.size());
        for (const auto& q : queries) out.push_back(execute(q));
        return out;
    }

private:
    [[nodiscard]] std::vector<float> normalized_coord(const Query3D& q) const {
        const double axis = static_cast<double>(cfg_.virtual_axis);
        const auto norm = [axis](double v) {
            if (!std::isfinite(v)) throw std::runtime_error("non-finite query coordinate");
            return static_cast<float>(std::clamp(v / axis, 0.0, 1.0));
        };
        return {norm(q.x), norm(q.y), norm(q.z)};
    }

    [[nodiscard]] std::vector<float> fuse_omega(const std::vector<float>& x) {
        if (x.size() != cfg_.feature_width) {
            throw std::runtime_error("HOLO_FUSE feature width mismatch");
        }
        std::vector<float> y(x.size());
        const std::size_t base = omega_head_ * cfg_.feature_width;
        for (std::size_t i = 0; i < x.size(); ++i) {
            const float history = omega_[base + i];
            y[i] = (1.0f - cfg_.omega_mix) * x[i] + cfg_.omega_mix * history;
            omega_[base + i] = x[i];
        }
        omega_head_ = (omega_head_ + 1) % cfg_.omega_slots;
        return y;
    }

    [[nodiscard]] static float information_entropy(const std::vector<float>& x) {
        if (x.empty()) return 0.0f;
        double sum = 0.0;
        for (float v : x) sum += std::abs(static_cast<double>(v));
        if (sum <= std::numeric_limits<double>::epsilon()) return 0.0f;
        double h = 0.0;
        for (float v : x) {
            const double p = std::abs(static_cast<double>(v)) / sum;
            if (p > 0.0) h -= p * std::log(p);
        }
        const double hmax = std::log(static_cast<double>(x.size()));
        return hmax > 0.0 ? static_cast<float>(h / hmax) : 0.0f;
    }

    VmConfig cfg_;
    BudgetTracker budget_;
    DenseMatrix fourier_;
    DenseMatrix layer1_;
    DenseMatrix layer2_;
    std::vector<float> omega_;
    std::size_t omega_head_ = 0;
};

inline const char* opcode_name(Opcode op) noexcept {
    switch (op) {
        case Opcode::LD_COORD: return "LD_COORD";
        case Opcode::FF_TRANS: return "FF_TRANS";
        case Opcode::SIN_ACT: return "SIN_ACT";
        case Opcode::MAT_MUL: return "MAT_MUL";
        case Opcode::HOLO_FUSE: return "HOLO_FUSE";
        case Opcode::BND_CHECK: return "BND_CHECK";
        case Opcode::RET_VAL: return "RET_VAL";
        case Opcode::HALT: return "HALT";
    }
    return "UNKNOWN";
}

} // namespace jarvisx::inrvm
