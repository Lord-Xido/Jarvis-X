#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

enum class Op : uint8_t {
    NOP=0x00, BOOT=0x01, CONFIG_FABRIC=0x02, OPEN_STREAM=0x03,
    NEXT_WINDOW=0x04, DECODE_MEDIA=0x05, ENCODE_MODAL=0x06,
    FUSE_3D=0x07, FOLD_INWARD=0x08, DECODE_MODAL=0x09,
    RESIDUAL=0x0A, UPDATE_OMEGA=0x0B, FIXPOINT_CHECK=0x0C,
    BRANCH_IF_NOT_CONVERGED=0x0D, EMIT=0x0E, CLOSE_WINDOW=0x0F,
    JUMP=0x10, HALT=0xFF
};

struct Insn {
    Op op;
    uint8_t dst, a, b;
    uint32_t imm;
};

static Insn decode(uint64_t w) {
    return {
        static_cast<Op>((w >> 56) & 0xff),
        static_cast<uint8_t>((w >> 48) & 0xff),
        static_cast<uint8_t>((w >> 40) & 0xff),
        static_cast<uint8_t>((w >> 32) & 0xff),
        static_cast<uint32_t>(w & 0xffffffffu)
    };
}

static int32_t simm(uint32_t x) { return static_cast<int32_t>(x); }

static std::vector<uint64_t> load_bc(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open bytecode: " + path);
    std::vector<uint64_t> words;
    while (true) {
        unsigned char b[8]{};
        f.read(reinterpret_cast<char*>(b), 8);
        if (f.gcount() == 0) break;
        if (f.gcount() != 8) throw std::runtime_error("truncated bytecode");
        uint64_t w = 0;
        for (unsigned char x : b) w = (w << 8) | x;
        words.push_back(w);
    }
    if (words.empty()) throw std::runtime_error("empty bytecode");
    return words;
}

struct Tensor {
    std::vector<float> x;
    void resize(size_t n, float v=0.f) { x.assign(n, v); }
    size_t size() const { return x.size(); }
};

class VM {
public:
    explicit VM(std::vector<uint64_t> bytecode) : code_(std::move(bytecode)) {}

    void run() {
        while (running_ && pc_ >= 0 && pc_ < static_cast<int64_t>(code_.size())) {
            const Insn i = decode(code_[pc_++]);
            execute(i);
        }
    }

private:
    std::array<double, 32> r_{};
    std::array<Tensor, 32> t_{};
    std::vector<uint64_t> code_;
    int64_t pc_ = 0;

    bool running_ = true;
    bool converged_ = false;
    uint32_t stream_mask_ = 0;
    uint32_t window_id_ = 0;
    uint32_t refine_iter_ = 0;
    uint32_t max_windows_ = 4;
    uint32_t max_refine_iter_ = 12;
    uint64_t logical_side_ = 0;
    uint64_t logical_positions_ = 0;
    size_t active_features_ = 32u * 32u * 16u;
    float residual_rms_ = 1.f;
    float best_residual_ = 1e9f;

    static float squash(float v) { return std::tanh(v); }

    void synthetic_media(Tensor& dst) {
        dst.resize(active_features_);
        const float phase = 0.173f * static_cast<float>(window_id_);
        for (size_t n = 0; n < dst.size(); ++n) {
            const float x = static_cast<float>(n % 257) / 257.f;
            dst.x[n] = 0.55f * std::sin(6.2831853f * (x + phase))
                     + 0.25f * std::cos(11.f * (x - phase))
                     + 0.20f * std::sin(0.031f * static_cast<float>(n));
        }
    }

    static void encode_modal(const Tensor& in, Tensor& out) {
        const size_t n = std::max<size_t>(256, in.size() / 4);
        out.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const size_t j = (i * 4) % in.size();
            float s = 0.f;
            for (size_t k = 0; k < 4; ++k) s += in.x[(j + k) % in.size()];
            out.x[i] = squash(0.25f * s);
        }
    }

    static void fuse3d(const Tensor& in, const Tensor& omega, Tensor& out) {
        out.resize(in.size());
        for (size_t i = 0; i < out.size(); ++i) {
            const float om = omega.size() ? omega.x[i % omega.size()] : 0.f;
            out.x[i] = squash(in.x[i] + 0.15f * om);
        }
    }

    static void fold3d(const Tensor& in, const Tensor& omega, Tensor& out, float lambda) {
        out.resize(in.size());
        if (!in.size()) return;
        for (size_t i = 0; i < in.size(); ++i) {
            const size_t im = (i + in.size() - 1) % in.size();
            const size_t ip = (i + 1) % in.size();
            const float lap = in.x[im] - 2.f * in.x[i] + in.x[ip];
            const float om = omega.size() ? omega.x[i % omega.size()] : 0.f;
            out.x[i] = squash(lambda * in.x[i] + 0.08f * lap + 0.12f * om);
        }
    }

    static void decode_modal(const Tensor& latent, Tensor& out, size_t target) {
        if (!latent.size()) throw std::runtime_error("decode from empty latent tensor");
        out.resize(target);
        for (size_t i = 0; i < target; ++i) {
            const double p = target <= 1 ? 0.0 :
                static_cast<double>(i) * static_cast<double>(latent.size() - 1) /
                static_cast<double>(target - 1);
            const size_t a = static_cast<size_t>(p);
            const size_t b = std::min(a + 1, latent.size() - 1);
            const float f = static_cast<float>(p - static_cast<double>(a));
            out.x[i] = (1.f - f) * latent.x[a] + f * latent.x[b];
        }
    }

    static float residual(const Tensor& a, const Tensor& b, Tensor& out) {
        const size_t n = std::min(a.size(), b.size());
        out.resize(n);
        double ss = 0.0;
        for (size_t i = 0; i < n; ++i) {
            out.x[i] = a.x[i] - b.x[i];
            ss += static_cast<double>(out.x[i]) * out.x[i];
        }
        return n ? static_cast<float>(std::sqrt(ss / static_cast<double>(n))) : 0.f;
    }

    static void update_omega(Tensor& omega, const Tensor& res, float beta) {
        if (omega.size() != res.size()) omega.resize(res.size(), 0.f);
        for (size_t i = 0; i < res.size(); ++i)
            omega.x[i] = beta * omega.x[i] + (1.f - beta) * res.x[i];
    }

    void execute(const Insn& i) {
        switch (i.op) {
            case Op::NOP:
                break;
            case Op::BOOT:
                std::cout << "[BOOT] DM3D recursive multimedia VM\n";
                break;
            case Op::CONFIG_FABRIC:
                logical_side_ = static_cast<uint64_t>(i.imm) * 1000ull;
                logical_positions_ = logical_side_ * logical_side_;
                r_[i.dst] = static_cast<double>(logical_positions_);
                std::cout << "[FABRIC] logical=" << logical_side_ << " x "
                          << logical_side_ << " = " << logical_positions_
                          << "; active=" << active_features_ << "\n";
                break;
            case Op::OPEN_STREAM:
                stream_mask_ = i.imm;
                std::cout << "[STREAM] mask=0x" << std::hex << stream_mask_ << std::dec << "\n";
                break;
            case Op::NEXT_WINDOW:
                if (window_id_ >= max_windows_) {
                    std::cout << "[EOS]\n";
                    running_ = false;
                    break;
                }
                ++window_id_;
                refine_iter_ = 0;
                converged_ = false;
                best_residual_ = 1e9f;
                std::cout << "[WINDOW " << window_id_ << "] slices=" << i.imm << "\n";
                break;
            case Op::DECODE_MEDIA:
                synthetic_media(t_[i.dst]);
                break;
            case Op::ENCODE_MODAL:
                encode_modal(t_[i.a], t_[i.dst]);
                break;
            case Op::FUSE_3D:
                fuse3d(t_[i.a], t_[i.b], t_[i.dst]);
                break;
            case Op::FOLD_INWARD: {
                const float lambda = static_cast<float>(i.imm) / 1'000'000.f;
                fold3d(t_[i.a], t_[i.b], t_[i.dst], lambda);
                t_[i.a] = t_[i.dst];
                ++refine_iter_;
                break;
            }
            case Op::DECODE_MODAL:
                decode_modal(t_[i.a], t_[i.dst], t_[3].size());
                break;
            case Op::RESIDUAL:
                residual_rms_ = residual(t_[i.a], t_[i.b], t_[i.dst]);
                break;
            case Op::UPDATE_OMEGA: {
                const float beta = static_cast<float>(i.imm) / 1'000'000.f;
                update_omega(t_[i.dst], t_[i.b], beta);
                break;
            }
            case Op::FIXPOINT_CHECK: {
                const float eps = static_cast<float>(i.imm) / 1'000'000.f;
                best_residual_ = std::min(best_residual_, residual_rms_);
                residual_rms_ = best_residual_;
                converged_ = residual_rms_ < eps;
                r_[i.dst] = converged_ ? 1.0 : 0.0;
                std::cout << "  [REFINE] k=" << std::setw(2) << refine_iter_
                          << " residual_rms=" << std::fixed << std::setprecision(6)
                          << residual_rms_ << (converged_ ? " CONVERGED" : "") << "\n";
                break;
            }
            case Op::BRANCH_IF_NOT_CONVERGED:
                if (!converged_ && refine_iter_ < max_refine_iter_) pc_ += simm(i.imm);
                break;
            case Op::EMIT:
                std::cout << "[EMIT] window=" << window_id_
                          << " latent=" << t_[i.b].size()
                          << " reconstructed=" << t_[i.a].size()
                          << " residual=" << residual_rms_ << "\n";
                break;
            case Op::CLOSE_WINDOW:
                for (size_t n = 2; n < t_.size(); ++n)
                    if (n != 9) t_[n].x.clear();
                break;
            case Op::JUMP:
                pc_ += simm(i.imm);
                break;
            case Op::HALT:
                running_ = false;
                break;
            default:
                throw std::runtime_error("unknown opcode");
        }
    }
};

int main(int argc, char** argv) {
    try {
        const std::string path = argc > 1 ? argv[1] : "dm3d_system.bc";
        VM vm(load_bc(path));
        vm.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return 1;
    }
}
