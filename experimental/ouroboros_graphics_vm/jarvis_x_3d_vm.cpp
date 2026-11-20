// JARVIS X: 3D neural-field + Ouroboros graphics VM (standalone C++17)
// Reconstructed from the uploaded 2026-11-20 React artifact.
// This is an executable model of its explicit operations, NOT a trained ANN
// and NOT a physical 10^(6,000,000)-pixel allocation.
// Compile: g++ -std=c++17 -O2 -Wall -Wextra -pedantic jarvis_x_3d_vm.cpp -o jarvis_x_3d_vm
// Usage:   ./jarvis_x_3d_vm --program pulse --cycles 2400 --frames 120
// Tests:   ./jarvis_x_3d_vm --self-test

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jx {
constexpr int WIDTH = 128;
constexpr int HEIGHT = 128;
constexpr std::size_t PIXELS = WIDTH * HEIGHT;
constexpr double PI = 3.141592653589793238462643383279502884;
using I64 = std::int64_t;

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}
std::string upper(std::string s) {
    for (auto &c : s) if (c >= 'a' && c <= 'z') c = char(c - ('a' - 'A'));
    return s;
}
I64 parseInteger(std::string s) {
    if (!s.empty() && s.front() == '#') s.erase(s.begin());
    if (s.empty()) throw std::runtime_error("empty numeric operand");
    std::size_t consumed = 0;
    I64 v = 0;
    const bool hex = (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0);
    try { v = std::stoll(s, &consumed, hex ? 16 : 10); }
    catch (const std::exception&) { throw std::runtime_error("invalid integer: " + s); }
    if (consumed != s.size()) throw std::runtime_error("invalid integer: " + s);
    return v;
}
I64 checkedAdd(I64 a, I64 b) {
    constexpr auto hi = std::numeric_limits<I64>::max();
    constexpr auto lo = std::numeric_limits<I64>::min();
    if ((b > 0 && a > hi - b) || (b < 0 && a < lo - b))
        throw std::overflow_error("signed ADD overflow");
    return a + b;
}
I64 checkedMul(I64 a, I64 b) {
#if defined(__GNUC__) || defined(__clang__)
    I64 r{};
    if (__builtin_mul_overflow(a, b, &r))
        throw std::overflow_error("signed MUL overflow");
    return r;
#else
    // Portable overflow checking without causing signed overflow.
    const auto hi = std::numeric_limits<I64>::max();
    const auto lo = std::numeric_limits<I64>::min();
    if (a == 0 || b == 0) return 0;
    if ((a == -1 && b == lo) || (b == -1 && a == lo))
        throw std::overflow_error("signed MUL overflow");
    if (a > 0) {
        if (b > 0 ? a > hi / b : b < lo / a)
            throw std::overflow_error("signed MUL overflow");
    } else {
        if (b > 0 ? a < lo / b : a < hi / b)
            throw std::overflow_error("signed MUL overflow");
    }
    return a * b;
#endif
}
std::size_t parseSize(std::string s) {
    const I64 value = parseInteger(std::move(s));
    if (value < 0) throw std::runtime_error("negative size is not permitted");
    return static_cast<std::size_t>(value);
}
int parseSmallInt(std::string s) {
    const I64 value = parseInteger(std::move(s));
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        throw std::runtime_error("integer argument out of int range");
    return static_cast<int>(value);
}
int pixel8(I64 a) { return static_cast<int>(std::clamp<I64>(a, 0, 255)); }
int wrap128(I64 a) { return static_cast<int>((a % WIDTH + WIDTH) % WIDTH); }
double unitClamp(double a) { return std::clamp(a, 0.0, 1.0); }

enum class Activation { ReLU, Sigmoid, Tanh, GELU };
double activation(Activation a, double x) {
    switch (a) {
        case Activation::ReLU: return std::max(0.0, x);
        case Activation::Sigmoid: return 1.0 / (1.0 + std::exp(-2.0 * x));
        case Activation::Tanh: return std::tanh(x);
        case Activation::GELU:
            return 0.5 * x * (1.0 + std::tanh(std::sqrt(2.0 / PI) *
                              (x + 0.044715 * x * x * x)));
    }
    return 0.0;
}
Activation parseActivation(std::string s) {
    s = upper(s);
    if (s == "RELU") return Activation::ReLU;
    if (s == "SIGMOID") return Activation::Sigmoid;
    if (s == "TANH") return Activation::Tanh;
    if (s == "GELU") return Activation::GELU;
    throw std::runtime_error("activation must be relu/sigmoid/tanh/gelu");
}

struct Vec3 { double x = 0, y = 0, z = 0; };
struct Neuron { Vec3 p; int layer = 0; double phase = 0, boost = 0, act = 0; };
struct Edge { std::size_t src = 0, dst = 0; bool feedback = false; };
struct Packet { std::size_t edge = 0; double progress = 0, speed = 1; };
struct FieldStats { std::size_t active = 0, emitted = 0, alivePackets = 0; double mean = 0; };

class NeuralField {
public:
    NeuralField(std::size_t requestedNodes, int layers, int depth, Activation act,
                double feedbackGain, std::uint64_t seed)
        : layers_(layers), depth_(depth), kind_(act), gain_(feedbackGain), rng_(seed) {
        if (requestedNodes < 1 || requestedNodes > 20000 ||
            layers < 1 || layers > 8 || depth < 1 || depth > 8 ||
            feedbackGain < 0 || feedbackGain > 2)
            throw std::runtime_error("invalid neural-field configuration");
        build(requestedNodes);
    }
    const std::vector<Neuron>& nodes() const { return nodes_; }
    const std::vector<Edge>& edges() const { return edges_; }
    std::size_t nLayers() const { return static_cast<std::size_t>(layers_); }
    void boost(std::size_t idx, double amount) {
        if (idx < nodes_.size()) nodes_[idx].boost = unitClamp(nodes_[idx].boost + amount);
    }
    void boostForPixel(std::size_t pixel) {
        if (!nodes_.empty()) boost(pixel % nodes_.size(), 1.0);
    }
    void boostForTriangle(const std::array<I64,6>& coords) {
        for (std::size_t j = 0; j < coords.size(); ++j) {
            // Match the artifact's vertex-to-neuron excitation rule.
            const auto idx = static_cast<std::size_t>(std::llabs(coords[j] * 7 +
                                  static_cast<I64>(j) * 13)) % nodes_.size();
            boost(idx, 0.7);
        }
    }
    void feedback() {
        for (auto &n : nodes_) boostRef(n, 0.12 * gain_);
    }
    void randomizePhases() {
        for (auto &n : nodes_) { n.phase = uni_(rng_) * 2 * PI; n.boost = uni_(rng_) * 0.7; }
    }
    void pulse() { pendingPulse_ = true; }
    FieldStats tick(double dt, double speed = 1.0) {
        if (dt < 0 || speed < 0) throw std::runtime_error("negative time or speed");
        time_ += dt * speed;
        FieldStats stat{};
        for (auto &n : nodes_) {
            const double z = 1.15 * std::sin(1.8 * time_ - 0.85 * n.layer +
                              0.4 * n.phase) + 2.2 * n.boost +
                              (pendingPulse_ ? 0.6 * std::sin(5 * time_ - 1.4 * n.layer) : 0);
            double v = activation(kind_, z);
            if (kind_ == Activation::ReLU) v = std::min(1.2, v / 1.4);
            else if (kind_ == Activation::Sigmoid) v = (v - 0.15) * 1.3;
            else if (kind_ == Activation::Tanh) v = (v + 1.0) / 2.0;
            else v += 0.35;
            n.act = unitClamp(v);
            n.boost *= 0.985; // Per-frame decay, as in the React source.
            stat.active += (n.act > 0.58);
            stat.mean += n.act;
        }
        stat.mean /= static_cast<double>(nodes_.size());
        const double chance = 0.25 + std::min(0.6, static_cast<double>(edges_.size()) / 20000.0);
        if (!edges_.empty() && uni_(rng_) < chance) {
            spawnPacket(0.0, 0.9 + uni_(rng_) * 1.4); ++stat.emitted;
        }
        if (pendingPulse_ && !edges_.empty()) {
            for (int i = 0; i < 28; ++i) {
                spawnPacket(uni_(rng_) * 0.4, 1.8 + uni_(rng_)); ++stat.emitted;
            }
        }
        pendingPulse_ = false;
        std::vector<Packet> surviving;
        surviving.reserve(packets_.size());
        for (auto p : packets_) {
            p.progress += dt * speed * p.speed;
            if (p.progress >= 1.0) {
                const auto &edge = edges_[p.edge];
                boost(edge.dst, 0.45 * (edge.feedback ? gain_ : 1.0));
            } else {
                surviving.push_back(p);
            }
        }
        if (surviving.size() > 900)
            surviving.erase(surviving.begin(), surviving.end() - 900);
        packets_ = std::move(surviving);
        stat.alivePackets = packets_.size();
        return stat;
    }
private:
    void boostRef(Neuron &n, double b) { n.boost = unitClamp(n.boost + b); }
    void spawnPacket(double t, double speed) {
        auto i = static_cast<std::size_t>(uni_(rng_) * static_cast<double>(edges_.size()));
        i = std::min(i, edges_.size() - 1);
        packets_.push_back({i, t, speed});
    }
    void build(std::size_t requested) {
        const auto perLayer = (requested + static_cast<std::size_t>(layers_) - 1) /
                              static_cast<std::size_t>(layers_);
        const double scale = 1.0 - (depth_ - 1) * 0.055;
        const double core = 430.0 * scale;
        nodes_.reserve(perLayer * static_cast<std::size_t>(layers_));
        for (int layer = 0; layer < layers_; ++layer) {
            for (std::size_t j = 0; j < perLayer; ++j) {
                double theta = 2 * PI * layer / layers_ + (depth_ - 1) * 0.12 *
                              std::sin(2 * PI * static_cast<double>(j) / static_cast<double>(perLayer));
                double phi = 2 * PI * static_cast<double>(j) / static_cast<double>(perLayer);
                double tube = 285.0 * scale * (1.0 - 0.42 * layer / layers_);
                double radial = core + tube * std::cos(phi);
                nodes_.push_back({{radial * std::cos(theta),
                                   0.92 * radial * std::sin(theta),
                                   tube * std::sin(phi) + 28 * std::sin(theta * depth_)},
                                  layer, uni_(rng_) * 2 * PI, 0, 0});
            }
        }
        for (int layer = 0; layer < layers_ - 1; ++layer) {
            std::size_t base = static_cast<std::size_t>(layer) * perLayer;
            std::size_t next = base + perLayer;
            for (std::size_t j = 0; j < perLayer; ++j) {
                int degree = uni_(rng_) < 0.35 ? 2 : 1;
                for (int q = 0; q < degree; ++q) {
                    const int jitterRange = static_cast<int>(std::min<std::size_t>(24, perLayer));
                    const auto jitter = static_cast<int>(std::floor((uni_(rng_) - 0.5) * jitterRange));
                    const auto dst = (static_cast<std::int64_t>(j) + jitter +
                                      static_cast<std::int64_t>(perLayer)) %
                                     static_cast<std::int64_t>(perLayer);
                    edges_.push_back({base + j, next + static_cast<std::size_t>(dst), false});
                }
            }
        }
        auto last = static_cast<std::size_t>(layers_ - 1) * perLayer;
        for (std::size_t j = 0; j < perLayer; ++j) {
            edges_.push_back({last + j, j, true});
            if (j % 3 == 0) edges_.push_back({last + j, (j + 1) % perLayer, true});
        }
    }
    int layers_, depth_;
    Activation kind_;
    double gain_, time_ = 0;
    bool pendingPulse_ = false;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<double> uni_{0.0, 1.0};
    std::vector<Neuron> nodes_;
    std::vector<Edge> edges_;
    std::vector<Packet> packets_;
};

struct Instruction { std::string op; std::vector<std::string> args; std::size_t line = 0; };
struct Program {
    std::vector<Instruction> code;
    std::unordered_map<std::string, std::size_t> labels;
};

std::pair<int,int> argRange(const std::string& op) {
    if (op == "LOAD" || op == "ADD" || op == "MUL" || op == "MOV" || op == "SIN") return {2,2};
    if (op == "PUSH" || op == "POP" || op == "JMP" || op == "JZ" ||
        op == "CALL") return {1,1};
    if (op == "DRAW_PIXEL") return {2,3};
    if (op == "DRAW_TRI") return {6,7};
    if (op == "NOISE") return {0,1};
    if (op == "RET" || op == "CLEAR" || op == "FEEDBACK" ||
        op == "HALT" || op == "NOP") return {0,0};
    throw std::runtime_error("unrecognized opcode: " + op);
}
Program assemble(const std::string& source) {
    Program p;
    std::istringstream in(source);
    std::string line;
    std::size_t lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        line = trim(line.substr(0, line.find(';')));
        if (line.empty()) continue;
        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string label = upper(trim(line.substr(0, colon)));
            if (label.empty() || !((label[0]>='A'&&label[0]<='Z')||label[0]=='_'))
                throw std::runtime_error("invalid label on line " + std::to_string(lineNo));
            for (char c : label) if (!((c>='A'&&c<='Z') || (c>='0'&&c<='9') || c=='_'))
                throw std::runtime_error("invalid label on line " + std::to_string(lineNo));
            if (!p.labels.emplace(label, p.code.size()).second)
                throw std::runtime_error("duplicate label: " + label);
            line = trim(line.substr(colon + 1));
            if (line.empty()) continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream words(line);
        Instruction ins;
        ins.line = lineNo;
        words >> ins.op;
        ins.op = upper(ins.op);
        std::string token;
        while (words >> token) ins.args.push_back(upper(token));
        auto [minimum, maximum] = argRange(ins.op);
        if (int(ins.args.size()) < minimum || int(ins.args.size()) > maximum)
            throw std::runtime_error("wrong operand count for " + ins.op +
                                     " on line " + std::to_string(lineNo));
        if (ins.op == "LOAD" || ins.op == "ADD" || ins.op == "MUL" ||
            ins.op == "MOV" || ins.op == "SIN" || ins.op == "POP") {
            const auto &dest = ins.args[0];
            if (dest != "ACC" && !(dest.size() == 2 && dest[0] == 'R' &&
                   dest[1] >= '0' && dest[1] <= '7'))
                throw std::runtime_error("invalid destination " + dest);
        }
        p.code.push_back(std::move(ins));
    }
    if (p.code.empty()) throw std::runtime_error("assembly program is empty");
    for (const auto& ins : p.code) {
        for (auto& a : ins.args) {
            if (a == "ACC" || a == "PC" || a == "SP" ||
                (a.size() == 2 && a[0] == 'R' && a[1] >= '0' && a[1] <= '7') ||
                p.labels.count(a)) continue;
            parseInteger(a); // Reject undefined symbols instead of silently reading 0.
        }
    }
    return p;
}

std::string preset(const std::string& name) {
    if (name == "pulse") return R"ASM(
; ANN Pulse, copied semantically from the attached app
LOAD R7 #0
LOOP:
  LOAD R0 #64
  LOAD R1 #64
  LOAD R2 #255
  DRAW_PIXEL R0 R1 R2
  ADD R7 #1
  LOAD R0 R7
  DRAW_PIXEL R0 R1 R2
  FEEDBACK
  JMP LOOP
)ASM";
    if (name == "triangle") return R"ASM(
LOAD R7 #0
LOOP:
  SIN R0 R7
  LOAD R1 #18
  SIN R2 R7
  ADD R2 #60
  LOAD R3 #108
  LOAD R4 #22
  LOAD R5 #104
  LOAD R6 #255
  DRAW_TRI R0 R1 R2 R3 R4 R5 R6
  ADD R7 #3
  JMP LOOP
)ASM";
    if (name == "plasma") return R"ASM(
LOAD R7 #0
LOOP:
  NOISE #220
  FEEDBACK
  ADD R7 #1
  JMP LOOP
)ASM";
    if (name == "spiral") return R"ASM(
LOAD R7 #0
LOOP:
  SIN R0 R7
  LOAD R1 #24
  SIN R2 R7
  ADD R2 #48
  LOAD R3 #96
  LOAD R4 #30
  LOAD R5 #110
  LOAD R6 #180
  DRAW_TRI R0 R1 R2 R3 R4 R5 R6
  ADD R7 #5
  FEEDBACK
  JMP LOOP
)ASM";
    throw std::runtime_error("program must be pulse/triangle/plasma/spiral");
}

struct VM {
    std::size_t pc = 0, sp = 0, vramPtr = 0;
    I64 acc = 0;
    std::array<I64,8> regs{};
    std::vector<I64> stack;
    std::array<std::uint8_t,256> mem{};
    std::array<std::uint8_t, PIXELS> vram{};
    std::uint64_t cycles = 0, frame = 0;
    bool halted = false;
};

class GraphicsVM {
public:
    explicit GraphicsVM(Program p, NeuralField *field, double gain, std::uint64_t seed)
        : code_(std::move(p)), field_(field), gain_(gain), rng_(seed) {
        if (gain < 0 || gain > 2) throw std::runtime_error("gain must be in [0, 2]");
    }
    const VM& state() const { return vm_; }
    bool step() {
        if (vm_.halted) return false;
        if (vm_.pc >= code_.code.size()) { vm_.halted = true; return false; }
        const auto &in = code_.code[vm_.pc];
        std::size_t next = vm_.pc + 1;
        const auto &a = in.args;
        auto v = [this](const std::string& s) { return value(s); };
        const std::string &op = in.op;
        try {
            if (op == "LOAD" || op == "MOV") assign(a[0], v(a[1]));
            else if (op == "ADD") assign(a[0], checkedAdd(v(a[0]), v(a[1])));
            else if (op == "MUL") assign(a[0], checkedMul(v(a[0]), v(a[1])));
            else if (op == "PUSH") push(v(a[0]));
            else if (op == "POP") {
                if (vm_.stack.empty()) throw std::runtime_error("POP on empty stack");
                const I64 n = vm_.stack.back(); vm_.stack.pop_back();
                vm_.sp = vm_.stack.size(); assign(a[0], n);
            }
            else if (op == "JMP") next = jump(v(a[0]));
            else if (op == "JZ") { if (vm_.acc == 0) next = jump(v(a[0])); }
            else if (op == "CALL") {
                auto target = jump(v(a[0])); push(static_cast<I64>(next)); next = target;
            }
            else if (op == "RET") {
                if (vm_.stack.empty()) throw std::runtime_error("RET on empty stack");
                auto target = jump(vm_.stack.back());
                vm_.stack.pop_back(); vm_.sp = vm_.stack.size(); next = target;
            }
            else if (op == "DRAW_PIXEL") {
                const int x = wrap128(v(a[0])), y = wrap128(v(a[1]));
                const auto index = static_cast<std::size_t>(y * WIDTH + x);
                vm_.vram[index] = static_cast<std::uint8_t>(a.size() > 2 ? pixel8(v(a[2])) : 255);
                vm_.vramPtr = index;
                if (field_) field_->boostForPixel(index);
            }
            else if (op == "DRAW_TRI") {
                const I64 intensity = a.size() > 6 ? v(a[6]) : 255;
                drawTriangle({v(a[0]),v(a[1]),v(a[2]),v(a[3]),v(a[4]),v(a[5])},
                             pixel8(intensity));
            }
            else if (op == "CLEAR") vm_.vram.fill(0);
            else if (op == "NOISE") {
                const I64 n = a.empty() ? 220 : v(a[0]);
                if (n < 0 || n > 1000000) throw std::runtime_error("NOISE count out of range");
                std::uniform_int_distribution<int> pos(0, static_cast<int>(PIXELS) - 1);
                std::uniform_int_distribution<int> byte(0, 255);
                for (I64 i = 0; i < n; ++i) {
                    const std::size_t index = static_cast<std::size_t>(pos(rng_));
                    vm_.vram[index] = static_cast<std::uint8_t>(byte(rng_));
                    vm_.vramPtr = index;
                }
            }
            else if (op == "SIN") {
                const double s = std::sin(static_cast<double>(v(a[1])) * 0.08) * 60 + 64;
                if (!std::isfinite(s)) throw std::runtime_error("SIN nonfinite value");
                assign(a[0], static_cast<I64>(std::llround(s)));
            }
            else if (op == "FEEDBACK") {
                const double factor = 0.92 + 0.08 * gain_;
                for (auto &p : vm_.vram)
                    p = static_cast<std::uint8_t>(pixel8(std::llround(p * factor)));
                if (field_) field_->feedback();
            }
            else if (op == "HALT") vm_.halted = true;
            else if (op != "NOP") throw std::runtime_error("bad opcode");
        } catch (const std::exception& ex) {
            vm_.halted = true;
            throw std::runtime_error("VM line " + std::to_string(in.line) + " (" + op + "): " + ex.what());
        }
        vm_.pc = next;
        ++vm_.cycles;
        vm_.mem[vm_.pc % vm_.mem.size()] = static_cast<std::uint8_t>(vm_.acc);
        if (vm_.cycles % 48 == 0) ++vm_.frame;
        if (vm_.pc >= code_.code.size()) vm_.halted = true;
        return true;
    }
    std::size_t run(std::size_t budget) {
        std::size_t executed = 0;
        while (executed < budget && step()) ++executed;
        return executed;
    }
private:
    I64 value(const std::string& s) const {
        if (s == "ACC") return vm_.acc;
        if (s == "PC") return static_cast<I64>(vm_.pc);
        if (s == "SP") return static_cast<I64>(vm_.sp);
        if (s.size() == 2 && s[0] == 'R' && s[1] >= '0' && s[1] <= '7')
            return vm_.regs[static_cast<std::size_t>(s[1] - '0')];
        const auto it = code_.labels.find(s);
        if (it != code_.labels.end()) return static_cast<I64>(it->second);
        return parseInteger(s);
    }
    void assign(const std::string& dest, I64 number) {
        if (dest.size() == 2 && dest[0] == 'R' && dest[1] >= '0' && dest[1] <= '7')
            vm_.regs[static_cast<std::size_t>(dest[1] - '0')] = number;
        // Original _t() always mirrors the assigned result into ACC.
        vm_.acc = number;
    }
    void push(I64 n) {
        if (vm_.stack.size() >= 65536) throw std::runtime_error("stack capacity exceeded");
        vm_.stack.push_back(n); vm_.sp = vm_.stack.size();
    }
    std::size_t jump(I64 target) const {
        if (target < 0 || static_cast<std::uint64_t>(target) > code_.code.size())
            throw std::runtime_error("jump target out of bounds");
        return static_cast<std::size_t>(target);
    }
    void drawTriangle(std::array<I64,6> p, int brightness) {
        // Bound extreme operands so all intermediate integer arithmetic is safe.
        for (I64 x : p)
            if (x < -1000000 || x > 1000000)
                throw std::runtime_error("triangle coordinate exceeds +/-1,000,000");
        // Pixel coordinates are integer and triangles are rasterized in XY.
        const auto [x0,y0,x1,y1,x2,y2] = p;
        const long double area = (static_cast<long double>(y1-y2) * (x0-x2) +
                                  static_cast<long double>(x2-x1) * (y0-y2));
        if (area == 0) return;
        const int left   = static_cast<int>(std::clamp<I64>(std::min({x0,x1,x2}),0,WIDTH-1));
        const int right  = static_cast<int>(std::clamp<I64>(std::max({x0,x1,x2}),0,WIDTH-1));
        const int top    = static_cast<int>(std::clamp<I64>(std::min({y0,y1,y2}),0,HEIGHT-1));
        const int bottom = static_cast<int>(std::clamp<I64>(std::max({y0,y1,y2}),0,HEIGHT-1));
        for (int y=top; y<=bottom; ++y) for (int x=left; x<=right; ++x) {
            const long double a = (static_cast<long double>(y1-y2) * (x-x2) +
                                   static_cast<long double>(x2-x1) * (y-y2))/area;
            const long double b = (static_cast<long double>(y2-y0) * (x-x2) +
                                   static_cast<long double>(x0-x2) * (y-y2))/area;
            const long double c = 1 - a - b;
            if (a >= -0.01L && b >= -0.01L && c >= -0.01L) {
                const auto index = static_cast<std::size_t>(y*WIDTH + x);
                vm_.vram[index] = static_cast<std::uint8_t>(brightness);
                vm_.vramPtr = index;
            }
        }
        if (field_) field_->boostForTriangle(p);
    }
    Program code_;
    NeuralField *field_;
    double gain_;
    std::mt19937_64 rng_;
    VM vm_{};
};

std::uint64_t hashFramebuffer(const VM& vm) {
    // Correct FNV-1a 64-bit offset basis and multiplier.
    std::uint64_t h = UINT64_C(14695981039346656037);
    for (std::uint8_t b : vm.vram) { h ^= b; h *= UINT64_C(1099511628211); }
    return h;
}
double frameDeltaMSE(const VM& state, const std::array<std::uint8_t,PIXELS>& prev) {
    long double ss = 0;
    for (std::size_t k=0; k<PIXELS; ++k) {
        const auto d = int(state.vram[k]) - int(prev[k]); ss += d*d;
    }
    return static_cast<double>(ss / (PIXELS * 255.0L * 255.0L));
}
void savePGM(const std::string& path, const VM& vm) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "P5\n" << WIDTH << ' ' << HEIGHT << "\n255\n";
    out.write(reinterpret_cast<const char*>(vm.vram.data()),
              static_cast<std::streamsize>(vm.vram.size()));
    if (!out) throw std::runtime_error("error writing " + path);
}
void savePLY(const std::string& path, const NeuralField& field) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "ply\nformat ascii 1.0\ncomment 3D field from JARVIS X Ouroboros VM\n"
        << "element vertex " << field.nodes().size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "element edge " << field.edges().size() << "\n"
        << "property int vertex1\nproperty int vertex2\nend_header\n";
    out << std::fixed << std::setprecision(5);
    for (const auto &n : field.nodes()) {
        const int r = pixel8(static_cast<I64>(std::llround(n.act*255)));
        const int b = pixel8(static_cast<I64>(std::llround((1-n.act)*255)));
        out << n.p.x << ' ' << n.p.y << ' ' << n.p.z << ' '
            << r << " 100 " << b << '\n';
    }
    for (const auto &e : field.edges()) out << e.src << ' ' << e.dst << '\n';
    if (!out) throw std::runtime_error("error writing " + path);
}

void require(bool ok, const std::string& msg) { if (!ok) throw std::runtime_error("TEST FAILED: " + msg); }
void selfTest() {
    {
        GraphicsVM vm(assemble("LOAD R0 #7\nADD R0 #3\nMUL R0 #2\nHALT\n"),nullptr,1,12);
        require(vm.run(20)==4,"exact four instructions executed");
        require(vm.state().regs[0]==20 && vm.state().acc==20,"arithmetic + ACC mirror");
        require(vm.state().halted && vm.state().cycles==4,"HALT/cycle count");
    }
    {
        GraphicsVM vm(assemble("LOAD R0 #0\nJZ SKIP\nLOAD R1 #99\nSKIP: CALL WORK\nHALT\nWORK: LOAD R2 #42\nRET\n"),nullptr,1,14);
        vm.run(100);
        require(vm.state().regs[1]==0 && vm.state().regs[2]==42,"branch+call+return");
        require(vm.state().stack.empty() && vm.state().sp==0,"stack balanced");
    }
    {
        GraphicsVM vm(assemble("DRAW_PIXEL #-1 #-1 #255\nHALT\n"),nullptr,1,15);
        vm.run(10);
        require(vm.state().vram[PIXELS-1]==255,"toroidal pixel wrap");
    }
    {
        GraphicsVM vm(assemble("DRAW_TRI #10 #10 #60 #10 #10 #60 #200\nHALT\n"),nullptr,1,16);
        vm.run(10);
        require(vm.state().vram[20*WIDTH+20]==200,"filled barycentric triangle");
    }
    for (int gain=0; gain<=2; ++gain) {
        GraphicsVM vm(assemble("DRAW_PIXEL #8 #9 #100\nFEEDBACK\nHALT\n"),nullptr,gain,17);
        vm.run(10);
        require(vm.state().vram[9*WIDTH+8]==(gain==0?92:gain==1?100:108),
                "feedback gain scaling");
    }
    {
        const auto p=assemble("NOISE #100\nHALT\n");
        GraphicsVM a(p,nullptr,1,18),b(p,nullptr,1,18);
        a.run(10); b.run(10);
        require(a.state().vram==b.state().vram,"seeded NOISE repeatability");
    }
    {
        bool rejected=false;
        try { (void)assemble("BADD R0 #2\n"); } catch (...) { rejected=true; }
        require(rejected,"invalid opcode rejected");
        rejected=false;
        try { (void)assemble("JMP UNKNOWN\n"); } catch (...) { rejected=true; }
        require(rejected,"unresolved target rejected");
    }
    {
        NeuralField field(500,5,3,Activation::GELU,1.0,19);
        require(field.nodes().size()==500,"3D nodes allocated exactly for divisible density");
        require(field.edges().size()>=500,"feedforward and feedback topology exists");
        std::size_t returning=0;
        for (const auto&e:field.edges()) {
            require(e.src<field.nodes().size() && e.dst<field.nodes().size(),"edge indices valid");
            returning+=e.feedback;
        }
        require(returning>0,"Ouroboros return edges");
        field.pulse(); auto s=field.tick(1.0/60.0);
        require(s.emitted>=28 && s.active<=500,"pulse launches packets");
        for (const auto& n:field.nodes()) {
            require(std::isfinite(n.p.x)&&std::isfinite(n.p.y)&&std::isfinite(n.p.z),"3D finite");
            require(n.act>=0 && n.act<=1,"activation bounded");
        }
    }
    {
        GraphicsVM vm(assemble("JMP LOOP\nLOOP: JMP LOOP\n"),nullptr,1,20);
        require(vm.run(48)==48 && vm.state().frame==1,"frame per 48 VM instructions");
    }
    std::cout << "PASS: 11 self-test groups (assembler, VM, raster, feedback, 3D, determinism)\n";
}

std::string readText(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read assembly source " + path);
    std::ostringstream s; s << f.rdbuf(); return s.str();
}

} // namespace jx

int main(int argc, char** argv) {
    using namespace jx;
    try {
        std::string programName="pulse", asmPath, output="jarvis_x";
        std::size_t cycles=2400, frames=120, density=6500;
        int layers=5, depth=3;
        double gain=1.0, speed=1.0;
        std::uint64_t seed=0x4A41525649535AULL;
        Activation kind=Activation::GELU;
        bool selftest=false;
        for (int i=1; i<argc; ++i) {
            const std::string flag=argv[i];
            if (flag=="--self-test") { selftest=true; continue; }
            if (flag=="--help") {
                std::cout << "Options: --self-test | --program pulse|triangle|plasma|spiral "
                    "| --asm FILE | --cycles N | --frames N | --density N | "
                    "--layers 3|5|8 | --depth 1..8 | --gain 0..2 | "
                    "--activation gelu|relu|sigmoid|tanh | --speed N | --seed N | --out PREFIX\n";
                return 0;
            }
            if (i+1>=argc) throw std::runtime_error("missing value for " + flag);
            const std::string val=argv[++i];
            if (flag=="--program") programName=val;
            else if (flag=="--asm") asmPath=val;
            else if (flag=="--out") output=val;
            else if (flag=="--cycles") cycles=parseSize(val);
            else if (flag=="--frames") frames=parseSize(val);
            else if (flag=="--density") density=parseSize(val);
            else if (flag=="--layers") layers=parseSmallInt(val);
            else if (flag=="--depth") depth=parseSmallInt(val);
            else if (flag=="--gain") gain=std::stod(val);
            else if (flag=="--speed") speed=std::stod(val);
            else if (flag=="--seed") seed=static_cast<std::uint64_t>(parseSize(val));
            else if (flag=="--activation") kind=parseActivation(val);
            else throw std::runtime_error("unknown option: " + flag);
        }
        if (selftest) { selfTest(); return 0; }
        if (frames < 1 || frames > 100000 || cycles > 100000000 || density < 1 || density > 20000 ||
            (layers!=3 && layers!=5 && layers!=8) || !std::isfinite(gain) ||
            !std::isfinite(speed) || speed < 0 || speed > 100)
            throw std::runtime_error("runtime arguments exceed supported bounds");
        NeuralField field(density,layers,depth,kind,gain,seed);
        GraphicsVM vm(assemble(asmPath.empty()?preset(programName):readText(asmPath)),
                      &field,gain,seed+1);
        std::ofstream csv(output + "_telemetry.csv");
        if (!csv) throw std::runtime_error("cannot create telemetry CSV");
        csv << "display_frame,vm_cycles,vm_frame,pc,active_neurons,mean_activation,"
               "emitted_packets,live_packets,frame_delta_mse,framebuffer_hash\n";
        std::array<std::uint8_t,PIXELS> prev{};
        std::size_t consumed=0, peakActive=0;
        FieldStats last{};
        for (std::size_t f=0; f<frames; ++f) {
            // Instruction budget spread across requested display frames.
            const std::size_t target=static_cast<std::size_t>(
                (static_cast<long double>(f+1)*cycles)/frames);
            const std::size_t allowance = target-consumed;
            consumed+=vm.run(allowance);
            last=field.tick(1.0/60.0,speed);
            peakActive=std::max(peakActive,last.active);
            const double mse=frameDeltaMSE(vm.state(),prev);
            prev=vm.state().vram;
            csv << f << ',' << vm.state().cycles << ',' << vm.state().frame << ','
                << vm.state().pc << ',' << last.active << ',' << std::setprecision(9)
                << last.mean << ',' << last.emitted << ',' << last.alivePackets << ','
                << mse << ',' << hashFramebuffer(vm.state()) << '\n';
        }
        csv.close();
        savePGM(output+"_framebuffer.pgm",vm.state());
        savePLY(output+"_neural_field.ply",field);
        std::cout << "JARVIS X 3D graphics VM / CPU reference\n"
                  << "Geometry: " << field.nodes().size() << " nodes, "
                  << field.edges().size() << " edges, " << field.nLayers() << " layers\n"
                  << "VM: " << vm.state().cycles << " instructions, "
                  << vm.state().frame << " VM frames, PC " << vm.state().pc
                  << ", halted=" << (vm.state().halted?"true":"false") << "\n"
                  << "Neuron activity: " << last.active << " active at last tick, peak "
                  << peakActive << "; mean=" << std::fixed << std::setprecision(4)
                  << last.mean << "\n"
                  << "Framebuffer: 128x128 (16384 bytes), FNV1a64=0x" << std::hex
                  << hashFramebuffer(vm.state()) << std::dec << "\n"
                  << "Logical display target: (10^6)^(10^6) = 10^6000000 "
                     "(NOT stored in memory)\n"
                  << "Output: " << output << "_framebuffer.pgm, " << output
                  << "_neural_field.ply, " << output << "_telemetry.csv\n";
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << '\n'; return 1;
    }
}
