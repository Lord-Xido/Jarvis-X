#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jarvisx::intelligence3d {

constexpr std::uint64_t kDecimalGB = 1000ULL * 1000ULL * 1000ULL;
constexpr std::uint64_t kDefaultAxisExtent = 1000ULL * kDecimalGB;
constexpr std::uint64_t kDefaultResidentLimit = 10ULL * kDecimalGB;

struct Coord3 {
    std::uint64_t x{};
    std::uint64_t y{};
    std::uint64_t z{};
};

struct PageKey {
    std::uint64_t x{};
    std::uint64_t y{};
    std::uint64_t z{};

    bool operator==(const PageKey& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
};

inline std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

struct PageKeyHash {
    std::size_t operator()(const PageKey& key) const noexcept {
        const auto a = mix64(key.x + 0x9e3779b97f4a7c15ULL);
        const auto b = mix64(key.y + 0x243f6a8885a308d3ULL);
        const auto c = mix64(key.z + 0x13198a2e03707344ULL);
        return static_cast<std::size_t>(a ^ (b << 1U) ^ (c >> 1U));
    }
};

struct VolumeStats {
    std::uint64_t reads{};
    std::uint64_t writes{};
    std::uint64_t page_faults{};
    std::uint64_t evictions{};
    std::uint64_t disk_loads{};
    std::uint64_t disk_stores{};
};

class VirtualVolume3D {
public:
    struct Config {
        std::uint64_t axis_extent{kDefaultAxisExtent};
        std::uint32_t page_edge{32U};
        std::uint64_t resident_limit_bytes{kDefaultResidentLimit};
        std::filesystem::path page_directory{"jarvisx-3d-state/pages"};
    };

    explicit VirtualVolume3D(Config config) : config_(std::move(config)) {
        if (config_.axis_extent == 0U) {
            throw std::invalid_argument("axis_extent must be positive");
        }
        if (config_.page_edge == 0U || config_.page_edge > 256U) {
            throw std::invalid_argument("page_edge must be in [1, 256]");
        }
        const auto edge = static_cast<std::uint64_t>(config_.page_edge);
        page_bytes_ = edge * edge * edge;
        if (config_.resident_limit_bytes < page_bytes_) {
            throw std::invalid_argument("resident_limit_bytes must hold at least one page");
        }
        if (config_.page_directory.empty()) {
            throw std::invalid_argument("page_directory must not be empty");
        }
        std::filesystem::create_directories(config_.page_directory);
    }

    ~VirtualVolume3D() {
        try {
            flush();
        } catch (...) {
            // Destructors must not throw. Explicit flush() reports I/O failures.
        }
    }

    VirtualVolume3D(const VirtualVolume3D&) = delete;
    VirtualVolume3D& operator=(const VirtualVolume3D&) = delete;

    std::uint8_t read(Coord3 coord) {
        validate(coord);
        ++stats_.reads;
        const auto key = key_for(coord);
        auto* page = find_page(key);
        if (page == nullptr) {
            const auto file = page_path(key);
            if (!std::filesystem::exists(file)) {
                return 0U;
            }
            page = &materialize(key, true);
        }
        page->last_touch = ++clock_;
        return page->bytes[offset_for(coord)];
    }

    void write(Coord3 coord, std::uint8_t value) {
        validate(coord);
        ++stats_.writes;
        const auto key = key_for(coord);
        auto* page = find_page(key);
        if (page == nullptr) {
            page = &materialize(key, std::filesystem::exists(page_path(key)));
        }
        page->bytes[offset_for(coord)] = value;
        page->dirty = true;
        page->last_touch = ++clock_;
    }

    void flush() {
        for (auto& entry : resident_) {
            store_if_dirty(entry.first, *entry.second);
        }
    }

    std::uint64_t axis_extent() const noexcept { return config_.axis_extent; }
    std::uint64_t page_bytes() const noexcept { return page_bytes_; }
    std::uint64_t resident_bytes() const noexcept {
        return static_cast<std::uint64_t>(resident_.size()) * page_bytes_;
    }
    std::uint64_t resident_limit_bytes() const noexcept { return config_.resident_limit_bytes; }
    std::size_t resident_pages() const noexcept { return resident_.size(); }
    const VolumeStats& stats() const noexcept { return stats_; }

    std::string conceptual_capacity() const {
        const long double axis = static_cast<long double>(config_.axis_extent);
        const long double volume = axis * axis * axis;
        std::ostringstream out;
        out << std::scientific << std::setprecision(3) << volume << " bytes";
        return out.str();
    }

private:
    struct Page {
        explicit Page(std::size_t byte_count) : bytes(byte_count, 0U) {}
        std::vector<std::uint8_t> bytes;
        bool dirty{false};
        std::uint64_t last_touch{};
    };

    Config config_;
    std::uint64_t page_bytes_{};
    std::uint64_t clock_{};
    VolumeStats stats_{};
    std::unordered_map<PageKey, std::unique_ptr<Page>, PageKeyHash> resident_;

    void validate(Coord3 coord) const {
        if (coord.x >= config_.axis_extent || coord.y >= config_.axis_extent ||
            coord.z >= config_.axis_extent) {
            throw std::out_of_range("3D coordinate lies outside virtual volume");
        }
    }

    PageKey key_for(Coord3 coord) const noexcept {
        const auto edge = static_cast<std::uint64_t>(config_.page_edge);
        return {coord.x / edge, coord.y / edge, coord.z / edge};
    }

    std::size_t offset_for(Coord3 coord) const noexcept {
        const auto edge = static_cast<std::uint64_t>(config_.page_edge);
        const auto lx = coord.x % edge;
        const auto ly = coord.y % edge;
        const auto lz = coord.z % edge;
        return static_cast<std::size_t>((lz * edge + ly) * edge + lx);
    }

    std::filesystem::path page_path(const PageKey& key) const {
        const auto shard = static_cast<unsigned>(mix64(key.x ^ key.y ^ key.z) & 0xffU);
        std::ostringstream shard_name;
        shard_name << std::hex << std::setw(2) << std::setfill('0') << shard;
        const auto dir = config_.page_directory / shard_name.str();
        std::ostringstream file_name;
        file_name << key.x << '_' << key.y << '_' << key.z << ".jxpg";
        return dir / file_name.str();
    }

    Page* find_page(const PageKey& key) noexcept {
        const auto it = resident_.find(key);
        return it == resident_.end() ? nullptr : it->second.get();
    }

    void evict_one() {
        if (resident_.empty()) {
            throw std::runtime_error("resident cache cannot evict from an empty set");
        }
        auto victim = resident_.begin();
        for (auto it = resident_.begin(); it != resident_.end(); ++it) {
            if (it->second->last_touch < victim->second->last_touch) {
                victim = it;
            }
        }
        store_if_dirty(victim->first, *victim->second);
        resident_.erase(victim);
        ++stats_.evictions;
    }

    Page& materialize(const PageKey& key, bool load_existing) {
        while (resident_bytes() + page_bytes_ > config_.resident_limit_bytes) {
            evict_one();
        }
        auto page = std::make_unique<Page>(static_cast<std::size_t>(page_bytes_));
        ++stats_.page_faults;
        if (load_existing) {
            const auto file = page_path(key);
            std::ifstream input(file, std::ios::binary);
            if (!input) {
                throw std::runtime_error("failed to open spilled page for reading: " + file.string());
            }
            input.read(reinterpret_cast<char*>(page->bytes.data()),
                       static_cast<std::streamsize>(page->bytes.size()));
            if (input.gcount() != static_cast<std::streamsize>(page->bytes.size())) {
                throw std::runtime_error("spilled page has invalid length: " + file.string());
            }
            ++stats_.disk_loads;
        }
        page->last_touch = ++clock_;
        auto [it, inserted] = resident_.emplace(key, std::move(page));
        if (!inserted) {
            throw std::runtime_error("internal error: duplicate resident page");
        }
        return *it->second;
    }

    void store_if_dirty(const PageKey& key, Page& page) {
        if (!page.dirty) {
            return;
        }
        const auto file = page_path(key);
        std::filesystem::create_directories(file.parent_path());
        const auto temporary = file.string() + ".tmp";
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) {
                throw std::runtime_error("failed to open page for writing: " + temporary);
            }
            output.write(reinterpret_cast<const char*>(page.bytes.data()),
                         static_cast<std::streamsize>(page.bytes.size()));
            if (!output) {
                throw std::runtime_error("failed to spill page: " + temporary);
            }
        }
        std::error_code ec;
        std::filesystem::remove(file, ec);
        ec.clear();
        std::filesystem::rename(temporary, file, ec);
        if (ec) {
            throw std::runtime_error("failed to commit page spill: " + ec.message());
        }
        page.dirty = false;
        ++stats_.disk_stores;
    }
};

class OctreeMask3D {
public:
    OctreeMask3D(std::uint64_t axis_extent, std::uint32_t depth)
        : axis_extent_(axis_extent), depth_(depth) {
        if (axis_extent_ == 0U) {
            throw std::invalid_argument("axis_extent must be positive");
        }
        if (depth_ > 48U) {
            throw std::invalid_argument("octree depth must be <= 48");
        }
    }

    double evaluate(Coord3 coord) const {
        if (coord.x >= axis_extent_ || coord.y >= axis_extent_ || coord.z >= axis_extent_) {
            throw std::out_of_range("mask coordinate lies outside virtual volume");
        }
        std::uint64_t x_lo = 0U, y_lo = 0U, z_lo = 0U;
        std::uint64_t x_hi = axis_extent_, y_hi = axis_extent_, z_hi = axis_extent_;
        for (std::uint32_t level = 0U; level < depth_; ++level) {
            const auto x_mid = x_lo + (x_hi - x_lo) / 2U;
            const auto y_mid = y_lo + (y_hi - y_lo) / 2U;
            const auto z_mid = z_lo + (z_hi - z_lo) / 2U;
            const int dx = coord.x >= x_mid ? 1 : 0;
            const int dy = coord.y >= y_mid ? 1 : 0;
            const int dz = coord.z >= z_mid ? 1 : 0;
            if ((dx + dy + dz) >= 2) {
                return 0.0;
            }
            if (dx != 0) x_lo = x_mid; else x_hi = x_mid;
            if (dy != 0) y_lo = y_mid; else y_hi = y_mid;
            if (dz != 0) z_lo = z_mid; else z_hi = z_mid;
        }
        return 1.0;
    }

private:
    std::uint64_t axis_extent_;
    std::uint32_t depth_;
};

struct PsiTrace {
    Coord3 coord{};
    double mask{};
    std::vector<double> encoded;
    std::vector<double> fused;
    std::uint8_t decoded{};
    double prediction_error{};
};

class PsiIntelligenceCore {
public:
    struct Config {
        std::size_t latent_dim{32U};
        std::uint32_t octree_depth{8U};
        double learning_rate{0.01};
        double momentum{0.95};
    };

    PsiIntelligenceCore(VirtualVolume3D& volume, Config config)
        : volume_(volume), config_(std::move(config)),
          mask_(volume.axis_extent(), config_.octree_depth), memory_(config_.latent_dim, 0.0) {
        if (config_.latent_dim == 0U || config_.latent_dim > 4096U) {
            throw std::invalid_argument("latent_dim must be in [1, 4096]");
        }
        if (!(config_.learning_rate > 0.0 && config_.learning_rate <= 1.0)) {
            throw std::invalid_argument("learning_rate must be in (0, 1]");
        }
        if (!(config_.momentum >= 0.0 && config_.momentum < 1.0)) {
            throw std::invalid_argument("momentum must be in [0, 1)");
        }
    }

    PsiTrace infer(Coord3 coord, bool learn) {
        PsiTrace trace;
        trace.coord = coord;
        trace.encoded = encode(coord);
        trace.mask = mask_.evaluate(coord);
        for (auto& value : trace.encoded) {
            value *= trace.mask;
        }
        trace.fused.resize(config_.latent_dim);
        for (std::size_t i = 0; i < config_.latent_dim; ++i) {
            trace.fused[i] = std::tanh(0.7 * trace.encoded[i] + 0.3 * memory_[i]);
        }
        trace.decoded = decode(trace.fused);
        const auto target = volume_.read(coord);
        trace.prediction_error = (static_cast<double>(target) - static_cast<double>(trace.decoded)) / 255.0;
        if (learn) {
            feedback(trace);
        }
        last_trace_ = trace;
        return trace;
    }

    const PsiTrace& last_trace() const noexcept { return last_trace_; }
    std::uint64_t refinement_steps() const noexcept { return refinement_steps_; }
    double best_abs_error() const noexcept { return best_abs_error_; }

private:
    VirtualVolume3D& volume_;
    Config config_;
    OctreeMask3D mask_;
    std::vector<double> memory_;
    PsiTrace last_trace_{};
    std::uint64_t refinement_steps_{};
    double best_abs_error_{std::numeric_limits<double>::infinity()};

    std::vector<double> encode(Coord3 coord) {
        std::array<double, 27> sample{};
        std::size_t cursor = 0U;
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    Coord3 q = coord;
                    q.x = bounded_offset(q.x, dx, volume_.axis_extent());
                    q.y = bounded_offset(q.y, dy, volume_.axis_extent());
                    q.z = bounded_offset(q.z, dz, volume_.axis_extent());
                    sample[cursor++] = static_cast<double>(volume_.read(q)) / 127.5 - 1.0;
                }
            }
        }
        std::vector<double> latent(config_.latent_dim, 0.0);
        for (std::size_t j = 0U; j < latent.size(); ++j) {
            double sum = 0.0;
            for (std::size_t i = 0U; i < sample.size(); ++i) {
                const double phase = static_cast<double>((i + 1U) * (j + 3U));
                const double weight = std::sin(phase * 0.17320508075688773) +
                                      0.5 * std::cos(phase * 0.07106781186547524);
                sum += sample[i] * weight;
            }
            latent[j] = std::tanh(sum / static_cast<double>(sample.size()));
        }
        return latent;
    }

    static std::uint64_t bounded_offset(std::uint64_t value, int delta, std::uint64_t extent) {
        if (delta < 0) {
            return value == 0U ? 0U : value - 1U;
        }
        if (delta > 0) {
            return value + 1U >= extent ? extent - 1U : value + 1U;
        }
        return value;
    }

    static std::uint8_t decode(const std::vector<double>& fused) {
        if (fused.empty()) {
            return 0U;
        }
        double accumulator = 0.0;
        for (std::size_t i = 0U; i < fused.size(); ++i) {
            const double weight = 0.5 + 0.5 * std::sin(static_cast<double>(i + 1U) * 1.6180339887498948);
            accumulator += fused[i] * weight;
        }
        const double normalized = std::tanh(accumulator / static_cast<double>(fused.size()));
        const double byte_value = (normalized + 1.0) * 127.5;
        return static_cast<std::uint8_t>(std::clamp(std::lround(byte_value), 0L, 255L));
    }

    void feedback(const PsiTrace& trace) {
        const double error = trace.prediction_error;
        for (std::size_t i = 0U; i < memory_.size(); ++i) {
            const double gradient = error * trace.encoded[i];
            memory_[i] = std::clamp(config_.momentum * memory_[i] + config_.learning_rate * gradient,
                                    -1.0, 1.0);
        }
        ++refinement_steps_;
        best_abs_error_ = std::min(best_abs_error_, std::abs(error));
    }
};

enum class Opcode : std::uint8_t {
    Nop = 0x00,
    MovImm = 0x01,
    Add = 0x02,
    Xor = 0x03,
    LoadVoxel = 0x10,
    StoreVoxel = 0x11,
    PsiInfer = 0x20,
    PsiLearn = 0x21,
    Jump = 0x30,
    JumpIfNonZero = 0x31,
    Halt = 0xff
};

struct Instruction {
    Opcode opcode{Opcode::Nop};
    std::uint8_t a{};
    std::uint8_t b{};
    std::uint8_t c{};
    std::uint64_t imm{};
};

class BytecodeProgram {
public:
    static constexpr std::array<char, 8> kMagic{{'J','X','3','D','V','M','1','\0'}};

    std::vector<Instruction> instructions;

    void save(const std::filesystem::path& path) const {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("failed to create bytecode file: " + path.string());
        }
        out.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
        write_u64(out, static_cast<std::uint64_t>(instructions.size()));
        for (const auto& instruction : instructions) {
            const auto op = static_cast<std::uint8_t>(instruction.opcode);
            out.put(static_cast<char>(op));
            out.put(static_cast<char>(instruction.a));
            out.put(static_cast<char>(instruction.b));
            out.put(static_cast<char>(instruction.c));
            out.put('\0'); out.put('\0'); out.put('\0'); out.put('\0');
            write_u64(out, instruction.imm);
        }
        if (!out) {
            throw std::runtime_error("failed to write bytecode file: " + path.string());
        }
    }

    static BytecodeProgram load(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("failed to open bytecode file: " + path.string());
        }
        std::array<char, 8> magic{};
        in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (magic != kMagic) {
            throw std::runtime_error("invalid Jarvis-X 3D bytecode magic");
        }
        const auto count = read_u64(in);
        if (count > 10'000'000ULL) {
            throw std::runtime_error("bytecode instruction count exceeds safety limit");
        }
        BytecodeProgram program;
        program.instructions.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t i = 0U; i < count; ++i) {
            Instruction instruction;
            const int op = in.get();
            const int a = in.get();
            const int b = in.get();
            const int c = in.get();
            if (op < 0 || a < 0 || b < 0 || c < 0) {
                throw std::runtime_error("truncated bytecode instruction");
            }
            instruction.opcode = static_cast<Opcode>(static_cast<std::uint8_t>(op));
            instruction.a = static_cast<std::uint8_t>(a);
            instruction.b = static_cast<std::uint8_t>(b);
            instruction.c = static_cast<std::uint8_t>(c);
            for (int pad = 0; pad < 4; ++pad) {
                if (in.get() < 0) throw std::runtime_error("truncated bytecode padding");
            }
            instruction.imm = read_u64(in);
            program.instructions.push_back(instruction);
        }
        return program;
    }

private:
    static void write_u64(std::ostream& out, std::uint64_t value) {
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            out.put(static_cast<char>((value >> shift) & 0xffU));
        }
    }

    static std::uint64_t read_u64(std::istream& in) {
        std::uint64_t value = 0U;
        for (unsigned shift = 0U; shift < 64U; shift += 8U) {
            const int byte = in.get();
            if (byte < 0) throw std::runtime_error("truncated bytecode integer");
            value |= static_cast<std::uint64_t>(static_cast<std::uint8_t>(byte)) << shift;
        }
        return value;
    }
};

struct VmStats {
    std::uint64_t steps{};
    std::uint64_t cycles{};
    std::uint64_t psi_inferences{};
    std::uint64_t psi_learning_steps{};
};

class IntelligenceVm3D {
public:
    IntelligenceVm3D(VirtualVolume3D& volume, PsiIntelligenceCore& psi)
        : volume_(volume), psi_(psi) {}

    void reset_registers() noexcept { registers_.fill(0U); }

    void run(const BytecodeProgram& program, std::uint64_t cycles, std::uint64_t max_steps_per_cycle) {
        if (program.instructions.empty()) {
            throw std::invalid_argument("bytecode program must not be empty");
        }
        if (max_steps_per_cycle == 0U) {
            throw std::invalid_argument("max_steps_per_cycle must be positive");
        }
        for (std::uint64_t cycle = 0U; cycle < cycles; ++cycle) {
            execute_cycle(program, max_steps_per_cycle);
            ++stats_.cycles;
        }
    }

    const std::array<std::uint64_t, 16>& registers() const noexcept { return registers_; }
    const VmStats& stats() const noexcept { return stats_; }

private:
    VirtualVolume3D& volume_;
    PsiIntelligenceCore& psi_;
    std::array<std::uint64_t, 16> registers_{};
    VmStats stats_{};

    static void validate_reg(std::uint8_t reg) {
        if (reg >= 16U) {
            throw std::runtime_error("bytecode register index out of range");
        }
    }

    Coord3 current_coord() const {
        return {registers_[0], registers_[1], registers_[2]};
    }

    void execute_cycle(const BytecodeProgram& program, std::uint64_t max_steps) {
        std::size_t pc = 0U;
        std::uint64_t local_steps = 0U;
        while (pc < program.instructions.size()) {
            if (++local_steps > max_steps) {
                throw std::runtime_error("bytecode cycle exceeded max_steps_per_cycle");
            }
            ++stats_.steps;
            const auto& ins = program.instructions[pc];
            switch (ins.opcode) {
                case Opcode::Nop:
                    ++pc;
                    break;
                case Opcode::MovImm:
                    validate_reg(ins.a);
                    registers_[ins.a] = ins.imm;
                    ++pc;
                    break;
                case Opcode::Add:
                    validate_reg(ins.a); validate_reg(ins.b); validate_reg(ins.c);
                    registers_[ins.a] = registers_[ins.b] + registers_[ins.c];
                    ++pc;
                    break;
                case Opcode::Xor:
                    validate_reg(ins.a); validate_reg(ins.b); validate_reg(ins.c);
                    registers_[ins.a] = registers_[ins.b] ^ registers_[ins.c];
                    ++pc;
                    break;
                case Opcode::LoadVoxel:
                    validate_reg(ins.a);
                    registers_[ins.a] = volume_.read(current_coord());
                    ++pc;
                    break;
                case Opcode::StoreVoxel:
                    validate_reg(ins.a);
                    volume_.write(current_coord(), static_cast<std::uint8_t>(registers_[ins.a] & 0xffU));
                    ++pc;
                    break;
                case Opcode::PsiInfer: {
                    validate_reg(ins.a);
                    const auto trace = psi_.infer(current_coord(), false);
                    registers_[ins.a] = trace.decoded;
                    ++stats_.psi_inferences;
                    ++pc;
                    break;
                }
                case Opcode::PsiLearn: {
                    validate_reg(ins.a);
                    const auto trace = psi_.infer(current_coord(), true);
                    registers_[ins.a] = trace.decoded;
                    ++stats_.psi_inferences;
                    ++stats_.psi_learning_steps;
                    ++pc;
                    break;
                }
                case Opcode::Jump:
                    if (ins.imm >= program.instructions.size()) {
                        throw std::runtime_error("jump target out of range");
                    }
                    pc = static_cast<std::size_t>(ins.imm);
                    break;
                case Opcode::JumpIfNonZero:
                    validate_reg(ins.a);
                    if (registers_[ins.a] != 0U) {
                        if (ins.imm >= program.instructions.size()) {
                            throw std::runtime_error("conditional jump target out of range");
                        }
                        pc = static_cast<std::size_t>(ins.imm);
                    } else {
                        ++pc;
                    }
                    break;
                case Opcode::Halt:
                    return;
                default:
                    throw std::runtime_error("unknown bytecode opcode");
            }
        }
    }
};

inline BytecodeProgram make_demo_program(std::uint64_t x, std::uint64_t y, std::uint64_t z,
                                         std::uint8_t value) {
    BytecodeProgram program;
    program.instructions = {
        {Opcode::MovImm, 0U, 0U, 0U, x},
        {Opcode::MovImm, 1U, 0U, 0U, y},
        {Opcode::MovImm, 2U, 0U, 0U, z},
        {Opcode::MovImm, 3U, 0U, 0U, value},
        {Opcode::StoreVoxel, 3U, 0U, 0U, 0U},
        {Opcode::PsiLearn, 4U, 0U, 0U, 0U},
        {Opcode::PsiInfer, 5U, 0U, 0U, 0U},
        {Opcode::Halt, 0U, 0U, 0U, 0U},
    };
    return program;
}

}  // namespace jarvisx::intelligence3d
