#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jarvisx::media8 {

constexpr std::size_t kTileEdge = 8U;
constexpr std::size_t kTileNodes = kTileEdge * kTileEdge * kTileEdge;
constexpr std::size_t kCoreEdge = 2U;
constexpr std::size_t kCoreNodes = kCoreEdge * kCoreEdge * kCoreEdge;

enum class MediaModality : std::uint8_t {
    Visual = 0U,
    Audio = 1U,
    Text = 2U,
    Generic = 3U,
};

inline const char* modality_name(MediaModality modality) noexcept {
    switch (modality) {
    case MediaModality::Visual: return "visual";
    case MediaModality::Audio: return "audio";
    case MediaModality::Text: return "text";
    case MediaModality::Generic: return "generic";
    }
    return "unknown";
}

enum class GateType : std::uint8_t {
    And = 0x01U,
    Or = 0x02U,
    Xor = 0x03U,
    Not = 0x04U,
};

enum class Opcode : std::uint8_t {
    IngestRaw = 0x10U,
    EncodeSpatial = 0x20U,
    VclGate = 0x30U,
    Conv3DInt8 = 0x40U,
    EvalEntropy = 0x50U,
    PruneLattice = 0x60U,
    DecodeBytecode = 0x70U,
    AutoEvolve = 0x80U,
    SyncLock = 0xFFU,
};

struct VCLNode {
    std::int8_t state{};
    std::int8_t weight{};
    std::uint8_t logic_mask{1U};
    std::uint8_t control_flag{1U};
};

struct TileMetrics {
    double reconstruction_mse{};
    double normalized_entropy{};
    std::size_t active_nodes{};
    std::uint64_t evolution_commits{};
    std::uint64_t evolution_rollbacks{};
    bool entropy_prune_requested{};
    bool synchronized{};
};

struct AdaptiveSnapshot {
    std::array<std::int8_t, kTileNodes> weights{};
    std::array<std::int16_t, kCoreNodes> omega{};
};

inline std::size_t tile_index(std::size_t x, std::size_t y, std::size_t z) noexcept {
    return (z * kTileEdge + y) * kTileEdge + x;
}

inline std::size_t core_index(std::size_t x, std::size_t y, std::size_t z) noexcept {
    return ((z & 1U) * kCoreEdge + (y & 1U)) * kCoreEdge + (x & 1U);
}

inline std::int8_t clamp_i8(int value) noexcept {
    return static_cast<std::int8_t>(std::clamp(value, -128, 127));
}

inline std::int8_t byte_to_centered(std::uint8_t value) noexcept {
    return clamp_i8(static_cast<int>(value) - 128);
}

inline std::uint8_t centered_to_byte(std::int8_t value) noexcept {
    const int restored = static_cast<int>(value) + 128;
    return static_cast<std::uint8_t>(std::clamp(restored, 0, 255));
}

class VCLTile8 {
public:
    VCLTile8() {
        for (VCLNode& node : nodes_) {
            node.logic_mask = 1U;
            node.control_flag = 1U;
        }
    }

    void ingest_tile(const std::array<std::uint8_t, kTileNodes>& bytes) {
        synchronized_ = false;
        entropy_prune_requested_ = false;
        normalized_entropy_ = 0.0;
        for (std::size_t i = 0U; i < kTileNodes; ++i) {
            const auto centered = byte_to_centered(bytes[i]);
            source_[i] = centered;
            reconstruction_[i] = centered;
            nodes_[i].state = centered;
            nodes_[i].logic_mask = 1U;
            nodes_[i].control_flag = 1U;
        }
    }

    void ingest_plane(std::uint8_t z_slice, std::uint8_t raw_value) {
        if (z_slice >= kTileEdge) {
            throw std::out_of_range("VCL INGEST_RAW z-slice outside 8x8x8 tile");
        }
        synchronized_ = false;
        const auto centered = byte_to_centered(raw_value);
        for (std::size_t y = 0U; y < kTileEdge; ++y) {
            for (std::size_t x = 0U; x < kTileEdge; ++x) {
                const std::size_t i = tile_index(x, y, z_slice);
                source_[i] = centered;
                reconstruction_[i] = centered;
                nodes_[i].state = centered;
                nodes_[i].logic_mask = 1U;
                nodes_[i].control_flag = 1U;
            }
        }
    }

    void conv3d(std::uint8_t kernel_id, std::uint8_t bias_byte) {
        const std::array<std::int8_t, kTileNodes> before = states();
        std::array<std::int8_t, kTileNodes> next = before;
        const int bias = bias_byte < 128U ? static_cast<int>(bias_byte)
                                         : static_cast<int>(bias_byte) - 256;
        for (std::size_t z = 0U; z < kTileEdge; ++z) {
            for (std::size_t y = 0U; y < kTileEdge; ++y) {
                for (std::size_t x = 0U; x < kTileEdge; ++x) {
                    const std::size_t out_index = tile_index(x, y, z);
                    if (!active(nodes_[out_index])) continue;
                    std::int32_t sum = 0;
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                const std::size_t sx = clamp_coord(static_cast<int>(x) + dx);
                                const std::size_t sy = clamp_coord(static_cast<int>(y) + dy);
                                const std::size_t sz = clamp_coord(static_cast<int>(z) + dz);
                                const int coefficient = kernel_coefficient(kernel_id, dx, dy, dz);
                                sum += static_cast<std::int32_t>(coefficient) *
                                       static_cast<std::int32_t>(before[tile_index(sx, sy, sz)]);
                            }
                        }
                    }
                    const int scaled = requantize_kernel(kernel_id, sum) + bias;
                    next[out_index] = clamp_i8(scaled);
                }
            }
        }
        for (std::size_t i = 0U; i < kTileNodes; ++i) nodes_[i].state = next[i];
    }

    void vcl_gate(std::uint8_t gate_type, std::uint8_t threshold) {
        for (VCLNode& node : nodes_) {
            const bool above = centered_to_byte(node.state) > threshold;
            const bool mask = node.logic_mask != 0U;
            bool value = false;
            switch (static_cast<GateType>(gate_type)) {
            case GateType::And: value = above && mask; break;
            case GateType::Or: value = above || mask; break;
            case GateType::Xor: value = above != mask; break;
            case GateType::Not: value = (!above) && mask; break;
            default: throw std::invalid_argument("unsupported VCL gate type");
            }
            node.control_flag = value ? 1U : 0U;
        }
    }

    void encode_spatial(std::uint8_t z_start, std::uint8_t z_end) {
        validate_range(z_start, z_end, "ENC_SPATIAL");
        std::array<std::int32_t, kCoreNodes> sums{};
        std::array<std::uint16_t, kCoreNodes> counts{};
        for (std::size_t z = z_start; z <= z_end; ++z) {
            for (std::size_t y = 0U; y < kTileEdge; ++y) {
                for (std::size_t x = 0U; x < kTileEdge; ++x) {
                    const std::size_t i = tile_index(x, y, z);
                    if (!active(nodes_[i])) continue;
                    const std::size_t c = core_index(x, y, z);
                    sums[c] += static_cast<std::int32_t>(nodes_[i].state);
                    ++counts[c];
                }
            }
        }
        for (std::size_t c = 0U; c < kCoreNodes; ++c) {
            const int raw = counts[c] == 0U ? 0 :
                static_cast<int>(sums[c] / static_cast<std::int32_t>(counts[c]));
            raw_core_[c] = clamp_i8(raw);
            const int fused = (3 * raw + static_cast<int>(omega_[c])) / 4;
            core_[c] = clamp_i8(fused);
        }
    }

    double eval_entropy(std::uint8_t shell_id, std::uint8_t threshold) {
        std::array<std::uint32_t, 256U> histogram{};
        std::uint64_t count = 0U;
        for (std::size_t z = 0U; z < kTileEdge; ++z) {
            for (std::size_t y = 0U; y < kTileEdge; ++y) {
                for (std::size_t x = 0U; x < kTileEdge; ++x) {
                    if (!in_shell(shell_id, x, y, z)) continue;
                    const VCLNode& node = nodes_[tile_index(x, y, z)];
                    if (node.logic_mask == 0U) continue;
                    ++histogram[centered_to_byte(node.state)];
                    ++count;
                }
            }
        }
        if (count == 0U) {
            normalized_entropy_ = 0.0;
        } else {
            double entropy = 0.0;
            for (const std::uint32_t frequency : histogram) {
                if (frequency == 0U) continue;
                const double probability = static_cast<double>(frequency) /
                                           static_cast<double>(count);
                entropy -= probability * std::log2(probability);
            }
            normalized_entropy_ = entropy / 8.0;
        }
        entropy_prune_requested_ = normalized_entropy_ * 255.0 <
                                   static_cast<double>(threshold);
        return normalized_entropy_;
    }

    void prune_lattice(std::uint8_t target_mask) {
        if (!entropy_prune_requested_) return;
        for (VCLNode& node : nodes_) {
            bool prune = false;
            if ((target_mask & 0x01U) != 0U && node.control_flag == 0U) prune = true;
            if ((target_mask & 0x02U) != 0U && std::abs(static_cast<int>(node.state)) < 8) {
                prune = true;
            }
            if (prune) node.logic_mask = 0U;
        }
    }

    void decode_bytecode(std::uint8_t z_start, std::uint8_t z_end) {
        validate_range(z_start, z_end, "DEC_BYTECODE");
        const auto decoded = shadow_decode();
        for (std::size_t z = z_start; z <= z_end; ++z) {
            for (std::size_t y = 0U; y < kTileEdge; ++y) {
                for (std::size_t x = 0U; x < kTileEdge; ++x) {
                    const std::size_t i = tile_index(x, y, z);
                    reconstruction_[i] = decoded[i];
                    nodes_[i].state = decoded[i];
                    nodes_[i].logic_mask = 1U;
                }
            }
        }
    }

    bool auto_evolve(std::uint8_t learning_rate) {
        const auto weights_before = weights();
        const auto baseline_reconstruction = shadow_decode();
        const double baseline = mse_against_source(baseline_reconstruction);

        for (std::size_t i = 0U; i < kTileNodes; ++i) {
            if (!active(nodes_[i])) continue;
            const int residual = static_cast<int>(source_[i]) -
                                 static_cast<int>(baseline_reconstruction[i]);
            const int delta = (static_cast<int>(learning_rate) * residual) >> 4;
            nodes_[i].weight = clamp_i8(static_cast<int>(nodes_[i].weight) + delta);
        }

        const auto candidate_reconstruction = shadow_decode();
        const double candidate = mse_against_source(candidate_reconstruction);
        if (std::isfinite(candidate) && candidate <= baseline + 1.0e-12) {
            ++evolution_commits_;
            return true;
        }

        restore_weights(weights_before);
        ++evolution_rollbacks_;
        return false;
    }

    void sync_lock() {
        for (std::size_t c = 0U; c < kCoreNodes; ++c) {
            omega_[c] = static_cast<std::int16_t>(
                (7 * static_cast<int>(omega_[c]) +
                 static_cast<int>(raw_core_[c])) / 8);
        }
        synchronized_ = true;
    }

    AdaptiveSnapshot adaptive_snapshot() const noexcept {
        return {weights(), omega_};
    }

    void restore_adaptive(const AdaptiveSnapshot& snapshot) noexcept {
        restore_weights(snapshot.weights);
        omega_ = snapshot.omega;
    }

    std::array<std::uint8_t, kTileNodes> reconstruction_bytes() const noexcept {
        std::array<std::uint8_t, kTileNodes> result{};
        for (std::size_t i = 0U; i < kTileNodes; ++i) {
            result[i] = centered_to_byte(reconstruction_[i]);
        }
        return result;
    }

    TileMetrics metrics() const noexcept {
        TileMetrics result;
        result.reconstruction_mse = mse_against_source(reconstruction_);
        result.normalized_entropy = normalized_entropy_;
        result.active_nodes = active_node_count();
        result.evolution_commits = evolution_commits_;
        result.evolution_rollbacks = evolution_rollbacks_;
        result.entropy_prune_requested = entropy_prune_requested_;
        result.synchronized = synchronized_;
        return result;
    }

    const std::array<VCLNode, kTileNodes>& nodes() const noexcept { return nodes_; }
    const std::array<std::int8_t, kCoreNodes>& core() const noexcept { return core_; }

private:
    std::array<VCLNode, kTileNodes> nodes_{};
    std::array<std::int8_t, kTileNodes> source_{};
    std::array<std::int8_t, kTileNodes> reconstruction_{};
    std::array<std::int8_t, kCoreNodes> raw_core_{};
    std::array<std::int8_t, kCoreNodes> core_{};
    std::array<std::int16_t, kCoreNodes> omega_{};
    double normalized_entropy_{};
    bool entropy_prune_requested_{};
    bool synchronized_{};
    std::uint64_t evolution_commits_{};
    std::uint64_t evolution_rollbacks_{};

    static bool active(const VCLNode& node) noexcept {
        return node.logic_mask != 0U && node.control_flag != 0U;
    }

    static std::size_t clamp_coord(int value) noexcept {
        return static_cast<std::size_t>(std::clamp(value, 0, static_cast<int>(kTileEdge - 1U)));
    }

    static void validate_range(std::uint8_t start, std::uint8_t end, const char* op) {
        if (start > end || end >= kTileEdge) {
            throw std::out_of_range(std::string(op) + " z-range outside 8x8x8 tile");
        }
    }

    static int kernel_coefficient(std::uint8_t kernel_id, int dx, int dy, int dz) {
        switch (kernel_id) {
        case 0U:
            return (dx == 0 && dy == 0 && dz == 0) ? 1 : 0;
        case 1U:
            return 1;
        case 2U:
            return (dx == 0 && dy == 0 && dz == 0) ? 26 : -1;
        case 3U: {
            const int manhattan = std::abs(dx) + std::abs(dy) + std::abs(dz);
            if (manhattan == 0) return 7;
            if (manhattan == 1) return -1;
            return 0;
        }
        default:
            throw std::invalid_argument("unknown CONV_3D_INT8 kernel id");
        }
    }

    static int requantize_kernel(std::uint8_t kernel_id, std::int32_t sum) noexcept {
        switch (kernel_id) {
        case 0U: return static_cast<int>(sum);
        case 1U: return static_cast<int>(sum / 27);
        case 2U: return static_cast<int>(sum / 16);
        case 3U: return static_cast<int>(sum);
        default: return 0;
        }
    }

    static bool in_shell(std::uint8_t shell, std::size_t x,
                         std::size_t y, std::size_t z) {
        switch (shell) {
        case 0U:
            return true;
        case 1U:
            return x == 0U || y == 0U || z == 0U ||
                   x + 1U == kTileEdge || y + 1U == kTileEdge || z + 1U == kTileEdge;
        case 2U:
            return x > 0U && y > 0U && z > 0U &&
                   x + 1U < kTileEdge && y + 1U < kTileEdge && z + 1U < kTileEdge;
        case 3U:
            return (x == 3U || x == 4U) &&
                   (y == 3U || y == 4U) &&
                   (z == 3U || z == 4U);
        default:
            throw std::invalid_argument("unknown entropy shell id");
        }
    }

    std::array<std::int8_t, kTileNodes> states() const noexcept {
        std::array<std::int8_t, kTileNodes> result{};
        for (std::size_t i = 0U; i < kTileNodes; ++i) result[i] = nodes_[i].state;
        return result;
    }

    std::array<std::int8_t, kTileNodes> weights() const noexcept {
        std::array<std::int8_t, kTileNodes> result{};
        for (std::size_t i = 0U; i < kTileNodes; ++i) result[i] = nodes_[i].weight;
        return result;
    }

    void restore_weights(const std::array<std::int8_t, kTileNodes>& values) noexcept {
        for (std::size_t i = 0U; i < kTileNodes; ++i) nodes_[i].weight = values[i];
    }

    std::array<std::int8_t, kTileNodes> shadow_decode() const noexcept {
        std::array<std::int8_t, kTileNodes> result{};
        for (std::size_t z = 0U; z < kTileEdge; ++z) {
            for (std::size_t y = 0U; y < kTileEdge; ++y) {
                for (std::size_t x = 0U; x < kTileEdge; ++x) {
                    const std::size_t i = tile_index(x, y, z);
                    const std::size_t c = core_index(x, y, z);
                    const int decoded = static_cast<int>(core_[c]) +
                                        static_cast<int>(nodes_[i].weight) / 8;
                    result[i] = clamp_i8(decoded);
                }
            }
        }
        return result;
    }

    double mse_against_source(const std::array<std::int8_t, kTileNodes>& values) const noexcept {
        double squared = 0.0;
        for (std::size_t i = 0U; i < kTileNodes; ++i) {
            const double delta = static_cast<double>(source_[i]) -
                                 static_cast<double>(values[i]);
            squared += delta * delta;
        }
        return squared / static_cast<double>(kTileNodes);
    }

    std::size_t active_node_count() const noexcept {
        std::size_t count = 0U;
        for (const VCLNode& node : nodes_) if (active(node)) ++count;
        return count;
    }
};

class VCLBVM8 {
public:
    explicit VCLBVM8(VCLTile8& tile) : tile_(tile) {}

    TileMetrics execute(const std::vector<std::uint8_t>& bytecode,
                        std::size_t instruction_limit = 4096U) {
        std::size_t ip = 0U;
        std::size_t instructions = 0U;
        while (ip < bytecode.size()) {
            if (++instructions > instruction_limit) {
                throw std::runtime_error("VCL-BVM-8 instruction limit exceeded");
            }
            const std::uint8_t header = bytecode[ip++];
            if (header == static_cast<std::uint8_t>(Opcode::SyncLock)) {
                tile_.sync_lock();
                return tile_.metrics();
            }
            const std::uint8_t opcode = header & 0xF0U;
            const std::uint8_t mode = header & 0x0FU;
            (void)mode;
            switch (opcode) {
            case static_cast<std::uint8_t>(Opcode::IngestRaw): {
                require(bytecode, ip, 2U);
                const auto z = bytecode[ip++];
                const auto value = bytecode[ip++];
                tile_.ingest_plane(z, value);
                break;
            }
            case static_cast<std::uint8_t>(Opcode::EncodeSpatial): {
                require(bytecode, ip, 2U);
                const auto start = bytecode[ip++];
                const auto end = bytecode[ip++];
                tile_.encode_spatial(start, end);
                break;
            }
            case static_cast<std::uint8_t>(Opcode::VclGate): {
                require(bytecode, ip, 2U);
                const auto gate = bytecode[ip++];
                const auto threshold = bytecode[ip++];
                tile_.vcl_gate(gate, threshold);
                break;
            }
            case static_cast<std::uint8_t>(Opcode::Conv3DInt8): {
                require(bytecode, ip, 2U);
                const auto kernel = bytecode[ip++];
                const auto bias = bytecode[ip++];
                tile_.conv3d(kernel, bias);
                break;
            }
            case static_cast<std::uint8_t>(Opcode::EvalEntropy): {
                require(bytecode, ip, 2U);
                const auto shell = bytecode[ip++];
                const auto threshold = bytecode[ip++];
                tile_.eval_entropy(shell, threshold);
                break;
            }
            case static_cast<std::uint8_t>(Opcode::PruneLattice): {
                require(bytecode, ip, 1U);
                tile_.prune_lattice(bytecode[ip++]);
                break;
            }
            case static_cast<std::uint8_t>(Opcode::DecodeBytecode): {
                require(bytecode, ip, 2U);
                const auto start = bytecode[ip++];
                const auto end = bytecode[ip++];
                tile_.decode_bytecode(start, end);
                break;
            }
            case static_cast<std::uint8_t>(Opcode::AutoEvolve): {
                require(bytecode, ip, 1U);
                tile_.auto_evolve(bytecode[ip++]);
                break;
            }
            default:
                throw std::runtime_error("unknown VCL-BVM-8 opcode header");
            }
        }
        throw std::runtime_error("VCL-BVM-8 stream ended without SYNC_LOCK");
    }

private:
    VCLTile8& tile_;

    static void require(const std::vector<std::uint8_t>& bytecode,
                        std::size_t ip, std::size_t count) {
        if (count > bytecode.size() - std::min(ip, bytecode.size())) {
            throw std::runtime_error("truncated VCL-BVM-8 instruction payload");
        }
    }
};

inline std::vector<std::uint8_t> default_media_program() {
    return {
        0x40U, 0x00U, 0x00U,       // identity 3D convolution, zero bias
        0x30U, 0x02U, 0x00U,       // OR gate keeps valid mask active
        0x20U, 0x00U, 0x07U,       // encode all eight Z slices inward
        0x50U, 0x00U, 0x08U,       // true entropy over whole tile
        0x60U, 0x01U,              // prune gate-disabled nodes if requested
        0x80U, 0x08U,              // bounded residual candidate update
        0x70U, 0x00U, 0x07U,       // decode the complete tile
        0xFFU,                      // Lambda/Omega synchronization lock
    };
}

struct ProcessorConfig {
    double max_output_mse{4096.0};
    bool fallback_on_regression{true};
    std::size_t instruction_limit{4096U};

    void validate() const {
        if (!std::isfinite(max_output_mse) || max_output_mse < 0.0) {
            throw std::invalid_argument("max_output_mse must be finite and non-negative");
        }
        if (instruction_limit == 0U) {
            throw std::invalid_argument("instruction_limit must be positive");
        }
    }
};

struct ProcessorMetrics {
    MediaModality modality{MediaModality::Generic};
    std::uint64_t tiles{};
    std::uint64_t accepted_tiles{};
    std::uint64_t rejected_tiles{};
    double average_candidate_mse{};
    double average_output_mse{};
    double hbar_semantic{};
    double average_entropy{};
    double average_active_nodes{};
    std::uint64_t evolution_commits{};
    std::uint64_t evolution_rollbacks{};
};

struct ProcessorResult {
    std::vector<std::uint8_t> bytes;
    ProcessorMetrics metrics;
};

class DrMoagiIntelligenceMediaProcessor {
public:
    explicit DrMoagiIntelligenceMediaProcessor(
        ProcessorConfig config = {},
        std::vector<std::uint8_t> program = default_media_program())
        : config_(config), program_(std::move(program)) {
        config_.validate();
        if (program_.empty()) throw std::invalid_argument("VCL media program must not be empty");
    }

    const std::vector<std::uint8_t>& program() const noexcept { return program_; }
    const VCLTile8& tile() const noexcept { return tile_; }

    void set_program(std::vector<std::uint8_t> program) {
        if (program.empty()) throw std::invalid_argument("VCL media program must not be empty");
        program_ = std::move(program);
    }

    ProcessorResult process(const std::vector<std::uint8_t>& input,
                            MediaModality modality = MediaModality::Generic) {
        ProcessorResult result;
        result.bytes.resize(input.size());
        result.metrics.modality = modality;
        if (input.empty()) return result;

        double candidate_mse_sum = 0.0;
        double output_mse_sum = 0.0;
        double entropy_sum = 0.0;
        double active_sum = 0.0;
        std::uint64_t previous_evolution_commits = tile_.metrics().evolution_commits;
        std::uint64_t previous_evolution_rollbacks = tile_.metrics().evolution_rollbacks;

        for (std::size_t offset = 0U; offset < input.size(); offset += kTileNodes) {
            const std::size_t available = std::min(kTileNodes, input.size() - offset);
            std::array<std::uint8_t, kTileNodes> tile_bytes{};
            tile_bytes.fill(128U);
            std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(offset),
                        static_cast<std::ptrdiff_t>(available), tile_bytes.begin());

            const AdaptiveSnapshot adaptive_before = tile_.adaptive_snapshot();
            tile_.ingest_tile(tile_bytes);
            VCLBVM8 vm(tile_);
            const TileMetrics tile_metrics = vm.execute(program_, config_.instruction_limit);
            const auto candidate = tile_.reconstruction_bytes();
            const double candidate_mse = mse_bytes(tile_bytes, candidate, available);
            const bool accept = std::isfinite(candidate_mse) &&
                                candidate_mse <= config_.max_output_mse;

            if (accept) {
                ++result.metrics.accepted_tiles;
                std::copy_n(candidate.begin(), static_cast<std::ptrdiff_t>(available),
                            result.bytes.begin() + static_cast<std::ptrdiff_t>(offset));
                output_mse_sum += candidate_mse;
            } else {
                ++result.metrics.rejected_tiles;
                tile_.restore_adaptive(adaptive_before);
                if (config_.fallback_on_regression) {
                    std::copy_n(input.begin() + static_cast<std::ptrdiff_t>(offset),
                                static_cast<std::ptrdiff_t>(available),
                                result.bytes.begin() + static_cast<std::ptrdiff_t>(offset));
                } else {
                    std::copy_n(candidate.begin(), static_cast<std::ptrdiff_t>(available),
                                result.bytes.begin() + static_cast<std::ptrdiff_t>(offset));
                    output_mse_sum += candidate_mse;
                }
            }

            ++result.metrics.tiles;
            candidate_mse_sum += candidate_mse;
            entropy_sum += tile_metrics.normalized_entropy;
            active_sum += static_cast<double>(tile_metrics.active_nodes);
        }

        const double tile_count = static_cast<double>(result.metrics.tiles);
        result.metrics.average_candidate_mse = candidate_mse_sum / tile_count;
        result.metrics.average_output_mse = output_mse_sum / tile_count;
        result.metrics.hbar_semantic =
            std::sqrt(result.metrics.average_output_mse) / 255.0;
        result.metrics.average_entropy = entropy_sum / tile_count;
        result.metrics.average_active_nodes = active_sum / tile_count;
        const TileMetrics final_tile = tile_.metrics();
        result.metrics.evolution_commits =
            final_tile.evolution_commits - previous_evolution_commits;
        result.metrics.evolution_rollbacks =
            final_tile.evolution_rollbacks - previous_evolution_rollbacks;
        return result;
    }

private:
    ProcessorConfig config_;
    std::vector<std::uint8_t> program_;
    VCLTile8 tile_;

    static double mse_bytes(const std::array<std::uint8_t, kTileNodes>& left,
                            const std::array<std::uint8_t, kTileNodes>& right,
                            std::size_t count) noexcept {
        if (count == 0U) return 0.0;
        double squared = 0.0;
        for (std::size_t i = 0U; i < count; ++i) {
            const double delta = static_cast<double>(left[i]) -
                                 static_cast<double>(right[i]);
            squared += delta * delta;
        }
        return squared / static_cast<double>(count);
    }
};

} // namespace jarvisx::media8
