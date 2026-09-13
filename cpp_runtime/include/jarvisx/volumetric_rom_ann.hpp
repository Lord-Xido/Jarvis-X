#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
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

namespace jarvisx::volumetric_rom {

constexpr std::uint32_t kAxisBits = 20;
constexpr std::uint32_t kAxisPositions = 1u << kAxisBits;
constexpr std::uint32_t kTileEdge = 32;
constexpr std::uint32_t kTileBits = 5;
constexpr std::uint32_t kTileVoxels = kTileEdge * kTileEdge * kTileEdge;
constexpr std::uint32_t kTileCoordBits = kAxisBits - kTileBits;
constexpr std::size_t kLatentDim = 8;
constexpr float kEpsilon = 1.0e-6F;

static_assert(3u * kAxisBits <= 64u, "Packed voxel address must fit in uint64_t");
static_assert(3u * kTileCoordBits <= 64u, "Packed tile address must fit in uint64_t");

struct Vec3u {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
};

inline bool operator==(const Vec3u& lhs, const Vec3u& rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

struct RuntimePolicy {
    std::uint32_t max_active_tiles = 64;
    std::uint32_t fixed_point_steps = 12;
    float fixed_point_tolerance = 1.0e-4F;
    float learning_rate = 1.0e-3F;
    float omega_decay = 0.92F;
    float sparse_residual_threshold = 0.20F;
};

struct PyramidLevel {
    std::uint32_t edge{};
    std::vector<float> values;
};

struct EngineStats {
    std::uint64_t cycles{};
    std::size_t active_tiles{};
    std::uint64_t physical_bytes{};
    float last_error{};
    float last_fixed_point_relative{};
    std::uint32_t fixed_point_steps{};
    std::size_t pyramid_levels{};
    std::uint64_t sparse_residual_voxels{};
    Vec3u last_position{};
    std::uint64_t last_address{};
    std::uint64_t last_tile_key{};
};

struct Tile {
    std::vector<float> signal;
    std::vector<float> reconstruction;
    std::vector<float> residual;
    std::uint64_t touches = 0;
    float last_error = 0.0F;

    Tile()
        : signal(kTileVoxels, 0.0F),
          reconstruction(kTileVoxels, 0.0F),
          residual(kTileVoxels, 0.0F) {}
};

class SparseVram {
public:
    Tile& fetch_or_allocate(std::uint64_t tile_key, const RuntimePolicy& policy) {
        auto it = tiles_.find(tile_key);
        if (it != tiles_.end()) {
            ++it->second.touches;
            return it->second;
        }

        if (policy.max_active_tiles == 0u) {
            throw std::invalid_argument("max_active_tiles must be greater than zero");
        }
        if (tiles_.size() >= policy.max_active_tiles) {
            evict_least_useful();
        }

        auto [inserted, ok] = tiles_.emplace(tile_key, Tile{});
        if (!ok) {
            throw std::runtime_error("sparse VRAM tile allocation failed");
        }
        inserted->second.touches = 1u;
        return inserted->second;
    }

    [[nodiscard]] std::size_t active_tile_count() const noexcept { return tiles_.size(); }

    [[nodiscard]] std::uint64_t physical_bytes_approx() const noexcept {
        return static_cast<std::uint64_t>(tiles_.size()) *
               static_cast<std::uint64_t>(kTileVoxels) * sizeof(float) * 3ull;
    }

private:
    std::unordered_map<std::uint64_t, Tile> tiles_;

    void evict_least_useful() {
        if (tiles_.empty()) {
            return;
        }
        auto victim = tiles_.begin();
        for (auto it = tiles_.begin(); it != tiles_.end(); ++it) {
            const double candidate = static_cast<double>(it->second.touches) +
                                     100.0 * static_cast<double>(it->second.last_error);
            const double current = static_cast<double>(victim->second.touches) +
                                   100.0 * static_cast<double>(victim->second.last_error);
            if (candidate < current) {
                victim = it;
            }
        }
        tiles_.erase(victim);
    }
};

enum class OpCode : std::uint8_t {
    Resolve,
    FetchAllocate,
    EncodePyramid,
    Contract,
    FixedPoint,
    Decode,
    Compare,
    UpdateOmega,
    UpdateTheta,
    OptimizeRuntime,
    Store,
    Recur,
    Halt
};

struct Instruction {
    OpCode opcode;
    const char* name;
};

inline constexpr std::array<Instruction, 13> kRom{{
    {OpCode::Resolve, "RESOLVE"},
    {OpCode::FetchAllocate, "FETCH_ALLOC"},
    {OpCode::EncodePyramid, "ENCODE_PYRAMID"},
    {OpCode::Contract, "CONTRACT"},
    {OpCode::FixedPoint, "FIXPOINT"},
    {OpCode::Decode, "DECODE"},
    {OpCode::Compare, "COMPARE"},
    {OpCode::UpdateOmega, "UPDATE_OMEGA"},
    {OpCode::UpdateTheta, "UPDATE_THETA"},
    {OpCode::OptimizeRuntime, "OPTIMIZE_RUNTIME"},
    {OpCode::Store, "STORE"},
    {OpCode::Recur, "RECUR"},
    {OpCode::Halt, "HALT"}
}};

class Engine {
public:
    explicit Engine(RuntimePolicy policy = {})
        : policy_(policy), rng_(0x5A17B1u), noise_(-1.0F, 1.0F) {
        if (policy_.max_active_tiles == 0u) {
            throw std::invalid_argument("max_active_tiles must be greater than zero");
        }
        for (std::size_t i = 0; i < kLatentDim; ++i) {
            theta_[i] = 0.15F + 0.03F * static_cast<float>(i);
            omega_[i] = 0.0F;
            latent_[i] = 0.0F;
        }
    }

    static std::uint64_t pack_voxel(const Vec3u& p) {
        if (p.x >= kAxisPositions || p.y >= kAxisPositions || p.z >= kAxisPositions) {
            throw std::out_of_range("voxel coordinate exceeds 20-bit axis range");
        }
        return (static_cast<std::uint64_t>(p.x) << 40u) |
               (static_cast<std::uint64_t>(p.y) << 20u) |
               static_cast<std::uint64_t>(p.z);
    }

    static Vec3u unpack_voxel(std::uint64_t address) noexcept {
        constexpr std::uint64_t mask = (1ull << kAxisBits) - 1ull;
        return {
            static_cast<std::uint32_t>((address >> 40u) & mask),
            static_cast<std::uint32_t>((address >> 20u) & mask),
            static_cast<std::uint32_t>(address & mask)
        };
    }

    static std::vector<PyramidLevel> build_pyramid(const std::vector<float>& input) {
        if (input.size() != kTileVoxels) {
            throw std::invalid_argument("pyramid input must contain exactly 32^3 voxels");
        }

        std::vector<PyramidLevel> pyramid;
        pyramid.push_back(PyramidLevel{kTileEdge, input});

        std::uint32_t edge = kTileEdge;
        while (edge > 1u) {
            const auto& fine = pyramid.back().values;
            const std::uint32_t coarse_edge = edge / 2u;
            std::vector<float> coarse(
                static_cast<std::size_t>(coarse_edge) * coarse_edge * coarse_edge, 0.0F);

            for (std::uint32_t z = 0; z < coarse_edge; ++z) {
                for (std::uint32_t y = 0; y < coarse_edge; ++y) {
                    for (std::uint32_t x = 0; x < coarse_edge; ++x) {
                        double sum = 0.0;
                        for (std::uint32_t dz = 0; dz < 2u; ++dz) {
                            for (std::uint32_t dy = 0; dy < 2u; ++dy) {
                                for (std::uint32_t dx = 0; dx < 2u; ++dx) {
                                    const std::uint32_t fx = 2u * x + dx;
                                    const std::uint32_t fy = 2u * y + dy;
                                    const std::uint32_t fz = 2u * z + dz;
                                    sum += static_cast<double>(fine[index3d(edge, fx, fy, fz)]);
                                }
                            }
                        }
                        coarse[index3d(coarse_edge, x, y, z)] = static_cast<float>(sum / 8.0);
                    }
                }
            }

            pyramid.push_back(PyramidLevel{coarse_edge, std::move(coarse)});
            edge = coarse_edge;
        }

        return pyramid;
    }

    EngineStats run(std::uint32_t cycles, bool quiet = false, std::ostream& out = std::cout) {
        if (cycles == 0u) {
            throw std::invalid_argument("cycles must be greater than zero");
        }
        if (!quiet) {
            print_banner(out);
        }
        for (std::uint32_t t = 0; t < cycles; ++t) {
            execute_cycle(t, quiet, out);
        }
        return stats();
    }

    EngineStats stats() const noexcept {
        EngineStats result{};
        result.cycles = cycles_completed_;
        result.active_tiles = vram_.active_tile_count();
        result.physical_bytes = vram_.physical_bytes_approx();
        result.last_error = current_error_;
        result.last_fixed_point_relative = last_fixed_point_relative_;
        result.fixed_point_steps = policy_.fixed_point_steps;
        result.pyramid_levels = pyramid_.size();
        result.sparse_residual_voxels = sparse_residual_voxels_;
        result.last_position = current_position_;
        result.last_address = current_address_;
        result.last_tile_key = current_tile_key_;
        return result;
    }

    [[nodiscard]] const RuntimePolicy& policy() const noexcept { return policy_; }

private:
    SparseVram vram_;
    RuntimePolicy policy_;
    std::array<float, kLatentDim> latent_{};
    std::array<float, kLatentDim> omega_{};
    std::array<float, kLatentDim> theta_{};
    std::vector<PyramidLevel> pyramid_;
    std::mt19937 rng_;
    std::uniform_real_distribution<float> noise_;

    Vec3u current_position_{};
    std::uint64_t current_address_ = 0;
    std::uint64_t current_tile_key_ = 0;
    Tile* current_tile_ = nullptr;
    float current_error_ = 0.0F;
    float last_fixed_point_relative_ = 0.0F;
    std::uint64_t sparse_residual_voxels_ = 0;
    std::uint64_t cycles_completed_ = 0;

    static std::size_t index3d(std::uint32_t edge,
                               std::uint32_t x,
                               std::uint32_t y,
                               std::uint32_t z) noexcept {
        return (static_cast<std::size_t>(z) * edge + y) * edge + x;
    }

    static std::uint64_t pack_tile(const Vec3u& p) noexcept {
        const std::uint32_t tx = p.x >> kTileBits;
        const std::uint32_t ty = p.y >> kTileBits;
        const std::uint32_t tz = p.z >> kTileBits;
        return (static_cast<std::uint64_t>(tx) << (2u * kTileCoordBits)) |
               (static_cast<std::uint64_t>(ty) << kTileCoordBits) |
               static_cast<std::uint64_t>(tz);
    }

    static std::uint32_t local_index(const Vec3u& p) noexcept {
        const std::uint32_t lx = p.x & (kTileEdge - 1u);
        const std::uint32_t ly = p.y & (kTileEdge - 1u);
        const std::uint32_t lz = p.z & (kTileEdge - 1u);
        return static_cast<std::uint32_t>(index3d(kTileEdge, lx, ly, lz));
    }

    static Vec3u generate_coordinate(std::uint32_t t) noexcept {
        const std::uint32_t mask = kAxisPositions - 1u;
        return {
            (t * 2654435761u + 17u) & mask,
            (t * 2246822519u + 12345u) & mask,
            (t * 3266489917u + 777u) & mask
        };
    }

    static float residual_energy(const PyramidLevel& fine, const PyramidLevel& coarse) {
        if (fine.edge != 2u * coarse.edge) {
            throw std::invalid_argument("pyramid levels are not adjacent");
        }
        double sum = 0.0;
        for (std::uint32_t z = 0; z < fine.edge; ++z) {
            for (std::uint32_t y = 0; y < fine.edge; ++y) {
                for (std::uint32_t x = 0; x < fine.edge; ++x) {
                    const float predicted = coarse.values[index3d(
                        coarse.edge, x / 2u, y / 2u, z / 2u)];
                    const float delta = fine.values[index3d(fine.edge, x, y, z)] - predicted;
                    sum += static_cast<double>(delta) * static_cast<double>(delta);
                }
            }
        }
        return static_cast<float>(std::sqrt(sum / static_cast<double>(fine.values.size())));
    }

    void execute_cycle(std::uint32_t t, bool quiet, std::ostream& out) {
        current_position_ = generate_coordinate(t);
        current_address_ = pack_voxel(current_position_);
        current_tile_ = nullptr;
        current_error_ = 0.0F;
        last_fixed_point_relative_ = 0.0F;
        sparse_residual_voxels_ = 0u;

        for (const auto& instruction : kRom) {
            if (instruction.opcode == OpCode::Halt) {
                break;
            }
            dispatch(instruction.opcode, t);
        }

        ++cycles_completed_;
        if (!quiet && (t < 8u || t % 8u == 0u)) {
            out << "cycle=" << std::setw(3) << t
                << " addr=0x" << std::hex << std::setw(15) << std::setfill('0')
                << current_address_ << std::dec << std::setfill(' ')
                << " xyz=(" << current_position_.x << ',' << current_position_.y << ','
                << current_position_.z << ')'
                << " tile=" << current_tile_key_
                << " mse=" << std::fixed << std::setprecision(6) << current_error_
                << " fp=" << last_fixed_point_relative_
                << " sparse_residuals=" << sparse_residual_voxels_
                << " active_tiles=" << vram_.active_tile_count() << '\n';
        }
    }

    void dispatch(OpCode opcode, std::uint32_t t) {
        switch (opcode) {
            case OpCode::Resolve: op_resolve(); break;
            case OpCode::FetchAllocate: op_fetch_allocate(t); break;
            case OpCode::EncodePyramid: op_encode_pyramid(); break;
            case OpCode::Contract: op_contract(); break;
            case OpCode::FixedPoint: op_fixed_point(); break;
            case OpCode::Decode: op_decode(); break;
            case OpCode::Compare: op_compare(); break;
            case OpCode::UpdateOmega: op_update_omega(); break;
            case OpCode::UpdateTheta: op_update_theta(); break;
            case OpCode::OptimizeRuntime: op_optimize_runtime(); break;
            case OpCode::Store: op_store(); break;
            case OpCode::Recur: op_recur(); break;
            case OpCode::Halt: break;
        }
    }

    void op_resolve() {
        const Vec3u decoded = unpack_voxel(current_address_);
        if (!(decoded == current_position_)) {
            throw std::runtime_error("60-bit address decoder mismatch");
        }
        current_tile_key_ = pack_tile(decoded);
    }

    void op_fetch_allocate(std::uint32_t t) {
        current_tile_ = &vram_.fetch_or_allocate(current_tile_key_, policy_);
        const float phase = 0.07F * static_cast<float>(t);

        for (std::uint32_t z = 0; z < kTileEdge; ++z) {
            for (std::uint32_t y = 0; y < kTileEdge; ++y) {
                for (std::uint32_t x = 0; x < kTileEdge; ++x) {
                    const float fx = static_cast<float>(x) / static_cast<float>(kTileEdge - 1u);
                    const float fy = static_cast<float>(y) / static_cast<float>(kTileEdge - 1u);
                    const float fz = static_cast<float>(z) / static_cast<float>(kTileEdge - 1u);
                    const float value =
                        0.35F * std::sin(6.28318530718F * (fx + phase)) +
                        0.25F * std::cos(6.28318530718F * (fy - 0.5F * phase)) +
                        0.20F * std::sin(6.28318530718F * (fz + 0.25F * phase)) +
                        0.03F * noise_(rng_);
                    current_tile_->signal[index3d(kTileEdge, x, y, z)] = value;
                }
            }
        }

        current_tile_->signal[local_index(current_position_)] += 0.75F;
    }

    void op_encode_pyramid() {
        require_tile();
        pyramid_ = build_pyramid(current_tile_->signal);

        const auto& x = current_tile_->signal;
        double mean = 0.0;
        double energy = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        for (float value : x) {
            mean += value;
            energy += static_cast<double>(value) * static_cast<double>(value);
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
        }
        mean /= static_cast<double>(x.size());
        energy = std::sqrt(energy / static_cast<double>(x.size()));

        latent_[0] = static_cast<float>(mean);
        latent_[1] = static_cast<float>(energy);
        latent_[2] = minimum;
        latent_[3] = maximum;
        for (std::size_t i = 0; i < 4u; ++i) {
            latent_[4u + i] = residual_energy(pyramid_[i], pyramid_[i + 1u]);
        }
    }

    void op_contract() {
        std::array<float, kLatentDim> next{};
        for (std::size_t i = 0; i < kLatentDim; ++i) {
            const float left = latent_[(i + kLatentDim - 1u) % kLatentDim];
            const float center = latent_[i];
            const float right = latent_[(i + 1u) % kLatentDim];
            next[i] = 0.2F * left + 0.6F * center + 0.2F * right;
        }
        latent_ = next;
    }

    void op_fixed_point() {
        float relative = std::numeric_limits<float>::infinity();
        for (std::uint32_t k = 0; k < policy_.fixed_point_steps; ++k) {
            std::array<float, kLatentDim> next{};
            float numerator = 0.0F;
            float denominator = 0.0F;
            for (std::size_t i = 0; i < kLatentDim; ++i) {
                const float coupling = latent_[(i + 1u) % kLatentDim];
                const float drive = 0.55F * latent_[i] +
                                    0.20F * coupling +
                                    0.15F * omega_[i] +
                                    0.10F * theta_[i];
                next[i] = std::tanh(drive);
                const float delta = next[i] - latent_[i];
                numerator += delta * delta;
                denominator += latent_[i] * latent_[i];
            }
            latent_ = next;
            relative = std::sqrt(numerator) / (std::sqrt(denominator) + kEpsilon);
            if (relative < policy_.fixed_point_tolerance) {
                break;
            }
        }
        last_fixed_point_relative_ = relative;
    }

    void op_decode() {
        require_tile();
        if (pyramid_.size() < 4u) {
            throw std::runtime_error("decoder requires a 4x4x4 coarse pyramid level");
        }
        const PyramidLevel& coarse = pyramid_[3u];
        const std::uint32_t scale = kTileEdge / coarse.edge;

        for (std::uint32_t z = 0; z < kTileEdge; ++z) {
            for (std::uint32_t y = 0; y < kTileEdge; ++y) {
                for (std::uint32_t x = 0; x < kTileEdge; ++x) {
                    const std::size_t index = index3d(kTileEdge, x, y, z);
                    const float base = coarse.values[index3d(
                        coarse.edge, x / scale, y / scale, z / scale)];
                    const float u = static_cast<float>(index) / static_cast<float>(kTileVoxels);
                    const float harmonic = latent_[1] *
                        std::sin(6.28318530718F * u + latent_[0]);
                    const float envelope = 0.5F * (latent_[3] - latent_[2]) *
                        std::cos(3.14159265359F * u);
                    current_tile_->reconstruction[index] =
                        0.82F * base + 0.10F * harmonic + 0.08F * envelope;
                }
            }
        }
    }

    void op_compare() {
        require_tile();
        double mse = 0.0;
        std::uint64_t sparse_count = 0u;
        for (std::uint32_t i = 0; i < kTileVoxels; ++i) {
            const float error = current_tile_->signal[i] - current_tile_->reconstruction[i];
            current_tile_->residual[i] = error;
            mse += static_cast<double>(error) * static_cast<double>(error);
            if (std::fabs(error) > policy_.sparse_residual_threshold) {
                ++sparse_count;
            }
        }
        sparse_residual_voxels_ = sparse_count;
        current_error_ = static_cast<float>(mse / static_cast<double>(kTileVoxels));
        current_tile_->last_error = current_error_;
    }

    void op_update_omega() {
        const float rho = policy_.omega_decay;
        for (std::size_t i = 0; i < kLatentDim; ++i) {
            const float error_drive = current_error_ *
                (0.5F + 0.05F * static_cast<float>(i));
            omega_[i] = rho * omega_[i] +
                        (1.0F - rho) * (latent_[i] + error_drive);
        }
    }

    void op_update_theta() {
        for (std::size_t i = 0; i < kLatentDim; ++i) {
            const float proxy_gradient = current_error_ * latent_[i];
            theta_[i] -= policy_.learning_rate * proxy_gradient;
            theta_[i] = std::clamp(theta_[i], -2.0F, 2.0F);
        }
    }

    void op_optimize_runtime() {
        if (current_error_ > 0.20F) {
            policy_.fixed_point_steps = std::min<std::uint32_t>(24u, policy_.fixed_point_steps + 1u);
        } else if (current_error_ < 0.05F && policy_.fixed_point_steps > 4u) {
            --policy_.fixed_point_steps;
        }
    }

    void op_store() noexcept {
        // State is already resident in sparse VRAM. A hardware backend can replace
        // this with compression, SSD offload, remote persistence, or dirty writeback.
    }

    void op_recur() noexcept {
        // ROM recursion boundary. The next cycle resolves another 60-bit voxel.
    }

    void require_tile() const {
        if (current_tile_ == nullptr) {
            throw std::runtime_error("no active sparse VRAM tile");
        }
    }

    static std::string human_bytes(std::uint64_t bytes) {
        constexpr std::array<const char*, 5> units{{"B", "KiB", "MiB", "GiB", "TiB"}};
        double value = static_cast<double>(bytes);
        std::size_t unit = 0u;
        while (value >= 1024.0 && unit + 1u < units.size()) {
            value /= 1024.0;
            ++unit;
        }
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << value << ' ' << units[unit];
        return stream.str();
    }

    void print_banner(std::ostream& out) const {
        out << "============================================================\n"
            << " Dr Moagi 1MiB x 1MiB x 1MiB Volumetric ROM ANN\n"
            << "============================================================\n"
            << "Virtual geometry : 2^20 x 2^20 x 2^20 = 2^60 voxels\n"
            << "Virtual capacity : 1 EiB at one byte per logical voxel\n"
            << "Physical tile    : 32^3 = " << kTileVoxels << " voxels\n"
            << "Pyramid          : 32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3 -> 1\n"
            << "Active tile cap  : " << policy_.max_active_tiles << "\n"
            << "ROM pipeline     : Resolve -> Fetch -> Encode -> Contract -> Fixpoint\n"
            << "                   -> Decode -> Compare -> Omega -> Theta -> Pi -> Recur\n\n";
    }
};

} // namespace jarvisx::volumetric_rom
