#pragma once

#include "jarvisx/intelligence_vm3d.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jarvisx::world {

constexpr std::uint64_t kVmadCoordExtent = 1ULL << 33U;
constexpr std::uint64_t kVmadCoordMask = kVmadCoordExtent - 1ULL;
constexpr std::size_t kVectorRegisterCount = 512U;
constexpr std::size_t kVectorBytes = 64U;
constexpr std::size_t kScalarRegisterCount = 512U;
constexpr std::size_t kVmadRegisterCount = 32U;
constexpr std::size_t kMaxTransferBytes = 4096U;
constexpr std::size_t kDefaultTileBytes = 1024U;

struct Vmad128 {
    std::uint64_t hi{};
    std::uint64_t lo{};

    static Vmad128 pack(std::uint16_t region, std::uint8_t modality, std::uint16_t attributes,
                        std::uint64_t x, std::uint64_t y, std::uint64_t z) {
        if (region >= 4096U) throw std::out_of_range("VMAD region exceeds 12 bits");
        if (attributes >= 512U) throw std::out_of_range("VMAD attributes exceed 9 bits");
        if (x >= kVmadCoordExtent || y >= kVmadCoordExtent || z >= kVmadCoordExtent) {
            throw std::out_of_range("VMAD coordinate exceeds 33-bit domain");
        }
        Vmad128 value;
        const std::uint64_t y_low31 = y & ((1ULL << 31U) - 1ULL);
        const std::uint64_t y_high2 = y >> 31U;
        value.lo = z | (y_low31 << 33U);
        value.hi = y_high2 |
                   (x << 2U) |
                   (static_cast<std::uint64_t>(attributes) << 35U) |
                   (static_cast<std::uint64_t>(modality) << 44U) |
                   (static_cast<std::uint64_t>(region) << 52U);
        return value;
    }

    std::uint16_t region() const noexcept {
        return static_cast<std::uint16_t>((hi >> 52U) & 0x0fffULL);
    }

    std::uint8_t modality() const noexcept {
        return static_cast<std::uint8_t>((hi >> 44U) & 0xffULL);
    }

    std::uint16_t attributes() const noexcept {
        return static_cast<std::uint16_t>((hi >> 35U) & 0x01ffULL);
    }

    std::uint64_t x() const noexcept { return (hi >> 2U) & kVmadCoordMask; }

    std::uint64_t y() const noexcept {
        const std::uint64_t low31 = (lo >> 33U) & ((1ULL << 31U) - 1ULL);
        const std::uint64_t high2 = hi & 0x3ULL;
        return low31 | (high2 << 31U);
    }

    std::uint64_t z() const noexcept { return lo & kVmadCoordMask; }

    intelligence3d::Coord3 coord() const noexcept { return {x(), y(), z()}; }

    bool operator==(const Vmad128& other) const noexcept {
        return hi == other.hi && lo == other.lo;
    }
};

inline std::uint64_t wrap_coord(std::uint64_t base, std::int64_t delta) noexcept {
    constexpr std::int64_t extent = static_cast<std::int64_t>(kVmadCoordExtent);
    const std::int64_t reduced = delta % extent;
    std::int64_t result = static_cast<std::int64_t>(base) + reduced;
    result %= extent;
    if (result < 0) result += extent;
    return static_cast<std::uint64_t>(result);
}

inline Vmad128 vmad_offset(const Vmad128& base, std::int64_t dx, std::int64_t dy, std::int64_t dz) {
    return Vmad128::pack(base.region(), base.modality(), base.attributes(),
                         wrap_coord(base.x(), dx), wrap_coord(base.y(), dy), wrap_coord(base.z(), dz));
}

inline Vmad128 vmad_advance_linear(const Vmad128& base, std::uint64_t bytes) {
    const std::uint64_t z_total = base.z() + bytes;
    const std::uint64_t z = z_total & kVmadCoordMask;
    const std::uint64_t carry_y = z_total >> 33U;
    const std::uint64_t y_total = base.y() + carry_y;
    const std::uint64_t y = y_total & kVmadCoordMask;
    const std::uint64_t carry_x = y_total >> 33U;
    const std::uint64_t x = (base.x() + carry_x) & kVmadCoordMask;
    return Vmad128::pack(base.region(), base.modality(), base.attributes(), x, y, z);
}

enum class MicroOpcode : std::uint8_t {
    Nop = 0x00,
    LoadVmad = 0x01,
    TileInVec = 0x10,
    StoreVec = 0x11,
    EncLatVol = 0x20,
    FuseAttn = 0x30,
    DecPixVol = 0x40,
    CalcDelta = 0x50,
    ProposeBias = 0x60,
    Validate = 0x70,
    CommitIf = 0x71,
    Halt = 0xff,
};

enum class KineticStage : std::uint8_t {
    Ingest = 0U,
    Reduce = 1U,
    Fuse = 2U,
    Reconstruct = 3U,
    Feedback = 4U,
};

struct MicroOp {
    MicroOpcode opcode{MicroOpcode::Nop};
    std::uint16_t dst{};
    std::uint16_t src0{};
    std::uint16_t src1{};
    std::uint8_t vmad_reg{};
    std::uint32_t imm24{};

    std::uint64_t encode() const {
        if (dst >= 512U || src0 >= 512U || src1 >= 512U) {
            throw std::out_of_range("micro-op register field exceeds 9 bits");
        }
        if (vmad_reg >= 32U) throw std::out_of_range("micro-op VMAD field exceeds 5 bits");
        if (imm24 >= (1U << 24U)) throw std::out_of_range("micro-op immediate exceeds 24 bits");
        return (static_cast<std::uint64_t>(static_cast<std::uint8_t>(opcode)) << 56U) |
               (static_cast<std::uint64_t>(dst) << 47U) |
               (static_cast<std::uint64_t>(src0) << 38U) |
               (static_cast<std::uint64_t>(src1) << 29U) |
               (static_cast<std::uint64_t>(vmad_reg) << 24U) |
               static_cast<std::uint64_t>(imm24);
    }

    static MicroOp decode(std::uint64_t word) noexcept {
        MicroOp op;
        op.opcode = static_cast<MicroOpcode>(static_cast<std::uint8_t>((word >> 56U) & 0xffULL));
        op.dst = static_cast<std::uint16_t>((word >> 47U) & 0x01ffULL);
        op.src0 = static_cast<std::uint16_t>((word >> 38U) & 0x01ffULL);
        op.src1 = static_cast<std::uint16_t>((word >> 29U) & 0x01ffULL);
        op.vmad_reg = static_cast<std::uint8_t>((word >> 24U) & 0x1fULL);
        op.imm24 = static_cast<std::uint32_t>(word & 0x00ffffffULL);
        return op;
    }
};

struct WorldProgram {
    std::vector<Vmad128> descriptors;
    std::vector<std::uint64_t> words;
};

struct KineticStats {
    std::uint64_t issued_micro_ops{};
    std::uint64_t logical_issue_cycles{};
    std::uint64_t estimated_pipeline_latency_cycles{};
    std::uint64_t bytes_ingested{};
    std::uint64_t bytes_stored{};
    std::uint64_t commits{};
    std::uint64_t rollbacks{};
    std::array<std::uint64_t, 5U> stage_issues{};
};

class WorldEngine128 {
public:
    using VectorRegister = std::array<std::uint8_t, kVectorBytes>;

    explicit WorldEngine128(intelligence3d::VirtualVolume3D& volume) : volume_(volume) {
        if (volume_.axis_extent() < kVmadCoordExtent) {
            throw std::invalid_argument("WorldEngine128 requires a virtual volume with at least 2^33 coordinates per axis");
        }
    }

    void reset() noexcept {
        for (auto& reg : vectors_) reg.fill(0U);
        scalars_.fill(0ULL);
        vmads_.fill(Vmad128{});
        authoritative_bias_.fill(0);
        candidate_bias_.fill(0);
        candidate_valid_ = false;
        last_delta_mean_ = 0U;
        stats_ = KineticStats{};
    }

    const VectorRegister& vector_register(std::size_t index) const {
        if (index >= vectors_.size()) throw std::out_of_range("vector register out of range");
        return vectors_[index];
    }

    void set_vector_register(std::size_t index, const VectorRegister& value) {
        if (index >= vectors_.size()) throw std::out_of_range("vector register out of range");
        vectors_[index] = value;
    }

    std::uint64_t scalar_register(std::size_t index) const {
        if (index >= scalars_.size()) throw std::out_of_range("scalar register out of range");
        return scalars_[index];
    }

    const std::array<std::int16_t, kVectorBytes>& biases() const noexcept { return authoritative_bias_; }
    const KineticStats& stats() const noexcept { return stats_; }
    std::uint32_t last_delta_mean() const noexcept { return last_delta_mean_; }

    void run(const WorldProgram& program, std::uint64_t max_steps = 100000ULL) {
        if (program.words.empty()) throw std::invalid_argument("world program must not be empty");
        if (max_steps == 0ULL) throw std::invalid_argument("max_steps must be positive");
        std::uint64_t steps = 0ULL;
        for (const std::uint64_t word : program.words) {
            if (++steps > max_steps) throw std::runtime_error("world program exceeded max_steps");
            const MicroOp op = MicroOp::decode(word);
            ++stats_.issued_micro_ops;
            ++stats_.logical_issue_cycles;
            if (execute(op, program)) break;
        }
        stats_.estimated_pipeline_latency_cycles =
            stats_.logical_issue_cycles == 0ULL ? 0ULL : stats_.logical_issue_cycles + 4ULL;
    }

private:
    intelligence3d::VirtualVolume3D& volume_;
    std::array<VectorRegister, kVectorRegisterCount> vectors_{};
    std::array<std::uint64_t, kScalarRegisterCount> scalars_{};
    std::array<Vmad128, kVmadRegisterCount> vmads_{};
    std::array<std::int16_t, kVectorBytes> authoritative_bias_{};
    std::array<std::int16_t, kVectorBytes> candidate_bias_{};
    bool candidate_valid_{};
    std::uint32_t last_delta_mean_{};
    KineticStats stats_{};

    static std::size_t transfer_size(std::uint32_t imm24) {
        const std::size_t bytes = imm24 == 0U ? kDefaultTileBytes : static_cast<std::size_t>(imm24);
        if (bytes == 0U || bytes > kMaxTransferBytes) {
            throw std::runtime_error("world-engine transfer exceeds bounded payload size");
        }
        return bytes;
    }

    static std::size_t register_span(std::size_t bytes) noexcept {
        return (bytes + kVectorBytes - 1U) / kVectorBytes;
    }

    static void validate_vector_span(std::uint16_t base, std::size_t bytes) {
        const std::size_t span = register_span(bytes);
        if (static_cast<std::size_t>(base) + span > kVectorRegisterCount) {
            throw std::runtime_error("vector register span exceeds V0-V511");
        }
    }

    static void validate_vector(std::uint16_t index) {
        if (index >= kVectorRegisterCount) throw std::runtime_error("vector register exceeds V0-V511");
    }

    static void validate_scalar(std::uint16_t index) {
        if (index >= kScalarRegisterCount) throw std::runtime_error("scalar register exceeds R0-R511");
    }

    static std::uint8_t clamp_byte(std::int64_t value) noexcept {
        return static_cast<std::uint8_t>(std::clamp<std::int64_t>(value, 0LL, 255LL));
    }

    static std::int16_t clamp_bias(std::int64_t value) noexcept {
        return static_cast<std::int16_t>(std::clamp<std::int64_t>(value, -127LL, 127LL));
    }

    std::uint8_t vector_byte(std::uint16_t base, std::size_t offset) const {
        const std::size_t reg = static_cast<std::size_t>(base) + offset / kVectorBytes;
        return vectors_[reg][offset % kVectorBytes];
    }

    void set_vector_byte(std::uint16_t base, std::size_t offset, std::uint8_t value) {
        const std::size_t reg = static_cast<std::size_t>(base) + offset / kVectorBytes;
        vectors_[reg][offset % kVectorBytes] = value;
    }

    intelligence3d::Coord3 checked_coord(const Vmad128& address) const {
        const intelligence3d::Coord3 coord = address.coord();
        if (coord.x >= volume_.axis_extent() || coord.y >= volume_.axis_extent() ||
            coord.z >= volume_.axis_extent()) {
            throw std::runtime_error("VMAD lies outside configured sparse volume");
        }
        return coord;
    }

    void note_stage(KineticStage stage) noexcept {
        ++stats_.stage_issues[static_cast<std::size_t>(stage)];
    }

    bool execute(const MicroOp& op, const WorldProgram& program) {
        switch (op.opcode) {
            case MicroOpcode::Nop:
                return false;
            case MicroOpcode::LoadVmad:
                if (op.vmad_reg >= kVmadRegisterCount) throw std::runtime_error("VMAD register exceeds A0-A31");
                if (op.imm24 >= program.descriptors.size()) throw std::runtime_error("VMAD descriptor index out of range");
                vmads_[op.vmad_reg] = program.descriptors[op.imm24];
                return false;
            case MicroOpcode::TileInVec:
                ingest(op);
                return false;
            case MicroOpcode::StoreVec:
                store(op);
                return false;
            case MicroOpcode::EncLatVol:
                encode_latent(op);
                return false;
            case MicroOpcode::FuseAttn:
                fuse_attention(op);
                return false;
            case MicroOpcode::DecPixVol:
                decode_expand(op);
                return false;
            case MicroOpcode::CalcDelta:
                calculate_delta(op);
                return false;
            case MicroOpcode::ProposeBias:
                propose_bias(op);
                return false;
            case MicroOpcode::Validate:
                validate_candidate(op);
                return false;
            case MicroOpcode::CommitIf:
                commit_if(op);
                return false;
            case MicroOpcode::Halt:
                return true;
            default:
                throw std::runtime_error("unknown world-engine micro-opcode");
        }
    }

    void ingest(const MicroOp& op) {
        if (op.vmad_reg >= kVmadRegisterCount) throw std::runtime_error("VMAD register exceeds A0-A31");
        const std::size_t bytes = transfer_size(op.imm24);
        validate_vector_span(op.dst, bytes);
        const Vmad128 base = vmads_[op.vmad_reg];
        for (std::size_t i = 0U; i < bytes; ++i) {
            const Vmad128 address = vmad_advance_linear(base, static_cast<std::uint64_t>(i));
            set_vector_byte(op.dst, i, volume_.read(checked_coord(address)));
        }
        stats_.bytes_ingested += static_cast<std::uint64_t>(bytes);
        note_stage(KineticStage::Ingest);
    }

    void store(const MicroOp& op) {
        if (op.vmad_reg >= kVmadRegisterCount) throw std::runtime_error("VMAD register exceeds A0-A31");
        const std::size_t bytes = transfer_size(op.imm24);
        validate_vector_span(op.src0, bytes);
        const Vmad128 base = vmads_[op.vmad_reg];
        for (std::size_t i = 0U; i < bytes; ++i) {
            const Vmad128 address = vmad_advance_linear(base, static_cast<std::uint64_t>(i));
            volume_.write(checked_coord(address), vector_byte(op.src0, i));
        }
        stats_.bytes_stored += static_cast<std::uint64_t>(bytes);
        note_stage(KineticStage::Reconstruct);
    }

    void encode_latent(const MicroOp& op) {
        const std::size_t bytes = transfer_size(op.imm24);
        validate_vector_span(op.src0, bytes);
        validate_vector(op.dst);
        vectors_[op.dst].fill(0U);
        for (std::size_t latent = 0U; latent < 32U; ++latent) {
            std::uint64_t sum = 0ULL;
            std::size_t count = 0U;
            for (std::size_t i = latent; i < bytes; i += 32U) {
                sum += static_cast<std::uint64_t>(vector_byte(op.src0, i));
                ++count;
            }
            const std::int64_t average = count == 0U ? 0LL :
                static_cast<std::int64_t>((sum + static_cast<std::uint64_t>(count / 2U)) /
                                          static_cast<std::uint64_t>(count));
            vectors_[op.dst][latent] = clamp_byte(average + authoritative_bias_[latent]);
        }
        note_stage(KineticStage::Reduce);
    }

    void fuse_attention(const MicroOp& op) {
        validate_vector(op.dst);
        validate_vector(op.src0);
        validate_vector(op.src1);
        std::int64_t dot = 0LL;
        for (std::size_t i = 0U; i < kVectorBytes; ++i) {
            const std::int64_t a = static_cast<std::int64_t>(vectors_[op.src0][i]) - 128LL;
            const std::int64_t b = static_cast<std::int64_t>(vectors_[op.src1][i]) - 128LL;
            dot += a * b;
        }
        const std::int64_t alpha64 = std::clamp<std::int64_t>(128LL + dot / 16384LL, 32LL, 223LL);
        const std::uint32_t alpha = static_cast<std::uint32_t>(alpha64);
        for (std::size_t i = 0U; i < kVectorBytes; ++i) {
            const std::uint32_t a = vectors_[op.src0][i];
            const std::uint32_t b = vectors_[op.src1][i];
            const std::uint32_t mixed = alpha * a + (255U - alpha) * b + 127U;
            vectors_[op.dst][i] = static_cast<std::uint8_t>(mixed / 255U);
        }
        note_stage(KineticStage::Fuse);
    }

    void decode_expand(const MicroOp& op) {
        if (op.vmad_reg >= kVmadRegisterCount) throw std::runtime_error("VMAD register exceeds A0-A31");
        const std::size_t bytes = transfer_size(op.imm24);
        validate_vector(op.src0);
        validate_vector_span(op.dst, bytes);
        const Vmad128 base = vmads_[op.vmad_reg];
        for (std::size_t i = 0U; i < bytes; ++i) {
            const std::size_t latent = i % 32U;
            const std::size_t next = (latent + 1U) % 32U;
            const std::uint32_t expanded =
                (3U * static_cast<std::uint32_t>(vectors_[op.src0][latent]) +
                 static_cast<std::uint32_t>(vectors_[op.src0][next]) + 2U) / 4U;
            const std::uint8_t value = static_cast<std::uint8_t>(expanded);
            set_vector_byte(op.dst, i, value);
            const Vmad128 address = vmad_advance_linear(base, static_cast<std::uint64_t>(i));
            volume_.write(checked_coord(address), value);
        }
        stats_.bytes_stored += static_cast<std::uint64_t>(bytes);
        note_stage(KineticStage::Reconstruct);
    }

    void calculate_delta(const MicroOp& op) {
        const std::size_t bytes = transfer_size(op.imm24);
        validate_vector_span(op.src0, bytes);
        validate_vector_span(op.src1, bytes);
        validate_vector_span(op.dst, bytes);
        std::uint64_t score = 0ULL;
        for (std::size_t i = 0U; i < bytes; ++i) {
            const std::int64_t source = static_cast<std::int64_t>(vector_byte(op.src0, i));
            const std::int64_t reconstructed = static_cast<std::int64_t>(vector_byte(op.src1, i));
            const std::int64_t delta = source - reconstructed;
            score += static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
            const std::int64_t bounded = std::clamp<std::int64_t>(delta, -127LL, 127LL);
            set_vector_byte(op.dst, i, static_cast<std::uint8_t>(bounded + 128LL));
        }
        last_delta_mean_ = static_cast<std::uint32_t>(score / static_cast<std::uint64_t>(bytes));
        note_stage(KineticStage::Feedback);
    }

    void propose_bias(const MicroOp& op) {
        const std::size_t bytes = transfer_size(op.imm24);
        validate_vector_span(op.src0, bytes);
        for (std::size_t lane = 0U; lane < kVectorBytes; ++lane) {
            std::int64_t sum = 0LL;
            std::size_t count = 0U;
            for (std::size_t i = lane; i < bytes; i += kVectorBytes) {
                sum += static_cast<std::int64_t>(vector_byte(op.src0, i)) - 128LL;
                ++count;
            }
            const std::int64_t mean = count == 0U ? 0LL : sum / static_cast<std::int64_t>(count);
            candidate_bias_[lane] = clamp_bias(static_cast<std::int64_t>(authoritative_bias_[lane]) + mean / 8LL);
        }
        candidate_valid_ = true;
        note_stage(KineticStage::Feedback);
    }

    void validate_candidate(const MicroOp& op) {
        validate_scalar(op.dst);
        const std::uint32_t threshold = op.imm24;
        scalars_[op.dst] = candidate_valid_ && last_delta_mean_ <= threshold ? 1ULL : 0ULL;
        note_stage(KineticStage::Feedback);
    }

    void commit_if(const MicroOp& op) {
        validate_scalar(op.src0);
        const bool accept = scalars_[op.src0] != 0ULL && candidate_valid_;
        if (accept) {
            authoritative_bias_ = candidate_bias_;
            ++stats_.commits;
        } else {
            ++stats_.rollbacks;
        }
        candidate_bias_ = authoritative_bias_;
        candidate_valid_ = false;
        note_stage(KineticStage::Feedback);
    }
};

inline WorldProgram make_world_demo_program(const Vmad128& source, const Vmad128& output) {
    WorldProgram program;
    program.descriptors = {source, output};
    const auto emit = [&](MicroOpcode opcode, std::uint16_t dst, std::uint16_t src0,
                          std::uint16_t src1, std::uint8_t address_reg, std::uint32_t imm24) {
        program.words.push_back(MicroOp{opcode, dst, src0, src1, address_reg, imm24}.encode());
    };
    emit(MicroOpcode::LoadVmad, 0U, 0U, 0U, 0U, 0U);
    emit(MicroOpcode::LoadVmad, 0U, 0U, 0U, 1U, 1U);
    emit(MicroOpcode::TileInVec, 2U, 0U, 0U, 0U, 1024U);
    emit(MicroOpcode::EncLatVol, 32U, 2U, 0U, 0U, 1024U);
    emit(MicroOpcode::FuseAttn, 40U, 32U, 32U, 0U, 0U);
    emit(MicroOpcode::DecPixVol, 64U, 40U, 0U, 1U, 1024U);
    emit(MicroOpcode::CalcDelta, 96U, 2U, 64U, 0U, 1024U);
    emit(MicroOpcode::ProposeBias, 0U, 96U, 0U, 0U, 1024U);
    emit(MicroOpcode::Validate, 20U, 0U, 0U, 0U, 255U);
    emit(MicroOpcode::CommitIf, 0U, 20U, 0U, 0U, 0U);
    emit(MicroOpcode::Halt, 0U, 0U, 0U, 0U, 0U);
    return program;
}

} // namespace jarvisx::world
