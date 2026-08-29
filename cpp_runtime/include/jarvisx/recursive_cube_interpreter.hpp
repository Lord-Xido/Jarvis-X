#pragma once

#include "jarvisx/world_engine_vmad.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace jarvisx::cube {

constexpr std::array<std::uint8_t, 8U> kCubeMagic{{'D','M','C','U','B','E','1',0U}};
constexpr std::uint16_t kCubeVersion = 1U;
constexpr std::size_t kCubeHeaderBytes = 16U;
constexpr std::size_t kCubeCommandBytes = 96U;
constexpr std::size_t kCubeDigestBytes = 8U;
constexpr std::size_t kCubeTileBytes = world::kDefaultTileBytes;
constexpr std::size_t kCubeLatentBytes = 32U;

inline std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t size) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0U; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

enum class CubeOpcode : std::uint8_t {
    EncodeRefine = 0x10,
    Decode = 0x20,
    Halt = 0xff,
};

struct CubeCommand {
    CubeOpcode opcode{CubeOpcode::Halt};
    std::uint8_t level{};
    std::uint16_t flags{};
    std::uint32_t tile_count{};
    std::uint32_t max_passes{};
    std::uint32_t epsilon{};
    world::Vmad128 source{};
    world::Vmad128 latent{};
    world::Vmad128 output{};
    world::Vmad128 shadow_latent{};
    world::Vmad128 shadow_output{};
};

struct CubeInterpreterConfig {
    std::uint64_t logical_extent{1000ULL * 1000ULL * 1000ULL};
    std::uint32_t max_commands{4096U};
    std::uint32_t max_tiles_per_command{1U << 20U};
    std::uint32_t max_passes_per_tile{64U};
    std::uint64_t max_total_tile_ops{1ULL << 22U};

    void validate() const {
        if (logical_extent == 0ULL || logical_extent > world::kVmadCoordExtent) {
            throw std::invalid_argument("cube logical extent must lie in [1, 2^33]");
        }
        if (max_commands == 0U || max_tiles_per_command == 0U || max_passes_per_tile == 0U ||
            max_total_tile_ops == 0ULL) {
            throw std::invalid_argument("cube interpreter limits must be positive");
        }
    }
};

class ByteWriter {
public:
    void u8(std::uint8_t value) { bytes_.push_back(value); }
    void u16(std::uint16_t value) {
        u8(static_cast<std::uint8_t>(value & 0xffU));
        u8(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
    }
    void u32(std::uint32_t value) {
        for (unsigned i = 0U; i < 4U; ++i) u8(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffU));
    }
    void u64(std::uint64_t value) {
        for (unsigned i = 0U; i < 8U; ++i) u8(static_cast<std::uint8_t>((value >> (8U * i)) & 0xffULL));
    }
    void vmad(const world::Vmad128& value) { u64(value.hi); u64(value.lo); }
    void append(const std::uint8_t* data, std::size_t size) {
        bytes_.insert(bytes_.end(), data, data + static_cast<std::ptrdiff_t>(size));
    }
    void append(const std::vector<std::uint8_t>& data) { bytes_.insert(bytes_.end(), data.begin(), data.end()); }
    std::vector<std::uint8_t> take() { return std::move(bytes_); }
private:
    std::vector<std::uint8_t> bytes_;
};

class ByteReader {
public:
    explicit ByteReader(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}
    std::uint8_t u8() { require(1U); return bytes_[offset_++]; }
    std::uint16_t u16() {
        const std::uint16_t a = u8();
        const std::uint16_t b = u8();
        return static_cast<std::uint16_t>(a | static_cast<std::uint16_t>(b << 8U));
    }
    std::uint32_t u32() {
        std::uint32_t value = 0U;
        for (unsigned i = 0U; i < 4U; ++i) value |= static_cast<std::uint32_t>(u8()) << (8U * i);
        return value;
    }
    std::uint64_t u64() {
        std::uint64_t value = 0ULL;
        for (unsigned i = 0U; i < 8U; ++i) value |= static_cast<std::uint64_t>(u8()) << (8U * i);
        return value;
    }
    world::Vmad128 vmad() { return world::Vmad128{u64(), u64()}; }
    std::size_t remaining() const noexcept { return bytes_.size() - offset_; }
private:
    const std::vector<std::uint8_t>& bytes_;
    std::size_t offset_{};
    void require(std::size_t count) const {
        if (count > bytes_.size() - offset_) throw std::runtime_error("truncated recursive-cube execution buffer");
    }
};

inline void append_command(ByteWriter& writer, const CubeCommand& command) {
    writer.u8(static_cast<std::uint8_t>(command.opcode));
    writer.u8(command.level);
    writer.u16(command.flags);
    writer.u32(command.tile_count);
    writer.u32(command.max_passes);
    writer.u32(command.epsilon);
    writer.vmad(command.source);
    writer.vmad(command.latent);
    writer.vmad(command.output);
    writer.vmad(command.shadow_latent);
    writer.vmad(command.shadow_output);
}

inline CubeCommand read_command(ByteReader& reader) {
    CubeCommand command;
    command.opcode = static_cast<CubeOpcode>(reader.u8());
    command.level = reader.u8();
    command.flags = reader.u16();
    command.tile_count = reader.u32();
    command.max_passes = reader.u32();
    command.epsilon = reader.u32();
    command.source = reader.vmad();
    command.latent = reader.vmad();
    command.output = reader.vmad();
    command.shadow_latent = reader.vmad();
    command.shadow_output = reader.vmad();
    return command;
}

inline std::vector<std::uint8_t> serialize_execution_buffer(const std::vector<CubeCommand>& commands) {
    if (commands.empty() || commands.size() > static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())) {
        throw std::invalid_argument("recursive-cube command count must fit uint16 and be non-zero");
    }
    ByteWriter writer;
    writer.append(kCubeMagic.data(), kCubeMagic.size());
    writer.u16(kCubeVersion);
    writer.u16(static_cast<std::uint16_t>(commands.size()));
    writer.u32(0U);
    for (const auto& command : commands) append_command(writer, command);
    std::vector<std::uint8_t> body = writer.take();
    const std::uint64_t digest = fnv1a64(body.data(), body.size());
    ByteWriter complete;
    complete.append(body);
    complete.u64(digest);
    return complete.take();
}

inline bool same_line_overlap(const world::Vmad128& a, std::uint64_t a_bytes,
                              const world::Vmad128& b, std::uint64_t b_bytes) noexcept {
    if (a.x() != b.x() || a.y() != b.y()) return false;
    const std::uint64_t a_end = a.z() + a_bytes;
    const std::uint64_t b_end = b.z() + b_bytes;
    return a.z() < b_end && b.z() < a_end;
}

inline void validate_span(const world::Vmad128& base, std::uint64_t bytes,
                          const CubeInterpreterConfig& config) {
    if (base.x() >= config.logical_extent || base.y() >= config.logical_extent || base.z() >= config.logical_extent) {
        throw std::runtime_error("cube VMAD lies outside configured logical extent");
    }
    if (bytes > config.logical_extent || base.z() > config.logical_extent - bytes) {
        throw std::runtime_error("cube VMAD span crosses configured logical extent");
    }
}

inline std::vector<CubeCommand> parse_execution_buffer(const std::vector<std::uint8_t>& bytes,
                                                        const CubeInterpreterConfig& config) {
    config.validate();
    if (bytes.size() < kCubeHeaderBytes + kCubeCommandBytes + kCubeDigestBytes) {
        throw std::runtime_error("recursive-cube execution buffer too small");
    }
    const std::size_t payload_size = bytes.size() - kCubeDigestBytes;
    std::uint64_t expected_digest = 0ULL;
    for (std::size_t i = 0U; i < 8U; ++i) {
        expected_digest |= static_cast<std::uint64_t>(bytes[payload_size + i]) << (8U * i);
    }
    if (fnv1a64(bytes.data(), payload_size) != expected_digest) {
        throw std::runtime_error("recursive-cube execution buffer digest mismatch");
    }
    std::vector<std::uint8_t> payload(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(payload_size));
    ByteReader reader(payload);
    for (const auto expected : kCubeMagic) {
        if (reader.u8() != expected) throw std::runtime_error("invalid recursive-cube execution-buffer magic");
    }
    if (reader.u16() != kCubeVersion) throw std::runtime_error("unsupported recursive-cube execution-buffer version");
    const std::uint16_t count = reader.u16();
    if (reader.u32() != 0U) throw std::runtime_error("recursive-cube header reserved field must be zero");
    if (count == 0U || count > config.max_commands) throw std::runtime_error("recursive-cube command count exceeds limit");
    const std::size_t expected_size = kCubeHeaderBytes + static_cast<std::size_t>(count) * kCubeCommandBytes;
    if (payload.size() != expected_size) throw std::runtime_error("recursive-cube execution-buffer size mismatch");

    std::vector<CubeCommand> commands;
    commands.reserve(count);
    std::uint64_t total_tile_ops = 0ULL;
    bool halted = false;
    for (std::uint16_t index = 0U; index < count; ++index) {
        CubeCommand command = read_command(reader);
        if (halted) throw std::runtime_error("recursive-cube command appears after HALT");
        if (command.flags != 0U) throw std::runtime_error("recursive-cube command flags are reserved");
        switch (command.opcode) {
            case CubeOpcode::EncodeRefine: {
                if (command.tile_count == 0U || command.tile_count > config.max_tiles_per_command) {
                    throw std::runtime_error("recursive-cube encode tile count exceeds limit");
                }
                if (command.max_passes == 0U || command.max_passes > config.max_passes_per_tile) {
                    throw std::runtime_error("recursive-cube refinement pass count exceeds limit");
                }
                if (command.epsilon > 255U) throw std::runtime_error("recursive-cube epsilon exceeds byte delta range");
                const std::uint64_t tile_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeTileBytes;
                const std::uint64_t latent_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeLatentBytes;
                validate_span(command.source, tile_bytes, config);
                validate_span(command.latent, latent_bytes, config);
                validate_span(command.output, tile_bytes, config);
                validate_span(command.shadow_latent, latent_bytes, config);
                validate_span(command.shadow_output, tile_bytes, config);
                const std::array<std::pair<world::Vmad128, std::uint64_t>, 5U> spans{{
                    {command.source, tile_bytes}, {command.latent, latent_bytes}, {command.output, tile_bytes},
                    {command.shadow_latent, latent_bytes}, {command.shadow_output, tile_bytes},
                }};
                for (std::size_t a = 0U; a < spans.size(); ++a) {
                    for (std::size_t b = a + 1U; b < spans.size(); ++b) {
                        if (same_line_overlap(spans[a].first, spans[a].second, spans[b].first, spans[b].second)) {
                            throw std::runtime_error("recursive-cube encode spans overlap in sparse world state");
                        }
                    }
                }
                total_tile_ops += static_cast<std::uint64_t>(command.tile_count) * command.max_passes;
                break;
            }
            case CubeOpcode::Decode: {
                if (command.tile_count == 0U || command.tile_count > config.max_tiles_per_command) {
                    throw std::runtime_error("recursive-cube decode tile count exceeds limit");
                }
                const std::uint64_t source_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeLatentBytes;
                const std::uint64_t output_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeTileBytes;
                validate_span(command.source, source_bytes, config);
                validate_span(command.output, output_bytes, config);
                if (same_line_overlap(command.source, source_bytes, command.output, output_bytes)) {
                    throw std::runtime_error("recursive-cube decode source/output overlap");
                }
                total_tile_ops += command.tile_count;
                break;
            }
            case CubeOpcode::Halt:
                if (index + 1U != count) throw std::runtime_error("recursive-cube HALT must be final command");
                halted = true;
                break;
            default:
                throw std::runtime_error("unknown recursive-cube opcode");
        }
        if (total_tile_ops > config.max_total_tile_ops) {
            throw std::runtime_error("recursive-cube execution exceeds total tile-operation limit");
        }
        commands.push_back(command);
    }
    if (!halted) throw std::runtime_error("recursive-cube execution buffer missing HALT");
    if (reader.remaining() != 0U) throw std::runtime_error("recursive-cube parser left trailing payload");
    return commands;
}

struct CubeCommandMetrics {
    CubeOpcode opcode{CubeOpcode::Halt};
    std::uint8_t level{};
    std::uint64_t tiles{};
    std::uint64_t passes{};
    std::uint64_t accepted_passes{};
    std::uint64_t rejected_passes{};
    std::uint64_t converged_tiles{};
    std::uint64_t input_bytes{};
    std::uint64_t latent_bytes{};
    std::uint64_t output_bytes{};
    double mean_final_delta{};
};

struct CubeRunMetrics {
    bool execution_buffer_validated{};
    std::uint64_t commands_executed{};
    std::uint64_t tiles_processed{};
    std::uint64_t total_passes{};
    std::uint64_t accepted_passes{};
    std::uint64_t rejected_passes{};
    std::uint64_t converged_tiles{};
    std::uint64_t encoded_input_bytes{};
    std::uint64_t latent_bytes_committed{};
    std::uint64_t decoded_output_bytes{};
    std::uint64_t aggregate_command_input_bytes{};
    std::uint64_t aggregate_command_output_bytes{};
    std::vector<CubeCommandMetrics> commands;
};

class RecursiveCubeInterpreter {
public:
    RecursiveCubeInterpreter(intelligence3d::VirtualVolume3D& volume,
                             CubeInterpreterConfig config = {})
        : volume_(volume), engine_(volume), config_(config) {
        config_.validate();
        if (volume_.axis_extent() < world::kVmadCoordExtent) {
            throw std::invalid_argument("recursive cube requires VMAD128-compatible virtual volume");
        }
    }

    CubeRunMetrics run(const std::vector<std::uint8_t>& execution_buffer) {
        const std::vector<CubeCommand> commands = parse_execution_buffer(execution_buffer, config_);
        CubeRunMetrics metrics;
        metrics.execution_buffer_validated = true;
        for (const auto& command : commands) {
            if (command.opcode == CubeOpcode::Halt) break;
            CubeCommandMetrics local;
            local.opcode = command.opcode;
            local.level = command.level;
            local.tiles = command.tile_count;
            if (command.opcode == CubeOpcode::EncodeRefine) execute_encode(command, local);
            else execute_decode(command, local);
            ++metrics.commands_executed;
            metrics.tiles_processed += local.tiles;
            metrics.total_passes += local.passes;
            metrics.accepted_passes += local.accepted_passes;
            metrics.rejected_passes += local.rejected_passes;
            metrics.converged_tiles += local.converged_tiles;
            metrics.aggregate_command_input_bytes += local.input_bytes;
            metrics.aggregate_command_output_bytes += local.output_bytes;
            if (command.opcode == CubeOpcode::EncodeRefine) {
                metrics.encoded_input_bytes += local.input_bytes;
                metrics.latent_bytes_committed += local.latent_bytes;
            } else if (command.opcode == CubeOpcode::Decode) {
                metrics.decoded_output_bytes += local.output_bytes;
            }
            metrics.commands.push_back(local);
        }
        return metrics;
    }

    const world::WorldEngine128& engine() const noexcept { return engine_; }

private:
    intelligence3d::VirtualVolume3D& volume_;
    world::WorldEngine128 engine_;
    CubeInterpreterConfig config_;

    static void emit(world::WorldProgram& program, world::MicroOpcode opcode, std::uint16_t dst,
                     std::uint16_t src0, std::uint16_t src1, std::uint8_t vmad_reg, std::uint32_t imm24) {
        program.words.push_back(world::MicroOp{opcode, dst, src0, src1, vmad_reg, imm24}.encode());
    }

    world::WorldProgram candidate_program(const world::Vmad128& source,
                                          const world::Vmad128& shadow_latent,
                                          const world::Vmad128& shadow_output,
                                          std::uint32_t threshold) const {
        world::WorldProgram program;
        program.descriptors = {source, shadow_latent, shadow_output};
        emit(program, world::MicroOpcode::LoadVmad, 0U, 0U, 0U, 0U, 0U);
        emit(program, world::MicroOpcode::LoadVmad, 0U, 0U, 0U, 1U, 1U);
        emit(program, world::MicroOpcode::LoadVmad, 0U, 0U, 0U, 2U, 2U);
        emit(program, world::MicroOpcode::TileInVec, 2U, 0U, 0U, 0U, static_cast<std::uint32_t>(kCubeTileBytes));
        emit(program, world::MicroOpcode::EncLatVol, 32U, 2U, 0U, 0U, static_cast<std::uint32_t>(kCubeTileBytes));
        emit(program, world::MicroOpcode::StoreVec, 0U, 32U, 0U, 1U, static_cast<std::uint32_t>(kCubeLatentBytes));
        emit(program, world::MicroOpcode::FuseAttn, 40U, 32U, 32U, 0U, 0U);
        emit(program, world::MicroOpcode::DecPixVol, 64U, 40U, 0U, 2U, static_cast<std::uint32_t>(kCubeTileBytes));
        emit(program, world::MicroOpcode::CalcDelta, 96U, 2U, 64U, 0U, static_cast<std::uint32_t>(kCubeTileBytes));
        emit(program, world::MicroOpcode::ProposeBias, 0U, 96U, 0U, 0U, static_cast<std::uint32_t>(kCubeTileBytes));
        emit(program, world::MicroOpcode::Validate, 20U, 0U, 0U, 0U, threshold);
        emit(program, world::MicroOpcode::Halt, 0U, 0U, 0U, 0U, 0U);
        return program;
    }

    static world::WorldProgram commit_program(bool accept) {
        world::WorldProgram program;
        const std::uint16_t gate = accept ? 20U : 511U;
        emit(program, world::MicroOpcode::CommitIf, 0U, gate, 0U, 0U, 0U);
        emit(program, world::MicroOpcode::Halt, 0U, 0U, 0U, 0U, 0U);
        return program;
    }

    world::WorldProgram decode_program(const world::Vmad128& source, const world::Vmad128& output) const {
        world::WorldProgram program;
        program.descriptors = {source, output};
        emit(program, world::MicroOpcode::LoadVmad, 0U, 0U, 0U, 0U, 0U);
        emit(program, world::MicroOpcode::LoadVmad, 0U, 0U, 0U, 1U, 1U);
        emit(program, world::MicroOpcode::TileInVec, 32U, 0U, 0U, 0U, static_cast<std::uint32_t>(kCubeLatentBytes));
        emit(program, world::MicroOpcode::DecPixVol, 64U, 32U, 0U, 1U, static_cast<std::uint32_t>(kCubeTileBytes));
        emit(program, world::MicroOpcode::Halt, 0U, 0U, 0U, 0U, 0U);
        return program;
    }

    void copy_span(const world::Vmad128& source, const world::Vmad128& destination, std::size_t bytes) {
        std::vector<std::uint8_t> staged(bytes, 0U);
        for (std::size_t i = 0U; i < bytes; ++i) {
            staged[i] = volume_.read(world::vmad_advance_linear(source, static_cast<std::uint64_t>(i)).coord());
        }
        for (std::size_t i = 0U; i < bytes; ++i) {
            volume_.write(world::vmad_advance_linear(destination, static_cast<std::uint64_t>(i)).coord(), staged[i]);
        }
    }

    void execute_encode(const CubeCommand& command, CubeCommandMetrics& metrics) {
        long double delta_sum = 0.0L;
        for (std::uint32_t tile = 0U; tile < command.tile_count; ++tile) {
            const auto source = world::vmad_advance_linear(command.source, static_cast<std::uint64_t>(tile) * kCubeTileBytes);
            const auto latent = world::vmad_advance_linear(command.latent, static_cast<std::uint64_t>(tile) * kCubeLatentBytes);
            const auto output = world::vmad_advance_linear(command.output, static_cast<std::uint64_t>(tile) * kCubeTileBytes);
            const auto shadow_latent = world::vmad_advance_linear(command.shadow_latent, static_cast<std::uint64_t>(tile) * kCubeLatentBytes);
            const auto shadow_output = world::vmad_advance_linear(command.shadow_output, static_cast<std::uint64_t>(tile) * kCubeTileBytes);
            std::uint32_t previous_delta = 256U;
            std::uint32_t last_accepted_delta = 255U;
            bool accepted_any = false;
            bool converged = false;
            for (std::uint32_t pass = 0U; pass < command.max_passes; ++pass) {
                const std::uint32_t threshold = std::min<std::uint32_t>(255U, previous_delta);
                engine_.run(candidate_program(source, shadow_latent, shadow_output, threshold), 64ULL);
                const std::uint32_t candidate_delta = engine_.last_delta_mean();
                const bool lambda_gate = engine_.scalar_register(20U) != 0ULL;
                const bool improved = pass == 0U || candidate_delta < previous_delta;
                const bool accept = lambda_gate && improved;
                ++metrics.passes;
                if (accept) {
                    copy_span(shadow_latent, latent, kCubeLatentBytes);
                    copy_span(shadow_output, output, kCubeTileBytes);
                    engine_.run(commit_program(true), 8ULL);
                    ++metrics.accepted_passes;
                    accepted_any = true;
                    previous_delta = candidate_delta;
                    last_accepted_delta = candidate_delta;
                    if (candidate_delta <= command.epsilon) {
                        converged = true;
                        break;
                    }
                } else {
                    engine_.run(commit_program(false), 8ULL);
                    ++metrics.rejected_passes;
                    break;
                }
            }
            if (converged) ++metrics.converged_tiles;
            delta_sum += static_cast<long double>(accepted_any ? last_accepted_delta : 255U);
        }
        metrics.input_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeTileBytes;
        metrics.latent_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeLatentBytes;
        metrics.output_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeTileBytes;
        metrics.mean_final_delta = command.tile_count == 0U ? 0.0 :
            static_cast<double>(delta_sum / static_cast<long double>(command.tile_count));
    }

    void execute_decode(const CubeCommand& command, CubeCommandMetrics& metrics) {
        for (std::uint32_t tile = 0U; tile < command.tile_count; ++tile) {
            const auto source = world::vmad_advance_linear(command.source, static_cast<std::uint64_t>(tile) * kCubeLatentBytes);
            const auto output = world::vmad_advance_linear(command.output, static_cast<std::uint64_t>(tile) * kCubeTileBytes);
            engine_.run(decode_program(source, output), 16ULL);
        }
        metrics.input_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeLatentBytes;
        metrics.output_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeTileBytes;
    }
};

struct CubeDemoPlan {
    std::vector<std::uint8_t> execution_buffer;
    world::Vmad128 source{};
    world::Vmad128 final_output{};
    std::uint32_t base_tiles{};
    std::uint8_t levels{};
};

inline world::Vmad128 demo_address(std::uint16_t region, std::uint64_t z) {
    return world::Vmad128::pack(region, 0U, 0U, 0U, 0U, z);
}

inline CubeDemoPlan make_demo_plan(std::uint32_t base_tiles = 32U, std::uint8_t levels = 2U,
                                   std::uint64_t logical_extent = 1000ULL * 1000ULL * 1000ULL) {
    if (base_tiles == 0U || levels == 0U || levels > 8U) throw std::invalid_argument("invalid recursive-cube demo geometry");
    std::uint64_t cursor = 0ULL;
    const auto allocate = [&](std::uint16_t region, std::uint64_t bytes, std::uint64_t& mutable_cursor) {
        if (bytes > logical_extent || mutable_cursor > logical_extent - bytes) {
            throw std::overflow_error("recursive-cube demo layout exceeds logical extent");
        }
        const auto address = demo_address(region, mutable_cursor);
        mutable_cursor += bytes + 4096ULL;
        if (mutable_cursor > logical_extent) throw std::overflow_error("recursive-cube demo padding exceeds logical extent");
        return address;
    };

    CubeDemoPlan plan;
    plan.base_tiles = base_tiles;
    plan.levels = levels;
    plan.source = allocate(1U, static_cast<std::uint64_t>(base_tiles) * kCubeTileBytes, cursor);

    std::vector<CubeCommand> commands;
    std::vector<world::Vmad128> level_latents;
    std::vector<std::uint32_t> level_tiles;
    world::Vmad128 source = plan.source;
    std::uint32_t tiles = base_tiles;
    for (std::uint8_t level = 0U; level < levels; ++level) {
        const std::uint64_t latent_bytes = static_cast<std::uint64_t>(tiles) * kCubeLatentBytes;
        const std::uint64_t tile_bytes = static_cast<std::uint64_t>(tiles) * kCubeTileBytes;
        CubeCommand command;
        command.opcode = CubeOpcode::EncodeRefine;
        command.level = level;
        command.tile_count = tiles;
        command.max_passes = 4U;
        command.epsilon = 8U;
        command.source = source;
        command.latent = allocate(static_cast<std::uint16_t>(100U + level), latent_bytes, cursor);
        command.output = allocate(static_cast<std::uint16_t>(200U + level), tile_bytes, cursor);
        command.shadow_latent = allocate(static_cast<std::uint16_t>(300U + level), latent_bytes, cursor);
        command.shadow_output = allocate(static_cast<std::uint16_t>(400U + level), tile_bytes, cursor);
        commands.push_back(command);
        level_latents.push_back(command.latent);
        level_tiles.push_back(tiles);
        source = command.latent;
        tiles = (tiles + 31U) / 32U;
    }

    world::Vmad128 decode_source = level_latents.back();
    for (std::size_t reverse = level_latents.size(); reverse-- > 0U;) {
        CubeCommand command;
        command.opcode = CubeOpcode::Decode;
        command.level = static_cast<std::uint8_t>(reverse);
        command.tile_count = level_tiles[reverse];
        command.max_passes = 1U;
        command.source = decode_source;
        const std::uint64_t output_bytes = static_cast<std::uint64_t>(command.tile_count) * kCubeTileBytes;
        command.output = allocate(static_cast<std::uint16_t>(500U + reverse), output_bytes, cursor);
        commands.push_back(command);
        decode_source = command.output;
        if (reverse == 0U) plan.final_output = command.output;
    }

    commands.push_back(CubeCommand{CubeOpcode::Halt});
    plan.execution_buffer = serialize_execution_buffer(commands);
    return plan;
}

} // namespace jarvisx::cube
