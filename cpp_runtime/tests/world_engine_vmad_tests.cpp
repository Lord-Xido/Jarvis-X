#include "jarvisx/bitwise_world500.hpp"
#include "jarvisx/world_engine_vmad.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::filesystem::path test_dir(const char* name) {
    return std::filesystem::temp_directory_path() / (std::string("jarvisx-world-") + name);
}

void test_world500_footprint_and_address_geometry() {
    namespace bw = jarvisx::world::bitwise500;
    require(bw::kWorldEdge == 500U, "World500 edge constant is incorrect");
    require(bw::kWorldAgentCount == 125000000ULL, "World500 agent count is incorrect");
    require(bw::kLatentDimensions == 16U, "World500 latent width is incorrect");
    require(bw::kBitsPerAgent == 512U, "World500 bits-per-agent is incorrect");
    require(bw::kWorldLatentBits == 64000000000ULL, "World500 latent bit footprint is incorrect");
    require(bw::kWorldLatentBytes == 8000000000ULL, "World500 latent byte footprint is incorrect");
    require(bw::kWorldLatentGiB > 7.45L && bw::kWorldLatentGiB < 7.46L,
            "World500 GiB conversion is incorrect");

    require(bw::linear_address(0U, 0U, 0U) == 0ULL, "World500 origin address is incorrect");
    const std::uint64_t last = bw::linear_address(499U, 499U, 499U);
    require(last == bw::kWorldAgentCount - 1ULL, "World500 terminal address is incorrect");
    require(bw::coordinate_from_address(last) == bw::Coord500{499U, 499U, 499U},
            "World500 inverse address mapping failed");

    const std::uint64_t interior = bw::linear_address(17U, 231U, 404U);
    require(bw::coordinate_from_address(interior) == bw::Coord500{17U, 231U, 404U},
            "World500 address mapping is not bijective");

    bool rejected = false;
    try {
        (void)bw::linear_address(500U, 0U, 0U);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    require(rejected, "World500 accepted an out-of-range coordinate");
}

void test_world500_bit_and_residual_semantics() {
    namespace bw = jarvisx::world::bitwise500;
    const std::vector<std::uint8_t> bits{1U, 0U, 1U, 1U, 0U, 0U, 1U, 0U};
    const auto packed = bw::pack_bits_lsb_first(bits);
    require(packed.size() == 1U && packed[0] == 0x4dU, "World500 LSB-first bit packing failed");
    require(bw::unpack_bits_lsb_first(packed, bits.size()) == bits,
            "World500 bit packing roundtrip failed");

    require(std::fabs(bw::normalize_byte(255U) - 1.0F) < 1.0e-6F,
            "World500 byte normalization failed");
    require(bw::xor_residual(0xb6U, 0xa6U) == 0x10U,
            "World500 byte XOR residual failed");
    require(std::fabs(bw::byte_xor_error_rate(0xb6U, 0xa6U) - 0.125F) < 1.0e-6F,
            "World500 XOR error density failed");
    require(bw::fp32_xor_residual(1.0F, 1.0F) == 0U,
            "World500 identical FP32 values produced a representation residual");
    require(bw::fp32_xor_residual(1.0F, 1.5F) != 0U,
            "World500 distinct FP32 values produced no representation residual");
    require(std::fabs(bw::numeric_abs_error(1.0F, 1.5F) - 0.5F) < 1.0e-6F,
            "World500 numeric residual failed");
    require(bw::byte_fixed_point(0x5aU, 0x5aU), "World500 byte fixed point failed");
    require(!bw::byte_fixed_point(0x5aU, 0x5bU), "World500 false byte fixed point accepted");
}

void test_world500_permeation_attention_memory_transaction() {
    namespace bw = jarvisx::world::bitwise500;
    require(std::fabs(bw::permeate_channel(1.0F, 0.0F) - 0.82F) < 1.0e-6F,
            "World500 0.82/0.18 permeation rule failed");

    std::array<float, bw::kLatentDimensions> query{};
    std::array<float, bw::kLatentDimensions> key{};
    query.fill(1.0F);
    key.fill(1.0F);
    require(std::fabs(bw::scaled_dot_logit(query, key) - 4.0F) < 1.0e-6F,
            "World500 scaled dot-product attention logit failed");

    const auto weights = bw::softmax({0.0F, 1.0F, 2.0F});
    require(weights.size() == 3U, "World500 softmax output width is incorrect");
    const float sum = weights[0] + weights[1] + weights[2];
    require(std::fabs(sum - 1.0F) < 1.0e-6F, "World500 softmax is not normalized");
    require(weights[2] > weights[1] && weights[1] > weights[0],
            "World500 softmax ordering is incorrect");

    require(std::fabs(bw::memory_update(0.0F, 1.0F, 0.9F) - 0.1F) < 1.0e-6F,
            "World500 Omega memory recurrence failed");
    require(std::fabs(bw::latent_velocity(1.0F, 2.0F, 3.0F, 4.0F,
                                          0.5F, 0.25F, 0.1F, 0.05F) - 0.1F) < 1.0e-6F,
            "World500 latent velocity equation failed");
    require(bw::should_commit(1.0, 0.5), "World500 strict improvement was not committed");
    require(!bw::should_commit(1.0, 1.0), "World500 accepted a non-improving equal-error candidate");
    require(!bw::should_commit(1.0, 1.5), "World500 accepted a worse candidate");
}

void test_vmad_pack_roundtrip() {
    const auto address = jarvisx::world::Vmad128::pack(
        4095U, 255U, 511U,
        jarvisx::world::kVmadCoordExtent - 1ULL,
        jarvisx::world::kVmadCoordExtent - 2ULL,
        jarvisx::world::kVmadCoordExtent - 3ULL);
    require(address.region() == 4095U, "VMAD region roundtrip failed");
    require(address.modality() == 255U, "VMAD modality roundtrip failed");
    require(address.attributes() == 511U, "VMAD attributes roundtrip failed");
    require(address.x() == jarvisx::world::kVmadCoordExtent - 1ULL, "VMAD X roundtrip failed");
    require(address.y() == jarvisx::world::kVmadCoordExtent - 2ULL, "VMAD Y roundtrip failed");
    require(address.z() == jarvisx::world::kVmadCoordExtent - 3ULL, "VMAD Z roundtrip failed");

    const auto wrapped = jarvisx::world::vmad_offset(address, 1, 2, 3);
    require(wrapped.x() == 0ULL && wrapped.y() == 0ULL && wrapped.z() == 0ULL,
            "VMAD toroidal offset failed");

    bool rejected = false;
    try {
        (void)jarvisx::world::Vmad128::pack(0U, 0U, 0U, jarvisx::world::kVmadCoordExtent, 0U, 0U);
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    require(rejected, "VMAD accepted a coordinate outside the 33-bit domain");
}

void test_micro_op_roundtrip() {
    const jarvisx::world::MicroOp original{
        jarvisx::world::MicroOpcode::CalcDelta, 511U, 510U, 509U, 31U, 0x00abcdefU,
    };
    const auto decoded = jarvisx::world::MicroOp::decode(original.encode());
    require(decoded.opcode == original.opcode, "micro-op opcode roundtrip failed");
    require(decoded.dst == original.dst && decoded.src0 == original.src0 && decoded.src1 == original.src1,
            "micro-op register roundtrip failed");
    require(decoded.vmad_reg == original.vmad_reg, "micro-op VMAD register roundtrip failed");
    require(decoded.imm24 == original.imm24, "micro-op immediate roundtrip failed");
}

void test_sparse_ingest_and_pipeline() {
    const auto dir = test_dir("pipeline");
    std::filesystem::remove_all(dir);
    jarvisx::intelligence3d::VirtualVolume3D volume({
        jarvisx::world::kVmadCoordExtent, 32U, 2ULL * 1024ULL * 1024ULL, dir / "pages",
    });
    const auto source = jarvisx::world::Vmad128::pack(1U, 1U, 0U, 5U, 7U, 11U);
    const auto output = jarvisx::world::Vmad128::pack(2U, 1U, 0U, 17U, 19U, 23U);
    for (std::size_t i = 0U; i < 1024U; ++i) {
        const auto address = jarvisx::world::vmad_advance_linear(source, static_cast<std::uint64_t>(i));
        volume.write(address.coord(), static_cast<std::uint8_t>((i * 13U + 17U) & 0xffU));
    }

    jarvisx::world::WorldEngine128 engine(volume);
    engine.run(jarvisx::world::make_world_demo_program(source, output), 1000ULL);
    const auto& stats = engine.stats();
    require(stats.bytes_ingested == 1024ULL, "pipeline did not ingest one 1024-byte quantum");
    require(stats.bytes_stored == 1024ULL, "pipeline did not reconstruct one 1024-byte quantum");
    require(stats.commits == 1ULL && stats.rollbacks == 0ULL, "pipeline candidate did not commit");
    require(stats.stage_issues[0] >= 1ULL, "ingest stage was not exercised");
    require(stats.stage_issues[1] >= 1ULL, "reduce stage was not exercised");
    require(stats.stage_issues[2] >= 1ULL, "fusion stage was not exercised");
    require(stats.stage_issues[3] >= 1ULL, "reconstruction stage was not exercised");
    require(stats.stage_issues[4] >= 1ULL, "feedback stage was not exercised");
    require(stats.estimated_pipeline_latency_cycles == stats.logical_issue_cycles + 4ULL,
            "pipeline latency accounting is inconsistent");
    require(volume.read(output.coord()) == engine.vector_register(64U)[0],
            "decoded vector and sparse output volume diverged");
    std::filesystem::remove_all(dir);
}

jarvisx::world::WorldProgram adaptation_program(std::uint32_t threshold) {
    jarvisx::world::WorldProgram program;
    const auto emit = [&](jarvisx::world::MicroOpcode opcode, std::uint16_t dst, std::uint16_t src0,
                          std::uint16_t src1, std::uint32_t imm24) {
        program.words.push_back(jarvisx::world::MicroOp{opcode, dst, src0, src1, 0U, imm24}.encode());
    };
    emit(jarvisx::world::MicroOpcode::CalcDelta, 2U, 0U, 1U, 64U);
    emit(jarvisx::world::MicroOpcode::ProposeBias, 0U, 2U, 0U, 64U);
    emit(jarvisx::world::MicroOpcode::Validate, 7U, 0U, 0U, threshold);
    emit(jarvisx::world::MicroOpcode::CommitIf, 0U, 7U, 0U, 0U);
    emit(jarvisx::world::MicroOpcode::Halt, 0U, 0U, 0U, 0U);
    return program;
}

void test_candidate_commit_and_rollback() {
    const auto dir = test_dir("transaction");
    std::filesystem::remove_all(dir);
    jarvisx::intelligence3d::VirtualVolume3D volume({
        jarvisx::world::kVmadCoordExtent, 32U, 1024ULL * 1024ULL, dir / "pages",
    });
    jarvisx::world::WorldEngine128 engine(volume);

    jarvisx::world::WorldEngine128::VectorRegister high{};
    jarvisx::world::WorldEngine128::VectorRegister low{};
    high.fill(255U);
    low.fill(0U);
    engine.set_vector_register(0U, high);
    engine.set_vector_register(1U, low);
    const auto before = engine.biases();

    engine.run(adaptation_program(0U), 100ULL);
    require(engine.stats().rollbacks == 1ULL, "failed quality gate did not roll back candidate");
    require(engine.biases() == before, "rollback mutated authoritative bias state");

    engine.run(adaptation_program(255U), 100ULL);
    require(engine.stats().commits == 1ULL, "validated candidate did not commit");
    require(engine.biases() != before, "commit did not change authoritative bias state");
    std::filesystem::remove_all(dir);
}

void test_transfer_bound() {
    const auto dir = test_dir("bounds");
    std::filesystem::remove_all(dir);
    jarvisx::intelligence3d::VirtualVolume3D volume({
        jarvisx::world::kVmadCoordExtent, 32U, 1024ULL * 1024ULL, dir / "pages",
    });
    jarvisx::world::WorldEngine128 engine(volume);
    jarvisx::world::WorldProgram program;
    program.descriptors.push_back(jarvisx::world::Vmad128::pack(0U, 0U, 0U, 0U, 0U, 0U));
    program.words.push_back(jarvisx::world::MicroOp{jarvisx::world::MicroOpcode::LoadVmad, 0U, 0U, 0U, 0U, 0U}.encode());
    program.words.push_back(jarvisx::world::MicroOp{jarvisx::world::MicroOpcode::TileInVec, 0U, 0U, 0U, 0U, 4097U}.encode());
    bool rejected = false;
    try {
        engine.run(program, 10ULL);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "world engine accepted an oversized transfer");
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    try {
        test_world500_footprint_and_address_geometry();
        test_world500_bit_and_residual_semantics();
        test_world500_permeation_attention_memory_transaction();
        test_vmad_pack_roundtrip();
        test_micro_op_roundtrip();
        test_sparse_ingest_and_pipeline();
        test_candidate_commit_and_rollback();
        test_transfer_bound();
        std::cout << "world-engine VMAD128 + World500 regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "world-engine regression failure: " << error.what() << '\n';
        return 1;
    }
}
