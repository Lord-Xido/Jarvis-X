#include "jarvisx/recursive_cube_interpreter.hpp"

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
    return std::filesystem::temp_directory_path() / (std::string("jarvisx-recursive-cube-") + name);
}

std::uint8_t sample(std::uint64_t index) noexcept {
    return static_cast<std::uint8_t>((index * 29ULL + (index >> 4U) + 7ULL) & 0xffULL);
}

void seed(jarvisx::intelligence3d::VirtualVolume3D& volume,
          const jarvisx::world::Vmad128& base, std::uint64_t bytes) {
    for (std::uint64_t i = 0ULL; i < bytes; ++i) {
        volume.write(jarvisx::world::vmad_advance_linear(base, i).coord(), sample(i));
    }
}

void test_execution_buffer_validation() {
    const auto plan = jarvisx::cube::make_demo_plan(32U, 2U);
    const auto commands = jarvisx::cube::parse_execution_buffer(plan.execution_buffer, {});
    require(commands.size() == 5U, "two-level recursive-cube plan should contain four actions plus HALT");
    require(commands.front().opcode == jarvisx::cube::CubeOpcode::EncodeRefine,
            "recursive-cube plan does not begin with inward encoding");
    require(commands[2].opcode == jarvisx::cube::CubeOpcode::Decode,
            "recursive-cube plan does not turn outward after hierarchy construction");
    require(commands.back().opcode == jarvisx::cube::CubeOpcode::Halt,
            "recursive-cube plan missing terminal HALT");

    auto corrupted = plan.execution_buffer;
    corrupted[20] ^= 0x5aU;
    bool rejected = false;
    try {
        (void)jarvisx::cube::parse_execution_buffer(corrupted, {});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "corrupted recursive-cube execution buffer was accepted");
}

void test_overlap_and_step_limits() {
    jarvisx::cube::CubeCommand command;
    command.opcode = jarvisx::cube::CubeOpcode::EncodeRefine;
    command.tile_count = 1U;
    command.max_passes = 1U;
    command.epsilon = 8U;
    command.source = jarvisx::cube::demo_address(1U, 1000U);
    command.latent = jarvisx::cube::demo_address(2U, 1000U);
    command.output = jarvisx::cube::demo_address(3U, 5000U);
    command.shadow_latent = jarvisx::cube::demo_address(4U, 9000U);
    command.shadow_output = jarvisx::cube::demo_address(5U, 13000U);
    std::vector<jarvisx::cube::CubeCommand> commands{command, jarvisx::cube::CubeCommand{jarvisx::cube::CubeOpcode::Halt}};
    bool overlap_rejected = false;
    try {
        (void)jarvisx::cube::parse_execution_buffer(jarvisx::cube::serialize_execution_buffer(commands), {});
    } catch (const std::runtime_error&) {
        overlap_rejected = true;
    }
    require(overlap_rejected, "overlapping recursive-cube source/latent spans were accepted");

    const auto plan = jarvisx::cube::make_demo_plan(4U, 1U);
    jarvisx::cube::CubeInterpreterConfig config;
    config.max_total_tile_ops = 1ULL;
    bool bound_rejected = false;
    try {
        (void)jarvisx::cube::parse_execution_buffer(plan.execution_buffer, config);
    } catch (const std::runtime_error&) {
        bound_rejected = true;
    }
    require(bound_rejected, "recursive-cube total tile-operation guard did not fire");
}

void test_end_to_end_recursive_execution() {
    const auto dir = test_dir("e2e");
    std::filesystem::remove_all(dir);
    jarvisx::intelligence3d::VirtualVolume3D volume({
        jarvisx::world::kVmadCoordExtent,
        32U,
        4ULL * 1024ULL * 1024ULL,
        dir / "pages",
    });

    const auto plan = jarvisx::cube::make_demo_plan(4U, 1U);
    const auto commands = jarvisx::cube::parse_execution_buffer(plan.execution_buffer, {});
    const std::uint64_t source_bytes = 4ULL * jarvisx::cube::kCubeTileBytes;
    const std::uint64_t latent_bytes = 4ULL * jarvisx::cube::kCubeLatentBytes;
    seed(volume, plan.source, source_bytes);

    jarvisx::cube::RecursiveCubeInterpreter interpreter(volume);
    const auto metrics = interpreter.run(plan.execution_buffer);
    require(metrics.execution_buffer_validated, "recursive-cube execution buffer did not validate");
    require(metrics.commands_executed == 2ULL, "single-level plan should execute one inward and one outward command");
    require(metrics.commands.size() == 2U, "recursive-cube command telemetry count mismatch");
    require(metrics.accepted_passes >= 4ULL, "each first tile candidate should pass the byte-range Lambda gate");
    require(metrics.commands[0].input_bytes == source_bytes,
            "recursive-cube inward command input byte count mismatch");
    require(metrics.commands[1].input_bytes == latent_bytes,
            "recursive-cube outward command latent input byte count mismatch");
    require(metrics.encoded_input_bytes == source_bytes,
            "recursive-cube encoded input should count inward source bytes only");
    require(metrics.latent_bytes_committed == latent_bytes,
            "recursive-cube committed latent byte count mismatch");
    require(metrics.decoded_output_bytes == source_bytes,
            "recursive-cube decoded output should count outward output bytes only");
    require(metrics.aggregate_command_input_bytes == source_bytes + latent_bytes,
            "recursive-cube aggregate command input byte count mismatch");
    require(metrics.aggregate_command_output_bytes == source_bytes + source_bytes,
            "recursive-cube aggregate command output byte count mismatch");
    require(interpreter.engine().stats().commits == metrics.accepted_passes,
            "world-engine commit telemetry diverges from recursive interpreter");

    long double authoritative_delta_sum = 0.0L;
    const auto& inward = commands.front();
    for (std::uint32_t tile = 0U; tile < inward.tile_count; ++tile) {
        const auto source = jarvisx::world::vmad_advance_linear(
            inward.source, static_cast<std::uint64_t>(tile) * jarvisx::cube::kCubeTileBytes);
        const auto authoritative = jarvisx::world::vmad_advance_linear(
            inward.output, static_cast<std::uint64_t>(tile) * jarvisx::cube::kCubeTileBytes);
        std::uint64_t score = 0ULL;
        for (std::size_t i = 0U; i < jarvisx::cube::kCubeTileBytes; ++i) {
            const int left = static_cast<int>(volume.read(
                jarvisx::world::vmad_advance_linear(source, static_cast<std::uint64_t>(i)).coord()));
            const int right = static_cast<int>(volume.read(
                jarvisx::world::vmad_advance_linear(authoritative, static_cast<std::uint64_t>(i)).coord()));
            const int delta = left - right;
            score += static_cast<std::uint64_t>(delta < 0 ? -delta : delta);
        }
        authoritative_delta_sum += static_cast<long double>(
            score / static_cast<std::uint64_t>(jarvisx::cube::kCubeTileBytes));
    }
    const double authoritative_mean = static_cast<double>(
        authoritative_delta_sum / static_cast<long double>(inward.tile_count));
    require(std::abs(metrics.commands[0].mean_final_delta - authoritative_mean) < 1.0e-9,
            "mean_final_delta does not describe authoritative committed reconstruction");

    std::uint64_t nonzero = 0ULL;
    for (std::uint64_t i = 0ULL; i < source_bytes; ++i) {
        if (volume.read(jarvisx::world::vmad_advance_linear(plan.final_output, i).coord()) != 0U) ++nonzero;
    }
    require(nonzero != 0ULL, "recursive outward decode produced an empty final world state");
    std::filesystem::remove_all(dir);
}

void test_data_is_not_implicitly_executable() {
    const auto dir = test_dir("data-not-code");
    std::filesystem::remove_all(dir);
    jarvisx::intelligence3d::VirtualVolume3D volume({
        jarvisx::world::kVmadCoordExtent,
        32U,
        1024ULL * 1024ULL,
        dir / "pages",
    });
    const auto plan = jarvisx::cube::make_demo_plan(1U, 1U);
    for (std::size_t i = 0U; i < jarvisx::cube::kCubeMagic.size(); ++i) {
        volume.write(jarvisx::world::vmad_advance_linear(plan.source, static_cast<std::uint64_t>(i)).coord(),
                     jarvisx::cube::kCubeMagic[i]);
    }
    jarvisx::cube::RecursiveCubeInterpreter interpreter(volume);
    const auto metrics = interpreter.run(plan.execution_buffer);
    require(metrics.commands_executed == 2ULL,
            "payload bytes altered interpreter control flow; decoded data must not auto-execute");
    std::filesystem::remove_all(dir);
}

} // namespace

int main() {
    try {
        test_execution_buffer_validation();
        test_overlap_and_step_limits();
        test_end_to_end_recursive_execution();
        test_data_is_not_implicitly_executable();
        std::cout << "recursive-cube interpreter regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "recursive-cube interpreter regression failure: " << error.what() << '\n';
        return 1;
    }
}
