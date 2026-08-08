#include "jarvisx/dmso_os.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace jarvisx::dmso;

    const std::uint64_t word = BinaryBytecodeCompiler::compile_instruction(0x80U, 45000U, 3U);
    const Instruction decoded = BinaryBytecodeCompiler::decompile_raw_chunk(word);
    assert(decoded.opcode == 0x80U);
    assert(decoded.alpha == 45000U);
    assert(decoded.beta == 3U);

    VirtualVRAMRegisters registers(8U);
    ExecutionEngineKernel kernel(registers);
    const std::vector<std::uint64_t> program{
        BinaryBytecodeCompiler::compile_instruction(
            BinaryBytecodeCompiler::RECURSE_SPACE,
            45000U,
            1U
        ),
        BinaryBytecodeCompiler::compile_instruction(
            BinaryBytecodeCompiler::EVAL_FIELD_GEO,
            0U,
            0U
        ),
        BinaryBytecodeCompiler::compile_instruction(
            BinaryBytecodeCompiler::EVAL_FIELD_TEX,
            0U,
            0U
        ),
    };

    const auto& frame = kernel.dispatch_forward_pass(program);
    assert(frame.size() == 8U * 8U * 3U);
    for (const double value : frame) {
        assert(std::isfinite(value));
        assert(value >= 0.0 && value <= 1.0);
    }

    bool rejected_opcode = false;
    try {
        kernel.dispatch_forward_pass({
            BinaryBytecodeCompiler::compile_instruction(0x01U, 0U, 0U),
        });
    } catch (const std::invalid_argument&) {
        rejected_opcode = true;
    }
    assert(rejected_opcode);

    AutonomousSystemOS os(8U);
    auto future = os.execute_system_lifecycle_async(2U, 0.05, 0.0);
    const LifecycleReport report = future.get();
    assert(report.epochs_executed == 2U);
    assert(report.accepted_updates + report.rejected_updates == report.epochs_executed);
    assert(std::isfinite(report.initial_loss));
    assert(std::isfinite(report.final_loss));
    assert(report.final_loss >= 0.0);
    assert(report.final_loss <= report.initial_loss);
    assert(report.elapsed_seconds >= 0.0);
    for (const double value : report.theta_g) {
        assert(std::isfinite(value));
    }
    for (const double value : report.theta_t) {
        assert(std::isfinite(value));
    }

    bool rejected_resolution = false;
    try {
        AutonomousSystemOS invalid(0U);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected_resolution = true;
    }
    assert(rejected_resolution);

    std::cout << "dmso-os regressions passed\n";
    return 0;
}
