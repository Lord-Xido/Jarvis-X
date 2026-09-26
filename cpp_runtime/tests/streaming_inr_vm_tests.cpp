#include "jarvisx/streaming_inr_vm.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

using namespace jarvisx::inrvm;

int main() {
    static_assert(static_cast<std::uint8_t>(Opcode::LD_COORD) == 0x01);
    static_assert(static_cast<std::uint8_t>(Opcode::FF_TRANS) == 0x02);
    static_assert(static_cast<std::uint8_t>(Opcode::SIN_ACT) == 0x03);
    static_assert(static_cast<std::uint8_t>(Opcode::MAT_MUL) == 0x04);
    static_assert(static_cast<std::uint8_t>(Opcode::HOLO_FUSE) == 0x05);
    static_assert(static_cast<std::uint8_t>(Opcode::BND_CHECK) == 0x06);
    static_assert(static_cast<std::uint8_t>(Opcode::RET_VAL) == 0x07);

    VmConfig cfg;
    cfg.feature_width = 16;
    cfg.fourier_width = 16;
    cfg.omega_slots = 8;
    cfg.entropy_cutoff = 0.0f;
    StreamingInrVm vm(cfg);

    const auto result = vm.execute({4'000'000.0, 2'000'000.0, 6'000'000.0, 0.0});
    assert(result.emitted);
    assert(!result.skipped);
    assert(result.feature.size() == cfg.feature_width);
    assert(result.entropy >= 0.0f && result.entropy <= 1.0f + 1e-5f);
    for (float v : result.feature) assert(std::isfinite(v));
    assert(vm.budget().total_used() < MemoryPlan::managed_limit);

    auto gated = StreamingInrVm::default_program(1.1f);
    const auto skipped = vm.execute({1.0, 2.0, 3.0, 1.0}, gated);
    assert(skipped.skipped);
    assert(!skipped.emitted);

    BudgetTracker budget;
    bool threw = false;
    try {
        budget.claim(Arena::Vm, MemoryPlan::vm_limit + 1);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "streaming INR VM tests passed\n";
    return 0;
}
