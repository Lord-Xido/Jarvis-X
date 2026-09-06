#include "jarvisx/compile_interpret_accelerator.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace accel = jarvisx::cube::acceleration;

namespace {

bool close(long double a, long double b, long double eps = 1.0e-12L) {
    return std::abs(a - b) <= eps * std::max(1.0L, std::max(std::abs(a), std::abs(b)));
}

void target_is_septillion_to_septillion_in_log_space() {
    const accel::AccelerationTarget target{};
    assert(std::isfinite(target.log10_speedup));
    assert(close(target.log10_speedup, 24.0L * 1.0e24L, 1.0e-18L));
}

void factor_product_is_logarithmically_composable() {
    accel::AccelerationFactors f;
    f.parallel = 8.0L;
    f.sparse = 4.0L;
    f.vectorization = 2.0L;
    assert(close(f.log10_product(), std::log10(64.0L)));
}

void recursive_fold_reduces_work_multiplicatively() {
    accel::RecursiveFoldProfile fold;
    fold.retention = {0.5L, 0.25L};
    assert(close(fold.retained_work_fraction(), 0.125L));
    assert(close(fold.log10_speedup(), std::log10(8.0L)));
}

void invalid_fold_is_rejected() {
    accel::RecursiveFoldProfile fold;
    fold.retention = {1.0L, 0.0L};
    bool threw = false;
    try {
        (void)fold.log10_speedup();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void compile_interpret_fusion_preserves_serial_commit_boundary() {
    accel::CompileInterpretTiming timing;
    timing.compile_ns = 10.0L;
    timing.interpret_ns = 20.0L;
    timing.commit_ns = 2.0L;
    assert(close(timing.sequential_ns(), 32.0L));
    assert(close(timing.fused_ns(), 22.0L));
    assert(close(timing.ideal_fusion_speedup(), 32.0L / 22.0L));
}

void kinetic_gain_is_bounded() {
    accel::AccelerationContract contract;
    contract.measured_factors.parallel = 64.0L;
    contract.fold.retention = {0.5L, 0.5L, 0.5L};
    const long double gain = contract.bounded_kinetic_gain();
    assert(gain >= 0.0L);
    assert(gain < 1.0L);
}

void harmonic_scheduler_stays_finite() {
    accel::AccelerationContract contract;
    contract.measured_factors.parallel = 8.0L;
    contract.measured_factors.sparse = 16.0L;
    accel::HarmonicCompileInterpretPhase phase;
    for (int i = 0; i < 10000; ++i) phase.advance(2.0L, 0.001L, contract);
    assert(std::isfinite(phase.phase));
    assert(phase.phase >= 0.0L);
    assert(phase.phase < 2.0L * 3.1415926535897932384626433832795L);
    const long double c = phase.compile_component();
    const long double s = phase.interpret_component();
    assert(close(c*c + s*s, 1.0L, 1.0e-10L));
}

void target_gap_never_materializes_hyper_exponential_number() {
    accel::AccelerationContract contract;
    contract.measured_factors.parallel = 128.0L;
    contract.measured_factors.sparse = 32.0L;
    contract.fold.retention = {0.25L, 0.25L};
    const long double modeled = contract.modeled_log10_speedup();
    const long double gap = contract.target_gap_log10();
    assert(std::isfinite(modeled));
    assert(std::isfinite(gap));
    assert(gap > 1.0e24L);
}

} // namespace

int main() {
    target_is_septillion_to_septillion_in_log_space();
    factor_product_is_logarithmically_composable();
    recursive_fold_reduces_work_multiplicatively();
    invalid_fold_is_rejected();
    compile_interpret_fusion_preserves_serial_commit_boundary();
    kinetic_gain_is_bounded();
    harmonic_scheduler_stays_finite();
    target_gap_never_materializes_hyper_exponential_number();
    std::cout << "compile-interpret accelerator regressions passed\n";
    return 0;
}
