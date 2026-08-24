#include "jarvisx/electromagnetic_flow.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool close(double lhs, double rhs, double tolerance = 1.0e-12) {
    return std::abs(lhs - rhs) <= tolerance;
}

void test_single_q16_lsb_toggle() {
    const std::vector<double> before{0.0};
    const std::vector<double> after{1.0 / 65536.0};
    const auto telemetry = jarvisx::ElectromagneticFlowLogic::measure_transition(before, after);

    assert(telemetry.words == 1U);
    assert(telemetry.raw_bit_toggles == 1U);
    assert(telemetry.dbi_bit_toggles == 1U);
    assert(telemetry.max_raw_toggles_per_word == 1U);
    assert(close(telemetry.raw_activity_factor, 1.0 / 32.0));
    assert(close(telemetry.dbi_activity_factor, 1.0 / 32.0));
}

void test_dbi_caps_payload_toggles() {
    const std::vector<double> before{0.0};
    const std::vector<double> after{-1.0 / 65536.0};
    const auto telemetry = jarvisx::ElectromagneticFlowLogic::measure_transition(before, after);

    // Q16.16 -1 LSB is the two's-complement word 0xffffffff. Relative to
    // zero this flips all 32 payload bits; ideal payload DBI selects its
    // complement, reducing the payload transition count to zero. The separate
    // DBI control-line transition is deliberately outside this metric.
    assert(telemetry.raw_bit_toggles == 32U);
    assert(telemetry.dbi_bit_toggles == 0U);
    assert(telemetry.max_raw_toggles_per_word == 32U);
    assert(close(telemetry.raw_activity_factor, 1.0));
    assert(close(telemetry.dbi_activity_factor, 0.0));
}

void test_physical_model_is_explicit() {
    jarvisx::SwitchingActivityTelemetry telemetry;
    telemetry.words = 1U;
    telemetry.raw_activity_factor = 0.5;
    telemetry.dbi_activity_factor = 0.25;

    jarvisx::PhysicalSwitchingModel model;
    model.effective_capacitance_f = 1.0e-12;
    model.supply_voltage_v = 1.0;
    model.cycle_frequency_hz = 1.0e9;

    const auto estimate =
        jarvisx::ElectromagneticFlowLogic::estimate_electrical_switching(telemetry, model);
    assert(close(estimate.raw_dynamic_power_w, 5.0e-4));
    assert(close(estimate.dbi_dynamic_power_w, 2.5e-4));
    assert(close(estimate.raw_average_current_a, 5.0e-4));
    assert(close(estimate.dbi_average_current_a, 2.5e-4));
}

void test_dimension_and_nonfinite_rejection() {
    bool dimension_threw = false;
    try {
        (void)jarvisx::ElectromagneticFlowLogic::measure_transition(
            std::vector<double>{0.0}, std::vector<double>{0.0, 1.0});
    } catch (const std::invalid_argument&) {
        dimension_threw = true;
    }
    assert(dimension_threw);

    bool finite_threw = false;
    try {
        (void)jarvisx::ElectromagneticFlowLogic::measure_transition(
            std::vector<double>{0.0}, std::vector<double>{std::nan("")});
    } catch (const std::invalid_argument&) {
        finite_threw = true;
    }
    assert(finite_threw);
}

void test_current_slew_proxy() {
    const double slew = jarvisx::ElectromagneticFlowLogic::current_slew_proxy(
        0.10, 0.13, 1.0e-9);
    assert(close(slew, 3.0e7, 1.0e-5));
}

} // namespace

int main() {
    test_single_q16_lsb_toggle();
    test_dbi_caps_payload_toggles();
    test_physical_model_is_explicit();
    test_dimension_and_nonfinite_rejection();
    test_current_slew_proxy();
    std::cout << "electromagnetic_flow_tests: PASS\n";
    return 0;
}
