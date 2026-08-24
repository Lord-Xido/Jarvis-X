#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jarvisx {

// Deterministic Q16.16 projection used only to measure logical switching
// activity. The authoritative state-space engine may retain a different
// numerical representation; this adapter makes the bit-level observation
// contract explicit instead of pretending floating-point object bytes are a
// stable hardware-independent logic encoding.
struct Q16_16SwitchWord {
    std::int32_t raw{};

    static Q16_16SwitchWord from_double(double value) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("switching telemetry requires finite state values");
        }
        constexpr double scale = 65536.0;
        constexpr double minimum =
            static_cast<double>(std::numeric_limits<std::int32_t>::min()) / scale;
        constexpr double maximum =
            static_cast<double>(std::numeric_limits<std::int32_t>::max()) / scale;
        const double bounded = std::clamp(value, minimum, maximum);
        const long long quantized = static_cast<long long>(std::llround(bounded * scale));
        return {static_cast<std::int32_t>(quantized)};
    }
};

struct SwitchingActivityTelemetry {
    std::size_t words{};
    std::uint64_t raw_bit_toggles{};
    std::uint64_t dbi_bit_toggles{};
    std::uint32_t max_raw_toggles_per_word{};
    double raw_activity_factor{};
    double dbi_activity_factor{};
};

struct ElectricalSwitchingEstimate {
    double raw_dynamic_power_w{};
    double dbi_dynamic_power_w{};
    double raw_average_current_a{};
    double dbi_average_current_a{};
};

struct PhysicalSwitchingModel {
    // Aggregate effective switched capacitance for the modeled domain. This is
    // intentionally supplied by a hardware model or measurement rather than
    // inferred from software state.
    double effective_capacitance_f{};
    double supply_voltage_v{};
    double cycle_frequency_hz{};

    void validate() const {
        if (!(effective_capacitance_f > 0.0) || !std::isfinite(effective_capacitance_f)) {
            throw std::invalid_argument("effective capacitance must be finite and positive");
        }
        if (!(supply_voltage_v > 0.0) || !std::isfinite(supply_voltage_v)) {
            throw std::invalid_argument("supply voltage must be finite and positive");
        }
        if (!(cycle_frequency_hz > 0.0) || !std::isfinite(cycle_frequency_hz)) {
            throw std::invalid_argument("cycle frequency must be finite and positive");
        }
    }
};

class ElectromagneticFlowLogic {
public:
    // Counts state transitions in a canonical 32-bit Q16.16 logic image.
    // DBI telemetry applies the ideal per-word transform min(H, 32-H), so its
    // result is an upper-level switching bound, not a claim that the executing
    // CPU/GPU interconnect actually implements Data Bus Inversion.
    static SwitchingActivityTelemetry measure_transition(
        const std::vector<double>& before,
        const std::vector<double>& after) {
        if (before.size() != after.size()) {
            throw std::invalid_argument("switching telemetry state dimensions must match");
        }

        SwitchingActivityTelemetry telemetry;
        telemetry.words = before.size();
        for (std::size_t index = 0; index < before.size(); ++index) {
            const auto lhs = Q16_16SwitchWord::from_double(before[index]);
            const auto rhs = Q16_16SwitchWord::from_double(after[index]);
            const std::uint32_t lhs_bits = static_cast<std::uint32_t>(lhs.raw);
            const std::uint32_t rhs_bits = static_cast<std::uint32_t>(rhs.raw);
            const std::uint32_t toggles = popcount32(lhs_bits ^ rhs_bits);
            const std::uint32_t dbi_toggles = std::min<std::uint32_t>(toggles, 32U - toggles);

            telemetry.raw_bit_toggles += static_cast<std::uint64_t>(toggles);
            telemetry.dbi_bit_toggles += static_cast<std::uint64_t>(dbi_toggles);
            telemetry.max_raw_toggles_per_word =
                std::max(telemetry.max_raw_toggles_per_word, toggles);
        }

        if (telemetry.words > 0U) {
            const double total_bits = static_cast<double>(telemetry.words) * 32.0;
            telemetry.raw_activity_factor =
                static_cast<double>(telemetry.raw_bit_toggles) / total_bits;
            telemetry.dbi_activity_factor =
                static_cast<double>(telemetry.dbi_bit_toggles) / total_bits;
        }
        return telemetry;
    }

    // Maps measured logical activity into the standard first-order digital
    // switching model P_dyn = alpha * C_eff * V_dd^2 * f. This is an electrical
    // estimate only. It does not solve package/interconnect Maxwell fields or
    // establish electromagnetic emissions fidelity.
    static ElectricalSwitchingEstimate estimate_electrical_switching(
        const SwitchingActivityTelemetry& telemetry,
        const PhysicalSwitchingModel& model) {
        model.validate();
        const double scale = model.effective_capacitance_f *
                             model.supply_voltage_v *
                             model.supply_voltage_v *
                             model.cycle_frequency_hz;

        ElectricalSwitchingEstimate estimate;
        estimate.raw_dynamic_power_w = telemetry.raw_activity_factor * scale;
        estimate.dbi_dynamic_power_w = telemetry.dbi_activity_factor * scale;
        estimate.raw_average_current_a =
            estimate.raw_dynamic_power_w / model.supply_voltage_v;
        estimate.dbi_average_current_a =
            estimate.dbi_dynamic_power_w / model.supply_voltage_v;
        return estimate;
    }

    static double current_slew_proxy(double previous_current_a,
                                     double current_a,
                                     double delta_t_s) {
        if (!std::isfinite(previous_current_a) || !std::isfinite(current_a)) {
            throw std::invalid_argument("current slew proxy requires finite current values");
        }
        if (!(delta_t_s > 0.0) || !std::isfinite(delta_t_s)) {
            throw std::invalid_argument("current slew proxy requires finite positive delta_t");
        }
        return std::abs(current_a - previous_current_a) / delta_t_s;
    }

private:
    static std::uint32_t popcount32(std::uint32_t value) noexcept {
        std::uint32_t count = 0U;
        while (value != 0U) {
            value &= value - 1U;
            ++count;
        }
        return count;
    }
};

} // namespace jarvisx
