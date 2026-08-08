#include "jarvisx/dmso_fused.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::uint64_t parse_repetitions(int argc, char** argv) {
    std::uint64_t repetitions = 500000;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--repetitions") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--repetitions requires a value");
            }
            const std::string value = argv[++index];
            const auto parsed = std::stoull(value);
            if (parsed == 0) {
                throw std::invalid_argument("repetitions must be positive");
            }
            repetitions = parsed;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    return repetitions;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        using jarvisx::dmso::Context;
        using jarvisx::dmso::Parameters;
        using jarvisx::dmso::execute_fused;
        using jarvisx::dmso::execute_primitive;
        using jarvisx::dmso::max_abs_error;

        const auto repetitions = parse_repetitions(argc, argv);
        const Context context{
            {0.25, -0.5, 0.125, -0.25},
            {0.1, 0.2, -0.15, 0.05},
            {0.25, -0.5, 0.125, -0.25},
            {0.7, -0.1, 0.3, 0.2},
            0.25,
        };
        const Parameters parameters{};
        const auto primitive_reference = execute_primitive(context, parameters);
        const auto fused_reference = execute_fused(context, parameters);
        const double error = max_abs_error(primitive_reference, fused_reference);
        if (error != 0.0) {
            throw std::runtime_error("fused semantic verification failed");
        }

        volatile double checksum = 0.0;
        const auto primitive_start = std::chrono::steady_clock::now();
        for (std::uint64_t index = 0; index < repetitions; ++index) {
            checksum += execute_primitive(context, parameters).value[0];
        }
        const auto primitive_end = std::chrono::steady_clock::now();

        const auto fused_start = std::chrono::steady_clock::now();
        for (std::uint64_t index = 0; index < repetitions; ++index) {
            checksum += execute_fused(context, parameters).value[0];
        }
        const auto fused_end = std::chrono::steady_clock::now();

        const auto primitive_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      primitive_end - primitive_start)
                                      .count();
        const auto fused_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  fused_end - fused_start)
                                  .count();
        const double primitive_per_call = static_cast<double>(primitive_ns) /
                                          static_cast<double>(repetitions);
        const double fused_per_call = static_cast<double>(fused_ns) /
                                      static_cast<double>(repetitions);
        const double speedup = primitive_per_call / fused_per_call;

        std::cout << std::fixed << std::setprecision(6)
                  << "{\n"
                  << "  \"repetitions\": " << repetitions << ",\n"
                  << "  \"primitive_dispatches\": 7,\n"
                  << "  \"fused_dispatches\": 1,\n"
                  << "  \"dispatch_reduction_ratio\": 7.0,\n"
                  << "  \"primitive_ns_per_call\": " << primitive_per_call << ",\n"
                  << "  \"fused_ns_per_call\": " << fused_per_call << ",\n"
                  << "  \"measured_speedup\": " << speedup << ",\n"
                  << "  \"output_max_abs_error\": " << error << ",\n"
                  << "  \"checksum\": " << checksum << "\n"
                  << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "dmso fused benchmark error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
