#include "jarvisx/dmso_fused.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    using jarvisx::dmso::Context;
    using jarvisx::dmso::Parameters;
    using jarvisx::dmso::execute_fused;
    using jarvisx::dmso::execute_primitive;
    using jarvisx::dmso::max_abs_error;

    const Context context{
        {0.25, -0.5},
        {0.1, 0.2},
        {0.25, -0.5},
        {0.7, -0.1},
        0.25,
    };
    const Parameters parameters{};
    const auto primitive = execute_primitive(context, parameters);
    const auto fused = execute_fused(context, parameters);
    require(primitive.dispatches == 7, "primitive path must dispatch seven operations");
    require(fused.dispatches == 1, "fused path must dispatch once");
    require(max_abs_error(primitive, fused) == 0.0, "fused path must preserve exact output");

    bool rejected = false;
    try {
        Context invalid = context;
        invalid.stimulus[0] = std::numeric_limits<double>::quiet_NaN();
        static_cast<void>(execute_fused(invalid, parameters));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "non-finite input must be rejected");

    std::cout << "DMSO fused native regression tests passed\n";
    return 0;
}
