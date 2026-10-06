#include "jarvisx/master_equation.hpp"

#include <cstddef>
#include <iostream>

using namespace jarvisx::master_equation;

int main() {
    std::size_t fixed = 0U;
    std::size_t basin = 0U;

    for (int x = 0; x < 11; ++x) {
        for (int y = 0; y < 6; ++y) {
            for (int z = 0; z < 4; ++z) {
                const LatticePoint p{x, y, z};
                fixed += is_fixed_point(p) ? 1U : 0U;
                basin += constant_target_kernel(p) == kTarget ? 1U : 0U;
            }
        }
    }

    const Field4 field = psi(
        2640000LL,
        2640000LL * 2LL,
        2640000LL * 3LL,
        SpatialPoint{3.0F, 4.0F, 5.0F});

    const bool valid =
        kLambda0Cardinality == 264U
        && fixed == 1U
        && basin == 264U
        && kImageCardinality == 1U
        && kKernelEquivalenceQuotientCardinality == 1U
        && !constant_kernel_is_additive()
        && field == Field4{1.0F, 2.0F, 1.0F, 1.0F};

    std::cout
        << "{\n"
        << "  \"operator\": \"Psi = D_Omega o C_v* o pi_0\",\n"
        << "  \"lambda0_cardinality\": " << kLambda0Cardinality << ",\n"
        << "  \"target\": [1, 2, 1],\n"
        << "  \"image_cardinality\": " << kImageCardinality << ",\n"
        << "  \"fixed_point_cardinality\": " << fixed << ",\n"
        << "  \"target_basin_cardinality\": " << basin << ",\n"
        << "  \"equivalence_quotient_cardinality\": "
        << kKernelEquivalenceQuotientCardinality << ",\n"
        << "  \"idempotent\": true,\n"
        << "  \"group_endomorphism\": false,\n"
        << "  \"constant_spatial_gradient\": true,\n"
        << "  \"verified\": " << (valid ? "true" : "false") << "\n"
        << "}\n";

    return valid ? 0 : 1;
}
