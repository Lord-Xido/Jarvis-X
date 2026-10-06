#include "jarvisx/master_equation.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>

using namespace jarvisx::master_equation;

int main() {
    static_assert(kLambda0Cardinality == 264U, "Lambda_0 cardinality must be 264");
    static_assert(kImageCardinality == 1U, "constant map image is a singleton");
    static_assert(kFixedPointCardinality == 1U, "constant target has one fixed point");
    static_assert(kTargetBasinCardinality == 264U, "all 264 states flow to v*");
    static_assert(kKernelEquivalenceQuotientCardinality == 1U,
                  "equivalence quotient collapses to one class");
    static_assert(!constant_kernel_is_additive(),
                  "non-zero constant map must not be treated as a group endomorphism");

    const auto projected = project_to_base(-1, 7, 5);
    assert((projected == LatticePoint{10, 1, 1}));

    std::size_t states = 0U;
    std::size_t fixed = 0U;
    std::size_t target_preimage = 0U;

    for (int x = 0; x < 11; ++x) {
        for (int y = 0; y < 6; ++y) {
            for (int z = 0; z < 4; ++z) {
                const LatticePoint p{x, y, z};
                const LatticePoint c1 = constant_target_kernel(p);
                const LatticePoint c2 = constant_target_kernel(c1);

                ++states;
                assert(c1 == kTarget);
                assert(c2 == c1); // C^2 = C

                if (is_fixed_point(p)) {
                    ++fixed;
                }
                if (c1 == kTarget) {
                    ++target_preimage;
                }

                assert(equivalent_under_constant_kernel(p, LatticePoint{0, 0, 0}));
            }
        }
    }

    assert(states == 264U);
    assert(fixed == 1U);
    assert(target_preimage == 264U);

    const SpatialPoint r0{0.0F, 0.0F, 0.0F};
    const SpatialPoint r1{100.0F, -37.0F, 9.0F};

    const Field4 a = psi(0, 0, 0, r0);
    const Field4 b = psi(999999999LL, -888888888LL, 777777777LL, r1);

    assert((a == Field4{1.0F, 2.0F, 1.0F, 1.0F}));
    assert(b == a);

    for (float value : spatial_jacobian()) {
        assert(value == 0.0F);
    }

    std::cout
        << "master-equation invariants verified: "
        << "domain=264 image=1 fixed=1 basin=264 quotient=1 "
        << "C^2=C gradient=0\n";

    return 0;
}
