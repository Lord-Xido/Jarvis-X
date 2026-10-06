#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace jarvisx::master_equation {

// Canonical finite base lattice:
//   Lambda_0 = Z_11 x Z_6 x Z_4
//   |Lambda_0| = 264
struct LatticePoint {
    int x{0};
    int y{0};
    int z{0};

    constexpr bool operator==(const LatticePoint& other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }

    constexpr bool operator!=(const LatticePoint& other) const noexcept {
        return !(*this == other);
    }
};

struct SpatialPoint {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
};

struct Field4 {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
    float w{1.0F};

    constexpr bool operator==(const Field4& other) const noexcept {
        return x == other.x && y == other.y && z == other.z && w == other.w;
    }
};

using Jacobian4x3 = std::array<float, 12>;

inline constexpr std::size_t kLambda0Cardinality = 11U * 6U * 4U;
inline constexpr LatticePoint kTarget{1, 2, 1};
inline constexpr std::size_t kImageCardinality = 1U;
inline constexpr std::size_t kFixedPointCardinality = 1U;
inline constexpr std::size_t kTargetBasinCardinality = kLambda0Cardinality;
inline constexpr std::size_t kKernelEquivalenceQuotientCardinality = 1U;

constexpr int positive_mod(std::int64_t value, int modulus) noexcept {
    const std::int64_t m = static_cast<std::int64_t>(modulus);
    const std::int64_t r = value % m;
    return static_cast<int>(r < 0 ? r + m : r);
}

// Operational representation of the canonical projection pi_0.
// Lambda_infinity is not physically instantiated; its integer-valued
// coordinate functionals are reduced into the finite base lattice.
constexpr LatticePoint project_to_base(
    std::int64_t f1,
    std::int64_t f2,
    std::int64_t f3) noexcept {
    return {
        positive_mod(f1, 11),
        positive_mod(f2, 6),
        positive_mod(f3, 4)
    };
}

constexpr LatticePoint normalize(LatticePoint p) noexcept {
    return project_to_base(p.x, p.y, p.z);
}

// C_{v*}(lambda) = v* for every lambda in Lambda_0.
constexpr LatticePoint constant_target_kernel(LatticePoint) noexcept {
    return kTarget;
}

constexpr bool is_fixed_point(LatticePoint p) noexcept {
    return normalize(p) == kTarget;
}

constexpr bool equivalent_under_constant_kernel(
    LatticePoint a,
    LatticePoint b) noexcept {
    return constant_target_kernel(a) == constant_target_kernel(b);
}

constexpr LatticePoint add_mod(LatticePoint a, LatticePoint b) noexcept {
    return project_to_base(
        static_cast<std::int64_t>(a.x) + b.x,
        static_cast<std::int64_t>(a.y) + b.y,
        static_cast<std::int64_t>(a.z) + b.z);
}

// A non-zero constant target map is idempotent as a function, but it is
// generally not an additive group endomorphism. This predicate encodes that
// distinction and protects the implementation from using a group-kernel
// quotient for this map.
constexpr bool constant_kernel_is_additive() noexcept {
    const LatticePoint zero{0, 0, 0};
    return constant_target_kernel(add_mod(zero, zero))
        == add_mod(constant_target_kernel(zero), constant_target_kernel(zero));
}

// D_Omega for the total-collapse limit. The fourth component is a homogeneous
// field/alpha coordinate so Psi maps into R^4 while v* remains the canonical
// three-component lattice target.
constexpr Field4 decode_constant_field(
    LatticePoint collapsed,
    SpatialPoint) noexcept {
    return {
        static_cast<float>(collapsed.x),
        static_cast<float>(collapsed.y),
        static_cast<float>(collapsed.z),
        1.0F
    };
}

// Psi = D_Omega o C_{v*} o pi_0.
constexpr Field4 psi(
    std::int64_t f1,
    std::int64_t f2,
    std::int64_t f3,
    SpatialPoint r) noexcept {
    return decode_constant_field(
        constant_target_kernel(project_to_base(f1, f2, f3)),
        r);
}

// The constant decoder has zero spatial Jacobian everywhere.
constexpr Jacobian4x3 spatial_jacobian() noexcept {
    return {0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.0F};
}

} // namespace jarvisx::master_equation
