#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace JarvisX {
using Q1616 = std::int32_t;
constexpr std::int64_t Q_SCALE = 65536;
constexpr std::size_t MAX_SPARSE_POINTS = 1000000;
constexpr std::size_t GRID_SIDE = 100;

// Checked conversion. Direct float -> int32 overflow is undefined behavior.
inline Q1616 to_q1616_checked(double v) {
    if (!std::isfinite(v) || v * static_cast<double>(Q_SCALE) >
         static_cast<double>(std::numeric_limits<Q1616>::max()) ||
         v * static_cast<double>(Q_SCALE) <
         static_cast<double>(std::numeric_limits<Q1616>::min())) {
        throw std::out_of_range("Q16.16 conversion out of range");
    }
    return static_cast<Q1616>(std::trunc(v * static_cast<double>(Q_SCALE)));
}
inline double to_double(Q1616 q) noexcept {
    return static_cast<double>(q) / static_cast<double>(Q_SCALE);
}
inline Q1616 mul_q1616_checked(Q1616 a, Q1616 b) {
    // Signed division is defined to truncate toward zero in C++17.
    const std::int64_t product = (static_cast<std::int64_t>(a) * b) / Q_SCALE;
    if (product > std::numeric_limits<Q1616>::max() ||
        product < std::numeric_limits<Q1616>::min()) {
        throw std::overflow_error("Q16.16 multiplication overflow");
    }
    return static_cast<Q1616>(product);
}
struct Point3D {
    Q1616 x{}, y{}, z{};
    bool operator==(const Point3D& p) const noexcept {
        return x == p.x && y == p.y && z == p.z;
    }
};

class SpatialEngine3D {
    std::vector<Point3D> points_;
public:
    void generate_sparse_grid(std::size_t count) {
        if (count > MAX_SPARSE_POINTS) {
            throw std::out_of_range("prototype supports at most 1,000,000 distinct points");
        }
        std::vector<Point3D> generated;
        generated.reserve(count);
        // Every grid coordinate is unique and nonzero. Spatial intervals are 1.5, 2.5, 3.5 units.
        // The regular grid has gaps in the ambient coordinate system; no spatial hash/octree is claimed.
        for (std::size_t i = 0; i < count; ++i) {
            const auto ix = 1 + i % GRID_SIDE;
            const auto iy = 1 + (i / GRID_SIDE) % GRID_SIDE;
            const auto iz = 1 + i / (GRID_SIDE * GRID_SIDE);
            generated.push_back({
                static_cast<Q1616>(ix * (3 * Q_SCALE / 2)),
                static_cast<Q1616>(iy * (5 * Q_SCALE / 2)),
                static_cast<Q1616>(iz * (7 * Q_SCALE / 2))
            });
        }
        points_.swap(generated);
    }
    void apply_contraction(double scale) {
        if (!std::isfinite(scale) || scale < 0.0 || scale >= 1.0) {
            throw std::invalid_argument("contraction scale must satisfy 0 <= scale < 1");
        }
        const Q1616 q = to_q1616_checked(scale);
        // Transactional: exceptions cannot leave a partly modified manifold.
        std::vector<Point3D> candidate = points_;
        for (auto& p : candidate) {
            p.x = mul_q1616_checked(p.x, q);
            p.y = mul_q1616_checked(p.y, q);
            p.z = mul_q1616_checked(p.z, q);
        }
        points_.swap(candidate);
    }
    const std::vector<Point3D>& points() const noexcept { return points_; }
    std::size_t size() const noexcept { return points_.size(); }

    // Deterministic byte-independent fingerprint of logical 32-bit coordinates.
    std::uint64_t fnv1a64() const noexcept {
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (const auto& p : points_) {
            for (Q1616 component : {p.x, p.y, p.z}) {
                const auto v = static_cast<std::uint32_t>(component);
                for (unsigned shift=0; shift<32; shift+=8) {
                    hash ^= static_cast<std::uint8_t>(v >> shift);
                    hash *= UINT64_C(1099511628211);
                }
            }
        }
        return hash;
    }
};
} // namespace JarvisX
