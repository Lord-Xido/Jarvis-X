#include "spatial_engine3d.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>

// Use active checks even in NDEBUG release builds.
#define CHECK(x) do { if(!(x)) throw std::runtime_error("CHECK FAILED: " #x); } while(false)
template<class F> void must_throw(F f) {
    bool thrown = false;
    try { f(); } catch (const std::exception&) { thrown = true; }
    CHECK(thrown);
}
int main() {
    using namespace JarvisX;
    CHECK(to_q1616_checked(0.5) == 32768);
    CHECK(to_q1616_checked(-0.5) == -32768);
    CHECK(mul_q1616_checked(3*Q_SCALE, to_q1616_checked(0.5)) == 98304);
    CHECK(mul_q1616_checked(-3*Q_SCALE, to_q1616_checked(0.5)) == -98304);
    must_throw([] { to_q1616_checked(33000); });
    must_throw([] { to_q1616_checked(std::numeric_limits<double>::infinity()); });
    must_throw([] { mul_q1616_checked(INT32_MAX, INT32_MAX); });
    SpatialEngine3D grid;
    grid.generate_sparse_grid(1000000);
    CHECK(grid.size() == 1000000);
    CHECK((grid.points().front() == Point3D{98304,163840,229376}));
    CHECK((grid.points().back() == Point3D{150*65536,250*65536,350*65536}));
    // 1,000,000 unique coordinate triples follow by base-100 indexing;
    // sample all coordinate points as a strong regression test.
    std::set<std::tuple<Q1616,Q1616,Q1616>> distinct;
    for (const auto& p : grid.points()) distinct.insert({p.x,p.y,p.z});
    CHECK(distinct.size() == grid.size());
    const auto initial = grid.fnv1a64();
    grid.apply_contraction(0.5);
    CHECK((grid.points().front() == Point3D{49152,81920,114688}));
    CHECK((grid.points().back() == Point3D{75*65536,125*65536,175*65536}));
    CHECK(initial != grid.fnv1a64());
    auto once = grid.fnv1a64();
    grid.apply_contraction(0.5);
    CHECK(grid.fnv1a64() != once);
    must_throw([&] { grid.apply_contraction(1.0); });
    must_throw([&] { grid.apply_contraction(-0.1); });
    must_throw([&] { grid.generate_sparse_grid(1000001); });
    CHECK(grid.size() == 1000000);
    grid.generate_sparse_grid(3);
    CHECK(grid.size() == 3);
    grid.generate_sparse_grid(0);
    CHECK(grid.size() == 0);
    std::cout << "PASS: Q16.16 bounds, deterministic 3D point grid, 1M uniqueness, contraction, rollback, empty grid\n";
}
