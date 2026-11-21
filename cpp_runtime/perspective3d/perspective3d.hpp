#pragma once

// DM-PTVP v0.1: bounded, dependency-free C++17 geometric/evidence reference.
// All volume transforms are exact index permutations; no ANN inference is made.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>

namespace jarvisx { namespace perspective3d {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

struct Rotation {
    // Row-major signed-permutation matrix with determinant +1: a proper cube rotation.
    std::array<int, 9> m{};
    bool operator==(const Rotation& other) const { return m == other.m; }
};

inline int determinant(const Rotation& r) {
    const auto& m = r.m;
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

inline std::vector<Rotation> proper_cube_rotations() {
    std::vector<Rotation> result;
    std::array<int, 3> permutation{{0, 1, 2}};
    do {
        for (int sx : {-1, 1}) for (int sy : {-1, 1}) for (int sz : {-1, 1}) {
            Rotation r;
            const int signs[3] = {sx, sy, sz};
            for (int row = 0; row < 3; ++row) r.m[static_cast<std::size_t>(row * 3 + permutation[static_cast<std::size_t>(row)])] = signs[row];
            if (determinant(r) == 1) result.push_back(r);
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return result;
}

inline Rotation inverse(const Rotation& r) {
    Rotation inv;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            inv.m[static_cast<std::size_t>(row * 3 + col)] = r.m[static_cast<std::size_t>(col * 3 + row)];
    return inv;
}

inline Rotation compose(const Rotation& a, const Rotation& b) {
    Rotation result;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            for (int k = 0; k < 3; ++k)
                result.m[static_cast<std::size_t>(row * 3 + col)] +=
                    a.m[static_cast<std::size_t>(row * 3 + k)] * b.m[static_cast<std::size_t>(k * 3 + col)];
    return result;
}

inline Vec3 apply(const Rotation& r, const Vec3& x) {
    const auto& a = r.m;
    return {a[0]*x.x + a[1]*x.y + a[2]*x.z,
            a[3]*x.x + a[4]*x.y + a[5]*x.z,
            a[6]*x.x + a[7]*x.y + a[8]*x.z};
}

struct Cube {
    std::size_t n;
    std::vector<std::uint8_t> values;

    explicit Cube(std::size_t edge) : n(edge), values() {
        // Explicit memory boundary: this is a finite reference, not a virtual 1GB^3 allocation.
        if (edge == 0 || edge > 128) throw std::invalid_argument("cube edge must be 1..128");
        values.resize(edge * edge * edge, 0);
    }
    std::size_t index(std::size_t x, std::size_t y, std::size_t z) const {
        if (x >= n || y >= n || z >= n) throw std::out_of_range("cube coordinate");
        return (z * n + y) * n + x;
    }
    std::uint8_t& at(std::size_t x, std::size_t y, std::size_t z) { return values[index(x,y,z)]; }
    std::uint8_t at(std::size_t x, std::size_t y, std::size_t z) const { return values[index(x,y,z)]; }
    bool operator==(const Cube& other) const { return n == other.n && values == other.values; }
};

inline Cube rotate(const Cube& src, const Rotation& r) {
    Cube dst(src.n);
    // A signed axis permutation is a bijection of the discrete cube.
    for (std::size_t z=0; z<src.n; ++z)
        for (std::size_t y=0; y<src.n; ++y)
            for (std::size_t x=0; x<src.n; ++x) {
                const std::size_t in[3] = {x, y, z};
                std::size_t out[3] = {};
                for (int row=0; row<3; ++row)
                    for (int col=0; col<3; ++col) {
                        const int v = r.m[static_cast<std::size_t>(row*3+col)];
                        if (v != 0) out[row] = v > 0 ? in[col] : src.n - 1 - in[col];
                    }
                dst.at(out[0],out[1],out[2]) = src.at(x,y,z);
            }
    return dst;
}

inline std::array<std::uint64_t, 256> histogram(const Cube& c) {
    std::array<std::uint64_t, 256> h{};
    for (auto byte : c.values) ++h[byte];
    return h;
}

inline Vec3 centred_weighted_centroid(const Cube& c) {
    const double centre = (static_cast<double>(c.n) - 1.0) / 2.0;
    double sx = 0.0, sy = 0.0, sz = 0.0, total = 0.0;
    for (std::size_t z=0; z<c.n; ++z)
        for (std::size_t y=0; y<c.n; ++y)
            for (std::size_t x=0; x<c.n; ++x) {
                const double w = c.at(x,y,z);
                sx += w*(static_cast<double>(x)-centre);
                sy += w*(static_cast<double>(y)-centre);
                sz += w*(static_cast<double>(z)-centre);
                total += w;
            }
    if (total == 0.0) return {};
    return {sx/total, sy/total, sz/total};
}

// Orthographic occupancy projection: non-injective map cube -> 2D sensor array.
inline std::vector<std::uint8_t> project_xy_occupancy(const Cube& c) {
    std::vector<std::uint8_t> output(c.n*c.n, 0);
    for (std::size_t y=0; y<c.n; ++y)
        for (std::size_t x=0; x<c.n; ++x)
            for (std::size_t z=0; z<c.n; ++z)
                if (c.at(x,y,z) != 0) { output[y*c.n+x] = 1; break; }
    return output;
}

struct Image2D {
    std::size_t width = 0, height = 0;
    std::vector<std::uint8_t> pixels;
    Image2D(std::size_t w, std::size_t h, std::vector<std::uint8_t> p)
        : width(w), height(h), pixels(std::move(p)) {
        if (w==0 || h==0 || w > 1000000 / h || pixels.size()!=w*h)
            throw std::invalid_argument("invalid bounded image dimensions/pixels");
    }
    bool operator==(const Image2D& other) const {
        return width == other.width && height == other.height && pixels == other.pixels;
    }
};

inline Image2D clockwise_90(const Image2D& src) {
    Image2D out(src.height, src.width, std::vector<std::uint8_t>(src.pixels.size(), 0));
    for (std::size_t y=0; y<src.height; ++y)
        for (std::size_t x=0; x<src.width; ++x)
            out.pixels[x*out.width + (src.height-1-y)] = src.pixels[y*src.width+x];
    return out;
}

enum class Verdict { verified, refuted, underdetermined };

// Same calibrated projection can disprove identity if it differs, but cannot establish it.
inline Verdict projection_identity(const std::vector<std::uint8_t>& a,
                                   const std::vector<std::uint8_t>& b) {
    return a == b ? Verdict::underdetermined : Verdict::refuted;
}

// Exact byte-state equality is decidable once full bounded ground-truth cubes are supplied.
inline Verdict exact_cube_identity(const Cube& a, const Cube& b) {
    return a == b ? Verdict::verified : Verdict::refuted;
}

}} // namespace jarvisx::perspective3d
