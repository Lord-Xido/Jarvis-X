// QSOL: integer-exact toroidal reverberation + read-only deterministic observation.
// C++17; no hardware, network, or rendering authority.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace qsol3d {

constexpr int kEdge = 16;
constexpr int kVoxels = kEdge * kEdge * kEdge;
constexpr int kObserverUnits = 256; // 8 x 8 x 4 pooling grid
constexpr std::int32_t kOne = 1 << 16; // Q16.16
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct Field {
    std::array<std::int32_t, kVoxels> cells{};
};

struct Observation {
    std::array<std::int32_t, kObserverUnits> units{};
    std::uint64_t render_hash{};
    int camera_dx{}, camera_dy{}; // illustrative integer layout offsets
};

inline int wrap(int n) { return (n + kEdge) % kEdge; } // neighbours only +/-1
inline int index(int x, int y, int z) {
    return (wrap(z) * kEdge + wrap(y)) * kEdge + wrap(x);
}

inline void validate(const Field& f) {
    for (std::int32_t q : f.cells)
        if (q < -kOne || q > kOne)
            throw std::invalid_argument("QSOL field outside normalized Q16.16 range");
}

// A = 3/4 I + 3/16 P, where P is the periodic six-neighbour mean.
// B = 1/16 I.  ||A||_infinity <= 15/16; bounded input <= 1.
// Every division uses C++ signed integer truncation toward zero.
inline Field advance(const Field& state, const Field& external) {
    validate(state);
    validate(external);
    Field next;
    for (int z = 0; z < kEdge; ++z)
        for (int y = 0; y < kEdge; ++y)
            for (int x = 0; x < kEdge; ++x) {
                const int p = index(x, y, z);
                const std::int64_t neighbours =
                    std::int64_t(state.cells[index(x-1,y,z)]) +
                    state.cells[index(x+1,y,z)] +
                    state.cells[index(x,y-1,z)] +
                    state.cells[index(x,y+1,z)] +
                    state.cells[index(x,y,z-1)] +
                    state.cells[index(x,y,z+1)];
                // Evaluate as ONE rational expression; avoid intermediate rounding.
                const std::int64_t numerator =
                    72LL * state.cells[p] + 3LL * neighbours + 6LL * external.cells[p];
                next.cells[p] = static_cast<std::int32_t>(numerator / 96LL);
            }
    validate(next);
    return next;
}

inline std::uint64_t hash_word(std::uint64_t h, std::uint32_t word) {
    // Explicit little-endian byte order, wraparound arithmetic defined for uint64.
    for (int i = 0; i < 4; ++i) {
        h = (h ^ ((word >> (8*i)) & 255U)) * kFnvPrime;
    }
    return h;
}

inline std::uint64_t state_hash(const Field& f) {
    std::uint64_t h = kFnvOffset;
    for (std::int32_t q : f.cells)
        h = hash_word(h, static_cast<std::uint32_t>(q));
    return h;
}

// C: 8x8x4 non-overlapping spatial averages; D: same average of external.
// y = (3 C x + D u)/4. No writes or mutation authority.
inline Observation observe(const Field& state, const Field& external, std::uint32_t frame) {
    validate(state);
    validate(external);
    Observation out;
    std::uint64_t h = kFnvOffset;
    for (int bz = 0; bz < 4; ++bz)
        for (int by = 0; by < 8; ++by)
            for (int bx = 0; bx < 8; ++bx) {
                std::int64_t sum_state = 0, sum_input = 0;
                for (int dz = 0; dz < 4; ++dz)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            int p = index(2*bx+dx,2*by+dy,4*bz+dz);
                            sum_state += state.cells[p];
                            sum_input += external.cells[p];
                        }
                const int i = (bz*8+by)*8+bx;
                out.units[i] = static_cast<std::int32_t>((3*sum_state+sum_input)/64);
                h = hash_word(h, static_cast<std::uint32_t>(out.units[i]));
            }
    out.render_hash = hash_word(h, frame);
    out.camera_dx = int(out.render_hash % 13ULL) - 6;
    out.camera_dy = int((out.render_hash >> 16) % 13ULL) - 6;
    return out;
}

// Pure exact-int framebuffer adapter. RGB channels are externally authoritative
// samples; the QSOL observer is downstream and never writes to them.
inline std::int32_t excitation_from_rgb(std::uint8_t r, std::uint8_t b) {
    return ((std::int32_t(r)-std::int32_t(b)) * kOne) / 255;
}

} // namespace qsol3d
