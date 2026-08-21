#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {
using fixed_t = std::int32_t;
using wide_t = std::int64_t;
constexpr int SHIFT = 16;
constexpr fixed_t ONE = fixed_t{1} << SHIFT;
constexpr std::size_t D_IN = 128;
constexpr std::size_t D_VOL = 64;
constexpr std::size_t D_VOXELS = D_VOL * D_VOL * D_VOL;

fixed_t sat(wide_t x) {
    if (x > std::numeric_limits<fixed_t>::max()) return std::numeric_limits<fixed_t>::max();
    if (x < std::numeric_limits<fixed_t>::min()) return std::numeric_limits<fixed_t>::min();
    return static_cast<fixed_t>(x);
}

fixed_t qmul(fixed_t a, fixed_t b) {
    return sat((static_cast<wide_t>(a) * b) >> SHIFT);
}

fixed_t qdiv(fixed_t a, fixed_t b) {
    if (b == 0) {
        return a >= 0 ? std::numeric_limits<fixed_t>::max()
                      : std::numeric_limits<fixed_t>::min();
    }
    return sat((static_cast<wide_t>(a) << SHIFT) / b);
}

fixed_t qtanh(fixed_t x) {
    constexpr fixed_t three = 3 * ONE;
    if (x >= three) return ONE;
    if (x <= -three) return -ONE;
    const fixed_t x2 = qmul(x, x);
    const fixed_t c27 = 27 * ONE;
    const fixed_t c9 = 9 * ONE;
    const fixed_t num = qmul(x, sat(static_cast<wide_t>(c27) + x2));
    const fixed_t den = sat(static_cast<wide_t>(c27) + qmul(c9, x2));
    return std::clamp(qdiv(num, den), -ONE, ONE);
}

void test_q16() {
    assert(qmul(ONE / 2, ONE / 2) == ONE / 4);
    assert(qtanh(10 * ONE) == ONE);
    assert(qtanh(-10 * ONE) == -ONE);

    fixed_t previous = -ONE;
    for (int i = -300; i <= 300; ++i) {
        const fixed_t x = static_cast<fixed_t>((static_cast<wide_t>(i) * ONE) / 100);
        const fixed_t y = qtanh(x);
        assert(y >= -ONE && y <= ONE);
        // Permit at most two Q16.16 LSBs of quantization ripple near saturation.
        assert(static_cast<wide_t>(y) + 2 >= previous);
        previous = std::max(previous, y);
    }
}

void test_lift_pool_identity() {
    static_assert(D_VOXELS % D_IN == 0,
                  "volume must evenly tile the state width");
    constexpr std::size_t replicas = D_VOXELS / D_IN;

    std::array<fixed_t, D_IN> state{};
    for (std::size_t i = 0; i < D_IN; ++i) {
        state[i] = static_cast<fixed_t>((static_cast<int>(i) - 64) * 257);
    }

    std::vector<fixed_t> volume(D_VOXELS);
    for (std::size_t v = 0; v < D_VOXELS; ++v) {
        volume[v] = state[v % D_IN];
    }

    std::array<fixed_t, D_IN> pooled{};
    for (std::size_t i = 0; i < D_IN; ++i) {
        wide_t acc = 0;
        for (std::size_t v = i; v < D_VOXELS; v += D_IN) {
            acc += volume[v];
        }
        pooled[i] = sat(acc / static_cast<wide_t>(replicas));
    }
    assert(pooled == state);
}

void test_memory_plan() {
    constexpr std::size_t max_batch = 256;
    constexpr std::size_t volume_bytes =
        max_batch * D_VOXELS * sizeof(fixed_t);
    constexpr std::size_t ping_pong_bytes = 2 * volume_bytes;
    static_assert(volume_bytes == 268435456ULL,
                  "expected 256 MiB per volume buffer");
    static_assert(ping_pong_bytes == 536870912ULL,
                  "expected 512 MiB for ping-pong volumes");
}
}  // namespace

int main() {
    test_q16();
    test_lift_pool_identity();
    test_memory_plan();
    std::cout << "DM-vOmegaXi+ infinity turbo contract tests: PASS\n";
    return 0;
}
