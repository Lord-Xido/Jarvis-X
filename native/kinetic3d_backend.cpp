#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

std::size_t block_index(
    int x,
    int y,
    int z,
    int sx,
    int sy,
    int sz,
    int factor,
    int &bx_count,
    int &by_count,
    int &bz_count
) {
    bx_count = (sx + factor - 1) / factor;
    by_count = (sy + factor - 1) / factor;
    bz_count = (sz + factor - 1) / factor;
    const int bx = x / factor;
    const int by = y / factor;
    const int bz = z / factor;
    return static_cast<std::size_t>((bz * by_count + by) * bx_count + bx);
}

}  // namespace

extern "C" int jx_kinetic3d_step(
    const double *current,
    const double *prediction,
    std::size_t count,
    int sx,
    int sy,
    int sz,
    double active_threshold,
    int coarse_factor,
    double refine_threshold,
    double *residual,
    double *reconstructed,
    std::uint8_t *active_mask,
    double *coarse_per_cell,
    double *fine_per_cell
) {
    if (
        current == nullptr || prediction == nullptr || residual == nullptr ||
        reconstructed == nullptr || active_mask == nullptr || coarse_per_cell == nullptr ||
        fine_per_cell == nullptr
    ) {
        return 1;
    }
    if (sx < 1 || sy < 1 || sz < 1 || coarse_factor < 1) {
        return 2;
    }
    const std::size_t expected = static_cast<std::size_t>(sx) * static_cast<std::size_t>(sy) *
                                 static_cast<std::size_t>(sz);
    if (expected != count) {
        return 3;
    }
    if (!std::isfinite(active_threshold) || active_threshold < 0.0 ||
        !std::isfinite(refine_threshold) || refine_threshold < 0.0) {
        return 4;
    }

    const int bx_count = (sx + coarse_factor - 1) / coarse_factor;
    const int by_count = (sy + coarse_factor - 1) / coarse_factor;
    const int bz_count = (sz + coarse_factor - 1) / coarse_factor;
    const std::size_t block_count = static_cast<std::size_t>(bx_count) *
                                    static_cast<std::size_t>(by_count) *
                                    static_cast<std::size_t>(bz_count);
    std::vector<double> block_sum(block_count, 0.0);
    std::vector<std::size_t> block_active(block_count, 0);

    for (std::size_t index = 0; index < count; ++index) {
        residual[index] = current[index] - prediction[index];
        reconstructed[index] = prediction[index];
        coarse_per_cell[index] = 0.0;
        fine_per_cell[index] = 0.0;
        const bool active = std::abs(residual[index]) > active_threshold;
        active_mask[index] = active ? 1U : 0U;
        if (!active) {
            continue;
        }

        const int plane = sx * sy;
        const int z = static_cast<int>(index) / plane;
        const int rem = static_cast<int>(index) % plane;
        const int y = rem / sx;
        const int x = rem % sx;
        const std::size_t block = static_cast<std::size_t>(
            ((z / coarse_factor) * by_count + (y / coarse_factor)) * bx_count +
            (x / coarse_factor)
        );
        block_sum[block] += residual[index];
        block_active[block] += 1;
    }

    std::vector<double> block_mean(block_count, 0.0);
    for (std::size_t block = 0; block < block_count; ++block) {
        if (block_active[block] != 0U) {
            block_mean[block] = block_sum[block] / static_cast<double>(block_active[block]);
        }
    }

    for (std::size_t index = 0; index < count; ++index) {
        if (active_mask[index] == 0U) {
            continue;
        }
        const int plane = sx * sy;
        const int z = static_cast<int>(index) / plane;
        const int rem = static_cast<int>(index) % plane;
        const int y = rem / sx;
        const int x = rem % sx;
        const std::size_t block = static_cast<std::size_t>(
            ((z / coarse_factor) * by_count + (y / coarse_factor)) * bx_count +
            (x / coarse_factor)
        );
        const double coarse = block_mean[block];
        coarse_per_cell[index] = coarse;
        reconstructed[index] += coarse;

        const double correction = residual[index] - coarse;
        if (std::abs(correction) > refine_threshold) {
            reconstructed[index] += correction;
            fine_per_cell[index] = correction;
        }
    }
    return 0;
}
