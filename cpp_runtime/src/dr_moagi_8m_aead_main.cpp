#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDefaultWidth = 4000U;
constexpr std::size_t kDefaultHeight = 2000U;
constexpr std::size_t kLatentEdge = 32U;
constexpr std::size_t kLatentCount = kLatentEdge * kLatentEdge * kLatentEdge;
constexpr double kLambda = 0.985;
constexpr double kDiffusion = 0.020;
constexpr double kOmegaBeta = 0.88;
constexpr double kCommitSlack = 1.0e-15;
constexpr std::size_t kFoldSteps = 6U;

struct Metrics {
    double mse{0.0};
    double mae{0.0};
    double max_error{0.0};
};

struct State {
    std::vector<double> latent;
    std::vector<double> omega;
    Metrics metrics;
    bool committed{false};
    bool bounded{true};
    bool finite{true};
};

std::size_t wrap(long value, std::size_t n) {
    const long nn = static_cast<long>(n);
    long r = value % nn;
    if (r < 0L) {
        r += nn;
    }
    return static_cast<std::size_t>(r);
}

std::size_t index3(std::size_t x, std::size_t y, std::size_t z) {
    return (z * kLatentEdge + y) * kLatentEdge + x;
}

std::array<std::size_t, 3U> pixel_to_latent(std::size_t pixel_index) {
    const std::size_t x = pixel_index % kLatentEdge;
    const std::size_t y = (pixel_index / kLatentEdge) % kLatentEdge;
    const std::size_t z =
        (pixel_index / (kLatentEdge * kLatentEdge)) % kLatentEdge;
    return {x, y, z};
}

double source_pixel(std::size_t x, std::size_t y, std::size_t width, std::size_t height) {
    const double nx = static_cast<double>(x) /
        static_cast<double>(std::max<std::size_t>(width - 1U, 1U));
    const double ny = static_cast<double>(y) /
        static_cast<double>(std::max<std::size_t>(height - 1U, 1U));
    const std::uint64_t mix =
        (static_cast<std::uint64_t>(x) * 0x9E3779B185EBCA87ULL) ^
        (static_cast<std::uint64_t>(y) * 0xC2B2AE3D27D4EB4FULL);
    const double texture =
        static_cast<double>((mix >> 56U) & 0xFFU) / 255.0;
    const double band = ((x / 64U + y / 64U) % 2U == 0U) ? 0.08 : -0.08;
    return std::clamp(0.12 + 0.46 * nx + 0.24 * ny + 0.10 * texture + band, 0.0, 1.0);
}

std::vector<double> encode(std::size_t width, std::size_t height) {
    std::vector<double> sums(kLatentCount, 0.0);
    std::vector<std::uint32_t> counts(kLatentCount, 0U);

    std::size_t p = 0U;
    for (std::size_t y = 0U; y < height; ++y) {
        for (std::size_t x = 0U; x < width; ++x, ++p) {
            const auto q = pixel_to_latent(p);
            const std::size_t li = index3(q[0], q[1], q[2]);
            sums[li] += source_pixel(x, y, width, height);
            ++counts[li];
        }
    }

    for (std::size_t i = 0U; i < kLatentCount; ++i) {
        if (counts[i] != 0U) {
            sums[i] /= static_cast<double>(counts[i]);
        }
    }
    return sums;
}

std::vector<double> inward_fold(const std::vector<double>& input) {
    std::vector<double> a = input;
    std::vector<double> b(kLatentCount, 0.0);

    for (std::size_t step = 0U; step < kFoldSteps; ++step) {
        double centre = std::accumulate(a.begin(), a.end(), 0.0) /
            static_cast<double>(a.size());

        for (std::size_t z = 0U; z < kLatentEdge; ++z) {
            for (std::size_t y = 0U; y < kLatentEdge; ++y) {
                for (std::size_t x = 0U; x < kLatentEdge; ++x) {
                    const auto at = [&](long dx, long dy, long dz) -> double {
                        const std::size_t xx = wrap(static_cast<long>(x) + dx, kLatentEdge);
                        const std::size_t yy = wrap(static_cast<long>(y) + dy, kLatentEdge);
                        const std::size_t zz = wrap(static_cast<long>(z) + dz, kLatentEdge);
                        return a[index3(xx, yy, zz)];
                    };

                    const std::size_t i = index3(x, y, z);
                    const double lap =
                        at(1L, 0L, 0L) + at(-1L, 0L, 0L) +
                        at(0L, 1L, 0L) + at(0L, -1L, 0L) +
                        at(0L, 0L, 1L) + at(0L, 0L, -1L) -
                        6.0 * a[i];

                    const double evolved = a[i] + kDiffusion * lap;
                    b[i] = std::clamp(
                        centre + kLambda * (evolved - centre),
                        0.0,
                        1.0
                    );
                }
            }
        }
        a.swap(b);
    }

    return a;
}

Metrics decode_metrics(
    const std::vector<double>& latent,
    std::size_t width,
    std::size_t height
) {
    long double sq = 0.0L;
    long double ab = 0.0L;
    double max_error = 0.0;
    std::size_t p = 0U;

    for (std::size_t y = 0U; y < height; ++y) {
        for (std::size_t x = 0U; x < width; ++x, ++p) {
            const auto q = pixel_to_latent(p);
            const double reconstructed = latent[index3(q[0], q[1], q[2])];
            const double target = source_pixel(x, y, width, height);
            const double error = target - reconstructed;
            sq += static_cast<long double>(error * error);
            ab += static_cast<long double>(std::abs(error));
            max_error = std::max(max_error, std::abs(error));
        }
    }

    const long double n = static_cast<long double>(width * height);
    return {
        static_cast<double>(sq / n),
        static_cast<double>(ab / n),
        max_error
    };
}

bool finite_and_bounded(const std::vector<double>& values) {
    for (double v : values) {
        if (!std::isfinite(v) || v < 0.0 || v > 1.0) {
            return false;
        }
    }
    return true;
}

double latent_fixed_point_residual(const std::vector<double>& a) {
    const std::vector<double> b = inward_fold(a);
    double max_delta = 0.0;
    for (std::size_t i = 0U; i < a.size(); ++i) {
        max_delta = std::max(max_delta, std::abs(b[i] - a[i]));
    }
    return max_delta;
}

State run_cycle(
    const State* previous,
    std::size_t width,
    std::size_t height
) {
    State state;
    const std::vector<double> encoded = encode(width, height);
    const Metrics baseline = decode_metrics(encoded, width, height);

    const std::vector<double> candidate = inward_fold(encoded);
    const Metrics candidate_metrics = decode_metrics(candidate, width, height);

    state.committed = candidate_metrics.mse <= baseline.mse + kCommitSlack;
    state.latent = state.committed ? candidate : encoded;
    state.metrics = state.committed ? candidate_metrics : baseline;

    state.omega.assign(kLatentCount, 0.0);
    for (std::size_t i = 0U; i < kLatentCount; ++i) {
        const double old_value =
            (previous != nullptr && previous->omega.size() == kLatentCount)
            ? previous->omega[i]
            : 0.0;
        state.omega[i] =
            kOmegaBeta * old_value + (1.0 - kOmegaBeta) * state.latent[i];
    }

    state.bounded = finite_and_bounded(state.latent) && finite_and_bounded(state.omega);
    state.finite =
        std::isfinite(state.metrics.mse) &&
        std::isfinite(state.metrics.mae) &&
        std::isfinite(state.metrics.max_error);

    return state;
}

std::size_t parse_size(const char* text, const char* name) {
    try {
        const unsigned long long parsed = std::stoull(text);
        if (parsed == 0ULL) {
            throw std::runtime_error(std::string(name) + " must be > 0");
        }
        if (parsed > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(std::string(name) + " exceeds size_t");
        }
        return static_cast<std::size_t>(parsed);
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t width = kDefaultWidth;
    std::size_t height = kDefaultHeight;
    std::size_t cycles = 2U;

    try {
        if (argc > 1) {
            width = parse_size(argv[1], "width");
        }
        if (argc > 2) {
            height = parse_size(argv[2], "height");
        }
        if (argc > 3) {
            cycles = parse_size(argv[3], "cycles");
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    if (width > std::numeric_limits<std::size_t>::max() / height) {
        std::cerr << "error: width*height overflow\n";
        return 2;
    }

    const std::size_t pixels = width * height;
    std::cout << "Dr Moagi 8M 3D AE/AD Master Equation Runtime\n";
    std::cout << "pixels=" << pixels
              << " frame=" << width << "x" << height
              << " latent=" << kLatentEdge << "^3"
              << " compression="
              << std::fixed << std::setprecision(3)
              << static_cast<double>(pixels) / static_cast<double>(kLatentCount)
              << ":1\n";

    State state;
    bool all_ok = true;
    for (std::size_t cycle = 0U; cycle < cycles; ++cycle) {
        const State* previous = cycle == 0U ? nullptr : &state;
        State next = run_cycle(previous, width, height);
        const double fp_residual = latent_fixed_point_residual(next.latent);

        const bool cycle_ok =
            next.bounded &&
            next.finite &&
            next.metrics.mse >= 0.0 &&
            std::isfinite(fp_residual);

        std::cout << "cycle=" << cycle
                  << " ctr=" << (next.committed ? "COMMIT" : "ROLLBACK")
                  << " mse=" << std::setprecision(12) << next.metrics.mse
                  << " mae=" << next.metrics.mae
                  << " max_error=" << next.metrics.max_error
                  << " fixed_point_residual=" << fp_residual
                  << " bounded=" << (next.bounded ? "yes" : "no")
                  << " finite=" << (next.finite ? "yes" : "no")
                  << "\n";

        all_ok = all_ok && cycle_ok;
        state = std::move(next);
    }

    std::cout << "Operator: B3D -> E_theta -> Phi_T3^K -> D_phi"
              << " -> CTR -> Omega -> verify -> recur\n";
    std::cout << "Verification: " << (all_ok ? "PASS" : "FAIL") << "\n";
    return all_ok ? 0 : 1;
}
