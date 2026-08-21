#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace dm3d_verification {

using fixed_t = std::int32_t;
using wide_t = std::int64_t;

static constexpr int SHIFT = 16;
static constexpr fixed_t ONE = fixed_t{1} << SHIFT;
static constexpr fixed_t MAX_FIXED = std::numeric_limits<fixed_t>::max();
static constexpr fixed_t MIN_FIXED = std::numeric_limits<fixed_t>::min();
static constexpr std::size_t D_MODEL = 128U;
static constexpr std::size_t D_LATENT = 48U;
static constexpr double PI = 3.141592653589793238462643383279502884;

enum class EvidenceLevel {
    ExactInvariant,
    AnalyticBound,
    DeterministicRegression,
    EmpiricalDiagnostic
};

const char* evidence_name(EvidenceLevel level) {
    switch (level) {
        case EvidenceLevel::ExactInvariant: return "exact invariant";
        case EvidenceLevel::AnalyticBound: return "analytic bound";
        case EvidenceLevel::DeterministicRegression: return "deterministic regression";
        case EvidenceLevel::EmpiricalDiagnostic: return "empirical diagnostic";
    }
    return "unknown";
}

fixed_t f_sat(wide_t value) {
    if (value > static_cast<wide_t>(MAX_FIXED)) return MAX_FIXED;
    if (value < static_cast<wide_t>(MIN_FIXED)) return MIN_FIXED;
    return static_cast<fixed_t>(value);
}

fixed_t f_add(fixed_t a, fixed_t b) {
    return f_sat(static_cast<wide_t>(a) + static_cast<wide_t>(b));
}

fixed_t f_sub(fixed_t a, fixed_t b) {
    return f_sat(static_cast<wide_t>(a) - static_cast<wide_t>(b));
}

fixed_t f_mul(fixed_t a, fixed_t b) {
    return f_sat((static_cast<wide_t>(a) * static_cast<wide_t>(b)) >> SHIFT);
}

fixed_t f_div(fixed_t a, fixed_t b) {
    if (b == 0) return (a >= 0) ? MAX_FIXED : MIN_FIXED;
    return f_sat((static_cast<wide_t>(a) << SHIFT) / static_cast<wide_t>(b));
}

fixed_t from_double(double value) {
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(ONE);
    if (scaled >= static_cast<long double>(MAX_FIXED)) return MAX_FIXED;
    if (scaled <= static_cast<long double>(MIN_FIXED)) return MIN_FIXED;
    return static_cast<fixed_t>(std::llround(scaled));
}

double to_double(fixed_t value) {
    return static_cast<double>(value) / static_cast<double>(ONE);
}

fixed_t f_tanh(fixed_t x) {
    const fixed_t three = from_double(3.0);
    if (x >= three) return ONE;
    if (x <= -three) return -ONE;

    const fixed_t x2 = f_mul(x, x);
    const fixed_t numerator = f_mul(x, f_add(from_double(27.0), x2));
    const fixed_t denominator = f_add(from_double(27.0), f_mul(from_double(9.0), x2));
    return std::clamp(f_div(numerator, denominator), -ONE, ONE);
}

struct XorShift32 {
    std::uint32_t state;
    explicit XorShift32(std::uint32_t seed) : state(seed == 0U ? 1U : seed) {}
    std::uint32_t next() {
        std::uint32_t x = state;
        x ^= x << 13U;
        x ^= x >> 17U;
        x ^= x << 5U;
        state = x;
        return x;
    }
};

template<std::size_t Rows, std::size_t Cols>
class FixedMatrix {
public:
    explicit FixedMatrix(std::uint32_t seed, double amplitude) {
        XorShift32 rng(seed);
        const fixed_t amp = from_double(amplitude);
        for (auto& value : weights_) {
            const std::int32_t centered = static_cast<std::int32_t>(rng.next() & 0xFFFFU) - 32768;
            value = f_sat((static_cast<wide_t>(centered) * static_cast<wide_t>(amp)) >> 15);
        }
    }

    template<std::size_t N = Cols>
    std::array<fixed_t, Rows> mul(const std::array<fixed_t, N>& input) const {
        static_assert(N == Cols, "dimension mismatch");
        std::array<fixed_t, Rows> output{};
        for (std::size_t row = 0; row < Rows; ++row) {
            wide_t acc = 0;
            for (std::size_t col = 0; col < Cols; ++col) {
                acc += (static_cast<wide_t>(weights_[row * Cols + col]) *
                        static_cast<wide_t>(input[col])) >> SHIFT;
            }
            output[row] = f_sat(acc);
        }
        return output;
    }

    const std::array<fixed_t, Rows * Cols>& weights() const noexcept { return weights_; }

private:
    std::array<fixed_t, Rows * Cols> weights_{};
};

class LatentLatticeAE {
public:
    LatentLatticeAE()
        : encoder_(0xD34D1234U, std::sqrt(6.0 / static_cast<double>(D_MODEL + D_LATENT))),
          decoder_(0xBEEF4321U, std::sqrt(6.0 / static_cast<double>(D_MODEL + D_LATENT))) {}

    std::array<fixed_t, D_LATENT> encode(const std::array<fixed_t, D_MODEL>& input) const {
        auto latent = encoder_.mul(input);
        for (auto& value : latent) value = f_tanh(value);
        return latent;
    }

    std::array<fixed_t, D_MODEL> decode(const std::array<fixed_t, D_LATENT>& latent) const {
        return decoder_.mul(latent);
    }

    const std::array<fixed_t, D_LATENT * D_MODEL>& encoder_weights() const noexcept {
        return encoder_.weights();
    }

private:
    FixedMatrix<D_LATENT, D_MODEL> encoder_;
    FixedMatrix<D_MODEL, D_LATENT> decoder_;
};

struct CheckResult {
    std::string name;
    EvidenceLevel evidence{};
    bool passed{};
    std::string statement;
    std::vector<std::pair<std::string, double>> metrics;
};

class VerificationCore {
public:
    std::vector<CheckResult> run_all() const {
        std::vector<CheckResult> results;
        results.push_back(check_fixed_point_integrity());
        results.push_back(check_manifold_bounds());
        results.push_back(check_periodic_diffusion_conservation());
        results.push_back(check_recursive_consistency());
        results.push_back(check_stream_separation());
        results.push_back(check_reference_loss_convexity());
        results.push_back(check_reference_hessian());
        results.push_back(check_latent_rank());
        results.push_back(check_spectral_bound());
        results.push_back(check_topology_regression());
        return results;
    }

private:
    static double rms_difference(const std::array<fixed_t, D_MODEL>& a,
                                 const std::array<fixed_t, D_MODEL>& b) {
        long double sum = 0.0L;
        for (std::size_t i = 0; i < D_MODEL; ++i) {
            const long double d = static_cast<long double>(to_double(f_sub(a[i], b[i])));
            sum += d * d;
        }
        return std::sqrt(static_cast<double>(sum / static_cast<long double>(D_MODEL)));
    }

    static std::array<fixed_t, D_MODEL> normalized_sine_state(double phase) {
        std::array<fixed_t, D_MODEL> state{};
        for (std::size_t i = 0; i < D_MODEL; ++i) {
            const double angle = 2.0 * PI * static_cast<double>(i) / static_cast<double>(D_MODEL) + phase;
            state[i] = from_double(0.5 * std::sin(angle));
        }
        return state;
    }

    CheckResult check_fixed_point_integrity() const {
        const fixed_t one_third = from_double(1.0 / 3.0);
        const fixed_t product = f_mul(one_third, one_third);
        const double product_error = std::abs(to_double(product) - 1.0 / 9.0);

        const fixed_t x = 12345;
        const fixed_t y = 67890;
        const fixed_t z = 11111;
        const bool associativity_in_safe_range =
            f_add(f_add(x, y), z) == f_add(x, f_add(y, z));
        const bool saturation = f_add(MAX_FIXED, MAX_FIXED) == MAX_FIXED &&
                                f_add(MIN_FIXED, MIN_FIXED) == MIN_FIXED;

        return {
            "Fixed-point arithmetic integrity",
            EvidenceLevel::ExactInvariant,
            product_error < 2.0 / static_cast<double>(ONE) && associativity_in_safe_range && saturation,
            "Q16.16 multiplication error is quantization-bounded and saturating addition is deterministic.",
            {{"one_ninth_error", product_error}}
        };
    }

    CheckResult check_manifold_bounds() const {
        double max_abs = 0.0;
        double max_slope = 0.0;
        bool monotone = true;
        fixed_t previous = f_tanh(from_double(-3.0));
        constexpr int samples = 12000;
        for (int i = 1; i <= samples; ++i) {
            const double x = -3.0 + 6.0 * static_cast<double>(i) / static_cast<double>(samples);
            const fixed_t current = f_tanh(from_double(x));
            max_abs = std::max(max_abs, std::abs(to_double(current)));
            if (current < previous - 2) monotone = false;
            const double dx = 6.0 / static_cast<double>(samples);
            const double slope = std::abs(to_double(f_sub(current, previous))) / dx;
            max_slope = std::max(max_slope, slope);
            previous = current;
        }
        return {
            "Bounded latent activation",
            EvidenceLevel::DeterministicRegression,
            monotone && max_abs <= 1.0 + 1e-12 && max_slope < 1.05,
            "The Q16.16 rational tanh approximation is monotone to fixed-point tolerance, bounded in [-1,1], and numerically near 1-Lipschitz on the sampled lattice.",
            {{"max_abs", max_abs}, {"sampled_max_slope", max_slope}}
        };
    }

    CheckResult check_periodic_diffusion_conservation() const {
        constexpr int edge = 4;
        using Grid = std::array<double, edge * edge * edge>;
        Grid grid{};
        auto index = [](int x, int y, int z) {
            return static_cast<std::size_t>(z * edge * edge + y * edge + x);
        };
        for (int z = 0; z < edge; ++z) {
            for (int y = 0; y < edge; ++y) {
                for (int x = 0; x < edge; ++x) {
                    grid[index(x, y, z)] = std::sin(static_cast<double>(x)) *
                                            std::cos(static_cast<double>(y)) *
                                            std::sin(static_cast<double>(z));
                }
            }
        }
        const double initial_mass = std::accumulate(grid.begin(), grid.end(), 0.0);
        constexpr double alpha = 0.05;
        for (int step = 0; step < 100; ++step) {
            Grid next{};
            for (int z = 0; z < edge; ++z) {
                for (int y = 0; y < edge; ++y) {
                    for (int x = 0; x < edge; ++x) {
                        const auto wrap = [](int v) { return (v + edge) % edge; };
                        const double center = grid[index(x, y, z)];
                        const double lap =
                            grid[index(wrap(x + 1), y, z)] + grid[index(wrap(x - 1), y, z)] +
                            grid[index(x, wrap(y + 1), z)] + grid[index(x, wrap(y - 1), z)] +
                            grid[index(x, y, wrap(z + 1))] + grid[index(x, y, wrap(z - 1))] -
                            6.0 * center;
                        next[index(x, y, z)] = center + alpha * lap;
                    }
                }
            }
            grid = next;
        }
        const double final_mass = std::accumulate(grid.begin(), grid.end(), 0.0);
        const double error = std::abs(final_mass - initial_mass);
        return {
            "3D periodic transport conservation",
            EvidenceLevel::DeterministicRegression,
            error < 1e-11,
            "The implemented periodic six-neighbor diffusion stencil conserves total scalar mass; this is not a claim of physical energy/momentum conservation.",
            {{"mass_conservation_error", error}}
        };
    }

    CheckResult check_recursive_consistency() const {
        LatentLatticeAE ae;
        auto state = normalized_sine_state(0.0);
        bool bounded = true;
        double final_delta = std::numeric_limits<double>::infinity();
        for (int depth = 0; depth < 7; ++depth) {
            const auto reconstruction = ae.decode(ae.encode(state));
            std::array<fixed_t, D_MODEL> next{};
            for (std::size_t i = 0; i < D_MODEL; ++i) {
                next[i] = f_add(f_mul(from_double(0.75), state[i]),
                                f_mul(from_double(0.25), reconstruction[i]));
            }
            final_delta = rms_difference(next, state);
            bounded = bounded && std::isfinite(final_delta) && final_delta < 4.0;
            state = next;
        }
        return {
            "Bounded recursive self-consistency",
            EvidenceLevel::EmpiricalDiagnostic,
            bounded,
            "Seven encode/decode residual recurrences remain finite and bounded for the deterministic reference state; this is a regression check, not a convergence proof.",
            {{"depth7_rms_delta", final_delta}}
        };
    }

    CheckResult check_stream_separation() const {
        std::array<std::array<double, D_MODEL>, 8> streams{};
        for (std::size_t s = 0; s < streams.size(); ++s) {
            for (std::size_t i = 0; i < D_MODEL; ++i) {
                streams[s][i] = std::sin(2.0 * PI * static_cast<double>((s + 1U) * i) /
                                         static_cast<double>(D_MODEL));
            }
        }
        double max_cosine = 0.0;
        for (std::size_t a = 0; a < streams.size(); ++a) {
            for (std::size_t b = a + 1U; b < streams.size(); ++b) {
                double dot = 0.0;
                double na = 0.0;
                double nb = 0.0;
                for (std::size_t i = 0; i < D_MODEL; ++i) {
                    dot += streams[a][i] * streams[b][i];
                    na += streams[a][i] * streams[a][i];
                    nb += streams[b][i] * streams[b][i];
                }
                const double cosine = std::abs(dot / std::sqrt(na * nb));
                max_cosine = std::max(max_cosine, cosine);
            }
        }
        return {
            "Reference stream separation",
            EvidenceLevel::AnalyticBound,
            max_cosine < 1e-12,
            "Eight harmonic reference channels are pairwise orthogonal under the discrete inner product; no mutual-information claim is made.",
            {{"max_abs_cosine", max_cosine}}
        };
    }

    static double quadratic_loss(const std::array<double, 8>& x) {
        double sum = 0.0;
        for (const double value : x) sum += value * value;
        return sum;
    }

    CheckResult check_reference_loss_convexity() const {
        bool convex = true;
        double worst_gap = -std::numeric_limits<double>::infinity();
        for (int trial = 0; trial < 256; ++trial) {
            std::array<double, 8> a{};
            std::array<double, 8> b{};
            std::array<double, 8> mid{};
            for (std::size_t i = 0; i < a.size(); ++i) {
                const double t = static_cast<double>(trial + static_cast<int>(i) * 17);
                a[i] = 0.1 * std::sin(t);
                b[i] = 0.1 * std::cos(0.5 * t);
                mid[i] = 0.5 * (a[i] + b[i]);
            }
            const double gap = quadratic_loss(mid) - 0.5 * (quadratic_loss(a) + quadratic_loss(b));
            worst_gap = std::max(worst_gap, gap);
            if (gap > 1e-12) convex = false;
        }
        return {
            "Reference quadratic loss convexity",
            EvidenceLevel::ExactInvariant,
            convex,
            "The reference L2 objective is convex. This does not imply that the full nonlinear autoencoder loss landscape is globally convex.",
            {{"worst_jensen_gap", worst_gap}}
        };
    }

    CheckResult check_reference_hessian() const {
        constexpr double min_eigenvalue = 2.0;
        return {
            "Reference Hessian definiteness",
            EvidenceLevel::ExactInvariant,
            min_eigenvalue > 0.0,
            "The Hessian of the declared quadratic reference loss is 2I and positive definite; no claim is made about the full model Hessian.",
            {{"min_eigenvalue", min_eigenvalue}}
        };
    }

    CheckResult check_latent_rank() const {
        LatentLatticeAE ae;
        std::vector<std::vector<double>> matrix(D_LATENT, std::vector<double>(D_MODEL));
        const auto& weights = ae.encoder_weights();
        for (std::size_t row = 0; row < D_LATENT; ++row) {
            for (std::size_t col = 0; col < D_MODEL; ++col) {
                matrix[row][col] = to_double(weights[row * D_MODEL + col]);
            }
        }

        std::size_t rank = 0U;
        constexpr double eps = 1e-10;
        for (std::size_t col = 0; col < D_MODEL && rank < D_LATENT; ++col) {
            std::size_t pivot = rank;
            for (std::size_t row = rank; row < D_LATENT; ++row) {
                if (std::abs(matrix[row][col]) > std::abs(matrix[pivot][col])) pivot = row;
            }
            if (std::abs(matrix[pivot][col]) <= eps) continue;
            std::swap(matrix[pivot], matrix[rank]);
            const double divisor = matrix[rank][col];
            for (std::size_t c = col; c < D_MODEL; ++c) matrix[rank][c] /= divisor;
            for (std::size_t row = 0; row < D_LATENT; ++row) {
                if (row == rank) continue;
                const double factor = matrix[row][col];
                for (std::size_t c = col; c < D_MODEL; ++c) {
                    matrix[row][c] -= factor * matrix[rank][c];
                }
            }
            ++rank;
        }
        const double coordinate_ratio = static_cast<double>(D_LATENT) / static_cast<double>(D_MODEL);
        return {
            "Latent bottleneck rank",
            EvidenceLevel::DeterministicRegression,
            rank == D_LATENT,
            "The 48x128 encoder has full row rank in the deterministic reference initialization. The 37.5% value is dimensionality retention, not Shannon information preservation.",
            {{"rank", static_cast<double>(rank)}, {"coordinate_ratio", coordinate_ratio}}
        };
    }

    CheckResult check_spectral_bound() const {
        constexpr std::size_t n = 32U;
        constexpr double gain = 0.90;
        double max_abs_row_sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double raw_sum = 0.0;
            std::array<double, n> row{};
            for (std::size_t j = 0; j < n; ++j) {
                const double distance = std::abs(static_cast<double>(i) - static_cast<double>(j));
                row[j] = std::exp(-distance / 4.0);
                raw_sum += row[j];
            }
            double row_sum = 0.0;
            for (const double value : row) row_sum += gain * value / raw_sum;
            max_abs_row_sum = std::max(max_abs_row_sum, row_sum);
        }
        return {
            "Transition spectral stability bound",
            EvidenceLevel::AnalyticBound,
            max_abs_row_sum < 1.0,
            "By the induced infinity norm, spectral radius rho(T) <= ||T||_inf; the normalized transition operator has row sum 0.9.",
            {{"spectral_radius_upper_bound", max_abs_row_sum}}
        };
    }

    static int connected_components(const std::vector<std::array<double, 3>>& points, double threshold) {
        std::vector<bool> visited(points.size(), false);
        int components = 0;
        for (std::size_t start = 0; start < points.size(); ++start) {
            if (visited[start]) continue;
            ++components;
            std::vector<std::size_t> stack{start};
            visited[start] = true;
            while (!stack.empty()) {
                const std::size_t current = stack.back();
                stack.pop_back();
                for (std::size_t j = 0; j < points.size(); ++j) {
                    if (visited[j]) continue;
                    double d2 = 0.0;
                    for (std::size_t k = 0; k < 3U; ++k) {
                        const double d = points[current][k] - points[j][k];
                        d2 += d * d;
                    }
                    if (std::sqrt(d2) < threshold) {
                        visited[j] = true;
                        stack.push_back(j);
                    }
                }
            }
        }
        return components;
    }

    CheckResult check_topology_regression() const {
        std::vector<std::array<double, 3>> original;
        std::vector<std::array<double, 3>> quantized;
        for (int cluster = 0; cluster < 2; ++cluster) {
            const double center = cluster == 0 ? -1.0 : 1.0;
            for (int i = 0; i < 16; ++i) {
                const double t = 2.0 * PI * static_cast<double>(i) / 16.0;
                std::array<double, 3> p{center + 0.1 * std::cos(t), 0.1 * std::sin(t), 0.02 * std::sin(2.0 * t)};
                original.push_back(p);
                for (double& v : p) v = std::round(v * 256.0) / 256.0;
                quantized.push_back(p);
            }
        }
        const int beta0_original = connected_components(original, 0.30);
        const int beta0_quantized = connected_components(quantized, 0.30);
        return {
            "Topology component regression",
            EvidenceLevel::DeterministicRegression,
            beta0_original == beta0_quantized && beta0_original == 2,
            "A two-component 3D fixture preserves beta_0 under the reference quantization. Higher Betti numbers are not claimed or hard-coded.",
            {{"beta0_original", static_cast<double>(beta0_original)},
             {"beta0_quantized", static_cast<double>(beta0_quantized)}}
        };
    }
};

int run() {
    std::cout << "DM-vOmegaXi+ Verification Core v4.1\n";
    std::cout << "Scope: executable operational invariants and diagnostics; not a general theorem prover.\n\n";

    const VerificationCore core;
    const auto results = core.run_all();
    std::size_t passed = 0U;

    for (const auto& result : results) {
        std::cout << (result.passed ? "[PASS] " : "[FAIL] ") << result.name
                  << " (" << evidence_name(result.evidence) << ")\n";
        std::cout << "       " << result.statement << "\n";
        for (const auto& metric : result.metrics) {
            std::cout << "       " << metric.first << " = " << std::setprecision(12) << metric.second << "\n";
        }
        if (result.passed) ++passed;
    }

    std::cout << "\nSummary: " << passed << "/" << results.size() << " checks passed.\n";
    std::cout << "Capability boundary: passing checks verify only their stated contracts;\n"
                 "they do not prove self-awareness, global optimality, topology preservation in general,\n"
                 "or any external mathematical conjecture.\n";

    return passed == results.size() ? 0 : 1;
}

} // namespace dm3d_verification

int main() {
    return dm3d_verification::run();
}
