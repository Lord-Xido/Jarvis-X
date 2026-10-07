#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef JARVISX_HAVE_OPENMP
#include <omp.h>
#endif

namespace jarvisx::mg3d {

using f32 = float;
using Clock = std::chrono::steady_clock;

constexpr f32 kReaction = 0.10f;
constexpr f32 kOmega = 0.72f;
constexpr f32 kMinPerm = 0.05f;
constexpr f32 kMaxPerm = 1.00f;
constexpr f32 kPi = 3.14159265358979323846f;

inline int max_threads() noexcept {
#ifdef JARVISX_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

struct WorkCounter {
    std::uint64_t flops = 0;
    void add(std::uint64_t n) noexcept { flops += n; }
};

struct Level3D {
    int D = 0;
    std::vector<f32> u;
    std::vector<f32> rhs;
    std::vector<f32> residual;
    std::vector<f32> k;
    std::vector<f32> scratch;

    explicit Level3D(int dim)
        : D(dim),
          u(static_cast<std::size_t>(dim) * dim * dim, 0.0f),
          rhs(u.size(), 0.0f),
          residual(u.size(), 0.0f),
          k(u.size(), 0.35f),
          scratch(u.size(), 0.0f) {
        if (dim < 4 || (dim & (dim - 1)) != 0) {
            throw std::invalid_argument("level dimension must be a power of two >= 4");
        }
    }

    [[nodiscard]] std::size_t index(int d, int h, int w) const noexcept {
        return (static_cast<std::size_t>(d) * D + static_cast<std::size_t>(h)) * D
             + static_cast<std::size_t>(w);
    }

    [[nodiscard]] int wrap(int x) const noexcept {
        x %= D;
        return x < 0 ? x + D : x;
    }

    [[nodiscard]] std::size_t bytes() const noexcept {
        return 5ull * u.size() * sizeof(f32);
    }
};

struct Pyramid3D {
    std::vector<Level3D> levels;
    std::vector<f32> truth;

    explicit Pyramid3D(int root_dim, int min_dim = 4) {
        if (root_dim < min_dim || (root_dim & (root_dim - 1)) != 0 ||
            (min_dim & (min_dim - 1)) != 0) {
            throw std::invalid_argument("root/min dimensions must be powers of two");
        }
        int d = root_dim;
        while (true) {
            levels.emplace_back(d);
            if (d == min_dim) break;
            d /= 2;
            if (d < min_dim) {
                throw std::invalid_argument("min dimension must divide root by powers of two");
            }
        }
        truth.assign(levels.front().u.size(), 0.0f);
    }

    [[nodiscard]] std::size_t total_bytes() const noexcept {
        std::size_t n = truth.size() * sizeof(f32);
        for (const auto& level : levels) n += level.bytes();
        return n;
    }
};

struct StencilValues {
    f32 diag = kReaction;
    f32 weighted_neighbors = 0.0f;
};

[[nodiscard]] inline StencilValues stencil_values(
    const Level3D& L,
    const std::vector<f32>& u,
    int d, int h, int w) noexcept {

    const std::size_t i = L.index(d, h, w);
    const f32 ki = L.k[i];
    StencilValues out{};

    const std::array<std::array<int, 3>, 6> offsets{{
        {{-1, 0, 0}}, {{1, 0, 0}}, {{0, -1, 0}},
        {{0, 1, 0}}, {{0, 0, -1}}, {{0, 0, 1}}
    }};

    for (const auto& o : offsets) {
        const int nd = L.wrap(d + o[0]);
        const int nh = L.wrap(h + o[1]);
        const int nw = L.wrap(w + o[2]);
        const std::size_t j = L.index(nd, nh, nw);
        const f32 wij = 0.5f * (ki + L.k[j]);
        out.diag += wij;
        out.weighted_neighbors += wij * u[j];
    }
    return out;
}

[[nodiscard]] inline f32 apply_operator_point(
    const Level3D& L,
    const std::vector<f32>& u,
    int d, int h, int w) noexcept {

    const auto s = stencil_values(L, u, d, h, w);
    return s.diag * u[L.index(d, h, w)] - s.weighted_neighbors;
}

void apply_operator(
    const Level3D& L,
    const std::vector<f32>& u,
    std::vector<f32>& out,
    WorkCounter* work = nullptr) {

    const int D = L.D;
    if (out.size() != u.size()) out.resize(u.size());

    std::uint64_t local = 0;
#pragma omp parallel for collapse(2) reduction(+:local) if(D >= 32)
    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < D; ++h) {
#if !defined(_MSC_VER)
#pragma omp simd
#endif
            for (int w = 0; w < D; ++w) {
                out[L.index(d,h,w)] = apply_operator_point(L, u, d, h, w);
                local += 31;
            }
        }
    }
    if (work != nullptr) work->add(local);
}

void smooth(Level3D& L, int sweeps, WorkCounter& work, f32 omega = kOmega) {
    const int D = L.D;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        std::uint64_t local = 0;

#pragma omp parallel for collapse(2) reduction(+:local) if(D >= 32)
        for (int d = 0; d < D; ++d) {
            for (int h = 0; h < D; ++h) {
#if !defined(_MSC_VER)
#pragma omp simd
#endif
                for (int w = 0; w < D; ++w) {
                    const std::size_t i = L.index(d,h,w);
                    const auto s = stencil_values(L, L.u, d, h, w);
                    const f32 jacobi =
                        (L.rhs[i] + s.weighted_neighbors) / s.diag;

                    L.scratch[i] =
                        (1.0f - omega) * L.u[i] + omega * jacobi;
                    local += 36;
                }
            }
        }

        L.u.swap(L.scratch);
        work.add(local);
    }
}

void compute_residual(Level3D& L, WorkCounter& work) {
    const int D = L.D;
    std::uint64_t local = 0;

#pragma omp parallel for collapse(2) reduction(+:local) if(D >= 32)
    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < D; ++h) {
#if !defined(_MSC_VER)
#pragma omp simd
#endif
            for (int w = 0; w < D; ++w) {
                const std::size_t i = L.index(d,h,w);
                L.residual[i] =
                    L.rhs[i] - apply_operator_point(L, L.u, d, h, w);
                local += 32;
            }
        }
    }

    work.add(local);
}

void restrict_full_weighting(
    const Level3D& fine,
    Level3D& coarse,
    const std::vector<f32>& fine_field,
    std::vector<f32>& coarse_field,
    WorkCounter* work = nullptr) {

    const int Dc = coarse.D;
    std::uint64_t local = 0;

#pragma omp parallel for collapse(2) reduction(+:local) if(Dc >= 16)
    for (int d = 0; d < Dc; ++d) {
        for (int h = 0; h < Dc; ++h) {
            for (int w = 0; w < Dc; ++w) {
                f32 sum = 0.0f;

                for (int a = -1; a <= 1; ++a) {
                    const int wa = a == 0 ? 2 : 1;

                    for (int b = -1; b <= 1; ++b) {
                        const int wb = b == 0 ? 2 : 1;

                        for (int c = -1; c <= 1; ++c) {
                            const int wc = c == 0 ? 2 : 1;
                            const int fd = fine.wrap(2*d + a);
                            const int fh = fine.wrap(2*h + b);
                            const int fw = fine.wrap(2*w + c);

                            sum += static_cast<f32>(wa*wb*wc)
                                 * fine_field[fine.index(fd,fh,fw)];
                        }
                    }
                }

                coarse_field[coarse.index(d,h,w)] =
                    sum * (1.0f/64.0f);
                local += 55;
            }
        }
    }

    if (work != nullptr) work->add(local);
}

void restrict_residual(
    Level3D& fine,
    Level3D& coarse,
    WorkCounter& work) {

    restrict_full_weighting(
        fine, coarse, fine.residual, coarse.rhs, &work);

    std::fill(coarse.u.begin(), coarse.u.end(), 0.0f);
}

void restrict_permeability(
    const Level3D& fine,
    Level3D& coarse) {

    const int Dc = coarse.D;

#pragma omp parallel for collapse(2) if(Dc >= 16)
    for (int d = 0; d < Dc; ++d) {
        for (int h = 0; h < Dc; ++h) {
            for (int w = 0; w < Dc; ++w) {
                f32 sum = 0.0f;

                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b)
                        for (int c = 0; c < 2; ++c)
                            sum += fine.k[
                                fine.index(2*d+a,2*h+b,2*w+c)];

                coarse.k[coarse.index(d,h,w)] =
                    sum * 0.125f;
            }
        }
    }
}

void propagate_permeability(Pyramid3D& P) {
    for (std::size_t l = 1; l < P.levels.size(); ++l) {
        restrict_permeability(
            P.levels[l-1], P.levels[l]);
    }
}

void prolong_trilinear_add(
    Level3D& fine,
    const Level3D& coarse,
    WorkCounter& work) {

    const int Df = fine.D;
    const int Dc = coarse.D;
    std::uint64_t local = 0;

#pragma omp parallel for collapse(2) reduction(+:local) if(Df >= 32)
    for (int d = 0; d < Df; ++d) {
        for (int h = 0; h < Df; ++h) {
#if !defined(_MSC_VER)
#pragma omp simd
#endif
            for (int w = 0; w < Df; ++w) {
                const int cd0 = d >> 1;
                const int ch0 = h >> 1;
                const int cw0 = w >> 1;

                const int cd1 = (cd0 + 1) % Dc;
                const int ch1 = (ch0 + 1) % Dc;
                const int cw1 = (cw0 + 1) % Dc;

                const f32 td = (d & 1) ? 0.5f : 0.0f;
                const f32 th = (h & 1) ? 0.5f : 0.0f;
                const f32 tw = (w & 1) ? 0.5f : 0.0f;

                const auto C = [&](int dd, int hh, int ww) -> f32 {
                    return coarse.u[coarse.index(dd,hh,ww)];
                };

                const f32 c000 = C(cd0,ch0,cw0);
                const f32 c100 = C(cd1,ch0,cw0);
                const f32 c010 = C(cd0,ch1,cw0);
                const f32 c110 = C(cd1,ch1,cw0);
                const f32 c001 = C(cd0,ch0,cw1);
                const f32 c101 = C(cd1,ch0,cw1);
                const f32 c011 = C(cd0,ch1,cw1);
                const f32 c111 = C(cd1,ch1,cw1);

                const f32 a00 = c000 + td*(c100-c000);
                const f32 a10 = c010 + td*(c110-c010);
                const f32 a01 = c001 + td*(c101-c001);
                const f32 a11 = c011 + td*(c111-c011);
                const f32 b0 = a00 + th*(a10-a00);
                const f32 b1 = a01 + th*(a11-a01);
                const f32 correction = b0 + tw*(b1-b0);

                fine.u[fine.index(d,h,w)] += correction;
                local += 21;
            }
        }
    }

    work.add(local);
}

void vcycle(
    Pyramid3D& P,
    std::size_t level,
    WorkCounter& work,
    int pre = 2,
    int post = 2,
    int coarse_sweeps = 60) {

    Level3D& L = P.levels[level];

    if (level + 1 == P.levels.size()) {
        smooth(L, coarse_sweeps, work);
        return;
    }

    smooth(L, pre, work);
    compute_residual(L, work);

    Level3D& C = P.levels[level+1];
    restrict_residual(L, C, work);

    vcycle(P, level+1, work, pre, post, coarse_sweeps);

    prolong_trilinear_add(L, C, work);
    smooth(L, post, work);
}

[[nodiscard]] double l2_norm(const std::vector<f32>& x) {
    long double sum = 0.0L;

#pragma omp parallel for reduction(+:sum) if(x.size() >= 32768)
    for (std::int64_t i = 0;
         i < static_cast<std::int64_t>(x.size());
         ++i) {

        const long double v =
            x[static_cast<std::size_t>(i)];
        sum += v*v;
    }

    return std::sqrt(static_cast<double>(sum));
}

[[nodiscard]] double mse(
    const std::vector<f32>& a,
    const std::vector<f32>& b) {

    if (a.size() != b.size())
        throw std::invalid_argument("mse size mismatch");

    long double sum = 0.0L;

#pragma omp parallel for reduction(+:sum) if(a.size() >= 32768)
    for (std::int64_t i = 0;
         i < static_cast<std::int64_t>(a.size());
         ++i) {

        const long double e =
            static_cast<long double>(
                a[static_cast<std::size_t>(i)])
          - static_cast<long double>(
                b[static_cast<std::size_t>(i)]);

        sum += e*e;
    }

    return static_cast<double>(
        sum / static_cast<long double>(a.size()));
}

void seed_truth(Pyramid3D& P) {
    Level3D& F = P.levels.front();
    const int D = F.D;

#pragma omp parallel for collapse(2) if(D >= 32)
    for (int d = 0; d < D; ++d) {
        for (int h = 0; h < D; ++h) {
            for (int w = 0; w < D; ++w) {
                const f32 x =
                    static_cast<f32>(d)
                    / static_cast<f32>(D);

                const f32 y =
                    static_cast<f32>(h)
                    / static_cast<f32>(D);

                const f32 z =
                    static_cast<f32>(w)
                    / static_cast<f32>(D);

                const f32 value =
                    0.45f
                  + 0.20f*std::cos(2.0f*kPi*x)
                  + 0.15f*std::sin(4.0f*kPi*y)
                  + 0.10f*std::cos(6.0f*kPi*z)
                  + 0.05f*std::sin(
                        2.0f*kPi*(x+y+z));

                P.truth[F.index(d,h,w)] = value;
            }
        }
    }
}

void rebuild_root_rhs(
    Pyramid3D& P,
    WorkCounter* work = nullptr) {

    Level3D& F = P.levels.front();
    apply_operator(F, P.truth, F.rhs, work);
}

struct Metrics {
    double rel_residual =
        std::numeric_limits<double>::infinity();

    double mse_truth =
        std::numeric_limits<double>::infinity();
};

Metrics metrics(
    Pyramid3D& P,
    WorkCounter& work) {

    Level3D& F = P.levels.front();
    compute_residual(F, work);

    const double rhs_norm =
        std::max(l2_norm(F.rhs), 1e-30);

    return {
        l2_norm(F.residual) / rhs_norm,
        mse(F.u, P.truth)
    };
}

struct SolveResult {
    Metrics metrics{};
    int cycles = 0;
};

SolveResult solve_until(
    Pyramid3D& P,
    double tol,
    int max_cycles,
    WorkCounter& work) {

    SolveResult result{};

    for (int cycle = 0; cycle < max_cycles; ++cycle) {
        vcycle(P, 0, work);
        result.cycles = cycle + 1;
        result.metrics = metrics(P, work);

        if (result.metrics.rel_residual <= tol)
            break;
    }

    return result;
}

struct OuterReceipt {
    int attempted = 0;
    int committed = 0;
    int rolled_back = 0;
    int total_vcycles = 0;
    Metrics final_metrics{};
    double mean_delta_k = 0.0;
    double ms = 0.0;
    std::uint64_t flops = 0;
};

OuterReceipt adaptive_outer_loop(
    Pyramid3D& P,
    double tol,
    int max_outer,
    int max_cycles,
    f32 adapt_rate,
    WorkCounter& work) {

    const auto t0 = Clock::now();
    const std::uint64_t f0 = work.flops;

    OuterReceipt receipt{};

    propagate_permeability(P);
    rebuild_root_rhs(P, &work);

    auto current =
        solve_until(P, tol, max_cycles, work);

    receipt.total_vcycles += current.cycles;
    Metrics accepted = current.metrics;

    for (int outer = 0; outer < max_outer; ++outer) {
        receipt.attempted++;

        Level3D& F = P.levels.front();

        const auto u_prev = F.u;
        const auto k_prev = F.k;
        const auto rhs_prev = F.rhs;

        f32 max_err = 0.0f;

// MSVC's /openmp supports neither simd nor reduction(max:...).
// Retain the parallel maximum reduction using per-thread accumulation
// and a single critical-section combine per worker, avoiding a serial
// fallback on Windows. GCC/Clang retain their native max reduction.
#if defined(_MSC_VER)
#pragma omp parallel if(F.u.size() >= 32768)
        {
            f32 thread_max_err = 0.0f;
#pragma omp for nowait
            for (std::int64_t i = 0;
                 i < static_cast<std::int64_t>(F.u.size());
                 ++i) {

                const auto j = static_cast<std::size_t>(i);
                thread_max_err = std::max(
                    thread_max_err,
                    std::fabs(P.truth[j] - F.u[j]));
            }
#pragma omp critical(jarvisx_mg3d_max_error)
            {
                max_err = std::max(max_err, thread_max_err);
            }
        }
#else
#pragma omp parallel for reduction(max:max_err) if(F.u.size() >= 32768)
        for (std::int64_t i = 0;
             i < static_cast<std::int64_t>(F.u.size());
             ++i) {

            const auto j = static_cast<std::size_t>(i);

            max_err = std::max(
                max_err,
                std::fabs(P.truth[j] - F.u[j]));
        }
#endif

        max_err = std::max(max_err, 1e-8f);

        long double delta_sum = 0.0L;

#pragma omp parallel for reduction(+:delta_sum) if(F.u.size() >= 32768)
        for (std::int64_t i = 0;
             i < static_cast<std::int64_t>(F.u.size());
             ++i) {

            const auto j = static_cast<std::size_t>(i);

            const f32 e =
                std::fabs(P.truth[j] - F.u[j])
                / max_err;

            const f32 desired =
                std::clamp(
                    0.25f + 0.75f*e,
                    kMinPerm,
                    kMaxPerm);

            const f32 candidate =
                std::clamp(
                    (1.0f-adapt_rate)*F.k[j]
                    + adapt_rate*desired,
                    kMinPerm,
                    kMaxPerm);

            delta_sum +=
                std::fabs(candidate - F.k[j]);

            F.k[j] = candidate;
        }

        const double mean_dk =
            static_cast<double>(
                delta_sum / F.k.size());

        propagate_permeability(P);
        rebuild_root_rhs(P, &work);

        auto candidate =
            solve_until(P, tol, max_cycles, work);

        receipt.total_vcycles += candidate.cycles;

        const bool residual_ok =
            candidate.metrics.rel_residual
            <= accepted.rel_residual * 1.001 + 1e-12;

        const bool mse_ok =
            candidate.metrics.mse_truth
            <= accepted.mse_truth * 1.001 + 1e-12;

        if (residual_ok && mse_ok) {
            accepted = candidate.metrics;
            receipt.mean_delta_k = mean_dk;
            receipt.committed++;

            if (mean_dk < 1e-4)
                break;
        } else {
            F.u = u_prev;
            F.k = k_prev;
            F.rhs = rhs_prev;

            propagate_permeability(P);

            receipt.rolled_back++;
            adapt_rate *= 0.5f;

            if (adapt_rate < 1e-3f)
                break;
        }
    }

    receipt.final_metrics = accepted;

    const auto t1 = Clock::now();
    receipt.ms =
        std::chrono::duration<double, std::milli>(
            t1-t0).count();

    receipt.flops = work.flops - f0;
    return receipt;
}

bool self_test() {
    WorkCounter work{};
    Pyramid3D P(32,4);

    seed_truth(P);
    propagate_permeability(P);
    rebuild_root_rhs(P, &work);

    std::vector<f32> Ax;
    apply_operator(
        P.levels.front(),
        P.truth,
        Ax,
        &work);

    if (mse(Ax, P.levels.front().rhs) > 1e-14)
        return false;

    Level3D constant_fine(8);
    Level3D constant_coarse(4);

    std::fill(
        constant_fine.residual.begin(),
        constant_fine.residual.end(),
        2.0f);

    WorkCounter testwork{};

    restrict_residual(
        constant_fine,
        constant_coarse,
        testwork);

    for (f32 v : constant_coarse.rhs)
        if (std::fabs(v-2.0f) > 1e-6f)
            return false;

    std::fill(
        constant_coarse.u.begin(),
        constant_coarse.u.end(),
        3.0f);

    std::fill(
        constant_fine.u.begin(),
        constant_fine.u.end(),
        0.0f);

    prolong_trilinear_add(
        constant_fine,
        constant_coarse,
        testwork);

    for (f32 v : constant_fine.u)
        if (std::fabs(v-3.0f) > 1e-6f)
            return false;

    const auto before = metrics(P, work);

    vcycle(P, 0, work);

    const auto after = metrics(P, work);

    if (!(after.rel_residual < before.rel_residual))
        return false;

    auto receipt =
        adaptive_outer_loop(
            P,
            1e-4,
            2,
            15,
            0.08f,
            work);

    if (!std::isfinite(
            receipt.final_metrics.rel_residual)
        || !std::isfinite(
            receipt.final_metrics.mse_truth))
        return false;

    if (receipt.final_metrics.rel_residual
        > after.rel_residual * 1.01)
        return false;

    return true;
}

struct Config {
    int dim = 128;
    int min_dim = 4;
    int max_outer = 4;
    int max_cycles = 30;
    double tol = 1e-5;
    f32 adapt_rate = 0.08f;
    bool self_test_mode = false;
};

Config parse_args(int argc, char** argv) {
    Config c{};

    for (int i=1; i<argc; ++i) {
        const std::string a = argv[i];

        auto next = [&](const char* name) -> std::string {
            if (i+1 >= argc)
                throw std::invalid_argument(
                    std::string(name)
                    + " requires a value");

            return argv[++i];
        };

        if (a == "--dim")
            c.dim = std::stoi(next("--dim"));

        else if (a == "--min-dim")
            c.min_dim =
                std::stoi(next("--min-dim"));

        else if (a == "--outer")
            c.max_outer =
                std::stoi(next("--outer"));

        else if (a == "--cycles")
            c.max_cycles =
                std::stoi(next("--cycles"));

        else if (a == "--tol")
            c.tol =
                std::stod(next("--tol"));

        else if (a == "--adapt-rate")
            c.adapt_rate =
                std::stof(next("--adapt-rate"));

        else if (a == "--self-test")
            c.self_test_mode = true;

        else if (a == "--help" || a == "-h") {
            std::cout
                << "DrMoagi Adaptive Multigrid 3D\n"
                << "  --dim N          root power-of-two dimension (default 128)\n"
                << "  --min-dim N      coarsest power-of-two dimension (default 4)\n"
                << "  --outer N        max permeability transactions\n"
                << "  --cycles N       max V-cycles per transaction\n"
                << "  --tol X          relative residual tolerance\n"
                << "  --adapt-rate X   permeability proposal rate\n"
                << "  --self-test      run deterministic invariants\n";

            std::exit(0);
        } else {
            throw std::invalid_argument(
                "unknown argument: " + a);
        }
    }

    if (!(c.tol > 0.0)
        || !(c.adapt_rate > 0.0f
             && c.adapt_rate <= 1.0f)) {

        throw std::invalid_argument(
            "invalid tolerance/adapt-rate");
    }

    return c;
}

void print_report(
    const Config& cfg,
    const Pyramid3D& P,
    const OuterReceipt& R) {

    std::cout
        << "============================================================\n"
        << " DR MOAGI ADAPTIVE MULTIGRID 3D — VERIFIED V-CYCLE\n"
        << "============================================================\n"
        << " topology           : periodic 3D torus\n"
        << " operator           : 0.1*u + div_k(grad u) discretization\n"
        << " threads            : " << max_threads() << "\n"
        << " root               : " << cfg.dim << "^3\n"
        << " hierarchy          : ";

    for (const auto& L : P.levels)
        std::cout
            << L.D
            << (L.D == cfg.min_dim ? "" : " -> ");

    std::cout
        << "\n resident state     : "
        << std::fixed
        << std::setprecision(2)
        << (static_cast<double>(
                P.total_bytes())
            / 1048576.0)
        << " MiB\n"
        << " transactions       : "
        << R.attempted << " attempted, "
        << R.committed << " committed, "
        << R.rolled_back << " rolled back\n"
        << " V-cycles           : "
        << R.total_vcycles << "\n"
        << " relative residual  : "
        << std::scientific
        << std::setprecision(6)
        << R.final_metrics.rel_residual << "\n"
        << " MSE vs truth       : "
        << R.final_metrics.mse_truth << "\n"
        << " mean |delta k|     : "
        << R.mean_delta_k << "\n"
        << " measured wall time : "
        << std::fixed
        << std::setprecision(3)
        << R.ms << " ms\n"
        << " counted FLOPs      : "
        << R.flops
        << " (approximate kernel count)\n";

    if (R.ms > 0.0) {
        std::cout
            << " kernel GFLOP/s     : "
            << std::setprecision(3)
            << (static_cast<double>(R.flops)
                / (R.ms*1e6))
            << "\n";
    }

    std::cout
        << "============================================================\n";
}

} // namespace jarvisx::mg3d

int main(int argc, char** argv) {
    try {
        using namespace jarvisx::mg3d;

        const Config cfg =
            parse_args(argc, argv);

        if (cfg.self_test_mode) {
            const bool ok = self_test();

            std::cout
                << (ok
                    ? "adaptive-multigrid-3d self-test: PASS\n"
                    : "adaptive-multigrid-3d self-test: FAIL\n");

            return ok ? 0 : 2;
        }

        WorkCounter work{};
        Pyramid3D P(cfg.dim, cfg.min_dim);

        seed_truth(P);
        propagate_permeability(P);
        rebuild_root_rhs(P, &work);

        const auto receipt =
            adaptive_outer_loop(
                P,
                cfg.tol,
                cfg.max_outer,
                cfg.max_cycles,
                cfg.adapt_rate,
                work);

        print_report(cfg, P, receipt);
        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "error: "
            << e.what()
            << "\n";

        return 1;
    }
}
