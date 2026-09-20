#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace jarvisx::dm6400 {

struct Geometry {
    static constexpr int nx = 20;
    static constexpr int ny = 20;
    static constexpr int nz = 16;
    static constexpr int panels = nx * ny * nz;
    static constexpr int cluster_side = 4;
    static constexpr int clusters_x = nx / cluster_side;
    static constexpr int clusters_y = ny / cluster_side;
    static constexpr int clusters_z = nz / cluster_side;
    static constexpr int clusters = clusters_x * clusters_y * clusters_z;
    static constexpr int panels_per_cluster = 64;
    static constexpr int latent = 8;

    static constexpr int id(int x, int y, int z) noexcept {
        return x + nx * y + nx * ny * z;
    }

    static constexpr int cluster_id(int x, int y, int z) noexcept {
        return (x / cluster_side)
             + clusters_x * (y / cluster_side)
             + clusters_x * clusters_y * (z / cluster_side);
    }

    static inline void xyz(int index, int& x, int& y, int& z) noexcept {
        z = index / (nx * ny);
        const int rem = index % (nx * ny);
        y = rem / nx;
        x = rem % nx;
    }

    static constexpr std::array<std::array<int, 3>, 6> multigrid_dims() noexcept {
        return {{{20, 20, 16}, {10, 10, 8}, {5, 5, 4}, {3, 3, 2}, {2, 2, 1}, {1, 1, 1}}};
    }

    static constexpr std::array<int, 6> multigrid_sizes() noexcept {
        return {{6400, 800, 100, 18, 4, 1}};
    }
};

using Latent = std::array<double, Geometry::latent>;

struct EngineConfig {
    int cycles = 4;
    int physical_workers = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int programmable_iterations = 1000;
    std::uint64_t seed = 7;
    double fixed_point_tolerance = 1.0e-5;
    double inner_tolerance = 2.5e-5;
    double active_tolerance = 5.0e-5;
    double relaxation = 0.28;
    double core_gain = 0.55;
    double smooth_gain = 0.18;
    double memory_rho = 0.90;
    int minimum_program_ops = 32;
    int convergence_window = 32;

    void validate() const {
        if (cycles <= 0) throw std::invalid_argument("cycles must be positive");
        if (physical_workers <= 0) throw std::invalid_argument("physical_workers must be positive");
        if (programmable_iterations <= 0) throw std::invalid_argument("programmable_iterations must be positive");
        if (!(fixed_point_tolerance >= 0.0)) throw std::invalid_argument("fixed_point_tolerance must be non-negative");
        if (!(inner_tolerance >= 0.0)) throw std::invalid_argument("inner_tolerance must be non-negative");
        if (!(active_tolerance >= 0.0)) throw std::invalid_argument("active_tolerance must be non-negative");
        if (!(relaxation > 0.0 && relaxation <= 1.0)) throw std::invalid_argument("relaxation must be in (0,1]");
        if (!(core_gain >= 0.0 && core_gain <= 1.0)) throw std::invalid_argument("core_gain must be in [0,1]");
        if (!(smooth_gain >= 0.0 && smooth_gain <= 1.0)) throw std::invalid_argument("smooth_gain must be in [0,1]");
        if (!(memory_rho >= 0.0 && memory_rho < 1.0)) throw std::invalid_argument("memory_rho must be in [0,1)");
        if (minimum_program_ops < 0) throw std::invalid_argument("minimum_program_ops must be non-negative");
        if (convergence_window <= 0) throw std::invalid_argument("convergence_window must be positive");
    }
};

struct PanelState {
    Latent x{};
    Latent z{};
    Latent omega{};
    Latent neighbour{};
    Latent reconstruction{};
    Latent reconstruction_residual{};
    Latent fixed_point_residual{};
    Latent previous_fixed_point_residual{};
    double reconstruction_mse = 0.0;
    double fixed_point_norm = 0.0;
    std::uint64_t logical_program_slots = 0;
    std::uint64_t executed_program_ops = 0;
    std::uint64_t corrections = 0;
    bool active = true;
    bool valid = true;
};

struct CycleReceipt {
    int cycle = 0;
    std::uint64_t logical_program_slots = 0;
    std::uint64_t executed_program_ops = 0;
    int active_panels = 0;
    int inactive_panels = 0;
    double execution_fraction = 0.0;
    double reconstruction_mse = 0.0;
    double fixed_point_residual = 0.0;
    double checksum = 0.0;
    std::uint64_t corrections = 0;
    double relaxation = 0.0;
    double memory_rho = 0.0;
    double cycle_ms = 0.0;
};

class WorkerPool {
public:
    explicit WorkerPool(int workers) : workers_(std::max(1, workers)) {}

    template <class Fn>
    void parallel_for(int n, Fn&& fn) const {
        std::atomic<int> next{0};
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(workers_));
        for (int worker = 0; worker < workers_; ++worker) {
            threads.emplace_back([&next, n, &fn]() {
                for (;;) {
                    const int index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= n) break;
                    fn(index);
                }
            });
        }
        for (auto& thread : threads) thread.join();
    }

    int size() const noexcept { return workers_; }

private:
    int workers_;
};

class DrMoagi6400x1000Engine {
public:
    explicit DrMoagi6400x1000Engine(EngineConfig config = {})
        : config_(std::move(config)),
          pool_(config_.physical_workers),
          panels_(static_cast<std::size_t>(Geometry::panels)),
          cluster_checksum_(static_cast<std::size_t>(Geometry::clusters), 0.0),
          cluster_loss_(static_cast<std::size_t>(Geometry::clusters), 0.0) {
        config_.validate();
        initialize();
    }

    const EngineConfig& config() const noexcept { return config_; }
    const std::vector<PanelState>& panels() const noexcept { return panels_; }
    int physical_workers() const noexcept { return pool_.size(); }

    CycleReceipt step() {
        using clock = std::chrono::steady_clock;
        const auto begin = clock::now();

        ingest(cycle_);
        exchange_neighbours();
        execute_programmable_kernels(cycle_);
        compute_fixed_point_residual();
        inward_outward_refine();
        residual_accelerate();
        decode_verify_remember();
        hierarchical_reduce();
        adapt_runtime_policy();

        CycleReceipt receipt;
        receipt.cycle = cycle_;
        receipt.logical_program_slots = static_cast<std::uint64_t>(Geometry::panels)
                                      * static_cast<std::uint64_t>(config_.programmable_iterations);
        receipt.executed_program_ops = last_executed_ops_;
        receipt.active_panels = last_active_panels_;
        receipt.inactive_panels = Geometry::panels - last_active_panels_;
        receipt.execution_fraction = receipt.logical_program_slots == 0U
            ? 0.0
            : static_cast<double>(receipt.executed_program_ops)
              / static_cast<double>(receipt.logical_program_slots);
        receipt.reconstruction_mse = global_loss_;
        receipt.fixed_point_residual = global_fixed_point_residual_;
        receipt.checksum = global_checksum_;
        receipt.corrections = corrections_total_;
        receipt.relaxation = config_.relaxation;
        receipt.memory_rho = config_.memory_rho;
        const auto end = clock::now();
        receipt.cycle_ms = std::chrono::duration<double, std::milli>(end - begin).count();

        receipts_.push_back(receipt);
        ++cycle_;
        return receipt;
    }

    std::vector<CycleReceipt> run(int cycles) {
        if (cycles < 0) throw std::invalid_argument("cycles must be non-negative");
        std::vector<CycleReceipt> result;
        result.reserve(static_cast<std::size_t>(cycles));
        for (int i = 0; i < cycles; ++i) result.push_back(step());
        return result;
    }

    const std::vector<CycleReceipt>& receipts() const noexcept { return receipts_; }

private:
    struct Level {
        int nx = 0;
        int ny = 0;
        int nz = 0;
        std::vector<Latent> values;
    };

    EngineConfig config_;
    WorkerPool pool_;
    std::vector<PanelState> panels_;
    std::vector<double> cluster_checksum_;
    std::vector<double> cluster_loss_;
    std::vector<CycleReceipt> receipts_;
    int cycle_ = 0;
    std::uint64_t last_executed_ops_ = 0;
    int last_active_panels_ = 0;
    std::uint64_t corrections_total_ = 0;
    double global_checksum_ = 0.0;
    double global_loss_ = 0.0;
    double global_fixed_point_residual_ = 0.0;
    double previous_global_loss_ = std::numeric_limits<double>::infinity();

    static double squash(double value) noexcept { return std::tanh(value); }

    static double norm_inf(const Latent& value) noexcept {
        double result = 0.0;
        for (const double lane : value) result = std::max(result, std::abs(lane));
        return result;
    }

    static double dot(const Latent& a, const Latent& b) noexcept {
        double value = 0.0;
        for (int lane = 0; lane < Geometry::latent; ++lane) {
            value += a[static_cast<std::size_t>(lane)] * b[static_cast<std::size_t>(lane)];
        }
        return value;
    }

    void initialize() {
        std::mt19937_64 rng(config_.seed);
        std::uniform_real_distribution<double> distribution(-0.35, 0.35);
        for (int index = 0; index < Geometry::panels; ++index) {
            int x = 0;
            int y = 0;
            int z = 0;
            Geometry::xyz(index, x, y, z);
            auto& panel = panels_[static_cast<std::size_t>(index)];
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const double coordinate = 0.010 * static_cast<double>(x)
                                        + 0.016 * static_cast<double>(y)
                                        + 0.022 * static_cast<double>(z)
                                        + 0.004 * static_cast<double>(lane);
                panel.x[static_cast<std::size_t>(lane)] = squash(coordinate + distribution(rng));
                panel.z[static_cast<std::size_t>(lane)] = 0.45 * panel.x[static_cast<std::size_t>(lane)];
            }
        }
    }

    void ingest(int cycle) {
        pool_.parallel_for(Geometry::panels, [&](int index) {
            int x = 0;
            int y = 0;
            int z = 0;
            Geometry::xyz(index, x, y, z);
            auto& panel = panels_[static_cast<std::size_t>(index)];
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const double wave = std::sin(0.031 * static_cast<double>((x + 1) * (cycle + 1)))
                                  + std::cos(0.027 * static_cast<double>((y + 1) * (lane + 1)))
                                  + std::sin(0.019 * static_cast<double>((z + 1) * (cycle + lane + 1)));
                const std::size_t k = static_cast<std::size_t>(lane);
                panel.x[k] = 0.74 * panel.x[k]
                           + 0.26 * squash(0.34 * wave + 0.14 * panel.omega[k]);
            }
        });
    }

    void exchange_neighbours() {
        pool_.parallel_for(Geometry::panels, [&](int index) {
            int x = 0;
            int y = 0;
            int z = 0;
            Geometry::xyz(index, x, y, z);
            auto& panel = panels_[static_cast<std::size_t>(index)];
            panel.neighbour.fill(0.0);
            int degree = 0;
            constexpr std::array<int, 6> dx{{-1, 1, 0, 0, 0, 0}};
            constexpr std::array<int, 6> dy{{0, 0, -1, 1, 0, 0}};
            constexpr std::array<int, 6> dz{{0, 0, 0, 0, -1, 1}};
            for (int direction = 0; direction < 6; ++direction) {
                const int xx = x + dx[static_cast<std::size_t>(direction)];
                const int yy = y + dy[static_cast<std::size_t>(direction)];
                const int zz = z + dz[static_cast<std::size_t>(direction)];
                if (xx < 0 || xx >= Geometry::nx || yy < 0 || yy >= Geometry::ny || zz < 0 || zz >= Geometry::nz) continue;
                const auto& neighbour = panels_[static_cast<std::size_t>(Geometry::id(xx, yy, zz))];
                for (int lane = 0; lane < Geometry::latent; ++lane) {
                    panel.neighbour[static_cast<std::size_t>(lane)] += neighbour.z[static_cast<std::size_t>(lane)];
                }
                ++degree;
            }
            if (degree > 0) {
                const double inv = 1.0 / static_cast<double>(degree);
                for (double& value : panel.neighbour) value *= inv;
            }
        });
    }

    void execute_programmable_kernels(int cycle) {
        std::atomic<std::uint64_t> executed{0};
        std::atomic<int> active{0};

        pool_.parallel_for(Geometry::panels, [&](int index) {
            auto& panel = panels_[static_cast<std::size_t>(index)];
            panel.logical_program_slots += static_cast<std::uint64_t>(config_.programmable_iterations);

            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                panel.z[k] = squash(0.58 * panel.x[k]
                                  + 0.24 * panel.omega[k]
                                  + 0.18 * panel.neighbour[k]);
            }

            Latent initial_residual{};
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                const std::size_t j = static_cast<std::size_t>((lane + 1) % Geometry::latent);
                const double target = squash(0.50 * panel.z[k]
                                           + 0.18 * panel.z[j]
                                           + 0.14 * panel.neighbour[k]
                                           + 0.10 * panel.omega[k]
                                           + 0.08 * panel.x[k]);
                initial_residual[k] = target - panel.z[k];
            }
            panel.active = norm_inf(initial_residual) > config_.active_tolerance;
            if (!panel.active) return;
            active.fetch_add(1, std::memory_order_relaxed);

            std::uint64_t state = (static_cast<std::uint64_t>(index) + 1U) * 0x9E3779B97F4A7C15ULL
                                ^ (static_cast<std::uint64_t>(cycle) + 1U) * 0xBF58476D1CE4E5B9ULL;
            int stable_count = 0;
            int performed = 0;
            for (int op = 0; op < config_.programmable_iterations; ++op) {
                state ^= state >> 12U;
                state ^= state << 25U;
                state ^= state >> 27U;
                const int lane = static_cast<int>((state + static_cast<std::uint64_t>(op))
                                                 % static_cast<std::uint64_t>(Geometry::latent));
                const int peer = (lane + 1 + (op & 3)) % Geometry::latent;
                const std::size_t k = static_cast<std::size_t>(lane);
                const std::size_t j = static_cast<std::size_t>(peer);
                const double target = squash(0.48 * panel.z[k]
                                           + 0.18 * panel.z[j]
                                           + 0.14 * panel.neighbour[k]
                                           + 0.11 * panel.omega[k]
                                           + 0.09 * panel.x[k]);
                const double delta = target - panel.z[k];
                panel.z[k] += config_.relaxation * delta;
                ++performed;

                if (std::abs(delta) <= config_.inner_tolerance) {
                    ++stable_count;
                } else {
                    stable_count = 0;
                }

                if (performed >= config_.minimum_program_ops
                    && stable_count >= config_.convergence_window) {
                    break;
                }
            }
            panel.executed_program_ops += static_cast<std::uint64_t>(performed);
            executed.fetch_add(static_cast<std::uint64_t>(performed), std::memory_order_relaxed);
        });

        last_executed_ops_ = executed.load(std::memory_order_relaxed);
        last_active_panels_ = active.load(std::memory_order_relaxed);
    }

    void compute_fixed_point_residual() {
        std::atomic<double> max_residual{0.0};
        pool_.parallel_for(Geometry::panels, [&](int index) {
            auto& panel = panels_[static_cast<std::size_t>(index)];
            panel.previous_fixed_point_residual = panel.fixed_point_residual;
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                const std::size_t j = static_cast<std::size_t>((lane + 1) % Geometry::latent);
                const double target = squash(0.52 * panel.z[k]
                                           + 0.16 * panel.z[j]
                                           + 0.14 * panel.neighbour[k]
                                           + 0.10 * panel.omega[k]
                                           + 0.08 * panel.x[k]);
                panel.fixed_point_residual[k] = target - panel.z[k];
            }
            panel.fixed_point_norm = norm_inf(panel.fixed_point_residual);

            double observed = max_residual.load(std::memory_order_relaxed);
            while (observed < panel.fixed_point_norm
                   && !max_residual.compare_exchange_weak(observed, panel.fixed_point_norm,
                                                          std::memory_order_relaxed,
                                                          std::memory_order_relaxed)) {
            }
        });
        global_fixed_point_residual_ = max_residual.load(std::memory_order_relaxed);
    }

    static Level restrict_level(const Level& fine, int coarse_x, int coarse_y, int coarse_z) {
        Level coarse;
        coarse.nx = coarse_x;
        coarse.ny = coarse_y;
        coarse.nz = coarse_z;
        const int count = coarse_x * coarse_y * coarse_z;
        coarse.values.assign(static_cast<std::size_t>(count), Latent{});
        std::vector<int> samples(static_cast<std::size_t>(count), 0);

        for (int z = 0; z < fine.nz; ++z) {
            const int cz = std::min(coarse_z - 1, (z * coarse_z) / fine.nz);
            for (int y = 0; y < fine.ny; ++y) {
                const int cy = std::min(coarse_y - 1, (y * coarse_y) / fine.ny);
                for (int x = 0; x < fine.nx; ++x) {
                    const int cx = std::min(coarse_x - 1, (x * coarse_x) / fine.nx);
                    const int fine_index = x + fine.nx * y + fine.nx * fine.ny * z;
                    const int coarse_index = cx + coarse_x * cy + coarse_x * coarse_y * cz;
                    auto& destination = coarse.values[static_cast<std::size_t>(coarse_index)];
                    const auto& source = fine.values[static_cast<std::size_t>(fine_index)];
                    for (int lane = 0; lane < Geometry::latent; ++lane) {
                        destination[static_cast<std::size_t>(lane)] += source[static_cast<std::size_t>(lane)];
                    }
                    ++samples[static_cast<std::size_t>(coarse_index)];
                }
            }
        }

        for (int index = 0; index < count; ++index) {
            const int n = samples[static_cast<std::size_t>(index)];
            if (n <= 0) continue;
            const double inv = 1.0 / static_cast<double>(n);
            for (double& lane : coarse.values[static_cast<std::size_t>(index)]) lane *= inv;
        }
        return coarse;
    }

    static Level prolongate(const Level& coarse, int fine_x, int fine_y, int fine_z) {
        Level fine;
        fine.nx = fine_x;
        fine.ny = fine_y;
        fine.nz = fine_z;
        fine.values.assign(static_cast<std::size_t>(fine_x * fine_y * fine_z), Latent{});

        for (int z = 0; z < fine_z; ++z) {
            const int cz = std::min(coarse.nz - 1, (z * coarse.nz) / fine_z);
            for (int y = 0; y < fine_y; ++y) {
                const int cy = std::min(coarse.ny - 1, (y * coarse.ny) / fine_y);
                for (int x = 0; x < fine_x; ++x) {
                    const int cx = std::min(coarse.nx - 1, (x * coarse.nx) / fine_x);
                    const int fine_index = x + fine_x * y + fine_x * fine_y * z;
                    const int coarse_index = cx + coarse.nx * cy + coarse.nx * coarse.ny * cz;
                    fine.values[static_cast<std::size_t>(fine_index)] = coarse.values[static_cast<std::size_t>(coarse_index)];
                }
            }
        }
        return fine;
    }

    void inward_outward_refine() {
        const auto dims = Geometry::multigrid_dims();
        std::vector<Level> levels;
        levels.reserve(dims.size());

        Level fine;
        fine.nx = Geometry::nx;
        fine.ny = Geometry::ny;
        fine.nz = Geometry::nz;
        fine.values.resize(static_cast<std::size_t>(Geometry::panels));
        for (int index = 0; index < Geometry::panels; ++index) {
            fine.values[static_cast<std::size_t>(index)] = panels_[static_cast<std::size_t>(index)].fixed_point_residual;
        }
        levels.push_back(std::move(fine));

        for (std::size_t level = 1; level < dims.size(); ++level) {
            levels.push_back(restrict_level(levels.back(),
                                            dims[level][0], dims[level][1], dims[level][2]));
        }

        Level correction = levels.back();
        for (auto& vector : correction.values) {
            for (double& lane : vector) lane *= config_.core_gain;
        }

        for (std::size_t reverse = levels.size() - 1; reverse > 0; --reverse) {
            const auto& target = levels[reverse - 1];
            Level expanded = prolongate(correction, target.nx, target.ny, target.nz);
            for (std::size_t index = 0; index < expanded.values.size(); ++index) {
                for (int lane = 0; lane < Geometry::latent; ++lane) {
                    const std::size_t k = static_cast<std::size_t>(lane);
                    expanded.values[index][k] += config_.smooth_gain * target.values[index][k];
                }
            }
            correction = std::move(expanded);
        }

        pool_.parallel_for(Geometry::panels, [&](int index) {
            auto& panel = panels_[static_cast<std::size_t>(index)];
            const auto& delta = correction.values[static_cast<std::size_t>(index)];
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                panel.z[k] = std::clamp(panel.z[k] + delta[k], -1.0, 1.0);
            }
        });
    }

    void residual_accelerate() {
        if (cycle_ == 0) return;
        pool_.parallel_for(Geometry::panels, [&](int index) {
            auto& panel = panels_[static_cast<std::size_t>(index)];
            Latent delta{};
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                delta[k] = panel.fixed_point_residual[k] - panel.previous_fixed_point_residual[k];
            }
            const double denominator = dot(delta, delta) + 1.0e-18;
            double alpha = -dot(panel.fixed_point_residual, delta) / denominator;
            alpha = std::clamp(alpha, -0.50, 1.00);
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                const double correction = 0.12 * alpha * delta[k];
                panel.z[k] = std::clamp(panel.z[k] + correction, -1.0, 1.0);
            }
        });
    }

    void decode_verify_remember() {
        pool_.parallel_for(Geometry::panels, [&](int index) {
            auto& panel = panels_[static_cast<std::size_t>(index)];
            double loss = 0.0;
            panel.valid = true;
            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                panel.reconstruction[k] = squash(0.90 * panel.z[k] + 0.10 * panel.omega[k]);
                panel.reconstruction_residual[k] = panel.x[k] - panel.reconstruction[k];
                loss += panel.reconstruction_residual[k] * panel.reconstruction_residual[k];
                panel.valid = panel.valid && std::isfinite(panel.z[k]) && std::isfinite(panel.reconstruction[k]);
            }
            panel.reconstruction_mse = loss / static_cast<double>(Geometry::latent);
            panel.valid = panel.valid && std::isfinite(panel.reconstruction_mse) && panel.reconstruction_mse < 4.0;

            if (!panel.valid) {
                ++panel.corrections;
                for (int lane = 0; lane < Geometry::latent; ++lane) {
                    const std::size_t k = static_cast<std::size_t>(lane);
                    panel.z[k] = std::clamp(std::isfinite(panel.z[k]) ? panel.z[k] : 0.0, -1.0, 1.0);
                    panel.omega[k] *= 0.5;
                    panel.reconstruction[k] = squash(panel.z[k]);
                    panel.reconstruction_residual[k] = panel.x[k] - panel.reconstruction[k];
                }
                loss = 0.0;
                for (const double residual : panel.reconstruction_residual) loss += residual * residual;
                panel.reconstruction_mse = loss / static_cast<double>(Geometry::latent);
                panel.valid = true;
            }

            for (int lane = 0; lane < Geometry::latent; ++lane) {
                const std::size_t k = static_cast<std::size_t>(lane);
                panel.omega[k] = config_.memory_rho * panel.omega[k]
                               + (1.0 - config_.memory_rho)
                                 * (panel.z[k] + 0.25 * panel.reconstruction_residual[k]);
            }
        });
    }

    void hierarchical_reduce() {
        std::fill(cluster_checksum_.begin(), cluster_checksum_.end(), 0.0);
        std::fill(cluster_loss_.begin(), cluster_loss_.end(), 0.0);

        pool_.parallel_for(Geometry::clusters, [&](int cluster) {
            const int cz = cluster / (Geometry::clusters_x * Geometry::clusters_y);
            const int rem = cluster % (Geometry::clusters_x * Geometry::clusters_y);
            const int cy = rem / Geometry::clusters_x;
            const int cx = rem % Geometry::clusters_x;
            double checksum = 0.0;
            double loss = 0.0;
            for (int lz = 0; lz < Geometry::cluster_side; ++lz) {
                for (int ly = 0; ly < Geometry::cluster_side; ++ly) {
                    for (int lx = 0; lx < Geometry::cluster_side; ++lx) {
                        const int x = cx * Geometry::cluster_side + lx;
                        const int y = cy * Geometry::cluster_side + ly;
                        const int z = cz * Geometry::cluster_side + lz;
                        const auto& panel = panels_[static_cast<std::size_t>(Geometry::id(x, y, z))];
                        checksum += std::accumulate(panel.reconstruction.begin(), panel.reconstruction.end(), 0.0);
                        loss += panel.reconstruction_mse;
                    }
                }
            }
            cluster_checksum_[static_cast<std::size_t>(cluster)] = checksum;
            cluster_loss_[static_cast<std::size_t>(cluster)] = loss / static_cast<double>(Geometry::panels_per_cluster);
        });

        global_checksum_ = std::accumulate(cluster_checksum_.begin(), cluster_checksum_.end(), 0.0);
        global_loss_ = std::accumulate(cluster_loss_.begin(), cluster_loss_.end(), 0.0)
                     / static_cast<double>(Geometry::clusters);
        corrections_total_ = 0;
        for (const auto& panel : panels_) corrections_total_ += panel.corrections;
    }

    void adapt_runtime_policy() {
        const bool improving = global_loss_ <= previous_global_loss_;
        if (!improving || global_fixed_point_residual_ > 0.20) {
            config_.relaxation = std::max(0.08, config_.relaxation * 0.96);
            config_.memory_rho = std::min(0.97, config_.memory_rho + 0.002);
        } else {
            config_.relaxation = std::min(0.45, config_.relaxation * 1.01);
            config_.memory_rho = std::max(0.82, config_.memory_rho - 0.0005);
        }
        previous_global_loss_ = global_loss_;
    }
};

} // namespace jarvisx::dm6400
