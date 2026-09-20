#include "jarvisx/dr_moagi_6400x1000.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    using namespace jarvisx::dm6400;

    require(Geometry::panels == 6400, "geometry must contain exactly 6400 panels");
    require(Geometry::clusters == 100, "geometry must contain exactly 100 clusters");
    require(Geometry::panels_per_cluster == 64, "cluster must contain 64 panels");
    require(Geometry::id(19, 19, 15) == 6399, "linear address mapping must end at 6399");
    require(Geometry::cluster_id(19, 19, 15) == 99, "cluster address mapping must end at 99");

    const auto sizes = Geometry::multigrid_sizes();
    require(sizes[0] == 6400 && sizes[1] == 800 && sizes[2] == 100
         && sizes[3] == 18 && sizes[4] == 4 && sizes[5] == 1,
            "inward hierarchy must be 6400->800->100->18->4->1");

    EngineConfig cfg;
    cfg.cycles = 1;
    cfg.physical_workers = 1;
    cfg.programmable_iterations = 64;
    cfg.minimum_program_ops = 8;
    cfg.convergence_window = 8;
    cfg.seed = 11;

    DrMoagi6400x1000Engine first(cfg);
    DrMoagi6400x1000Engine second(cfg);
    const CycleReceipt a = first.step();
    const CycleReceipt b = second.step();

    require(a.logical_program_slots == 6400ULL * 64ULL, "logical slot count must be exact");
    require(a.executed_program_ops <= a.logical_program_slots, "executed work cannot exceed logical slots");
    require(a.active_panels >= 0 && a.active_panels <= 6400, "active panel count must be bounded");
    require(std::isfinite(a.reconstruction_mse), "reconstruction MSE must be finite");
    require(std::isfinite(a.fixed_point_residual), "fixed-point residual must be finite");
    require(std::isfinite(a.checksum), "checksum must be finite");
    require(std::abs(a.reconstruction_mse - b.reconstruction_mse) < 1.0e-12,
            "single-worker seeded runs must be deterministic");
    require(std::abs(a.checksum - b.checksum) < 1.0e-12,
            "single-worker seeded checksums must be deterministic");

    for (const auto& panel : first.panels()) {
        require(panel.valid, "all panels must pass verification after correction");
        require(std::isfinite(panel.reconstruction_mse), "panel MSE must remain finite");
        for (const double z : panel.z) {
            require(std::isfinite(z), "latent state must remain finite");
            require(std::abs(z) <= 1.000001, "latent state must remain bounded");
        }
    }

    std::cout << "dr-moagi-6400x1000 tests passed\n";
    return 0;
}
