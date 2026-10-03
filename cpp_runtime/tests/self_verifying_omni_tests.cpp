#include "jarvisx/self_verifying_omni.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}

template <typename Function> void rejects(Function function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, message);
}

} // namespace

int main() {
    using namespace jarvisx::omni;

    // Reproduce the submitted polynomial's algebraic failure, not floating drift.
    const double t2 = pi * pi;
    const double legacy_c = 1.0 - t2 / 2.0 + t2 * t2 / 24.0;
    const double legacy_s = pi * (1.0 - t2 / 6.0 + t2 * t2 / 120.0);
    require(std::abs(legacy_c*legacy_c + legacy_s*legacy_s - 1.0) > 0.5,
            "legacy truncated rotation must demonstrate loss of norm");
    require(std::pow(0.8, 16) > 0.01 && std::pow(0.8, 21) < 0.01,
            "16 steps cannot reach the budget cutoff; earliest possible is 21");
    for (int angle = -200; angle <= 200; ++angle) {
        double x = 1.0, y = 0.0;
        for (int repeat = 0; repeat < 64; ++repeat) rotate_token(x, y, pi * angle / 200.0);
        require(std::abs(x*x + y*y - 1.0) < norm_tolerance, "rotation must conserve norm across full angle domain");
    }

    const SpatialAttractorGrid grid;
    require(grid.query(1.0F, 1.0F, 1.0F) == grid.field_density[4095], "upper endpoint must address last cell");
    require(grid.query(-1.0F, 0.0F, 0.0F) == grid.query(0.0F, 0.0F, 0.0F), "negative lookup must clamp safely");
    require(grid.query(2.0F, 1.0F, 1.0F) == grid.query(1.0F, 1.0F, 1.0F), "large lookup must clamp safely");
    rejects([&] { grid.query(std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F); }, "nonfinite lookup must reject");

    Config config;
    config.pathways = 1031; // non-multiple of SIMD width and static thread partitions
    config.requested_threads = 1;
    config.seed = 17;
    SelfVerifyingOmniEngine serial(config);
    require(sizeof(serial) < 65536, "million-pathway payload must not live on the stack");
    const auto aligned = [](const auto& vector) { return reinterpret_cast<std::uintptr_t>(vector.data()) % 64U == 0U; };
    for (const auto& axis : serial.manifold().position) require(aligned(axis), "position must be 64-byte aligned");
    for (const auto& channel : serial.manifold().state) require(aligned(channel), "state channel must be 64-byte aligned");
    for (const auto& token : serial.manifold().token) require(aligned(token), "verification token must be aligned");
    require(serial.resident_payload_bytes() == 80U * config.pathways + 16384U, "resident payload must be explicit");
    require(serial.verify_end_to_end_mechanics().passed(), "fresh manifold must satisfy audit");
    const auto stats = serial.execute_verified_inward_loop();
    const auto report = serial.verify_end_to_end_mechanics();
    require(report.passed(), "16-step mechanics must pass independent verification");
    require(report.pruned == 0 && report.active == config.pathways, "16-step run must not claim pruning");
    require(stats.executed_updates == config.pathways * config.depth, "executed updates must be counted exactly");
    require(report.final_mean_radius < report.initial_mean_radius, "actual geometry must contract inward");
    require(report.converged < config.pathways, "valid execution must remain distinct from full convergence");

    Config parallel_config = config;
    parallel_config.requested_threads = 4;
    SelfVerifyingOmniEngine parallel(parallel_config);
    const auto parallel_stats = parallel.execute_verified_inward_loop();
    require(parallel_stats.worker_threads >= 1 && parallel_stats.worker_threads <= 4, "actual team must be bounded");
    require(parallel_stats.executed_updates == stats.executed_updates, "parallel scheduling must count same work");
    require(parallel.verify_end_to_end_mechanics().passed(), "parallel mechanics must verify");
    const auto& a = serial.manifold();
    const auto& b = parallel.manifold();
    require(a.position == b.position && a.state == b.state && a.token == b.token &&
            a.prediction_error == b.prediction_error && a.compute_budget == b.compute_budget &&
            a.active_mask == b.active_mask, "seeded authoritative arrays must match across thread counts");

    // Corrupt copies of public audit inputs: a green aggregate cannot hide bad nodes.
    auto bad = serial.manifold();
    bad.token[0][0] = std::sqrt(1.01); bad.token[1][0] = 0.0;
    bad.token[0][1] = std::sqrt(0.99); bad.token[1][1] = 0.0;
    require(!verify_manifold(bad, config, stats).passed(), "opposite norm errors must not cancel into a pass");
    bad = serial.manifold(); bad.active_mask[0] = 2;
    require(!verify_manifold(bad, config, stats).passed(), "invalid mask values must fail");
    bad = serial.manifold(); bad.compute_budget[0] = -1.0F;
    require(!verify_manifold(bad, config, stats).passed(), "invalid budget must fail");
    bad = serial.manifold(); bad.state[0][0] = std::numeric_limits<float>::quiet_NaN();
    require(!verify_manifold(bad, config, stats).passed(), "nonfinite cognitive state must fail");
    bad = serial.manifold(); bad.token[0][0] = std::numeric_limits<double>::infinity();
    require(!verify_manifold(bad, config, stats).passed(), "infinite verification token must fail");
    bad = serial.manifold(); bad.prediction_error[0] = 0.5F;
    require(!verify_manifold(bad, config, stats).passed(), "cached residual must agree with independent calculation");
    bad = serial.manifold(); bad.active_mask[0] = 0; bad.compute_budget[0] = 0.0F;
    require(!verify_manifold(bad, config, stats).passed(), "premature pruning must fail");
    bad = serial.manifold(); bad.position[0][0] = 1.5F;
    require(!verify_manifold(bad, config, stats).passed(), "out-of-domain geometry must fail");
    bad = serial.manifold(); bad.state[0].pop_back();
    require(!verify_manifold(bad, config, stats).passed(), "malformed shape must fail before indexed reads");
    auto failed_stats = stats; failed_stats.rejected_updates = 1;
    require(!verify_manifold(serial.manifold(), config, failed_stats).passed(), "rejected candidate receipt must fail audit");

    Config deep_config;
    deep_config.pathways = 257;
    deep_config.depth = 1024;
    deep_config.requested_threads = 2;
    SelfVerifyingOmniEngine deep(deep_config);
    const auto deep_stats = deep.execute_verified_inward_loop();
    const auto deep_report = deep.verify_end_to_end_mechanics();
    require(deep_report.passed() && deep_report.converged == deep_config.pathways &&
            deep_report.pruned == deep_config.pathways, "long recurrence must converge and prune only admissible pathways");
    require(deep_stats.executed_updates < deep_config.pathways * deep_config.depth, "pruning must skip real updates");
    const auto frozen = deep.manifold();
    require(deep.execute_verified_inward_loop().executed_updates == 0, "pruned pathways must remain inactive");
    require(deep.manifold().position == frozen.position && deep.manifold().state == frozen.state &&
            deep.manifold().token == frozen.token && deep.verify_end_to_end_mechanics().passed(),
            "continued execution must preserve frozen converged state");

    Config strict = config;
    strict.depth = 1024;
    strict.residual_mse_tolerance = 1.0e-20;
    SelfVerifyingOmniEngine strict_engine(strict);
    strict_engine.execute_verified_inward_loop();
    require(strict_engine.verify_end_to_end_mechanics().passed() &&
            strict_engine.verify_end_to_end_mechanics().converged < strict.pathways,
            "bounded termination must not imply convergence below floating precision");

    Config invalid = config; invalid.pathways = 0;
    rejects([&] { SelfVerifyingOmniEngine engine(invalid); }, "zero allocation must reject");
    invalid.pathways = maximum_pathways + 1;
    rejects([&] { SelfVerifyingOmniEngine engine(invalid); }, "oversize allocation must reject before allocating");
    invalid = config; invalid.depth = 0;
    rejects([&] { SelfVerifyingOmniEngine engine(invalid); }, "zero depth must reject");
    invalid.depth = 4097;
    rejects([&] { SelfVerifyingOmniEngine engine(invalid); }, "oversize depth must reject");
    invalid = config; invalid.requested_threads = 0;
    rejects([&] { SelfVerifyingOmniEngine engine(invalid); }, "zero thread count must reject");
    invalid = config; invalid.spatial_tolerance = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { SelfVerifyingOmniEngine engine(invalid); }, "nonfinite tolerance must reject");

    std::cout << "self-verifying omni regressions passed\n";
}
