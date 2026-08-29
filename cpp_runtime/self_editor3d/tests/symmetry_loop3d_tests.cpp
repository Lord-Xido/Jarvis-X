#include "jarvisx/symmetry_loop3d.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace jarvisx::symmetry3d;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same_grid(const Grid& a, const Grid& b, double eps = 1.0e-12) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

Grid fixture() {
    return Grid{
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    };
}

void test_exact_reconstruction() {
    SymmetryCodec codec(3);
    const Grid x = fixture();
    const Grid reconstructed = codec.decode_majority(codec.encode(x));
    require(same_grid(x, reconstructed), "D(E(X)) must reconstruct binary X exactly");
}

void test_exact_cyclic_period_and_example() {
    SymmetryCodec codec(3);
    const Grid x = fixture();
    const Tensor3 z0 = codec.encode(x);
    const Tensor3 z1 = codec.cyclic_shift_exact(z0);
    const Tensor3 z2 = codec.cyclic_shift_exact(z1);
    const Tensor3 z3 = codec.cyclic_shift_exact(z2);
    for (std::size_t k = 0; k < 3; ++k) {
        require(same_grid(z0.layer[k], z3.layer[k]), "exact latent shift must satisfy P^3=I");
    }

    const Grid expected_x1{
        1.0, 1.0, 1.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const Grid x1 = codec.decode_majority(z1);
    require(same_grid(x1, expected_x1), "example cyclic decode must match supplied X1");

    const Grid x2 = codec.decode_majority(codec.cyclic_shift_exact(codec.encode(x1)));
    require(same_grid(x2, x), "supplied fixture must have decoded period two");
}

void test_transport_is_row_stochastic() {
    const auto w = Transport3::cyclic_inward().weights();
    for (std::size_t r = 0; r < 3; ++r) {
        double sum = 0.0;
        for (std::size_t c = 0; c < 3; ++c) {
            require(w[r * 3 + c] >= 0.0 && w[r * 3 + c] <= 1.0,
                    "transport probability out of range");
            sum += w[r * 3 + c];
        }
        require(std::abs(sum - 1.0) < 1.0e-12, "transport row must sum to one");
    }
}

void test_closed_loop_optimizes_and_converges() {
    LoopConfig config;
    config.max_optimization_sweeps = 64;
    config.max_feedback_steps = 64;
    ClosedLoopOptimizer optimizer(3, config);
    const Grid x = fixture();
    const ClosedLoopReport report = optimizer.run(x, Transport3::cyclic_inward());

    require(report.optimization.final_loss.objective <
                0.01 * report.optimization.initial_loss.objective,
            "closed loop must materially lower the objective");
    require(report.optimization.accepted_moves > 0,
            "closed loop must accept at least one parameter improvement");
    require(report.feedback.converged, "feedback loop must reach a fixed point");
    require(report.feedback.fixed_point_mse <= config.fixed_point_tolerance,
            "fixed-point residual exceeds tolerance");
    require(report.feedback.reference_mse < 1.0e-8,
            "optimized feedback state must reconstruct the invariant reference");
    require(same_grid(report.feedback.final_binary_state, x),
            "hard decoded fixed point must equal original binary frame");
}

}  // namespace

int main() {
    try {
        test_exact_reconstruction();
        test_exact_cyclic_period_and_example();
        test_transport_is_row_stochastic();
        test_closed_loop_optimizes_and_converges();
        std::cout << "symmetry-loop3d regressions: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "symmetry-loop3d regressions: FAIL: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
