#include "jarvisx/symmetry_loop3d.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

using jarvisx::symmetry3d::ClosedLoopOptimizer;
using jarvisx::symmetry3d::Grid;
using jarvisx::symmetry3d::SymmetryCodec;
using jarvisx::symmetry3d::Transport3;
using jarvisx::symmetry3d::format_grid;
using jarvisx::symmetry3d::format_transport;

int main() {
    const Grid x0{
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    };

    SymmetryCodec codec(3);
    const auto latent = codec.encode(x0);
    const auto shifted = codec.cyclic_shift_exact(latent);
    const auto x1 = codec.decode_majority(shifted);

    std::cout << "Jarvis-X 3D symmetry autoencoder closed loop\n\n";
    std::cout << "X0:\n" << format_grid(x0, 3, true) << "\n\n";
    std::cout << "Exact cyclic decoded state:\n" << format_grid(x1, 3, true) << "\n\n";

    ClosedLoopOptimizer optimizer(3);
    const auto report = optimizer.run(x0, Transport3::cyclic_inward());

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "initial objective: " << report.optimization.initial_loss.objective << '\n';
    std::cout << "final objective:   " << report.optimization.final_loss.objective << '\n';
    std::cout << "accepted moves:    " << report.optimization.accepted_moves << '\n';
    std::cout << "optimization sweeps: " << report.optimization.sweeps << "\n\n";
    std::cout << "optimized transport P_theta:\n"
              << format_transport(report.optimization.transport) << "\n\n";
    std::cout << "feedback steps:    " << report.feedback.steps << '\n';
    std::cout << "fixed-point MSE:   " << report.feedback.fixed_point_mse << '\n';
    std::cout << "reference MSE:     " << report.feedback.reference_mse << '\n';
    std::cout << "converged:         " << (report.feedback.converged ? "yes" : "no") << "\n\n";
    std::cout << "final binary state:\n"
              << format_grid(report.feedback.final_binary_state, 3, true) << '\n';

    return report.feedback.converged &&
                   report.optimization.final_loss.objective < report.optimization.initial_loss.objective
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}
