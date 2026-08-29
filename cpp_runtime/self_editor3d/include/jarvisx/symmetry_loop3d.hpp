#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace jarvisx {
namespace symmetry3d {

using Grid = std::vector<double>;

struct Tensor3 {
    std::size_t n{};
    std::array<Grid, 3> layer;
};

struct Transport3 {
    std::array<double, 9> logits{};

    static Transport3 identity(double strength = 6.0);
    static Transport3 cyclic_inward(double strength = 6.0);
    std::array<double, 9> weights() const;
};

struct LossMetrics {
    double reconstruction_mse{};
    double latent_cycle_mse{};
    double fixed_point_mse{};
    double transport_entropy{};
    double objective{};
};

struct LoopConfig {
    double cycle_weight{0.25};
    double fixed_point_weight{0.25};
    double entropy_weight{0.002};
    double coordinate_step{1.0};
    double min_coordinate_step{1.0e-4};
    double improvement_epsilon{1.0e-12};
    double fixed_point_tolerance{1.0e-12};
    double hard_threshold{0.5};
    std::size_t max_optimization_sweeps{64};
    std::size_t max_feedback_steps{64};
};

struct OptimizationReport {
    Transport3 transport;
    LossMetrics initial_loss;
    LossMetrics final_loss;
    std::size_t sweeps{};
    std::size_t accepted_moves{};
    double final_coordinate_step{};
};

struct FeedbackReport {
    Grid final_state;
    Grid final_binary_state;
    std::size_t steps{};
    double fixed_point_mse{};
    double reference_mse{};
    bool converged{false};
};

struct ClosedLoopReport {
    OptimizationReport optimization;
    FeedbackReport feedback;
};

class SymmetryCodec {
public:
    explicit SymmetryCodec(std::size_t n);

    std::size_t n() const noexcept { return n_; }
    Tensor3 encode(const Grid& x) const;
    Tensor3 apply_transport(const Tensor3& latent, const Transport3& transport) const;
    Tensor3 cyclic_shift_exact(const Tensor3& latent) const;
    Grid decode_soft(const Tensor3& latent) const;
    Grid decode_majority(const Tensor3& latent) const;
    Grid hard_threshold(const Grid& x, double threshold = 0.5) const;

private:
    std::size_t n_{};

    void validate_grid(const Grid& x) const;
    Grid horizontal(const Grid& x) const;
    Grid vertical(const Grid& x) const;
};

class ClosedLoopOptimizer {
public:
    ClosedLoopOptimizer(std::size_t n, LoopConfig config = {});

    LossMetrics evaluate(const Grid& source,
                         const Grid& reference,
                         const Transport3& transport) const;

    OptimizationReport optimize(const Grid& reference,
                                Transport3 initial = Transport3::cyclic_inward()) const;

    FeedbackReport close_feedback(const Grid& seed,
                                  const Grid& reference,
                                  const Transport3& optimized) const;

    ClosedLoopReport run(const Grid& reference,
                         Transport3 initial = Transport3::cyclic_inward()) const;

private:
    SymmetryCodec codec_;
    LoopConfig config_;

    static double mse(const Grid& a, const Grid& b);
    static double tensor_mse(const Tensor3& a, const Tensor3& b);
    static double entropy(const Transport3& transport);
};

std::string format_grid(const Grid& grid, std::size_t n, bool binary = false);
std::string format_transport(const Transport3& transport);

}  // namespace symmetry3d
}  // namespace jarvisx
