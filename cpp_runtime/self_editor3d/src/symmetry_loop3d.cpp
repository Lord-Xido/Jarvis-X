#include "jarvisx/symmetry_loop3d.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace jarvisx {
namespace symmetry3d {
namespace {

std::size_t idx(std::size_t n, std::size_t i, std::size_t j) {
    return i * n + j;
}

double clamp_logit(double x) {
    return std::max(-8.0, std::min(8.0, x));
}

}  // namespace

Transport3 Transport3::identity(double strength) {
    Transport3 t;
    t.logits.fill(-strength);
    for (std::size_t r = 0; r < 3; ++r) {
        t.logits[r * 3 + r] = strength;
    }
    return t;
}

Transport3 Transport3::cyclic_inward(double strength) {
    // Operational convention from the supplied example:
    // [L0, L1, L2] -> [L2, L0, L1].
    Transport3 t;
    t.logits.fill(-strength);
    const std::array<std::size_t, 3> source{{2, 0, 1}};
    for (std::size_t r = 0; r < 3; ++r) {
        t.logits[r * 3 + source[r]] = strength;
    }
    return t;
}

std::array<double, 9> Transport3::weights() const {
    std::array<double, 9> w{};
    for (std::size_t r = 0; r < 3; ++r) {
        double row_max = logits[r * 3];
        for (std::size_t c = 1; c < 3; ++c) {
            row_max = std::max(row_max, logits[r * 3 + c]);
        }
        double denom = 0.0;
        for (std::size_t c = 0; c < 3; ++c) {
            const double e = std::exp(logits[r * 3 + c] - row_max);
            w[r * 3 + c] = e;
            denom += e;
        }
        for (std::size_t c = 0; c < 3; ++c) {
            w[r * 3 + c] /= denom;
        }
    }
    return w;
}

SymmetryCodec::SymmetryCodec(std::size_t n) : n_(n) {
    if (n_ == 0) {
        throw std::invalid_argument("grid side must be positive");
    }
}

void SymmetryCodec::validate_grid(const Grid& x) const {
    if (x.size() != n_ * n_) {
        throw std::invalid_argument("grid size does not match codec side");
    }
    for (double v : x) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("grid contains a non-finite value");
        }
    }
}

Grid SymmetryCodec::horizontal(const Grid& x) const {
    validate_grid(x);
    Grid out(x.size(), 0.0);
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = 0; j < n_; ++j) {
            out[idx(n_, i, j)] = x[idx(n_, i, n_ - 1 - j)];
        }
    }
    return out;
}

Grid SymmetryCodec::vertical(const Grid& x) const {
    validate_grid(x);
    Grid out(x.size(), 0.0);
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = 0; j < n_; ++j) {
            out[idx(n_, i, j)] = x[idx(n_, n_ - 1 - i, j)];
        }
    }
    return out;
}

Tensor3 SymmetryCodec::encode(const Grid& x) const {
    validate_grid(x);
    Tensor3 latent;
    latent.n = n_;
    latent.layer[0] = x;
    latent.layer[1] = horizontal(x);
    latent.layer[2] = vertical(x);
    return latent;
}

Tensor3 SymmetryCodec::apply_transport(const Tensor3& latent, const Transport3& transport) const {
    if (latent.n != n_) {
        throw std::invalid_argument("latent side does not match codec side");
    }
    const auto w = transport.weights();
    Tensor3 out;
    out.n = n_;
    for (auto& layer : out.layer) {
        layer.assign(n_ * n_, 0.0);
    }
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            const double weight = w[r * 3 + c];
            for (std::size_t p = 0; p < n_ * n_; ++p) {
                out.layer[r][p] += weight * latent.layer[c][p];
            }
        }
    }
    return out;
}

Tensor3 SymmetryCodec::cyclic_shift_exact(const Tensor3& latent) const {
    if (latent.n != n_) {
        throw std::invalid_argument("latent side does not match codec side");
    }
    Tensor3 out;
    out.n = n_;
    out.layer[0] = latent.layer[2];
    out.layer[1] = latent.layer[0];
    out.layer[2] = latent.layer[1];
    return out;
}

Grid SymmetryCodec::decode_soft(const Tensor3& latent) const {
    if (latent.n != n_) {
        throw std::invalid_argument("latent side does not match codec side");
    }
    const Grid aligned0 = latent.layer[0];
    const Grid aligned1 = horizontal(latent.layer[1]);
    const Grid aligned2 = vertical(latent.layer[2]);
    Grid out(n_ * n_, 0.0);
    for (std::size_t p = 0; p < out.size(); ++p) {
        out[p] = (aligned0[p] + aligned1[p] + aligned2[p]) / 3.0;
    }
    return out;
}

Grid SymmetryCodec::decode_majority(const Tensor3& latent) const {
    if (latent.n != n_) {
        throw std::invalid_argument("latent side does not match codec side");
    }
    const Grid aligned0 = latent.layer[0];
    const Grid aligned1 = horizontal(latent.layer[1]);
    const Grid aligned2 = vertical(latent.layer[2]);
    Grid out(n_ * n_, 0.0);
    for (std::size_t p = 0; p < out.size(); ++p) {
        const double votes = aligned0[p] + aligned1[p] + aligned2[p];
        out[p] = votes >= 2.0 ? 1.0 : 0.0;
    }
    return out;
}

Grid SymmetryCodec::hard_threshold(const Grid& x, double threshold) const {
    validate_grid(x);
    Grid out(x.size(), 0.0);
    for (std::size_t p = 0; p < x.size(); ++p) {
        out[p] = x[p] >= threshold ? 1.0 : 0.0;
    }
    return out;
}

ClosedLoopOptimizer::ClosedLoopOptimizer(std::size_t n, LoopConfig config)
    : codec_(n), config_(config) {
    if (config_.coordinate_step <= 0.0 || config_.min_coordinate_step <= 0.0) {
        throw std::invalid_argument("coordinate search steps must be positive");
    }
    if (config_.min_coordinate_step > config_.coordinate_step) {
        throw std::invalid_argument("minimum coordinate step exceeds initial step");
    }
}

double ClosedLoopOptimizer::mse(const Grid& a, const Grid& b) {
    if (a.size() != b.size() || a.empty()) {
        throw std::invalid_argument("mse requires equal non-empty grids");
    }
    double sum = 0.0;
    for (std::size_t p = 0; p < a.size(); ++p) {
        const double d = a[p] - b[p];
        sum += d * d;
    }
    return sum / static_cast<double>(a.size());
}

double ClosedLoopOptimizer::tensor_mse(const Tensor3& a, const Tensor3& b) {
    if (a.n != b.n || a.n == 0) {
        throw std::invalid_argument("tensor mse requires equal non-empty tensors");
    }
    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t k = 0; k < 3; ++k) {
        if (a.layer[k].size() != b.layer[k].size()) {
            throw std::invalid_argument("tensor layer sizes differ");
        }
        for (std::size_t p = 0; p < a.layer[k].size(); ++p) {
            const double d = a.layer[k][p] - b.layer[k][p];
            sum += d * d;
            ++count;
        }
    }
    return sum / static_cast<double>(count);
}

double ClosedLoopOptimizer::entropy(const Transport3& transport) {
    const auto w = transport.weights();
    double total = 0.0;
    for (std::size_t r = 0; r < 3; ++r) {
        double row = 0.0;
        for (std::size_t c = 0; c < 3; ++c) {
            const double p = w[r * 3 + c];
            if (p > 0.0) {
                row -= p * std::log(p);
            }
        }
        total += row;
    }
    return total / 3.0;
}

LossMetrics ClosedLoopOptimizer::evaluate(const Grid& source,
                                          const Grid& reference,
                                          const Transport3& transport) const {
    const Tensor3 encoded = codec_.encode(source);
    const Tensor3 transported = codec_.apply_transport(encoded, transport);
    const Grid decoded = codec_.decode_soft(transported);
    const Tensor3 reencoded = codec_.encode(decoded);

    LossMetrics loss;
    loss.reconstruction_mse = mse(decoded, reference);
    loss.latent_cycle_mse = tensor_mse(reencoded, transported);
    loss.fixed_point_mse = mse(decoded, source);
    loss.transport_entropy = entropy(transport);
    loss.objective = loss.reconstruction_mse
        + config_.cycle_weight * loss.latent_cycle_mse
        + config_.fixed_point_weight * loss.fixed_point_mse
        + config_.entropy_weight * loss.transport_entropy;
    return loss;
}

OptimizationReport ClosedLoopOptimizer::optimize(const Grid& reference, Transport3 initial) const {
    // Keep the original frame immutable during parameter search. Otherwise an
    // early distorted recurrent state can become its own target and create a
    // false fixed point.
    OptimizationReport report;
    report.transport = initial;
    report.initial_loss = evaluate(reference, reference, report.transport);
    LossMetrics best = report.initial_loss;
    double step = config_.coordinate_step;

    for (std::size_t sweep = 0; sweep < config_.max_optimization_sweeps; ++sweep) {
        bool improved = false;
        for (std::size_t q = 0; q < report.transport.logits.size(); ++q) {
            for (double sign : {1.0, -1.0}) {
                Transport3 candidate = report.transport;
                candidate.logits[q] = clamp_logit(candidate.logits[q] + sign * step);
                const LossMetrics candidate_loss = evaluate(reference, reference, candidate);
                if (candidate_loss.objective + config_.improvement_epsilon < best.objective) {
                    report.transport = candidate;
                    best = candidate_loss;
                    improved = true;
                    ++report.accepted_moves;
                }
            }
        }
        report.sweeps = sweep + 1;
        if (!improved) {
            step *= 0.5;
            if (step < config_.min_coordinate_step) {
                break;
            }
        }
    }

    report.final_coordinate_step = step;
    report.final_loss = best;
    return report;
}

FeedbackReport ClosedLoopOptimizer::close_feedback(const Grid& seed,
                                                   const Grid& reference,
                                                   const Transport3& optimized) const {
    FeedbackReport report;
    Grid current = seed;

    for (std::size_t step = 0; step < config_.max_feedback_steps; ++step) {
        const Tensor3 encoded = codec_.encode(current);
        const Tensor3 transported = codec_.apply_transport(encoded, optimized);
        const Grid next = codec_.decode_soft(transported);
        const double delta = mse(next, current);
        current = next;
        report.steps = step + 1;
        report.fixed_point_mse = delta;
        if (delta <= config_.fixed_point_tolerance) {
            report.converged = true;
            break;
        }
    }

    report.final_state = current;
    report.final_binary_state = codec_.hard_threshold(current, config_.hard_threshold);
    report.reference_mse = mse(current, reference);
    return report;
}

ClosedLoopReport ClosedLoopOptimizer::run(const Grid& reference, Transport3 initial) const {
    ClosedLoopReport report;
    report.optimization = optimize(reference, initial);
    report.feedback = close_feedback(reference, reference, report.optimization.transport);
    return report;
}

std::string format_grid(const Grid& grid, std::size_t n, bool binary) {
    if (n == 0 || grid.size() != n * n) {
        throw std::invalid_argument("format_grid shape mismatch");
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(binary ? 0 : 6);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j != 0) {
                out << ' ';
            }
            out << grid[idx(n, i, j)];
        }
        if (i + 1 != n) {
            out << '\n';
        }
    }
    return out.str();
}

std::string format_transport(const Transport3& transport) {
    const auto w = transport.weights();
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    for (std::size_t r = 0; r < 3; ++r) {
        for (std::size_t c = 0; c < 3; ++c) {
            if (c != 0) {
                out << ' ';
            }
            out << w[r * 3 + c];
        }
        if (r != 2) {
            out << '\n';
        }
    }
    return out.str();
}

}  // namespace symmetry3d
}  // namespace jarvisx
