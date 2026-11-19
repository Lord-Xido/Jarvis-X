#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

// Jarvis-X 3D investigative planner: deterministic research simulation.
// An exact self-printing quine in default mode; bounded planning in --run mode.
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    double norm2() const { return x * x + y * y + z * z; }
    bool finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

class ChaoticExploration3D {
public:
    Vec3 lorenz{0.1, 0.0, 0.0};
    double logistic = 0.371;
    double scale = 0.04;

    Vec3 step() {
        constexpr double sigma = 10.0;
        constexpr double rho = 28.0;
        constexpr double beta = 8.0 / 3.0;
        constexpr double dt = 0.005;
        const double dx = sigma * (lorenz.y - lorenz.x);
        const double dy = lorenz.x * (rho - lorenz.z) - lorenz.y;
        const double dz = lorenz.x * lorenz.y - beta * lorenz.z;
        lorenz = lorenz + Vec3{dx, dy, dz} * dt;
        logistic = 3.92 * logistic * (1.0 - logistic);
        if (!lorenz.finite() || !std::isfinite(logistic))
            throw std::runtime_error("chaos state diverged");
        const double modulation = (logistic - 0.5) * 2.0;
        return {
            std::tanh(lorenz.x * 0.05) * scale * modulation,
            std::tanh(lorenz.y * 0.05) * scale * modulation,
            std::tanh((lorenz.z - 25.0) * 0.05) * scale * modulation
        };
    }
};

struct Plan3D {
    Vec3 action{};
    double objective = std::numeric_limits<double>::infinity();
    double zero_action_objective = std::numeric_limits<double>::infinity();
};

struct Motion3D {
    Vec3 position{};
    Vec3 velocity{};
};

class InvestigativeEngine3D {
public:
    static constexpr int D = 9; // pos3, velocity3, full target displacement3
    static constexpr int HORIZON = 12;
    static constexpr int K_PATHS = 256;
    static constexpr double GAMMA = 0.95;

    std::array<double, D> h_state{};
    std::array<std::array<double, D>, D> W_q{}, W_k{}, W_v{};
    std::array<std::array<double, D * 2>, D> W_z{}, W_r{}, W_h{};
    std::array<std::array<double, D + 3>, 3> W_dyn{};

    InvestigativeEngine3D() {
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                W_q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 0.8 : 0.05 * std::sin(i + j + 1);
                W_k[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 0.8 : 0.05 * std::cos(i - j + 1);
                W_v[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 1.0 : 0.03 * std::sin(i * 2 + j);
            }
            for (int j = 0; j < D * 2; ++j) {
                W_z[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.08 * std::cos(static_cast<double>(i) + j * 0.5);
                W_r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.08 * std::sin(i * 0.7 + j);
                W_h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (j == i) ? 0.45 : 0.06 * std::tanh(i - j);
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            for (int j = 0; j < D; ++j)
                W_dyn[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)] = 0.04 * std::sin(axis + j + 1);
            W_dyn[static_cast<std::size_t>(axis)][static_cast<std::size_t>(D + axis)] = 0.85;
        }
    }

    static double sigmoid(double x) {
        if (x >= 0.0) { const double e = std::exp(-x); return 1.0 / (1.0 + e); }
        const double e = std::exp(x);
        return e / (1.0 + e);
    }

    // Feature-wise attention-like gating, not conventional token self-attention.
    std::array<double, D> attention(const std::array<double, D>& x) const {
        std::array<double, D> q{}, k{}, v{}, scores{}, out{};
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) {
                q[static_cast<std::size_t>(i)] += W_q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
                k[static_cast<std::size_t>(i)] += W_k[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
                v[static_cast<std::size_t>(i)] += W_v[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
            }
        double max_score = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < D; ++i) {
            scores[static_cast<std::size_t>(i)] = q[static_cast<std::size_t>(i)] * k[static_cast<std::size_t>(i)] / std::sqrt(static_cast<double>(D));
            max_score = std::max(max_score, scores[static_cast<std::size_t>(i)]);
        }
        double denominator = 0.0;
        for (int i = 0; i < D; ++i) {
            scores[static_cast<std::size_t>(i)] = std::exp(scores[static_cast<std::size_t>(i)] - max_score);
            denominator += scores[static_cast<std::size_t>(i)];
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator))
            throw std::runtime_error("nonfinite feature attention");
        for (int i = 0; i < D; ++i)
            out[static_cast<std::size_t>(i)] = (scores[static_cast<std::size_t>(i)] / denominator) * v[static_cast<std::size_t>(i)] + x[static_cast<std::size_t>(i)];
        return out;
    }

    std::array<double, D> gru_step(const std::array<double, D>& x,
                                   const std::array<double, D>& h_prev) const {
        std::array<double, D * 2> concat{};
        for (int i = 0; i < D; ++i) {
            concat[static_cast<std::size_t>(i)] = h_prev[static_cast<std::size_t>(i)];
            concat[static_cast<std::size_t>(D + i)] = x[static_cast<std::size_t>(i)];
        }
        std::array<double, D> z{}, r{}, h_next{};
        for (int i = 0; i < D; ++i) {
            double sz = 0.0, sr = 0.0;
            for (int j = 0; j < D * 2; ++j) {
                sz += W_z[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * concat[static_cast<std::size_t>(j)];
                sr += W_r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * concat[static_cast<std::size_t>(j)];
            }
            z[static_cast<std::size_t>(i)] = sigmoid(sz);
            r[static_cast<std::size_t>(i)] = sigmoid(sr);
        }
        for (int i = 0; i < D; ++i) {
            double sh = 0.0;
            for (int j = 0; j < D; ++j) sh += W_h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * r[static_cast<std::size_t>(j)] * h_prev[static_cast<std::size_t>(j)];
            for (int j = 0; j < D; ++j) sh += W_h[static_cast<std::size_t>(i)][static_cast<std::size_t>(D + j)] * x[static_cast<std::size_t>(j)];
            const double h_candidate = std::tanh(sh);
            h_next[static_cast<std::size_t>(i)] = (1.0 - z[static_cast<std::size_t>(i)]) * h_prev[static_cast<std::size_t>(i)] + z[static_cast<std::size_t>(i)] * h_candidate;
        }
        return h_next;
    }

    Vec3 f_theta(const std::array<double, D>& h, const Vec3& action) const {
        std::array<double, D + 3> in{};
        for (int i = 0; i < D; ++i) in[static_cast<std::size_t>(i)] = h[static_cast<std::size_t>(i)];
        in[D] = action.x;
        in[D + 1] = action.y;
        in[D + 2] = action.z;
        Vec3 delta{};
        for (int j = 0; j < D + 3; ++j) {
            delta.x += W_dyn[0][static_cast<std::size_t>(j)] * in[static_cast<std::size_t>(j)];
            delta.y += W_dyn[1][static_cast<std::size_t>(j)] * in[static_cast<std::size_t>(j)];
            delta.z += W_dyn[2][static_cast<std::size_t>(j)] * in[static_cast<std::size_t>(j)];
        }
        return delta;
    }

    static double surprise_nll(const Vec3& pred_position, const Vec3& target) {
        constexpr double variance = 0.25;
        constexpr double two_pi = 6.283185307179586;
        const double error_squared = (pred_position - target).norm2();
        return 0.5 * (3.0 * std::log(two_pi * variance) + error_squared / variance);
    }

    static std::array<double, D> observe(const Vec3& pos, const Vec3& vel,
                                         const Vec3& target, const Vec3& exploration) {
        return {{pos.x + exploration.x, pos.y + exploration.y, pos.z + exploration.z,
                 vel.x, vel.y, vel.z,
                 target.x - pos.x, target.y - pos.y, target.z - pos.z}};
    }

    double evaluate_path(const Vec3& initial_pos, const Vec3& initial_vel,
                         const Vec3& target, const ChaoticExploration3D& initial_chaos,
                         const std::array<Vec3, HORIZON>& actions) const {
        Vec3 position = initial_pos;
        Vec3 velocity = initial_vel;
        std::array<double, D> hidden = h_state;
        ChaoticExploration3D chaos = initial_chaos; // common exploration path per candidate
        double score = 0.0;
        double discount = 1.0;
        for (int t = 0; t < HORIZON; ++t) {
            const auto features = observe(position, velocity, target, chaos.step());
            hidden = gru_step(attention(features), hidden);
            velocity = f_theta(hidden, actions[static_cast<std::size_t>(t)]);
            position = position + velocity;
            score += discount * (surprise_nll(position, target) +
                       0.02 * actions[static_cast<std::size_t>(t)].norm2());
            discount *= GAMMA;
        }
        if (!std::isfinite(score)) throw std::runtime_error("nonfinite candidate objective");
        return score;
    }

    Plan3D plan_one_shot(const Vec3& pos, const Vec3& vel, const Vec3& target,
                         const ChaoticExploration3D& chaos) const {
        if (!pos.finite() || !vel.finite() || !target.finite())
            throw std::invalid_argument("planning input is not finite");
        std::mt19937 rng(42);
        std::normal_distribution<double> normal(0.0, 0.35);
        Plan3D best;
        for (int k = 0; k < K_PATHS; ++k) {
            std::array<Vec3, HORIZON> actions{};
            if (k != 0) {
                for (auto& action : actions) {
                    action = {std::clamp(normal(rng), -1.0, 1.0),
                              std::clamp(normal(rng), -1.0, 1.0),
                              std::clamp(normal(rng), -1.0, 1.0)};
                }
            }
            const double score = evaluate_path(pos, vel, target, chaos, actions);
            if (k == 0) best.zero_action_objective = score;
            if (score < best.objective) {
                best.objective = score;
                best.action = actions[0];
            }
        }
        return best; // does not mutate h_state or the caller's chaos state
    }

    Motion3D apply_first_action(const Motion3D& current, const Vec3& target,
                                 const Vec3& action, ChaoticExploration3D& chaos) {
        if (!current.position.finite() || !current.velocity.finite() ||
            !target.finite() || !action.finite())
            throw std::invalid_argument("nonfinite transition input");
        const auto features = observe(current.position, current.velocity, target, chaos.step());
        h_state = gru_step(attention(features), h_state);
        const Vec3 new_velocity = f_theta(h_state, action);
        const Vec3 new_position = current.position + new_velocity;
        if (!new_position.finite() || !new_velocity.finite())
            throw std::runtime_error("nonfinite transition");
        return {new_position, new_velocity};
    }
};

static bool nearly_equal(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

static void self_test() {
    InvestigativeEngine3D engine;
    ChaoticExploration3D chaos;
    const Vec3 pos{0.0, 0.0, 0.0};
    const Vec3 vel{0.0, 0.0, 0.0};
    const Vec3 target{2.5, -1.5, 3.0};
    const auto initial_h = engine.h_state;
    const Vec3 initial_lorenz = chaos.lorenz;
    const double initial_logistic = chaos.logistic;
    const Plan3D plan = engine.plan_one_shot(pos, vel, target, chaos);
    if (engine.h_state != initial_h || !nearly_equal(chaos.logistic, initial_logistic) ||
        !(chaos.lorenz.x == initial_lorenz.x && chaos.lorenz.y == initial_lorenz.y &&
          chaos.lorenz.z == initial_lorenz.z))
        throw std::runtime_error("planner mutated committed state");
    if (!plan.action.finite() || !std::isfinite(plan.objective) ||
        plan.objective > plan.zero_action_objective + 1.0e-12)
        throw std::runtime_error("sampled planner failed baseline contract");
    const Plan3D repeat = engine.plan_one_shot(pos, vel, target, chaos);
    if (!(repeat.objective == plan.objective && repeat.action.x == plan.action.x &&
          repeat.action.y == plan.action.y && repeat.action.z == plan.action.z))
        throw std::runtime_error("planner is not deterministic");
    std::array<double, InvestigativeEngine3D::D> probe = engine.observe(pos, vel, target, {});
    if (!nearly_equal(probe[8], 3.0))
        throw std::runtime_error("z-target displacement feature is missing");
    const auto alternate = engine.observe(pos, vel, {2.5, -1.5, 30.0}, {});
    if (alternate[8] == probe[8])
        throw std::runtime_error("z target is not observable");
    Motion3D first = engine.apply_first_action({pos, vel}, target, plan.action, chaos);
    if (!first.position.finite() || !first.velocity.finite() ||
        engine.h_state == initial_h || nearly_equal(chaos.logistic, initial_logistic))
        throw std::runtime_error("transition did not commit exactly one state step");
    bool rejected = false;
    try {
        const double inf = std::numeric_limits<double>::infinity();
        (void)engine.plan_one_shot({inf, 0.0, 0.0}, vel, target, chaos);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("nonfinite planning input accepted");
    std::cout << std::setprecision(12)
              << "investigative3d_self_test=PASS baseline=" << plan.zero_action_objective
              << " selected=" << plan.objective << "\n";
}

const char* SRC = R"JXINV(#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

// Jarvis-X 3D investigative planner: deterministic research simulation.
// An exact self-printing quine in default mode; bounded planning in --run mode.
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    double norm2() const { return x * x + y * y + z * z; }
    bool finite() const { return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }
};

class ChaoticExploration3D {
public:
    Vec3 lorenz{0.1, 0.0, 0.0};
    double logistic = 0.371;
    double scale = 0.04;

    Vec3 step() {
        constexpr double sigma = 10.0;
        constexpr double rho = 28.0;
        constexpr double beta = 8.0 / 3.0;
        constexpr double dt = 0.005;
        const double dx = sigma * (lorenz.y - lorenz.x);
        const double dy = lorenz.x * (rho - lorenz.z) - lorenz.y;
        const double dz = lorenz.x * lorenz.y - beta * lorenz.z;
        lorenz = lorenz + Vec3{dx, dy, dz} * dt;
        logistic = 3.92 * logistic * (1.0 - logistic);
        if (!lorenz.finite() || !std::isfinite(logistic))
            throw std::runtime_error("chaos state diverged");
        const double modulation = (logistic - 0.5) * 2.0;
        return {
            std::tanh(lorenz.x * 0.05) * scale * modulation,
            std::tanh(lorenz.y * 0.05) * scale * modulation,
            std::tanh((lorenz.z - 25.0) * 0.05) * scale * modulation
        };
    }
};

struct Plan3D {
    Vec3 action{};
    double objective = std::numeric_limits<double>::infinity();
    double zero_action_objective = std::numeric_limits<double>::infinity();
};

struct Motion3D {
    Vec3 position{};
    Vec3 velocity{};
};

class InvestigativeEngine3D {
public:
    static constexpr int D = 9; // pos3, velocity3, full target displacement3
    static constexpr int HORIZON = 12;
    static constexpr int K_PATHS = 256;
    static constexpr double GAMMA = 0.95;

    std::array<double, D> h_state{};
    std::array<std::array<double, D>, D> W_q{}, W_k{}, W_v{};
    std::array<std::array<double, D * 2>, D> W_z{}, W_r{}, W_h{};
    std::array<std::array<double, D + 3>, 3> W_dyn{};

    InvestigativeEngine3D() {
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                W_q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 0.8 : 0.05 * std::sin(i + j + 1);
                W_k[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 0.8 : 0.05 * std::cos(i - j + 1);
                W_v[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (i == j) ? 1.0 : 0.03 * std::sin(i * 2 + j);
            }
            for (int j = 0; j < D * 2; ++j) {
                W_z[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.08 * std::cos(static_cast<double>(i) + j * 0.5);
                W_r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.08 * std::sin(i * 0.7 + j);
                W_h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = (j == i) ? 0.45 : 0.06 * std::tanh(i - j);
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            for (int j = 0; j < D; ++j)
                W_dyn[static_cast<std::size_t>(axis)][static_cast<std::size_t>(j)] = 0.04 * std::sin(axis + j + 1);
            W_dyn[static_cast<std::size_t>(axis)][static_cast<std::size_t>(D + axis)] = 0.85;
        }
    }

    static double sigmoid(double x) {
        if (x >= 0.0) { const double e = std::exp(-x); return 1.0 / (1.0 + e); }
        const double e = std::exp(x);
        return e / (1.0 + e);
    }

    // Feature-wise attention-like gating, not conventional token self-attention.
    std::array<double, D> attention(const std::array<double, D>& x) const {
        std::array<double, D> q{}, k{}, v{}, scores{}, out{};
        for (int i = 0; i < D; ++i)
            for (int j = 0; j < D; ++j) {
                q[static_cast<std::size_t>(i)] += W_q[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
                k[static_cast<std::size_t>(i)] += W_k[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
                v[static_cast<std::size_t>(i)] += W_v[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
            }
        double max_score = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < D; ++i) {
            scores[static_cast<std::size_t>(i)] = q[static_cast<std::size_t>(i)] * k[static_cast<std::size_t>(i)] / std::sqrt(static_cast<double>(D));
            max_score = std::max(max_score, scores[static_cast<std::size_t>(i)]);
        }
        double denominator = 0.0;
        for (int i = 0; i < D; ++i) {
            scores[static_cast<std::size_t>(i)] = std::exp(scores[static_cast<std::size_t>(i)] - max_score);
            denominator += scores[static_cast<std::size_t>(i)];
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator))
            throw std::runtime_error("nonfinite feature attention");
        for (int i = 0; i < D; ++i)
            out[static_cast<std::size_t>(i)] = (scores[static_cast<std::size_t>(i)] / denominator) * v[static_cast<std::size_t>(i)] + x[static_cast<std::size_t>(i)];
        return out;
    }

    std::array<double, D> gru_step(const std::array<double, D>& x,
                                   const std::array<double, D>& h_prev) const {
        std::array<double, D * 2> concat{};
        for (int i = 0; i < D; ++i) {
            concat[static_cast<std::size_t>(i)] = h_prev[static_cast<std::size_t>(i)];
            concat[static_cast<std::size_t>(D + i)] = x[static_cast<std::size_t>(i)];
        }
        std::array<double, D> z{}, r{}, h_next{};
        for (int i = 0; i < D; ++i) {
            double sz = 0.0, sr = 0.0;
            for (int j = 0; j < D * 2; ++j) {
                sz += W_z[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * concat[static_cast<std::size_t>(j)];
                sr += W_r[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * concat[static_cast<std::size_t>(j)];
            }
            z[static_cast<std::size_t>(i)] = sigmoid(sz);
            r[static_cast<std::size_t>(i)] = sigmoid(sr);
        }
        for (int i = 0; i < D; ++i) {
            double sh = 0.0;
            for (int j = 0; j < D; ++j) sh += W_h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * r[static_cast<std::size_t>(j)] * h_prev[static_cast<std::size_t>(j)];
            for (int j = 0; j < D; ++j) sh += W_h[static_cast<std::size_t>(i)][static_cast<std::size_t>(D + j)] * x[static_cast<std::size_t>(j)];
            const double h_candidate = std::tanh(sh);
            h_next[static_cast<std::size_t>(i)] = (1.0 - z[static_cast<std::size_t>(i)]) * h_prev[static_cast<std::size_t>(i)] + z[static_cast<std::size_t>(i)] * h_candidate;
        }
        return h_next;
    }

    Vec3 f_theta(const std::array<double, D>& h, const Vec3& action) const {
        std::array<double, D + 3> in{};
        for (int i = 0; i < D; ++i) in[static_cast<std::size_t>(i)] = h[static_cast<std::size_t>(i)];
        in[D] = action.x;
        in[D + 1] = action.y;
        in[D + 2] = action.z;
        Vec3 delta{};
        for (int j = 0; j < D + 3; ++j) {
            delta.x += W_dyn[0][static_cast<std::size_t>(j)] * in[static_cast<std::size_t>(j)];
            delta.y += W_dyn[1][static_cast<std::size_t>(j)] * in[static_cast<std::size_t>(j)];
            delta.z += W_dyn[2][static_cast<std::size_t>(j)] * in[static_cast<std::size_t>(j)];
        }
        return delta;
    }

    static double surprise_nll(const Vec3& pred_position, const Vec3& target) {
        constexpr double variance = 0.25;
        constexpr double two_pi = 6.283185307179586;
        const double error_squared = (pred_position - target).norm2();
        return 0.5 * (3.0 * std::log(two_pi * variance) + error_squared / variance);
    }

    static std::array<double, D> observe(const Vec3& pos, const Vec3& vel,
                                         const Vec3& target, const Vec3& exploration) {
        return {{pos.x + exploration.x, pos.y + exploration.y, pos.z + exploration.z,
                 vel.x, vel.y, vel.z,
                 target.x - pos.x, target.y - pos.y, target.z - pos.z}};
    }

    double evaluate_path(const Vec3& initial_pos, const Vec3& initial_vel,
                         const Vec3& target, const ChaoticExploration3D& initial_chaos,
                         const std::array<Vec3, HORIZON>& actions) const {
        Vec3 position = initial_pos;
        Vec3 velocity = initial_vel;
        std::array<double, D> hidden = h_state;
        ChaoticExploration3D chaos = initial_chaos; // common exploration path per candidate
        double score = 0.0;
        double discount = 1.0;
        for (int t = 0; t < HORIZON; ++t) {
            const auto features = observe(position, velocity, target, chaos.step());
            hidden = gru_step(attention(features), hidden);
            velocity = f_theta(hidden, actions[static_cast<std::size_t>(t)]);
            position = position + velocity;
            score += discount * (surprise_nll(position, target) +
                       0.02 * actions[static_cast<std::size_t>(t)].norm2());
            discount *= GAMMA;
        }
        if (!std::isfinite(score)) throw std::runtime_error("nonfinite candidate objective");
        return score;
    }

    Plan3D plan_one_shot(const Vec3& pos, const Vec3& vel, const Vec3& target,
                         const ChaoticExploration3D& chaos) const {
        if (!pos.finite() || !vel.finite() || !target.finite())
            throw std::invalid_argument("planning input is not finite");
        std::mt19937 rng(42);
        std::normal_distribution<double> normal(0.0, 0.35);
        Plan3D best;
        for (int k = 0; k < K_PATHS; ++k) {
            std::array<Vec3, HORIZON> actions{};
            if (k != 0) {
                for (auto& action : actions) {
                    action = {std::clamp(normal(rng), -1.0, 1.0),
                              std::clamp(normal(rng), -1.0, 1.0),
                              std::clamp(normal(rng), -1.0, 1.0)};
                }
            }
            const double score = evaluate_path(pos, vel, target, chaos, actions);
            if (k == 0) best.zero_action_objective = score;
            if (score < best.objective) {
                best.objective = score;
                best.action = actions[0];
            }
        }
        return best; // does not mutate h_state or the caller's chaos state
    }

    Motion3D apply_first_action(const Motion3D& current, const Vec3& target,
                                 const Vec3& action, ChaoticExploration3D& chaos) {
        if (!current.position.finite() || !current.velocity.finite() ||
            !target.finite() || !action.finite())
            throw std::invalid_argument("nonfinite transition input");
        const auto features = observe(current.position, current.velocity, target, chaos.step());
        h_state = gru_step(attention(features), h_state);
        const Vec3 new_velocity = f_theta(h_state, action);
        const Vec3 new_position = current.position + new_velocity;
        if (!new_position.finite() || !new_velocity.finite())
            throw std::runtime_error("nonfinite transition");
        return {new_position, new_velocity};
    }
};

static bool nearly_equal(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

static void self_test() {
    InvestigativeEngine3D engine;
    ChaoticExploration3D chaos;
    const Vec3 pos{0.0, 0.0, 0.0};
    const Vec3 vel{0.0, 0.0, 0.0};
    const Vec3 target{2.5, -1.5, 3.0};
    const auto initial_h = engine.h_state;
    const Vec3 initial_lorenz = chaos.lorenz;
    const double initial_logistic = chaos.logistic;
    const Plan3D plan = engine.plan_one_shot(pos, vel, target, chaos);
    if (engine.h_state != initial_h || !nearly_equal(chaos.logistic, initial_logistic) ||
        !(chaos.lorenz.x == initial_lorenz.x && chaos.lorenz.y == initial_lorenz.y &&
          chaos.lorenz.z == initial_lorenz.z))
        throw std::runtime_error("planner mutated committed state");
    if (!plan.action.finite() || !std::isfinite(plan.objective) ||
        plan.objective > plan.zero_action_objective + 1.0e-12)
        throw std::runtime_error("sampled planner failed baseline contract");
    const Plan3D repeat = engine.plan_one_shot(pos, vel, target, chaos);
    if (!(repeat.objective == plan.objective && repeat.action.x == plan.action.x &&
          repeat.action.y == plan.action.y && repeat.action.z == plan.action.z))
        throw std::runtime_error("planner is not deterministic");
    std::array<double, InvestigativeEngine3D::D> probe = engine.observe(pos, vel, target, {});
    if (!nearly_equal(probe[8], 3.0))
        throw std::runtime_error("z-target displacement feature is missing");
    const auto alternate = engine.observe(pos, vel, {2.5, -1.5, 30.0}, {});
    if (alternate[8] == probe[8])
        throw std::runtime_error("z target is not observable");
    Motion3D first = engine.apply_first_action({pos, vel}, target, plan.action, chaos);
    if (!first.position.finite() || !first.velocity.finite() ||
        engine.h_state == initial_h || nearly_equal(chaos.logistic, initial_logistic))
        throw std::runtime_error("transition did not commit exactly one state step");
    bool rejected = false;
    try {
        const double inf = std::numeric_limits<double>::infinity();
        (void)engine.plan_one_shot({inf, 0.0, 0.0}, vel, target, chaos);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("nonfinite planning input accepted");
    std::cout << std::setprecision(12)
              << "investigative3d_self_test=PASS baseline=" << plan.zero_action_objective
              << " selected=" << plan.objective << "\n";
}

const char* SRC = R%cJXINV(%s)JXINV%c;
int main(int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--print-source")) {
            if (std::printf(SRC, 34, SRC, 34) < 0)
                throw std::runtime_error("unable to print quine");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: jarvisx-investigative3d [--print-source|--self-test|--run|--steps N]\n";
            return 0;
        }
        int steps = 1;
        if (argc == 2 && std::string(argv[1]) == "--run") {
            steps = 1;
        } else if (argc == 3 && std::string(argv[1]) == "--steps") {
            const std::string arg = argv[2];
            if (arg.empty() || arg[0] == '-' || arg[0] == '+')
                throw std::invalid_argument("invalid positive step count");
            std::size_t consumed = 0;
            const unsigned long parsed = std::stoul(arg, &consumed, 10);
            if (consumed != arg.size() || parsed < 1UL || parsed > 50UL)
                throw std::invalid_argument("steps must be 1..50");
            steps = static_cast<int>(parsed);
        } else {
            throw std::invalid_argument("unknown command or argument");
        }
        ChaoticExploration3D chaos;
        InvestigativeEngine3D engine;
        Motion3D motion{};
        const Vec3 target{2.5, -1.5, 3.0};
        std::cout << std::setprecision(10);
        for (int i = 0; i < steps; ++i) {
            const Plan3D plan = engine.plan_one_shot(motion.position, motion.velocity, target, chaos);
            motion = engine.apply_first_action(motion, target, plan.action, chaos);
            std::cout << "step=" << i + 1
                      << " a_t_star=(" << plan.action.x << "," << plan.action.y << "," << plan.action.z << ")"
                      << " next_position=(" << motion.position.x << "," << motion.position.y << "," << motion.position.z << ")"
                      << " target_distance=" << std::sqrt((motion.position - target).norm2())
                      << " planned_objective=" << plan.objective
                      << " zero_control_objective=" << plan.zero_action_objective << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "investigative3d failure: " << e.what() << '\n';
        return 1;
    }
}
)JXINV";
int main(int argc, char** argv) {
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--print-source")) {
            if (std::printf(SRC, 34, SRC, 34) < 0)
                throw std::runtime_error("unable to print quine");
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: jarvisx-investigative3d [--print-source|--self-test|--run|--steps N]\n";
            return 0;
        }
        int steps = 1;
        if (argc == 2 && std::string(argv[1]) == "--run") {
            steps = 1;
        } else if (argc == 3 && std::string(argv[1]) == "--steps") {
            const std::string arg = argv[2];
            if (arg.empty() || arg[0] == '-' || arg[0] == '+')
                throw std::invalid_argument("invalid positive step count");
            std::size_t consumed = 0;
            const unsigned long parsed = std::stoul(arg, &consumed, 10);
            if (consumed != arg.size() || parsed < 1UL || parsed > 50UL)
                throw std::invalid_argument("steps must be 1..50");
            steps = static_cast<int>(parsed);
        } else {
            throw std::invalid_argument("unknown command or argument");
        }
        ChaoticExploration3D chaos;
        InvestigativeEngine3D engine;
        Motion3D motion{};
        const Vec3 target{2.5, -1.5, 3.0};
        std::cout << std::setprecision(10);
        for (int i = 0; i < steps; ++i) {
            const Plan3D plan = engine.plan_one_shot(motion.position, motion.velocity, target, chaos);
            motion = engine.apply_first_action(motion, target, plan.action, chaos);
            std::cout << "step=" << i + 1
                      << " a_t_star=(" << plan.action.x << "," << plan.action.y << "," << plan.action.z << ")"
                      << " next_position=(" << motion.position.x << "," << motion.position.y << "," << motion.position.z << ")"
                      << " target_distance=" << std::sqrt((motion.position - target).norm2())
                      << " planned_objective=" << plan.objective
                      << " zero_control_objective=" << plan.zero_action_objective << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "investigative3d failure: " << e.what() << '\n';
        return 1;
    }
}
