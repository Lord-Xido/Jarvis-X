#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jarvisx {

struct KineticVec3 {
    float x{};
    float y{};
    float z{};

    KineticVec3 operator+(const KineticVec3& other) const noexcept {
        return {x + other.x, y + other.y, z + other.z};
    }
    KineticVec3 operator-(const KineticVec3& other) const noexcept {
        return {x - other.x, y - other.y, z - other.z};
    }
    KineticVec3 operator*(float scalar) const noexcept {
        return {x * scalar, y * scalar, z * scalar};
    }
    KineticVec3& operator+=(const KineticVec3& other) noexcept {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    float norm_sq() const noexcept { return x * x + y * y + z * z; }
    float norm() const noexcept { return std::sqrt(norm_sq()); }
    KineticVec3 normalized() const noexcept {
        const float n = norm();
        return n > 1.0e-8F ? (*this) * (1.0F / n) : KineticVec3{};
    }
};

using KineticNodeId = std::uint64_t;

enum class KineticOperationType : std::uint8_t {
    Encode,
    Decode,
    Physics,
    AI,
    Echo,
    Learn,
    Swarm,
    Optimize,
    Spawn,
    Prune
};

enum class KineticScope : std::uint8_t {
    Direct,
    Global
};

struct KineticOperation {
    KineticOperationType type{KineticOperationType::Encode};
    KineticScope scope{KineticScope::Global};
    KineticNodeId target{};
    float strength{1.0F};
    std::uint16_t ttl{8U};
};

struct KineticSynapse {
    KineticNodeId target{};
    float weight{0.5F};
    float rest_length{1.0F};
    float plasticity{0.1F};
};

struct KineticNode {
    KineticNodeId id{};
    KineticVec3 anchor_position{};
    KineticVec3 position{};
    KineticVec3 velocity{};
    KineticVec3 normal{0.0F, 0.0F, 1.0F};
    float curvature{0.1F};
    float spectral_weight{1.0F};
    float activation{0.0F};
    float potential{0.0F};
    float threshold{0.5F};
    float decay{0.99F};
    float residual_error{0.0F};
    float fitness{0.5F};
    float energy{1.0F};
    std::vector<KineticSynapse> synapses;
};

struct KineticConfig {
    float dt{0.016F};
    float gravity{-9.8F};
    float velocity_damping{0.98F};
    float echo_damping{0.5F};
    float echo_epsilon{1.0e-4F};
    float learning_rate{0.01F};
    float max_speed{50.0F};
    float max_displacement{1.0F};
    float max_abs_position{1.0e4F};
    std::size_t max_nodes{200000U};
    std::size_t max_events_per_step{100000U};

    void validate() const {
        if (!std::isfinite(dt) || dt <= 0.0F || dt > 1.0F) {
            throw std::invalid_argument("kinetic dt must be in (0, 1]");
        }
        if (!std::isfinite(velocity_damping) || velocity_damping < 0.0F ||
            velocity_damping > 1.0F) {
            throw std::invalid_argument("velocity damping must be in [0, 1]");
        }
        if (!std::isfinite(echo_damping) || echo_damping < 0.0F ||
            echo_damping > 1.0F) {
            throw std::invalid_argument("echo damping must be in [0, 1]");
        }
        if (!std::isfinite(max_speed) || max_speed <= 0.0F ||
            !std::isfinite(max_displacement) || max_displacement <= 0.0F ||
            !std::isfinite(max_abs_position) || max_abs_position <= 0.0F) {
            throw std::invalid_argument("kinetic bounds must be finite and positive");
        }
        if (max_nodes == 0U || max_events_per_step == 0U) {
            throw std::invalid_argument("kinetic resource bounds must be non-zero");
        }
    }
};

struct KineticTelemetry {
    std::uint64_t cycle{};
    std::size_t nodes_before{};
    std::size_t nodes_after{};
    std::size_t operations_consumed{};
    std::size_t propagated_events{};
    std::size_t spawned_nodes{};
    std::size_t pruned_synapses{};
    float mean_activation{};
    float mean_residual{};
    float max_speed{};
    float max_displacement{};
    bool committed{};
    const char* rejection_reason{""};
};

class KineticSystemLoop {
public:
    explicit KineticSystemLoop(KineticConfig config = {}) : config_(config) {
        config_.validate();
    }

    KineticNodeId add_node(KineticNode node) {
        if (nodes_.size() >= config_.max_nodes) {
            throw std::runtime_error("kinetic node budget exceeded");
        }
        node.id = next_id_++;
        node.anchor_position = node.position;
        sanitize_node(node);
        if (std::fabs(node.position.x) > config_.max_abs_position ||
            std::fabs(node.position.y) > config_.max_abs_position ||
            std::fabs(node.position.z) > config_.max_abs_position) {
            throw std::invalid_argument("kinetic node outside position bounds");
        }
        nodes_.push_back(std::move(node));
        rebuild_index();
        return nodes_.back().id;
    }

    void connect(KineticNodeId source, KineticNodeId target, float weight,
                 float rest_length = 1.0F, float plasticity = 0.1F) {
        auto& node = nodes_.at(index_of(source));
        (void)index_of(target);
        if (!std::isfinite(weight) || !std::isfinite(rest_length) ||
            rest_length < 0.0F || !std::isfinite(plasticity) || plasticity < 0.0F) {
            throw std::invalid_argument("invalid kinetic synapse");
        }
        node.synapses.push_back({target, clamp(weight, -1.0F, 1.0F),
                                 rest_length, clamp(plasticity, 0.0F, 1.0F)});
    }

    void enqueue(KineticOperation operation) {
        if (!std::isfinite(operation.strength)) {
            throw std::invalid_argument("operation strength must be finite");
        }
        if (operation.type == KineticOperationType::Spawn &&
            operation.scope != KineticScope::Direct) {
            throw std::invalid_argument("spawn requires direct scope");
        }
        if (operation.scope == KineticScope::Direct) {
            (void)index_of(operation.target);
        }
        pending_.push_back(operation);
    }

    const std::vector<KineticNode>& nodes() const noexcept { return nodes_; }
    const KineticTelemetry& telemetry() const noexcept { return telemetry_; }

    const KineticNode& node(KineticNodeId id) const { return nodes_.at(index_of(id)); }

    bool step() {
        telemetry_ = {};
        telemetry_.cycle = ++cycle_;
        telemetry_.nodes_before = nodes_.size();

        const std::vector<KineticNode> snapshot = nodes_;
        const auto snapshot_index = make_index(snapshot);
        std::vector<KineticNode> candidate = snapshot;
        std::vector<KineticVec3> forces(candidate.size());
        std::vector<SpawnRequest> spawn_requests;
        KineticNodeId next_id_candidate = next_id_;

        std::deque<KineticOperation> work;
        work.swap(pending_);

        while (!work.empty()) {
            if (telemetry_.operations_consumed >= config_.max_events_per_step) {
                return reject("event budget exceeded");
            }
            const KineticOperation operation = work.front();
            work.pop_front();
            ++telemetry_.operations_consumed;

            if (operation.scope == KineticScope::Global) {
                for (std::size_t i = 0; i < snapshot.size(); ++i) {
                    apply_operation(operation, i, snapshot, snapshot_index,
                                    candidate, forces, work, spawn_requests);
                    if (telemetry_.operations_consumed + work.size() >
                        config_.max_events_per_step) {
                        return reject("propagation budget exceeded");
                    }
                }
            } else {
                const auto found = snapshot_index.find(operation.target);
                if (found == snapshot_index.end()) {
                    return reject("direct target missing from snapshot");
                }
                apply_operation(operation, found->second, snapshot, snapshot_index,
                                candidate, forces, work, spawn_requests);
            }
        }

        integrate(snapshot, candidate, forces);
        if (!validate_candidate(candidate)) {
            return reject("candidate projection failed");
        }

        for (const SpawnRequest& request : spawn_requests) {
            if (candidate.size() >= config_.max_nodes) {
                return reject("spawn would exceed node budget");
            }
            const auto candidate_index = make_index(candidate);
            const auto found = candidate_index.find(request.parent);
            if (found == candidate_index.end()) {
                return reject("spawn parent missing");
            }
            const KineticNode parent = candidate[found->second];
            for (std::size_t ordinal = 0; ordinal < request.count; ++ordinal) {
                if (candidate.size() >= config_.max_nodes) {
                    return reject("spawn would exceed node budget");
                }
                KineticNode child = parent;
                child.id = next_id_candidate++;
                const float sign = (ordinal % 2U == 0U) ? 1.0F : -1.0F;
                const float scale = 0.05F * static_cast<float>(1U + ordinal);
                child.position += child.normal.normalized() * (sign * scale);
                child.anchor_position = child.position;
                child.velocity = {};
                child.activation = 0.5F;
                child.potential = 0.0F;
                child.synapses.clear();
                candidate.push_back(std::move(child));
                ++telemetry_.spawned_nodes;
            }
        }

        if (!validate_candidate(candidate)) {
            return reject("post-topology projection failed");
        }

        nodes_ = std::move(candidate);
        next_id_ = next_id_candidate;
        rebuild_index();
        telemetry_.nodes_after = nodes_.size();
        finalize_telemetry();
        telemetry_.committed = true;
        telemetry_.rejection_reason = "";
        return true;
    }

private:
    struct SpawnRequest {
        KineticNodeId parent{};
        std::size_t count{};
    };

    KineticConfig config_;
    std::vector<KineticNode> nodes_;
    std::unordered_map<KineticNodeId, std::size_t> index_;
    std::deque<KineticOperation> pending_;
    KineticNodeId next_id_{1U};
    std::uint64_t cycle_{};
    KineticTelemetry telemetry_{};

    static float clamp(float value, float low, float high) noexcept {
        return std::max(low, std::min(value, high));
    }

    static float sigmoid(float x) noexcept {
        if (x >= 0.0F) {
            const float e = std::exp(-x);
            return 1.0F / (1.0F + e);
        }
        const float e = std::exp(x);
        return e / (1.0F + e);
    }

    static bool finite_vec(const KineticVec3& v) noexcept {
        return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
    }

    static std::unordered_map<KineticNodeId, std::size_t>
    make_index(const std::vector<KineticNode>& nodes) {
        std::unordered_map<KineticNodeId, std::size_t> result;
        result.reserve(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            result.emplace(nodes[i].id, i);
        }
        return result;
    }

    void rebuild_index() { index_ = make_index(nodes_); }

    std::size_t index_of(KineticNodeId id) const {
        const auto found = index_.find(id);
        if (found == index_.end()) throw std::out_of_range("unknown kinetic node id");
        return found->second;
    }

    static void sanitize_node(KineticNode& node) {
        if (!finite_vec(node.position) || !finite_vec(node.velocity) ||
            !finite_vec(node.normal)) {
            throw std::invalid_argument("non-finite kinetic vector state");
        }
        node.normal = node.normal.normalized();
        if (node.normal.norm_sq() < 1.0e-8F) node.normal = {0.0F, 0.0F, 1.0F};
        node.curvature = clamp(node.curvature, 0.0F, 10.0F);
        node.spectral_weight = clamp(node.spectral_weight, 0.01F, 10.0F);
        node.activation = clamp(node.activation, 0.0F, 1.0F);
        node.threshold = clamp(node.threshold, 0.01F, 0.99F);
        node.decay = clamp(node.decay, 0.0F, 1.0F);
        node.fitness = clamp(node.fitness, 0.0F, 1.0F);
        node.energy = clamp(node.energy, 0.0F, 1000.0F);
    }

    void apply_operation(
        const KineticOperation& operation,
        std::size_t i,
        const std::vector<KineticNode>& snapshot,
        const std::unordered_map<KineticNodeId, std::size_t>& snapshot_index,
        std::vector<KineticNode>& candidate,
        std::vector<KineticVec3>& forces,
        std::deque<KineticOperation>& work,
        std::vector<SpawnRequest>& spawn_requests) {

        const KineticNode& source = snapshot[i];
        KineticNode& target = candidate[i];
        const float strength = operation.strength;

        switch (operation.type) {
            case KineticOperationType::Encode: {
                const float latent = std::tanh(
                    0.50F * source.activation + 0.25F * source.potential +
                    0.15F * source.curvature + 0.10F * source.spectral_weight);
                const float decoded = sigmoid(
                    source.spectral_weight * latent - source.threshold);
                target.residual_error = source.activation - decoded;
                target.potential -= strength * target.residual_error * 0.05F;
                break;
            }
            case KineticOperationType::Decode: {
                forces[i] += source.normal.normalized() *
                    (strength * source.activation * source.spectral_weight);
                break;
            }
            case KineticOperationType::Physics: {
                forces[i].y += config_.gravity * strength;
                for (const KineticSynapse& synapse : source.synapses) {
                    const auto found = snapshot_index.find(synapse.target);
                    if (found == snapshot_index.end()) continue;
                    const KineticVec3 diff = snapshot[found->second].position - source.position;
                    const float distance = diff.norm();
                    if (distance <= 1.0e-8F) continue;
                    const float extension = distance - synapse.rest_length;
                    forces[i] += diff * ((0.1F * synapse.weight * extension) / distance);
                }
                break;
            }
            case KineticOperationType::AI: {
                float prediction = 0.0F;
                float normalizer = 0.0F;
                for (const KineticSynapse& synapse : source.synapses) {
                    const auto found = snapshot_index.find(synapse.target);
                    if (found == snapshot_index.end()) continue;
                    prediction += synapse.weight * snapshot[found->second].activation;
                    normalizer += std::fabs(synapse.weight);
                }
                if (normalizer > 1.0e-8F) prediction /= normalizer;
                const float error = source.activation - prediction;
                target.potential -= strength * error * 0.05F;
                target.residual_error = 0.5F * target.residual_error + 0.5F * error;
                break;
            }
            case KineticOperationType::Echo: {
                target.potential += strength * 0.1F;
                if (operation.ttl > 0U && std::fabs(strength) >= config_.echo_epsilon) {
                    for (const KineticSynapse& synapse : source.synapses) {
                        const float next_strength = strength * synapse.weight *
                                                    config_.echo_damping;
                        if (std::fabs(next_strength) < config_.echo_epsilon) continue;
                        work.push_back({KineticOperationType::Echo,
                                        KineticScope::Direct,
                                        synapse.target,
                                        next_strength,
                                        static_cast<std::uint16_t>(operation.ttl - 1U)});
                        ++telemetry_.propagated_events;
                    }
                }
                break;
            }
            case KineticOperationType::Learn: {
                for (KineticSynapse& synapse : target.synapses) {
                    const auto found = snapshot_index.find(synapse.target);
                    if (found == snapshot_index.end()) continue;
                    const float target_activation = snapshot[found->second].activation;
                    const float dw = config_.learning_rate * strength *
                        (source.activation * target_activation - 0.01F * synapse.weight);
                    synapse.weight = clamp(synapse.weight + dw, -1.0F, 1.0F);
                }
                break;
            }
            case KineticOperationType::Swarm: {
                KineticVec3 center{};
                std::size_t count = 0U;
                for (const KineticSynapse& synapse : source.synapses) {
                    const auto found = snapshot_index.find(synapse.target);
                    if (found == snapshot_index.end()) continue;
                    center += snapshot[found->second].position;
                    ++count;
                }
                if (count > 0U) {
                    center = center * (1.0F / static_cast<float>(count));
                    forces[i] += (center - source.position) * (0.05F * strength);
                }
                break;
            }
            case KineticOperationType::Optimize: {
                target.threshold = clamp(
                    source.threshold - config_.learning_rate * strength *
                    source.residual_error,
                    0.01F, 0.99F);
                target.spectral_weight = clamp(
                    source.spectral_weight - 0.1F * config_.learning_rate * strength *
                    source.residual_error,
                    0.01F, 10.0F);
                break;
            }
            case KineticOperationType::Spawn: {
                if (operation.scope != KineticScope::Direct) {
                    throw std::invalid_argument("spawn requires direct scope");
                }
                const float requested = std::max(0.0F, strength);
                const std::size_t count = static_cast<std::size_t>(std::floor(requested));
                if (count > 0U) spawn_requests.push_back({source.id, count});
                break;
            }
            case KineticOperationType::Prune: {
                const float threshold = std::fabs(strength);
                const auto before = target.synapses.size();
                target.synapses.erase(
                    std::remove_if(target.synapses.begin(), target.synapses.end(),
                        [threshold](const KineticSynapse& synapse) {
                            return std::fabs(synapse.weight) < threshold;
                        }),
                    target.synapses.end());
                telemetry_.pruned_synapses += before - target.synapses.size();
                break;
            }
        }
    }

    void integrate(const std::vector<KineticNode>& snapshot,
                   std::vector<KineticNode>& candidate,
                   const std::vector<KineticVec3>& forces) {
        for (std::size_t i = 0; i < candidate.size(); ++i) {
            KineticNode& node = candidate[i];
            const KineticNode& old = snapshot[i];
            node.potential *= old.decay;
            node.activation = sigmoid((node.potential - node.threshold) * 2.0F);

            KineticVec3 velocity = (old.velocity + forces[i] * config_.dt) *
                                   config_.velocity_damping;
            const float speed = velocity.norm();
            if (speed > config_.max_speed) {
                velocity = velocity * (config_.max_speed / speed);
            }
            KineticVec3 displacement = velocity * config_.dt;
            const float displacement_norm = displacement.norm();
            if (displacement_norm > config_.max_displacement) {
                displacement = displacement *
                    (config_.max_displacement / displacement_norm);
            }
            node.velocity = velocity;
            node.position = old.position + displacement;
            telemetry_.max_speed = std::max(telemetry_.max_speed, velocity.norm());
            telemetry_.max_displacement = std::max(
                telemetry_.max_displacement, displacement.norm());
        }
    }

    bool validate_candidate(const std::vector<KineticNode>& candidate) const {
        if (candidate.size() > config_.max_nodes) return false;
        std::unordered_map<KineticNodeId, bool> ids;
        ids.reserve(candidate.size());
        for (const KineticNode& node : candidate) {
            if (node.id == 0U || !ids.emplace(node.id, true).second) return false;
        }
        for (const KineticNode& node : candidate) {
            if (!finite_vec(node.position) || !finite_vec(node.velocity) ||
                !finite_vec(node.normal)) return false;
            if (!std::isfinite(node.activation) || !std::isfinite(node.potential) ||
                !std::isfinite(node.threshold) || !std::isfinite(node.residual_error) ||
                !std::isfinite(node.spectral_weight)) return false;
            if (std::fabs(node.position.x) > config_.max_abs_position ||
                std::fabs(node.position.y) > config_.max_abs_position ||
                std::fabs(node.position.z) > config_.max_abs_position) return false;
            if (node.velocity.norm() > config_.max_speed + 1.0e-4F) return false;
            for (const KineticSynapse& synapse : node.synapses) {
                if (!std::isfinite(synapse.weight) || synapse.weight < -1.0F ||
                    synapse.weight > 1.0F || ids.find(synapse.target) == ids.end()) {
                    return false;
                }
            }
        }
        return true;
    }

    bool reject(const char* reason) {
        telemetry_.nodes_after = nodes_.size();
        telemetry_.committed = false;
        telemetry_.rejection_reason = reason;
        return false;
    }

    void finalize_telemetry() {
        if (nodes_.empty()) return;
        double activation = 0.0;
        double residual = 0.0;
        for (const KineticNode& node : nodes_) {
            activation += node.activation;
            residual += std::fabs(node.residual_error);
        }
        telemetry_.mean_activation = static_cast<float>(
            activation / static_cast<double>(nodes_.size()));
        telemetry_.mean_residual = static_cast<float>(
            residual / static_cast<double>(nodes_.size()));
    }
};

}  // namespace jarvisx
