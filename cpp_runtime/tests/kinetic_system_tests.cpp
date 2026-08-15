#include "jarvisx/kinetic_system.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <set>

using namespace jarvisx;

static KineticNode make_node(float x, float activation) {
    KineticNode node;
    node.position = {x, 0.0F, 0.0F};
    node.normal = {1.0F, 0.0F, 0.0F};
    node.activation = activation;
    node.potential = activation;
    node.threshold = 0.4F;
    return node;
}

int main() {
    {
        KineticSystemLoop runtime;
        const auto a = runtime.add_node(make_node(0.0F, 0.8F));
        const auto b = runtime.add_node(make_node(2.0F, 0.2F));
        runtime.connect(a, b, 0.5F, 1.0F);
        runtime.connect(b, a, 0.5F, 1.0F);
        runtime.enqueue({KineticOperationType::Physics, KineticScope::Global, 0U, 0.1F, 0U});
        runtime.enqueue({KineticOperationType::Swarm, KineticScope::Global, 0U, 1.0F, 0U});
        assert(runtime.step());
        assert(runtime.telemetry().committed);
        assert(runtime.nodes().size() == 2U);
        assert(runtime.node(a).position.x > 0.0F);
        assert(runtime.node(b).position.x < 2.0F);
    }

    {
        KineticConfig config;
        config.echo_damping = 0.5F;
        config.echo_epsilon = 1.0e-3F;
        config.max_events_per_step = 100U;
        KineticSystemLoop runtime(config);
        const auto a = runtime.add_node(make_node(0.0F, 0.5F));
        const auto b = runtime.add_node(make_node(1.0F, 0.5F));
        runtime.connect(a, b, 0.5F);
        runtime.connect(b, a, 0.5F);
        runtime.enqueue({KineticOperationType::Echo, KineticScope::Direct, a, 1.0F, 6U});
        assert(runtime.step());
        assert(runtime.telemetry().propagated_events > 0U);
        assert(runtime.telemetry().operations_consumed <= config.max_events_per_step);
    }

    {
        KineticSystemLoop runtime;
        const auto parent = runtime.add_node(make_node(0.0F, 0.5F));
        runtime.enqueue({KineticOperationType::Spawn, KineticScope::Direct, parent, 3.0F, 0U});
        assert(runtime.step());
        assert(runtime.nodes().size() == 4U);
        assert(runtime.telemetry().spawned_nodes == 3U);
        std::set<KineticNodeId> ids;
        for (const auto& node : runtime.nodes()) ids.insert(node.id);
        assert(ids.size() == runtime.nodes().size());
    }

    {
        KineticSystemLoop runtime;
        const auto a = runtime.add_node(make_node(0.0F, 0.5F));
        const auto b = runtime.add_node(make_node(1.0F, 0.5F));
        runtime.connect(a, b, 0.01F);
        runtime.connect(a, b, 0.5F);
        runtime.enqueue({KineticOperationType::Prune, KineticScope::Direct, a, 0.05F, 0U});
        assert(runtime.step());
        assert(runtime.node(a).synapses.size() == 1U);
        assert(runtime.telemetry().pruned_synapses == 1U);
    }

    {
        KineticConfig config;
        config.max_abs_position = 0.5F;
        KineticSystemLoop runtime(config);
        auto node = make_node(0.49F, 1.0F);
        node.spectral_weight = 10.0F;
        const auto id = runtime.add_node(node);
        const auto before = runtime.node(id).position.x;
        runtime.enqueue({KineticOperationType::Decode, KineticScope::Global, 0U, 100.0F, 0U});
        assert(!runtime.step());
        assert(!runtime.telemetry().committed);
        assert(runtime.node(id).position.x == before);
    }

    {
        KineticSystemLoop runtime;
        const auto a = runtime.add_node(make_node(0.0F, 0.8F));
        const auto b = runtime.add_node(make_node(1.0F, 0.2F));
        runtime.connect(a, b, 0.7F);
        const float old_weight = runtime.node(a).synapses.front().weight;
        runtime.enqueue({KineticOperationType::Encode, KineticScope::Global, 0U, 1.0F, 0U});
        runtime.enqueue({KineticOperationType::AI, KineticScope::Global, 0U, 1.0F, 0U});
        runtime.enqueue({KineticOperationType::Learn, KineticScope::Global, 0U, 1.0F, 0U});
        runtime.enqueue({KineticOperationType::Optimize, KineticScope::Global, 0U, 1.0F, 0U});
        assert(runtime.step());
        assert(std::isfinite(runtime.telemetry().mean_residual));
        assert(runtime.node(a).synapses.front().weight != old_weight);
    }

    std::cout << "kinetic system regressions passed\n";
    return 0;
}
