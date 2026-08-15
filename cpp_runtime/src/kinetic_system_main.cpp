#include "jarvisx/kinetic_system.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace jarvisx;

namespace {

std::size_t parse_size(const char* value, const char* name) {
    try {
        const auto parsed = std::stoull(value);
        if (parsed == 0U) throw std::invalid_argument("zero");
        return static_cast<std::size_t>(parsed);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t cycles = 120U;
    std::size_t node_count = 64U;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--cycles" && i + 1 < argc) {
            cycles = parse_size(argv[++i], "cycle count");
        } else if (arg == "--nodes" && i + 1 < argc) {
            node_count = parse_size(argv[++i], "node count");
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--help") {
            std::cout << "usage: jarvisx-kinetic-loop [--cycles N] [--nodes N] [--quiet]\n";
            return 0;
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }

    KineticConfig config;
    config.max_nodes = std::max<std::size_t>(node_count * 4U, 128U);
    KineticSystemLoop runtime(config);
    std::vector<KineticNodeId> ids;
    ids.reserve(node_count);

    constexpr float kPi = 3.14159265358979323846F;
    for (std::size_t i = 0; i < node_count; ++i) {
        const float phase = 2.0F * kPi * static_cast<float>(i) /
                            static_cast<float>(node_count);
        KineticNode node;
        node.position = {10.0F * std::cos(phase),
                         2.0F * std::sin(phase * 3.0F),
                         10.0F * std::sin(phase)};
        node.normal = node.position.normalized();
        node.activation = 0.45F + 0.35F * std::sin(phase);
        node.potential = node.activation;
        node.threshold = 0.45F;
        node.curvature = 0.1F + 0.1F * std::fabs(std::cos(phase));
        ids.push_back(runtime.add_node(node));
    }

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const std::size_t next = (i + 1U) % ids.size();
        const std::size_t prev = (i + ids.size() - 1U) % ids.size();
        runtime.connect(ids[i], ids[next], 0.55F, 1.0F, 0.1F);
        runtime.connect(ids[i], ids[prev], 0.55F, 1.0F, 0.1F);
    }

    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        runtime.enqueue({KineticOperationType::Encode, KineticScope::Global, 0U, 1.0F, 0U});
        runtime.enqueue({KineticOperationType::AI, KineticScope::Global, 0U, 0.5F, 0U});
        runtime.enqueue({KineticOperationType::Physics, KineticScope::Global, 0U, 0.02F, 0U});
        runtime.enqueue({KineticOperationType::Swarm, KineticScope::Global, 0U, 0.2F, 0U});
        runtime.enqueue({KineticOperationType::Decode, KineticScope::Global, 0U, 0.1F, 0U});
        runtime.enqueue({KineticOperationType::Learn, KineticScope::Global, 0U, 0.25F, 0U});
        runtime.enqueue({KineticOperationType::Optimize, KineticScope::Global, 0U, 0.1F, 0U});
        if (cycle % 20U == 0U) {
            runtime.enqueue({KineticOperationType::Echo, KineticScope::Direct,
                             ids.front(), 1.0F, 6U});
        }

        if (!runtime.step()) {
            std::cerr << "kinetic transaction rejected at cycle " << cycle
                      << ": " << runtime.telemetry().rejection_reason << "\n";
            return 1;
        }

        if (!quiet && (cycle % 10U == 0U || cycle + 1U == cycles)) {
            const auto& t = runtime.telemetry();
            std::cout << "cycle=" << std::setw(4) << t.cycle
                      << " nodes=" << std::setw(4) << t.nodes_after
                      << " ops=" << std::setw(5) << t.operations_consumed
                      << " echo=" << std::setw(4) << t.propagated_events
                      << " act=" << std::fixed << std::setprecision(4)
                      << t.mean_activation
                      << " residual=" << t.mean_residual
                      << " vmax=" << t.max_speed
                      << " dxmax=" << t.max_displacement << "\n";
        }
    }

    return 0;
}
