#include "jarvisx/processor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_constructor_normalizes_before_allocation() {
    jarvisx::Genome genome;
    genome.feature_dim = 1;
    genome.latent_dim = 1024;
    genome.iterations = 0;

    jarvisx::Processor processor(genome, jarvisx::text_packet("normalization"));
    const auto& normalized = processor.genome();

    require(normalized.feature_dim == 32, "feature_dim was not normalized");
    require(normalized.latent_dim == 32, "latent_dim was not normalized");
    require(normalized.iterations == 2, "iterations were not normalized");

    processor.run();
    require(processor.metrics().committed + processor.metrics().rejected == 2,
            "normalized iteration count was not executed");
}

void test_evaluation_is_deterministic() {
    jarvisx::Genome genome;
    const auto packet = jarvisx::text_packet("Jarvis X deterministic replay");

    const auto first = jarvisx::evaluate(genome, packet);
    const auto second = jarvisx::evaluate(genome, packet);

    require(first.valid && second.valid, "evaluation was invalid");
    require(first.metrics.cycles == second.metrics.cycles, "cycle count drifted");
    require(first.metrics.committed == second.metrics.committed,
            "commit count drifted");
    require(first.metrics.rejected == second.metrics.rejected,
            "reject count drifted");
    require(first.metrics.mse == second.metrics.mse, "MSE drifted");
    require(first.metrics.coherence == second.metrics.coherence,
            "coherence drifted");
    require(first.metrics.energy == second.metrics.energy, "energy drifted");
    require(first.tiles == second.tiles, "tile count drifted");
    require(first.memory_bytes == second.memory_bytes, "memory estimate drifted");
    require(first.fitness == second.fitness, "fitness drifted");
}

void test_latency_is_telemetry_not_fitness() {
    jarvisx::Evaluation fast;
    fast.valid = true;
    fast.metrics.mse = 0.01F;
    fast.metrics.coherence = 0.8F;
    fast.metrics.energy = 0.2F;
    fast.memory_bytes = 4096;
    fast.elapsed_ms = 1.0;

    jarvisx::Evaluation slow = fast;
    slow.elapsed_ms = 100000.0;

    require(jarvisx::fitness(fast) == jarvisx::fitness(slow),
            "wall-clock latency changed deterministic fitness");
}

} // namespace

int main() {
    try {
        test_constructor_normalizes_before_allocation();
        test_evaluation_is_deterministic();
        test_latency_is_telemetry_not_fitness();
        std::cout << "processor regression tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "processor regression test failed: " << error.what() << '\n';
        return 1;
    }
}
