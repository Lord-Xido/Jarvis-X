#include "jarvisx/runtime.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_throws(Function function, const char* message) {
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
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

void test_latency_is_telemetry_not_selection() {
    jarvisx::Evaluation fast;
    fast.valid = true;
    fast.metrics.mse = 0.01F;
    fast.metrics.coherence = 0.8F;
    fast.metrics.energy = 0.2F;
    fast.memory_bytes = 4096;
    fast.elapsed_ms = 1.0;
    fast.fitness = jarvisx::fitness(fast);

    jarvisx::Evaluation slow = fast;
    slow.elapsed_ms = 100000.0;

    require(jarvisx::fitness(fast) == jarvisx::fitness(slow),
            "wall-clock latency changed deterministic fitness");
    require(!jarvisx::evaluation_less(fast, slow) &&
                !jarvisx::evaluation_less(slow, fast),
            "wall-clock latency changed equal-fitness selection order");
}

void test_bounded_unsigned_mutation_does_not_wrap() {
    require(jarvisx::bounded_u16(2, -4, 2, 128) == 2,
            "iteration mutation wrapped below its lower bound");
    require(jarvisx::bounded_u16(1, -30, 1, 400) == 1,
            "learning mutation wrapped below its lower bound");
    require(jarvisx::bounded_u16(9000, 500, 0, 9000) == 9000,
            "coherence mutation exceeded its upper bound");
    require(jarvisx::bounded_size(32, -64, 32, 512) == 32,
            "feature mutation wrapped below its lower bound");
}

void test_checkpoint_round_trip_and_tamper_detection() {
    jarvisx::Genome genome;
    genome.generation = 7;
    genome.seed = 123456789ULL;
    genome.clamp();

    const std::string encoded = jarvisx::serialize(genome);
    const jarvisx::Genome restored = jarvisx::deserialize(encoded);

    require(restored.generation == genome.generation,
            "checkpoint generation did not round trip");
    require(restored.seed == genome.seed, "checkpoint seed did not round trip");
    require(restored.fingerprint() == genome.fingerprint(),
            "checkpoint fingerprint did not round trip");

    std::string corrupted = encoded;
    const auto fingerprint = corrupted.find("fingerprint=");
    require(fingerprint != std::string::npos, "checkpoint fingerprint was absent");
    corrupted.replace(fingerprint + 12, genome.fingerprint().size(), "CORRUPTED");

    require_throws(
        [&corrupted]() { static_cast<void>(jarvisx::deserialize(corrupted)); },
        "corrupt checkpoint fingerprint was accepted");

    const auto required_field = corrupted.find("iterations=");
    require(required_field != std::string::npos, "checkpoint field was absent");
    const auto required_end = corrupted.find('\n', required_field);
    corrupted.erase(required_field, required_end - required_field + 1);

    require_throws(
        [&corrupted]() { static_cast<void>(jarvisx::deserialize(corrupted)); },
        "checkpoint with a missing required field was accepted");
}

} // namespace

int main() {
    try {
        test_constructor_normalizes_before_allocation();
        test_evaluation_is_deterministic();
        test_latency_is_telemetry_not_selection();
        test_bounded_unsigned_mutation_does_not_wrap();
        test_checkpoint_round_trip_and_tamper_detection();
        std::cout << "processor regression tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "processor regression test failed: " << error.what() << '\n';
        return 1;
    }
}
