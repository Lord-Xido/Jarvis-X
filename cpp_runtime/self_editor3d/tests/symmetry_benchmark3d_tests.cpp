#include "jarvisx/symmetry_benchmark3d.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace jarvisx::symmetry3d;
using namespace jarvisx::symmetry3d::bench;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool same_grid(const Grid& a, const Grid& b) {
    return a == b;
}

void test_theoretical_majority_error() {
    require(std::abs(majority_vote_bit_error(0.0)) < 1.0e-15,
            "zero latent noise must have zero theoretical BER");
    require(std::abs(majority_vote_bit_error(0.2) - 0.104) < 1.0e-12,
            "three-copy majority BER formula is incorrect");
    require(std::abs(majority_vote_bit_error(0.5) - 0.5) < 1.0e-12,
            "majority BER must be 0.5 at p=0.5");
}

void test_corruption_is_deterministic() {
    const Grid source = make_reference_pattern(32, 1234);
    const Grid a = corrupt_grid(source, 0.2, 99);
    const Grid b = corrupt_grid(source, 0.2, 99);
    const Grid c = corrupt_grid(source, 0.2, 100);
    require(same_grid(a, b), "corruption must replay exactly for the same seed");
    require(!same_grid(a, c), "different corruption seeds should produce different fixtures");
    require(same_grid(source, corrupt_grid(source, 0.0, 99)),
            "zero corruption must preserve the source exactly");
}

void test_independent_latent_redundancy_improves_recovery() {
    const std::size_t n = 128;
    const double p = 0.2;
    const Grid reference = make_reference_pattern(n, 777);
    SymmetryCodec codec(n);
    const Tensor3 clean = codec.encode(reference);

    double single_error = 0.0;
    double majority_error = 0.0;
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
        const Tensor3 noisy = corrupt_latent_independently(clean, p, seed * 1009ULL);
        single_error += 1.0 - bit_accuracy(noisy.layer[0], reference);
        majority_error += 1.0 - bit_accuracy(codec.decode_majority(noisy), reference);
    }
    single_error /= 4.0;
    majority_error /= 4.0;

    require(majority_error < single_error,
            "independent three-copy majority decoding must beat a single noisy copy at p=0.2");
    require(std::abs(majority_error - majority_vote_bit_error(p)) < 0.02,
            "empirical majority BER deviates unexpectedly from the theoretical repetition-code BER");
}

void test_benchmark_matrix_and_csv() {
    BenchmarkConfig config;
    config.sizes = {8};
    config.noise_levels = {0.0, 0.2};
    config.repeats = 1;
    config.loop.max_optimization_sweeps = 16;
    config.loop.max_feedback_steps = 16;

    BenchmarkRunner runner(config);
    const auto rows = runner.run();
    require(rows.size() == 12,
            "one size x two noise levels x two domains x three models must produce 12 rows");

    bool saw_common_exact = false;
    bool saw_latent_majority = false;
    bool saw_closed = false;
    for (const BenchmarkRow& row : rows) {
        require(row.n == 8, "benchmark row has wrong grid size");
        require(row.bit_accuracy >= 0.0 && row.bit_accuracy <= 1.0,
                "bit accuracy must remain in [0,1]");
        require(row.mse >= 0.0, "MSE must be non-negative");
        require(row.latency_us >= 0.0, "latency must be non-negative");
        require(row.throughput_mpix_s >= 0.0, "throughput must be non-negative");
        require(row.estimated_working_set_bytes > 0, "working-set estimate must be positive");

        if (row.model == "symmetry-exact" && row.noise_domain == "input-common") {
            saw_common_exact = true;
        }
        if (row.model == "symmetry-majority" && row.noise_domain == "latent-independent") {
            saw_latent_majority = true;
        }
        if (row.model == "symmetry-closed-loop" && row.noise_domain == "input-common") {
            saw_closed = true;
            require(row.final_objective <= row.initial_objective,
                    "closed-loop benchmark must not report an objective regression");
            require(row.latent_expansion_ratio == 3.0,
                    "symmetry latent representation must report threefold scalar expansion");
        }
    }
    require(saw_common_exact && saw_latent_majority && saw_closed,
            "benchmark matrix is missing required models or noise domains");

    const BenchmarkSummary summary = runner.summarize(rows);
    require(summary.row_count == rows.size(), "summary row count mismatch");

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "jarvisx-symmetry-benchmark3d-test.csv";
    runner.write_csv(path, rows);
    std::ifstream input(path);
    require(static_cast<bool>(input), "benchmark CSV was not created");
    std::string header;
    std::getline(input, header);
    require(header.find("bit_accuracy") != std::string::npos,
            "benchmark CSV header is missing bit accuracy");
    require(header.find("optimization_latency_us") != std::string::npos,
            "benchmark CSV header is missing optimization latency");
    require(header.find("latent_expansion_ratio") != std::string::npos,
            "benchmark CSV header is missing latent expansion ratio");
    input.close();
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    try {
        test_theoretical_majority_error();
        test_corruption_is_deterministic();
        test_independent_latent_redundancy_improves_recovery();
        test_benchmark_matrix_and_csv();
        std::cout << "symmetry-benchmark3d regressions: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "symmetry-benchmark3d regressions: FAIL: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
