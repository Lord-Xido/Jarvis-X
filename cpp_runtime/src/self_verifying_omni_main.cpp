#include "jarvisx/self_verifying_omni.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace {

std::uint64_t integer(std::string_view value) {
    std::uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::invalid_argument("expected an unsigned integer");
    }
    return result;
}

void number(double value) {
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}

void usage() {
    std::cout << "Usage: jarvisx-self-verifying-omni [--pathways 1..10000000]\n"
              << "       [--depth 1..4096] [--threads 1..1024] [--seed UINT64]\n"
              << "       [--json] [--require-convergence]\n"
              << "Defaults: 1000000 pathways, 16 steps, up to 64 worker threads, seed 0.\n"
              << "Exit: 0 verified; 1 verification failure; 2 convergence required but incomplete;\n"
              << "      64 invalid arguments or allocation failure.\n";
}

} // namespace

int main(int argc, char** argv) {
    using namespace jarvisx::omni;
    try {
        Config config;
        bool json = false;
        bool require_convergence = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") { usage(); return 0; }
            if (option == "--json") { json = true; continue; }
            if (option == "--require-convergence") { require_convergence = true; continue; }
            if (option != "--pathways" && option != "--depth" && option != "--threads" && option != "--seed") {
                throw std::invalid_argument("unknown argument: " + option);
            }
            if (++i == argc) throw std::invalid_argument("missing value for " + option);
            const auto value = integer(argv[i]);
            if (option == "--pathways") {
                if (value == 0 || value > maximum_pathways) throw std::invalid_argument("pathways outside 1..10000000");
                config.pathways = static_cast<std::size_t>(value);
            } else if (option == "--depth") {
                if (value == 0 || value > 4096) throw std::invalid_argument("depth outside 1..4096");
                config.depth = static_cast<std::uint32_t>(value);
            } else if (option == "--threads") {
                if (value == 0 || value > 1024) throw std::invalid_argument("threads outside 1..1024");
                config.requested_threads = static_cast<int>(value);
            } else {
                config.seed = value;
            }
        }

        SelfVerifyingOmniEngine engine(config);
        const RunStats stats = engine.execute_verified_inward_loop();
        const VerificationReport report = engine.verify_end_to_end_mechanics();
        const bool passed = report.passed();
        const bool all_converged = report.converged == config.pathways;
        const auto possible_updates = static_cast<std::uint64_t>(config.pathways) * config.depth;
        std::cout << std::setprecision(17) << std::boolalpha;
        if (json) {
            std::cout << "{\"schema_version\":1,\"pathways\":" << config.pathways
                      << ",\"depth\":" << config.depth
                      << ",\"seed\":" << config.seed
                      << ",\"requested_threads\":" << config.requested_threads
                      << ",\"worker_threads\":" << stats.worker_threads
#ifdef _OPENMP
                      << ",\"openmp_enabled\":true"
#else
                      << ",\"openmp_enabled\":false"
#endif
                      << ",\"resident_payload_bytes\":" << engine.resident_payload_bytes()
                      << ",\"executed_updates\":" << stats.executed_updates
                      << ",\"skipped_updates\":" << possible_updates - stats.executed_updates
                      << ",\"rejected_updates\":" << stats.rejected_updates
                      << ",\"elapsed_ms\":";
            number(stats.elapsed_ms);
            std::cout << ",\"active\":" << report.active << ",\"pruned\":" << report.pruned
                      << ",\"converged\":" << report.converged
                      << ",\"premature_prunes\":" << report.premature_prunes
                      << ",\"nonfinite_pathways\":" << report.nonfinite_pathways
                      << ",\"invalid_pathways\":" << report.invalid_pathways
                      << ",\"storage_valid\":" << report.storage_valid
                      << ",\"execution_valid\":" << report.execution_valid
                      << ",\"total_norm\":";
            number(report.total_norm);
            std::cout << ",\"norm_sum_error\":"; number(report.norm_sum_error);
            std::cout << ",\"max_norm_error\":"; number(report.max_norm_error);
            std::cout << ",\"mean_absolute_norm_error\":"; number(report.mean_absolute_norm_error);
            std::cout << ",\"initial_mean_radius\":"; number(report.initial_mean_radius);
            std::cout << ",\"final_mean_radius\":"; number(report.final_mean_radius);
            std::cout << ",\"max_residual_mse\":"; number(report.max_residual_mse);
            std::cout << ",\"state_checksum\":"; number(report.state_checksum);
            std::cout << ",\"total_budget\":"; number(report.total_budget);
            std::cout << ",\"verification_passed\":" << passed
                      << ",\"all_converged\":" << all_converged << "}\n";
        } else {
            std::cout << "=== Dr Moagi bounded self-verifying inward engine ===\n"
                      << "Pathways: " << config.pathways << "; channels: " << channels
                      << "; recurrence steps: " << config.depth << "\n"
                      << "Worker threads used: " << stats.worker_threads
                      << " (requested " << config.requested_threads << ")\n"
                      << "Resident array/grid payload: " << engine.resident_payload_bytes() << " bytes\n"
                      << "Inward loop: " << stats.elapsed_ms << " ms; updates: " << stats.executed_updates << "\n"
                      << "Unit-norm sum error: " << report.norm_sum_error
                      << "; maximum per-pathway error: " << report.max_norm_error << "\n"
                      << "Mean radius: " << report.initial_mean_radius << " -> " << report.final_mean_radius << "\n"
                      << "Active: " << report.active << "; pruned: " << report.pruned
                      << "; converged: " << report.converged << "\n"
                      << "Rejected updates: " << stats.rejected_updates
                      << "; non-finite pathways: " << report.nonfinite_pathways
                      << "; invalid pathways: " << report.invalid_pathways
                      << "; premature prunes: " << report.premature_prunes << "\n"
                      << "Verification: " << (passed ? "PASS" : "FAIL")
                      << "; full convergence: " << all_converged << "\n";
        }
        if (!passed) return 1;
        if (require_convergence && !all_converged) return 2;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 64;
    }
}
