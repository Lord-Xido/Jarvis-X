#include "jarvisx/dmso_os.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::size_t resolution = 32U;
        std::size_t epochs = 20U;
        if (argc > 1) {
            resolution = static_cast<std::size_t>(std::stoul(argv[1]));
        }
        if (argc > 2) {
            epochs = static_cast<std::size_t>(std::stoul(argv[2]));
        }

        jarvisx::dmso::AutonomousSystemOS runtime(resolution);
        auto future = runtime.execute_system_lifecycle_async(epochs);
        const auto report = future.get();

        std::cout << std::fixed << std::setprecision(8)
                  << "{\n"
                  << "  \"epochs_executed\": " << report.epochs_executed << ",\n"
                  << "  \"accepted_updates\": " << report.accepted_updates << ",\n"
                  << "  \"rejected_updates\": " << report.rejected_updates << ",\n"
                  << "  \"initial_loss\": " << report.initial_loss << ",\n"
                  << "  \"final_loss\": " << report.final_loss << ",\n"
                  << "  \"converged\": " << (report.converged ? "true" : "false") << ",\n"
                  << "  \"elapsed_seconds\": " << report.elapsed_seconds << "\n"
                  << "}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "jarvisx-dmso-os: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
