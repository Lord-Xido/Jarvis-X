// Regression coverage for the executable's hash and command-line boundary.
// Rename the production entry point to exercise its exact implementation.
#define main jarvisx_gcain_entry_under_test
#include "../src/global_cognitive_action_kinetics_main.cpp"
#undef main

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static bool rejects_argument(const std::string& argument) {
    char program[] = "gcain-regressions";
    std::vector<char> mutable_argument(argument.begin(), argument.end());
    mutable_argument.push_back('\0');
    char* argv[] = {program, mutable_argument.data(), nullptr};
    std::ostringstream errors;
    auto* previous = std::cerr.rdbuf(errors.rdbuf());
    const int status = jarvisx_gcain_entry_under_test(2, argv);
    std::cerr.rdbuf(previous);
    return status == 2 && errors.str().find("Usage:") != std::string::npos;
}

int main() {
    // Independent canonical FNV-1a 64-bit test vectors.
    if (gcain::fnv1a64("") != 0xcbf29ce484222325ULL ||
        gcain::fnv1a64("a") != 0xaf63dc4c8601ec8cULL ||
        gcain::fnv1a64("hello") != 0xa430d84680aabd0bULL ||
        gcain::fnv1a64("foobar") != 0x85944171f73967e8ULL) {
        std::cerr << "FNV-1a standard vectors failed\n";
        return 1;
    }

    // All invalid inputs must fail before entering the ambient loop.
    for (const char* invalid : {"-1", "-0", "-123", "+1", "1junk",
                                "1.5", " 1", "", "184467440737095516160"}) {
        if (!rejects_argument(invalid)) {
            std::cerr << "Accepted invalid steps argument: [" << invalid << "]\n";
            return 1;
        }
    }

    char program[] = "gcain-regressions";
    char zero[] = "0";
    char* argv[] = {program, zero, nullptr};
    std::ostringstream output;
    auto* previous = std::cout.rdbuf(output.rdbuf());
    const int status = jarvisx_gcain_entry_under_test(2, argv);
    std::cout.rdbuf(previous);
    if (status != 0 || output.str().find("Verification: PASS") == std::string::npos) {
        std::cerr << "Valid zero-step execution did not verify\n";
        return 1;
    }

    std::cout << "FNV-1a and CLI regression checks: PASS\n";
    return 0;
}
