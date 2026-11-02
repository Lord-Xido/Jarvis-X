#pragma once
#include "bitcube3d/engine.hpp"
#include <cstdint>
#include <iosfwd>
#include <string>

namespace bitcube3d {
// Safe, bounded line-oriented scripting runtime backed by the native C++ bit cube.
// This is a bytecode-like DSL interpreter, not an arbitrary C++ compiler or sandbox.
class Program {
public:
    explicit Program(std::ostream& output, std::uint64_t max_bricks = 100'000);
    // Execute one command. Throws std::runtime_error with validation details.
    void run_line(const std::string& line);
    // Throws a line-numbered exception on the first failed command.
    void run_script(std::istream& script);
    const Snapshot& snapshot() const noexcept { return snap_; }
    std::uint64_t executed_commands() const noexcept { return executed_; }
private:
    Snapshot snap_{};
    std::ostream* output_;
    std::uint64_t max_bricks_;
    std::uint64_t executed_ = 0;
    void require_plain() const;
    void check_brick_budget() const;
};

// Emit a buildable C++17 executable which replays the supplied validated DSL
// via Program, with one raw C++ string literal per command. No shell execution.
void generate_cpp(std::istream& input, std::ostream& output);
}  // namespace bitcube3d
