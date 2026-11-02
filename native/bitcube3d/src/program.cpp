#include "bitcube3d/program.hpp"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_set>

namespace bitcube3d {
namespace {
constexpr std::uint64_t MAX_ROUNDS = 10'000;
constexpr std::uint64_t MAX_FILL = 1'000'000;
constexpr std::uint64_t MAX_LINES = 100'000;

std::string trim(std::string s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}
std::vector<std::string> tokenize(const std::string& text) {
    // Commands intentionally use whitespace-delimited arguments. Filenames
    // cannot include whitespace. '#' and '//' denote comments.
    std::istringstream is(text);
    std::string item;
    std::vector<std::string> tokens;
    while (is >> item) {
        if (item[0] == '#' || item.rfind("//", 0) == 0) break;
        tokens.push_back(item);
    }
    if (!tokens.empty()) {
        for (char& c : tokens[0]) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return tokens;
}
void arity(const std::vector<std::string>& t, unsigned lo, unsigned hi) {
    if (t.size() < lo || t.size() > hi) {
        throw std::runtime_error(t[0] + ": expected " + std::to_string(lo - 1) +
                                 (lo == hi ? "" : " to " + std::to_string(hi - 1)) + " arguments");
    }
}
std::uint64_t number(const std::string& text, std::uint64_t max, const char* label) {
    if (text.empty() || !std::all_of(text.begin(), text.end(),
             [](unsigned char c){return std::isdigit(c) != 0;}))
        throw std::runtime_error(std::string(label) + " must be an unsigned decimal integer");
    std::uint64_t n = 0;
    try {
        std::size_t used = 0;
        n = std::stoull(text, &used, 10);
        if (used != text.size() || n > max) throw std::out_of_range("range");
    } catch (const std::exception&) {
        throw std::runtime_error(std::string(label) + " out of range (max " + std::to_string(max) + ")");
    }
    return n;
}
std::uint32_t coord(const std::string& token) {
    return static_cast<std::uint32_t>(number(token, SIDE - 1, "coordinate"));
}
void check_path(const std::string& path) {
    if (path.empty() || path.size() > 4096 || path == "." || path == "..")
        throw std::runtime_error("invalid path");
}
void help(std::ostream& os) {
    os << "Commands: RESET | SET x y z 0|1 | CLEAR x y z | FLIP x y z | GET x y z\n"
       << "  SPHERE cx cy cz radius  (radius <=32; solid, clipped to domain)\n"
       << "  BOX x0 y0 z0 x1 y1 z1  (inclusive, <=1,000,000 bits)\n"
       << "  ENCODE [rounds] | DECODE [rounds] | VERIFY [rounds]\n"
       << "  ASSERT x y z 0|1 | STATS | SAVE file.b3d | LOAD file.b3d | HELP\n"
       << "  Files are B3D1, compatible with BitCube3D v1 C++/JavaScript.\n";
}
} // namespace

Program::Program(std::ostream& output, std::uint64_t max_bricks)
    : output_(&output), max_bricks_(max_bricks) {
    if (max_bricks == 0 || max_bricks > 1'000'000)
        throw std::invalid_argument("max_bricks must be 1..1,000,000");
}
void Program::require_plain() const {
    if (snap_.rounds != 0)
        throw std::runtime_error("state is encoded; DECODE outstanding rounds before editing");
}
void Program::check_brick_budget() const {
    if (snap_.engine.active_bricks() > max_bricks_)
        throw std::runtime_error("active brick budget exceeded");
}
void Program::run_line(const std::string& line) {
    const auto t = tokenize(trim(line));
    if (t.empty()) return;
    const auto& cmd = t[0];
    if (cmd == "HELP") {
        arity(t, 1, 1); help(*output_);
    } else if (cmd == "RESET") {
        arity(t, 1, 1);
        snap_ = Snapshot{};
        *output_ << "RESET OK\n";
    } else if (cmd == "SET" || cmd == "CLEAR" || cmd == "FLIP" || cmd == "GET" || cmd == "ASSERT") {
        arity(t, (cmd == "SET" || cmd == "ASSERT") ? 5 : 4, (cmd == "SET" || cmd == "ASSERT") ? 5 : 4);
        const auto x = coord(t[1]), y = coord(t[2]), z = coord(t[3]);
        if (cmd == "GET") {
            *output_ << "BIT " << x << " " << y << " " << z << " = " << snap_.engine.get(x,y,z) << '\n';
        } else if (cmd == "ASSERT") {
            const bool expected = number(t[4], 1, "bit") == 1;
            if (snap_.engine.get(x,y,z) != expected)
                throw std::runtime_error("ASSERT failed for voxel (" + std::to_string(x) + "," +
                                         std::to_string(y) + "," + std::to_string(z) + ")");
            *output_ << "ASSERT PASS\n";
        } else {
            require_plain();
            if (cmd == "SET") {
                bool value = number(t[4], 1, "bit") != 0;
                if (value && !snap_.engine.contains_brick(x,y,z) && snap_.engine.active_bricks() >= max_bricks_)
                    throw std::runtime_error("brick limit reached; edit refused");
                snap_.engine.set(x,y,z,value);
            } else if (cmd == "CLEAR") snap_.engine.set(x,y,z,false);
            else {
                if (!snap_.engine.contains_brick(x,y,z) && snap_.engine.active_bricks() >= max_bricks_)
                    throw std::runtime_error("brick limit reached; flip refused");
                snap_.engine.flip(x,y,z);
            }
        }
    } else if (cmd == "BOX" || cmd == "SPHERE") {
        require_plain();
        if (cmd == "BOX") {
            arity(t, 7, 7);
            std::uint32_t lo[3] = {coord(t[1]), coord(t[2]), coord(t[3])};
            std::uint32_t hi[3] = {coord(t[4]), coord(t[5]), coord(t[6])};
            std::uint64_t volume = 1;
            for (int i=0; i<3; ++i) {
                if (hi[i] < lo[i]) throw std::runtime_error("BOX upper corner precedes lower corner");
                const auto width = static_cast<std::uint64_t>(hi[i]) - lo[i] + 1;
                if (width > MAX_FILL || volume > MAX_FILL / width)
                    throw std::runtime_error("BOX exceeds 1,000,000 voxels");
                volume *= width;
            }
            // Preflight actual missing bricks, not a pessimistic total estimate.
            std::uint64_t additional = 0;
            for (std::uint64_t bz=lo[2]/BRICK_EDGE; bz<=hi[2]/BRICK_EDGE; ++bz)
                for (std::uint64_t by=lo[1]/BRICK_EDGE; by<=hi[1]/BRICK_EDGE; ++by)
                    for (std::uint64_t bx=lo[0]/BRICK_EDGE; bx<=hi[0]/BRICK_EDGE; ++bx)
                        if (!snap_.engine.contains_brick(static_cast<std::uint32_t>(bx*BRICK_EDGE),
                                static_cast<std::uint32_t>(by*BRICK_EDGE), static_cast<std::uint32_t>(bz*BRICK_EDGE)))
                            ++additional;
            if (additional > max_bricks_ - snap_.engine.active_bricks())
                throw std::runtime_error("BOX exceeds brick budget");
            for (std::uint64_t z=lo[2]; z<=hi[2]; ++z)
                for (std::uint64_t y=lo[1]; y<=hi[1]; ++y)
                    for (std::uint64_t x=lo[0]; x<=hi[0]; ++x)
                        snap_.engine.set(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z), true);
            *output_ << "BOX voxels=" << volume << '\n';
        } else {
            arity(t, 5, 5);
            const auto cx = coord(t[1]), cy = coord(t[2]), cz = coord(t[3]);
            const auto r = number(t[4], 32, "radius");
            const auto span = static_cast<std::int64_t>(r);
            // Stage the sparse brick coordinates before writing any bits.
            std::unordered_set<BrickCoord,BrickHash> missing;
            for (std::int64_t dz=-span; dz<=span; ++dz)
                for (std::int64_t dy=-span; dy<=span; ++dy)
                    for (std::int64_t dx=-span; dx<=span; ++dx) {
                        if (dx*dx + dy*dy + dz*dz > span*span) continue;
                        const auto x = static_cast<std::int64_t>(cx) + dx;
                        const auto y = static_cast<std::int64_t>(cy) + dy;
                        const auto z = static_cast<std::int64_t>(cz) + dz;
                        if (x<0 || y<0 || z<0 || x>=SIDE || y>=SIDE || z>=SIDE) continue;
                        const auto xx=static_cast<std::uint32_t>(x), yy=static_cast<std::uint32_t>(y), zz=static_cast<std::uint32_t>(z);
                        if (!snap_.engine.contains_brick(xx,yy,zz))
                            missing.insert({xx / BRICK_EDGE, yy / BRICK_EDGE, zz / BRICK_EDGE});
                    }
            if (missing.size() > max_bricks_ - snap_.engine.active_bricks())
                throw std::runtime_error("SPHERE exceeds brick budget");
            std::uint64_t written = 0;
            for (std::int64_t dz=-span; dz<=span; ++dz)
                for (std::int64_t dy=-span; dy<=span; ++dy)
                    for (std::int64_t dx=-span; dx<=span; ++dx) {
                        if (dx*dx + dy*dy + dz*dz > span*span) continue;
                        const auto x = static_cast<std::int64_t>(cx) + dx;
                        const auto y = static_cast<std::int64_t>(cy) + dy;
                        const auto z = static_cast<std::int64_t>(cz) + dz;
                        if (x<0 || y<0 || z<0 || x>=SIDE || y>=SIDE || z>=SIDE) continue;
                        snap_.engine.set(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), static_cast<std::uint32_t>(z),true);
                        ++written;
                    }
            *output_ << "SPHERE voxels=" << written << '\n';
        }
        check_brick_budget();
    } else if (cmd == "ENCODE") {
        arity(t, 1, 2);
        auto n = t.size()==2 ? number(t[1], MAX_ROUNDS, "rounds") : 1;
        if (n > MAX_ROUNDS - snap_.rounds) throw std::runtime_error("round metadata limit exceeded");
        snap_.engine.forward(static_cast<std::uint32_t>(n));
        snap_.rounds += static_cast<std::uint32_t>(n);
        *output_ << "ENCODE rounds=" << snap_.rounds << '\n';
    } else if (cmd == "DECODE") {
        arity(t, 1, 2);
        const auto n = t.size()==2 ? number(t[1], MAX_ROUNDS, "rounds") : snap_.rounds;
        if (n > snap_.rounds) throw std::runtime_error("cannot decode more rounds than outstanding");
        snap_.engine.inverse(static_cast<std::uint32_t>(n));
        snap_.rounds -= static_cast<std::uint32_t>(n);
        *output_ << "DECODE remaining_rounds=" << snap_.rounds << '\n';
    } else if (cmd == "VERIFY") {
        arity(t, 1, 2);
        const auto n = t.size()==2 ? number(t[1], MAX_ROUNDS, "rounds") : 8;
        auto trial = snap_.engine;
        trial.forward(static_cast<std::uint32_t>(n));
        trial.inverse(static_cast<std::uint32_t>(n));
        if (!(trial == snap_.engine)) throw std::runtime_error("VERIFY failed: bitwise mismatch");
        *output_ << "VERIFY PASS iterations=" << n << " active_bricks=" << snap_.engine.active_bricks() << '\n';
    } else if (cmd == "STATS") {
        arity(t, 1, 1);
        *output_ << "STATS side=" << SIDE << " logical_bits=10^27 active_bricks=" << snap_.engine.active_bricks()
                 << " active_bits=" << snap_.engine.active_bits() << " payload_bytes=" << snap_.engine.payload_bytes()
                 << " pending_rounds=" << snap_.rounds << '\n';
    } else if (cmd == "SAVE") {
        arity(t, 2, 2); check_path(t[1]);
        save(snap_, t[1]); *output_ << "SAVE " << t[1] << " OK\n";
    } else if (cmd == "LOAD") {
        arity(t, 2, 2); check_path(t[1]);
        const std::uint64_t max_bytes = 20 + max_bricks_ * (12 + BRICK_BYTES);
        const auto file_bytes = std::filesystem::file_size(t[1]);
        if (file_bytes > max_bytes) throw std::runtime_error("file exceeds allowed brick budget");
        Snapshot next = load(t[1]);
        if (next.rounds > MAX_ROUNDS || next.engine.active_bricks() > max_bricks_)
            throw std::runtime_error("file metadata exceeds runtime limits");
        snap_ = std::move(next);
        *output_ << "LOAD " << t[1] << " OK\n";
    } else {
        throw std::runtime_error("unknown opcode: " + cmd);
    }
    ++executed_;
}
void Program::run_script(std::istream& script) {
    std::string line;
    std::uint64_t line_number = 0;
    while (std::getline(script,line)) {
        if (++line_number > MAX_LINES) throw std::runtime_error("script exceeds 100,000 lines");
        try { run_line(line); }
        catch (const std::exception& e) {
            throw std::runtime_error("line " + std::to_string(line_number) + ": " + e.what());
        }
    }
    if (!script.eof()) throw std::runtime_error("input stream read failure");
}
void generate_cpp(std::istream& input, std::ostream& output) {
    std::string line;
    std::uint64_t count = 0;
    output << "// Generated BitCube3D C++17 program; build against bitcube3d_program.\n"
           << "#include \"bitcube3d/program.hpp\"\n"
           << "#include <exception>\n#include <iostream>\nint main() {\n"
           << "  try {\n    bitcube3d::Program vm(std::cout);\n";
    while (std::getline(input, line)) {
        if (++count > MAX_LINES) throw std::runtime_error("script exceeds 100,000 lines");
        if (line.size()>8192) throw std::runtime_error("script line exceeds 8192 chars");
        const auto t = tokenize(line);
        if (t.empty()) continue;
        // Use per-line raw literals with a delimiter absent from the original.
        // C++ compiles these to native program strings; Program validates them at runtime.
        const std::string delimiter = "B3DDSL";
        if (line.find(")" + delimiter + "\"") != std::string::npos)
            throw std::runtime_error("script includes unsupported raw string delimiter sequence");
        output << "    vm.run_line(R\"" << delimiter << "(" << line << ")" << delimiter << "\");\n";
    }
    output << "    return 0;\n  } catch (const std::exception& e) {\n"
           << "    std::cerr << \"ERROR: \" << e.what() << '\\n';\n    return 1;\n  }\n}\n";
    if (!output) throw std::runtime_error("cannot write generated C++ program");
}
} // namespace bitcube3d
