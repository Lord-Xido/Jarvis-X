// Moagi 3D virtual code runtime: sparse, deterministic addressing of 10^24
// logical C++ line instances. Strict ISO C++17; no 128-bit compiler extension.
// This is an addressing and regeneration demonstration, NOT 10^24 executions.
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace jx3d {
constexpr std::uint64_t OUTER = 1'000'000'000ULL;
constexpr std::uint64_t INNER = 1'000'000ULL;
constexpr std::uint64_t LINES = 1'000'000'000ULL;
constexpr std::uint64_t MASK30 = (1ULL << 30) - 1ULL;
constexpr std::uint64_t MASK20 = (1ULL << 20) - 1ULL;
constexpr std::uint64_t MASK14 = (1ULL << 14) - 1ULL;
constexpr std::uint64_t PRIME = 1'099'511'628'211ULL;
constexpr std::uint64_t INIT = 1'469'598'103'934'665'603ULL;

struct Address { std::uint64_t outer, inner, line; };
struct Packed80 { std::uint16_t high; std::uint64_t low; };

std::uint64_t parse_uint(const std::string& v) {
    if (v.empty()) throw std::invalid_argument("Empty coordinate");
    for (char c : v) if (c < '0' || c > '9')
        throw std::invalid_argument("Coordinates must be unsigned decimal integers");
    std::size_t consumed = 0;
    const auto result = std::stoull(v, &consumed, 10);
    if (consumed != v.size()) throw std::invalid_argument("Invalid coordinate");
    return result;
}
void validate(Address a) {
    if (a.outer >= OUTER || a.inner >= INNER || a.line >= LINES)
        throw std::out_of_range("Logical coordinate outside bounds");
}
Packed80 pack(Address a) {
    validate(a);
    // 80-bit packing: [outer:30 bits][inner:20 bits][line:30 bits].
    return {static_cast<std::uint16_t>(a.outer >> 14),
            ((a.outer & MASK14) << 50) | (a.inner << 30) | a.line};
}
Address unpack(Packed80 p) {
    const Address a{(std::uint64_t(p.high) << 14) | ((p.low >> 50) & MASK14),
                    (p.low >> 30) & MASK20,
                    p.low & MASK30};
    validate(a);
    return a;
}
std::string to_hex(Packed80 p) {
    std::ostringstream o;
    o << std::hex << std::setfill('0') << std::setw(4) << p.high
      << std::setw(16) << p.low;
    return o.str();
}
Packed80 from_hex(const std::string& h) {
    if (h.size() != 20) throw std::invalid_argument("Packed coordinate requires 20 hex digits");
    for (char c : h) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
        throw std::invalid_argument("Invalid hex coordinate");
    return {static_cast<std::uint16_t>(std::stoul(h.substr(0, 4), nullptr, 16)),
            static_cast<std::uint64_t>(std::stoull(h.substr(4), nullptr, 16))};
}
std::uint64_t mix64(std::uint64_t v) {
    v += 0x9e3779b97f4a7c15ULL;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
    return v ^ (v >> 31);
}
std::uint64_t site_seed(Address a) {
    validate(a);
    std::uint64_t s = mix64(0x4d4f414749334431ULL ^ a.outer);
    s = mix64(s ^ a.inner);
    return mix64(s ^ a.line);
}
std::uint64_t constant(Address a) {
    // True 3D dependency: six face-adjacent sites, periodic boundaries.
    // Each site is computed independently from its immutable coordinates;
    // this is not a mutable, iterated cellular automaton.
    validate(a);
    const Address neighbours[6] = {
        {(a.outer + OUTER - 1) % OUTER, a.inner, a.line},
        {(a.outer + 1) % OUTER, a.inner, a.line},
        {a.outer, (a.inner + INNER - 1) % INNER, a.line},
        {a.outer, (a.inner + 1) % INNER, a.line},
        {a.outer, a.inner, (a.line + LINES - 1) % LINES},
        {a.outer, a.inner, (a.line + 1) % LINES}
    };
    std::uint64_t s = site_seed(a);
    for (Address neighbour : neighbours) s = mix64(s ^ site_seed(neighbour));
    return s;
}
void emit_statement(std::ostream& o, Address a) {
    o << "  acc = (acc ^ " << constant(a) << "ULL) * 1099511628211ULL;\n";
}
std::uint64_t evaluate(Address a, std::uint64_t n) {
    if (n > LINES) throw std::out_of_range("Requested sample exceeds logical program size");
    std::uint64_t acc = INIT;
    for (std::uint64_t k = 0; k < n; ++k) {
        a.line = k;
        acc = (acc ^ constant(a)) * PRIME;
    }
    return acc;
}
void emit_demo(Address a, std::uint64_t n, const std::string& filename) {
    if (n == 0 || n > 1'000'000ULL) throw std::out_of_range("Demo limited to 1..1000000 statements");
    validate(a);
    std::ofstream o(filename, std::ios::binary);
    if (!o) throw std::runtime_error("Cannot open source output");
    o << "#include <cstdint>\n#include <iostream>\nint main() {\n"
      << "  std::uint64_t acc = 1469598103934665603ULL;\n";
    for (std::uint64_t k = 0; k < n; ++k) {
        a.line = k;
        emit_statement(o, a);
    }
    o << "  std::cout << acc << '\\n';\n}\n";
    o.close();
    if (!o) throw std::runtime_error("Source write failure");
    std::cout << "source=" << filename << " statements=" << n
              << " file_lines=" << n + 6 << " expected_output=" << evaluate(a, n) << '\n';
}
void self_test() {
    const Address cases[] = {
        {0, 0, 0}, {0, 0, LINES - 1}, {OUTER - 1, INNER - 1, LINES - 1},
        {OUTER - 1, 0, 0}, {0, INNER - 1, 0}, {123456789, 543210, 987654321}
    };
    std::uint64_t tested = 0;
    for (auto a : cases) {
        auto b = unpack(from_hex(to_hex(pack(a))));
        if (a.outer != b.outer || a.inner != b.inner || a.line != b.line)
            throw std::runtime_error("80-bit packing roundtrip failed");
        ++tested;
    }
    for (std::uint64_t k = 0; k < 100'000; ++k) {
        const Address a{mix64(k) % OUTER, mix64(k ^ 0xabbccddeeffULL) % INNER,
                        mix64(k ^ 0xfeeddccbbaaULL) % LINES};
        auto b = unpack(pack(a));
        if (a.outer != b.outer || a.inner != b.inner || a.line != b.line)
            throw std::runtime_error("Random address roundtrip failed");
        if (constant(a) != constant(b))
            throw std::runtime_error("Regeneration mismatch");
        ++tested;
    }
    try { pack({OUTER, 0, 0}); throw std::runtime_error("Missing outer-bound rejection"); }
    catch (const std::out_of_range&) {}
    try { pack({0, INNER, 0}); throw std::runtime_error("Missing inner-bound rejection"); }
    catch (const std::out_of_range&) {}
    try { pack({0, 0, LINES}); throw std::runtime_error("Missing line-bound rejection"); }
    catch (const std::out_of_range&) {}
    std::cout << "verification=PASS address_roundtrips=" << tested
              << " bounds_checks=3 key_bits=80 logical_line_instances=10^24\n";
    std::cout << "first_address=" << to_hex(pack({0,0,0})) << '\n';
    std::cout << "last_address=" << to_hex(pack({OUTER-1,INNER-1,LINES-1})) << '\n';
}
}  // namespace jx3d

int main(int argc, char** argv) {
    try {
        using namespace jx3d;
        if (argc == 2 && std::string(argv[1]) == "--verify") {
            self_test();
        } else if (argc == 5 && std::string(argv[1]) == "--line") {
            emit_statement(std::cout, {parse_uint(argv[2]), parse_uint(argv[3]), parse_uint(argv[4])});
        } else if (argc == 5 && std::string(argv[1]) == "--pack") {
            std::cout << to_hex(pack({parse_uint(argv[2]), parse_uint(argv[3]), parse_uint(argv[4])})) << '\n';
        } else if (argc == 3 && std::string(argv[1]) == "--unpack") {
            auto a = unpack(from_hex(argv[2]));
            std::cout << a.outer << ' ' << a.inner << ' ' << a.line << '\n';
        } else if (argc == 6 && std::string(argv[1]) == "--emit-demo") {
            emit_demo({parse_uint(argv[2]), parse_uint(argv[3]), 0}, parse_uint(argv[4]), argv[5]);
        } else {
            std::cerr << "Usage:\n"
                      << "  runtime --verify\n  runtime --pack OUTER INNER LINE\n"
                      << "  runtime --unpack HEX80\n  runtime --line OUTER INNER LINE\n"
                      << "  runtime --emit-demo OUTER INNER NUM_STATEMENTS FILE.cpp\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 2;
    }
}
