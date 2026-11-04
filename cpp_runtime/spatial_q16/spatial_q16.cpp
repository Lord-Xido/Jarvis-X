// Jarvis-X Q16.16 deterministic sparse 3D field and reversible page transform.
// Integration candidate: this is NOT a trained ANN, compression, encryption,
// bytecode interpreter, quantum simulation, or a proof of autonomous convergence.
// Build: c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic spatial_q16.cpp -o spatial_q16
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace jarvisx::spatial {
constexpr std::uint32_t DIM = 1000;
constexpr std::size_t PAGE_BYTES = 4096;
constexpr std::size_t LATENTS = PAGE_BYTES / 4;
constexpr std::uint32_t MAX_ROUNDS = 512;
constexpr std::uint32_t SCALE = 65536;
constexpr std::size_t HEADER_BYTES = 32;
constexpr std::array<std::uint8_t, 8> MAGIC = {'J','X','Q','1','6','V','1',0};

class Fixed16_16 {
    std::int32_t value_ = 0;
    static std::int32_t checked(std::int64_t v) {
        if (v < std::numeric_limits<std::int32_t>::min() ||
            v > std::numeric_limits<std::int32_t>::max())
            throw std::overflow_error("Q16.16 arithmetic overflow");
        return static_cast<std::int32_t>(v);
    }
public:
    static constexpr Fixed16_16 from_raw(std::int32_t v) {
        Fixed16_16 r;
        r.value_ = v;
        return r;
    }
    static Fixed16_16 from_ratio(std::int32_t n, std::int32_t d) {
        if (d == 0) throw std::invalid_argument("zero denominator");
        return from_raw(checked((std::int64_t(n) * SCALE) / d));
    }
    std::int32_t raw() const { return value_; }
    double to_double() const { return double(value_) / SCALE; }
    friend Fixed16_16 operator+(Fixed16_16 a, Fixed16_16 b) {
        return from_raw(checked(std::int64_t(a.value_) + b.value_));
    }
    friend Fixed16_16 operator-(Fixed16_16 a, Fixed16_16 b) {
        return from_raw(checked(std::int64_t(a.value_) - b.value_));
    }
    friend Fixed16_16 operator*(Fixed16_16 a, Fixed16_16 b) {
        const std::int64_t prod = std::int64_t(a.value_) * b.value_;
        // Explicit floor division, independent of implementation-defined
        // right shifts of negative signed integers.
        std::int64_t q = prod / SCALE;
        if (prod < 0 && prod % SCALE != 0) --q;
        return from_raw(checked(q));
    }
};

struct SpatialCoord3D {
    std::uint32_t x, y, z;
    bool operator==(const SpatialCoord3D&) const = default;
};
inline void validate(const SpatialCoord3D& c) {
    if (c.x >= DIM || c.y >= DIM || c.z >= DIM)
        throw std::out_of_range("3D coordinate outside [0,999]");
}
inline std::uint64_t linear_index(const SpatialCoord3D& c) {
    validate(c);
    return (std::uint64_t(c.x) * DIM + c.y) * DIM + c.z;
}
struct SpatialCoordHash {
    std::size_t operator()(const SpatialCoord3D& c) const {
        return static_cast<std::size_t>(linear_index(c));
    }
};
struct BytecodePage {
    std::array<std::uint8_t, PAGE_BYTES> instructions{};
    std::array<Fixed16_16, LATENTS> latent_weights{};
    bool is_dirty = true;
    BytecodePage() {
        instructions.fill(0x90); // Data only; never executed as machine code.
        latent_weights.fill(Fixed16_16::from_ratio(1, 2));
    }
};
struct SystemStackState {
    Fixed16_16 psi_state = Fixed16_16::from_ratio(1, 1);
    Fixed16_16 phi_filter = Fixed16_16::from_ratio(985, 1000);
    Fixed16_16 lambda_constraint = Fixed16_16::from_ratio(15, 1000);
    Fixed16_16 omega_memory = Fixed16_16::from_ratio(777, 1000);
    Fixed16_16 theta_field = Fixed16_16::from_ratio(420, 1000);
};
inline Fixed16_16 bias_for(const SpatialCoord3D& c, std::size_t i) {
    // Bounded 8-bit spatial lattice bias in [0, 255/256].
    const std::uint32_t code = c.x ^ c.y ^ c.z ^
                               (static_cast<std::uint32_t>(i) * 131U);
    return Fixed16_16::from_raw(static_cast<std::int32_t>((code & 255U) * 256U));
}
inline std::uint32_t evolve_page(BytecodePage& p, const SpatialCoord3D& c,
                                const SystemStackState& stack) {
    const Fixed16_16 alpha = Fixed16_16::from_ratio(1, 8);
    std::uint32_t greatest = 0;
    for (std::size_t i = 0; i < LATENTS; ++i) {
        const auto before = p.latent_weights[i];
        // Fixed-point gradient descent:
        // w' = w - alpha*(phi*w - bias + lambda*(w-theta)).
        const auto gradient = before * stack.phi_filter - bias_for(c, i) +
                              (before - stack.theta_field) * stack.lambda_constraint;
        const auto after = before - gradient * alpha;
        p.latent_weights[i] = after;
        const std::int64_t diff = std::int64_t(after.raw()) - before.raw();
        greatest = std::max(greatest, static_cast<std::uint32_t>(diff < 0 ? -diff : diff));
    }
    p.is_dirty = greatest != 0;
    return greatest;
}
inline void evolve_exact_steps(BytecodePage& p, const SpatialCoord3D& c,
                               std::uint32_t rounds) {
    validate(c);
    if (rounds > MAX_ROUNDS) throw std::out_of_range("round count exceeds 512");
    const SystemStackState stack{};
    for (std::uint32_t r = 0; r < rounds; ++r) evolve_page(p, c, stack);
}

struct RunSummary {
    std::uint32_t performed = 0;
    std::uint32_t max_delta_raw = 0;
    std::size_t active_pages = 0;
    bool converged = false;
};

struct WavefrontSummary {
    std::uint32_t shells = 0;
    std::uint64_t unique_tiles = 0;
    std::uint64_t covered_voxels = 0;
    std::uint64_t parent_transfers = 0;
    std::uint32_t max_delta_raw = 0;
};

class DrMoagiSpatialEngine {
    std::unordered_map<SpatialCoord3D, std::unique_ptr<BytecodePage>,
                       SpatialCoordHash> memory_arena_;
    SystemStackState stack_;
    std::uint64_t iteration_counter_ = 0;
public:
    BytecodePage& page(const SpatialCoord3D& c) {
        validate(c);
        auto [it, inserted] = memory_arena_.try_emplace(c);
        if (inserted) it->second = std::make_unique<BytecodePage>();
        return *it->second;
    }
    RunSummary execute_iterations(const SpatialCoord3D& origin,
                                  std::uint32_t max_iterations) {
        if (origin.x >= DIM - 1 || origin.y >= DIM - 1 || origin.z >= DIM - 1)
            throw std::out_of_range("2x2x2 neighborhood exceeds 3D bounds");
        if (max_iterations > MAX_ROUNDS)
            throw std::out_of_range("iteration count exceeds 512");
        RunSummary result{};
        // All pages are present even if max_iterations == 0.
        for (std::uint32_t dx = 0; dx < 2; ++dx)
            for (std::uint32_t dy = 0; dy < 2; ++dy)
                for (std::uint32_t dz = 0; dz < 2; ++dz)
                    page({origin.x + dx, origin.y + dy, origin.z + dz});

        for (std::uint32_t r = 0; r < max_iterations; ++r) {
            std::uint32_t max_change = 0;
            for (std::uint32_t dx = 0; dx < 2; ++dx)
                for (std::uint32_t dy = 0; dy < 2; ++dy)
                    for (std::uint32_t dz = 0; dz < 2; ++dz) {
                        const SpatialCoord3D c{origin.x + dx, origin.y + dy,
                                               origin.z + dz};
                        max_change = std::max(max_change, evolve_page(page(c), c, stack_));
                    }
            stack_.psi_state = stack_.psi_state * stack_.phi_filter;
            const auto magnitude = Fixed16_16::from_raw(
                static_cast<std::int32_t>(std::min(max_change, SCALE)));
            stack_.omega_memory =
                stack_.omega_memory *
                    (Fixed16_16::from_ratio(1, 1) - stack_.lambda_constraint) +
                magnitude * stack_.lambda_constraint;
            ++iteration_counter_;
            result.performed++;
            result.max_delta_raw = max_change;
            result.converged = max_change <= 1;
            if (result.converged) break;
        }
        result.active_pages = memory_arena_.size();
        return result;
    }

    // 26-neighbor (Chebyshev metric) breadth-first wavefront in tile space.
    // Every reachable tile is visited exactly once. A deterministic parent
    // on the previous shell contributes one quarter of its latent field.
    // This is numerical state transfer, not learned neural inference.
    WavefrontSummary permeate_tiles(const SpatialCoord3D& voxel_origin,
                                    std::uint32_t max_shell_radius,
                                    std::size_t page_budget = 20000) {
        validate(voxel_origin);
        if (max_shell_radius >= 100)
            throw std::out_of_range("wavefront radius must be <= 99 tiles");
        if (!memory_arena_.empty())
            throw std::logic_error("wavefront requires a fresh engine");
        const SpatialCoord3D seed{voxel_origin.x / 10U,
                                  voxel_origin.y / 10U,
                                  voxel_origin.z / 10U};
        const auto lo = [](std::uint32_t v, std::uint32_t r) {
            return v > r ? v - r : 0U;
        };
        const auto hi = [](std::uint32_t v, std::uint32_t r) {
            return std::min(99U, v + r);
        };
        const std::uint64_t xcount = hi(seed.x, max_shell_radius) -
                                     lo(seed.x, max_shell_radius) + 1U;
        const std::uint64_t ycount = hi(seed.y, max_shell_radius) -
                                     lo(seed.y, max_shell_radius) + 1U;
        const std::uint64_t zcount = hi(seed.z, max_shell_radius) -
                                     lo(seed.z, max_shell_radius) + 1U;
        const std::uint64_t requested = xcount * ycount * zcount;
        if (page_budget == 0 || requested > page_budget)
            throw std::length_error("wavefront exceeds the page budget");
        const auto toward = [](std::uint32_t c, std::uint32_t s) {
            return c > s ? c - 1U : (c < s ? c + 1U : c);
        };
        const Fixed16_16 three_quarters = Fixed16_16::from_ratio(3, 4);
        const Fixed16_16 one_quarter = Fixed16_16::from_ratio(1, 4);
        WavefrontSummary result{};
        for (std::uint32_t r = 0; r <= max_shell_radius; ++r) {
            for (std::uint32_t x = lo(seed.x, r); x <= hi(seed.x, r); ++x)
                for (std::uint32_t y = lo(seed.y, r); y <= hi(seed.y, r); ++y)
                    for (std::uint32_t z = lo(seed.z, r); z <= hi(seed.z, r); ++z) {
                        const std::uint32_t dist = std::max({
                            x > seed.x ? x - seed.x : seed.x - x,
                            y > seed.y ? y - seed.y : seed.y - y,
                            z > seed.z ? z - seed.z : seed.z - z
                        });
                        if (dist != r) continue;
                        const SpatialCoord3D c{x,y,z};
                        auto& active = page(c);
                        if (r == 0) {
                            // Signal injection at source; all other weights are 0.5.
                            active.latent_weights[0] = Fixed16_16::from_ratio(1,1);
                        } else {
                            const SpatialCoord3D parent{
                                toward(x, seed.x), toward(y, seed.y),
                                toward(z, seed.z)
                            };
                            // Parent is guaranteed to be in shell r-1.
                            const auto& prior = page(parent);
                            for (std::size_t i = 0; i < LATENTS; ++i)
                                active.latent_weights[i] =
                                    active.latent_weights[i] * three_quarters +
                                    prior.latent_weights[i] * one_quarter;
                            ++result.parent_transfers;
                        }
                        result.max_delta_raw = std::max(
                            result.max_delta_raw, evolve_page(active, c, stack_));
                        ++result.unique_tiles;
                    }
            ++result.shells;
            ++iteration_counter_;
            // Global feedback advances once per wavefront shell, not once per voxel.
            stack_.psi_state = stack_.psi_state * stack_.phi_filter;
        }
        result.covered_voxels = result.unique_tiles * 1000ULL;
        if (result.unique_tiles != requested)
            throw std::logic_error("wavefront visited an unexpected tile count");
        return result;
    }

    const SystemStackState& stack() const { return stack_; }
    std::size_t active_pages() const { return memory_arena_.size(); }
};

using PageBytes = std::array<std::uint8_t, PAGE_BYTES>;
inline std::uint8_t mask_for(const BytecodePage& page, const SpatialCoord3D& c,
                             std::size_t i) {
    const std::uint32_t bits = static_cast<std::uint32_t>(
        page.latent_weights[i].raw());
    const std::uint32_t k = (bits >> 8U) ^ c.x ^ (c.y * 17U) ^
                            (c.z * 31U) ^ static_cast<std::uint32_t>(i);
    return static_cast<std::uint8_t>(k & 255U);
}
inline PageBytes transform(const PageBytes& input, const SpatialCoord3D& c,
                           std::uint32_t rounds, bool inverse) {
    BytecodePage model;
    evolve_exact_steps(model, c, rounds);
    PageBytes output{};
    for (std::size_t i = 0; i < LATENTS; ++i) {
        const std::size_t o = 4 * i;
        const std::uint8_t key = mask_for(model, c, i);
        if (!inverse) {
            output[o] = input[o] ^ key;
            for (std::size_t k = 1; k < 4; ++k)
                output[o + k] = input[o + k] ^ input[o + k - 1];
        } else {
            output[o] = input[o] ^ key;
            for (std::size_t k = 1; k < 4; ++k)
                output[o + k] = input[o + k] ^ output[o + k - 1];
        }
    }
    return output;
}
inline std::uint32_t crc32(const PageBytes& src) {
    std::uint32_t crc = 0xFFFFFFFFU;
    for (std::uint8_t b : src) {
        crc ^= b;
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1U) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
    }
    return ~crc;
}
inline void append_u32(std::vector<std::uint8_t>& v, std::uint32_t n) {
    for (int i = 0; i < 4; ++i) v.push_back(static_cast<std::uint8_t>(n >> (8 * i)));
}
inline std::uint32_t read_u32(const std::vector<std::uint8_t>& v, std::size_t off) {
    if (off + 4 > v.size()) throw std::runtime_error("truncated header");
    std::uint32_t n = 0;
    for (int i = 0; i < 4; ++i)
        n |= std::uint32_t(v[off + static_cast<std::size_t>(i)]) << (8 * i);
    return n;
}
inline std::vector<std::uint8_t> encode(const PageBytes& input,
                                       const SpatialCoord3D& c,
                                       std::uint32_t rounds) {
    const PageBytes payload = transform(input, c, rounds, false);
    std::vector<std::uint8_t> frame(MAGIC.begin(), MAGIC.end());
    append_u32(frame, c.x);
    append_u32(frame, c.y);
    append_u32(frame, c.z);
    append_u32(frame, rounds);
    append_u32(frame, crc32(input));
    append_u32(frame, crc32(payload));
    frame.insert(frame.end(), payload.begin(), payload.end());
    return frame; // 32-byte header + 4096-byte payload. No compression.
}
inline PageBytes decode(const std::vector<std::uint8_t>& frame) {
    if (frame.size() != HEADER_BYTES + PAGE_BYTES)
        throw std::runtime_error("invalid page frame length");
    if (!std::equal(MAGIC.begin(), MAGIC.end(), frame.begin()))
        throw std::runtime_error("invalid format magic");
    const SpatialCoord3D c{read_u32(frame, 8), read_u32(frame, 12),
                           read_u32(frame, 16)};
    const auto rounds = read_u32(frame, 20);
    validate(c);
    if (rounds > MAX_ROUNDS) throw std::runtime_error("unsupported round count");
    PageBytes payload{};
    std::copy(frame.begin() + HEADER_BYTES, frame.end(), payload.begin());
    if (crc32(payload) != read_u32(frame, 28))
        throw std::runtime_error("encoded payload CRC mismatch");
    const PageBytes decoded = transform(payload, c, rounds, true);
    if (crc32(decoded) != read_u32(frame, 24))
        throw std::runtime_error("decoded payload CRC mismatch");
    return decoded;
}
inline std::uint32_t parse_uint(const std::string& s, std::uint32_t max) {
    if (s.empty() || !std::all_of(s.begin(), s.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        })) throw std::invalid_argument("expected unsigned decimal integer");
    const std::uint64_t v = std::stoull(s);
    if (v > max) throw std::out_of_range("numeric argument out of range");
    return static_cast<std::uint32_t>(v);
}
inline std::vector<std::uint8_t> read_file(const std::string& path,
                                           std::size_t expected) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open input " + path);
    const auto length = f.tellg();
    if (length < 0 || static_cast<std::uint64_t>(length) != expected)
        throw std::runtime_error("input has unexpected byte length");
    f.seekg(0);
    std::vector<std::uint8_t> out(expected);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(expected));
    if (!f) throw std::runtime_error("input read failed");
    return out;
}
inline void write_file(const std::string& path,
                       const std::vector<std::uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot open output " + path);
    f.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
    if (!f) throw std::runtime_error("output write failed");
}
inline void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(std::string("selftest: ") + message);
}
inline void selftest() {
    check(Fixed16_16::from_ratio(-1, 2).raw() == -32768, "negative ratio");
    check((Fixed16_16::from_raw(-1) * Fixed16_16::from_raw(1)).raw() == -1,
          "negative multiplication floor");
    bool overflow = false;
    try {
        (void)(Fixed16_16::from_raw(std::numeric_limits<std::int32_t>::max()) +
               Fixed16_16::from_raw(1));
    } catch (const std::overflow_error&) { overflow = true; }
    check(overflow, "overflow must throw");
    check(linear_index({999,999,999}) == 999999999ULL, "billion-space index");
    DrMoagiSpatialEngine e;
    const auto summary = e.execute_iterations({500,500,500}, 256);
    check(summary.active_pages == 8, "2x2x2 neighborhood allocation");
    check(summary.performed <= 256, "bounded execution");
    check(summary.converged, "fixed-point stop criterion");
    PageBytes input{};
    std::uint32_t rng = 0x12345678U;
    for (int pattern = 0; pattern < 3; ++pattern) {
        for (std::size_t i = 0; i < input.size(); ++i) {
            rng ^= rng << 13U;
            rng ^= rng >> 17U;
            rng ^= rng << 5U;
            input[i] = pattern == 0 ? 0 : (pattern == 1 ? 
                       static_cast<std::uint8_t>(i) :
                       static_cast<std::uint8_t>(rng));
        }
        for (auto c : {SpatialCoord3D{0,0,0}, SpatialCoord3D{500,500,500},
                       SpatialCoord3D{999,999,999}}) {
            for (auto rounds : {0U, 1U, 30U, 128U}) {
                const auto frame = encode(input, c, rounds);
                check(frame.size() == HEADER_BYTES + PAGE_BYTES, "frame size");
                check(frame == encode(input, c, rounds), "replay exact");
                check(decode(frame) == input, "lossless round trip");
                auto corrupt = frame;
                corrupt.back() ^= 0x01;
                bool rejected = false;
                try { (void)decode(corrupt); }
                catch (const std::runtime_error&) { rejected = true; }
                check(rejected, "corruption rejected");
            }
        }
    }
    auto invalid = encode(input, {0,0,0}, 1);
    invalid[8] = 0xE8; invalid[9] = 0x03; // x=1000
    bool rejected = false;
    try { (void)decode(invalid); }
    catch (const std::out_of_range&) { rejected = true; }
    check(rejected, "out-of-range coordinate rejected");

    // Unique-tile wavefront: 3^3 at an interior seed, 2^3 at a boundary.
    DrMoagiSpatialEngine wave;
    const auto ws = wave.permeate_tiles({500,500,500}, 1, 27);
    check(ws.shells == 2 && ws.unique_tiles == 27, "interior wavefront count");
    check(ws.parent_transfers == 26 && ws.covered_voxels == 27000,
          "unique transfers and coverage");
    BytecodePage independent;
    (void)evolve_page(independent, {51,50,50}, SystemStackState{});
    check(wave.page({51,50,50}).latent_weights[0].raw() !=
          independent.latent_weights[0].raw(), "real neighbor field transfer");
    DrMoagiSpatialEngine replay_wave;
    const auto wr = replay_wave.permeate_tiles({500,500,500}, 1, 27);
    check(wr.unique_tiles == 27 &&
          replay_wave.page({51,50,50}).latent_weights[0].raw() ==
              wave.page({51,50,50}).latent_weights[0].raw(),
          "wavefront deterministic replay");
    DrMoagiSpatialEngine corner_wave;
    const auto cs = corner_wave.permeate_tiles({999,999,999}, 1, 8);
    check(cs.unique_tiles == 8 && cs.parent_transfers == 7,
          "corner clipping and unique tile visits");
    bool budget_rejected = false;
    try {
        DrMoagiSpatialEngine too_large;
        (void)too_large.permeate_tiles({500,500,500}, 2, 100);
    } catch (const std::length_error&) { budget_rejected = true; }
    check(budget_rejected, "pre-allocation page budget");

    std::cout << "PASS Q16.16 overflow, topology, convergence, 36 page "
                 "round trips, replay and corruption detection\n";
}
} // namespace jarvisx::spatial

int main(int argc, char** argv) {
    try {
        using namespace jarvisx::spatial;
        if (argc == 2 && std::string(argv[1]) == "selftest") {
            selftest();
            return 0;
        }
        if (argc >= 2 && std::string(argv[1]) == "demo" && argc <= 3) {
            const auto rounds = argc == 3 ? parse_uint(argv[2], MAX_ROUNDS) : 30U;
            DrMoagiSpatialEngine e;
            const auto r = e.execute_iterations({500,500,500}, rounds);
            std::cout << "steps=" << r.performed
                      << " active_pages=" << r.active_pages
                      << " max_delta_raw=" << r.max_delta_raw
                      << " converged=" << (r.converged ? "true" : "false")
                      << " psi=" << e.stack().psi_state.to_double()
                      << " omega=" << e.stack().omega_memory.to_double() << "\n";
            return 0;
        }

        if ((argc == 6 || argc == 7) &&
            std::string(argv[1]) == "permeate") {
            const SpatialCoord3D voxel{
                parse_uint(argv[2], DIM-1),
                parse_uint(argv[3], DIM-1),
                parse_uint(argv[4], DIM-1)
            };
            const auto radius = parse_uint(argv[5], 99U);
            const auto budget = argc == 7 ?
                parse_uint(argv[6], 1000000U) : 20000U;
            DrMoagiSpatialEngine e;
            const auto result = e.permeate_tiles(voxel, radius, budget);
            std::cout << "shells=" << result.shells
                      << " unique_tiles=" << result.unique_tiles
                      << " covered_voxels=" << result.covered_voxels
                      << " parent_transfers=" << result.parent_transfers
                      << " max_delta_raw=" << result.max_delta_raw
                      << " psi=" << e.stack().psi_state.to_double()
                      << "\n";
            return 0;
        }
        if (argc == 8 && std::string(argv[1]) == "encode") {
            const auto bytes = read_file(argv[2], PAGE_BYTES);
            PageBytes input{};
            std::copy(bytes.begin(), bytes.end(), input.begin());
            const SpatialCoord3D c{parse_uint(argv[4], DIM-1),
                                   parse_uint(argv[5], DIM-1),
                                   parse_uint(argv[6], DIM-1)};
            const auto rounds = parse_uint(argv[7], MAX_ROUNDS);
            write_file(argv[3], encode(input, c, rounds));
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "decode") {
            const auto frame = read_file(argv[2], HEADER_BYTES + PAGE_BYTES);
            const PageBytes output = decode(frame);
            write_file(argv[3], {output.begin(), output.end()});
            return 0;
        }
        std::cerr << "Usage:\n"
                  << "  spatial_q16 selftest\n"
                  << "  spatial_q16 demo [iterations 0..512]\n"
                  << "  spatial_q16 permeate X Y Z RADIUS_TILES [PAGE_BUDGET]\n"
                  << "  spatial_q16 encode INPUT_4096_BYTES OUTPUT.jxq16 X Y Z ROUNDS\n"
                  << "  spatial_q16 decode INPUT.jxq16 OUTPUT_4096_BYTES\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
