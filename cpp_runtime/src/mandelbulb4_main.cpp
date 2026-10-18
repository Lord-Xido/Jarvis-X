// Jarvis-X DM3D Mandelbulb four-way hierarchical lossless reference codec.
// v1: bounded 3D byte fields, alternating XY/YZ/ZX 4-child lifting,
// run-length zero / zigzag-ULEB residuals, raw fallback and FNV-1a integrity.
// No claim of a neural codec or of universally certified fractal distance.
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mb4 {
using Bytes = std::vector<std::uint8_t>;
using Grid = std::vector<std::int32_t>;
struct Shape { int x, y, z; };
struct Stage { Shape fine, coarse; Grid detail; };
constexpr std::size_t header_bytes = 17;
constexpr std::size_t max_archive_bytes = 32u * 1024u * 1024u;

void require(bool condition, const char* why) {
    if (!condition) throw std::runtime_error(why);
}
std::size_t count(Shape s) {
    return static_cast<std::size_t>(s.x) * static_cast<std::size_t>(s.y) * static_cast<std::size_t>(s.z);
}
std::size_t at(Shape s, int x, int y, int z) {
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(s.y)
           + static_cast<std::size_t>(y)) * static_cast<std::size_t>(s.x)
           + static_cast<std::size_t>(x);
}
int axis_value(Shape s, int axis) {
    return axis == 0 ? s.x : (axis == 1 ? s.y : s.z);
}
void set_axis(Shape& s, int axis, int v) {
    if (axis == 0) s.x = v;
    else if (axis == 1) s.y = v;
    else s.z = v;
}
std::array<int, 2> split_axes(unsigned level) {
    if (level % 3u == 0u) return {{0, 1}}; // XY
    if (level % 3u == 1u) return {{1, 2}}; // YZ
    return {{2, 0}};                        // ZX
}
bool split_possible(Shape s, unsigned level) {
    auto a = split_axes(level);
    return axis_value(s, a[0]) >= 2 && axis_value(s, a[1]) >= 2
        && axis_value(s, a[0]) % 2 == 0 && axis_value(s, a[1]) % 2 == 0;
}
Shape reduced(Shape s, unsigned level) {
    require(split_possible(s, level), "invalid four-way split");
    auto a = split_axes(level);
    set_axis(s, a[0], axis_value(s, a[0]) / 2);
    set_axis(s, a[1], axis_value(s, a[1]) / 2);
    return s;
}
unsigned levels_for(unsigned side) {
    require(side >= 4 && side <= 128 && (side & (side - 1u)) == 0u,
            "side must be a power of two from 4 to 128");
    Shape s{static_cast<int>(side), static_cast<int>(side), static_cast<int>(side)};
    unsigned levels = 0;
    while (split_possible(s, levels)) {
        s = reduced(s, levels);
        ++levels;
    }
    return levels;
}
std::array<int, 3> child_coords(int x, int y, int z, unsigned level, int child) {
    std::array<int, 3> c{{x, y, z}};
    const auto axes = split_axes(level);
    c[static_cast<std::size_t>(axes[0])] = 2 * c[static_cast<std::size_t>(axes[0])] + (child & 1);
    c[static_cast<std::size_t>(axes[1])] = 2 * c[static_cast<std::size_t>(axes[1])] + ((child >> 1) & 1);
    return c;
}
std::vector<Stage> forward(const Bytes& voxels, unsigned side, Grid& coarse) {
    const unsigned max_levels = levels_for(side);
    require(voxels.size() == static_cast<std::size_t>(side) * side * side, "wrong voxel count");
    Shape current{static_cast<int>(side), static_cast<int>(side), static_cast<int>(side)};
    Grid field(voxels.begin(), voxels.end());
    std::vector<Stage> stages;
    for (unsigned level = 0; level < max_levels; ++level) {
        const Shape next = reduced(current, level);
        Grid parent(count(next)), detail(3u * count(next));
        for (int z = 0; z < next.z; ++z)
            for (int y = 0; y < next.y; ++y)
                for (int x = 0; x < next.x; ++x) {
                    const std::size_t p = at(next, x, y, z);
                    const auto anchor = child_coords(x, y, z, level, 0);
                    const std::int32_t base = field[at(current, anchor[0], anchor[1], anchor[2])];
                    parent[p] = base;
                    for (int j = 1; j < 4; ++j) {
                        const auto c = child_coords(x, y, z, level, j);
                        detail[3u * p + static_cast<std::size_t>(j - 1)] =
                            field[at(current, c[0], c[1], c[2])] - base;
                    }
                }
        stages.push_back(Stage{current, next, std::move(detail)});
        field.swap(parent);
        current = next;
    }
    coarse = std::move(field);
    return stages;
}
Grid backward(const Grid& coarse, const std::vector<Stage>& stages) {
    Grid field = coarse;
    for (std::size_t i = stages.size(); i > 0; --i) {
        const std::size_t level = i - 1u;
        const Stage& stage = stages[level];
        require(field.size() == count(stage.coarse), "coarse shape mismatch");
        require(stage.detail.size() == 3u * count(stage.coarse), "detail shape mismatch");
        Grid finer(count(stage.fine));
        for (int z = 0; z < stage.coarse.z; ++z)
            for (int y = 0; y < stage.coarse.y; ++y)
                for (int x = 0; x < stage.coarse.x; ++x) {
                    const std::size_t p = at(stage.coarse, x, y, z);
                    for (int j = 0; j < 4; ++j) {
                        const auto c = child_coords(x, y, z, static_cast<unsigned>(level), j);
                        const std::int32_t value = field[p] +
                            (j ? stage.detail[3u * p + static_cast<std::size_t>(j - 1)] : 0);
                        require(value >= 0 && value <= 255, "invalid decoded voxel");
                        finer[at(stage.fine, c[0], c[1], c[2])] = value;
                    }
                }
        field.swap(finer);
    }
    return field;
}
void put_u16(Bytes& out, unsigned v) {
    out.push_back(static_cast<std::uint8_t>(v & 255u));
    out.push_back(static_cast<std::uint8_t>((v >> 8u) & 255u));
}
void put_u64(Bytes& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>(v & 255u));
        v >>= 8u;
    }
}
std::uint8_t get_byte(const Bytes& in, std::size_t& pos) {
    require(pos < in.size(), "truncated archive");
    return in[pos++];
}
unsigned get_u16(const Bytes& in, std::size_t& pos) {
    unsigned a = get_byte(in, pos);
    return a | (static_cast<unsigned>(get_byte(in, pos)) << 8u);
}
std::uint64_t get_u64(const Bytes& in, std::size_t& pos) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8u; ++i)
        value |= static_cast<std::uint64_t>(get_byte(in, pos)) << (8u * i);
    return value;
}
std::uint64_t checksum(const Bytes& input) {
    std::uint64_t h = UINT64_C(14695981039346656037);
    for (std::uint8_t v : input) {
        h ^= v;
        h *= UINT64_C(1099511628211);
    }
    return h;
}
void put_var(Bytes& out, std::uint32_t value) {
    do {
        std::uint8_t b = static_cast<std::uint8_t>(value & 127u);
        value >>= 7u;
        if (value) b = static_cast<std::uint8_t>(b | 128u);
        out.push_back(b);
    } while (value);
}
std::uint32_t get_var(const Bytes& in, std::size_t& pos) {
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift <= 28u; shift += 7u) {
        const std::uint8_t b = get_byte(in, pos);
        require(!(shift == 28u && (b & 240u) != 0u), "varint overflow");
        value |= static_cast<std::uint32_t>(b & 127u) << shift;
        if (!(b & 128u)) return value;
    }
    throw std::runtime_error("varint too long");
}
std::uint32_t zigzag(std::int32_t value) {
    const std::int64_t v = value;
    return static_cast<std::uint32_t>(v >= 0 ? 2 * v : -2 * v - 1);
}
std::int32_t unzigzag(std::uint32_t v) {
    const std::int64_t result = (v & 1u) ? -(static_cast<std::int64_t>(v >> 1u) + 1)
                                        : static_cast<std::int64_t>(v >> 1u);
    return static_cast<std::int32_t>(result);
}
void write_sequence(Bytes& out, const Grid& seq) {
    for (std::size_t i = 0; i < seq.size();) {
        if (seq[i] == 0) {
            std::size_t end = i + 1;
            while (end < seq.size() && seq[end] == 0 && end - i < UINT32_MAX) ++end;
            out.push_back(0);
            put_var(out, static_cast<std::uint32_t>(end - i));
            i = end;
        } else {
            out.push_back(1);
            put_var(out, zigzag(seq[i]));
            ++i;
        }
    }
}
Grid read_sequence(const Bytes& in, std::size_t& pos, std::size_t n) {
    Grid values;
    values.reserve(n);
    while (values.size() < n) {
        const std::uint8_t token = get_byte(in, pos);
        if (token == 0) {
            const std::uint32_t run = get_var(in, pos);
            require(run != 0 && run <= n - values.size(), "invalid zero run");
            values.insert(values.end(), run, 0);
        } else if (token == 1) {
            const std::int32_t v = unzigzag(get_var(in, pos));
            require(v != 0, "noncanonical zero literal");
            values.push_back(v);
        } else {
            throw std::runtime_error("invalid sequence token");
        }
    }
    return values;
}
Bytes archive_header(unsigned side, std::uint8_t mode, unsigned depth, const Bytes& original) {
    Bytes out{'M', 'B', '4', 'C', 1, mode};
    put_u16(out, side);
    out.push_back(static_cast<std::uint8_t>(depth));
    put_u64(out, checksum(original));
    return out;
}
#include "mandelbulb4_adaptive.inc"
#include "mandelbulb4_bounded.inc"

Bytes encode(const Bytes& voxels, unsigned side, bool force_hierarchy = false) {
    const unsigned depth = levels_for(side);
    require(voxels.size() == static_cast<std::size_t>(side) * side * side, "wrong voxel count");
    Bytes raw = archive_header(side, 0, 0, voxels);
    raw.insert(raw.end(), voxels.begin(), voxels.end());
    Grid coarse;
    const auto stages = forward(voxels, side, coarse);
    Bytes packed = archive_header(side, 1, depth, voxels);
    write_sequence(packed, coarse);
    for (auto iter = stages.rbegin(); iter != stages.rend(); ++iter)
        write_sequence(packed, iter->detail);
    return force_hierarchy || packed.size() < raw.size() ? packed : raw;
}
Bytes decode(const Bytes& in) {
    require(in.size() >= header_bytes && in.size() <= max_archive_bytes, "invalid archive size");
    std::size_t pos = 0;
    require(get_byte(in, pos) == 'M' && get_byte(in, pos) == 'B'
            && get_byte(in, pos) == '4' && get_byte(in, pos) == 'C', "invalid magic");
    require(get_byte(in, pos) == 1, "unsupported archive version");
    const unsigned mode = get_byte(in, pos);
    const unsigned side = get_u16(in, pos);
    const unsigned depth = get_byte(in, pos);
    const std::uint64_t expected = get_u64(in, pos);
    const unsigned max_depth = levels_for(side);
    Bytes reconstructed;
    if (mode == 0) {
        require(depth == 0, "invalid raw depth");
        const std::size_t n = static_cast<std::size_t>(side) * side * side;
        require(in.size() - pos == n, "raw archive length mismatch");
        reconstructed.assign(in.begin() + static_cast<std::ptrdiff_t>(pos), in.end());
        pos = in.size();
    } else if (mode == 1) {
        require(depth == max_depth, "invalid hierarchy depth");
        Shape fine{static_cast<int>(side), static_cast<int>(side), static_cast<int>(side)};
        std::vector<Stage> stages;
        for (unsigned i = 0; i < depth; ++i) {
            const Shape coarse_shape = reduced(fine, i);
            stages.push_back(Stage{fine, coarse_shape, Grid{}});
            fine = coarse_shape;
        }
        Grid coarse = read_sequence(in, pos, count(fine));
        for (std::size_t i = stages.size(); i > 0; --i)
            stages[i - 1u].detail = read_sequence(in, pos, 3u * count(stages[i - 1u].coarse));
        Grid out = backward(coarse, stages);
        reconstructed.reserve(out.size());
        for (std::int32_t v : out) {
            require(v >= 0 && v <= 255, "out-of-range voxel");
            reconstructed.push_back(static_cast<std::uint8_t>(v));
        }
    } else if (mode == 2) {
        require(depth == max_depth, "invalid adaptive hierarchy depth");
        reconstructed = decode_adaptive(in, pos, side);
    } else if (mode == 3) {
        require(depth == max_depth, "invalid bounded hierarchy depth");
        reconstructed = decode_bounded(in, pos, side);
    } else {
        throw std::runtime_error("unknown archive mode");
    }
    require(pos == in.size(), "trailing archive data");
    require(checksum(reconstructed) == expected, "checksum mismatch");
    return reconstructed;
}
std::uint8_t mandelbulb_sample(double cx, double cy, double cz, unsigned iterations) {
    double x = 0, y = 0, z = 0;
    for (unsigned k = 0; k < iterations; ++k) {
        const double r = std::sqrt(x*x + y*y + z*z);
        if (r > 4.0 || !std::isfinite(r))
            return static_cast<std::uint8_t>(255u * k / iterations);
        const double theta = r > 0.0 ? std::acos(std::max(-1.0, std::min(1.0, z/r))) : 0.0;
        const double phi = std::atan2(y, x);
        const double radius = std::pow(r, 8.0);
        const double sin_theta = std::sin(8.0 * theta);
        const double new_x = radius * sin_theta * std::cos(8.0 * phi) + cx;
        const double new_y = radius * sin_theta * std::sin(8.0 * phi) + cy;
        const double new_z = radius * std::cos(8.0 * theta) + cz;
        x = new_x; y = new_y; z = new_z;
    }
    return 255u;
}
Bytes generate(unsigned side, unsigned iterations) {
    levels_for(side);
    require(iterations >= 2 && iterations <= 64, "iterations must be 2..64");
    const std::size_t n = static_cast<std::size_t>(side) * side * side;
    Bytes voxels(n);
    for (unsigned z = 0; z < side; ++z)
        for (unsigned y = 0; y < side; ++y)
            for (unsigned x = 0; x < side; ++x) {
                const double cx = 3.0 * (static_cast<double>(x) + 0.5) / side - 1.5;
                const double cy = 3.0 * (static_cast<double>(y) + 0.5) / side - 1.5;
                const double cz = 3.0 * (static_cast<double>(z) + 0.5) / side - 1.5;
                voxels[at(Shape{static_cast<int>(side), static_cast<int>(side), static_cast<int>(side)},
                          static_cast<int>(x), static_cast<int>(y), static_cast<int>(z))] =
                    mandelbulb_sample(cx, cy, cz, iterations);
            }
    return voxels;
}
Bytes fixture_volume(unsigned side, unsigned iterations, const std::string& fixture) {
    if (fixture == "mandelbulb") return generate(side, iterations);
    const std::size_t n = static_cast<std::size_t>(side) * side * side;
    if (fixture == "uniform") return Bytes(n, 127);
    if (fixture == "noise") {
        Bytes output(n);
        std::uint32_t state = UINT32_C(0x1234abcd);
        for (std::uint8_t& b : output) {
            state = 1664525u * state + 1013904223u;
            b = static_cast<std::uint8_t>(state >> 24u);
        }
        return output;
    }
    throw std::runtime_error("unknown fixture: " + fixture);
}
Bytes read_file(const std::string& path, std::size_t limit) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(in), "cannot open input");
    const std::streamoff length = in.tellg();
    require(length >= 0 && static_cast<std::uint64_t>(length) <= limit, "input exceeds limit");
    in.seekg(0);
    Bytes result(static_cast<std::size_t>(length));
    if (!result.empty())
        require(static_cast<bool>(in.read(reinterpret_cast<char*>(result.data()), length)),
                "input read failure");
    return result;
}
void write_file(const std::string& path, const Bytes& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(out), "cannot open output");
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
    require(static_cast<bool>(out), "output write failure");
}
unsigned maximum_error(const Bytes& a, const Bytes& b) {
    require(a.size() == b.size(), "source and reconstruction lengths differ");
    unsigned error = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const unsigned x = a[i], y = b[i];
        error = std::max(error, x > y ? x-y : y-x);
    }
    return error;
}

void benchmark(const Bytes& voxels, unsigned side, const std::string& fixture,
               unsigned tolerance = 0) {
    using Clock = std::chrono::steady_clock;
    const char* labels[] = {"raw", "hierarchical", "adaptive", "bounded"};
    for (unsigned mode = 0; mode < (tolerance > 0 ? 4u : 3u); ++mode) {
        std::array<std::int64_t, 3> enc_us{}, dec_us{};
        std::size_t bytes = 0;
        AdaptiveStats stats{};
        BoundedStats bounded{};
        unsigned max_err = 0;
        for (unsigned repetition = 0; repetition < 3; ++repetition) {
            const auto start = Clock::now();
            Bytes encoded;
            if (mode == 0) {
                encoded = archive_header(side, 0, 0, voxels);
                encoded.insert(encoded.end(), voxels.begin(), voxels.end());
            } else if (mode == 1) {
                encoded = encode(voxels, side, true);
            } else if (mode == 2) {
                stats = AdaptiveStats{};
                encoded = encode_adaptive(voxels, side, &stats);
            } else {
                bounded = BoundedStats{};
                encoded = encode_bounded(voxels, side, tolerance, &bounded);
            }
            const auto encoded_at = Clock::now();
            const Bytes restored = decode(encoded);
            const auto decoded_at = Clock::now();
            max_err = maximum_error(restored, voxels);
            require(max_err <= (mode == 3 ? tolerance : 0u),
                    "benchmark CTR maximum-error bound failed");
            bytes = encoded.size();
            enc_us[repetition] = std::chrono::duration_cast<std::chrono::microseconds>(
                encoded_at - start).count();
            dec_us[repetition] = std::chrono::duration_cast<std::chrono::microseconds>(
                decoded_at - encoded_at).count();
        }
        std::sort(enc_us.begin(), enc_us.end());
        std::sort(dec_us.begin(), dec_us.end());
        std::cout << "MB4_BENCH fixture=" << fixture
                  << " side=" << side << " mode=" << labels[mode]
                  << " original_bytes=" << voxels.size()
                  << " archive_bytes=" << bytes
                  << " ratio=" << static_cast<double>(voxels.size()) / static_cast<double>(bytes)
                  << " median_encode_us=" << enc_us[1]
                  << " median_decode_us=" << dec_us[1]
                  << " max_error=" << max_err;
        if (mode == 2) {
            std::cout << " tree_nodes=" << stats.nodes
                      << " splits=" << stats.split_nodes
                      << " constant_leaves=" << stats.constant_leaves
                      << " raw_leaves=" << stats.raw_leaves
                      << " deepest_split=" << stats.deepest_split;
        }
        if (mode == 3) {
            std::cout << " tolerance=" << tolerance
                      << " tree_nodes=" << bounded.nodes
                      << " splits=" << bounded.splits
                      << " approximate_leaves=" << bounded.approximations
                      << " approximate_voxels=" << bounded.covered_approx_voxels;
        }
        std::cout << "\n";
    }
}
void self_test() {
    require(levels_for(16) == 6, "wrong level count");
    for (unsigned side : {4u, 8u, 16u}) {
        Bytes fixture = generate(side, 18);
        require(decode(encode(fixture, side, true)) == fixture, "Mandelbulb lossless roundtrip");
        require(decode(encode(fixture, side)) == fixture, "Mandelbulb auto mode");
        require(decode(encode_adaptive(fixture, side)) == fixture,
                "adaptive Mandelbulb roundtrip");
    }
    Bytes zeros(16u * 16u * 16u, 0);
    const Bytes packed = encode(zeros, 16);
    require(packed[5] == 1 && packed.size() < zeros.size(), "zero field should compress");
    require(decode(packed) == zeros, "zero field roundtrip");
    Bytes noisy(8u * 8u * 8u);
    std::uint32_t state = 0x1234abcd;
    for (std::uint8_t& value : noisy) {
        state = 1664525u * state + 1013904223u;
        value = static_cast<std::uint8_t>(state >> 24u);
    }
    // A synthetic plane-aligned interface must be partitioned into coarse constant blocks.
    Bytes boundary(16u * 16u * 16u);
    for (int z = 0; z < 16; ++z)
        for (int y = 0; y < 16; ++y)
            for (int x = 0; x < 16; ++x)
                boundary[at(Shape{16,16,16},x,y,z)] = x < 8 ? 11 : 200;
    AdaptiveStats boundary_stats{};
    const Bytes boundary_archive = encode_adaptive(boundary, 16, &boundary_stats);
    require(decode(boundary_archive) == boundary, "adaptive geometric interface roundtrip");
    require(boundary_stats.nodes == boundary_stats.split_nodes
            + boundary_stats.constant_leaves + boundary_stats.raw_leaves,
            "invalid adaptive node statistics");
    require(boundary_stats.split_nodes > 0 && boundary_stats.constant_leaves >= 2,
            "adaptive geometry did not choose coarse regions");
    require(boundary_archive.size() < boundary.size(), "adaptive interface compression failed");
    // Random data is not declared compressible: root literal fallback is optimal here.
    AdaptiveStats random_stats{};
    const Bytes random_archive = encode_adaptive(noisy, 8, &random_stats);
    require(decode(random_archive) == noisy, "adaptive noise roundtrip");
    require(random_stats.raw_leaves >= 1, "adaptive fallback not exercised");
    // Corrupt adaptive payload / depth / node type / suffix must be rejected.
    for (int mutation = 0; mutation < 4; ++mutation) {
        Bytes bad = boundary_archive;
        if (mutation == 0) bad[9] ^= 1u;                  // checksum
        else if (mutation == 1) bad[8] = 0;              // invalid depth
        else if (mutation == 2) bad[header_bytes] = 255; // unknown node tag
        else bad.push_back(0);                           // trailing content
        bool rejected_adaptive = false;
        try { (void)decode(bad); }
        catch (const std::runtime_error&) { rejected_adaptive = true; }
        require(rejected_adaptive, "malformed adaptive archive accepted");
    }
    // Exhaustive region range checking bounds every sampled voxel's error.
    Bytes near_uniform(16u*16u*16u);
    for (std::size_t i = 0; i < near_uniform.size(); ++i)
        near_uniform[i] = static_cast<std::uint8_t>(40 + (i % 5u));
    BoundedStats bounded_stats{};
    const Bytes bounded_archive = encode_bounded(near_uniform,16,2,&bounded_stats);
    const Bytes bounded_output = decode(bounded_archive);
    require(bounded_archive[5] == 3 && bounded_stats.approximations > 0,
            "bounded approximation not selected");
    require(maximum_error(near_uniform,bounded_output) == 2,
            "bounded field expected exact measured maximum error 2");
    require(bounded_stats.measured_max_error == 2,
            "bounded certificate statistics incorrect");
    // Tight tolerance 0 never loses bytes.
    require(decode(encode_bounded(near_uniform,16,0)) == near_uniform,
            "zero tolerance should be exact");
    // Falsified per-leaf reported range cannot exceed archive tolerance.
    Bytes invalid_envelope = bounded_archive;
    require(invalid_envelope[header_bytes+9u] == 3,
            "bounded test requires a representative root leaf");
    invalid_envelope[header_bytes+9u+3u] = 255;
    bool bad_envelope_rejected = false;
    try { (void)decode(invalid_envelope); }
    catch (const std::runtime_error&) { bad_envelope_rejected = true; }
    require(bad_envelope_rejected, "invalid tolerance envelope was accepted");
    Bytes truncated_bounded = bounded_archive;
    truncated_bounded.pop_back();
    bool truncated_bounded_rejected = false;
    try { (void)decode(truncated_bounded); }
    catch (const std::runtime_error&) { truncated_bounded_rejected = true; }
    require(truncated_bounded_rejected, "truncated bounded archive was accepted");
    Bytes shortened = boundary_archive;
    shortened.pop_back();
    bool shortened_rejected = false;
    try { (void)decode(shortened); }
    catch (const std::runtime_error&) { shortened_rejected = true; }
    require(shortened_rejected, "truncated adaptive archive accepted");
    const Bytes auto_archive = encode(noisy, 8);
    require(auto_archive[5] == 0, "incompressible source should use raw fallback");
    require(decode(auto_archive) == noisy, "raw fallback roundtrip");
    Bytes damaged = packed;
    damaged[9] ^= 1u;
    bool rejected = false;
    try { (void)decode(damaged); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "checksum corruption not rejected");
    damaged = packed;
    damaged.pop_back();
    rejected = false;
    try { (void)decode(damaged); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "truncated archive not rejected");
    damaged = packed;
    damaged.push_back(0);
    rejected = false;
    try { (void)decode(damaged); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "trailing bytes not rejected");
    std::cout << "MB4C_SELF_TEST PASS; exact and adaptive roundtrip; "
                 "cost-gated geometric branching; raw fallback; malformed archive rejection\n";
}
} // namespace mb4

int main(int argc, char** argv) {
    try {
        unsigned side = 32, iterations = 18;
        bool force_hierarchy = false, adaptive = false, auto_codec = false;
        bool test = false, run_benchmark = false;
        int bound = -1;
        std::string fixture = "mandelbulb";
        std::string input, archive = "mandelbulb4.mb4", decoded_path, decode_source;
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            const auto arg = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("missing option value");
                return argv[i];
            };
            if (option == "--side") side = static_cast<unsigned>(std::stoul(arg()));
            else if (option == "--iterations") iterations = static_cast<unsigned>(std::stoul(arg()));
            else if (option == "--input") input = arg();
            else if (option == "--decode-archive") decode_source = arg();
            else if (option == "--out") archive = arg();
            else if (option == "--decoded") decoded_path = arg();
            else if (option == "--force-hierarchy") force_hierarchy = true;
            else if (option == "--adaptive") adaptive = true;
            else if (option == "--auto-codec") auto_codec = true;
            else if (option == "--benchmark") run_benchmark = true;
            else if (option == "--bounded-error") bound = std::stoi(arg());
            else if (option == "--fixture") fixture = arg();
            else if (option == "--self-test") test = true;
            else throw std::runtime_error("unknown option: " + option);
        }
        if (test) { mb4::self_test(); return 0; }
        if (!decode_source.empty()) {
            mb4::require(!decoded_path.empty(), "decode mode requires --decoded output path");
            const mb4::Bytes file = mb4::read_file(decode_source, mb4::max_archive_bytes);
            const mb4::Bytes output = mb4::decode(file);
            mb4::write_file(decoded_path, output);
            std::cout << "MB4C verified=1 independent_decode=1 decoded_bytes="
                      << output.size() << " max_error=0\n";
            return 0;
        }
        mb4::levels_for(side);
        mb4::require(!(force_hierarchy && (adaptive || auto_codec || bound >= 0))
                     && !(adaptive && (auto_codec || bound >= 0))
                     && !(auto_codec && bound >= 0), "conflicting codec options");
        mb4::require(bound <= 64, "bounded tolerance must be 0..64");
        const mb4::Bytes voxels = input.empty()
            ? mb4::fixture_volume(side, iterations, fixture)
            : mb4::read_file(input, static_cast<std::size_t>(side) * side * side);
        mb4::require(voxels.size() == static_cast<std::size_t>(side) * side * side,
                     "source must contain exactly side cubed bytes");
        if (run_benchmark) {
            mb4::benchmark(voxels, side, input.empty() ? fixture : "input",
                            bound > 0 ? static_cast<unsigned>(bound) : 0u);
            return 0;
        }
        mb4::AdaptiveStats adaptive_stats{};
        mb4::Bytes packed;
        mb4::BoundedStats bounded_stats{};
        if (bound >= 0) packed = mb4::encode_bounded(voxels, side,
                                 static_cast<unsigned>(bound), &bounded_stats);
        else if (adaptive) packed = mb4::encode_adaptive(voxels, side, &adaptive_stats);
        else if (auto_codec) {
            packed = mb4::encode(voxels, side);
            mb4::Bytes candidate = mb4::encode_adaptive(voxels, side, &adaptive_stats);
            if (candidate.size() < packed.size()) packed.swap(candidate);
        } else packed = mb4::encode(voxels, side, force_hierarchy);
        const mb4::Bytes restored = mb4::decode(packed);
        const unsigned error = mb4::maximum_error(restored, voxels);
        mb4::require(error <= (bound >= 0 ? static_cast<unsigned>(bound) : 0u),
                     "CTR sampled-field maximum error exceeds declared tolerance");
        mb4::write_file(archive, packed);
        if (!decoded_path.empty()) mb4::write_file(decoded_path, restored);
        std::cout << "MB4C verified=1 side=" << side
                  << " depth=" << mb4::levels_for(side)
                  << " mode=" << (packed[5] == 3 ? "bounded" : packed[5] == 2 ? "adaptive" : packed[5] == 1 ? "fourway" : "raw")
                  << " original_bytes=" << voxels.size()
                  << " archive_bytes=" << packed.size()
                  << " ratio=" << static_cast<double>(voxels.size()) / static_cast<double>(packed.size())
                  << " max_error=" << error;
        if (packed[5] == 2)
            std::cout << " tree_nodes=" << adaptive_stats.nodes
                      << " splits=" << adaptive_stats.split_nodes
                      << " constant_leaves=" << adaptive_stats.constant_leaves
                      << " raw_leaves=" << adaptive_stats.raw_leaves;
        if (packed[5] == 3)
            std::cout << " bound=" << bound
                      << " approximate_leaves=" << bounded_stats.approximations
                      << " approximate_voxels=" << bounded_stats.covered_approx_voxels;
        std::cout << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "MB4C error: " << e.what() << "\n";
        return 1;
    }
}
