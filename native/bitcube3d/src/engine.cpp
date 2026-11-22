#include "bitcube3d/engine.hpp"
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace bitcube3d {

std::size_t BrickHash::operator()(BrickCoord const &c) const noexcept {
    // Coordinates are hashed individually; never flatten a 10^27-space to uint64.
    std::uint64_t h = 14695981039346656037ULL;
    for (std::uint32_t v : {c.x, c.y, c.z}) {
        h ^= static_cast<std::uint64_t>(v);
        h *= 1099511628211ULL;
        h ^= h >> 32;
    }
    return static_cast<std::size_t>(h);
}

bool Brick::get(unsigned i) const noexcept {
    return (data[i >> 3] >> (i & 7)) & 1U;
}
void Brick::set(unsigned i, bool value) noexcept {
    auto &b = data[i >> 3];
    const auto mask = static_cast<std::uint8_t>(1U << (i & 7));
    if (value) b = static_cast<std::uint8_t>(b | mask);
    else b = static_cast<std::uint8_t>(b & ~mask);
}
void Brick::flip(unsigned i) noexcept {
    data[i >> 3] ^= static_cast<std::uint8_t>(1U << (i & 7));
}
bool Brick::empty() const noexcept {
    return std::all_of(data.begin(), data.end(), [](std::uint8_t b) { return b == 0; });
}

void Engine::check_coord(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    if (x >= SIDE || y >= SIDE || z >= SIDE)
        throw std::out_of_range("voxel coordinate must be in [0, 999999999]");
}
BrickCoord Engine::brick_coord(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return {x / BRICK_EDGE, y / BRICK_EDGE, z / BRICK_EDGE};
}
unsigned Engine::bit_index(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return (x % BRICK_EDGE) + BRICK_EDGE * ((y % BRICK_EDGE) + BRICK_EDGE * (z % BRICK_EDGE));
}
bool Engine::contains_brick(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
    check_coord(x, y, z);
    return bricks_.find(brick_coord(x, y, z)) != bricks_.end();
}
bool Engine::get(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
    check_coord(x, y, z);
    auto it = bricks_.find(brick_coord(x, y, z));
    return it != bricks_.end() && it->second.get(bit_index(x, y, z));
}
void Engine::set(std::uint32_t x, std::uint32_t y, std::uint32_t z, bool value) {
    check_coord(x, y, z);
    BrickCoord key = brick_coord(x, y, z);
    if (!value) {
        auto it = bricks_.find(key);
        if (it != bricks_.end()) {
            it->second.set(bit_index(x, y, z), false);
            if (it->second.empty()) bricks_.erase(it);
        }
        return;
    }
    bricks_[key].set(bit_index(x, y, z), true);
}
void Engine::flip(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    set(x, y, z, !get(x, y, z));
}

void Engine::transform_pass(Brick &b, unsigned axis, unsigned stride) {
    const unsigned delta = axis == 0 ? stride : axis == 1 ? 16 * stride : 256 * stride;
    for (unsigned z = 0; z < 16; ++z) {
        for (unsigned y = 0; y < 16; ++y) {
            for (unsigned x = 0; x < 16; ++x) {
                const unsigned v = axis == 0 ? x : axis == 1 ? y : z;
                if ((v & stride) != 0) continue;
                const unsigned lo = x + 16 * y + 256 * z;
                if (b.get(lo)) b.flip(lo + delta);
            }
        }
    }
}
void Engine::transform_round(bool invert) {
    for (auto &kv : bricks_) {
        if (!invert) {
            for (unsigned axis = 0; axis < 3; ++axis)
                for (unsigned stride : {1U, 2U, 4U, 8U})
                    transform_pass(kv.second, axis, stride);
        } else {
            for (unsigned axis : {2U, 1U, 0U})
                for (unsigned stride : {8U, 4U, 2U, 1U})
                    transform_pass(kv.second, axis, stride);
        }
    }
}
void Engine::forward(std::uint32_t iterations) {
    for (std::uint32_t n = 0; n < iterations; ++n) transform_round(false);
}
void Engine::inverse(std::uint32_t iterations) {
    for (std::uint32_t n = 0; n < iterations; ++n) transform_round(true);
}
std::uint64_t Engine::active_bits() const noexcept {
    std::uint64_t count = 0;
    for (auto const &kv : bricks_) for (auto b : kv.second.data) {
#if defined(__GNUC__) || defined(__clang__)
        count += static_cast<unsigned>(__builtin_popcount(static_cast<unsigned>(b)));
#else
        while (b) { b &= static_cast<std::uint8_t>(b - 1); ++count; }
#endif
    }
    return count;
}
std::vector<std::pair<BrickCoord, Brick>> Engine::ordered_bricks() const {
    std::vector<std::pair<BrickCoord, Brick>> v;
    v.reserve(bricks_.size());
    for (const auto &kv : bricks_) v.push_back(kv);
    std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    return v;
}
void Engine::insert_brick(BrickCoord c, const Brick &brick) {
    if (c.x >= SIDE / BRICK_EDGE || c.y >= SIDE / BRICK_EDGE || c.z >= SIDE / BRICK_EDGE)
        throw std::runtime_error("brick coordinate outside virtual cube");
    if (brick.empty()) throw std::runtime_error("B3D1 forbids empty bricks");
    if (!bricks_.emplace(c, brick).second) throw std::runtime_error("duplicate brick coordinate");
}

static void put_u32(std::ostream &os, std::uint32_t v) {
    for (int j = 0; j < 4; ++j) os.put(static_cast<char>((v >> (j * 8)) & 255));
}
static void put_u64(std::ostream &os, std::uint64_t v) {
    for (int j = 0; j < 8; ++j) os.put(static_cast<char>((v >> (j * 8)) & 255));
}
static std::uint64_t read_le(std::istream &is, int n) {
    std::uint64_t result = 0;
    for (int j = 0; j < n; ++j) {
        int c = is.get();
        if (c == EOF) throw std::runtime_error("truncated B3D1 header or coordinate");
        result |= static_cast<std::uint64_t>(static_cast<unsigned>(c)) << (8 * j);
    }
    return result;
}
void save(const Snapshot &snap, const std::string &path) {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) throw std::runtime_error("cannot write: " + path);
    os.write("B3D1", 4);
    put_u32(os, SIDE);
    put_u32(os, snap.rounds);
    put_u64(os, static_cast<std::uint64_t>(snap.engine.active_bricks()));
    for (auto const &kv : snap.engine.ordered_bricks()) {
        put_u32(os, kv.first.x); put_u32(os, kv.first.y); put_u32(os, kv.first.z);
        os.write(reinterpret_cast<const char *>(kv.second.data.data()), BRICK_BYTES);
    }
    if (!os) throw std::runtime_error("write failed: " + path);
}
Snapshot load(const std::string &path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("cannot read: " + path);
    char magic[4];
    if (!is.read(magic, 4) || std::string(magic, 4) != "B3D1")
        throw std::runtime_error("not a B3D1 file");
    if (read_le(is, 4) != SIDE) throw std::runtime_error("unsupported virtual side length");
    Snapshot snap;
    snap.rounds = static_cast<std::uint32_t>(read_le(is, 4));
    const std::uint64_t count = read_le(is, 8);
    // Refuse impossible files before attempting allocations: a brick costs 524 bytes on disk.
    is.seekg(0, std::ios::end);
    const auto end = is.tellg();
    if (end < std::streamoff(20)) throw std::runtime_error("invalid file length");
    const auto payload = static_cast<std::uint64_t>(end - std::streamoff(20));
    if (count > payload / (12U + BRICK_BYTES) || payload != count * (12U + BRICK_BYTES))
        throw std::runtime_error("B3D1 byte length/brick count mismatch");
    is.seekg(20, std::ios::beg);
    BrickCoord previous{};
    bool have_previous = false;
    for (std::uint64_t i = 0; i < count; ++i) {
        BrickCoord c;
        c.x = static_cast<std::uint32_t>(read_le(is, 4));
        c.y = static_cast<std::uint32_t>(read_le(is, 4));
        c.z = static_cast<std::uint32_t>(read_le(is, 4));
        // Writers emit sorted bricks; reject noncanonical order in every reader.
        if (have_previous && !(previous < c))
            throw std::runtime_error("B3D1 brick coordinates must be strictly sorted");
        previous = c;
        have_previous = true;
        Brick b;
        if (!is.read(reinterpret_cast<char *>(b.data.data()), BRICK_BYTES))
            throw std::runtime_error("truncated B3D1 brick");
        snap.engine.insert_brick(c, b);
    }
    return snap;
}
} // namespace bitcube3d
