#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bitcube3d {

inline constexpr std::uint32_t SIDE = 1'000'000'000U;
inline constexpr unsigned BRICK_EDGE = 16;
inline constexpr unsigned BRICK_BITS = 16 * 16 * 16;
inline constexpr unsigned BRICK_BYTES = BRICK_BITS / 8;

struct BrickCoord {
    std::uint32_t x = 0, y = 0, z = 0;
    bool operator==(const BrickCoord &other) const noexcept {
        return x == other.x && y == other.y && z == other.z;
    }
    bool operator<(const BrickCoord &other) const noexcept {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct BrickHash {
    std::size_t operator()(BrickCoord const &c) const noexcept;
};

struct Brick {
    std::array<std::uint8_t, BRICK_BYTES> data{};
    bool get(unsigned bit) const noexcept;
    void set(unsigned bit, bool value) noexcept;
    void flip(unsigned bit) noexcept;
    bool empty() const noexcept;
    bool operator==(const Brick &rhs) const noexcept { return data == rhs.data; }
};

class Engine {
public:
    bool get(std::uint32_t x, std::uint32_t y, std::uint32_t z) const;
    void set(std::uint32_t x, std::uint32_t y, std::uint32_t z, bool value);
    void flip(std::uint32_t x, std::uint32_t y, std::uint32_t z);
    void forward(std::uint32_t iterations = 1);
    void inverse(std::uint32_t iterations = 1);

    std::size_t active_bricks() const noexcept { return bricks_.size(); }
    bool contains_brick(std::uint32_t x, std::uint32_t y, std::uint32_t z) const;
    std::uint64_t active_bits() const noexcept;
    std::uint64_t payload_bytes() const noexcept {
        return static_cast<std::uint64_t>(bricks_.size()) * BRICK_BYTES;
    }
    bool operator==(const Engine &other) const noexcept { return bricks_ == other.bricks_; }
    std::vector<std::pair<BrickCoord, Brick>> ordered_bricks() const;
    void insert_brick(BrickCoord c, const Brick &brick);

private:
    std::unordered_map<BrickCoord, Brick, BrickHash> bricks_;
    static void check_coord(std::uint32_t x, std::uint32_t y, std::uint32_t z);
    static BrickCoord brick_coord(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept;
    static unsigned bit_index(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept;
    static void transform_pass(Brick &b, unsigned axis, unsigned stride);
    void transform_round(bool invert);
};

// B3D1: magic[4], side:u32le, rounds:u32le, count:u64le,
//        repeated count times: brick_x:u32le, brick_y:u32le,
//        brick_z:u32le, packed_16x16x16_bits[512].
// Coordinates use x-fastest indexing, bit 0 = least significant bit.
// Only non-zero bricks are serialized; blocks are sorted by (x,y,z).
// File is LOSSLESS but has NO authentication, encryption, or checksum.
struct Snapshot {
    Engine engine;
    std::uint32_t rounds = 0;
};

void save(const Snapshot &snap, const std::string &path);
Snapshot load(const std::string &path);

} // namespace bitcube3d
