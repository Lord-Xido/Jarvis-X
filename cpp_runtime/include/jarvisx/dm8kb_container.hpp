#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace jarvisx::dm8kb {

constexpr std::size_t kContainerBytes = 8192U;
constexpr std::size_t kRegionCount = 10U;

enum class Region : std::uint8_t {
    Control = 0U,
    VclState = 1U,
    Theta = 2U,
    Masks = 3U,
    Omega = 4U,
    Microcode = 5U,
    Residual = 6U,
    Features = 7U,
    Shadow = 8U,
    Integrity = 9U,
};

struct RegionSpec {
    Region region;
    std::size_t offset;
    std::size_t size;
    const char* name;
};

constexpr std::array<RegionSpec, kRegionCount> kRegions{{
    {Region::Control, 0x0000U, 128U, "CONTROL"},
    {Region::VclState, 0x0080U, 512U, "VCL_STATE"},
    {Region::Theta, 0x0280U, 512U, "THETA"},
    {Region::Masks, 0x0480U, 128U, "MASKS"},
    {Region::Omega, 0x0500U, 512U, "OMEGA"},
    {Region::Microcode, 0x0700U, 1024U, "MICROCODE"},
    {Region::Residual, 0x0B00U, 2048U, "RESIDUAL"},
    {Region::Features, 0x1300U, 2048U, "FEATURES"},
    {Region::Shadow, 0x1B00U, 1024U, "SHADOW"},
    {Region::Integrity, 0x1F00U, 256U, "INTEGRITY"},
}};

constexpr std::size_t region_offset(Region region) noexcept {
    return kRegions[static_cast<std::size_t>(region)].offset;
}

constexpr std::size_t region_size(Region region) noexcept {
    return kRegions[static_cast<std::size_t>(region)].size;
}

constexpr bool layout_is_exact() noexcept {
    std::size_t cursor = 0U;
    for (const RegionSpec& spec : kRegions) {
        if (spec.offset != cursor) return false;
        cursor += spec.size;
    }
    return cursor == kContainerBytes;
}

static_assert(layout_is_exact(), "DM-vOmegaXi+ container map must total exactly 8192 bytes");

struct CommitReceipt {
    std::uint64_t prior_digest{};
    std::uint64_t candidate_digest{};
    std::uint64_t result_digest{};
    std::uint64_t microcode_digest{};
    std::uint64_t epoch{};
    std::uint16_t changed_regions{};
    std::uint32_t semantic_gap_q24{};
    std::uint32_t state_delta_q24{};
    bool accepted{};
};

inline std::uint64_t fnv1a64(const std::uint8_t* data, std::size_t size) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::size_t i = 0U; i < size; ++i) {
        hash ^= static_cast<std::uint64_t>(data[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

class Container {
public:
    Container() { reset(); }

    void reset() noexcept {
        bytes_.fill(std::uint8_t{0U});
        constexpr std::array<std::uint8_t, 8U> magic{{
            'D', 'M', '8', 'K', 'B', '1', 0U, 0U,
        }};
        std::copy(magic.begin(), magic.end(), bytes_.begin());
        write_u16_unchecked(8U, 1U);
        write_u16_unchecked(10U, static_cast<std::uint16_t>(kContainerBytes));
    }

    const std::array<std::uint8_t, kContainerBytes>& bytes() const noexcept { return bytes_; }
    std::array<std::uint8_t, kContainerBytes> snapshot() const noexcept { return bytes_; }

    void restore(const std::array<std::uint8_t, kContainerBytes>& snapshot) noexcept {
        bytes_ = snapshot;
    }

    std::uint8_t read(std::size_t offset) const {
        check_bounds(offset, 1U);
        return bytes_[offset];
    }

    void write(std::size_t offset, std::uint8_t value) {
        check_bounds(offset, 1U);
        bytes_[offset] = value;
    }

    void write_region(Region region, const std::uint8_t* data, std::size_t size) {
        if (size > region_size(region)) {
            throw std::out_of_range(std::string("DM8KB region overflow: ") +
                                    kRegions[static_cast<std::size_t>(region)].name);
        }
        const std::size_t offset = region_offset(region);
        std::fill(bytes_.begin() + static_cast<std::ptrdiff_t>(offset),
                  bytes_.begin() + static_cast<std::ptrdiff_t>(offset + region_size(region)),
                  std::uint8_t{0U});
        if (size != 0U) {
            std::copy_n(data, static_cast<std::ptrdiff_t>(size),
                        bytes_.begin() + static_cast<std::ptrdiff_t>(offset));
        }
    }

    void write_region(Region region, const std::vector<std::uint8_t>& values) {
        write_region(region, values.data(), values.size());
    }

    std::vector<std::uint8_t> read_region(Region region, std::size_t size) const {
        if (size > region_size(region)) throw std::out_of_range("DM8KB region read overflow");
        const std::size_t offset = region_offset(region);
        return std::vector<std::uint8_t>(
            bytes_.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes_.begin() + static_cast<std::ptrdiff_t>(offset + size));
    }

    void set_microcode(const std::vector<std::uint8_t>& program) {
        if (program.empty() || program.size() > region_size(Region::Microcode)) {
            throw std::invalid_argument("DM8KB microcode must fit in 1024 bytes and be non-empty");
        }
        write_region(Region::Microcode, program);
        write_u16(12U, static_cast<std::uint16_t>(program.size()));
    }

    std::size_t microcode_size() const noexcept { return static_cast<std::size_t>(read_u16_unchecked(12U)); }

    std::vector<std::uint8_t> microcode() const {
        const std::size_t size = microcode_size();
        if (size == 0U || size > region_size(Region::Microcode)) {
            throw std::runtime_error("DM8KB invalid microcode length");
        }
        return read_region(Region::Microcode, size);
    }

    void stage_shadow(const std::vector<std::uint8_t>& candidate) {
        if (candidate.size() > region_size(Region::Shadow)) {
            throw std::out_of_range("DM8KB shadow candidate exceeds 1024 bytes");
        }
        write_region(Region::Shadow, candidate);
        write_u16(14U, static_cast<std::uint16_t>(candidate.size()));
    }

    std::size_t shadow_size() const noexcept { return static_cast<std::size_t>(read_u16_unchecked(14U)); }

    std::uint64_t epoch() const noexcept { return read_u64_unchecked(16U); }

    void set_epoch(std::uint64_t value) noexcept { write_u64_unchecked(16U, value); }

    void increment_epoch() noexcept { set_epoch(epoch() + 1ULL); }

    std::uint64_t digest() const noexcept {
        return fnv1a64(bytes_.data(), region_offset(Region::Integrity));
    }

    std::uint64_t microcode_digest() const {
        const std::size_t size = microcode_size();
        if (size == 0U || size > region_size(Region::Microcode)) return 0ULL;
        return fnv1a64(bytes_.data() + region_offset(Region::Microcode), size);
    }

    void write_receipt(const CommitReceipt& receipt) noexcept {
        const std::size_t base = region_offset(Region::Integrity);
        std::fill(bytes_.begin() + static_cast<std::ptrdiff_t>(base), bytes_.end(), std::uint8_t{0U});
        write_u64_unchecked(base + 0U, receipt.prior_digest);
        write_u64_unchecked(base + 8U, receipt.candidate_digest);
        write_u64_unchecked(base + 16U, receipt.result_digest);
        write_u64_unchecked(base + 24U, receipt.microcode_digest);
        write_u64_unchecked(base + 32U, receipt.epoch);
        write_u16_unchecked(base + 40U, receipt.changed_regions);
        write_u32_unchecked(base + 42U, receipt.semantic_gap_q24);
        write_u32_unchecked(base + 46U, receipt.state_delta_q24);
        bytes_[base + 50U] = receipt.accepted ? 1U : 0U;
    }

    CommitReceipt receipt() const noexcept {
        const std::size_t base = region_offset(Region::Integrity);
        CommitReceipt result;
        result.prior_digest = read_u64_unchecked(base + 0U);
        result.candidate_digest = read_u64_unchecked(base + 8U);
        result.result_digest = read_u64_unchecked(base + 16U);
        result.microcode_digest = read_u64_unchecked(base + 24U);
        result.epoch = read_u64_unchecked(base + 32U);
        result.changed_regions = read_u16_unchecked(base + 40U);
        result.semantic_gap_q24 = read_u32_unchecked(base + 42U);
        result.state_delta_q24 = read_u32_unchecked(base + 46U);
        result.accepted = bytes_[base + 50U] != 0U;
        return result;
    }

    static std::uint16_t changed_region_bitmap(
        const std::array<std::uint8_t, kContainerBytes>& before,
        const std::array<std::uint8_t, kContainerBytes>& after) noexcept {
        std::uint16_t bitmap = 0U;
        for (std::size_t r = 0U; r < kRegions.size(); ++r) {
            const RegionSpec& spec = kRegions[r];
            bool changed = false;
            for (std::size_t i = 0U; i < spec.size; ++i) {
                if (before[spec.offset + i] != after[spec.offset + i]) {
                    changed = true;
                    break;
                }
            }
            if (changed) bitmap = static_cast<std::uint16_t>(bitmap | static_cast<std::uint16_t>(1U << r));
        }
        return bitmap;
    }

    static double normalized_delta(
        const std::array<std::uint8_t, kContainerBytes>& before,
        const std::array<std::uint8_t, kContainerBytes>& after) noexcept {
        std::size_t changed = 0U;
        for (std::size_t i = 0U; i < kContainerBytes; ++i) {
            if (before[i] != after[i]) ++changed;
        }
        return static_cast<double>(changed) / static_cast<double>(kContainerBytes);
    }

    static std::size_t torus_index(int x, int y, int z) noexcept {
        const auto wrap = [](int value) noexcept {
            constexpr int edge = 8;
            int result = value % edge;
            if (result < 0) result += edge;
            return static_cast<std::size_t>(result);
        };
        const std::size_t wx = wrap(x);
        const std::size_t wy = wrap(y);
        const std::size_t wz = wrap(z);
        return (wz * 8U + wy) * 8U + wx;
    }

private:
    std::array<std::uint8_t, kContainerBytes> bytes_{};

    static void check_bounds(std::size_t offset, std::size_t size) {
        if (offset > kContainerBytes || size > kContainerBytes - offset) {
            throw std::out_of_range("DM8KB container-local access outside 8192-byte boundary");
        }
    }

    void write_u16(std::size_t offset, std::uint16_t value) {
        check_bounds(offset, 2U);
        write_u16_unchecked(offset, value);
    }

    void write_u16_unchecked(std::size_t offset, std::uint16_t value) noexcept {
        bytes_[offset] = static_cast<std::uint8_t>(value & 0xFFU);
        bytes_[offset + 1U] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
    }

    std::uint16_t read_u16_unchecked(std::size_t offset) const noexcept {
        return static_cast<std::uint16_t>(bytes_[offset]) |
               static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes_[offset + 1U]) << 8U);
    }

    void write_u32_unchecked(std::size_t offset, std::uint32_t value) noexcept {
        for (std::size_t i = 0U; i < 4U; ++i) {
            bytes_[offset + i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFU);
        }
    }

    std::uint32_t read_u32_unchecked(std::size_t offset) const noexcept {
        std::uint32_t value = 0U;
        for (std::size_t i = 0U; i < 4U; ++i) {
            value |= static_cast<std::uint32_t>(bytes_[offset + i]) << (8U * i);
        }
        return value;
    }

    void write_u64_unchecked(std::size_t offset, std::uint64_t value) noexcept {
        for (std::size_t i = 0U; i < 8U; ++i) {
            bytes_[offset + i] = static_cast<std::uint8_t>((value >> (8U * i)) & 0xFFULL);
        }
    }

    std::uint64_t read_u64_unchecked(std::size_t offset) const noexcept {
        std::uint64_t value = 0ULL;
        for (std::size_t i = 0U; i < 8U; ++i) {
            value |= static_cast<std::uint64_t>(bytes_[offset + i]) << (8U * i);
        }
        return value;
    }
};

inline std::uint32_t q24(double value) noexcept {
    const double clamped = std::clamp(value, 0.0, 255.9999999403953552);
    return static_cast<std::uint32_t>(clamped * 16777216.0);
}

} // namespace jarvisx::dm8kb
