#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace dm3d {
namespace fs = std::filesystem;
constexpr uint64_t AXIS = 1000000000000ULL; // 1000 decimal GB of logical coordinate positions
constexpr uint32_t TILE_EDGE = 16;
constexpr size_t TILE_BYTES = 4096; // 16 x 16 x 16 bytes
using U128 = unsigned __int128; // GCC/Clang; portable 128-bit pair can replace this
struct Coord { uint64_t x, y, z; };
U128 flatten(Coord c);
Coord unflatten(U128 value);
std::string decimal(U128 n);
U128 parse_u128(const std::string& str);
Coord tile_origin(uint64_t tile_id);
// Append-only sparse voxel memory: unallocated coordinates implicitly contain zero.
void voxel_set(const fs::path& volume, Coord c, uint8_t value);
std::optional<uint8_t> voxel_get(const fs::path& volume, Coord c);

enum class Modality : uint32_t { Binary=0, Text=1, Image=2, Audio=3, Video=4, Document=5 };
std::string modality_name(Modality m);
Modality detect_modality(const fs::path& path, const std::vector<uint8_t>& sample);
struct Features {
    Modality modality = Modality::Binary;
    uint64_t bytes = 0, lines = 0, words = 0;
    double entropy_bits_per_byte = 0;
    std::string detail;
};
Features analyze_file(const fs::path& path);

uint64_t hash_bytes(const uint8_t* p, size_t n, uint64_t seed=14695981039346656037ULL);
uint64_t hash_file(const fs::path& path);
struct LatentResult {
    double change = 0, mean = 0;
    unsigned iterations = 0;
};
LatentResult inward_fold(const std::array<uint8_t,TILE_BYTES>& tile, unsigned iterations);
struct EncodedTile {
    uint64_t tile_id = 0;
    Coord origin{0,0,0};
    uint32_t valid = 0;
    uint8_t mode = 0; // 0 raw; 1..5 reversible 3D spatial predictors + zero RLE
    uint8_t tested = 0; // number of verified inward codec proposals
    uint64_t checksum = 0;
    LatentResult latent;
    std::vector<uint8_t> payload;
};
EncodedTile encode_tile(const std::array<uint8_t,TILE_BYTES>& tile,
                        uint32_t valid, uint64_t tile_id, unsigned iterations,
                        unsigned refine_passes=5, uint8_t preferred_mode=1);
std::array<uint8_t,TILE_BYTES> decode_tile(const EncodedTile& tile);

struct ArchiveInfo {
    Modality modality = Modality::Binary;
    uint64_t original_size = 0;
    uint64_t tiles = 0;
    uint64_t file_hash = 0;
    uint64_t stored_size = 0;
};
ArchiveInfo inspect_archive(const fs::path& archive);

struct Stats {
    ArchiveInfo archive;
    double mean_latent_change = 0;
    double mean_latent_value = 0;
    uint64_t original_bytes = 0, encoded_bytes = 0;
    unsigned threads = 1;
    unsigned fold_iterations = 8;
    unsigned refine_passes = 5;
    uint64_t candidate_trials = 0;
    uint64_t raw_fallback_tiles = 0;
    uint64_t saved_payload_bytes = 0;
};
Stats encode_file(const fs::path& source, const fs::path& dest,
                  unsigned threads=4, unsigned fold_iterations=8, unsigned refine_passes=5);
ArchiveInfo decode_file(const fs::path& archive, const fs::path& dest);
void verify_file(const fs::path& archive, const fs::path& original);
void run_selftest();
} // namespace dm3d
