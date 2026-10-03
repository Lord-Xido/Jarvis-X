// Dr Moagi TiB3D streaming processor
// Logical geometry: 8192 x 8192 x 16384 bytes = 2^40 bytes = 1 TiB.
// The 1000 GB/s figure is a measured target, never an assumed capability.

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace jarvisx::tib3d {

constexpr std::uint64_t NX = 8192;
constexpr std::uint64_t NY = 8192;
constexpr std::uint64_t NZ = 16384;
constexpr std::uint64_t PLANE_BYTES = NX * NY;
constexpr std::uint64_t TOTAL_BYTES = NX * NY * NZ;
constexpr std::uint64_t ONE_TIB = 1ull << 40;

static_assert(PLANE_BYTES == (64ull << 20));
static_assert(TOTAL_BYTES == ONE_TIB);

struct Coord3D {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
};

[[nodiscard]] constexpr std::uint64_t xyz_to_linear(
    std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept {
    return std::uint64_t{x} + NX * (std::uint64_t{y} + NY * std::uint64_t{z});
}

[[nodiscard]] constexpr Coord3D linear_to_xyz(std::uint64_t address) noexcept {
    Coord3D c{};
    c.z = static_cast<std::uint32_t>(address / (NX * NY));
    address %= (NX * NY);
    c.y = static_cast<std::uint32_t>(address / NX);
    c.x = static_cast<std::uint32_t>(address % NX);
    return c;
}

[[nodiscard]] constexpr bool valid_coord(const Coord3D& c) noexcept {
    return c.x < NX && c.y < NY && c.z < NZ;
}

[[nodiscard]] inline std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

class AlignedPlane {
public:
    AlignedPlane() {
        constexpr std::size_t alignment = 64;
#if defined(_MSC_VER)
        ptr_ = static_cast<std::uint8_t*>(_aligned_malloc(PLANE_BYTES, alignment));
        if (ptr_ == nullptr) {
            throw std::bad_alloc{};
        }
#else
        ptr_ = static_cast<std::uint8_t*>(std::aligned_alloc(alignment, PLANE_BYTES));
        if (ptr_ == nullptr) {
            throw std::bad_alloc{};
        }
#endif
        auto* q = reinterpret_cast<std::uint64_t*>(ptr_);
        const auto n = static_cast<std::size_t>(PLANE_BYTES / sizeof(std::uint64_t));
        for (std::size_t i = 0; i < n; ++i) {
            q[i] = splitmix64(static_cast<std::uint64_t>(i));
        }
    }

    ~AlignedPlane() {
#if defined(_MSC_VER)
        _aligned_free(ptr_);
#else
        std::free(ptr_);
#endif
    }

    AlignedPlane(const AlignedPlane&) = delete;
    AlignedPlane& operator=(const AlignedPlane&) = delete;

    [[nodiscard]] std::uint8_t* data() noexcept { return ptr_; }

private:
    std::uint8_t* ptr_{nullptr};
};

enum class Kernel { Xor, Checksum };

struct Config {
    unsigned threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::uint32_t passes = 1;
    Kernel kernel = Kernel::Xor;
    double target_gbps = 1000.0;
    std::uint64_t task_limit = 64;
    bool full_sweep = false;
    bool self_test = false;
    bool json = false;
};

struct Stats {
    double seconds{};
    double payload_gbps{};
    double payload_gibps{};
    std::uint64_t bytes{};
    std::uint64_t tasks{};
    std::uint64_t digest{};
    bool target_met{};
};

inline void xor_bytes(std::uint8_t* p, std::size_t n, std::uint64_t key64) noexcept {
    std::size_t i = 0;
#if defined(__AVX512F__)
    const __m512i key = _mm512_set1_epi64(static_cast<long long>(key64));
    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(p + i));
        v = _mm512_xor_si512(v, key);
        _mm512_storeu_si512(reinterpret_cast<void*>(p + i), v);
    }
#elif defined(__AVX2__)
    const __m256i key = _mm256_set1_epi64x(static_cast<long long>(key64));
    for (; i + 32 <= n; i += 32) {
        __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i));
        v = _mm256_xor_si256(v, key);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(p + i), v);
    }
#endif
    for (; i + sizeof(std::uint64_t) <= n; i += sizeof(std::uint64_t)) {
        std::uint64_t value{};
        std::memcpy(&value, p + i, sizeof(value));
        value ^= key64;
        std::memcpy(p + i, &value, sizeof(value));
    }
    if (i < n) {
        const auto* key = reinterpret_cast<const std::uint8_t*>(&key64);
        for (std::size_t lane = 0; i < n; ++i, ++lane) {
            p[i] ^= key[lane & 7u];
        }
    }
}

inline void xor_plane(std::uint8_t* p, std::uint64_t plane_index) noexcept {
    xor_bytes(p, static_cast<std::size_t>(PLANE_BYTES),
              splitmix64(plane_index ^ 0xD3A59B17C4E26F01ull));
}

[[nodiscard]] inline std::uint64_t checksum_plane(
    const std::uint8_t* p, std::uint64_t plane_index) noexcept {
    const auto* q = reinterpret_cast<const std::uint64_t*>(p);
    const auto n = static_cast<std::size_t>(PLANE_BYTES / sizeof(std::uint64_t));

    std::uint64_t acc0 = splitmix64(plane_index);
    std::uint64_t acc1 = 0x6a09e667f3bcc909ull;
    std::uint64_t acc2 = 0xbb67ae8584caa73bull;
    std::uint64_t acc3 = 0x3c6ef372fe94f82bull;

    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc0 ^= q[i + 0];
        acc1 += q[i + 1];
        acc2 ^= (q[i + 2] << 13) | (q[i + 2] >> 51);
        acc3 += q[i + 3] * 0x9e3779b97f4a7c15ull;
    }
    for (; i < n; ++i) {
        acc0 ^= q[i];
    }
    return splitmix64(acc0 ^ acc1 ^ acc2 ^ acc3);
}

[[nodiscard]] bool self_test() {
    constexpr Coord3D probes[] = {
        {0, 0, 0},
        {8191, 0, 0},
        {0, 8191, 0},
        {0, 0, 16383},
        {8191, 8191, 16383},
        {17, 4096, 8192},
    };

    for (const auto& p : probes) {
        if (!valid_coord(p)) {
            return false;
        }
        const auto recovered = linear_to_xyz(xyz_to_linear(p.x, p.y, p.z));
        if (recovered.x != p.x || recovered.y != p.y || recovered.z != p.z) {
            return false;
        }
    }

    if (xyz_to_linear(8191, 8191, 16383) != TOTAL_BYTES - 1) {
        return false;
    }

    std::vector<std::uint8_t> block(4096);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<std::uint8_t>(
            splitmix64(static_cast<std::uint64_t>(i)) & 0xffu);
    }
    const auto original = block;
    constexpr std::uint64_t key = 0x0123456789ABCDEFull;
    xor_bytes(block.data(), block.size(), key);
    if (block == original) {
        return false;
    }
    xor_bytes(block.data(), block.size(), key);
    return block == original;
}

[[nodiscard]] Stats run(const Config& cfg) {
    if (cfg.threads == 0) {
        throw std::invalid_argument("threads must be >= 1");
    }
    if (cfg.passes == 0) {
        throw std::invalid_argument("passes must be >= 1");
    }

    const std::uint64_t max_tasks = NZ * std::uint64_t{cfg.passes};
    const std::uint64_t selected_tasks = cfg.full_sweep
        ? max_tasks
        : std::min(max_tasks, std::max<std::uint64_t>(1, cfg.task_limit));
    const std::uint64_t logical_bytes = selected_tasks * PLANE_BYTES;

    std::atomic<std::uint64_t> next_task{0};
    std::atomic<std::uint64_t> global_digest{0};
    std::barrier sync_point(static_cast<std::ptrdiff_t>(cfg.threads + 1));
    std::vector<std::jthread> pool;
    pool.reserve(cfg.threads);

    for (unsigned tid = 0; tid < cfg.threads; ++tid) {
        pool.emplace_back([&, tid] {
            AlignedPlane plane;
            std::uint64_t local_digest = splitmix64(static_cast<std::uint64_t>(tid));
            sync_point.arrive_and_wait();

            for (;;) {
                const auto task = next_task.fetch_add(1, std::memory_order_relaxed);
                if (task >= selected_tasks) {
                    break;
                }
                const auto z = task % NZ;
                if (cfg.kernel == Kernel::Xor) {
                    xor_plane(plane.data(), z);
                    const auto* words =
                        reinterpret_cast<const std::uint64_t*>(plane.data());
                    local_digest ^= words[static_cast<std::size_t>(z & 1023u)];
                } else {
                    local_digest ^= checksum_plane(plane.data(), z);
                }
            }
            global_digest.fetch_xor(local_digest, std::memory_order_relaxed);
        });
    }

    sync_point.arrive_and_wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (auto& t : pool) {
        if (t.joinable()) {
            t.join();
        }
    }
    const auto t1 = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    Stats s{};
    s.seconds = seconds;
    s.bytes = logical_bytes;
    s.tasks = selected_tasks;
    s.payload_gbps = (static_cast<double>(logical_bytes) / seconds) / 1e9;
    s.payload_gibps = (static_cast<double>(logical_bytes) / seconds) /
                      static_cast<double>(1ull << 30);
    s.digest = global_digest.load(std::memory_order_relaxed);
    s.target_met = s.payload_gbps >= cfg.target_gbps;
    return s;
}

[[nodiscard]] Kernel parse_kernel(const std::string& s) {
    if (s == "xor") {
        return Kernel::Xor;
    }
    if (s == "checksum") {
        return Kernel::Checksum;
    }
    throw std::invalid_argument("unknown kernel: " + s);
}

[[nodiscard]] Config parse_args(int argc, char** argv) {
    Config c{};
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* option) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string(option) + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--threads") {
            c.threads = static_cast<unsigned>(
                std::stoul(require_value("--threads")));
        } else if (arg == "--passes") {
            c.passes = static_cast<std::uint32_t>(
                std::stoul(require_value("--passes")));
        } else if (arg == "--kernel") {
            c.kernel = parse_kernel(require_value("--kernel"));
        } else if (arg == "--target-gbps") {
            c.target_gbps = std::stod(require_value("--target-gbps"));
        } else if (arg == "--task-limit") {
            c.task_limit = std::stoull(require_value("--task-limit"));
        } else if (arg == "--full-sweep") {
            c.full_sweep = true;
        } else if (arg == "--self-test") {
            c.self_test = true;
        } else if (arg == "--json") {
            c.json = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "DrMoagi TiB3D streaming processor\n"
                << "  --threads N       worker threads (bounded default <= 8)\n"
                << "  --passes N        logical sweep count\n"
                << "  --kernel K        xor|checksum\n"
                << "  --task-limit N    process N x 64 MiB planes (default 64)\n"
                << "  --full-sweep      process all 16,384 planes per pass\n"
                << "  --target-gbps N   measured target (default 1000)\n"
                << "  --self-test       verify geometry and reversible XOR\n"
                << "  --json            emit machine-readable result\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    return c;
}

void print_geometry() {
    std::cout
        << "Geometry\n"
        << "  X: " << NX << " bytes\n"
        << "  Y: " << NY << " rows\n"
        << "  Z: " << NZ << " planes\n"
        << "  Plane: " << (PLANE_BYTES >> 20) << " MiB\n"
        << "  Volume: " << TOTAL_BYTES << " bytes = 1 TiB\n"
        << "  Address: A(x,y,z) = x + 8192*(y + 8192*z)\n\n";
}

void print_json(const Config& cfg, const Stats& s) {
    std::cout
        << "{\"geometry\":{\"x\":8192,\"y\":8192,\"z\":16384,\"bytes\":"
        << TOTAL_BYTES << "},\"execution\":{\"threads\":" << cfg.threads
        << ",\"passes\":" << cfg.passes << ",\"tasks\":" << s.tasks
        << ",\"kernel\":\"" << (cfg.kernel == Kernel::Xor ? "xor" : "checksum")
        << "\"},\"result\":{\"seconds\":" << std::setprecision(9) << s.seconds
        << ",\"payload_gbps\":" << s.payload_gbps
        << ",\"payload_gibps\":" << s.payload_gibps
        << ",\"target_gbps\":" << cfg.target_gbps
        << ",\"target_met\":" << (s.target_met ? "true" : "false")
        << ",\"digest\":\"0x" << std::hex << s.digest << std::dec << "\"}}\n";
}

} // namespace jarvisx::tib3d

int main(int argc, char** argv) {
    try {
        using namespace jarvisx::tib3d;
        const Config cfg = parse_args(argc, argv);

        if (cfg.self_test) {
            const bool ok = self_test();
            std::cout << (ok ? "TiB3D self-test: PASS\n" : "TiB3D self-test: FAIL\n");
            return ok ? 0 : 2;
        }

        const Stats stats = run(cfg);
        if (cfg.json) {
            print_json(cfg, stats);
            return 0;
        }

        print_geometry();
        std::cout
            << "Execution\n"
            << "  Threads: " << cfg.threads << '\n'
            << "  Passes: " << cfg.passes << '\n'
            << "  Tasks: " << stats.tasks << " / "
            << (NZ * std::uint64_t{cfg.passes}) << '\n'
            << "  Kernel: "
            << (cfg.kernel == Kernel::Xor ? "xor (reversible)" : "checksum") << '\n'
            << "  Target: " << std::fixed << std::setprecision(2)
            << cfg.target_gbps << " GB/s\n\n"
            << "Result\n"
            << "  Payload bytes: " << stats.bytes << '\n'
            << "  Time: " << std::setprecision(6) << stats.seconds << " s\n"
            << "  Payload: " << std::setprecision(2) << stats.payload_gbps << " GB/s\n"
            << "  Payload: " << stats.payload_gibps << " GiB/s\n"
            << "  Digest: 0x" << std::hex << stats.digest << std::dec << '\n'
            << "  Target status: " << (stats.target_met ? "MET" : "NOT MET") << '\n';

        if (cfg.kernel == Kernel::Xor) {
            std::cout
                << "\nIn-place XOR reads and writes each payload byte; approximate "
                   "memory-fabric traffic is ~2x payload throughput.\n";
        }
        if (!cfg.full_sweep) {
            std::cout
                << "Bounded mode is active. Use --full-sweep for the complete "
                   "1 TiB logical traversal.\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
