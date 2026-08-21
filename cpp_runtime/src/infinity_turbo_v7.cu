#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dm3d_infinity_turbo {

using fixed_t = std::int32_t;
using wide_t = std::int64_t;

constexpr int SHIFT = 16;
constexpr fixed_t ONE = fixed_t{1} << SHIFT;
constexpr fixed_t HALF = ONE / 2;
constexpr fixed_t THREE = 3 * ONE;
constexpr std::size_t D_IN = 128;
constexpr std::size_t D_LAT = 48;
constexpr std::size_t D_VOL = 64;
constexpr std::size_t D_VOXELS = D_VOL * D_VOL * D_VOL;
constexpr std::size_t MAX_BATCH = 256;
constexpr int MAX_RADIUS = 4;
constexpr double PI = 3.14159265358979323846;

__host__ __device__ inline fixed_t sat64(wide_t x) {
    if (x > static_cast<wide_t>(INT32_MAX)) return INT32_MAX;
    if (x < static_cast<wide_t>(INT32_MIN)) return INT32_MIN;
    return static_cast<fixed_t>(x);
}

__host__ __device__ inline fixed_t q_add(fixed_t a, fixed_t b) {
    return sat64(static_cast<wide_t>(a) + static_cast<wide_t>(b));
}

__host__ __device__ inline fixed_t q_sub(fixed_t a, fixed_t b) {
    return sat64(static_cast<wide_t>(a) - static_cast<wide_t>(b));
}

__host__ __device__ inline fixed_t q_mul(fixed_t a, fixed_t b) {
    return sat64((static_cast<wide_t>(a) * static_cast<wide_t>(b)) >> SHIFT);
}

__host__ __device__ inline fixed_t q_div(fixed_t a, fixed_t b) {
    if (b == 0) return a >= 0 ? INT32_MAX : INT32_MIN;
    return sat64((static_cast<wide_t>(a) << SHIFT) / static_cast<wide_t>(b));
}

__host__ __device__ inline fixed_t q_tanh(fixed_t x) {
    if (x >= THREE) return ONE;
    if (x <= -THREE) return -ONE;
    const fixed_t x2 = q_mul(x, x);
    const fixed_t c27 = 27 * ONE;
    const fixed_t c9 = 9 * ONE;
    const fixed_t num = q_mul(x, q_add(c27, x2));
    const fixed_t den = q_add(c27, q_mul(c9, x2));
    fixed_t y = q_div(num, den);
    if (y > ONE) y = ONE;
    if (y < -ONE) y = -ONE;
    return y;
}

inline fixed_t q_from_double(double x) {
    const double scaled = x * static_cast<double>(ONE);
    if (scaled > static_cast<double>(INT32_MAX)) return INT32_MAX;
    if (scaled < static_cast<double>(INT32_MIN)) return INT32_MIN;
    return static_cast<fixed_t>(std::llround(scaled));
}

inline double q_to_double(fixed_t x) {
    return static_cast<double>(x) / static_cast<double>(ONE);
}

inline void cuda_check(cudaError_t status, const char* expr,
                       const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") +
                                 cudaGetErrorString(status) + " | " + expr +
                                 " @ " + file + ":" + std::to_string(line));
    }
}

#define DM_CUDA_CHECK(expr) \
    ::dm3d_infinity_turbo::cuda_check((expr), #expr, __FILE__, __LINE__)

struct Config {
    int batch = 8;
    int radius = 2;
    int recursion_depth = 3;
    int cycles = 5;
    fixed_t feedback_gain = q_from_double(0.05);
    std::uint32_t seed = 0xD34D7001u;

    void validate() const {
        if (batch < 1 || batch > static_cast<int>(MAX_BATCH)) {
            throw std::invalid_argument("batch must be in [1,256]");
        }
        if (radius < 1 || radius > MAX_RADIUS) {
            throw std::invalid_argument("radius must be in [1,4]");
        }
        if (recursion_depth < 1 || recursion_depth > 8) {
            throw std::invalid_argument("recursion depth must be in [1,8]");
        }
        if (cycles < 1 || cycles > 100000) {
            throw std::invalid_argument("cycles must be in [1,100000]");
        }
    }
};

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { reset(); }

    void allocate(std::size_t count) {
        reset();
        if (count == 0) return;
        DM_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_),
                                 count * sizeof(T)));
        count_ = count;
    }
    void reset() noexcept {
        if (ptr_) cudaFree(ptr_);
        ptr_ = nullptr;
        count_ = 0;
    }
    T* get() noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return count_; }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() { DM_CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }
private:
    cudaStream_t stream_{};
};

class Event {
public:
    Event() { DM_CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { if (event_) cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t get() const noexcept { return event_; }
private:
    cudaEvent_t event_{};
};

__global__ void dense_tanh_kernel(const fixed_t* input,
                                  fixed_t* output,
                                  const fixed_t* weights,
                                  const fixed_t* biases,
                                  int batch,
                                  int in_dim,
                                  int out_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / out_dim;
    const int o = idx % out_dim;
    if (b >= batch) return;

    wide_t acc = biases[o];
    for (int j = 0; j < in_dim; ++j) {
        acc += static_cast<wide_t>(
            q_mul(input[b * in_dim + j], weights[o * in_dim + j]));
    }
    output[b * out_dim + o] = q_tanh(sat64(acc));
}

__global__ void lift_state_to_volume_kernel(const fixed_t* state,
                                            fixed_t* volume,
                                            int batch,
                                            int voxels) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / voxels;
    const int v = idx % voxels;
    if (b >= batch) return;
    volume[b * voxels + v] =
        state[b * static_cast<int>(D_IN) + (v % static_cast<int>(D_IN))];
}

__global__ void process_volume_3d_kernel(const fixed_t* input,
                                         fixed_t* output,
                                         const fixed_t* stencil,
                                         int batch,
                                         int radius) {
    const int voxels = static_cast<int>(D_VOXELS);
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / voxels;
    const int v = idx % voxels;
    if (b >= batch) return;

    const int x = v / static_cast<int>(D_VOL * D_VOL);
    const int y = (v / static_cast<int>(D_VOL)) % static_cast<int>(D_VOL);
    const int z = v % static_cast<int>(D_VOL);

    wide_t acc = 0;
    int k = 0;
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dz = -radius; dz <= radius; ++dz, ++k) {
                const int nx =
                    (x + dx + static_cast<int>(D_VOL)) % static_cast<int>(D_VOL);
                const int ny =
                    (y + dy + static_cast<int>(D_VOL)) % static_cast<int>(D_VOL);
                const int nz =
                    (z + dz + static_cast<int>(D_VOL)) % static_cast<int>(D_VOL);
                const int nv = nx * static_cast<int>(D_VOL * D_VOL) +
                               ny * static_cast<int>(D_VOL) + nz;
                acc += static_cast<wide_t>(
                    q_mul(input[b * voxels + nv], stencil[k]));
            }
        }
    }
    output[b * voxels + v] = q_tanh(sat64(acc));
}

__global__ void pool_volume_to_state_kernel(const fixed_t* volume,
                                            fixed_t* state,
                                            int batch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / static_cast<int>(D_IN);
    const int i = idx % static_cast<int>(D_IN);
    if (b >= batch) return;

    constexpr int replicas = static_cast<int>(D_VOXELS / D_IN);
    wide_t acc = 0;
    for (int v = i; v < static_cast<int>(D_VOXELS);
         v += static_cast<int>(D_IN)) {
        acc += volume[b * static_cast<int>(D_VOXELS) + v];
    }
    state[b * static_cast<int>(D_IN) + i] = sat64(acc / replicas);
}

__global__ void compute_health_kernel(const fixed_t* previous,
                                      const fixed_t* current,
                                      fixed_t* metrics,
                                      int batch) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;

    wide_t abs_error = 0;
    for (int i = 0; i < static_cast<int>(D_IN); ++i) {
        wide_t d =
            static_cast<wide_t>(previous[b * static_cast<int>(D_IN) + i]) -
            current[b * static_cast<int>(D_IN) + i];
        if (d < 0) d = -d;
        if (d > ONE) d = ONE;
        abs_error += d;
    }
    const fixed_t mean_error =
        sat64(abs_error / static_cast<wide_t>(D_IN));
    metrics[b] = q_sub(ONE, mean_error);
}

__global__ void diagnostic_feedback_kernel(fixed_t* states,
                                           const fixed_t* metrics,
                                           fixed_t gain,
                                           int batch) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = idx / static_cast<int>(D_IN);
    const int i = idx % static_cast<int>(D_IN);
    if (b >= batch) return;

    const fixed_t self = states[b * static_cast<int>(D_IN) + i];
    const fixed_t centered_health = q_sub(metrics[b], HALF);
    const fixed_t correction = q_mul(gain, q_mul(centered_health, self));
    states[b * static_cast<int>(D_IN) + i] =
        q_tanh(q_add(self, correction));
}

struct CycleStats {
    float gpu_ms = 0.0F;
    std::uint64_t estimated_ops = 0;
    double estimated_ops_per_second = 0.0;
    double mean_health = 0.0;
};

class InfinityTurboCore {
public:
    explicit InfinityTurboCore(Config config) : cfg_(config) {
        cfg_.validate();
        check_device_and_memory();
        allocate();
        initialize();
    }

    CycleStats cycle() {
        constexpr int threads = 256;
        Event start;
        Event stop;
        DM_CUDA_CHECK(cudaEventRecord(start.get(), stream_.get()));

        const std::size_t state_count =
            static_cast<std::size_t>(cfg_.batch) * D_IN;
        const std::size_t latent_count =
            static_cast<std::size_t>(cfg_.batch) * D_LAT;
        DM_CUDA_CHECK(cudaMemcpyAsync(
            d_previous_.get(), d_state_.get(), state_count * sizeof(fixed_t),
            cudaMemcpyDeviceToDevice, stream_.get()));

        int blocks =
            static_cast<int>((latent_count + threads - 1U) / threads);
        dense_tanh_kernel<<<blocks, threads, 0, stream_.get()>>>(
            d_state_.get(), d_latent_.get(), d_enc_w_.get(), d_enc_b_.get(),
            cfg_.batch, static_cast<int>(D_IN), static_cast<int>(D_LAT));
        kernel_check();

        blocks = static_cast<int>((state_count + threads - 1U) / threads);
        dense_tanh_kernel<<<blocks, threads, 0, stream_.get()>>>(
            d_latent_.get(), d_recon_.get(), d_dec_w_.get(), d_dec_b_.get(),
            cfg_.batch, static_cast<int>(D_LAT), static_cast<int>(D_IN));
        kernel_check();

        const std::size_t volume_count =
            static_cast<std::size_t>(cfg_.batch) * D_VOXELS;
        blocks = static_cast<int>((volume_count + threads - 1U) / threads);
        lift_state_to_volume_kernel<<<blocks, threads, 0, stream_.get()>>>(
            d_recon_.get(), d_volume_a_.get(), cfg_.batch,
            static_cast<int>(D_VOXELS));
        kernel_check();

        fixed_t* in = d_volume_a_.get();
        fixed_t* out = d_volume_b_.get();
        for (int depth = 0; depth < cfg_.recursion_depth; ++depth) {
            process_volume_3d_kernel<<<blocks, threads, 0, stream_.get()>>>(
                in, out, d_stencil_.get(), cfg_.batch, cfg_.radius);
            kernel_check();
            std::swap(in, out);
        }

        blocks = static_cast<int>((state_count + threads - 1U) / threads);
        pool_volume_to_state_kernel<<<blocks, threads, 0, stream_.get()>>>(
            in, d_state_.get(), cfg_.batch);
        kernel_check();

        blocks = (cfg_.batch + threads - 1) / threads;
        compute_health_kernel<<<blocks, threads, 0, stream_.get()>>>(
            d_previous_.get(), d_state_.get(), d_metrics_.get(), cfg_.batch);
        kernel_check();

        blocks = static_cast<int>((state_count + threads - 1U) / threads);
        diagnostic_feedback_kernel<<<blocks, threads, 0, stream_.get()>>>(
            d_state_.get(), d_metrics_.get(), cfg_.feedback_gain, cfg_.batch);
        kernel_check();

        DM_CUDA_CHECK(cudaEventRecord(stop.get(), stream_.get()));
        DM_CUDA_CHECK(cudaEventSynchronize(stop.get()));

        float ms = 0.0F;
        DM_CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));

        std::vector<fixed_t> health(static_cast<std::size_t>(cfg_.batch));
        DM_CUDA_CHECK(cudaMemcpy(health.data(), d_metrics_.get(),
                                 health.size() * sizeof(fixed_t),
                                 cudaMemcpyDeviceToHost));
        double mean_health = 0.0;
        for (fixed_t h : health) mean_health += q_to_double(h);
        mean_health /= static_cast<double>(health.size());

        const std::uint64_t estimated = estimate_ops();
        const double seconds = static_cast<double>(ms) / 1000.0;
        CycleStats stats;
        stats.gpu_ms = ms;
        stats.estimated_ops = estimated;
        stats.estimated_ops_per_second =
            seconds > 0.0 ? static_cast<double>(estimated) / seconds : 0.0;
        stats.mean_health = mean_health;
        return stats;
    }

    std::array<fixed_t, D_IN> state(std::size_t batch_index = 0) const {
        if (batch_index >= static_cast<std::size_t>(cfg_.batch)) {
            throw std::out_of_range("batch index out of range");
        }
        std::array<fixed_t, D_IN> out{};
        DM_CUDA_CHECK(cudaMemcpy(
            out.data(), d_state_.get() + batch_index * D_IN,
            D_IN * sizeof(fixed_t), cudaMemcpyDeviceToHost));
        return out;
    }

    const Config& config() const noexcept { return cfg_; }

private:
    Config cfg_;
    Stream stream_;
    DeviceBuffer<fixed_t> d_enc_w_;
    DeviceBuffer<fixed_t> d_dec_w_;
    DeviceBuffer<fixed_t> d_enc_b_;
    DeviceBuffer<fixed_t> d_dec_b_;
    DeviceBuffer<fixed_t> d_state_;
    DeviceBuffer<fixed_t> d_previous_;
    DeviceBuffer<fixed_t> d_latent_;
    DeviceBuffer<fixed_t> d_recon_;
    DeviceBuffer<fixed_t> d_metrics_;
    DeviceBuffer<fixed_t> d_volume_a_;
    DeviceBuffer<fixed_t> d_volume_b_;
    DeviceBuffer<fixed_t> d_stencil_;

    void kernel_check() { DM_CUDA_CHECK(cudaPeekAtLastError()); }

    void check_device_and_memory() {
        int devices = 0;
        DM_CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices < 1) throw std::runtime_error("no CUDA device detected");

        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        DM_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        const std::size_t volume_bytes =
            static_cast<std::size_t>(cfg_.batch) * D_VOXELS * sizeof(fixed_t);
        const std::size_t dense_bytes =
            static_cast<std::size_t>(cfg_.batch) *
            (D_IN * 3U + D_LAT) * sizeof(fixed_t);
        const std::size_t weight_bytes =
            (D_LAT * D_IN + D_IN * D_LAT + D_LAT + D_IN) * sizeof(fixed_t);
        const std::size_t required =
            2U * volume_bytes + dense_bytes + weight_bytes + 4U * 1024U * 1024U;
        if (required > free_bytes * 8U / 10U) {
            throw std::runtime_error(
                "requested batch exceeds conservative 80% free-GPU-memory budget");
        }
        (void)total_bytes;
    }

    void allocate() {
        const std::size_t batch = static_cast<std::size_t>(cfg_.batch);
        d_enc_w_.allocate(D_LAT * D_IN);
        d_dec_w_.allocate(D_IN * D_LAT);
        d_enc_b_.allocate(D_LAT);
        d_dec_b_.allocate(D_IN);
        d_state_.allocate(batch * D_IN);
        d_previous_.allocate(batch * D_IN);
        d_latent_.allocate(batch * D_LAT);
        d_recon_.allocate(batch * D_IN);
        d_metrics_.allocate(batch);
        d_volume_a_.allocate(batch * D_VOXELS);
        d_volume_b_.allocate(batch * D_VOXELS);
        const std::size_t side = static_cast<std::size_t>(2 * cfg_.radius + 1);
        d_stencil_.allocate(side * side * side);
    }

    static std::uint32_t xorshift(std::uint32_t& s) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }

    static std::vector<fixed_t> init_weights(std::size_t count,
                                              std::size_t fan_in,
                                              std::size_t fan_out,
                                              std::uint32_t seed) {
        std::vector<fixed_t> w(count);
        const double amp =
            std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
        std::uint32_t state = seed ? seed : 1U;
        for (auto& v : w) {
            const std::uint32_t u = xorshift(state);
            const double unit =
                (static_cast<double>(u) / static_cast<double>(UINT32_MAX)) *
                    2.0 -
                1.0;
            v = q_from_double(unit * amp);
        }
        return w;
    }

    std::vector<fixed_t> make_stencil() const {
        const int side = 2 * cfg_.radius + 1;
        std::vector<double> raw(
            static_cast<std::size_t>(side * side * side));
        double sum = 0.0;
        std::size_t k = 0;
        for (int dx = -cfg_.radius; dx <= cfg_.radius; ++dx) {
            for (int dy = -cfg_.radius; dy <= cfg_.radius; ++dy) {
                for (int dz = -cfg_.radius; dz <= cfg_.radius; ++dz, ++k) {
                    const double dist =
                        std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
                    raw[k] = std::exp(-dist);
                    sum += raw[k];
                }
            }
        }
        std::vector<fixed_t> q(raw.size());
        wide_t qsum = 0;
        for (std::size_t i = 0; i < raw.size(); ++i) {
            q[i] = q_from_double(raw[i] / sum);
            qsum += q[i];
        }
        const std::size_t center = q.size() / 2U;
        q[center] = q_add(
            q[center], sat64(static_cast<wide_t>(ONE) - qsum));
        return q;
    }

    void initialize() {
        auto enc = init_weights(D_LAT * D_IN, D_IN, D_LAT, cfg_.seed);
        auto dec = init_weights(D_IN * D_LAT, D_LAT, D_IN,
                                cfg_.seed ^ 0xBEEF1234u);
        std::vector<fixed_t> enc_b(D_LAT, 0);
        std::vector<fixed_t> dec_b(D_IN, 0);
        auto stencil = make_stencil();

        const std::size_t batch = static_cast<std::size_t>(cfg_.batch);
        std::vector<fixed_t> state(batch * D_IN);
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t i = 0; i < D_IN; ++i) {
                const double phase =
                    2.0 * PI * static_cast<double>(i) /
                        static_cast<double>(D_IN) +
                    0.1 * static_cast<double>(b);
                state[b * D_IN + i] =
                    q_from_double(0.5 * std::sin(phase));
            }
        }

        DM_CUDA_CHECK(cudaMemcpyAsync(
            d_enc_w_.get(), enc.data(), enc.size() * sizeof(fixed_t),
            cudaMemcpyHostToDevice, stream_.get()));
        DM_CUDA_CHECK(cudaMemcpyAsync(
            d_dec_w_.get(), dec.data(), dec.size() * sizeof(fixed_t),
            cudaMemcpyHostToDevice, stream_.get()));
        DM_CUDA_CHECK(cudaMemcpyAsync(
            d_enc_b_.get(), enc_b.data(), enc_b.size() * sizeof(fixed_t),
            cudaMemcpyHostToDevice, stream_.get()));
        DM_CUDA_CHECK(cudaMemcpyAsync(
            d_dec_b_.get(), dec_b.data(), dec_b.size() * sizeof(fixed_t),
            cudaMemcpyHostToDevice, stream_.get()));
        DM_CUDA_CHECK(cudaMemcpyAsync(
            d_state_.get(), state.data(), state.size() * sizeof(fixed_t),
            cudaMemcpyHostToDevice, stream_.get()));
        DM_CUDA_CHECK(cudaMemcpyAsync(
            d_stencil_.get(), stencil.data(), stencil.size() * sizeof(fixed_t),
            cudaMemcpyHostToDevice, stream_.get()));
        DM_CUDA_CHECK(cudaMemsetAsync(
            d_metrics_.get(), 0, d_metrics_.size() * sizeof(fixed_t),
            stream_.get()));
        DM_CUDA_CHECK(cudaMemsetAsync(
            d_volume_a_.get(), 0, d_volume_a_.size() * sizeof(fixed_t),
            stream_.get()));
        DM_CUDA_CHECK(cudaMemsetAsync(
            d_volume_b_.get(), 0, d_volume_b_.size() * sizeof(fixed_t),
            stream_.get()));
        DM_CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    }

    std::uint64_t estimate_ops() const {
        const std::uint64_t batch = static_cast<std::uint64_t>(cfg_.batch);
        const std::uint64_t side =
            static_cast<std::uint64_t>(2 * cfg_.radius + 1);
        const std::uint64_t neighbors = side * side * side;
        const std::uint64_t dense =
            batch * 2ULL *
            (static_cast<std::uint64_t>(D_IN) * D_LAT +
             static_cast<std::uint64_t>(D_LAT) * D_IN);
        const std::uint64_t volume =
            batch * static_cast<std::uint64_t>(D_VOXELS) * neighbors * 2ULL *
            static_cast<std::uint64_t>(cfg_.recursion_depth);
        const std::uint64_t lift_pool_feedback =
            batch * (2ULL * static_cast<std::uint64_t>(D_VOXELS) +
                     8ULL * static_cast<std::uint64_t>(D_IN));
        return dense + volume + lift_pool_feedback;
    }
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") +
                                            name);
            }
            return argv[++i];
        };
        if (a == "--batch") cfg.batch = std::stoi(next("--batch"));
        else if (a == "--radius") cfg.radius = std::stoi(next("--radius"));
        else if (a == "--depth")
            cfg.recursion_depth = std::stoi(next("--depth"));
        else if (a == "--cycles") cfg.cycles = std::stoi(next("--cycles"));
        else if (a == "--feedback")
            cfg.feedback_gain = q_from_double(std::stod(next("--feedback")));
        else if (a == "--seed")
            cfg.seed = static_cast<std::uint32_t>(
                std::stoul(next("--seed")));
        else if (a == "--help") {
            std::cout
                << "Usage: jarvisx-infinity-turbo [--batch N] [--radius 1..4] "
                   "[--depth 1..8] [--cycles N] [--feedback X] [--seed N]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + a);
        }
    }
    cfg.validate();
    return cfg;
}

}  // namespace dm3d_infinity_turbo

int main(int argc, char** argv) {
    using namespace dm3d_infinity_turbo;
    try {
        const Config cfg = parse_args(argc, argv);
        InfinityTurboCore core(cfg);

        std::cout
            << "DM-vOmegaXi+ Infinity Turbo Core v7.1 (experimental)\n"
            << "Q16.16 | 128D -> 48D -> 128D | volume=" << D_VOL << '^' << 3
            << " | batch=" << cfg.batch << " | radius=" << cfg.radius
            << " | depth=" << cfg.recursion_depth << "\n"
            << "Performance claims are measured at runtime; no fixed 10^9x "
               "speedup is asserted.\n\n";

        double total_ops = 0.0;
        double total_seconds = 0.0;
        double health = 0.0;
        for (int c = 0; c < cfg.cycles; ++c) {
            const CycleStats s = core.cycle();
            total_ops += static_cast<double>(s.estimated_ops);
            total_seconds += static_cast<double>(s.gpu_ms) / 1000.0;
            health = s.mean_health;
            std::cout << "cycle=" << (c + 1)
                      << " gpu_ms=" << std::fixed << std::setprecision(3)
                      << s.gpu_ms << " est_Gop/s=" << std::setprecision(3)
                      << s.estimated_ops_per_second / 1.0e9
                      << " diagnostic_health=" << std::setprecision(4)
                      << s.mean_health << '\n';
        }

        const auto s0 = core.state(0);
        std::cout
            << "\naggregate_est_Gop/s=" << std::fixed << std::setprecision(3)
            << (total_seconds > 0.0
                    ? total_ops / total_seconds / 1.0e9
                    : 0.0)
            << "\nstate[0]=" << s0[0] << " (" << std::setprecision(6)
            << q_to_double(s0[0]) << ")"
            << "\nmean_diagnostic_health=" << std::setprecision(4) << health
            << "\nstatus=CUDA execution path completed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
