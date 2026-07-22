// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Dr Matladi Maxwell Moagi
//
// Jarvis-X SE(3) CUDA reference kernel
// One CUDA thread integrates one rigid-body twist and emits one 3x4 pose.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t error__ = (call);                                                  \
        if (error__ != cudaSuccess) {                                                        \
            std::cerr << "CUDA failure at " << __FILE__ << ':' << __LINE__ << ": "         \
                      << cudaGetErrorString(error__) << '\n';                                \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (false)

struct alignas(16) Float4 {
    float x;
    float y;
    float z;
    float w;
};

// Two aligned vector loads per pose. omega and velocity are rates; the kernel
// multiplies both by dt before evaluating the finite SE(3) exponential map.
struct alignas(16) Twist8 {
    Float4 omega;
    Float4 velocity;
};

// Row-major [R | t]. The invariant homogeneous row [0, 0, 0, 1] is omitted.
struct alignas(16) Pose3x4 {
    Float4 row0;
    Float4 row1;
    Float4 row2;
};

static_assert(sizeof(Float4) == 16, "Float4 must remain a 16-byte vector");
static_assert(sizeof(Twist8) == 32, "Twist8 must remain two vector loads");
static_assert(sizeof(Pose3x4) == 48, "Pose3x4 must remain three vector stores");

struct CoefficientsF {
    float a;
    float b;
    float c;
};

__device__ __forceinline__ CoefficientsF se3_coefficients(float theta2) {
    // Stable Maclaurin evaluation around the removable singularity theta=0.
    if (theta2 < 1.0e-8f) {
        const float theta4 = theta2 * theta2;
        const float theta6 = theta4 * theta2;
        return {
            1.0f - theta2 * (1.0f / 6.0f) + theta4 * (1.0f / 120.0f) -
                theta6 * (1.0f / 5040.0f),
            0.5f - theta2 * (1.0f / 24.0f) + theta4 * (1.0f / 720.0f) -
                theta6 * (1.0f / 40320.0f),
            (1.0f / 6.0f) - theta2 * (1.0f / 120.0f) +
                theta4 * (1.0f / 5040.0f) - theta6 * (1.0f / 362880.0f),
        };
    }

    const float theta = sqrtf(theta2);
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(theta, &sine, &cosine);
    const float inverse_theta = 1.0f / theta;
    const float inverse_theta2 = 1.0f / theta2;
    const float a = sine * inverse_theta;
    return {
        a,
        (1.0f - cosine) * inverse_theta2,
        (1.0f - a) * inverse_theta2,
    };
}

__global__ void se3_exp_kernel(
    const Twist8* __restrict__ twists,
    Pose3x4* __restrict__ poses,
    std::size_t count,
    float dt) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const Twist8 twist = twists[index];

    // Finite exponential coordinates phi = omega * dt, rho = velocity * dt.
    const float x = twist.omega.x * dt;
    const float y = twist.omega.y * dt;
    const float z = twist.omega.z * dt;
    const float rx = twist.velocity.x * dt;
    const float ry = twist.velocity.y * dt;
    const float rz = twist.velocity.z * dt;

    const float theta2 = fmaf(x, x, fmaf(y, y, z * z));
    const CoefficientsF coefficient = se3_coefficients(theta2);

    // Rodrigues rotation: R = I + A[phi]x + B[phi]x^2.
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;

    const float r00 = 1.0f - coefficient.b * (yy + zz);
    const float r01 = coefficient.b * xy - coefficient.a * z;
    const float r02 = coefficient.b * xz + coefficient.a * y;
    const float r10 = coefficient.b * xy + coefficient.a * z;
    const float r11 = 1.0f - coefficient.b * (xx + zz);
    const float r12 = coefficient.b * yz - coefficient.a * x;
    const float r20 = coefficient.b * xz - coefficient.a * y;
    const float r21 = coefficient.b * yz + coefficient.a * x;
    const float r22 = 1.0f - coefficient.b * (xx + yy);

    // Left Jacobian action without materializing a matrix:
    // t = rho + B(phi x rho) + C(phi x (phi x rho)).
    const float cross_x = y * rz - z * ry;
    const float cross_y = z * rx - x * rz;
    const float cross_z = x * ry - y * rx;

    const float double_cross_x = y * cross_z - z * cross_y;
    const float double_cross_y = z * cross_x - x * cross_z;
    const float double_cross_z = x * cross_y - y * cross_x;

    const float tx = fmaf(
        coefficient.c,
        double_cross_x,
        fmaf(coefficient.b, cross_x, rx));
    const float ty = fmaf(
        coefficient.c,
        double_cross_y,
        fmaf(coefficient.b, cross_y, ry));
    const float tz = fmaf(
        coefficient.c,
        double_cross_z,
        fmaf(coefficient.b, cross_z, rz));

    poses[index] = {
        {r00, r01, r02, tx},
        {r10, r11, r12, ty},
        {r20, r21, r22, tz},
    };
}

struct Options {
    std::size_t count = 1u << 20;
    int repeats = 100;
    int warmup = 10;
    int device = 0;
    float dt = 0.01f;
};

struct PoseReference {
    double values[12];
};

struct ValidationMetrics {
    double rotation_rms_rad = 0.0;
    double rotation_max_rad = 0.0;
    double translation_rms = 0.0;
    double translation_max = 0.0;
    double orthogonality_max = 0.0;
    double determinant_error_max = 0.0;
};

[[noreturn]] void usage(const char* program, int status) {
    std::ostream& output = status == EXIT_SUCCESS ? std::cout : std::cerr;
    output << "Usage: " << program << " [options]\n"
           << "  --count N      Number of poses (default 1048576)\n"
           << "  --repeats N    Timed kernel repetitions (default 100)\n"
           << "  --warmup N     Warm-up launches (default 10)\n"
           << "  --dt SECONDS   Integration interval (default 0.01)\n"
           << "  --device N     CUDA device index (default 0)\n"
           << "  --help         Show this message\n";
    std::exit(status);
}

template <typename T>
T parse_number(const char* option, const char* text);

template <>
std::size_t parse_number<std::size_t>(const char* option, const char* text) {
    try {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != std::strlen(text) || value == 0) {
            throw std::invalid_argument("not a positive integer");
        }
        return static_cast<std::size_t>(value);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string(option) + ": " + error.what());
    }
}

template <>
int parse_number<int>(const char* option, const char* text) {
    try {
        std::size_t consumed = 0;
        const long value = std::stol(text, &consumed);
        if (consumed != std::strlen(text) || value < 0 ||
            value > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("outside the supported integer range");
        }
        return static_cast<int>(value);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string(option) + ": " + error.what());
    }
}

template <>
float parse_number<float>(const char* option, const char* text) {
    try {
        std::size_t consumed = 0;
        const float value = std::stof(text, &consumed);
        if (consumed != std::strlen(text) || !std::isfinite(value) || value <= 0.0f) {
            throw std::invalid_argument("must be finite and greater than zero");
        }
        return value;
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string(option) + ": " + error.what());
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            usage(argv[0], EXIT_SUCCESS);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument(argument + " requires a value");
        }
        const char* value = argv[++index];
        if (argument == "--count") {
            options.count = parse_number<std::size_t>(argument.c_str(), value);
        } else if (argument == "--repeats") {
            options.repeats = parse_number<int>(argument.c_str(), value);
        } else if (argument == "--warmup") {
            options.warmup = parse_number<int>(argument.c_str(), value);
        } else if (argument == "--dt") {
            options.dt = parse_number<float>(argument.c_str(), value);
        } else if (argument == "--device") {
            options.device = parse_number<int>(argument.c_str(), value);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.repeats == 0) {
        throw std::invalid_argument("--repeats must be greater than zero");
    }
    return options;
}

void generate_twists(Twist8* twists, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        const double phase = static_cast<double>(index) * 0.0009765625;
        Twist8 value{};
        value.omega = {
            static_cast<float>(2.25 * std::sin(phase * 1.11)),
            static_cast<float>(1.75 * std::cos(phase * 0.73)),
            static_cast<float>(2.50 * std::sin(phase * 0.37 + 0.2)),
            0.0f,
        };
        value.velocity = {
            static_cast<float>(4.0 * std::cos(phase * 0.19)),
            static_cast<float>(2.0 * std::sin(phase * 0.41)),
            static_cast<float>(3.0 * std::cos(phase * 0.67 + 0.4)),
            0.0f,
        };

        // Force exact singular and pure-translation cases into every batch.
        if ((index & 4095u) == 0u) {
            value.omega = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        if ((index & 16383u) == 0u) {
            value.velocity = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        twists[index] = value;
    }
}

PoseReference se3_exp_reference(const Twist8& twist, double dt) {
    const double x = static_cast<double>(twist.omega.x) * dt;
    const double y = static_cast<double>(twist.omega.y) * dt;
    const double z = static_cast<double>(twist.omega.z) * dt;
    const double rx = static_cast<double>(twist.velocity.x) * dt;
    const double ry = static_cast<double>(twist.velocity.y) * dt;
    const double rz = static_cast<double>(twist.velocity.z) * dt;
    const double theta2 = x * x + y * y + z * z;

    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    if (theta2 < 1.0e-12) {
        const double theta4 = theta2 * theta2;
        const double theta6 = theta4 * theta2;
        a = 1.0 - theta2 / 6.0 + theta4 / 120.0 - theta6 / 5040.0;
        b = 0.5 - theta2 / 24.0 + theta4 / 720.0 - theta6 / 40320.0;
        c = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0 - theta6 / 362880.0;
    } else {
        const double theta = std::sqrt(theta2);
        a = std::sin(theta) / theta;
        b = (1.0 - std::cos(theta)) / theta2;
        c = (1.0 - a) / theta2;
    }

    const double cross_x = y * rz - z * ry;
    const double cross_y = z * rx - x * rz;
    const double cross_z = x * ry - y * rx;
    const double double_cross_x = y * cross_z - z * cross_y;
    const double double_cross_y = z * cross_x - x * cross_z;
    const double double_cross_z = x * cross_y - y * cross_x;

    return {{
        1.0 - b * (y * y + z * z),
        b * x * y - a * z,
        b * x * z + a * y,
        rx + b * cross_x + c * double_cross_x,
        b * x * y + a * z,
        1.0 - b * (x * x + z * z),
        b * y * z - a * x,
        ry + b * cross_y + c * double_cross_y,
        b * x * z - a * y,
        b * y * z + a * x,
        1.0 - b * (x * x + y * y),
        rz + b * cross_z + c * double_cross_z,
    }};
}

double clamp_unit(double value) {
    return std::max(-1.0, std::min(1.0, value));
}

ValidationMetrics validate(
    const Twist8* input,
    const Pose3x4* output,
    std::size_t count,
    double dt) {
    long double rotation_squared = 0.0;
    long double translation_squared = 0.0;
    ValidationMetrics metrics;

    for (std::size_t index = 0; index < count; ++index) {
        const PoseReference reference = se3_exp_reference(input[index], dt);
        const Pose3x4& pose = output[index];
        const double gpu[12] = {
            pose.row0.x, pose.row0.y, pose.row0.z, pose.row0.w,
            pose.row1.x, pose.row1.y, pose.row1.z, pose.row1.w,
            pose.row2.x, pose.row2.y, pose.row2.z, pose.row2.w,
        };

        const double rotation_inner_product =
            reference.values[0] * gpu[0] + reference.values[1] * gpu[1] +
            reference.values[2] * gpu[2] + reference.values[4] * gpu[4] +
            reference.values[5] * gpu[5] + reference.values[6] * gpu[6] +
            reference.values[8] * gpu[8] + reference.values[9] * gpu[9] +
            reference.values[10] * gpu[10];
        const double rotation_error =
            std::acos(clamp_unit((rotation_inner_product - 1.0) * 0.5));

        const double tx = gpu[3] - reference.values[3];
        const double ty = gpu[7] - reference.values[7];
        const double tz = gpu[11] - reference.values[11];
        const double translation_error = std::sqrt(tx * tx + ty * ty + tz * tz);

        rotation_squared += rotation_error * rotation_error;
        translation_squared += translation_error * translation_error;
        metrics.rotation_max_rad = std::max(metrics.rotation_max_rad, rotation_error);
        metrics.translation_max = std::max(metrics.translation_max, translation_error);

        const double r00 = gpu[0];
        const double r01 = gpu[1];
        const double r02 = gpu[2];
        const double r10 = gpu[4];
        const double r11 = gpu[5];
        const double r12 = gpu[6];
        const double r20 = gpu[8];
        const double r21 = gpu[9];
        const double r22 = gpu[10];

        const double orthogonality_terms[6] = {
            r00 * r00 + r10 * r10 + r20 * r20 - 1.0,
            r01 * r01 + r11 * r11 + r21 * r21 - 1.0,
            r02 * r02 + r12 * r12 + r22 * r22 - 1.0,
            r00 * r01 + r10 * r11 + r20 * r21,
            r00 * r02 + r10 * r12 + r20 * r22,
            r01 * r02 + r11 * r12 + r21 * r22,
        };
        for (double term : orthogonality_terms) {
            metrics.orthogonality_max =
                std::max(metrics.orthogonality_max, std::abs(term));
        }

        const double determinant =
            r00 * (r11 * r22 - r12 * r21) -
            r01 * (r10 * r22 - r12 * r20) +
            r02 * (r10 * r21 - r11 * r20);
        metrics.determinant_error_max =
            std::max(metrics.determinant_error_max, std::abs(determinant - 1.0));
    }

    metrics.rotation_rms_rad =
        std::sqrt(static_cast<double>(rotation_squared / count));
    metrics.translation_rms =
        std::sqrt(static_cast<double>(translation_squared / count));
    return metrics;
}

float time_copy(
    cudaEvent_t start,
    cudaEvent_t stop,
    cudaStream_t stream,
    void* destination,
    const void* source,
    std::size_t bytes,
    cudaMemcpyKind kind) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyAsync(destination, source, bytes, kind, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

int run(const Options& options) {
    CUDA_CHECK(cudaSetDevice(options.device));

    cudaDeviceProp property{};
    CUDA_CHECK(cudaGetDeviceProperties(&property, options.device));

    Twist8* host_input = nullptr;
    Pose3x4* host_output = nullptr;
    Twist8* device_input = nullptr;
    Pose3x4* device_output = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    if (options.count > std::numeric_limits<std::size_t>::max() / sizeof(Twist8) ||
        options.count > std::numeric_limits<std::size_t>::max() / sizeof(Pose3x4)) {
        throw std::overflow_error("pose count overflows allocation size");
    }
    const std::size_t input_bytes = options.count * sizeof(Twist8);
    const std::size_t output_bytes = options.count * sizeof(Pose3x4);

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_input), input_bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&host_output), output_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_input), input_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    generate_twists(host_input, options.count);

    const float h2d_ms = time_copy(
        start,
        stop,
        stream,
        device_input,
        host_input,
        input_bytes,
        cudaMemcpyHostToDevice);

    constexpr int threads_per_block = 256;
    const std::size_t raw_blocks =
        (options.count + threads_per_block - 1) / threads_per_block;
    if (raw_blocks > static_cast<std::size_t>(property.maxGridSize[0])) {
        throw std::overflow_error("pose count exceeds one-dimensional CUDA grid capacity");
    }
    const unsigned int blocks = static_cast<unsigned int>(raw_blocks);

    for (int index = 0; index < options.warmup; ++index) {
        se3_exp_kernel<<<blocks, threads_per_block, 0, stream>>>(
            device_input,
            device_output,
            options.count,
            options.dt);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int index = 0; index < options.repeats; ++index) {
        se3_exp_kernel<<<blocks, threads_per_block, 0, stream>>>(
            device_input,
            device_output,
            options.count,
            options.dt);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_total_ms, start, stop));
    const double kernel_ms = kernel_total_ms / options.repeats;

    const float d2h_ms = time_copy(
        start,
        stop,
        stream,
        host_output,
        device_output,
        output_bytes,
        cudaMemcpyDeviceToHost);

    const ValidationMetrics metrics =
        validate(host_input, host_output, options.count, options.dt);

    const double seconds = kernel_ms * 1.0e-3;
    const double poses_per_second = static_cast<double>(options.count) / seconds;
    const double kernel_bytes = static_cast<double>(input_bytes + output_bytes);
    const double effective_gigabytes_per_second = kernel_bytes / seconds / 1.0e9;
    const double end_to_end_ms = h2d_ms + kernel_ms + d2h_ms;

    std::cout << std::fixed << std::setprecision(6)
              << "Jarvis-X SE(3) CUDA reference\n"
              << "device=" << property.name << " cc=" << property.major << '.'
              << property.minor << "\n"
              << "poses=" << options.count << " dt_s=" << options.dt
              << " warmup=" << options.warmup << " repeats=" << options.repeats << "\n"
              << "layout_bytes_per_pose=" << (sizeof(Twist8) + sizeof(Pose3x4))
              << " input_MiB=" << input_bytes / 1048576.0
              << " output_MiB=" << output_bytes / 1048576.0 << "\n"
              << "h2d_ms=" << h2d_ms << " kernel_ms=" << kernel_ms
              << " d2h_ms=" << d2h_ms << " end_to_end_ms=" << end_to_end_ms << "\n"
              << "poses_per_second=" << poses_per_second
              << " effective_kernel_GBps=" << effective_gigabytes_per_second << "\n"
              << "rotation_rms_rad=" << metrics.rotation_rms_rad
              << " rotation_max_rad=" << metrics.rotation_max_rad << "\n"
              << "translation_rms=" << metrics.translation_rms
              << " translation_max=" << metrics.translation_max << "\n"
              << "orthogonality_max=" << metrics.orthogonality_max
              << " determinant_error_max=" << metrics.determinant_error_max << "\n";

    // Conservative reference thresholds for this deterministic input domain.
    const bool valid =
        metrics.rotation_max_rad <= 2.0e-3 &&
        metrics.translation_max <= 5.0e-5 &&
        metrics.orthogonality_max <= 5.0e-5 &&
        metrics.determinant_error_max <= 5.0e-5;
    std::cout << "validation=" << (valid ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(device_output));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFreeHost(host_output));
    CUDA_CHECK(cudaFreeHost(host_input));

    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
