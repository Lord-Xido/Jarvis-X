// Dr Moagi 3D ANN inward loop -- bounded, reproducible research reference.
// C++17. Authorship: based on the Matladi Maxwell Moagi ANN concept
// provided 2026-11-16; revised implementation for complete backpropagation.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace moagi {
constexpr int SIDE = 32;
constexpr int PATCH = 4;
constexpr int PATCH_DIM = PATCH * PATCH * PATCH;
constexpr int GRID = SIDE / PATCH;
constexpr int VOXELS = SIDE * SIDE * SIDE;
constexpr int PATCH_COUNT = GRID * GRID * GRID;
constexpr int HIDDEN_DIM = 32;
constexpr int LATENT_DIM = 8;
constexpr float LEARNING_RATE = 0.025f;
constexpr int EPOCHS = 30;
constexpr int LOOP_STEPS = 50;
constexpr int FLIP_CENTERS = 16;
static_assert(SIDE % PATCH == 0, "exact tile division required");

inline float act(float x) { return x > 0.0f ? x : 0.05f * x; }
inline float act_grad(float x) { return x > 0.0f ? 1.0f : 0.05f; }
inline float sigmoid(float x) {
    if (x >= 0) { float q = std::exp(-x); return 1.0f / (1.0f + q); }
    float q = std::exp(x); return q / (1.0f + q);
}

struct Lattice3D {
    std::array<std::uint8_t, VOXELS> bits{};
    static size_t index(int x, int y, int z) {
        const auto wrap = [](int n) { return (n % SIDE + SIDE) % SIDE; };
        return static_cast<size_t>((wrap(z) * SIDE + wrap(y)) * SIDE + wrap(x));
    }
    std::uint8_t get(int x, int y, int z) const { return bits[index(x, y, z)]; }
    void flip(int x, int y, int z) { bits[index(x, y, z)] ^= 1u; }
};

using Patch = std::array<float, PATCH_DIM>;
using PatchBits = std::array<std::uint8_t, PATCH_DIM>;

Patch load_patch(const Lattice3D &l, int tile) {
    Patch p{};
    const int tz = tile / (GRID * GRID);
    const int ty = (tile / GRID) % GRID;
    const int tx = tile % GRID;
    for (int z = 0; z < PATCH; ++z)
        for (int y = 0; y < PATCH; ++y)
            for (int x = 0; x < PATCH; ++x)
                p[(z * PATCH + y) * PATCH + x] = static_cast<float>(
                    l.get(tx * PATCH + x, ty * PATCH + y, tz * PATCH + z));
    return p;
}
void store_patch(Lattice3D &l, int tile, const Patch &p) {
    const int tz = tile / (GRID * GRID);
    const int ty = (tile / GRID) % GRID;
    const int tx = tile % GRID;
    for (int z = 0; z < PATCH; ++z)
        for (int y = 0; y < PATCH; ++y)
            for (int x = 0; x < PATCH; ++x)
                l.bits[Lattice3D::index(tx * PATCH + x, ty * PATCH + y, tz * PATCH + z)] =
                    static_cast<std::uint8_t>(p[(z * PATCH + y) * PATCH + x] >= 0.5f);
}

// Spatially repeated synthetic motifs make low-dimensional bottlenecks testable.
// This deliberately does not assert that arbitrary random 3D bits are compressible.
Lattice3D reference_lattice() {
    Lattice3D l;
    for (int z = 0; z < SIDE; ++z) for (int y = 0; y < SIDE; ++y)
        for (int x = 0; x < SIDE; ++x) {
            const int lx = x % PATCH, ly = y % PATCH, lz = z % PATCH;
            const int kind = (x / PATCH + 2 * (y / PATCH) + 3 * (z / PATCH)) % 4;
            bool value = false;
            switch (kind) {
                case 0: value = ((lx - 1.5f) * (lx - 1.5f) +
                                 (ly - 1.5f) * (ly - 1.5f) +
                                 (lz - 1.5f) * (lz - 1.5f)) <= 3.0f; break;
                case 1: value = lx == 1 || lx == 2; break;
                case 2: value = (lx + ly + lz) % 2 == 0; break;
                default: value = (lx == ly || ly == lz); break;
            }
            l.bits[Lattice3D::index(x, y, z)] = static_cast<std::uint8_t>(value);
        }
    return l;
}

// Each layer owns a contiguous matrix, cached input and pre-activations.
// Backward propagates through PRE-UPDATE weights, then performs ordinary SGD.
template <size_t IN, size_t OUT>
struct Layer {
    std::array<float, IN * OUT> W{};
    std::array<float, OUT> b{}, pre{};
    std::array<float, IN> input{};

    explicit Layer(std::mt19937 &rng) {
        std::normal_distribution<float> normal(0.0f, std::sqrt(2.0f / float(IN)));
        for (float &w : W) w = normal(rng);
    }
    std::array<float, OUT> forward(const std::array<float, IN> &x, bool output_sigmoid) {
        input = x;
        std::array<float, OUT> output{};
        for (size_t o = 0; o < OUT; ++o) {
            float sum = b[o];
            for (size_t i = 0; i < IN; ++i) sum += W[o * IN + i] * x[i];
            pre[o] = sum;
            output[o] = output_sigmoid ? sigmoid(sum) : act(sum);
        }
        return output;
    }
    // delta is dLoss/d(preactivation); activation derivative belongs to caller.
    std::array<float, IN> backward_delta(const std::array<float, OUT> &delta, float lr) {
        std::array<float, IN> grad_in{};
        for (size_t o = 0; o < OUT; ++o) {
            for (size_t i = 0; i < IN; ++i) {
                const size_t k = o * IN + i;
                grad_in[i] += W[k] * delta[o]; // old weights, before the update
                W[k] -= lr * delta[o] * input[i];
            }
            b[o] -= lr * delta[o];
        }
        return grad_in;
    }
};

template<size_t N>
std::array<float, N> with_act_derivative(
    const std::array<float, N> &gradient, const std::array<float, N> &pre) {
    std::array<float, N> result{};
    for (size_t i = 0; i < N; ++i) result[i] = gradient[i] * act_grad(pre[i]);
    return result;
}

struct MoagiANN3D {
    Layer<PATCH_DIM, HIDDEN_DIM> enc1;
    Layer<HIDDEN_DIM, LATENT_DIM> enc2;
    Layer<LATENT_DIM, HIDDEN_DIM> dec1;
    Layer<HIDDEN_DIM, PATCH_DIM> dec2;
    std::array<float, LATENT_DIM> latent{};

    explicit MoagiANN3D(std::mt19937 &rng)
        : enc1(rng), enc2(rng), dec1(rng), dec2(rng) {}

    Patch forward(const Patch &in) {
        const auto h1 = enc1.forward(in, false);
        latent = enc2.forward(h1, false);
        const auto h2 = dec1.forward(latent, false);
        return dec2.forward(h2, true);
    }
    // BCE with sigmoid output: dL/dz = (prediction-target)/PATCH_DIM.
    double train_step(const Patch &noisy, const Patch &target) {
        const Patch pred = forward(noisy);
        std::array<float, PATCH_DIM> delta{};
        double loss = 0.0;
        for (int i = 0; i < PATCH_DIM; ++i) {
            const double p = std::clamp(static_cast<double>(pred[i]), 1e-7, 1.0 - 1e-7);
            const double t = target[i];
            loss -= (t * std::log(p) + (1.0 - t) * std::log(1.0 - p)) / PATCH_DIM;
            delta[i] = (pred[i] - target[i]) / PATCH_DIM;
        }
        const auto gd1 = dec2.backward_delta(delta, LEARNING_RATE);
        const auto gz = dec1.backward_delta(with_act_derivative(gd1, dec1.pre), LEARNING_RATE);
        const auto ge1 = enc2.backward_delta(with_act_derivative(gz, enc2.pre), LEARNING_RATE);
        enc1.backward_delta(with_act_derivative(ge1, enc1.pre), LEARNING_RATE);
        return loss;
    }
};

int bit_errors(const Lattice3D &a, const Lattice3D &b) {
    int n = 0;
    for (size_t i = 0; i < a.bits.size(); ++i) n += a.bits[i] != b.bits[i];
    return n;
}
std::uint64_t lattice_hash(const Lattice3D &l) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (std::uint8_t bit : l.bits) { hash ^= bit; hash *= 1099511628211ULL; }
    return hash;
}

// One center event toggles center plus six distinct toroidal neighbors.
int apply_virtual_flips(Lattice3D &l, std::mt19937 &rng, int centers) {
    std::uniform_int_distribution<int> d(0, SIDE - 1);
    for (int n = 0; n < centers; ++n) {
        const int x = d(rng), y = d(rng), z = d(rng);
        l.flip(x, y, z);
        l.flip(x - 1, y, z); l.flip(x + 1, y, z);
        l.flip(x, y - 1, z); l.flip(x, y + 1, z);
        l.flip(x, y, z - 1); l.flip(x, y, z + 1);
    }
    return centers * 7; // attempted toggles, not guaranteed net-changed bits
}

// Unbiased bit-noise generator for supervised synthetic denoising training.
Patch corrupt(const Patch &clean, std::mt19937 &rng, float probability) {
    std::bernoulli_distribution b(probability);
    Patch noisy = clean;
    for (float &v : noisy) if (b(rng)) v = 1.0f - v;
    return noisy;
}

double pretrain(MoagiANN3D &net, const Lattice3D &truth, std::mt19937 &rng, int epochs) {
    std::vector<int> order(PATCH_COUNT);
    std::iota(order.begin(), order.end(), 0);
    double last = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        double loss = 0.0;
        for (int tile : order) {
            const auto clean = load_patch(truth, tile);
            loss += net.train_step(corrupt(clean, rng, 0.08f), clean);
        }
        last = loss / PATCH_COUNT;
        if (epoch == 0 || epoch == epochs - 1 || (epoch + 1) % 10 == 0) {
            std::cout << "train_epoch=" << (epoch + 1) << " noisy_to_clean_BCE="
                      << std::fixed << std::setprecision(6) << last << "\n";
        }
    }
    return last;
}

struct Reconstructed {
    Lattice3D hard;
    double latent_energy = 0;
    double target_bce = 0;
};
Reconstructed reconstruct(MoagiANN3D &net, const Lattice3D &input,
                          const Lattice3D &truth) {
    Reconstructed result;
    for (int tile = 0; tile < PATCH_COUNT; ++tile) {
        const auto p = net.forward(load_patch(input, tile));
        const auto t = load_patch(truth, tile);
        store_patch(result.hard, tile, p);
        for (int i = 0; i < PATCH_DIM; ++i) {
            const double v = std::clamp(static_cast<double>(p[i]), 1e-7, 1.0 - 1e-7);
            result.target_bce -= (t[i] * std::log(v) + (1.0 - t[i]) * std::log(1.0 - v)) / VOXELS;
        }
        for (float z : net.latent) result.latent_energy += double(z) * z / (PATCH_COUNT * LATENT_DIM);
    }
    return result;
}

void self_test() {
    if (Lattice3D::index(-1, 0, 0) != Lattice3D::index(31, 0, 0))
        throw std::runtime_error("periodic negative index failed");
    if (Lattice3D::index(32, 0, 0) != Lattice3D::index(0, 0, 0))
        throw std::runtime_error("periodic positive index failed");
    auto truth = reference_lattice();
    Lattice3D two_flips = truth;
    std::mt19937 gen_a(17), gen_b(17);
    apply_virtual_flips(two_flips, gen_a, 50);
    apply_virtual_flips(two_flips, gen_b, 50);
    if (bit_errors(two_flips, truth) != 0)
        throw std::runtime_error("XOR flip involution failed");
    Lattice3D roundtrip;
    for (int tile = 0; tile < PATCH_COUNT; ++tile)
        store_patch(roundtrip, tile, load_patch(truth, tile));
    if (bit_errors(roundtrip, truth) != 0)
        throw std::runtime_error("3D tiled load/store lossless mapping failed");
    // Full-backprop sanity: learn one repeated motif with a fixed noisy input.
    std::mt19937 seed(1234);
    MoagiANN3D net(seed);
    const Patch target = load_patch(truth, 0);
    Patch noisy = target;
    noisy[0] = 1.0f - noisy[0]; noisy[8] = 1.0f - noisy[8];
    auto bce = [&](const Patch &p) {
        const auto out = net.forward(p);
        double v = 0;
        for (int i = 0; i < PATCH_DIM; ++i) {
            const double y = std::clamp(double(out[i]), 1e-7, 1.0 - 1e-7);
            v -= (target[i] * std::log(y) + (1.0 - target[i]) * std::log(1.0 - y)) / PATCH_DIM;
        }
        return v;
    };
    const double initial = bce(noisy);
    for (int i = 0; i < 500; ++i) net.train_step(noisy, target);
    const double final = bce(noisy);
    if (!(final < initial * 0.6))
        throw std::runtime_error("4-layer backprop learning regression");
    std::cout << "SELF_TEST_PASS index, XOR, 3D patch roundtrip, 4-layer learning"
              << " initial_BCE=" << initial << " final_BCE=" << final << "\n";
}

void run() {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    std::mt19937 rng(1234);
    auto target = reference_lattice();
    MoagiANN3D net(rng);
    std::cout << "DR_MOAGI_3D_ANN_INWARD_LOOP v1\n"
              << "voxels=" << VOXELS << " patch_edge=" << PATCH
              << " patches=" << PATCH_COUNT << " patch_dims=" << PATCH_DIM
              << " hidden=" << HIDDEN_DIM << " latent=" << LATENT_DIM << "\n"
              << "per_patch_nominal_dimensional_ratio=" << (PATCH_DIM / LATENT_DIM)
              << ":1 (NOT byte compression)\n"
              << "mode=supervised_synthetic_pretrain_then_recursive_inference\n"
              << "claimed_1e24_flips_per_ns=NOT_IMPLEMENTED_OR_BENCHMARKED\n";
    pretrain(net, target, rng, EPOCHS);
    Lattice3D current = target;
    apply_virtual_flips(current, rng, FLIP_CENTERS);
    for (int step = 0; step < LOOP_STEPS; ++step) {
        const int incoming_error = bit_errors(current, target);
        const auto output = reconstruct(net, current, target);
        const int reconstructed_error = bit_errors(output.hard, target);
        // Feed model reconstruction back into the state, then perturb locally.
        current = output.hard;
        const int attempted = apply_virtual_flips(current, rng, FLIP_CENTERS);
        const int next_error = bit_errors(current, target);
        if (step % 5 == 0 || step == LOOP_STEPS - 1) {
            std::cout << "loop=" << step
                      << " incoming_bit_errors=" << incoming_error
                      << " reconstructed_bit_errors=" << reconstructed_error
                      << " next_bit_errors=" << next_error
                      << " BCE_vs_reference=" << std::fixed << std::setprecision(6)
                      << output.target_bce
                      << " latent_energy=" << output.latent_energy
                      << " attempted_toggles=" << attempted
                      << " state_FNV64=" << std::hex << lattice_hash(current) << std::dec << "\n";
        }
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    std::cout << "elapsed_wall_seconds=" << std::fixed << std::setprecision(3) << seconds
              << " (includes training; machine-specific)\n"
              << "LOOP_COMPLETE synthetic fixed-target experiment; no autonomous intelligence claim\n";
}
} // namespace moagi

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") moagi::self_test();
        else if (argc == 1) moagi::run();
        else { std::cerr << "Usage: moagi_ann3d [--self-test]\n"; return 2; }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR " << e.what() << '\n'; return 1;
    }
}
