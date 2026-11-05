// recursive_multimodal_engine.cpp
// Bounded recursive inward multimodal engine for Jarvis-X.
// Compile: g++ -std=c++17 -O3 -pthread -o recursive_engine recursive_multimodal_engine.cpp
//
// Mechanics:
//   X -> Q(X) -> E_theta -> Phi_in -> fixed-point latent core -> D_phi -> X_hat
//                                                    ^                  |
//                                                    |------ residual --|
//
// Notes:
// - The physical resident field is N^3 voxels. Larger spaces must be virtualized/tiled.
// - "Self-programming" here means bounded gradient updates of trainable parameters.
// - Fixed-point status is measured from the latent iteration residual; it is not assumed.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

constexpr int N = 16;                 // Physical resident edge; N^3 = 4096 voxels.
constexpr int CH_3D = 4;             // RGB + density / signed-distance-like scalar.
constexpr int CH_AUDIO = 2;          // Spatial audio amplitude channels.
constexpr int TOTAL_CH = CH_3D + CH_AUDIO;
constexpr int TILE_SIZE = 4;
constexpr int ENCODED_DIM = 256;
constexpr int LATENT_DIM = 128;
constexpr float LAMBDA_INIT = 0.73F;
constexpr float ETA_INIT = 0.002F;    // Conservative SGD rate for dense decoder.
constexpr int MAX_CYCLES = 64;
constexpr int FIXPOINT_STEPS = 32;
constexpr float FIXPOINT_TOL = 1.0e-5F;
constexpr float RECURRENT_MIX = 0.45F;
constexpr float RECURRENT_ROW_L1_MAX = 0.70F;
constexpr float PI_F = 3.14159265358979323846F;

struct Options {
    int cycles{MAX_CYCLES};
    float tolerance{1.0e-4F};
    bool quiet{false};
};

Options parse_options(int argc, char** argv) {
    Options options;
    auto need_value = [&](int& i, const string& flag) -> string {
        if (i + 1 >= argc) throw invalid_argument("missing value after " + flag);
        return string(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "--cycles") {
            options.cycles = stoi(need_value(i, arg));
            if (options.cycles < 1 || options.cycles > 10000) {
                throw invalid_argument("--cycles must be in [1,10000]");
            }
        } else if (arg == "--tolerance") {
            options.tolerance = stof(need_value(i, arg));
            if (!(options.tolerance > 0.0F) || !isfinite(options.tolerance)) {
                throw invalid_argument("--tolerance must be finite and > 0");
            }
        } else if (arg == "--quiet") {
            options.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            cout << "Usage: recursive_engine [--cycles N] [--tolerance X] [--quiet]\n";
            exit(0);
        } else {
            throw invalid_argument("unknown option: " + arg);
        }
    }
    return options;
}

class MultimodalField {
public:
    vector<float> data;
    int n{};
    int c{};

    explicit MultimodalField(int n_ = N, int c_ = TOTAL_CH) : n(n_), c(c_) {
        if (n <= 0 || c <= 0) throw invalid_argument("invalid field shape");
        data.assign(static_cast<size_t>(n) * static_cast<size_t>(n) *
                        static_cast<size_t>(n) * static_cast<size_t>(c),
                    0.0F);
    }

    float& operator()(int x, int y, int z, int ch) {
        return data[index(x, y, z, ch)];
    }

    const float& operator()(int x, int y, int z, int ch) const {
        return data[index(x, y, z, ch)];
    }

    size_t size() const { return data.size(); }

    float compute_mse(const MultimodalField& other) const {
        if (data.size() != other.data.size()) throw invalid_argument("field size mismatch");
        double sum = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            const double d = static_cast<double>(data[i]) - static_cast<double>(other.data[i]);
            sum += d * d;
        }
        return static_cast<float>(sum / static_cast<double>(data.size()));
    }

    void seed_multimodal_signals(mt19937& rng) {
        uniform_real_distribution<float> dist(-1.0F, 1.0F);
        const float inv = 1.0F / static_cast<float>(max(1, n - 1));

        for (int z = 0; z < n; ++z) {
            for (int y = 0; y < n; ++y) {
                for (int x = 0; x < n; ++x) {
                    for (int ch = 0; ch < CH_3D; ++ch) {
                        (*this)(x, y, z, ch) = dist(rng);
                    }

                    // Static samples of two spatial pressure waves. This is a bounded
                    // spatial-audio field, not a claim that the 3D axes are time samples.
                    const float nx = static_cast<float>(x) * inv;
                    const float ny = static_cast<float>(y) * inv;
                    const float nz = static_cast<float>(z) * inv;
                    const float phase_l = 2.0F * PI_F * (2.0F * nx + 3.0F * ny + 1.0F * nz);
                    const float phase_r =
                        2.0F * PI_F * (1.0F * nx + 2.0F * ny + 4.0F * nz) + 0.35F;
                    (*this)(x, y, z, CH_3D) = sinf(phase_l);
                    (*this)(x, y, z, CH_3D + 1) = sinf(phase_r);
                }
            }
        }
    }

private:
    size_t index(int x, int y, int z, int ch) const {
        if (x < 0 || y < 0 || z < 0 || ch < 0 || x >= n || y >= n || z >= n || ch >= c) {
            throw out_of_range("multimodal field index");
        }
        return (((static_cast<size_t>(z) * static_cast<size_t>(n) + static_cast<size_t>(y)) *
                     static_cast<size_t>(n) +
                 static_cast<size_t>(x)) *
                    static_cast<size_t>(c) +
                static_cast<size_t>(ch));
    }
};

struct NeuralLayer {
    int in_dim{};
    int out_dim{};
    vector<float> W;
    vector<float> b;

    NeuralLayer(int in_d, int out_d, uint32_t seed) : in_dim(in_d), out_dim(out_d) {
        if (in_dim <= 0 || out_dim <= 0) throw invalid_argument("invalid neural layer shape");
        W.resize(static_cast<size_t>(in_dim) * static_cast<size_t>(out_dim));
        b.resize(static_cast<size_t>(out_dim), 0.0F);
        const float scale = sqrtf(2.0F / static_cast<float>(in_dim + out_dim));
        mt19937 rng(seed);
        normal_distribution<float> nd(0.0F, scale);
        for (float& w : W) w = nd(rng);
    }

    vector<float> forward(const vector<float>& x) const {
        if (static_cast<int>(x.size()) != in_dim) throw invalid_argument("layer input size mismatch");
        vector<float> y(static_cast<size_t>(out_dim), 0.0F);
        for (int o = 0; o < out_dim; ++o) {
            float sum = b[static_cast<size_t>(o)];
            const size_t base = static_cast<size_t>(o) * static_cast<size_t>(in_dim);
            for (int i = 0; i < in_dim; ++i) {
                sum += W[base + static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
            }
            y[static_cast<size_t>(o)] = tanhf(sum);
        }
        return y;
    }

    vector<float> backward_update(const vector<float>& x,
                                  const vector<float>& y,
                                  const vector<float>& grad_output,
                                  float learning_rate) {
        if (static_cast<int>(x.size()) != in_dim || static_cast<int>(y.size()) != out_dim ||
            static_cast<int>(grad_output.size()) != out_dim) {
            throw invalid_argument("layer backward size mismatch");
        }

        vector<float> delta(static_cast<size_t>(out_dim), 0.0F);
        vector<float> grad_input(static_cast<size_t>(in_dim), 0.0F);
        for (int o = 0; o < out_dim; ++o) {
            const float yo = y[static_cast<size_t>(o)];
            delta[static_cast<size_t>(o)] =
                grad_output[static_cast<size_t>(o)] * (1.0F - yo * yo);
        }

        for (int o = 0; o < out_dim; ++o) {
            const float d = delta[static_cast<size_t>(o)];
            const size_t base = static_cast<size_t>(o) * static_cast<size_t>(in_dim);
            for (int i = 0; i < in_dim; ++i) {
                grad_input[static_cast<size_t>(i)] += W[base + static_cast<size_t>(i)] * d;
            }
        }

        for (int o = 0; o < out_dim; ++o) {
            const float d = delta[static_cast<size_t>(o)];
            const size_t base = static_cast<size_t>(o) * static_cast<size_t>(in_dim);
            for (int i = 0; i < in_dim; ++i) {
                W[base + static_cast<size_t>(i)] -=
                    learning_rate * d * x[static_cast<size_t>(i)];
            }
            b[static_cast<size_t>(o)] -= learning_rate * d;
        }
        return grad_input;
    }

    void bound_row_l1(float max_l1) {
        for (int o = 0; o < out_dim; ++o) {
            const size_t base = static_cast<size_t>(o) * static_cast<size_t>(in_dim);
            float l1 = 0.0F;
            for (int i = 0; i < in_dim; ++i) {
                l1 += fabsf(W[base + static_cast<size_t>(i)]);
            }
            if (l1 > max_l1 && l1 > 0.0F) {
                const float s = max_l1 / l1;
                for (int i = 0; i < in_dim; ++i) {
                    W[base + static_cast<size_t>(i)] *= s;
                }
            }
        }
    }
};

struct FixedPointResult {
    vector<float> z;
    int iterations{};
    float residual{numeric_limits<float>::infinity()};
    bool converged{};
};

class RecursiveMultimodalEngine {
public:
    MultimodalField target;
    MultimodalField reconstruction;
    NeuralLayer encoder;
    NeuralLayer latent_core_proj;
    NeuralLayer recurrent_core;
    NeuralLayer decoder;

    float lambda{LAMBDA_INIT};
    float eta{ETA_INIT};
    int cycle{};
    float current_mse{numeric_limits<float>::infinity()};
    float previous_mse{numeric_limits<float>::infinity()};
    float latent_radius{};
    int active_tiles{};
    bool latent_fixedpoint_converged{};
    int latent_fixedpoint_iterations{};
    float latent_fixedpoint_residual{numeric_limits<float>::infinity()};

    RecursiveMultimodalEngine()
        : target(N, TOTAL_CH),
          reconstruction(N, TOTAL_CH),
          encoder((N / 2) * (N / 2) * (N / 2) * TOTAL_CH, ENCODED_DIM, 0xE001U),
          latent_core_proj(ENCODED_DIM, LATENT_DIM, 0xE002U),
          recurrent_core(LATENT_DIM, LATENT_DIM, 0xE003U),
          decoder(LATENT_DIM, N * N * N * TOTAL_CH, 0xE004U) {
        mt19937 rng(1337U);
        target.seed_multimodal_signals(rng);
        recurrent_core.bound_row_l1(RECURRENT_ROW_L1_MAX);
        active_tiles = (N / TILE_SIZE) * (N / TILE_SIZE) * (N / TILE_SIZE);
    }

    vector<float> downsample_multimodal_field() const {
        const int sub_n = N / 2;
        vector<float> downsampled(
            static_cast<size_t>(sub_n) * static_cast<size_t>(sub_n) *
                static_cast<size_t>(sub_n) * static_cast<size_t>(TOTAL_CH),
            0.0F);

        for (int z = 0; z < sub_n; ++z) {
            for (int y = 0; y < sub_n; ++y) {
                for (int x = 0; x < sub_n; ++x) {
                    for (int ch = 0; ch < TOTAL_CH; ++ch) {
                        float sum = 0.0F;
                        for (int dz = 0; dz < 2; ++dz) {
                            for (int dy = 0; dy < 2; ++dy) {
                                for (int dx = 0; dx < 2; ++dx) {
                                    sum += target(x * 2 + dx, y * 2 + dy, z * 2 + dz, ch);
                                }
                            }
                        }
                        const size_t index =
                            (((static_cast<size_t>(z) * static_cast<size_t>(sub_n) +
                               static_cast<size_t>(y)) *
                                  static_cast<size_t>(sub_n) +
                              static_cast<size_t>(x)) *
                                 static_cast<size_t>(TOTAL_CH) +
                             static_cast<size_t>(ch));
                        downsampled[index] = sum * 0.125F;
                    }
                }
            }
        }
        return downsampled;
    }

    vector<float> inward_fold(const vector<float>& z) const {
        vector<float> folded(z.size(), 0.0F);
        const size_t sz = z.size();
        const float radial_scale = expf(-lambda);
        for (size_t i = 0; i < sz; ++i) {
            const size_t j = (i * 11U + 17U) % sz;
            folded[i] = 0.65F * radial_scale * z[i] + 0.35F * z[j];
        }
        return folded;
    }

    vector<float> inward_fold_backward(const vector<float>& grad_folded) const {
        vector<float> grad_encoded(grad_folded.size(), 0.0F);
        const size_t sz = grad_folded.size();
        const float radial_scale = expf(-lambda);
        for (size_t i = 0; i < sz; ++i) {
            const size_t j = (i * 11U + 17U) % sz;
            const float g = grad_folded[i];
            grad_encoded[i] += 0.65F * radial_scale * g;
            grad_encoded[j] += 0.35F * g;
        }
        return grad_encoded;
    }

    FixedPointResult solve_latent_fixedpoint(const vector<float>& anchor) const {
        if (anchor.size() != static_cast<size_t>(LATENT_DIM)) {
            throw invalid_argument("latent anchor size mismatch");
        }
        FixedPointResult result;
        result.z = anchor;

        for (int k = 0; k < FIXPOINT_STEPS; ++k) {
            const vector<float> rec = recurrent_core.forward(result.z);
            vector<float> next(result.z.size(), 0.0F);
            double d2 = 0.0;
            for (size_t i = 0; i < next.size(); ++i) {
                next[i] =
                    (1.0F - RECURRENT_MIX) * anchor[i] + RECURRENT_MIX * rec[i];
                const double d =
                    static_cast<double>(next[i]) - static_cast<double>(result.z[i]);
                d2 += d * d;
            }
            result.residual =
                static_cast<float>(sqrt(d2 / static_cast<double>(next.size())));
            result.z.swap(next);
            result.iterations = k + 1;
            if (result.residual <= FIXPOINT_TOL) {
                result.converged = true;
                break;
            }
        }
        return result;
    }

    MultimodalField decode_multimodal_field(const vector<float>& z_star,
                                            vector<float>& flat_out) const {
        flat_out = decoder.forward(z_star);
        MultimodalField decoded(N, TOTAL_CH);
        if (flat_out.size() != decoded.data.size()) {
            throw runtime_error("decoder output mismatch");
        }
        decoded.data = flat_out;
        return decoded;
    }

    void execute_recursive_cycle() {
        previous_mse = current_mse;

        const vector<float> downsampled = downsample_multimodal_field();
        const vector<float> z_encoded = encoder.forward(downsampled);
        const vector<float> z_folded = inward_fold(z_encoded);
        const vector<float> z_anchor = latent_core_proj.forward(z_folded);
        const FixedPointResult fp = solve_latent_fixedpoint(z_anchor);

        latent_fixedpoint_converged = fp.converged;
        latent_fixedpoint_iterations = fp.iterations;
        latent_fixedpoint_residual = fp.residual;

        double r2 = 0.0;
        for (float v : fp.z) {
            r2 += static_cast<double>(v) * static_cast<double>(v);
        }
        latent_radius =
            static_cast<float>(sqrt(r2 / static_cast<double>(fp.z.size())));

        vector<float> flat_out;
        reconstruction = decode_multimodal_field(fp.z, flat_out);
        current_mse = target.compute_mse(reconstruction);

        vector<float> grad_output(flat_out.size(), 0.0F);
        const float inv_count = 1.0F / static_cast<float>(flat_out.size());
        for (size_t i = 0; i < flat_out.size(); ++i) {
            grad_output[i] = 2.0F * (flat_out[i] - target.data[i]);
        }

        const vector<float> grad_z_star =
            decoder.backward_update(fp.z, flat_out, grad_output, eta);

        // Truncated fixed-point gradient: d z*/d anchor is approximated as I.
        const vector<float> grad_folded = latent_core_proj.backward_update(
            z_folded, z_anchor, grad_z_star, eta * inv_count * 0.35F);
        const vector<float> grad_encoded = inward_fold_backward(grad_folded);
        (void)encoder.backward_update(
            downsampled, z_encoded, grad_encoded, eta * inv_count * 0.20F);

        lambda = LAMBDA_INIT *
                 (1.0F - 0.35F * (current_mse / (current_mse + 1.0F)));
        eta = ETA_INIT / (1.0F + 0.02F * static_cast<float>(cycle + 1));
        ++cycle;
    }

    bool reconstruction_converged(float tolerance) const {
        return current_mse <= tolerance;
    }

    void render_telemetry() const {
        cout << fixed << setprecision(6)
             << "cycle=" << setw(3) << cycle
             << " voxels=" << (N * N * N)
             << " tiles=" << active_tiles
             << " mse=" << current_mse
             << " latent_r=" << latent_radius
             << " fp_residual=" << latent_fixedpoint_residual
             << " fp_steps=" << latent_fixedpoint_iterations
             << " fp=" << (latent_fixedpoint_converged ? "converged" : "open")
             << " lambda=" << lambda
             << " eta=" << eta << '\n';
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        if (!options.quiet) {
            cout << "RECURSIVE 3D MULTIMODAL AUTOENCODING ENGINE\n"
                 << "closed-loop residual learning + contractive latent fixed point\n\n";
        }

        RecursiveMultimodalEngine engine;
        const auto start = chrono::steady_clock::now();

        for (int i = 0; i < options.cycles; ++i) {
            engine.execute_recursive_cycle();
            if (!options.quiet &&
                (i == 0 || i + 1 == options.cycles || (i + 1) % 8 == 0)) {
                engine.render_telemetry();
            }
            if (engine.reconstruction_converged(options.tolerance)) break;
        }

        const auto end = chrono::steady_clock::now();
        const double elapsed_ms =
            chrono::duration<double, milli>(end - start).count();

        if (!options.quiet) {
            cout << "\nexecution_complete\n"
                 << "cycles=" << engine.cycle << '\n'
                 << "elapsed_ms=" << elapsed_ms << '\n'
                 << "mean_cycle_ms="
                 << (elapsed_ms / static_cast<double>(max(1, engine.cycle))) << '\n'
                 << "final_mse=" << engine.current_mse << '\n'
                 << "latent_fixedpoint="
                 << (engine.latent_fixedpoint_converged ? "converged"
                                                       : "not_converged")
                 << '\n'
                 << "reconstruction="
                 << (engine.reconstruction_converged(options.tolerance)
                         ? "within_tolerance"
                         : "above_tolerance")
                 << '\n';
        }

        if (!isfinite(engine.current_mse) || !engine.latent_fixedpoint_converged) {
            return 2;
        }
        return 0;
    } catch (const exception& error) {
        cerr << "recursive_multimodal_engine failure: " << error.what() << '\n';
        return 1;
    }
}
