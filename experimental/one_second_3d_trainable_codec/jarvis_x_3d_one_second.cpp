// JARVIS X | one-shot geometrically 3D trainable autoencoder and exact residual codec.
// C++17, no external dependencies. This is a bounded CPU research demonstration,
// not a pretrained foundation model or a universal one-second performance guarantee.
// Build: g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror jarvis_x_3d_one_second.cpp -o jx3d
// Run:   ./jx3d --budget-ms 1000 --epochs 120 --output jx3d
// Tests: ./jx3d --self-test
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace jx3d {
using Clock = std::chrono::steady_clock;
constexpr int N = 16;                 // Side length of spatial 3-torus.
constexpr int H = N / 2;             // Side length of encoded coarse grid.
constexpr int Voxels = N * N * N;    // 4,096 physical source voxels.
constexpr int Cells = H * H * H;     // 512 latent cells.
constexpr int Phases = 8;           // Eight sub-voxels per encoded cell.
constexpr int Kernel = 27;          // 3x3x3 periodic coarse-neighborhood.
constexpr int Parameters = Phases * (Kernel + 1);  // 224 trainable parameters.
constexpr double Pi = 3.141592653589793238462643383279502884;
using Volume = std::array<std::uint8_t, Voxels>;
using Latent = std::array<std::uint8_t, Cells>;
using Residual = std::array<std::int16_t, Voxels>;
using Prediction = std::array<float, Voxels>;

int wrap(int n, int side) { return (n + side) % side; }  // n is at least -1.
int idx(int x, int y, int z, int side) { return (z * side + y) * side + x; }
int phase(int x, int y, int z) { return (x & 1) | ((y & 1) << 1) | ((z & 1) << 2); }
float clip(float v) { return std::max(0.0f, std::min(1.0f, v)); }
std::uint8_t byte(float v) { return static_cast<std::uint8_t>(std::lround(255.0f * clip(v))); }

struct Prng {
    std::uint64_t state;
    explicit Prng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        auto z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

// Synthetic, spatially periodic hold-out data. All training fields use a
// distinct phase seed. This makes the experiment reproducible, not general.
Volume synthetic(std::uint64_t seed) {
    Prng rng(seed);
    const double px = 2.0 * Pi * rng.uniform();
    const double py = 2.0 * Pi * rng.uniform();
    const double pz = 2.0 * Pi * rng.uniform();
    const double pm = 2.0 * Pi * rng.uniform();
    const double ph = 2.0 * Pi * rng.uniform();
    Volume v{};
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const double a = 2.0 * Pi / static_cast<double>(N);
            const double value = 0.5
              + 0.13 * std::sin(a * x + px)
              + 0.11 * std::cos(a * y + py)
              + 0.08 * std::sin(a * z + pz)
              + 0.07 * std::sin(a * (x + y + z) + pm)
              + 0.025 * std::cos(2.0 * a * (x - y) + ph);
            v[static_cast<std::size_t>(idx(x,y,z,N))] = byte(static_cast<float>(value));
        }
    return v;
}

// Explicit out-of-distribution check: independent random 3D bytes.
Volume random_volume(std::uint64_t seed) {
    Prng rng(seed);
    Volume v{};
    for (auto& element : v)
        element = static_cast<std::uint8_t>(rng.next() & 0xffULL);
    return v;
}

struct Encoded {
    Latent latent{};
    Residual residual{};
};

Encoded encode(const Volume& x) {
    Encoded e;
    for (int z = 0; z < H; ++z)
      for (int y = 0; y < H; ++y)
        for (int a = 0; a < H; ++a) {
            int sum = 0;
            for (int oz = 0; oz < 2; ++oz)
              for (int oy = 0; oy < 2; ++oy)
                for (int ox = 0; ox < 2; ++ox)
                    sum += x[static_cast<std::size_t>(idx(2*a+ox,2*y+oy,2*z+oz,N))];
            const std::uint8_t c = static_cast<std::uint8_t>(sum / 8);
            e.latent[static_cast<std::size_t>(idx(a,y,z,H))] = c;
            for (int oz = 0; oz < 2; ++oz)
              for (int oy = 0; oy < 2; ++oy)
                for (int ox = 0; ox < 2; ++ox) {
                    const int i = idx(2*a+ox,2*y+oy,2*z+oz,N);
                    e.residual[static_cast<std::size_t>(i)] =
                        static_cast<std::int16_t>(static_cast<int>(x[static_cast<std::size_t>(i)]) - static_cast<int>(c));
                }
        }
    return e;
}

Volume decode_exact(const Encoded& e) {
    Volume output{};
    for (int z = 0; z < N; ++z)
      for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            const int i = idx(x,y,z,N);
            const int c = idx(x/2,y/2,z/2,H);
            const int v = static_cast<int>(e.latent[static_cast<std::size_t>(c)])
                        + static_cast<int>(e.residual[static_cast<std::size_t>(i)]);
            if (v < 0 || v > 255) throw std::runtime_error("decoded voxel exceeds byte range");
            output[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(v);
        }
    return output;
}

// Neural decoder feature: 27 relative intensities on the physical 3D torus.
struct Cell {
    std::array<float, Kernel> neighbor_delta{};
    std::array<float, Phases> target{};
    float base = 0.0f;
};
using Dataset = std::vector<Cell>;

Dataset build_dataset(const Volume& source) {
    const Encoded e = encode(source);
    Dataset data;
    data.reserve(Cells);
    for (int z = 0; z < H; ++z)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < H; ++x) {
            Cell cell;
            const int center = idx(x,y,z,H);
            const int base = e.latent[static_cast<std::size_t>(center)];
            cell.base = static_cast<float>(base) / 255.0f;
            int k = 0;
            for (int dz = -1; dz <= 1; ++dz)
              for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int j = idx(wrap(x+dx,H),wrap(y+dy,H),wrap(z+dz,H),H);
                    cell.neighbor_delta[static_cast<std::size_t>(k++)] =
                        static_cast<float>(static_cast<int>(e.latent[static_cast<std::size_t>(j)]) - base) / 255.0f;
                }
            for (int oz = 0; oz < 2; ++oz)
              for (int oy = 0; oy < 2; ++oy)
                for (int ox = 0; ox < 2; ++ox) {
                    const int p = phase(ox,oy,oz);
                    cell.target[static_cast<std::size_t>(p)] =
                        static_cast<float>(source[static_cast<std::size_t>(idx(2*x+ox,2*y+oy,2*z+oz,N))]) / 255.0f;
                }
            data.push_back(cell);
        }
    return data;
}

struct Decoder {
    std::array<float, Parameters> weights{}; // 8 learned 3x3x3 filters plus 8 biases.
    float predict(const Cell& cell, int p) const {
        const int begin = p * (Kernel + 1);
        float z = weights[static_cast<std::size_t>(begin + Kernel)];
        for (int k = 0; k < Kernel; ++k)
            z += weights[static_cast<std::size_t>(begin + k)] * cell.neighbor_delta[static_cast<std::size_t>(k)];
        return clip(cell.base + 0.32f * std::tanh(z));
    }
};

double mse(const std::vector<Dataset>& sets, const Decoder& decoder) {
    double sum = 0.0;
    std::size_t count = 0;
    for (const Dataset& set : sets)
      for (const Cell& cell : set)
        for (int p = 0; p < Phases; ++p) {
            const double diff = static_cast<double>(decoder.predict(cell,p) - cell.target[static_cast<std::size_t>(p)]);
            sum += diff * diff;
            ++count;
        }
    return sum / static_cast<double>(count);
}

// Full-batch Adam; gradients are analytic through tanh and clipping.
struct Optimizer {
    std::array<double, Parameters> first{};
    std::array<double, Parameters> second{};
    int iteration = 0;
    void epoch(Decoder& model, const std::vector<Dataset>& training) {
        std::array<double, Parameters> grad{};
        std::size_t count = 0;
        for (const Dataset& set : training)
          for (const Cell& cell : set)
            for (int p = 0; p < Phases; ++p) {
                const int base = p * (Kernel + 1);
                double z = static_cast<double>(model.weights[static_cast<std::size_t>(base+Kernel)]);
                for (int k = 0; k < Kernel; ++k)
                    z += static_cast<double>(model.weights[static_cast<std::size_t>(base+k)]) * cell.neighbor_delta[static_cast<std::size_t>(k)];
                const double a = std::tanh(z);
                const double value = static_cast<double>(cell.base) + 0.32 * a;
                const double prediction = std::min(1.0, std::max(0.0, value));
                const double error = prediction - cell.target[static_cast<std::size_t>(p)];
                const double d = (value > 0.0 && value < 1.0)
                    ? 2.0 * error * 0.32 * (1.0-a*a) : 0.0;
                grad[static_cast<std::size_t>(base+Kernel)] += d;
                for (int k = 0; k < Kernel; ++k)
                    grad[static_cast<std::size_t>(base+k)] += d * cell.neighbor_delta[static_cast<std::size_t>(k)];
                ++count;
            }
        ++iteration;
        const double n = static_cast<double>(count);
        const double bc1 = 1.0 - std::pow(0.9, iteration);
        const double bc2 = 1.0 - std::pow(0.999, iteration);
        for (int k = 0; k < Parameters; ++k) {
            const auto i = static_cast<std::size_t>(k);
            const double g = grad[i] / n + 1e-6 * model.weights[i];
            first[i] = 0.9 * first[i] + 0.1 * g;
            second[i] = 0.999 * second[i] + 0.001 * g*g;
            const double update = 0.032 * (first[i] / bc1) / (std::sqrt(second[i] / bc2) + 1e-8);
            model.weights[i] = static_cast<float>(static_cast<double>(model.weights[i]) - update);
            if (!std::isfinite(model.weights[i]) || std::abs(model.weights[i]) > 100.0f)
                throw std::runtime_error("unstable neural decoder parameters");
        }
    }
};

Prediction project(const Dataset& set, const Decoder& model) {
    if (set.size() != static_cast<std::size_t>(Cells))
        throw std::runtime_error("incorrect encoded 3D grid size");
    Prediction v{};
    for (int z = 0; z < H; ++z)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < H; ++x) {
            const Cell& c = set[static_cast<std::size_t>(idx(x,y,z,H))];
            for (int oz = 0; oz < 2; ++oz)
              for (int oy = 0; oy < 2; ++oy)
                for (int ox = 0; ox < 2; ++ox)
                    v[static_cast<std::size_t>(idx(2*x+ox,2*y+oy,2*z+oz,N))] = model.predict(c,phase(ox,oy,oz));
        }
    return v;
}

double rms_to(const Prediction& v, const Volume& target) {
    double sum = 0.0;
    for (int i = 0; i < Voxels; ++i) {
        const double diff = static_cast<double>(v[static_cast<std::size_t>(i)]) - static_cast<double>(target[static_cast<std::size_t>(i)]) / 255.0;
        sum += diff*diff;
    }
    return std::sqrt(sum/static_cast<double>(Voxels));
}

struct Feedback {
    Prediction state{};
    double max_fixed_point_error = 0.0;
};

Feedback controlled_feedback(const Prediction& start, const Prediction& destination, int steps) {
    Feedback f{start,0.0};
    const double initial = [&](){
        double m = 0.0;
        for(int i=0;i<Voxels;++i) m=std::max(m,std::abs(static_cast<double>(start[static_cast<std::size_t>(i)]-destination[static_cast<std::size_t>(i)])));
        return m;
    }();
    for (int t = 0; t < steps; ++t)
        for (int i = 0; i < Voxels; ++i) {
            auto& value = f.state[static_cast<std::size_t>(i)];
            value = 0.5f * value + 0.5f * destination[static_cast<std::size_t>(i)];
            if (!(value >= 0.0f && value <= 1.0f))
                throw std::runtime_error("feedback escaped the admissible [0,1] region");
        }
    for(int i=0;i<Voxels;++i)
        f.max_fixed_point_error = std::max(f.max_fixed_point_error,std::abs(static_cast<double>(f.state[static_cast<std::size_t>(i)]-destination[static_cast<std::size_t>(i)])));
    if (f.max_fixed_point_error > initial * std::pow(0.5,steps) + 5e-7)
        throw std::runtime_error("feedback contraction check failed");
    return f;
}

struct Args {
    int budget_ms = 1000;
    int max_epochs = 120;
    std::uint64_t seed = 20261120ULL;
    std::string output;
    bool self_test = false;
};

int parse_positive(const std::string& s, int upper) {
    if(s.empty() || s[0]=='-') throw std::runtime_error("negative or empty value");
    std::size_t used=0;
    const long long v=std::stoll(s,&used,10);
    if (used != s.size() || v <= 0 || v > upper) throw std::runtime_error("out-of-range numeric value");
    return static_cast<int>(v);
}
Args parse(int argc, char** argv) {
    Args opts;
    for (int i=1; i<argc; ++i) {
        const std::string key=argv[i];
        if(key=="--self-test") {opts.self_test=true;continue;}
        if (i+1 >= argc) throw std::runtime_error("missing value for "+key);
        const std::string value=argv[++i];
        if(key=="--budget-ms") opts.budget_ms=parse_positive(value,60000);
        else if(key=="--epochs") opts.max_epochs=parse_positive(value,5000);
        else if(key=="--output") opts.output=value;
        else if(key=="--seed") { 
            if(value.empty() || value[0]=='-')throw std::runtime_error("negative seed");
            std::size_t used=0; opts.seed=std::stoull(value,&used); if(used!=value.size())throw std::runtime_error("bad seed");
        } else throw std::runtime_error("unknown option: "+key);
    }
    return opts;
}

struct Result {
    int epochs=0;
    double time_ms=0;
    double train_mse_before=0, train_mse_after=0;
    double heldout_baseline_rmse=0,heldout_learned_rmse=0,heldout_feedback_rmse=0;
    double feedback_fixedpoint_max_error=0;
    double ood_baseline_rmse=0, ood_learned_rmse=0;
    std::uint64_t exact_voxels=0;
    bool within_budget=false;
};

Result run(const Args& args) {
    const auto start=Clock::now();
    const auto elapsed=[&](){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    const std::uint64_t base_seed=args.seed;
    std::vector<Dataset> train;
    train.reserve(6);
    for(int i=0;i<6;++i)
        train.push_back(build_dataset(synthetic(base_seed + static_cast<std::uint64_t>(i)*7919ULL)));
    const Volume holdout=synthetic(base_seed+900001ULL);
    const Dataset holdout_data=build_dataset(holdout);
    Decoder model;
    const double before=mse(train,model);
    Optimizer optimizer;
    int epochs=0;
    for (;epochs < args.max_epochs; ++epochs) {
        // Leave time to verify exact inverse and export measurements.
        if (elapsed() >= 0.70 * static_cast<double>(args.budget_ms)) break;
        optimizer.epoch(model,train);
    }
    const double after=mse(train,model);
    const Decoder baseline{};
    const auto random_source=random_volume(base_seed+1200001ULL);
    const auto random_data=build_dataset(random_source);
    const Prediction random_baseline=project(random_data,baseline);
    const Prediction random_learned=project(random_data,model);
    const Prediction untrained=project(holdout_data,baseline);
    const Prediction learned=project(holdout_data,model);
    const Feedback recurrent=controlled_feedback(untrained,learned,6);
    const Encoded encoded=encode(holdout);
    const Volume exact=decode_exact(encoded);
    std::uint64_t exact_voxels=0;
    for (int i=0; i<Voxels; ++i)
        if(exact[static_cast<std::size_t>(i)]==holdout[static_cast<std::size_t>(i)])++exact_voxels;
    if(exact_voxels!=static_cast<std::uint64_t>(Voxels))
        throw std::runtime_error("lossless codec roundtrip failed");
    if(!args.output.empty()) {
        std::ofstream out(args.output+"_voxels.csv",std::ios::binary);
        if(!out)throw std::runtime_error("cannot create voxel CSV");
        out << "x,y,z,source,coarse,decoded_learned,feedback,lossless_recovered\n" << std::setprecision(9);
        for(int z=0;z<N;++z)
          for(int y=0;y<N;++y)
            for(int x=0;x<N;++x){
                const auto i=static_cast<std::size_t>(idx(x,y,z,N));
                out << x << ',' << y << ',' << z << ',' << static_cast<int>(holdout[i]) << ','
                    << untrained[i] << ',' << learned[i] << ',' << recurrent.state[i] << ','
                    << static_cast<int>(exact[i]) << '\n';
            }
        if(!out)throw std::runtime_error("failed writing voxel CSV");
    }
    Result r;
    r.epochs=epochs;
    r.train_mse_before=before; r.train_mse_after=after;
    r.heldout_baseline_rmse=rms_to(untrained,holdout);
    r.heldout_learned_rmse=rms_to(learned,holdout);
    r.heldout_feedback_rmse=rms_to(recurrent.state,holdout);
    r.feedback_fixedpoint_max_error=recurrent.max_fixed_point_error;
    r.ood_baseline_rmse=rms_to(random_baseline,random_source);
    r.ood_learned_rmse=rms_to(random_learned,random_source);
    r.exact_voxels=exact_voxels;
    r.time_ms=elapsed();
    r.within_budget=r.time_ms <= static_cast<double>(args.budget_ms);
    return r;
}

void print(const Result& r,const Args& args) {
    std::cout << std::fixed << std::setprecision(7)
      << "JARVIS_X_3D_ONE_SHOT\n"
      << "spatial_torus=" << N << 'x' << N << 'x' << N << "  latent=" << H << 'x' << H << 'x' << H << "  parameters=" << Parameters << '\n'
      << "training_fields=6 heldout_fields=1  epochs=" << r.epochs << '\n'
      << "training_MSE_before=" << r.train_mse_before << " after=" << r.train_mse_after << '\n'
      << "heldout_baseline_RMSE=" << r.heldout_baseline_rmse << '\n'
      << "heldout_learned_RMSE=" << r.heldout_learned_rmse << '\n'
      << "heldout_feedback6_RMSE=" << r.heldout_feedback_rmse << '\n'
      << "learned_improvement_percent=" << 100.0*(1.0-r.heldout_learned_rmse/r.heldout_baseline_rmse) << '\n'
      << "feedback_fixedpoint_Linf=" << r.feedback_fixedpoint_max_error << '\n'
      << "OOD_random_baseline_RMSE=" << r.ood_baseline_rmse << '\n'
      << "OOD_random_learned_RMSE=" << r.ood_learned_rmse << '\n'
      << "lossless_roundtrip_voxels=" << r.exact_voxels << '/' << Voxels << '\n'
      << "elapsed_ms=" << r.time_ms << " budget_ms=" << args.budget_ms
      << " within_budget=" << (r.within_budget?"YES":"NO") << '\n';
}

void self_test() {
    for (std::uint64_t seed : {1ULL, 42ULL, 20261120ULL}) {
        const auto original = synthetic(seed);
        const auto e=encode(original);
        if(decode_exact(e)!=original)throw std::runtime_error("codec test failed");
        const auto d=build_dataset(original);
        if(d.size()!=static_cast<std::size_t>(Cells))throw std::runtime_error("geometry test failed");
        for (const Cell& c : d) {
            if(c.base < 0 || c.base > 1)throw std::runtime_error("bad encoded value");
            for(float f : c.neighbor_delta) if(!std::isfinite(f))throw std::runtime_error("invalid 3D feature");
        }
    }
    Args a; a.max_epochs=60; a.budget_ms=5000;
    const Result r=run(a);
    if (r.train_mse_after >= r.train_mse_before)throw std::runtime_error("no measurable learning on train split");
    if (r.heldout_learned_rmse >= r.heldout_baseline_rmse)throw std::runtime_error("no improvement on held-out split");
    if (r.exact_voxels!=static_cast<std::uint64_t>(Voxels))throw std::runtime_error("incorrect roundtrip count");
    if (r.feedback_fixedpoint_max_error > 0.01)throw std::runtime_error("feedback failed to approach fixed point");
    std::cout << "SELF_TEST PASS: periodic geometry, deterministic encoding, exact inverse, trained held-out gain, bounded contracting feedback\n";
}
} // namespace jx3d

int main(int argc,char** argv) {
    try {
        const auto args=jx3d::parse(argc,argv);
        if(args.self_test){jx3d::self_test();return 0;}
        const auto result=jx3d::run(args);
        jx3d::print(result,args);
        return result.within_budget?0:2;
    } catch(const std::exception& e){
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}