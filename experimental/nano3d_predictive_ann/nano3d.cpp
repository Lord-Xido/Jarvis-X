// Jarvis X Nano3D: bounded 3D ANN-precomputed code-template predictor.
// Research demonstrator. Prediction lookup is NOT neural inference or compile-time validation.
// Build: g++ -O3 -std=c++17 -Wall -Wextra -Wpedantic nano3d.cpp -o nano3d
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <limits>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nano3d {
constexpr unsigned SIDE = 32;
constexpr unsigned CELL_COUNT = SIDE * SIDE * SIDE;
constexpr unsigned N_CLASS = 8;
constexpr unsigned FEATURE_DIM = 3;
using Clock = std::chrono::steady_clock;

uint32_t xorshift32(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return s;
}

uint16_t morton3_5(unsigned x, unsigned y, unsigned z) {
    uint16_t out = 0;
    for (unsigned bit = 0; bit < 5; ++bit) {
        out |= static_cast<uint16_t>(((x >> bit) & 1u) << (3*bit));
        out |= static_cast<uint16_t>(((y >> bit) & 1u) << (3*bit + 1));
        out |= static_cast<uint16_t>(((z >> bit) & 1u) << (3*bit + 2));
    }
    return out;
}

unsigned oracle_class(unsigned x, unsigned y, unsigned z) {
    return (x >= SIDE / 2u ? 4u : 0u)
         | (y >= SIDE / 2u ? 2u : 0u)
         | (z >= SIDE / 2u ? 1u : 0u);
}

struct Point { uint8_t x,y,z; };
struct Prediction { uint8_t kind, confidence; };
static_assert(sizeof(Prediction) == 2, "Compact 2-byte prediction record");

struct ANN {
    std::array<std::array<double, FEATURE_DIM>, N_CLASS> w{};
    std::array<double, N_CLASS> bias{};

    static std::array<double, 3> features(unsigned x, unsigned y, unsigned z) {
        return {{(2.0*x+1.0)/SIDE-1.0, (2.0*y+1.0)/SIDE-1.0, (2.0*z+1.0)/SIDE-1.0}};
    }
    std::array<double, N_CLASS> logits(const std::array<double, 3>& f) const {
        std::array<double, N_CLASS> a{};
        for (unsigned c=0; c<N_CLASS; ++c) {
            a[c]=bias[c];
            for (unsigned j=0; j<3; ++j) a[c]+=w[c][j]*f[j];
        }
        return a;
    }
    Prediction predict(unsigned x, unsigned y, unsigned z) const {
        const auto a=logits(features(x,y,z));
        const auto best=static_cast<unsigned>(std::max_element(a.begin(), a.end())-a.begin());
        double sum=0.0;
        for (auto v : a) sum+=std::exp(v-a[best]);
        const double probability=1.0/sum;
        const auto conf=static_cast<unsigned>(std::clamp(std::lround(255.0*probability),0l,255l));
        return {static_cast<uint8_t>(best),static_cast<uint8_t>(conf)};
    }
    void train(unsigned epochs=350, unsigned samples_per_epoch=4096) {
        uint32_t rng=0xC0FFEE42u;
        for (unsigned epoch=0; epoch<epochs; ++epoch) {
            const double eta=0.025/(1.0+epoch/200.0);
            for (unsigned n=0;n<samples_per_epoch;++n) {
                const unsigned x=xorshift32(rng)&31u;
                const unsigned y=xorshift32(rng)&31u;
                const unsigned z=xorshift32(rng)&31u;
                const unsigned correct=oracle_class(x,y,z);
                const auto f=features(x,y,z);
                auto a=logits(f);
                const double mx=*std::max_element(a.begin(),a.end());
                double denom=0;
                for (auto& v : a) { v=std::exp(v-mx); denom+=v; }
                for (unsigned c=0;c<N_CLASS;++c) {
                    const double grad=(a[c]/denom-(c==correct?1.0:0.0));
                    for (unsigned j=0;j<3;++j) w[c][j]-=eta*grad*f[j];
                    bias[c]-=eta*grad;
                }
            }
        }
    }
};

struct Engine {
    ANN net;
    std::array<Prediction, CELL_COUNT> table{};
    std::array<Point, CELL_COUNT> by_address{};
    unsigned correct = 0;

    void precompute() {
        correct=0;
        for (unsigned z=0;z<SIDE;++z)
        for (unsigned y=0;y<SIDE;++y)
        for (unsigned x=0;x<SIDE;++x) {
            const auto addr=morton3_5(x,y,z);
            by_address[addr]={static_cast<uint8_t>(x),static_cast<uint8_t>(y),static_cast<uint8_t>(z)};
            table[addr]=net.predict(x,y,z);
            correct += table[addr].kind==oracle_class(x,y,z);
        }
    }

    void self_test() const {
        std::vector<bool> seen(CELL_COUNT, false);
        for (unsigned z=0;z<SIDE;++z)
        for (unsigned y=0;y<SIDE;++y)
        for (unsigned x=0;x<SIDE;++x) {
            const auto addr=morton3_5(x,y,z);
            if (addr >= CELL_COUNT || seen[addr]) throw std::runtime_error("Morton alias");
            seen[addr]=true;
            const auto point=by_address[addr];
            if (point.x!=x || point.y!=y || point.z!=z) throw std::runtime_error("address decode mismatch");
            if (table[addr].kind>=N_CLASS) throw std::runtime_error("invalid classification");
        }
        if (std::find(seen.begin(),seen.end(),false)!=seen.end()) throw std::runtime_error("Morton missing address");
    }
};

// Eight verified C++ expression templates. All are total over signed integer inputs
// by converting to uint32_t where arithmetic may overflow.
const char* expr(unsigned cls) {
    static const char* const expressions[N_CLASS]={
        "a + b", "a ^ b", "a | b", "a & b",
        "a - b", "a * 3u + b", "(a << 1u) ^ b", "(a >> 1u) + b"
    };
    return expressions[cls];
}

void emit(const Engine& e, const std::filesystem::path& out_dir, unsigned file_count, unsigned lines_per_file) {
    if (file_count > 10000 || lines_per_file > 100000 || file_count==0 || lines_per_file==0 ||
        static_cast<uint64_t>(file_count)*lines_per_file > 1000000ULL)
        throw std::invalid_argument("bounded files and lines must be positive");
    std::filesystem::create_directories(out_dir);
    uint64_t source_lines=0;
    uint64_t pred_mismatches=0;
    for (unsigned f=0;f<file_count;++f) {
        char name[32];
        std::snprintf(name,sizeof(name),"generated_%05u.cpp",f);
        std::ofstream stream(out_dir/name, std::ios::binary);
        if (!stream) throw std::runtime_error("failed to open source output");
        for (unsigned line=0;line<lines_per_file;++line) {
            // Mapping from a deterministic task ID to 3D features; not a learned task parser.
            uint32_t state=0x9E3779B9u ^ static_cast<uint32_t>(f*131071u+line*8191u);
            const unsigned x=xorshift32(state)&31u;
            const unsigned y=xorshift32(state)&31u;
            const unsigned z=xorshift32(state)&31u;
            const auto pred=e.table[morton3_5(x,y,z)];
            pred_mismatches += pred.kind!=oracle_class(x,y,z);
            // Each emitted LOC is an independent valid C++ function definition.
            stream<<"constexpr unsigned jx_"<<f<<"_"<<line
                  <<"(unsigned a,unsigned b) noexcept { return "<<expr(pred.kind)<<"; }\n";
            ++source_lines;
        }
        stream.flush();
        if (!stream) throw std::runtime_error("output write failed");
    }
    std::cout<<"emitted_files="<<file_count<<" emitted_cpp_lines="<<source_lines
             <<" synthetic_prediction_mismatches="<<pred_mismatches<<'\n';
}

struct BenchResult { double median_ns; uint64_t checksum; };
template <class F>
BenchResult benchmark(F fn, unsigned rounds) {
    std::vector<double> times;
    uint64_t overall=0;
    for (unsigned r=0;r<rounds;++r) {
        const auto begin=Clock::now();
        const auto value=fn();
        const auto end=Clock::now();
        overall^=value;
        const double sec=std::chrono::duration<double>(end-begin).count();
        times.push_back(sec);
    }
    std::sort(times.begin(),times.end());
    return {times[times.size()/2]*1e9,overall};
}

void run_benchmark(const Engine& e, unsigned n=2000000) {
    if(n==0 || n>20000000) throw std::invalid_argument("bench iterations must be 1..20,000,000");
    std::vector<uint16_t> addresses(n);
    uint32_t rng=0x12345678u;
    for (auto& a: addresses) a=static_cast<uint16_t>(xorshift32(rng)&(CELL_COUNT-1));
    const auto cached=benchmark([&](){
        uint64_t checksum=0;
        for (const auto addr:addresses) checksum+=(static_cast<uint64_t>(e.table[addr].kind)+1u);
        return checksum;
    },7);
    const auto online=benchmark([&](){
        uint64_t checksum=0;
        for (const auto addr:addresses) {
            const auto p=e.by_address[addr];
            checksum+=static_cast<uint64_t>(e.net.predict(p.x,p.y,p.z).kind)+1u;
        }
        return checksum;
    },7);
    std::cout<<std::fixed<<std::setprecision(4)
       <<"cached_lookup_median_ns_per_query="<<cached.median_ns/n<<'\n'
       <<"online_ann_median_ns_per_query="<<online.median_ns/n<<'\n'
       <<"cached_queries_per_s="<<n/(cached.median_ns/1e9)<<'\n'
       <<"workload=synthetic 32^3 ANN octant classification; fully hot local CPU, no I/O or compiler\n";
    // The checksum is printed to make measurements observable and guard elimination.
    std::cout<<"benchmark_checksum="<<cached.checksum<<":"<<online.checksum<<'\n';
}

} // namespace nano3d

unsigned parse_positive(const std::string& v) {
    if (v.empty() || !std::all_of(v.begin(), v.end(), [](unsigned char c) { return c >= '0' && c <= '9'; }))
        throw std::invalid_argument("expected positive decimal count");
    const unsigned long n = std::stoul(v);
    if (n == 0 || n > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("count out of range");
    return static_cast<unsigned>(n);
}

int main(int argc, char** argv) {
    try {
        unsigned files=8, lines=850, loops=2000000;
        std::filesystem::path out="generated";
        bool no_emit=false;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if ((arg=="--files" || arg=="--lines" || arg=="--loops" || arg=="--out") && i+1>=argc)
                throw std::invalid_argument("missing value for "+arg);
            if (arg=="--files") files=parse_positive(argv[++i]);
            else if (arg=="--lines") lines=parse_positive(argv[++i]);
            else if (arg=="--loops") loops=parse_positive(argv[++i]);
            else if (arg=="--out") out=argv[++i];
            else if (arg=="--no-emit") no_emit=true;
            else throw std::invalid_argument("unknown argument: "+arg);
        }
        nano3d::Engine engine;
        const auto start=nano3d::Clock::now();
        engine.net.train();
        const auto trained=nano3d::Clock::now();
        engine.precompute();
        engine.self_test();
        const auto done=nano3d::Clock::now();
        std::cout<<"grid_side="<<nano3d::SIDE<<" cells="<<nano3d::CELL_COUNT
                 <<" table_bytes="<<sizeof(engine.table)<<'\n';
        std::cout<<"trained_correct="<<engine.correct<<"/"<<nano3d::CELL_COUNT<<'\n';
        std::cout<<std::fixed<<std::setprecision(6)
                 <<"training_seconds="<<std::chrono::duration<double>(trained-start).count()<<'\n'
                 <<"precompute_and_test_seconds="<<std::chrono::duration<double>(done-trained).count()<<'\n';
        nano3d::run_benchmark(engine,loops);
        if (!no_emit) {
            const auto emit_begin=nano3d::Clock::now();
            nano3d::emit(engine,out,files,lines);
            const auto emit_end=nano3d::Clock::now();
            const double emit_seconds=std::chrono::duration<double>(emit_end-emit_begin).count();
            std::cout<<std::fixed<<std::setprecision(6)
                     <<"emit_seconds="<<emit_seconds<<'\n'
                     <<"template_loc_per_s="<<(static_cast<double>(files)*lines/emit_seconds)<<'\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<'\n';
        return 1;
    }
}