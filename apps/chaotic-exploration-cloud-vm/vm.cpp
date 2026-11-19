#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// JARVIS X Phi3D v2: bounded, explicitly interpreted 3D CPU reference VM.
// The fixed program is an application-level ISA; it is not a hardware hypervisor
// or a trained GRU/Transformer. See MATHEMATICS.md for the exact recurrence.
namespace {
constexpr double PI = 3.141592653589793238462643383279502884;
enum class Op : uint8_t {
    INGEST=0x10, LORENZ=0x20, ATTEND_GRU=0x30,
    FOLD_3D=0x40, PLAN=0x50, VERIFY=0x60, HALT=0xff
};
constexpr std::array<Op,7> PROGRAM = {Op::INGEST, Op::LORENZ, Op::ATTEND_GRU,
                                     Op::FOLD_3D, Op::PLAN, Op::VERIFY, Op::HALT};
struct Config {
    int side=12, steps=8, seed=42, horizon=8, input_stdin=0, emit_field=0, trace=1;
    double alpha=.09, beta=.12, chaos=.03, epsilon=.0001, decay=.94;
};
int integer(const std::string &s, int low, int high) {
    std::size_t end=0;
    const long long n=std::stoll(s,&end);
    if(end!=s.size() || n<low || n>high) throw std::invalid_argument("integer out of range");
    return static_cast<int>(n);
}
double real(const std::string &s,double low,double high) {
    std::size_t end=0;
    const double x=std::stod(s,&end);
    if(end!=s.size() || !std::isfinite(x) || x<low || x>high)
        throw std::invalid_argument("float out of range");
    return x;
}
std::uint64_t splitmix64(std::uint64_t &x) {
    std::uint64_t z=(x+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
double sigmoid(double x) { return 1.0/(1.0+std::exp(-x)); }
using Vec3 = std::array<double,3>;
Vec3 add(Vec3 a, Vec3 b, double scale=1.0) {
    for(std::size_t i=0;i<3;++i) a[i]+=scale*b[i];
    return a;
}
Vec3 derivative(const Vec3 &q) {
    return {10.0*(q[1]-q[0]), q[0]*(28.0-q[2])-q[1],
            q[0]*q[1]-(8.0/3.0)*q[2]};
}
Vec3 rk4(Vec3 q) {
    constexpr double h=0.005;
    const Vec3 a=derivative(q);
    const Vec3 b=derivative(add(q,a,h/2.0));
    const Vec3 c=derivative(add(q,b,h/2.0));
    const Vec3 d=derivative(add(q,c,h));
    for(std::size_t i=0;i<3;++i) q[i]+=h*(a[i]+2*b[i]+2*c[i]+d[i])/6.0;
    return q;
}
struct Trace {
    int iteration=0, path_id=0;
    double rms=0, linf=0, field_energy=0, maxabs=0, mean=0;
    double path_cost=0, fold_mean_error=0, forcing=0, context=0;
};
struct Engine {
    Config cfg;
    std::vector<double> grid, previous, temp;
    std::vector<Trace> traces;
    Vec3 q{0.1,0.0,0.0};
    Trace last;
    double forcing=0.0, context=0.0, fold_mean_error=0.0;
    int steps_done=0;
    bool stationary_observed=false;
    explicit Engine(Config c):cfg(c),grid(static_cast<std::size_t>(c.side)*c.side*c.side),
          previous(grid.size()),temp(grid.size()) { traces.reserve(static_cast<std::size_t>(c.steps)); }
    std::size_t idx(int x,int y,int z) const {
        const int n=cfg.side;
        x=(x+n)%n; y=(y+n)%n; z=(z+n)%n;
        return (static_cast<std::size_t>(z)*n+y)*n+x;
    }
    double average(const std::vector<double> &v) const {
        double sum=0;
        for(double x:v) sum+=x;
        return sum/static_cast<double>(v.size());
    }
    void ingest() {
        if(cfg.input_stdin) {
            for(double &v:grid) {
                if(!(std::cin>>v) || !std::isfinite(v) || std::abs(v)>1.0)
                    throw std::invalid_argument("input voxel must be a finite number in [-1,1]");
            }
            std::string surplus;
            if(std::cin>>surplus) throw std::invalid_argument("extra input voxels");
        } else {
            std::uint64_t seed=static_cast<std::uint64_t>(cfg.seed);
            for(double &v:grid)
                v=2.0*static_cast<double>(splitmix64(seed)>>11)*(1.0/9007199254740992.0)-1.0;
        }
    }
    void lorenz(int step) {
        q=rk4(q);
        for(double v:q) if(!std::isfinite(v)) throw std::runtime_error("invalid Lorenz state");
        forcing=cfg.chaos*std::pow(cfg.decay,step)*std::tanh(q[0]/20.0);
        const int n=cfg.side;
        for(int z=0;z<n;++z) for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            // Nonuniform bounded spatial injection, not a broadcast constant.
            const double basis=std::sin(2.0*PI*(x+0.5)/n)*
                               std::cos(2.0*PI*(y+0.5)/n)*
                               std::sin(2.0*PI*(z+0.5)/n);
            const auto i=idx(x,y,z);
            grid[i]=std::tanh(grid[i]+forcing*basis);
        }
    }
    void attend_gru() {
        // Deterministic scalar attention + gated recurrence; NO learned weights.
        double numerator=0.0, denominator=0.0;
        for(double v:grid){const double weight=std::exp(v);
            numerator+=weight*v;denominator+=weight;}
        context=numerator/denominator;
        for(double &v:grid) {
            const double gate=sigmoid(v+context);
            v=(1.0-gate)*v+gate*std::tanh(0.75*v+0.25*context);
        }
    }
    void fold() {
        const int n=cfg.side;
        const double before=average(grid);
        for(int z=0;z<n;++z) for(int y=0;y<n;++y) for(int x=0;x<n;++x) {
            const auto i=idx(x,y,z);
            const double v=grid[i];
            const double neighbors=grid[idx(x-1,y,z)]+grid[idx(x+1,y,z)]+
                grid[idx(x,y-1,z)]+grid[idx(x,y+1,z)]+
                grid[idx(x,y,z-1)]+grid[idx(x,y,z+1)];
            temp[i]=(1.0-6.0*cfg.alpha-cfg.beta)*v+
                    cfg.alpha*neighbors+cfg.beta*before;
        }
        grid.swap(temp);
        fold_mean_error=std::abs(average(grid)-before);
        // The toroidal Laplacian and mean contraction must preserve the mean.
        if(fold_mean_error>1e-10) throw std::runtime_error("fold mean invariant failed");
    }
    void plan(Trace &t) const {
        const int n=cfg.side, center=n/2;
        constexpr int dirs[4][3]={{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
        t.path_cost=std::numeric_limits<double>::infinity();
        for(int d=0;d<4;++d) {
            double cost=0;
            for(int k=0;k<cfg.horizon;++k) {
                const int x=(center+dirs[d][0]*k)%n;
                const int y=(center+dirs[d][1]*k)%n;
                const int z=(center+dirs[d][2]*k)%n;
                const double v=grid[idx(x,y,z)];
                cost+=v*v+0.01*std::abs(v);
            }
            cost/=cfg.horizon;
            if(cost<t.path_cost) {t.path_cost=cost;t.path_id=d;}
        }
    }
    void verify(Trace &t) const {
        double sq=0, energy=0, sum=0, maxabs=0, linf=0;
        for(std::size_t i=0;i<grid.size();++i) {
            const double v=grid[i];
            if(!std::isfinite(v) || std::abs(v)>1.0+1e-10)
                throw std::runtime_error("bounded voxel invariant failed");
            const double delta=v-previous[i];
            sq+=delta*delta;linf=std::max(linf,std::abs(delta));
            energy+=v*v;sum+=v;maxabs=std::max(maxabs,std::abs(v));
        }
        const double n=static_cast<double>(grid.size());
        t.rms=std::sqrt(sq/n); t.linf=linf; t.field_energy=energy/n;
        t.mean=sum/n; t.maxabs=maxabs;t.fold_mean_error=fold_mean_error;
        t.forcing=forcing;t.context=context;
        for(double v:{t.rms,t.linf,t.field_energy,t.path_cost,t.mean,t.maxabs,t.context})
            if(!std::isfinite(v)) throw std::runtime_error("nonfinite verification metric");
    }
    void execute() {
        ingest(); // Initialization occurs BEFORE the first residual baseline.
        for(int step=0;step<cfg.steps;++step) {
            previous=grid;
            Trace t; t.iteration=step+1;
            for(Op op:PROGRAM) {
                switch(op) {
                case Op::INGEST: break; // Ingest is executed exactly once.
                case Op::LORENZ: lorenz(step);break;
                case Op::ATTEND_GRU: attend_gru();break;
                case Op::FOLD_3D: fold();break;
                case Op::PLAN: plan(t);break;
                case Op::VERIFY: verify(t);break;
                case Op::HALT: break;
                }
            }
            last=t;
            if(cfg.trace) traces.push_back(t);
            steps_done=step+1;
            // Empirical stationarity gate. Does NOT establish asymptotic convergence.
            if(cfg.epsilon>0 && t.rms<=cfg.epsilon &&
               std::abs(t.forcing)<=cfg.epsilon) {
                stationary_observed=true;break;
            }
        }
    }
    // Noncryptographic FNV-1a over quantized end-state values.
    std::uint64_t fingerprint() const {
        std::uint64_t h=14695981039346656037ULL;
        for(double v:grid) {
            const auto quantized=static_cast<std::int32_t>(std::round(v*10000.0));
            const auto bits=static_cast<std::uint32_t>(quantized);
            for(int k=0;k<4;++k) {h^=(bits>>(8*k))&255U;h*=1099511628211ULL;}
        }
        return h;
    }
};
void print_trace(const Trace &t) {
    std::cout<<"{\"iteration\":"<<t.iteration<<",\"residual_rms\":"<<t.rms
        <<",\"residual_linf\":"<<t.linf<<",\"field_energy\":"<<t.field_energy
        <<",\"max_abs\":"<<t.maxabs<<",\"mean\":"<<t.mean
        <<",\"path_id\":"<<t.path_id<<",\"path_cost\":"<<t.path_cost
        <<",\"fold_mean_error\":"<<t.fold_mean_error
        <<",\"forcing_amplitude\":"<<t.forcing<<",\"context\":"<<t.context<<"}";
}
}
int main(int argc,char**argv) {
    try {
        Config cfg;
        if((argc-1)%2) throw std::invalid_argument("expected --key value pairs");
        for(int i=1;i<argc;i+=2) {
            const std::string key=argv[i], value=argv[i+1];
            if(key=="--side")cfg.side=integer(value,4,32);
            else if(key=="--steps")cfg.steps=integer(value,1,100);
            else if(key=="--seed")cfg.seed=integer(value,0,2147483647);
            else if(key=="--horizon")cfg.horizon=integer(value,1,32);
            else if(key=="--alpha")cfg.alpha=real(value,0,1.0/6.0);
            else if(key=="--beta")cfg.beta=real(value,0,0.5);
            else if(key=="--chaos")cfg.chaos=real(value,0,0.2);
            else if(key=="--epsilon")cfg.epsilon=real(value,0,1.0);
            else if(key=="--decay")cfg.decay=real(value,0,1.0);
            else if(key=="--input-stdin")cfg.input_stdin=integer(value,0,1);
            else if(key=="--emit-field")cfg.emit_field=integer(value,0,1);
            else if(key=="--trace")cfg.trace=integer(value,0,1);
            else throw std::invalid_argument("unknown option: "+key);
        }
        if(6.0*cfg.alpha+cfg.beta>1.0+1e-12)
            throw std::invalid_argument("unstable spatial weights: require 6*alpha + beta <= 1");
        if(cfg.emit_field && cfg.side>16)
            throw std::invalid_argument("full field export is bounded to side <= 16");
        const auto start=std::chrono::steady_clock::now();
        Engine engine(cfg);engine.execute();
        const auto end=std::chrono::steady_clock::now();
        const auto ms=std::chrono::duration<double,std::milli>(end-start).count();
        std::cout<<std::fixed<<std::setprecision(12);
        std::cout<<"{\"engine\":\"jarvisx-phi3d-native-cpu-v2\",\"execution\":\"native\","
            <<"\"verified_finite\":true,\"spatial_invariants_verified\":true,"
            <<"\"stationary_observed\":"<<(engine.stationary_observed?"true":"false")
            <<",\"asymptotic_convergence_proven\":false"
            <<",\"side\":"<<cfg.side<<",\"voxels\":"<<engine.grid.size()
            <<",\"iterations\":"<<engine.steps_done
            <<",\"residual_rms\":"<<engine.last.rms
            <<",\"residual_linf\":"<<engine.last.linf
            <<",\"field_energy\":"<<engine.last.field_energy
            <<",\"max_abs\":"<<engine.last.maxabs
            <<",\"field_mean\":"<<engine.last.mean
            <<",\"fold_mean_error\":"<<engine.last.fold_mean_error
            <<",\"path_id\":"<<engine.last.path_id
            <<",\"path_cost\":"<<engine.last.path_cost
            <<",\"forcing_amplitude\":"<<engine.last.forcing
            <<",\"stability_margin\":"<<(1.0-6.0*cfg.alpha-cfg.beta)
            <<",\"lorenz\":["<<engine.q[0]<<","<<engine.q[1]<<","<<engine.q[2]<<"]"
            <<",\"input_mode\":\""<<(cfg.input_stdin?"explicit_voxels":"seeded")<<"\""
            <<",\"fingerprint64\":\""<<std::hex<<std::setw(16)<<std::setfill('0')
            <<engine.fingerprint()<<std::dec<<std::setfill(' ')
            <<"\",\"fingerprint_cryptographic\":false,\"runtime_ms\":"<<ms
            <<",\"program_hex\":\"10 20 30 40 50 60 ff\""
            <<",\"opcodes\":[\"INGEST\",\"LORENZ\",\"ATTEND_GRU\",\"FOLD_3D\",\"PLAN\",\"VERIFY\",\"HALT\"]";
        std::cout<<",\"slice_z\":"<<cfg.side/2<<",\"xy_slice\":[";
        for(int y=0;y<cfg.side;++y) for(int x=0;x<cfg.side;++x) {
            if(x || y) std::cout<<',';
            std::cout<<engine.grid[engine.idx(x,y,cfg.side/2)];
        }
        std::cout<<']';
        if(cfg.emit_field) {
            std::cout<<",\"field_zyx\":[";
            for(std::size_t i=0;i<engine.grid.size();++i) {
                if(i) std::cout<<',';
                std::cout<<engine.grid[i];
            }
            std::cout<<']';
        }
        if(cfg.trace) {
            std::cout<<",\"trace\":[";
            for(std::size_t i=0;i<engine.traces.size();++i) {
                if(i) std::cout<<',';
                print_trace(engine.traces[i]);
            }
            std::cout<<']';
        }
        std::cout<<"}\n";
        return 0;
    } catch(const std::exception &e) {
        std::cerr<<"VM ERROR: "<<e.what()<<'\n';return 2;
    }
}
