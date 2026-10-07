// dm3d_codegen.cpp
// 3D evolutionary C++ code-generation system.
//
// Maps a toroidal 3D genotype g=(gx,gy,gz) into a concrete specialization
// of a 3D stencil kernel. Candidates are executed against a scalar reference,
// benchmarked, selected, folded toward the elite, mutated, and the best
// candidate is emitted as standalone C++ source.
//
// Build:
//   g++ -O3 -std=c++17 -pthread dm3d_codegen.cpp -o dm3d_codegen
//
// Run:
//   ./dm3d_codegen [generations population N output.cpp]
//
// Mathematical loop:
//   G = Z_GX x Z_GY x Z_GZ
//   theta = D(g)
//   y_theta = K_theta(x)
//   e(theta) = ||y_theta-y_ref||_inf
//   g' = Fold(g,g_elite) + mutation
//
// This is explicit search + verification + code emission.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static constexpr float ALPHA = 0.78f;
static constexpr float BETA  = 0.055f;
static constexpr float GAMMA = 0.015f;
static constexpr float DT    = 0.08f;
static constexpr float TOL   = 2.0e-6f;

struct Volume {
    int N{};
    std::vector<float> v;
    explicit Volume(int n=0) : N(n), v(static_cast<size_t>(n)*n*n, 0.0f) {}
    inline size_t idx(int x, int y, int z) const {
        return (static_cast<size_t>(z)*N + y)*N + x;
    }
};

static Volume make_input(int N) {
    Volume a(N);
    for (int z=0; z<N; ++z) {
        for (int y=0; y<N; ++y) {
            for (int x=0; x<N; ++x) {
                const float fx = static_cast<float>(x) / N;
                const float fy = static_cast<float>(y) / N;
                const float fz = static_cast<float>(z) / N;
                a.v[a.idx(x,y,z)] =
                    0.55f*std::sin(6.2831853f*fx) +
                    0.25f*std::cos(12.5663706f*fy) +
                    0.20f*std::sin(18.8495559f*fz + 3.1415926f*fx);
            }
        }
    }
    return a;
}

static inline float cell_update(const Volume& in, int x, int y, int z) {
    const int N = in.N;
    const int xm = (x==0   ? N-1 : x-1);
    const int xp = (x==N-1 ? 0   : x+1);
    const int ym = (y==0   ? N-1 : y-1);
    const int yp = (y==N-1 ? 0   : y+1);
    const int zm = (z==0   ? N-1 : z-1);
    const int zp = (z==N-1 ? 0   : z+1);
    const size_t c = in.idx(x,y,z);
    const float center = in.v[c];
    const float lap =
        in.v[in.idx(xm,y,z)] + in.v[in.idx(xp,y,z)] +
        in.v[in.idx(x,ym,z)] + in.v[in.idx(x,yp,z)] +
        in.v[in.idx(x,y,zm)] + in.v[in.idx(x,y,zp)] - 6.0f*center;
    const float residual = std::tanh(center) - center;
    return ALPHA*center + BETA*lap + GAMMA*residual + DT*center*residual;
}

static Volume reference(const Volume& in) {
    Volume out(in.N);
    for (int z=0; z<in.N; ++z)
        for (int y=0; y<in.N; ++y)
            for (int x=0; x<in.N; ++x)
                out.v[out.idx(x,y,z)] = cell_update(in,x,y,z);
    return out;
}

struct Gene3D {
    int x{}, y{}, z{};
};

struct KernelSpec {
    int tileX{8};
    int tileY{8};
    int tileZ{8};
    int unrollX{1};
    int order{0}; // 0=ZYX, 1=YZX, 2=ZXY, 3=XZY, 4=YXZ, 5=XYZ
    bool branchlessWrap{false};
};

static constexpr int GX = 8, GY = 8, GZ = 12;

static int toroidal_delta(int from, int to, int N) {
    int d = (to - from) % N;
    if (d < 0) d += N;
    if (d > N/2) d -= N;
    return d;
}

static int wrap_gene(int x, int N) {
    x %= N;
    return x < 0 ? x+N : x;
}

static KernelSpec decode_gene(const Gene3D& g) {
    static constexpr int tiles[8] = {2,4,6,8,12,16,24,32};
    static constexpr int unrolls[4] = {1,2,4,8};
    KernelSpec s;
    s.tileX = tiles[g.x % 8];
    s.tileY = tiles[(g.y + 2*g.z) % 8];
    s.tileZ = tiles[(g.z + g.x) % 8];
    s.unrollX = unrolls[g.y % 4];
    s.order = g.z % 6;
    s.branchlessWrap = ((g.x + g.y + g.z) & 1) != 0;
    return s;
}

static inline float update_branchless(const Volume& in, int x, int y, int z) {
    const int N = in.N;
    const int xm = x + (x==0   ? N-1 : -1);
    const int xp = x + (x==N-1 ? -(N-1) : 1);
    const int ym = y + (y==0   ? N-1 : -1);
    const int yp = y + (y==N-1 ? -(N-1) : 1);
    const int zm = z + (z==0   ? N-1 : -1);
    const int zp = z + (z==N-1 ? -(N-1) : 1);
    const float center = in.v[in.idx(x,y,z)];
    const float lap =
        in.v[in.idx(xm,y,z)] + in.v[in.idx(xp,y,z)] +
        in.v[in.idx(x,ym,z)] + in.v[in.idx(x,yp,z)] +
        in.v[in.idx(x,y,zm)] + in.v[in.idx(x,y,zp)] - 6.0f*center;
    const float residual = std::tanh(center) - center;
    return ALPHA*center + BETA*lap + GAMMA*residual + DT*center*residual;
}

template <class F>
static void tiled_xyz(int N, const KernelSpec& s, F&& f) {
    auto run_tile = [&](int bx, int by, int bz) {
        const int xe = std::min(N, bx+s.tileX);
        const int ye = std::min(N, by+s.tileY);
        const int ze = std::min(N, bz+s.tileZ);
        for (int z=bz; z<ze; ++z) {
            for (int y=by; y<ye; ++y) {
                int x=bx;
                for (; x+s.unrollX<=xe; x+=s.unrollX)
                    for (int u=0; u<s.unrollX; ++u) f(x+u,y,z);
                for (; x<xe; ++x) f(x,y,z);
            }
        }
    };

    const int tx=s.tileX, ty=s.tileY, tz=s.tileZ;
    switch (s.order) {
        case 0:
            for (int bz=0;bz<N;bz+=tz) for (int by=0;by<N;by+=ty) for (int bx=0;bx<N;bx+=tx) run_tile(bx,by,bz); break;
        case 1:
            for (int by=0;by<N;by+=ty) for (int bz=0;bz<N;bz+=tz) for (int bx=0;bx<N;bx+=tx) run_tile(bx,by,bz); break;
        case 2:
            for (int bz=0;bz<N;bz+=tz) for (int bx=0;bx<N;bx+=tx) for (int by=0;by<N;by+=ty) run_tile(bx,by,bz); break;
        case 3:
            for (int bx=0;bx<N;bx+=tx) for (int bz=0;bz<N;bz+=tz) for (int by=0;by<N;by+=ty) run_tile(bx,by,bz); break;
        case 4:
            for (int by=0;by<N;by+=ty) for (int bx=0;bx<N;bx+=tx) for (int bz=0;bz<N;bz+=tz) run_tile(bx,by,bz); break;
        default:
            for (int bx=0;bx<N;bx+=tx) for (int by=0;by<N;by+=ty) for (int bz=0;bz<N;bz+=tz) run_tile(bx,by,bz); break;
    }
}

static void run_candidate(const Volume& in, Volume& out, const KernelSpec& s) {
    tiled_xyz(in.N, s, [&](int x,int y,int z) {
        out.v[out.idx(x,y,z)] = s.branchlessWrap ?
            update_branchless(in,x,y,z) : cell_update(in,x,y,z);
    });
}

struct Eval {
    Gene3D gene{};
    KernelSpec spec{};
    double mvox_s{};
    double max_error{};
    double fitness{-1e300};
};

static double max_error(const Volume& a, const Volume& b) {
    double e=0.0;
    for (size_t i=0;i<a.v.size();++i)
        e = std::max(e, static_cast<double>(std::abs(a.v[i]-b.v[i])));
    return e;
}

static Eval evaluate(const Gene3D& g, const Volume& input, const Volume& ref, int repeats=3) {
    Eval e;
    e.gene=g;
    e.spec=decode_gene(g);
    Volume out(input.N);

    run_candidate(input,out,e.spec);
    e.max_error=max_error(out,ref);
    if (e.max_error > TOL || !std::isfinite(e.max_error)) {
        e.fitness = -1e200;
        return e;
    }

    double best=std::numeric_limits<double>::infinity();
    for (int r=0;r<repeats;++r) {
        auto t0=Clock::now();
        run_candidate(input,out,e.spec);
        auto t1=Clock::now();
        const double sec=std::chrono::duration<double>(t1-t0).count();
        best=std::min(best,sec);
    }

    const double vox=static_cast<double>(input.v.size());
    e.mvox_s = vox/best/1e6;
    const double complexity =
        1.0 + 0.00002*e.spec.tileX*e.spec.tileY*e.spec.tileZ +
        0.002*e.spec.unrollX;
    e.fitness = e.mvox_s / complexity;
    return e;
}

static Gene3D inward_fold_gene(const Gene3D& g, const Gene3D& elite,
                               std::mt19937& rng, double contract) {
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::uniform_int_distribution<int> step(-1,1);

    auto fold_axis = [&](int a, int b, int N) {
        const int d=toroidal_delta(a,b,N);
        int next=wrap_gene(a + static_cast<int>(std::lround(contract*d)), N);
        const double p_mut=0.34*(1.0-contract) + 0.08;
        if (U(rng) < p_mut) next=wrap_gene(next+step(rng),N);
        return next;
    };

    return {
        fold_axis(g.x,elite.x,GX),
        fold_axis(g.y,elite.y,GY),
        fold_axis(g.z,elite.z,GZ)
    };
}

static std::string spec_string(const KernelSpec& s) {
    std::ostringstream os;
    os << "tile="<<s.tileX<<"x"<<s.tileY<<"x"<<s.tileZ
       << " unrollX="<<s.unrollX << " order="<<s.order
       << " wrap="<<(s.branchlessWrap?"branchless":"conditional");
    return os.str();
}

static std::string emit_cpp(const Eval& best) {
    const KernelSpec& s=best.spec;
    std::ostringstream c;
    c << R"CPP(// AUTO-GENERATED by dm3d_codegen.cpp
// Best 3D genotype -> specialized C++ kernel.
// Build: g++ -O3 -std=c++17 best_kernel.cpp -o best_kernel

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

static constexpr float ALPHA=0.78f, BETA=0.055f, GAMMA=0.015f, DT=0.08f;
static constexpr int TILE_X=)CPP" << s.tileX << ";\n"
      << "static constexpr int TILE_Y=" << s.tileY << ";\n"
      << "static constexpr int TILE_Z=" << s.tileZ << ";\n"
      << "static constexpr int UNROLL_X=" << s.unrollX << ";\n"
      << "static constexpr int ORDER=" << s.order << ";\n"
      << "static constexpr bool BRANCHLESS_WRAP=" << (s.branchlessWrap?"true":"false") << ";\n\n";

    c << R"CPP(struct Volume {
    int N;
    std::vector<float> v;
    explicit Volume(int n):N(n),v(static_cast<size_t>(n)*n*n,0.0f){}
    inline size_t idx(int x,int y,int z) const { return (static_cast<size_t>(z)*N+y)*N+x; }
};

static inline float update(const Volume& in,int x,int y,int z) {
    const int N=in.N;
    const int xm = BRANCHLESS_WRAP ? x + (x==0 ? N-1 : -1) : (x==0?N-1:x-1);
    const int xp = BRANCHLESS_WRAP ? x + (x==N-1 ? -(N-1) : 1) : (x==N-1?0:x+1);
    const int ym = BRANCHLESS_WRAP ? y + (y==0 ? N-1 : -1) : (y==0?N-1:y-1);
    const int yp = BRANCHLESS_WRAP ? y + (y==N-1 ? -(N-1) : 1) : (y==N-1?0:y+1);
    const int zm = BRANCHLESS_WRAP ? z + (z==0 ? N-1 : -1) : (z==0?N-1:z-1);
    const int zp = BRANCHLESS_WRAP ? z + (z==N-1 ? -(N-1) : 1) : (z==N-1?0:z+1);
    const float center=in.v[in.idx(x,y,z)];
    const float lap=
        in.v[in.idx(xm,y,z)]+in.v[in.idx(xp,y,z)]+
        in.v[in.idx(x,ym,z)]+in.v[in.idx(x,yp,z)]+
        in.v[in.idx(x,y,zm)]+in.v[in.idx(x,y,zp)]-6.0f*center;
    const float residual=std::tanh(center)-center;
    return ALPHA*center+BETA*lap+GAMMA*residual+DT*center*residual;
}

template<class F>
static void run_tiles(int N,F&& f) {
    auto tile=[&](int bx,int by,int bz){
        const int xe=std::min(N,bx+TILE_X), ye=std::min(N,by+TILE_Y), ze=std::min(N,bz+TILE_Z);
        for(int z=bz;z<ze;++z) for(int y=by;y<ye;++y){
            int x=bx;
            for(;x+UNROLL_X<=xe;x+=UNROLL_X) for(int u=0;u<UNROLL_X;++u) f(x+u,y,z);
            for(;x<xe;++x) f(x,y,z);
        }
    };
    switch(ORDER){
        case 0: for(int bz=0;bz<N;bz+=TILE_Z) for(int by=0;by<N;by+=TILE_Y) for(int bx=0;bx<N;bx+=TILE_X) tile(bx,by,bz); break;
        case 1: for(int by=0;by<N;by+=TILE_Y) for(int bz=0;bz<N;bz+=TILE_Z) for(int bx=0;bx<N;bx+=TILE_X) tile(bx,by,bz); break;
        case 2: for(int bz=0;bz<N;bz+=TILE_Z) for(int bx=0;bx<N;bx+=TILE_X) for(int by=0;by<N;by+=TILE_Y) tile(bx,by,bz); break;
        case 3: for(int bx=0;bx<N;bx+=TILE_X) for(int bz=0;bz<N;bz+=TILE_Z) for(int by=0;by<N;by+=TILE_Y) tile(bx,by,bz); break;
        case 4: for(int by=0;by<N;by+=TILE_Y) for(int bx=0;bx<N;bx+=TILE_X) for(int bz=0;bz<N;bz+=TILE_Z) tile(bx,by,bz); break;
        default:for(int bx=0;bx<N;bx+=TILE_X) for(int by=0;by<N;by+=TILE_Y) for(int bz=0;bz<N;bz+=TILE_Z) tile(bx,by,bz); break;
    }
}

static void kernel(const Volume& in,Volume& out){
    run_tiles(in.N,[&](int x,int y,int z){out.v[out.idx(x,y,z)]=update(in,x,y,z);});
}

int main(int argc,char** argv){
    const int N=(argc>1)?std::max(8,std::atoi(argv[1])):64;
    Volume a(N),b(N);
    for(int z=0;z<N;++z) for(int y=0;y<N;++y) for(int x=0;x<N;++x){
        float fx=float(x)/N,fy=float(y)/N,fz=float(z)/N;
        a.v[a.idx(x,y,z)]=0.55f*std::sin(6.2831853f*fx)+0.25f*std::cos(12.5663706f*fy)+0.20f*std::sin(18.8495559f*fz+3.1415926f*fx);
    }
    kernel(a,b);
    double checksum=0.0;
    for(float q:b.v) checksum+=q;
    std::cout<<std::setprecision(12)<<"N="<<N<<" checksum="<<checksum<<"\n";
    std::cout<<"spec tile="<<TILE_X<<"x"<<TILE_Y<<"x"<<TILE_Z<<" unrollX="<<UNROLL_X<<" order="<<ORDER<<"\n";
    return 0;
}
)CPP";
    return c.str();
}

int main(int argc,char** argv) {
    const int generations = argc>1 ? std::clamp(std::atoi(argv[1]),1,50) : 8;
    const int population  = argc>2 ? std::clamp(std::atoi(argv[2]),4,128) : 12;
    const int N           = argc>3 ? std::clamp(std::atoi(argv[3]),16,256) : 48;
    const fs::path outPath= argc>4 ? fs::path(argv[4]) : fs::path("generated/best_kernel.cpp");

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dx(0,GX-1),dy(0,GY-1),dz(0,GZ-1);

    const Volume input=make_input(N);
    const Volume ref=reference(input);

    std::vector<Gene3D> pop(population);
    for(auto& g:pop) g={dx(rng),dy(rng),dz(rng)};

    Eval globalBest;
    globalBest.fitness=-1e300;

    std::cout << "DM3D C++ CODE GENERATOR\n"
              << "manifold="<<GX<<"x"<<GY<<"x"<<GZ
              << " generations="<<generations<<" population="<<population
              << " validation_volume="<<N<<"^3\n\n";

    for(int gen=0;gen<generations;++gen){
        std::vector<Eval> evals;
        evals.reserve(pop.size());
        for(const auto& g:pop) evals.push_back(evaluate(g,input,ref));
        std::sort(evals.begin(),evals.end(),[](const Eval&a,const Eval&b){return a.fitness>b.fitness;});
        if(evals.front().fitness>globalBest.fitness) globalBest=evals.front();

        const Eval& elite=evals.front();
        std::cout << "gen "<<std::setw(2)<<gen
                  << " elite g=("<<elite.gene.x<<","<<elite.gene.y<<","<<elite.gene.z<<") "
                  << std::fixed<<std::setprecision(2)<<elite.mvox_s<<" Mvox/s "
                  << "err="<<std::scientific<<elite.max_error<<std::fixed
                  << "  "<<spec_string(elite.spec)<<"\n";

        const double contract=std::min(0.82,0.22+0.60*double(gen+1)/generations);
        std::vector<Gene3D> next;
        next.reserve(population);
        next.push_back(elite.gene);
        if(population>1) next.push_back(evals[std::min<size_t>(1,evals.size()-1)].gene);
        while(static_cast<int>(next.size())<population){
            const Eval& parent=evals[static_cast<size_t>(next.size()) % std::min<size_t>(4,evals.size())];
            next.push_back(inward_fold_gene(parent.gene,elite.gene,rng,contract));
        }
        pop=std::move(next);
    }

    fs::create_directories(outPath.parent_path().empty()?fs::path("."):outPath.parent_path());
    std::ofstream out(outPath);
    out << emit_cpp(globalBest);
    out.close();

    std::cout << "\nBEST\n"
              << "g=("<<globalBest.gene.x<<","<<globalBest.gene.y<<","<<globalBest.gene.z<<")\n"
              << spec_string(globalBest.spec)<<"\n"
              << std::fixed<<std::setprecision(2)<<globalBest.mvox_s<<" Mvox/s, error="
              << std::scientific<<globalBest.max_error<<"\n"
              << "emitted: "<<outPath.string()<<"\n";

    fs::path binPath = outPath;
    binPath.replace_extension("");
    const fs::path runLog = outPath.parent_path() / "generated_kernel_run.log";

    std::ostringstream compileCmd;
    compileCmd << "g++ -O3 -std=c++17 \"" << outPath.string()
               << "\" -o \"" << binPath.string() << "\"";

    std::cout << "\n[CLOSE-LOOP] compile generated kernel\n"
              << compileCmd.str() << "\n";
    const int compileRc = std::system(compileCmd.str().c_str());
    if (compileRc != 0) {
        std::cerr << "[CLOSE-LOOP] compile FAILED rc=" << compileRc << "\n";
        return 2;
    }

    const int runN = std::max(64, N);
    std::ostringstream runCmd;
    runCmd << "\"" << binPath.string() << "\" " << runN
           << " | tee \"" << runLog.string() << "\"";

    std::cout << "[CLOSE-LOOP] execute emitted artifact at N=" << runN << "\n";
    const int runRc = std::system(runCmd.str().c_str());
    if (runRc != 0) {
        std::cerr << "[CLOSE-LOOP] generated kernel FAILED rc=" << runRc << "\n";
        return 3;
    }

    std::cout << "[CLOSE-LOOP] PASS\n"
              << "generated binary: " << binPath.string() << "\n"
              << "run log: " << runLog.string() << "\n"
              << "pipeline: genome -> decode -> execute -> verify -> benchmark -> select -> fold/mutate -> emit -> compile -> execute\n";
    return 0;
}
