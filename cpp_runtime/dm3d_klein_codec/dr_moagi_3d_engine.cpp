// Dr Moagi 3D — operational C++17 reference codec and CTR verifier.
// No dependencies beyond the C++17 standard library. OpenMP is optional.
// Build: g++ -std=c++17 -O3 -Wall -Wextra -pedantic dr_moagi_3d_engine.cpp -o dm3d
// Parallel: g++ -std=c++17 -O3 -fopenmp dr_moagi_3d_engine.cpp -o dm3d
// Run: ./dm3d --grid 32 --steps 16 --eta 0.38 --beta 0.86 --mu 0.05
//      --inject-after-eight --csv trace.csv --json trace.json --obj klein.obj
// Test: ./dm3d --self-test
// Geometry: a 256-point Klein-inspired immersion and twisted-seam graph.
// Numerical codec: orthogonal projection over {1,sx,sy,sz} on 2x2x2 blocks.
// This is NOT a trained artificial neural network or a 1 GB RAM allocator.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace dm3d {
using Volume = std::vector<float>;
using Latent = std::vector<float>;
constexpr std::array<int, 8> SHELLS{{63,54,45,36,27,18,9,0}};
constexpr double PI = 3.1415926535897932384626433832795;
constexpr int C = 3, K = 4;

std::size_t mul_checked(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("grid size overflow");
    return a*b;
}
struct Shape {
    std::size_t n, m, voxels, blocks, nxyz;
    explicit Shape(int n_) : n(static_cast<std::size_t>(n_)), m(n/2),
        voxels(mul_checked(C, mul_checked(n,mul_checked(n,n)))),
        blocks(mul_checked(C, mul_checked(m,mul_checked(m,m)))),
        nxyz(mul_checked(n,mul_checked(n,n))) {
        if (n_<2 || n_%2) throw std::invalid_argument("grid must be even and >= 2");
        if (blocks > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::invalid_argument("grid too large for parallel indexing");
    }
    std::size_t pos(std::size_t c, std::size_t z, std::size_t y, std::size_t x) const {
        return ((c*n+z)*n+y)*n+x;
    }
    std::size_t latent_count() const { return mul_checked(blocks,K); }
};

Latent encode(const Shape& s, const Volume& x) {
    if (x.size()!=s.voxels) throw std::invalid_argument("encode: incorrect volume shape");
    Latent out(s.latent_count(), 0.0f);
    const auto count=static_cast<std::int64_t>(s.blocks);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count>4096)
#endif
    for (std::int64_t ib=0; ib<count; ++ib) {
        std::size_t t=static_cast<std::size_t>(ib);
        const auto bx=t%s.m; t/=s.m; const auto by=t%s.m; t/=s.m;
        const auto bz=t%s.m; t/=s.m; const auto c=t;
        // Integer signs are the four orthogonal Walsh basis vectors.
        double sum[4]{0.,0.,0.,0.};
        for (int iz=0;iz<2;++iz) for(int iy=0;iy<2;++iy) for(int ix=0;ix<2;++ix) {
            const float a=x[s.pos(c,2*bz+iz,2*by+iy,2*bx+ix)];
            sum[0]+=a; sum[1]+=(ix?1.:-1.)*a;
            sum[2]+=(iy?1.:-1.)*a; sum[3]+=(iz?1.:-1.)*a;
        }
        for(int k=0;k<4;++k) out[static_cast<std::size_t>(ib)*4+k]=static_cast<float>(sum[k]/8.);
    }
    return out;
}

void decode_into(const Shape& s, const Latent& z, Volume& out) {
    if(z.size()!=s.latent_count()) throw std::invalid_argument("decode: incorrect latent shape");
    if(out.size()!=s.voxels) throw std::invalid_argument("decode: incorrect output shape");
    const auto count=static_cast<std::int64_t>(s.blocks);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(count>4096)
#endif
    for(std::int64_t ib=0;ib<count;++ib) {
        std::size_t t=static_cast<std::size_t>(ib);
        const auto bx=t%s.m; t/=s.m; const auto by=t%s.m; t/=s.m;
        const auto bz=t%s.m; t/=s.m; const auto c=t;
        const float a=z[static_cast<std::size_t>(ib)*4+0],
                    dx=z[static_cast<std::size_t>(ib)*4+1],
                    dy=z[static_cast<std::size_t>(ib)*4+2],
                    dz=z[static_cast<std::size_t>(ib)*4+3];
        for(int iz=0;iz<2;++iz) for(int iy=0;iy<2;++iy) for(int ix=0;ix<2;++ix)
            out[s.pos(c,2*bz+iz,2*by+iy,2*bx+ix)] =
                a+(ix?dx:-dx)+(iy?dy:-dy)+(iz?dz:-dz);
    }
}

double mse(const Volume& a, const Volume& b) {
    if(a.size()!=b.size()) throw std::invalid_argument("MSE: unequal shapes");
    // Sequential double accumulation is deterministic across OpenMP thread counts.
    double acc=0.;
    for(std::size_t i=0;i<a.size();++i) {double r=double(a[i])-double(b[i]);acc+=r*r;}
    return acc/static_cast<double>(a.size());
}

double rms(const Latent& a) {
    double acc=0.; for(float f:a) acc+=double(f)*double(f);
    return std::sqrt(acc/a.size());
}

Volume synthetic(const Shape& s) {
    Volume a(s.voxels);
    const auto n=static_cast<std::int64_t>(s.nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(n>4096)
#endif
    for(std::int64_t i=0;i<n;++i) {
        std::size_t z=static_cast<std::size_t>(i)/ (s.n*s.n);
        std::size_t y=(static_cast<std::size_t>(i)/s.n)%s.n;
        std::size_t x=static_cast<std::size_t>(i)%s.n;
        double nx=(x+.5-double(s.n)/2)/(double(s.n)/2),
               ny=(y+.5-double(s.n)/2)/(double(s.n)/2),
               nz=(z+.5-double(s.n)/2)/(double(s.n)/2);
        double r=std::sqrt(nx*nx+ny*ny+nz*nz);
        double blob=std::exp(-7.*(std::pow(nx+.18,2)+std::pow(ny-.08,2)+std::pow(nz+.09,2)));
        double ring=std::exp(-13.*(std::pow(std::hypot(nx,ny)-.58,2)+.25*nz*nz));
        double vis=.07+.70*blob+.32*ring*(.55+.45*std::cos(6*std::atan2(ny,nx)));
        double aud=.04+.55*(.5+.5*std::sin(17*r+4*nz))*std::exp(-6*nz*nz)
                     +.16*std::exp(-6*(nx*nx+ny*ny));
        bool edge_x=std::abs(nx+.35)<.095 && std::abs(ny)<.62;
        bool edge_y=std::abs(ny-.15)<.095 && nx>-.36 && nx<.18;
        bool diag=std::abs(ny-1.4*(nx-.22))<.11 && nx>.15 && nx<.65;
        double txt=(edge_x||edge_y||diag?.92:.035)*std::exp(-17*nz*nz);
        auto i0=static_cast<std::size_t>(i);
        a[i0]=static_cast<float>(std::clamp(vis,0.,1.));
        a[s.nxyz+i0]=static_cast<float>(std::clamp(aud,0.,1.));
        a[2*s.nxyz+i0]=static_cast<float>(std::clamp(txt,0.,1.));
    }
    return a;
}

struct Row {
    unsigned step{}; int shell{}; double loss{}, excess{}, psnr{}, omega_rms{};
    bool accepted{}; double effective_eta{}; std::string event{"fold"};
};
struct Verification {
    bool passed=false; double orth_inf=0, decomposition_error=0, floor_error=0, latent_roundtrip_inf=0;
    double loss=0, floor=0, excess=0;
};
class Engine {
public:
    Shape s;
    double eta, beta, mu;
    Volume x, decoded, scratch;
    Latent target, z, omega, candidate;
    double floor{}, loss{};
    unsigned t=0, accepted=0, rejected=0;
    std::vector<Row> trace;

    Engine(int n, double eta_, double beta_, double mu_)
        : s(n), eta(eta_), beta(beta_), mu(mu_),
          x(synthetic(s)), decoded(s.voxels,0), scratch(s.voxels,0),
          target(encode(s,x)), z(s.latent_count(),0),
          omega(s.latent_count(),0),candidate(s.latent_count(),0) {
        if(!(eta>0 && eta<=1 && beta>=0 && beta<1 && mu>=0) ||
           !std::isfinite(eta) || !std::isfinite(beta) || !std::isfinite(mu))
            throw std::invalid_argument("need 0<eta<=1, 0<=beta<1, mu>=0, all finite");
        decode_into(s,target,scratch);
        floor=mse(x,scratch); loss=mse(x,decoded);
    }

    Row step() {
        bool ok=false; double accepted_gain=0, trial_loss=loss;
        for(int feedback_try=0;feedback_try<(mu>0?2:1) && !ok;++feedback_try) {
            const double memory_gain=(feedback_try==0?mu:0.); // fallback ignores active memory
            double gain=eta;
            for(int attempt=0;attempt<8 && !ok;++attempt, gain*=.5) {
                for(std::size_t j=0;j<z.size();++j) {
                    double correction=double(target[j])-double(z[j]);
                    if(memory_gain>0) correction+=memory_gain*(double(omega[j])-double(z[j]));
                    candidate[j]=static_cast<float>(double(z[j])+gain*correction);
                }
                decode_into(s,candidate,scratch);
                trial_loss=mse(x,scratch);
                if(std::isfinite(trial_loss) && trial_loss<=loss+1e-10) {
                    ok=true; accepted_gain=gain;
                }
            }
        }
        if(ok) {
            z.swap(candidate);
            decoded.swap(scratch);
            loss=trial_loss;
            for(std::size_t j=0;j<omega.size();++j)
                omega[j]=static_cast<float>(beta*double(omega[j])+(1.-beta)*double(z[j]));
            ++accepted;
        } else ++rejected;
        ++t;
        Row row{t, SHELLS[(t-1)%8],loss,loss-floor,
                loss>0?10*std::log10(1./loss):std::numeric_limits<double>::infinity(),
                rms(omega),ok,accepted_gain,"fold"};
        trace.push_back(row);
        return row;
    }

    void inject(int channel, std::uint64_t seed) {
        if(channel<0 || channel>=3) throw std::invalid_argument("inject channel 0..2");
        std::mt19937_64 rng(seed); std::uniform_real_distribution<double> u(.32,.68);
        std::array<int,3> center{{int(s.n*u(rng)),int(s.n*u(rng)),int(s.n*u(rng))}};
        for(std::size_t z0=0;z0<s.n;++z0) for(std::size_t y0=0;y0<s.n;++y0)
            for(std::size_t x0=0;x0<s.n;++x0) {
                double d=(std::pow(double(x0-center[0]),2)+std::pow(double(y0-center[1]),2)
                          +std::pow(double(z0-center[2]),2))/(2*std::pow(s.n*.10,2));
                double wave=channel==1?.55+.45*std::cos(std::sqrt(d)*9.):1.;
                std::size_t ix=s.pos(channel,z0,y0,x0);
                x[ix]=static_cast<float>(std::clamp(double(x[ix])+.38*std::exp(-d)*wave,0.,1.));
            }
        target=encode(s,x);
        decode_into(s,target,scratch);
        floor=mse(x,scratch);
        loss=mse(x,decoded); // new baseline; no cross-stimulus monotonicity claim
    }

    Verification verify() const {
        Volume optimum(s.voxels); decode_into(s,target,optimum);
        Volume residual(s.voxels);
        for(std::size_t i=0;i<x.size();++i) residual[i]=x[i]-optimum[i];
        Latent er=encode(s,residual);
        Latent roundtrip=encode(s,optimum);
        double orth=0., latent_roundtrip=0., sum=0.;
        for(std::size_t i=0;i<z.size();++i) {
            orth=std::max(orth,std::abs(double(er[i])));
            latent_roundtrip=std::max(latent_roundtrip,std::abs(double(roundtrip[i])-target[i]));
            double d=double(z[i])-double(target[i]); sum+=d*d;
        }
        const double lo=mse(x,optimum);
        const double predicted=lo+8.*sum/double(x.size());
        const double direct=mse(x,decoded);
        Verification q;
        q.orth_inf=orth; q.latent_roundtrip_inf=latent_roundtrip;
        q.decomposition_error=std::abs(direct-predicted);
        q.floor_error=std::abs(floor-lo);
        q.loss=direct; q.floor=lo; q.excess=direct-lo;
        q.passed=orth<5e-6 && latent_roundtrip<5e-6 && q.decomposition_error<2e-6
              && q.floor_error<1e-8 && direct+1e-8>=lo && std::isfinite(direct);
        return q;
    }
    std::size_t resident_bytes() const {
        return (x.size()+decoded.size()+scratch.size()+target.size()+z.size()+omega.size()+candidate.size())*sizeof(float);
    }
};

struct Vec3 {double x=0,y=0,z=0;};
Vec3 klein_node(int i, int j, double phase) {
    const double u=2*PI*i/16, v=2*PI*j/16;
    const int depth=SHELLS[j/2];
    const double A=2.2+std::cos(u/2)*std::sin(v)-std::sin(u/2)*std::sin(2*v);
    const double radius=.78*(.42+double(depth)/63*.85);
    const Vec3 original{radius*A*std::cos(u),radius*A*std::sin(u),
                        radius*(std::sin(u/2)*std::sin(v)+std::cos(u/2)*std::sin(2*v))};
    double f=double(depth)/63;
    double contract=1.-phase*.78*(.4+.8*f);
    double q=phase*.6*f;
    return {contract*original.x+.08*std::sin(q+.8*original.y),
            contract*original.y+.08*std::cos(q+original.x),contract*original.z};
}
// OBJ indexing is 1-based. The u-boundary reverses v, following Klein identification.
void export_obj(const std::string& path, double phase) {
    std::ofstream out(path);
    if(!out) throw std::runtime_error("cannot write OBJ: "+path);
    out << "# DM3D Klein-inspired 256-node geometry; phase="<<phase<<"\n";
    out<<"# Twisted u seam: (u=2pi,v) = (u=0,-v), topology is a graph approximation\n";
    out<<std::setprecision(10);
    for(int j=0;j<16;++j) for(int i=0;i<16;++i) {
        const auto p=klein_node(i,j,phase);
        out<<"v "<<p.x<<' '<<p.y<<' '<<p.z<<"\n";
    }
    auto id=[](int i,int j)->int {return ((j+16)%16)*16+i+1;};
    for(int j=0;j<16;++j) for(int i=0;i<16;++i) {
        int jnext=(j+1)%16, ni=(i+1)%16;
        int bj=(i==15 ? (16-j)%16 : j);
        int cj=(i==15 ? (16-jnext)%16 : jnext);
        int a=id(i,j), b=id(ni,bj), c=id(ni,cj), d=id(i,jnext);
        out<<"f "<<a<<' '<<b<<' '<<c<<"\n";
        out<<"f "<<a<<' '<<c<<' '<<d<<"\n";
    }
    if(!out) throw std::runtime_error("OBJ write failed");
}

std::string escape_json(const std::string& s) {
    std::string t; for(char c:s) {if(c=='"'||c=='\\')t+='\\';t+=c;}return t;
}
void export_csv(const std::string& path, const Engine& e) {
    std::ofstream f(path); if(!f) throw std::runtime_error("cannot write CSV");
    f<<"step,shell,mse,excess_mse,psnr_db,accepted,effective_eta,omega_rms,event\n";
    f<<std::setprecision(12);
    for(const auto& r:e.trace)
        f<<r.step<<','<<r.shell<<','<<r.loss<<','<<r.excess<<','<<r.psnr<<','
         <<(r.accepted?1:0)<<','<<r.effective_eta<<','<<r.omega_rms<<','<<r.event<<'\n';
}
void export_json(const std::string& path, const Engine& e, const Verification& q, double initial) {
    std::ofstream f(path); if(!f) throw std::runtime_error("cannot write JSON");
    f<<std::setprecision(12);
    f<<"{\n  \"implementation\": \"DM3D C++17 orthogonal 3D block codec\",\n"
      <<"  \"grid\": "<<e.s.n<<", \"channels\": 3, \"coefficients_per_block\": 4,\n"
      <<"  \"source_voxels\": "<<e.x.size()<<", \"latent_coefficients\": "<<e.z.size()<<",\n"
      <<"  \"coefficient_ratio\": "<<double(e.x.size())/e.z.size()<<",\n"
      <<"  \"resident_array_bytes\": "<<e.resident_bytes()<<",\n"
      <<"  \"eta\": "<<e.eta<<", \"beta\": "<<e.beta<<", \"mu\": "<<e.mu<<",\n"
      <<"  \"initial_mse\": "<<initial<<", \"final_mse\": "<<e.loss<<",\n"
      <<"  \"accepted\": "<<e.accepted<<", \"rejected\": "<<e.rejected<<",\n"
      <<"  \"verification\": {\"passed\": "<<(q.passed?"true":"false")
      <<", \"residual_orthogonality_inf\": "<<q.orth_inf
      <<", \"latent_roundtrip_inf\": "<<q.latent_roundtrip_inf
      <<", \"floor_mse\": "<<q.floor
      <<", \"loss_decomposition_error\": "<<q.decomposition_error<<"},\n"
      <<"  \"shell_depths\": [63,54,45,36,27,18,9,0], \"klein_nodes\": 256,\n"
      <<"  \"trace\": [\n";
    for(std::size_t i=0;i<e.trace.size();++i) {
        const auto& r=e.trace[i];
        f<<"    {\"step\": "<<r.step<<", \"shell\": "<<r.shell
         <<", \"mse\": "<<r.loss<<", \"excess_mse\": "<<r.excess
         <<", \"accepted\": "<<(r.accepted?"true":"false")
         <<", \"effective_eta\": "<<r.effective_eta
         <<", \"omega_rms\": "<<r.omega_rms<<", \"event\": \""<<escape_json(r.event)<<"\"}";
        if(i+1<e.trace.size())f<<',';
        f<<'\n';
    }
    f<<"  ]\n}\n";
}

struct Options {
    int grid=32, steps=8; double eta=.38, beta=.86, mu=0, phase=.65;
    bool inject=false, self_test=false;
    std::string csv, json, obj;
};
void help() {
    std::cout << "DM3D C++17 end-to-end 3D codec\n"
              << "  --grid N (even) --steps K --eta A --beta B --mu M\n"
              << "  --inject-after-eight --csv TRACE.csv --json TRACE.json\n"
              << "  --obj GEOMETRY.obj --phase P (0..1) --self-test --help\n";
}
Options parse(int argc, char** argv) {
    Options o;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto val=[&]()->std::string {if(i+1>=argc)throw std::invalid_argument("missing value after "+a);return argv[++i];};
        if(a=="--grid")o.grid=std::stoi(val());
        else if(a=="--steps")o.steps=std::stoi(val());
        else if(a=="--eta")o.eta=std::stod(val());
        else if(a=="--beta")o.beta=std::stod(val());
        else if(a=="--mu")o.mu=std::stod(val());
        else if(a=="--phase")o.phase=std::stod(val());
        else if(a=="--csv")o.csv=val();
        else if(a=="--json")o.json=val();
        else if(a=="--obj")o.obj=val();
        else if(a=="--inject-after-eight")o.inject=true;
        else if(a=="--self-test")o.self_test=true;
        else if(a=="--help" || a=="-h") {help();std::exit(0);}
        else throw std::invalid_argument("unknown option "+a);
    }
    if(o.steps<0 || o.steps>1000000)throw std::invalid_argument("steps must be 0..1,000,000");
    if(!(o.phase>=0 && o.phase<=1) || !std::isfinite(o.phase))
        throw std::invalid_argument("phase must be in [0,1]");
    return o;
}
void run_case(int n, bool injection, double mu) {
    Engine e(n,.38,.86,mu);
    auto q=e.verify(); if(!q.passed)throw std::runtime_error("initial verification failed");
    for(int i=0;i<(injection?16:8);++i) {
        double old=e.loss;
        Row r=e.step();
        if(!r.accepted || r.loss>old+1e-9)
            throw std::runtime_error("CTR violation or unexpected rejection");
        if(injection && i==7)e.inject(1,7);
        q=e.verify();
        if(!q.passed)throw std::runtime_error("verification failed after step/injection");
    }
    if(!injection && !(e.loss<e.trace.front().loss))
        throw std::runtime_error("static-input loss failed to decrease");
    std::cout<<"TEST PASS grid="<<n<<" mu="<<mu<<" inject="<<injection
             <<" final_MSE="<<std::setprecision(10)<<e.loss
             <<" orth="<<q.orth_inf<<" decomposition="<<q.decomposition_error<<'\n';
}
int main_fn(int argc,char** argv) {
    Options o=parse(argc,argv);
    if(o.self_test) {
        for(int n:{16,24,32})run_case(n,false,0);
        run_case(24,true,.05);
        return 0;
    }
    Engine e(o.grid,o.eta,o.beta,o.mu);
    double first=e.loss;
    std::cout<<std::fixed<<std::setprecision(9)
             <<"DM3D X=[3,"<<o.grid<<','<<o.grid<<','<<o.grid<<"] Z=[3,"
             <<o.grid/2<<','<<o.grid/2<<','<<o.grid/2<<",4]\n"
             <<"Initial MSE="<<first<<" irreducible_floor="<<e.floor
             <<" coefficient_ratio="<<double(e.x.size())/e.z.size()
             <<" resident_arrays="<<e.resident_bytes()<<" bytes\n";
    for(int i=0;i<o.steps;++i) {
        auto r=e.step();
        std::cout<<"t="<<std::setw(2)<<r.step<<" shell="<<std::setw(2)<<r.shell
                 <<" MSE="<<r.loss<<" excess="<<r.excess
                 <<" CTR="<<(r.accepted?"ACCEPT":"REJECT")
                 <<" gain="<<r.effective_eta<<" omega_RMS="<<r.omega_rms<<'\n';
        if(o.inject && (i+1)%8==0) {
            e.inject((i/8)%3,7+i);
            e.trace.back().event="fold+injection_after";
            std::cout<<"  INPUT INJECTED channel="<<(i/8)%3<<" new_target_MSE="<<e.loss
                     <<" new_floor="<<e.floor<<"\n";
        }
    }
    auto q=e.verify();
    std::cout<<"VERIFY "<<(q.passed?"PASS":"FAIL")
             <<" orth_inf="<<q.orth_inf<<" E(D(z*))_inf="<<q.latent_roundtrip_inf
             <<" decomposition_error="<<q.decomposition_error<<"\n";
    if(!o.csv.empty()) {export_csv(o.csv,e);std::cout<<"WROTE CSV "<<o.csv<<'\n';}
    if(!o.json.empty()) {export_json(o.json,e,q,first);std::cout<<"WROTE JSON "<<o.json<<'\n';}
    if(!o.obj.empty()) {export_obj(o.obj,o.phase);std::cout<<"WROTE 256-node OBJ "<<o.obj<<'\n';}
    return q.passed?0:2;
}
} // namespace dm3d
int main(int argc,char**argv) {
    try{return dm3d::main_fn(argc,argv);}
    catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}