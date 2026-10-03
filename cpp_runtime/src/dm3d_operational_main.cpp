// Jarvis-X / Dr Moagi DM3D operational engine.
// C++17 reference runtime: exact per-brick least squares, geometry-coupled
// inward folding, face-consistency coupling, OpenMP multiplexing, and CTR-style
// monotone commit/reject verification.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dm3d {

constexpr int K = 8;
constexpr double PI = 3.1415926535897932384626433832795;
using Latent = std::array<double,K>;
using Gram = std::array<std::array<double,K>,K>;

struct V3 { double x=0,y=0,z=0; };
static double norm(V3 p){ return std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z); }
static V3 rx(V3 p,double a){ double c=std::cos(a),s=std::sin(a); return {p.x,c*p.y-s*p.z,s*p.y+c*p.z}; }
static V3 ry(V3 p,double a){ double c=std::cos(a),s=std::sin(a); return {c*p.x+s*p.z,p.y,-s*p.x+c*p.z}; }
static V3 rz(V3 p,double a){ double c=std::cos(a),s=std::sin(a); return {c*p.x-s*p.y,s*p.x+c*p.y,p.z}; }
static V3 rotate(V3 p,double ax,double ay,double az){ return rz(ry(rx(p,ax),ay),az); }

struct Config {
    int side=1000, brick=25, iterations=4, threads=0;
    std::uint64_t max_bricks=256;
    bool full=false, toroidal=false, quiet=false;
    double contraction=0.96, coupling=0.02, curvature=0.04, tol=1e-12;
    std::string out;
};

struct Shape { int x0=0,y0=0,z0=0,nx=0,ny=0,nz=0; };
struct Transform { double lambda=1,ax=0,ay=0,az=0,curvature=0; };
struct Stats { Gram g{}; Latent b{}; long double c=0; std::uint64_t n=0; };

struct State {
    Shape shape{};
    Transform t{};
    Stats st{};
    Latent z{};
    V3 q0{},q{};
    double raw=0, fitted=0, mse=0;
    std::uint64_t geo_accept=0, fallbacks=0, coupling_accept=0;
    bool monotone=true, contractive=true;
};

struct Summary {
    std::uint64_t logical_voxels=0,logical_lanes=0,total_bricks=0,active=0,touched=0;
    std::uint64_t geo_accept=0,fallbacks=0,coupling_accept=0,failures=0;
    double initial_mse=0,fitted_mse=0,final_mse=0,gain=0;
    double r0=0,r1=0,face0=0,face1=0,seconds=0;
};

static std::uint64_t cube(std::uint64_t n){
    if(n && n>std::numeric_limits<std::uint64_t>::max()/n) throw std::overflow_error("side^2 overflow");
    std::uint64_t n2=n*n;
    if(n && n2>std::numeric_limits<std::uint64_t>::max()/n) throw std::overflow_error("side^3 overflow");
    return n2*n;
}

static std::uint64_t index3(int x,int y,int z,int n){
    return static_cast<std::uint64_t>(x)+static_cast<std::uint64_t>(n)*
        (static_cast<std::uint64_t>(y)+static_cast<std::uint64_t>(n)*static_cast<std::uint64_t>(z));
}

static void validate(const Config& c){
    if(c.side<=0||c.brick<=0||c.iterations<0) throw std::invalid_argument("invalid dimensions/iterations");
    if(!(c.contraction>0&&c.contraction<=1)) throw std::invalid_argument("contraction must be in (0,1]");
    if(c.coupling<0||c.coupling>0.25) throw std::invalid_argument("coupling must be in [0,0.25]");
    if(c.curvature<0||c.curvature>0.5) throw std::invalid_argument("curvature must be in [0,0.5]");
    if(c.tol<0) throw std::invalid_argument("tol must be non-negative");
    (void)cube(static_cast<std::uint64_t>(c.side));
}

static V3 global_p(int x,int y,int z,int n){
    double s=1.0/static_cast<double>(n);
    return {2*(static_cast<double>(x)+0.5)*s-1,
            2*(static_cast<double>(y)+0.5)*s-1,
            2*(static_cast<double>(z)+0.5)*s-1};
}
static double axis(int i,int n){ return n<=1?0.0:2.0*static_cast<double>(i)/static_cast<double>(n-1)-1.0; }

static double source(int x,int y,int z,int n){
    V3 p=global_p(x,y,z,n);
    double r2=p.x*p.x+p.y*p.y+p.z*p.z;
    double smooth=.38+.20*p.x-.14*p.y+.11*p.z+.08*p.x*p.y-.07*p.y*p.z+.05*p.z*p.x+.04*p.x*p.y*p.z;
    double wave=.055*std::sin(5*PI*p.x+3*PI*p.y)*std::cos(4*PI*p.z);
    return smooth+wave+.12*std::exp(-4.5*r2);
}

static V3 warp(V3 p,const Transform& t){
    V3 r=rotate(p,t.ax,t.ay,t.az);
    V3 h{r.y*r.z,r.z*r.x,r.x*r.y};
    return {t.lambda*(r.x+t.curvature*h.x),
            t.lambda*(r.y+t.curvature*h.y),
            t.lambda*(r.z+t.curvature*h.z)};
}
static Latent phi(V3 p,const Transform& t){
    V3 w=warp(p,t);
    return {1,w.x,w.y,w.z,w.x*w.y,w.y*w.z,w.z*w.x,w.x*w.y*w.z};
}
static double decode(const Latent& z,V3 p,const Transform& t){
    Latent f=phi(p,t); double s=0;
    for(int j=0;j<K;++j) s+=z[static_cast<std::size_t>(j)]*f[static_cast<std::size_t>(j)];
    return s;
}

static Shape shape(std::uint64_t id,const Config& c){
    int g=(c.side+c.brick-1)/c.brick;
    std::uint64_t gu=static_cast<std::uint64_t>(g), plane=gu*gu;
    int bz=static_cast<int>(id/plane);
    std::uint64_t rem=id%plane;
    int by=static_cast<int>(rem/gu), bx=static_cast<int>(rem%gu);
    return {bx*c.brick,by*c.brick,bz*c.brick,
            std::min(c.brick,c.side-bx*c.brick),
            std::min(c.brick,c.side-by*c.brick),
            std::min(c.brick,c.side-bz*c.brick)};
}

static Stats scan(const Shape& s,const Transform& t,const Config& c){
    Stats a;
    for(int kz=0;kz<s.nz;++kz){ double w=axis(kz,s.nz); int gz=s.z0+kz;
        for(int ky=0;ky<s.ny;++ky){ double v=axis(ky,s.ny); int gy=s.y0+ky;
            for(int kx=0;kx<s.nx;++kx){ double u=axis(kx,s.nx); int gx=s.x0+kx;
                Latent f=phi({u,v,w},t); double x=source(gx,gy,gz,c.side);
                a.c+=static_cast<long double>(x)*x; ++a.n;
                for(int i=0;i<K;++i){ std::size_t ii=static_cast<std::size_t>(i); a.b[ii]+=x*f[ii];
                    for(int j=0;j<K;++j){ std::size_t jj=static_cast<std::size_t>(j); a.g[ii][jj]+=f[ii]*f[jj]; }
                }
            }
        }
    }
    return a;
}

static Latent solve(const Stats& s){
    std::array<std::array<double,K+1>,K> a{};
    double md=0;
    for(int i=0;i<K;++i){ std::size_t ii=static_cast<std::size_t>(i); md=std::max(md,std::abs(s.g[ii][ii])); }
    double ridge=std::max(1e-14,md*1e-13);
    for(int i=0;i<K;++i){ std::size_t ii=static_cast<std::size_t>(i);
        for(int j=0;j<K;++j) a[ii][static_cast<std::size_t>(j)]=s.g[ii][static_cast<std::size_t>(j)];
        a[ii][ii]+=ridge; a[ii][K]=s.b[ii];
    }
    for(int col=0;col<K;++col){
        int piv=col; double best=std::abs(a[static_cast<std::size_t>(piv)][static_cast<std::size_t>(col)]);
        for(int r=col+1;r<K;++r){ double q=std::abs(a[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)]); if(q>best){best=q;piv=r;} }
        if(best<1e-20) continue;
        if(piv!=col) std::swap(a[static_cast<std::size_t>(piv)],a[static_cast<std::size_t>(col)]);
        double d=a[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
        for(int j=col;j<=K;++j) a[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)]/=d;
        for(int r=0;r<K;++r) if(r!=col){ double f=a[static_cast<std::size_t>(r)][static_cast<std::size_t>(col)];
            for(int j=col;j<=K;++j) a[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)]-=f*a[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
        }
    }
    Latent z{}; for(int i=0;i<K;++i) z[static_cast<std::size_t>(i)]=a[static_cast<std::size_t>(i)][K]; return z;
}

static double mse(const Stats& s,const Latent& z){
    if(!s.n) return 0;
    long double l=s.c;
    for(int i=0;i<K;++i){ std::size_t ii=static_cast<std::size_t>(i); l-=2.0L*static_cast<long double>(z[ii])*s.b[ii];
        for(int j=0;j<K;++j){ std::size_t jj=static_cast<std::size_t>(j); l+=static_cast<long double>(z[ii])*s.g[ii][jj]*z[jj]; }
    }
    long double v=l/static_cast<long double>(s.n);
    if(v<0&&std::abs(static_cast<double>(v))<1e-13) {
        return 0;
    }
    return static_cast<double>(v);
}

static Transform explore(const Transform& cur,const Config& c,std::uint64_t id,int it){
    Transform n=cur; n.lambda=std::max(.10,cur.lambda*c.contraction);
    double p=.010*static_cast<double>(it+1)+1e-5*static_cast<double>(id%10007ULL);
    n.ax+=p; n.ay+=.7*p; n.az+=1.3*p;
    n.curvature=std::clamp(c.curvature*(1+.15*std::sin(7*p)),0.0,.5);
    return n;
}

static void fallback(State& s,double q){
    if(q==1) return;
    double q2=q*q,q3=q2*q; Latent d{1,q,q,q,q2,q2,q2,q3};
    for(int i=0;i<K;++i){ std::size_t ii=static_cast<std::size_t>(i); s.st.b[ii]*=d[ii]; s.z[ii]/=d[ii];
        for(int j=0;j<K;++j){ std::size_t jj=static_cast<std::size_t>(j); s.st.g[ii][jj]*=d[ii]*d[jj]; }
    }
    s.t.lambda*=q; s.q={q*s.q.x,q*s.q.y,q*s.q.z}; ++s.fallbacks;
}

static State process(std::uint64_t id,const Config& c){
    State s; s.shape=shape(id,c);
    int cx=s.shape.x0+s.shape.nx/2, cy=s.shape.y0+s.shape.ny/2, cz=s.shape.z0+s.shape.nz/2;
    s.q0=global_p(cx,cy,cz,c.side); s.q=s.q0;
    s.st=scan(s.shape,s.t,c); Latent zero{}; s.raw=mse(s.st,zero);
    s.z=solve(s.st); s.fitted=mse(s.st,s.z); s.mse=s.fitted; double prev=s.mse;
    for(int it=0;it<c.iterations;++it){
        Transform nt=explore(s.t,c,id,it); Stats ns=scan(s.shape,nt,c); Latent nz=solve(ns); double nm=mse(ns,nz);
        double before=norm(s.q);
        if(nm<=s.mse+c.tol){
            double scale=nt.lambda/s.t.lambda;
            V3 qq=rotate(s.q,nt.ax-s.t.ax,nt.ay-s.t.ay,nt.az-s.t.az);
            s.q={scale*qq.x,scale*qq.y,scale*qq.z};
            s.t=nt; s.st=ns; s.z=nz; s.mse=nm; ++s.geo_accept;
        } else fallback(s,c.contraction);
        if(norm(s.q)>before+1e-11) s.contractive=false;
        if(s.mse>prev+c.tol) {
            s.monotone=false;
        }
        prev=s.mse;
    }
    return s;
}

static std::uint64_t gid(int x,int y,int z,int g){
    return static_cast<std::uint64_t>(x)+static_cast<std::uint64_t>(g)*
        (static_cast<std::uint64_t>(y)+static_cast<std::uint64_t>(g)*static_cast<std::uint64_t>(z));
}
static bool neighbor(int x,int y,int z,int dx,int dy,int dz,int g,bool tor,int& nx,int& ny,int& nz){
    nx=x+dx; ny=y+dy; nz=z+dz;
    if(tor){ auto w=[g](int v){ int r=v%g; return r<0?r+g:r; }; nx=w(nx);ny=w(ny);nz=w(nz);return true; }
    return nx>=0&&nx<g&&ny>=0&&ny<g&&nz>=0&&nz<g;
}
static V3 face(int dx,int dy,int dz){ return {static_cast<double>(dx),static_cast<double>(dy),static_cast<double>(dz)}; }

static double face_jump(const std::vector<State>& s,std::uint64_t active,const Config& c){
    int g=(c.side+c.brick-1)/c.brick; int d[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    long double sum=0; std::uint64_t n=0,gu=static_cast<std::uint64_t>(g),plane=gu*gu;
    for(std::uint64_t id=0;id<active;++id){ int z=static_cast<int>(id/plane); std::uint64_t rem=id%plane; int y=static_cast<int>(rem/gu),x=static_cast<int>(rem%gu);
        for(auto& v:d){ int nx=0,ny=0,nz=0; if(!neighbor(x,y,z,v[0],v[1],v[2],g,c.toroidal,nx,ny,nz)) continue;
            std::uint64_t j=gid(nx,ny,nz,g); if(j>=active) continue;
            double a=decode(s[static_cast<std::size_t>(id)].z,face(v[0],v[1],v[2]),s[static_cast<std::size_t>(id)].t);
            double b=decode(s[static_cast<std::size_t>(j)].z,face(-v[0],-v[1],-v[2]),s[static_cast<std::size_t>(j)].t);
            sum+=std::abs(a-b); ++n;
        }
    }
    return n?static_cast<double>(sum/static_cast<long double>(n)):0;
}

static void couple(std::vector<State>& s,std::uint64_t active,const Config& c){
    if(c.coupling<=0||!active) return;
    int g=(c.side+c.brick-1)/c.brick; std::uint64_t gu=static_cast<std::uint64_t>(g),plane=gu*gu;
    std::vector<double> candidate(static_cast<std::size_t>(active));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(std::int64_t raw=0;raw<static_cast<std::int64_t>(active);++raw){
        std::uint64_t id=static_cast<std::uint64_t>(raw); int z=static_cast<int>(id/plane); std::uint64_t rem=id%plane; int y=static_cast<int>(rem/gu),x=static_cast<int>(rem%gu);
        int d[6][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}}; double corr=0; int count=0;
        const State& self=s[static_cast<std::size_t>(id)];
        for(auto& v:d){ int nx=0,ny=0,nz=0; if(!neighbor(x,y,z,v[0],v[1],v[2],g,c.toroidal,nx,ny,nz)) continue;
            std::uint64_t j=gid(nx,ny,nz,g); if(j>=active) continue; const State& nb=s[static_cast<std::size_t>(j)];
            corr+=decode(nb.z,face(-v[0],-v[1],-v[2]),nb.t)-decode(self.z,face(v[0],v[1],v[2]),self.t); ++count;
        }
        candidate[static_cast<std::size_t>(id)]=self.z[0]+(count?c.coupling*corr/static_cast<double>(count):0);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(std::int64_t raw=0;raw<static_cast<std::int64_t>(active);++raw){
        State& q=s[static_cast<std::size_t>(raw)]; Latent z=q.z; z[0]=candidate[static_cast<std::size_t>(raw)]; double m=mse(q.st,z);
        if(m<=q.mse+c.tol){ q.z=z; q.mse=m; ++q.coupling_accept; }
    }
}

static Summary run(const Config& c){
    validate(c);
#ifdef _OPENMP
    if(c.threads>0) omp_set_num_threads(c.threads);
#endif
    int g=(c.side+c.brick-1)/c.brick; std::uint64_t gu=static_cast<std::uint64_t>(g),total=gu*gu*gu;
    std::uint64_t active=c.full?total:std::min(c.max_bricks,total);
    auto t0=std::chrono::steady_clock::now(); std::vector<State> states(static_cast<std::size_t>(active));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(std::int64_t raw=0;raw<static_cast<std::int64_t>(active);++raw) states[static_cast<std::size_t>(raw)]=process(static_cast<std::uint64_t>(raw),c);
    double f0=face_jump(states,active,c); couple(states,active,c); double f1=face_jump(states,active,c);
    long double a=0,b=0,d=0,r0=0,r1=0; std::uint64_t touched=0,ga=0,fb=0,ca=0,fail=0;
    for(const State& s:states){
        std::uint64_t n=static_cast<std::uint64_t>(s.shape.nx)*static_cast<std::uint64_t>(s.shape.ny)*static_cast<std::uint64_t>(s.shape.nz);
        touched+=n;
        const long double weight=static_cast<long double>(n);
        a+=static_cast<long double>(s.raw)*weight;
        b+=static_cast<long double>(s.fitted)*weight;
        d+=static_cast<long double>(s.mse)*weight;
        r0+=norm(s.q0);
        r1+=norm(s.q);
        ga+=s.geo_accept; fb+=s.fallbacks; ca+=s.coupling_accept;
        if(!s.monotone||!s.contractive||!std::isfinite(s.mse)||s.mse>s.raw+c.tol||norm(s.q)>norm(s.q0)+1e-10) ++fail;
    }
    auto t1=std::chrono::steady_clock::now(); Summary o; o.logical_voxels=cube(static_cast<std::uint64_t>(c.side));
    o.logical_lanes=static_cast<std::uint64_t>(c.side)*static_cast<std::uint64_t>(c.side); o.total_bricks=total;o.active=active;o.touched=touched;
    o.geo_accept=ga;o.fallbacks=fb;o.coupling_accept=ca;o.failures=fail;
    if(touched){ long double n=static_cast<long double>(touched);o.initial_mse=static_cast<double>(a/n);o.fitted_mse=static_cast<double>(b/n);o.final_mse=static_cast<double>(d/n); }
    o.gain=o.initial_mse>0?(o.initial_mse-o.final_mse)/o.initial_mse:0;
    if(active){ long double n=static_cast<long double>(active);o.r0=static_cast<double>(r0/n);o.r1=static_cast<double>(r1/n); }
    o.face0=f0;o.face1=f1;o.seconds=std::chrono::duration<double>(t1-t0).count(); return o;
}

static bool self_test(){
    Config c;c.side=24;c.brick=8;c.iterations=3;c.full=true;c.contraction=.95;c.coupling=.02;c.curvature=.04;c.tol=1e-11;
    Summary s=run(c);
    return cube(24)==13824ULL&&index3(23,23,23,24)==13823ULL&&s.total_bricks==27&&s.active==27&&s.touched==13824&&
        s.fitted_mse<=s.initial_mse+1e-10&&s.final_mse<=s.initial_mse+1e-10&&s.r1<=s.r0+1e-10&&s.failures==0;
}

static void write_json(const Config& c,const Summary& s){
    if(c.out.empty()) {
        return;
    }
    std::ofstream f(c.out);
    if(!f) {
        throw std::runtime_error("cannot write "+c.out);
    }
    f<<std::setprecision(17);
    f<<"{\n  \"engine\":\"Jarvis-X DM3D Operational\",\n  \"side\":"<<c.side<<",\n  \"brick\":"<<c.brick<<",\n  \"iterations\":"<<c.iterations
     <<",\n  \"logical_voxels\":"<<s.logical_voxels<<",\n  \"active_bricks\":"<<s.active<<",\n  \"touched_voxels\":"<<s.touched
     <<",\n  \"initial_mse\":"<<s.initial_mse<<",\n  \"fitted_mse\":"<<s.fitted_mse<<",\n  \"final_mse\":"<<s.final_mse
     <<",\n  \"geometry_accepts\":"<<s.geo_accept<<",\n  \"contraction_fallbacks\":"<<s.fallbacks<<",\n  \"coupling_accepts\":"<<s.coupling_accept
     <<",\n  \"face_jump_before\":"<<s.face0<<",\n  \"face_jump_after\":"<<s.face1<<",\n  \"invariant_failures\":"<<s.failures
     <<",\n  \"elapsed_seconds\":"<<s.seconds<<"\n}\n";
}

static Config parse(int argc,char** argv,bool& test,bool& health){
    Config c; auto need=[&](int& i){ if(i+1>=argc) throw std::invalid_argument(std::string("missing value for ")+argv[i]); return std::string(argv[++i]); };
    for(int i=1;i<argc;++i){ std::string a=argv[i];
        if(a=="--side")c.side=std::stoi(need(i)); else if(a=="--brick")c.brick=std::stoi(need(i)); else if(a=="--iterations")c.iterations=std::stoi(need(i));
        else if(a=="--bricks")c.max_bricks=std::stoull(need(i)); else if(a=="--threads")c.threads=std::stoi(need(i)); else if(a=="--contraction")c.contraction=std::stod(need(i));
        else if(a=="--coupling")c.coupling=std::stod(need(i)); else if(a=="--curvature")c.curvature=std::stod(need(i)); else if(a=="--out")c.out=need(i);
        else if(a=="--full")c.full=true; else if(a=="--toroidal")c.toroidal=true; else if(a=="--quiet")c.quiet=true; else if(a=="--self-test")test=true; else if(a=="--health")health=true;
        else if(a=="--help"||a=="-h"){ std::cout<<"Usage: "<<argv[0]<<" [--side N] [--brick N] [--iterations N] [--bricks N|--full] [--threads N] [--contraction X] [--coupling X] [--curvature X] [--toroidal] [--out FILE] [--self-test] [--health] [--quiet]\n"; std::exit(0); }
        else throw std::invalid_argument("unknown option: "+a);
    } return c;
}

} // namespace dm3d

int main(int argc,char** argv){
    try{
        bool test=false,health=false; dm3d::Config c=dm3d::parse(argc,argv,test,health);
        if(health){std::cout<<"DM3D_HEALTH_OK\n";return 0;}
        if(test){bool ok=dm3d::self_test();std::cout<<(ok?"DM3D_SELF_TEST_PASS\n":"DM3D_SELF_TEST_FAIL\n");return ok?0:1;}
        dm3d::Summary s=dm3d::run(c); dm3d::write_json(c,s);
        if(!c.quiet){std::cout<<std::fixed<<std::setprecision(10)
            <<"=== Jarvis-X DM3D Operational ===\n"
            <<"logical lanes          : "<<s.logical_lanes<<"\nlogical voxels         : "<<s.logical_voxels
            <<"\ntotal bricks           : "<<s.total_bricks<<"\nactive bricks          : "<<s.active
            <<"\ntouched voxels         : "<<s.touched<<"\ninitial MSE            : "<<s.initial_mse
            <<"\nexact fitted MSE       : "<<s.fitted_mse<<"\nfinal MSE              : "<<s.final_mse
            <<"\nrelative MSE gain      : "<<s.gain<<"\nmean radius initial    : "<<s.r0
            <<"\nmean radius final      : "<<s.r1<<"\ngeometry accepts       : "<<s.geo_accept
            <<"\ncontraction fallbacks  : "<<s.fallbacks<<"\ncoupling accepts       : "<<s.coupling_accept
            <<"\nface jump before       : "<<s.face0<<"\nface jump after        : "<<s.face1
            <<"\ninvariant failures     : "<<s.failures<<"\nelapsed seconds        : "<<s.seconds<<"\n";}
        return s.failures?2:0;
    }catch(const std::exception& e){std::cerr<<"fatal: "<<e.what()<<"\n";return 64;}
}
