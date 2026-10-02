// moagi_systemwide_3d_map.cpp
//
// Self-contained C++17 reference implementation of the Jarvis-X / Moagi
// system-wide 3D architecture.
//
// Build:
//   g++ -std=c++17 -O2 -Wall -Wextra -pedantic moagi_systemwide_3d_map.cpp -o moagi3d
//
// Run:
//   ./moagi3d --tile-side 16 --iterations 10 --out moagi_system
//
// Outputs:
//   PREFIX.obj   - 3D wireframe system map + active sparse tile points
//   PREFIX.dot   - graph topology
//   PREFIX.json  - execution telemetry / CTR receipts
//
// Scale contract:
// Each virtual axis spans 1000 decimal MB = 1,000,000,000 logical byte
// positions, so the logical address universe contains 10^27 coordinates.
// This program never allocates 10^27 cells. It materializes only bounded tiles.
//
// The encoder/decoder are deterministic block codecs so this file has no
// dependencies beyond the C++ standard library. Their interfaces are intended
// to be replaced by libtorch/CUDA Conv3D kernels in a trained implementation.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moagi {

constexpr double PI = 3.141592653589793238462643383279502884;

// ---------------------------------------------------------------------------
// 0. Basic 3D geometry
// ---------------------------------------------------------------------------

struct Vec3 {
    double x{}, y{}, z{};
    Vec3 operator+(const Vec3& o) const { return {x+o.x,y+o.y,z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x,y-o.y,z-o.z}; }
    Vec3 operator*(double s) const { return {x*s,y*s,z*s}; }
    Vec3 operator/(double s) const { return {x/s,y/s,z/s}; }
};

inline double dot(const Vec3& a,const Vec3& b){
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
inline double norm(const Vec3& v){ return std::sqrt(dot(v,v)); }
inline Vec3 normalize(Vec3 v){
    const double n=norm(v);
    return n<1e-12?Vec3{0,0,0}:v/n;
}

struct Mat3 {
    double m[3][3]{};
    Vec3 apply(const Vec3& v) const {
        return {
            m[0][0]*v.x+m[0][1]*v.y+m[0][2]*v.z,
            m[1][0]*v.x+m[1][1]*v.y+m[1][2]*v.z,
            m[2][0]*v.x+m[2][1]*v.y+m[2][2]*v.z
        };
    }
    Mat3 transpose() const {
        Mat3 r;
        for(int i=0;i<3;++i) for(int j=0;j<3;++j) r.m[i][j]=m[j][i];
        return r;
    }
};

Mat3 mmul(const Mat3& A,const Mat3& B){
    Mat3 R;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k)
        R.m[i][j]+=A.m[i][k]*B.m[k][j];
    return R;
}

Mat3 rotation_matrix(double ax,double ay,double az){
    const double cx=std::cos(ax),sx=std::sin(ax);
    const double cy=std::cos(ay),sy=std::sin(ay);
    const double cz=std::cos(az),sz=std::sin(az);
    Mat3 Rx{},Ry{},Rz{};
    Rx.m[0][0]=1; Rx.m[1][1]=cx; Rx.m[1][2]=-sx; Rx.m[2][1]=sx; Rx.m[2][2]=cx;
    Ry.m[0][0]=cy; Ry.m[0][2]=sy; Ry.m[1][1]=1; Ry.m[2][0]=-sy; Ry.m[2][2]=cy;
    Rz.m[0][0]=cz; Rz.m[0][1]=-sz; Rz.m[1][0]=sz; Rz.m[1][1]=cz; Rz.m[2][2]=1;
    return mmul(Rz,mmul(Ry,Rx));
}

// ---------------------------------------------------------------------------
// 1. Configuration
// ---------------------------------------------------------------------------

struct Config {
    std::uint64_t logical_axis_bytes=1'000'000'000ULL; // 1000 decimal MB
    int tile_side=16;
    int block=4;
    int max_tiles_per_sample=8;
    int iterations=10;

    double beta_memory=0.88;
    double residual_gain=0.70;
    double context_gain=0.22;

    double lambda_contract=0.985;
    double dt=0.08;
    double a_torus=0.55;
    double a_lorenz=0.25;
    double a_hyper=0.20;
    double possibility_gain=0.05;
    double possibility_temp=0.18;

    std::uint32_t seed=7;
    std::string out_prefix="moagi_systemwide_3d";

    int latent_side() const {
        if(tile_side%block!=0) throw std::runtime_error("tile_side must be divisible by block");
        return tile_side/block;
    }
    std::size_t tile_bytes() const {
        return static_cast<std::size_t>(tile_side)*tile_side*tile_side;
    }
};

// ---------------------------------------------------------------------------
// 2. Multimodal packets
// ---------------------------------------------------------------------------

enum class Modality:int { Text=0,Image=1,Audio=2,Video=3,Bytes=4 };

std::string modality_name(Modality m){
    switch(m){
        case Modality::Text:return "text";
        case Modality::Image:return "image";
        case Modality::Audio:return "audio";
        case Modality::Video:return "video";
        default:return "bytes";
    }
}

struct MediaSample {
    Modality modality{};
    std::string name;
    std::vector<std::uint8_t> bytes;
};

std::vector<MediaSample> make_samples(const Config& cfg){
    std::vector<MediaSample> out;

    // Text
    {
        const std::string seed=
            "Jarvis X / Dr Moagi 3D runtime. Sparse spatial materialization, "
            "multimodal encoding, inward latent refinement, CTR verification. ";
        std::string text;
        for(int i=0;i<18;++i) text+=seed;
        out.push_back({Modality::Text,"synthetic-text",
                       std::vector<std::uint8_t>(text.begin(),text.end())});
    }

    // Image: 32x32 RGB
    {
        constexpr int H=32,W=32,C=3;
        std::vector<std::uint8_t> b(H*W*C);
        for(int y=0;y<H;++y) for(int x=0;x<W;++x){
            const auto i=static_cast<std::size_t>((y*W+x)*C);
            b[i]=static_cast<std::uint8_t>(255.0*x/(W-1));
            b[i+1]=static_cast<std::uint8_t>(255.0*y/(H-1));
            b[i+2]=static_cast<std::uint8_t>((x*y)%256);
        }
        out.push_back({Modality::Image,"synthetic-image",std::move(b)});
    }

    // Audio: PCM16
    {
        constexpr int N=2048;
        constexpr double sr=8000.0;
        std::vector<std::uint8_t> b;
        b.reserve(N*2);
        for(int n=0;n<N;++n){
            const double t=n/sr;
            const double v=0.55*std::sin(2*PI*220*t)+0.25*std::sin(2*PI*440*t);
            const auto q=static_cast<std::int16_t>(std::clamp(v,-1.0,1.0)*32767.0);
            b.push_back(static_cast<std::uint8_t>(q&0xff));
            b.push_back(static_cast<std::uint8_t>((q>>8)&0xff));
        }
        out.push_back({Modality::Audio,"synthetic-audio",std::move(b)});
    }

    // Video: 4 x 16 x 16 x 3 raw frames
    {
        constexpr int T=4,H=16,W=16,C=3;
        std::vector<std::uint8_t> b(T*H*W*C);
        for(int t=0;t<T;++t) for(int y=0;y<H;++y) for(int x=0;x<W;++x){
            const int cx=3+2*t,cy=4+t;
            const double dx=x-cx,dy=y-cy;
            const double blob=std::exp(-(dx*dx+dy*dy)/12.0);
            const auto i=static_cast<std::size_t>(((t*H+y)*W+x)*C);
            b[i]=static_cast<std::uint8_t>(std::clamp(255.0*blob,0.0,255.0));
            b[i+1]=static_cast<std::uint8_t>((60+25*t)%256);
            b[i+2]=static_cast<std::uint8_t>((20*t)%256);
        }
        out.push_back({Modality::Video,"synthetic-video",std::move(b)});
    }

    // Opaque bytes
    {
        std::mt19937 rng(cfg.seed+99);
        std::uniform_int_distribution<int> d(0,255);
        std::vector<std::uint8_t> b(4096);
        for(auto& x:b) x=static_cast<std::uint8_t>(d(rng));
        out.push_back({Modality::Bytes,"synthetic-bytes",std::move(b)});
    }

    return out;
}

// ---------------------------------------------------------------------------
// 3. Sparse virtual 1000MB x 1000MB x 1000MB fabric
// ---------------------------------------------------------------------------

struct TileAddress {
    std::uint64_t x{},y{},z{};
    int modality{};
    std::uint64_t sequence{};
};

struct Tile {
    TileAddress address;
    std::vector<std::uint8_t> bytes;
    std::size_t valid_bytes{};
    std::string sample_name;
};

std::uint64_t splitmix64(std::uint64_t x){
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}

std::uint64_t hash_string64(const std::string& s){
    std::uint64_t h=1469598103934665603ULL;
    for(unsigned char c:s){ h^=c; h*=1099511628211ULL; }
    return h;
}

class SparseVirtualCube {
public:
    explicit SparseVirtualCube(const Config& cfg):cfg_(cfg){}

    std::vector<Tile> materialize(const MediaSample& sample) const {
        const std::size_t B=cfg_.tile_bytes();
        const std::size_t nbytes=std::max<std::size_t>(1,sample.bytes.size());
        const std::size_t chunks=(nbytes+B-1)/B;
        const std::size_t ntiles=std::min<std::size_t>(chunks,cfg_.max_tiles_per_sample);

        std::vector<std::size_t> indices;
        if(chunks<=ntiles){
            for(std::size_t i=0;i<chunks;++i) indices.push_back(i);
        }else{
            for(std::size_t i=0;i<ntiles;++i){
                const double u=ntiles==1?0.0:static_cast<double>(i)/static_cast<double>(ntiles-1);
                indices.push_back(static_cast<std::size_t>(std::llround(u*static_cast<double>(chunks-1))));
            }
        }

        std::vector<Tile> out;
        const std::uint64_t base=hash_string64(sample.name)
            ^(static_cast<std::uint64_t>(sample.modality)<<32);

        for(const std::size_t chunk_index:indices){
            const std::size_t begin=chunk_index*B;
            const std::size_t valid=begin<sample.bytes.size()
                ?std::min(B,sample.bytes.size()-begin):0;
            std::vector<std::uint8_t> resident(B,0);
            if(valid){
                std::copy_n(sample.bytes.begin()+static_cast<std::ptrdiff_t>(begin),
                            valid,resident.begin());
            }

            const std::uint64_t h0=splitmix64(base^chunk_index);
            const std::uint64_t h1=splitmix64(h0);
            const std::uint64_t h2=splitmix64(h1);
            TileAddress a{
                h0%cfg_.logical_axis_bytes,
                h1%cfg_.logical_axis_bytes,
                h2%cfg_.logical_axis_bytes,
                static_cast<int>(sample.modality),
                static_cast<std::uint64_t>(chunk_index)
            };
            out.push_back({a,std::move(resident),valid,sample.name});
        }
        return out;
    }

private:
    Config cfg_;
};

// ---------------------------------------------------------------------------
// 4. Cubic scalar volumes + deterministic block codec
// ---------------------------------------------------------------------------

struct Volume {
    int n{};
    std::vector<double> v;
    Volume()=default;
    explicit Volume(int N,double init=0.0)
        :n(N),v(static_cast<std::size_t>(N)*N*N,init){}
    std::size_t idx(int x,int y,int z) const {
        return (static_cast<std::size_t>(x)*n+y)*n+z;
    }
    double& at(int x,int y,int z){ return v[idx(x,y,z)]; }
    double at(int x,int y,int z) const { return v[idx(x,y,z)]; }
    std::size_t size() const { return v.size(); }
};

Volume tile_to_volume(const Tile& tile,int side){
    Volume x(side);
    for(std::size_t i=0;i<x.size();++i) x.v[i]=tile.bytes[i]/255.0;
    return x;
}

Volume block_encode(const Volume& x,int block){
    if(x.n%block!=0) throw std::runtime_error("block_encode: incompatible shape");
    const int m=x.n/block;
    Volume z(m,0.0);
    const double inv=1.0/static_cast<double>(block*block*block);
    for(int i=0;i<m;++i) for(int j=0;j<m;++j) for(int k=0;k<m;++k){
        double s=0;
        for(int a=0;a<block;++a) for(int b=0;b<block;++b) for(int c=0;c<block;++c)
            s+=x.at(i*block+a,j*block+b,k*block+c);
        z.at(i,j,k)=s*inv;
    }
    return z;
}

Volume block_decode(const Volume& z,int block){
    Volume x(z.n*block,0.0);
    for(int i=0;i<z.n;++i) for(int j=0;j<z.n;++j) for(int k=0;k<z.n;++k)
        for(int a=0;a<block;++a) for(int b=0;b<block;++b) for(int c=0;c<block;++c)
            x.at(i*block+a,j*block+b,k*block+c)=z.at(i,j,k);
    return x;
}

double mse(const Volume& a,const Volume& b){
    if(a.n!=b.n) throw std::runtime_error("mse: shape mismatch");
    long double s=0;
    for(std::size_t i=0;i<a.size();++i){
        const long double d=a.v[i]-b.v[i];
        s+=d*d;
    }
    return static_cast<double>(s/a.size());
}

double rms(const Volume& a){
    long double s=0;
    for(double x:a.v) s+=static_cast<long double>(x)*x;
    return std::sqrt(static_cast<double>(s/a.size()));
}

Volume add(const Volume& a,const Volume& b,double sa=1.0,double sb=1.0){
    if(a.n!=b.n) throw std::runtime_error("add: shape mismatch");
    Volume r(a.n);
    for(std::size_t i=0;i<a.size();++i) r.v[i]=sa*a.v[i]+sb*b.v[i];
    return r;
}

void clamp01(Volume& a){
    for(double& x:a.v) x=std::clamp(x,0.0,1.0);
}

// ---------------------------------------------------------------------------
// 5. Tile encoder + orchestrated fusion
// ---------------------------------------------------------------------------

struct EncodedTile {
    Tile tile;
    Volume source;
    Volume latent;
    double weight{};
};

double tile_complexity(const Volume& x){
    const long double mean=std::accumulate(x.v.begin(),x.v.end(),0.0L)/x.size();
    long double var=0;
    for(double q:x.v){ const long double d=q-mean; var+=d*d; }
    var/=x.size();

    long double grad=0;
    std::size_t cnt=0;
    for(int i=0;i<x.n-1;++i) for(int j=0;j<x.n-1;++j) for(int k=0;k<x.n-1;++k){
        grad+=std::abs(x.at(i+1,j,k)-x.at(i,j,k));
        grad+=std::abs(x.at(i,j+1,k)-x.at(i,j,k));
        grad+=std::abs(x.at(i,j,k+1)-x.at(i,j,k));
        cnt+=3;
    }
    const double g=cnt?static_cast<double>(grad/cnt):0.0;
    return std::sqrt(static_cast<double>(var))+0.5*g;
}

std::vector<EncodedTile> encode_tiles(const std::vector<Tile>& tiles,const Config& cfg){
    std::vector<EncodedTile> out;
    std::vector<double> raw;
    out.reserve(tiles.size()); raw.reserve(tiles.size());

    for(const auto& t:tiles){
        Volume x=tile_to_volume(t,cfg.tile_side);
        Volume z=block_encode(x,cfg.block);
        const double c=tile_complexity(x);
        const double modality_prior=1.0+0.04*t.address.modality;
        raw.push_back(std::exp(std::min(4.0,c*modality_prior)));
        out.push_back({t,std::move(x),std::move(z),0.0});
    }

    const double sumw=std::accumulate(raw.begin(),raw.end(),0.0);
    for(std::size_t i=0;i<out.size();++i) out[i].weight=raw[i]/std::max(sumw,1e-12);
    return out;
}

Volume fuse_latents(const std::vector<EncodedTile>& tiles,int latent_n){
    Volume z(latent_n,0.0);
    for(const auto& t:tiles)
        for(std::size_t i=0;i<z.size();++i) z.v[i]+=t.weight*t.latent.v[i];
    return z;
}

// ---------------------------------------------------------------------------
// 6. Possibility field + geometric inward recurrence
// ---------------------------------------------------------------------------

Vec3 torus_field(const Vec3& p){
    const double rho=std::sqrt(p.x*p.x+p.y*p.y)+1e-6;
    return normalize({-p.y/rho,p.x/rho,0.45*std::sin(2*PI*p.z)});
}

Vec3 lorenz_field(const Vec3& p){
    constexpr double sigma=10.0,rho=28.0,beta=8.0/3.0;
    const double X=12*p.x,Y=12*p.y,Z=14*(p.z+1.0);
    return normalize({sigma*(Y-X),X*(rho-Z)-Y,X*Y-beta*Z});
}

Vec3 hyperfold_field(const Vec3& p){
    const double r2=p.x*p.x+p.y*p.y+p.z*p.z;
    const double inv=1.0/(0.30+r2);
    return normalize({
        inv*(p.y-p.z)-0.25*p.x,
        inv*(p.z-p.x)-0.25*p.y,
        inv*(p.x-p.y)-0.25*p.z
    });
}

Vec3 combined_field(const Vec3& p,const Config& cfg){
    const Vec3 a=torus_field(p),b=lorenz_field(p),c=hyperfold_field(p);
    return normalize(a*cfg.a_torus+b*cfg.a_lorenz+c*cfg.a_hyper);
}

double sample_trilinear_clamp(const Volume& v,double sx,double sy,double sz){
    const double hi=v.n-1.000001;
    sx=std::clamp(sx,0.0,hi); sy=std::clamp(sy,0.0,hi); sz=std::clamp(sz,0.0,hi);
    const int x0=static_cast<int>(std::floor(sx)),x1=std::min(x0+1,v.n-1);
    const int y0=static_cast<int>(std::floor(sy)),y1=std::min(y0+1,v.n-1);
    const int z0=static_cast<int>(std::floor(sz)),z1=std::min(z0+1,v.n-1);
    const double wx=sx-x0,wy=sy-y0,wz=sz-z0;

    const double c000=v.at(x0,y0,z0),c100=v.at(x1,y0,z0);
    const double c010=v.at(x0,y1,z0),c110=v.at(x1,y1,z0);
    const double c001=v.at(x0,y0,z1),c101=v.at(x1,y0,z1);
    const double c011=v.at(x0,y1,z1),c111=v.at(x1,y1,z1);
    const double c00=c000*(1-wx)+c100*wx;
    const double c10=c010*(1-wx)+c110*wx;
    const double c01=c001*(1-wx)+c101*wx;
    const double c11=c011*(1-wx)+c111*wx;
    const double c0=c00*(1-wy)+c10*wy;
    const double c1=c01*(1-wy)+c11*wy;
    return c0*(1-wz)+c1*wz;
}

Volume possibility_field(const Volume& z,const Volume& omega,const Config& cfg){
    Volume p(z.n);
    double lo=std::numeric_limits<double>::infinity();
    double hi=-std::numeric_limits<double>::infinity();
    for(std::size_t i=0;i<z.size();++i){
        const double q=std::exp(-std::abs(omega.v[i])/std::max(cfg.possibility_temp,1e-6));
        p.v[i]=q; lo=std::min(lo,q); hi=std::max(hi,q);
    }
    const double d=std::max(hi-lo,1e-12);
    for(double& q:p.v) q=(q-lo)/d;
    return p;
}

Volume inward_fold(const Volume& z,const Volume& omega,int iter,const Config& cfg){
    Volume out(z.n);
    const Volume poss=possibility_field(z,omega,cfg);
    const Mat3 invR=rotation_matrix(
        0.020*(iter+1),0.016*(iter+1),0.024*(iter+1)
    ).transpose();

    for(int i=0;i<z.n;++i) for(int j=0;j<z.n;++j) for(int k=0;k<z.n;++k){
        const Vec3 p{
            -1.0+2.0*i/std::max(1,z.n-1),
            -1.0+2.0*j/std::max(1,z.n-1),
            -1.0+2.0*k/std::max(1,z.n-1)
        };

        Vec3 q=invR.apply(p)/cfg.lambda_contract;
        const Vec3 v0=combined_field(q,cfg);
        const Vec3 qh=q-v0*(0.5*cfg.dt);
        const Vec3 vh=combined_field(qh,cfg);
        const Vec3 qs=q-vh*cfg.dt;

        const double sx=(qs.x+1.0)*0.5*(z.n-1);
        const double sy=(qs.y+1.0)*0.5*(z.n-1);
        const double sz=(qs.z+1.0)*0.5*(z.n-1);
        const double warped=sample_trilinear_clamp(z,sx,sy,sz);
        const double envelope=std::exp(-0.020*(iter+1)*norm(p)*norm(p));
        const std::size_t id=z.idx(i,j,k);

        out.v[id]=warped*envelope
                 +0.08*omega.v[id]
                 +cfg.possibility_gain*(poss.v[id]-0.5);
    }
    clamp01(out);
    return out;
}

// ---------------------------------------------------------------------------
// 7. Decoder, residual memory and CTR
// ---------------------------------------------------------------------------

double decode_score(const std::vector<EncodedTile>& tiles,const Volume& global,const Config& cfg){
    long double s=0;
    std::size_t count=0;
    for(const auto& t:tiles){
        Volume contextual=add(t.latent,global,1.0,cfg.context_gain);
        clamp01(contextual);
        const Volume xhat=block_decode(contextual,cfg.block);
        s+=mse(t.source,xhat)*static_cast<double>(t.source.size());
        count+=t.source.size();
    }
    return count?static_cast<double>(s/count):0.0;
}

Volume aggregate_residual_latent(
    const std::vector<EncodedTile>& tiles,const Volume& global,const Config& cfg
){
    Volume r(global.n,0.0);
    double totalw=0.0;
    for(const auto& t:tiles){
        Volume contextual=add(t.latent,global,1.0,cfg.context_gain);
        clamp01(contextual);
        const Volume xhat=block_decode(contextual,cfg.block);
        Volume e(t.source.n);
        for(std::size_t i=0;i<e.size();++i) e.v[i]=t.source.v[i]-xhat.v[i];
        const Volume er=block_encode(e,cfg.block);
        for(std::size_t i=0;i<r.size();++i) r.v[i]+=t.weight*er.v[i];
        totalw+=t.weight;
    }
    if(totalw>1e-12) for(double& x:r.v) x/=totalw;
    return r;
}

struct Receipt {
    int iter{};
    std::string branch;
    double mse{};
    double omega_rms{};
    double latent_step_rms{};
    bool accepted{};
};

struct EngineResult {
    Volume zstar;
    Volume omega;
    std::vector<Receipt> receipts;
    double initial_mse{};
    double final_mse{};
};

EngineResult run_engine(const std::vector<EncodedTile>& tiles,const Config& cfg){
    if(tiles.empty()) throw std::runtime_error("no active tiles");

    Volume z=fuse_latents(tiles,cfg.latent_side());
    Volume omega(z.n,0.0);
    double current=decode_score(tiles,z,cfg);
    const double initial=current;

    std::vector<Receipt> receipts{{0,"initial",current,0.0,0.0,true}};

    for(int k=1;k<=cfg.iterations;++k){
        const Volume residual=aggregate_residual_latent(tiles,z,cfg);
        Volume omega_candidate(z.n);
        for(std::size_t i=0;i<z.size();++i)
            omega_candidate.v[i]=cfg.beta_memory*omega.v[i]
                +(1.0-cfg.beta_memory)*residual.v[i];

        Volume c_res=add(z,omega_candidate,1.0,cfg.residual_gain);
        clamp01(c_res);

        const Volume folded=inward_fold(z,omega_candidate,k,cfg);
        Volume c_geo(z.n);
        for(std::size_t i=0;i<z.size();++i)
            c_geo.v[i]=0.90*z.v[i]+0.10*folded.v[i]
                +cfg.residual_gain*omega_candidate.v[i];
        clamp01(c_geo);

        const double loss_res=decode_score(tiles,c_res,cfg);
        const double loss_geo=decode_score(tiles,c_geo,cfg);

        std::string branch="rollback";
        Volume* winner=nullptr;
        double win_loss=current;
        if(std::isfinite(loss_res)&&loss_res<win_loss){
            winner=&c_res; win_loss=loss_res; branch="residual";
        }
        if(std::isfinite(loss_geo)&&loss_geo<win_loss){
            winner=&c_geo; win_loss=loss_geo; branch="geometric";
        }

        if(winner){
            const Volume step=add(*winner,z,1.0,-1.0);
            z=*winner;
            omega=omega_candidate;
            current=win_loss;
            receipts.push_back({k,branch,current,rms(omega),rms(step),true});
        }else{
            for(std::size_t i=0;i<omega.size();++i)
                omega.v[i]=0.98*omega.v[i]+0.02*omega_candidate.v[i];
            receipts.push_back({k,"rollback",current,rms(omega),0.0,false});
        }
    }

    return {z,omega,receipts,initial,current};
}

// ---------------------------------------------------------------------------
// 8. Arithmetic planner integration point
// ---------------------------------------------------------------------------

enum class ArithmeticPlan { Execute,Reduce3D,ComposeOperator,SolveFixedPoint };

std::string plan_name(ArithmeticPlan p){
    switch(p){
        case ArithmeticPlan::Reduce3D:return "REDUCE_3D";
        case ArithmeticPlan::ComposeOperator:return "COMPOSE_OPERATOR";
        case ArithmeticPlan::SolveFixedPoint:return "SOLVE_FIXED_POINT";
        default:return "EXECUTE";
    }
}

ArithmeticPlan select_plan(bool associative,bool repeated,bool contractive){
    if(contractive) return ArithmeticPlan::SolveFixedPoint;
    if(repeated) return ArithmeticPlan::ComposeOperator;
    if(associative) return ArithmeticPlan::Reduce3D;
    return ArithmeticPlan::Execute;
}

// ---------------------------------------------------------------------------
// 9. End-to-end 3D architecture map
// ---------------------------------------------------------------------------

struct MapNode {
    std::string id,label;
    Vec3 p;
    double size{0.8};
};
struct MapEdge { int a{},b{}; std::string label; };

class GeometryMap {
public:
    int add_node(std::string id,std::string label,Vec3 p,double size=0.8){
        nodes.push_back({std::move(id),std::move(label),p,size});
        return static_cast<int>(nodes.size()-1);
    }
    void add_edge(int a,int b,std::string label=""){
        edges.push_back({a,b,std::move(label)});
    }

    void export_obj(const std::string& path,const std::vector<Tile>& active,const Config& cfg) const {
        std::ofstream f(path);
        if(!f) throw std::runtime_error("cannot write "+path);
        f<<"# Jarvis-X / Moagi end-to-end 3D architecture\n";
        f<<"# 1000MB^3 logical substrate = 10^27 virtual coordinates; schematic / not to scale\n\n";
        std::size_t vertex_base=1;

        auto emit_cube=[&](const MapNode& n){
            const double s=n.size/2.0;
            const std::array<Vec3,8> v{{
                {n.p.x-s,n.p.y-s,n.p.z-s},{n.p.x+s,n.p.y-s,n.p.z-s},
                {n.p.x+s,n.p.y+s,n.p.z-s},{n.p.x-s,n.p.y+s,n.p.z-s},
                {n.p.x-s,n.p.y-s,n.p.z+s},{n.p.x+s,n.p.y-s,n.p.z+s},
                {n.p.x+s,n.p.y+s,n.p.z+s},{n.p.x-s,n.p.y+s,n.p.z+s}
            }};
            f<<"o "<<n.id<<"\n# "<<n.label<<"\n";
            for(const auto& p:v) f<<"v "<<p.x<<" "<<p.y<<" "<<p.z<<"\n";
            const std::array<std::pair<int,int>,12> e{{
                {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
                {0,4},{1,5},{2,6},{3,7}
            }};
            for(auto [a,b]:e) f<<"l "<<vertex_base+a<<" "<<vertex_base+b<<"\n";
            vertex_base+=8;
        };

        for(const auto& n:nodes) emit_cube(n);

        f<<"\no stage_connections\n";
        const std::size_t centers=vertex_base;
        for(const auto& n:nodes) f<<"v "<<n.p.x<<" "<<n.p.y<<" "<<n.p.z<<"\n";
        for(const auto& e:edges) f<<"l "<<centers+e.a<<" "<<centers+e.b<<"\n";
        vertex_base+=nodes.size();

        // Active logical coordinates normalized into a local cube around SPARSE.
        f<<"\no active_sparse_tiles\n";
        const Vec3 anchor{-7,0,0};
        for(const auto& t:active){
            const double nx=static_cast<double>(t.address.x)/static_cast<double>(cfg.logical_axis_bytes)-0.5;
            const double ny=static_cast<double>(t.address.y)/static_cast<double>(cfg.logical_axis_bytes)-0.5;
            const double nz=static_cast<double>(t.address.z)/static_cast<double>(cfg.logical_axis_bytes)-0.5;
            f<<"v "<<anchor.x+2.2*nx<<" "<<anchor.y+2.2*ny<<" "<<anchor.z+2.2*nz<<"\n";
        }
        if(!active.empty()){
            f<<"p";
            for(std::size_t i=0;i<active.size();++i) f<<" "<<vertex_base+i;
            f<<"\n";
            vertex_base+=active.size();
        }

        // Nested fixed-point shells around the inward core.
        const Vec3 core{2,0,0};
        for(double s:{2.4,1.8,1.3,0.90,0.55,0.30})
            emit_cube({"latent_shell","inward latent shell",core,s});

        // Closed Reality feedback path from ACT/SERVE back toward REALITY.
        f<<"\no reality_feedback_loop\n";
        constexpr int N=160;
        const std::size_t start=vertex_base;
        for(int i=0;i<N;++i){
            const double u=static_cast<double>(i)/(N-1);
            const double x=10.5-21.5*u;
            const double y=3.0+1.15*std::sin(2*PI*u);
            const double z=0.75*std::sin(4*PI*u);
            f<<"v "<<x<<" "<<y<<" "<<z<<"\n";
        }
        for(int i=0;i<N-1;++i) f<<"l "<<start+i<<" "<<start+i+1<<"\n";
    }

    void export_dot(const std::string& path) const {
        std::ofstream f(path);
        if(!f) throw std::runtime_error("cannot write "+path);
        f<<"digraph Moagi3D {\n  rankdir=LR;\n  node [shape=box,style=rounded];\n";
        for(const auto& n:nodes) f<<"  \""<<n.id<<"\" [label=\""<<n.label<<"\"];\n";
        for(const auto& e:edges){
            f<<"  \""<<nodes[e.a].id<<"\" -> \""<<nodes[e.b].id<<"\"";
            if(!e.label.empty()) f<<" [label=\""<<e.label<<"\"]";
            f<<";\n";
        }
        f<<"}\n";
    }

    std::vector<MapNode> nodes;
    std::vector<MapEdge> edges;
};

GeometryMap build_system_map(){
    GeometryMap g;
    const int reality=g.add_node("reality","REALITY / DATA SOURCES",{-11,0,0},1.0);
    const int ingest=g.add_node("ingest","MULTIMODAL INGEST",{-9,0,0},0.9);
    const int sparse=g.add_node("sparse","SPARSE VIRTUAL 1000MB^3",{-7,0,0},1.2);
    const int tiles=g.add_node("tiles","ACTIVE TILE FABRIC",{-5,0,0},0.9);
    const int encoder=g.add_node("encoder","ENCODER E_theta",{-3,0,0},0.9);
    const int latents=g.add_node("latents","DISTRIBUTED TILE LATENTS",{-1.2,1.1,0},0.9);
    const int memory=g.add_node("memory","MEMORY Omega",{-1.2,-1.35,0},0.75);
    const int fusion=g.add_node("fusion","ORCHESTRATOR / FUSION",{0.4,1.1,0},0.85);
    const int poss=g.add_node("possibility","POSSIBILITY FIELD",{0.4,-1.35,0},0.75);
    const int inward=g.add_node("inward","PHI_in / Z*",{2,0,0},1.0);
    const int arith=g.add_node("arithmetic","ARITHMETIC PLANNER",{2,2.0,0},0.75);
    const int decoder=g.add_node("decoder","DECODER D_phi",{4,0,0},0.9);
    const int resid=g.add_node("residual","RESIDUAL e",{5.8,-1.2,0},0.7);
    const int codec=g.add_node("codec","QUANTIZE / CODEC",{6.0,1.2,0},0.8);
    const int ctr=g.add_node("ctr","CTR VERIFY / RECKON",{8,0,0},1.0);
    const int serve=g.add_node("serve","COMMIT / SERVE / ACT",{10.5,0,0},1.0);

    g.add_edge(reality,ingest);
    g.add_edge(ingest,sparse);
    g.add_edge(sparse,tiles);
    g.add_edge(tiles,encoder);
    g.add_edge(encoder,latents);
    g.add_edge(latents,fusion);
    g.add_edge(latents,memory,"state");
    g.add_edge(memory,inward,"Omega");
    g.add_edge(fusion,inward,"Z0");
    g.add_edge(poss,inward,"P(Z)");
    g.add_edge(tiles,arith,"structured work");
    g.add_edge(arith,inward,"solver/reduction");
    g.add_edge(inward,decoder,"Z*");
    g.add_edge(decoder,resid,"Xhat");
    g.add_edge(resid,memory,"update");
    g.add_edge(resid,poss,"update");
    g.add_edge(decoder,codec);
    g.add_edge(codec,ctr);
    g.add_edge(resid,ctr,"metrics");
    g.add_edge(ctr,serve,"commit");
    g.add_edge(serve,reality,"closed loop");
    return g;
}

// ---------------------------------------------------------------------------
// 10. Telemetry
// ---------------------------------------------------------------------------

std::string json_escape(const std::string& s){
    std::ostringstream o;
    for(char c:s){
        if(c=='\\') o<<"\\\\";
        else if(c=='\"') o<<"\\\"";
        else if(c=='\n') o<<"\\n";
        else o<<c;
    }
    return o.str();
}

void write_json(
    const std::string& path,
    const Config& cfg,
    const std::vector<MediaSample>& samples,
    const std::vector<Tile>& tiles,
    const EngineResult& result,
    ArithmeticPlan plan,
    const GeometryMap& map
){
    std::ofstream f(path);
    if(!f) throw std::runtime_error("cannot write "+path);
    f<<std::setprecision(12);
    f<<"{\n";
    f<<"  \"logical_space\": {\n";
    f<<"    \"axis_bytes\": "<<cfg.logical_axis_bytes<<",\n";
    f<<"    \"coordinate_count_symbolic\": \"(10^9)^3 = 10^27\",\n";
    f<<"    \"dense_allocation\": false,\n";
    f<<"    \"resident_tile_count\": "<<tiles.size()<<"\n";
    f<<"  },\n";

    f<<"  \"samples\": [\n";
    for(std::size_t i=0;i<samples.size();++i){
        f<<"    {\"name\":\""<<json_escape(samples[i].name)
         <<"\",\"modality\":\""<<modality_name(samples[i].modality)
         <<"\",\"bytes\":"<<samples[i].bytes.size()<<"}";
        if(i+1<samples.size()) f<<",";
        f<<"\n";
    }
    f<<"  ],\n";

    f<<"  \"active_tiles\": [\n";
    for(std::size_t i=0;i<tiles.size();++i){
        const auto& t=tiles[i];
        f<<"    {\"sample\":\""<<json_escape(t.sample_name)
         <<"\",\"modality\":"<<t.address.modality
         <<",\"sequence\":"<<t.address.sequence
         <<",\"x\":"<<t.address.x
         <<",\"y\":"<<t.address.y
         <<",\"z\":"<<t.address.z
         <<",\"valid_bytes\":"<<t.valid_bytes<<"}";
        if(i+1<tiles.size()) f<<",";
        f<<"\n";
    }
    f<<"  ],\n";

    f<<"  \"latent_engine\": {\n";
    f<<"    \"latent_side\": "<<result.zstar.n<<",\n";
    f<<"    \"initial_mse\": "<<result.initial_mse<<",\n";
    f<<"    \"final_mse\": "<<result.final_mse<<",\n";
    f<<"    \"omega_rms\": "<<rms(result.omega)<<",\n";
    f<<"    \"receipts\": [\n";
    for(std::size_t i=0;i<result.receipts.size();++i){
        const auto& r=result.receipts[i];
        f<<"      {\"iter\":"<<r.iter
         <<",\"branch\":\""<<r.branch
         <<"\",\"mse\":"<<r.mse
         <<",\"omega_rms\":"<<r.omega_rms
         <<",\"latent_step_rms\":"<<r.latent_step_rms
         <<",\"accepted\":"<<(r.accepted?"true":"false")<<"}";
        if(i+1<result.receipts.size()) f<<",";
        f<<"\n";
    }
    f<<"    ]\n";
    f<<"  },\n";

    f<<"  \"arithmetic_planner\": {\"selected_plan\": \""<<plan_name(plan)<<"\"},\n";
    f<<"  \"geometry_map\": {\"nodes\": "<<map.nodes.size()
     <<", \"edges\": "<<map.edges.size()
     <<", \"obj\": \""<<json_escape(cfg.out_prefix+".obj")
     <<"\", \"dot\": \""<<json_escape(cfg.out_prefix+".dot")<<"\"}\n";
    f<<"}\n";
}

// ---------------------------------------------------------------------------
// 11. CLI
// ---------------------------------------------------------------------------

Config parse_args(int argc,char** argv){
    Config cfg;
    for(int i=1;i<argc;++i){
        const std::string a=argv[i];
        auto need=[&](const std::string& opt){
            if(i+1>=argc) throw std::runtime_error("missing value for "+opt);
            return std::string(argv[++i]);
        };
        if(a=="--tile-side") cfg.tile_side=std::stoi(need(a));
        else if(a=="--block") cfg.block=std::stoi(need(a));
        else if(a=="--max-tiles") cfg.max_tiles_per_sample=std::stoi(need(a));
        else if(a=="--iterations") cfg.iterations=std::stoi(need(a));
        else if(a=="--seed") cfg.seed=static_cast<std::uint32_t>(std::stoul(need(a)));
        else if(a=="--out") cfg.out_prefix=need(a);
        else if(a=="--help"){
            std::cout
                <<"Usage: "<<argv[0]<<" [options]\n"
                <<"  --tile-side N\n  --block N\n  --max-tiles N\n"
                <<"  --iterations N\n  --seed N\n  --out PREFIX\n";
            std::exit(0);
        }else throw std::runtime_error("unknown argument: "+a);
    }
    if(cfg.tile_side<4||cfg.block<1||cfg.tile_side%cfg.block!=0)
        throw std::runtime_error("tile-side must be >=4 and divisible by block");
    if(cfg.max_tiles_per_sample<1||cfg.iterations<1)
        throw std::runtime_error("max-tiles and iterations must be >=1");
    return cfg;
}

} // namespace moagi

int main(int argc,char** argv){
    using namespace moagi;
    try{
        const Config cfg=parse_args(argc,argv);
        const auto samples=make_samples(cfg);
        const SparseVirtualCube cube(cfg);

        std::vector<Tile> active_tiles;
        for(const auto& s:samples){
            auto ts=cube.materialize(s);
            active_tiles.insert(active_tiles.end(),
                std::make_move_iterator(ts.begin()),std::make_move_iterator(ts.end()));
        }

        const auto encoded=encode_tiles(active_tiles,cfg);
        const EngineResult result=run_engine(encoded,cfg);

        // Illustrative planner choice for repeated contractive structured work.
        const ArithmeticPlan plan=select_plan(true,true,true);

        const GeometryMap map=build_system_map();
        map.export_obj(cfg.out_prefix+".obj",active_tiles,cfg);
        map.export_dot(cfg.out_prefix+".dot");
        write_json(cfg.out_prefix+".json",cfg,samples,active_tiles,result,plan,map);

        const int accepts=static_cast<int>(std::count_if(
            result.receipts.begin()+1,result.receipts.end(),
            [](const Receipt& r){return r.accepted;}
        ));

        std::cout<<std::fixed<<std::setprecision(8);
        std::cout<<"=== Jarvis-X / Moagi 3D system map ===\n";
        std::cout<<"logical substrate  : 1000MB x 1000MB x 1000MB = 10^27 logical coordinates\n";
        std::cout<<"resident tiles     : "<<active_tiles.size()<<"\n";
        std::cout<<"tile volume        : "<<cfg.tile_side<<"^3 = "<<cfg.tile_bytes()<<" bytes\n";
        std::cout<<"latent volume      : "<<cfg.latent_side()<<"^3 scalar cells\n";
        std::cout<<"initial MSE        : "<<result.initial_mse<<"\n";
        std::cout<<"final MSE          : "<<result.final_mse<<"\n";
        std::cout<<"Omega RMS          : "<<rms(result.omega)<<"\n";
        std::cout<<"CTR accepts        : "<<accepts<<"/"<<cfg.iterations<<"\n";
        std::cout<<"arithmetic plan    : "<<plan_name(plan)<<"\n";
        std::cout<<"3D OBJ map         : "<<cfg.out_prefix<<".obj\n";
        std::cout<<"graph topology     : "<<cfg.out_prefix<<".dot\n";
        std::cout<<"telemetry          : "<<cfg.out_prefix<<".json\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"fatal: "<<e.what()<<"\n";
        return 1;
    }
}