// inward_graphics_ann_main.cpp
// Closed-loop 3D graphics + 3D ANN terminal engine.
//
// Internal framebuffer: 400 x 200 = exactly 80,000 RGB/depth pixels.
// Neural field: 16 x 16 x 16 voxels x 8 hidden channels.
//
// Closed loop per frame:
//   3D geometry -> raster framebuffer X_t
//   -> depth-aware 3D encoder E(X_t) -> latent volume H_t
//   -> recurrent 3D ANN N_theta(H_t)
//   -> differentiable-ish voxel decoder D(H_t) -> Xhat_t
//   -> residual R_t = X_t - Xhat_t
//   -> depth backprojection B(R_t) -> feedback volume
//   -> H_{t+1}, and bounded latent deformation of next geometry.
//
// This is a finite software emulation. It performs real numerical updates,
// but makes no claim of autonomous intelligence or unbounded self-improvement.
//
// Linux/macOS:
//   g++ -std=c++17 -O3 -march=native inward_graphics_ann_main.cpp -o inward_ann3d
//   ./inward_ann3d
//
// Headless verification:
//   ./inward_ann3d --headless --frames 120
//
// Windows (MSVC Developer Prompt):
//   cl /std:c++17 /O2 /EHsc inward_graphics_ann_main.cpp

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "qsol_reverberation3d.hpp"

#ifdef _WIN32
  #define NOMINMAX
  #include <windows.h>
#else
  #include <sys/ioctl.h>
  #include <unistd.h>
#endif

namespace inward3d {

constexpr int FB_W = 400;
constexpr int FB_H = 200;
constexpr int FB_PIXELS = FB_W * FB_H;
static_assert(FB_PIXELS == 80000, "Framebuffer must be exactly 80K pixels");

constexpr int LAT = 16;
constexpr int LAT_VOX = LAT * LAT * LAT;
constexpr int H = 8;
constexpr float PI = 3.14159265358979323846f;
constexpr float INF = std::numeric_limits<float>::infinity();
constexpr float WORLD_MIN = -4.0f;
constexpr float WORLD_MAX =  4.0f;
constexpr float WORLD_SPAN = WORLD_MAX - WORLD_MIN;

struct Vec3 {
    float x{}, y{}, z{};
    Vec3 operator+(const Vec3& b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(const Vec3& b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
};
static inline float dot(const Vec3& a, const Vec3& b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
static inline Vec3 normalize(Vec3 v) {
    float inv = 1.0f/std::sqrt(std::max(1e-12f, dot(v,v)));
    return v*inv;
}
static inline float sigmoid(float x) {
    x = std::clamp(x, -12.0f, 12.0f);
    return 1.0f/(1.0f+std::exp(-x));
}
static inline float clamp01(float x) { return std::clamp(x,0.0f,1.0f); }

struct RGB { uint8_t r{},g{},b{}; };
struct Camera { Vec3 pos{0,0,-8}; Vec3 target{0,0,0}; float fov_deg=64.0f; };
struct Basis { Vec3 right,up,forward; };

static Basis camera_basis(const Camera& cam) {
    Vec3 f=normalize(cam.target-cam.pos);
    Vec3 r=normalize(cross(f,{0,1,0}));
    Vec3 u=normalize(cross(r,f));
    return {r,u,f};
}

static Vec3 rotate_xyz(Vec3 p,float ax,float ay,float az) {
    float c=std::cos(ax),s=std::sin(ax);
    p={p.x,p.y*c-p.z*s,p.y*s+p.z*c};
    c=std::cos(ay); s=std::sin(ay);
    p={p.x*c+p.z*s,p.y,-p.x*s+p.z*c};
    c=std::cos(az); s=std::sin(az);
    p={p.x*c-p.y*s,p.x*s+p.y*c,p.z};
    return p;
}

struct Framebuffer {
    // 80,000 RGB pixels and 80,000 depth values take about 560 KB.
    // Keep the large arrays on the heap: MSVC reserves a small default
    // thread stack, and nested self-tests otherwise exceed that reserve.
    std::vector<RGB> color;
    std::vector<float> depth;
    Framebuffer() : color(FB_PIXELS), depth(FB_PIXELS) {}
    void clear(float t) {
        for(int y=0;y<FB_H;++y){
            float v=float(y)/float(FB_H-1);
            for(int x=0;x<FB_W;++x){
                int i=y*FB_W+x;
                float wave=.5f+.5f*std::sin(.025f*x+.035f*y+t*.7f);
                color[i]={uint8_t(2+4*wave),uint8_t(5+7*(1-v)),uint8_t(10+12*v)};
                depth[i]=INF;
            }
        }
    }
    inline void put(int x,int y,float z,RGB c){
        if((unsigned)x>=(unsigned)FB_W||(unsigned)y>=(unsigned)FB_H||z<=0) return;
        int i=y*FB_W+x;
        if(z<depth[i]){depth[i]=z;color[i]=c;}
    }
};

struct FloatFramebuffer {
    std::vector<std::array<float,3>> rgb;
    std::vector<float> w;
    FloatFramebuffer():rgb(FB_PIXELS),w(FB_PIXELS){}
    void clear(){ std::fill(rgb.begin(),rgb.end(),std::array<float,3>{0,0,0}); std::fill(w.begin(),w.end(),0.0f); }
    void add(int x,int y,float a,const std::array<float,3>& c){
        if((unsigned)x>=(unsigned)FB_W||(unsigned)y>=(unsigned)FB_H||a<=0) return;
        int i=y*FB_W+x;
        rgb[i][0]+=a*c[0]; rgb[i][1]+=a*c[1]; rgb[i][2]+=a*c[2]; w[i]+=a;
    }
    std::array<float,3> get(int i) const {
        if(w[i]<=1e-8f) return {0,0,0};
        return {clamp01(rgb[i][0]/w[i]),clamp01(rgb[i][1]/w[i]),clamp01(rgb[i][2]/w[i])};
    }
};

struct Projected { int x{},y{}; float z{}; bool ok=false; };
static Projected project(const Vec3& p,const Camera& cam,const Basis& b){
    Vec3 rel=p-cam.pos;
    float vx=dot(rel,b.right),vy=dot(rel,b.up),vz=dot(rel,b.forward);
    if(vz<.08f) return {};
    float f=1.0f/std::tan(cam.fov_deg*PI/360.0f), aspect=float(FB_W)/FB_H;
    float nx=(vx/vz)*f/aspect, ny=(vy/vz)*f;
    int sx=int((nx*.5f+.5f)*(FB_W-1));
    int sy=int((-ny*.5f+.5f)*(FB_H-1));
    if(sx<-8||sx>FB_W+8||sy<-8||sy>FB_H+8) return {};
    return {sx,sy,vz,true};
}

static Vec3 unproject(int sx,int sy,float vz,const Camera& cam,const Basis& b){
    float f=1.0f/std::tan(cam.fov_deg*PI/360.0f), aspect=float(FB_W)/FB_H;
    float nx=2.0f*float(sx)/float(FB_W-1)-1.0f;
    float ny=1.0f-2.0f*float(sy)/float(FB_H-1);
    float vx=nx*vz*aspect/f;
    float vy=ny*vz/f;
    return cam.pos + b.right*vx + b.up*vy + b.forward*vz;
}

static RGB shade(RGB base,Vec3 normal,Vec3 worldPos,const Camera& cam,float pulse,float neuralTint){
    Vec3 key=normalize({-.6f,.9f,-.7f});
    Vec3 rim=normalize(cam.pos-worldPos);
    float lam=std::max(0.0f,dot(normal,key));
    float edge=std::pow(std::max(0.0f,1.0f-std::abs(dot(normal,rim))),2.0f);
    float light=.18f+.70f*lam+.40f*edge+.10f*pulse;
    float n=std::clamp(neuralTint,-1.0f,1.0f);
    auto cv=[&](float basev,float bias){return uint8_t(std::clamp(basev*light + bias*n,0.0f,255.0f));};
    return {cv(base.r,28),cv(base.g,-12),cv(base.b,34)};
}

static void splat(Framebuffer& fb,int x,int y,float z,RGB c,int radius=1){
    radius=std::clamp(radius,1,3);
    for(int oy=-radius;oy<=radius;++oy)for(int ox=-radius;ox<=radius;++ox){
        float d2=float(ox*ox+oy*oy); if(d2>radius*radius+.25f) continue;
        float a=std::max(0.0f,1.0f-.16f*d2);
        fb.put(x+ox,y+oy,z+.0002f*d2,{uint8_t(c.r*a),uint8_t(c.g*a),uint8_t(c.b*a)});
    }
}

struct Vertex { Vec3 p,n; RGB base; };
static std::vector<Vertex> make_torus(int nu=160,int nv=72,float R=1.85f,float r=.68f){
    std::vector<Vertex> out; out.reserve(size_t(nu)*nv);
    for(int iu=0;iu<nu;++iu){
        float u=2*PI*iu/nu;
        for(int iv=0;iv<nv;++iv){
            float v=2*PI*iv/nv,cv=std::cos(v),sv=std::sin(v),cu=std::cos(u),su=std::sin(u);
            float h=.5f+.5f*std::sin(3*u+2*v);
            out.push_back({{(R+r*cv)*cu,r*sv,(R+r*cv)*su},{cv*cu,sv,cv*su},
                {uint8_t(45+95*h),uint8_t(105+110*(1-h)),uint8_t(175+70*h)}});
        }
    }
    return out;
}

static int wrapi(int v){ v%=LAT; return v<0?v+LAT:v; }
static int li(int x,int y,int z){ return (wrapi(z)*LAT+wrapi(y))*LAT+wrapi(x); }
static int world_to_coord(float p){ return std::clamp(int((p-WORLD_MIN)/WORLD_SPAN*LAT),0,LAT-1); }
static Vec3 coord_to_world(int x,int y,int z){
    return {WORLD_MIN+(x+.5f)*WORLD_SPAN/LAT,
            WORLD_MIN+(y+.5f)*WORLD_SPAN/LAT,
            WORLD_MIN+(z+.5f)*WORLD_SPAN/LAT};
}

struct ANNStats {
    float reconstruction_mse=0;
    float residual_rms=0;
    float latent_rms=0;
    float feedback_energy=0;
    int active_voxels=0;
    double step_ms=0;
};
struct CycleReceipt {
    std::uint64_t frame=0;
    float baseline_mse=0;
    float candidate_mse=0;
    float authoritative_mse=0;
    float feedback_energy=0;
    float latent_rms=0;
    int active_voxels=0;
    bool parameter_commit=false;
    double step_ms=0;
};


class NeuralField3D {
public:
    using HVec=std::array<float,H>;
    std::vector<HVec> h,enc,fb,next;
    std::vector<float> encCount,fbCount;
    std::array<std::array<float,H>,H> Wself{},Wnbr{};
    std::array<std::array<float,4>,H> Win{};
    std::array<std::array<float,3>,H> Wfb{};
    std::array<float,H> bias{};
    std::array<std::array<float,H>,3> Wdec{};
    std::array<float,3> bdec{};
    std::array<float,H> Wdens{};
    float bdens=-1.4f;
    FloatFramebuffer recon;
    ANNStats stats;

    NeuralField3D():h(LAT_VOX),enc(LAT_VOX),fb(LAT_VOX),next(LAT_VOX),encCount(LAT_VOX),fbCount(LAT_VOX){init();}

    void init(){
        std::mt19937 rng(7); std::normal_distribution<float> n(0.0f,1.0f);
        for(int o=0;o<H;++o){
            bias[o]=.02f*n(rng);
            for(int i=0;i<H;++i){
                Wself[o][i]=((o==i)?.62f:0.0f)+.035f*n(rng);
                Wnbr[o][i]=((o==i)?.20f:0.0f)+.020f*n(rng);
            }
            for(int i=0;i<4;++i) Win[o][i]=.16f*n(rng);
            for(int i=0;i<3;++i) Wfb[o][i]=.11f*n(rng);
            Wdens[o]=.12f*n(rng);
        }
        for(int c=0;c<3;++c){
            bdec[c]=0;
            for(int j=0;j<H;++j) Wdec[c][j]=.13f*n(rng);
        }
    }

    HVec sample(const Vec3& p) const { return h[li(world_to_coord(p.x),world_to_coord(p.y),world_to_coord(p.z))]; }

    void encode(const Framebuffer& src,const Camera& cam,const Basis& basis){
        std::fill(enc.begin(),enc.end(),HVec{}); std::fill(encCount.begin(),encCount.end(),0.0f);
        // Depth-aware backprojection of framebuffer observations into the latent 3D lattice.
        for(int y=0;y<FB_H;y+=2) for(int x=0;x<FB_W;x+=2){
            int pi=y*FB_W+x; float z=src.depth[pi]; if(!std::isfinite(z)||z>14.0f) continue;
            Vec3 p=unproject(x,y,z,cam,basis);
            int k=li(world_to_coord(p.x),world_to_coord(p.y),world_to_coord(p.z));
            float r=src.color[pi].r/255.0f,g=src.color[pi].g/255.0f,b=src.color[pi].b/255.0f,d=clamp01(z/12.0f);
            enc[k][0]+=r; enc[k][1]+=g; enc[k][2]+=b; enc[k][3]+=1.0f-d; encCount[k]+=1.0f;
        }
        for(int k=0;k<LAT_VOX;++k) if(encCount[k]>0){
            float inv=1.0f/encCount[k]; for(int c=0;c<4;++c) enc[k][c]*=inv;
        }
    }

    void recurrent_step(int iterations=2){
        for(int it=0;it<iterations;++it){
            for(int z=0;z<LAT;++z)for(int y=0;y<LAT;++y)for(int x=0;x<LAT;++x){
                int k=li(x,y,z); HVec navg{};
                int nb[6]={li(x-1,y,z),li(x+1,y,z),li(x,y-1,z),li(x,y+1,z),li(x,y,z-1),li(x,y,z+1)};
                for(int q:nb)for(int c=0;c<H;++c) navg[c]+=h[q][c]/6.0f;
                for(int o=0;o<H;++o){
                    float a=bias[o];
                    for(int j=0;j<H;++j) a+=Wself[o][j]*h[k][j]+Wnbr[o][j]*navg[j];
                    for(int j=0;j<4;++j) a+=Win[o][j]*enc[k][j];
                    for(int j=0;j<3;++j) a+=Wfb[o][j]*fb[k][j];
                    next[k][o]=std::tanh(a);
                }
            }
            h.swap(next);
        }
    }

    std::array<float,3> decode_color(const HVec& hv) const {
        std::array<float,3> c{};
        for(int o=0;o<3;++o){float a=bdec[o];for(int j=0;j<H;++j)a+=Wdec[o][j]*hv[j];c[o]=sigmoid(a);}return c;
    }
    float density(const HVec& hv) const { float a=bdens;for(int j=0;j<H;++j)a+=Wdens[j]*hv[j];return sigmoid(a); }

    void decode_and_learn(const Framebuffer& target,const Camera& cam,const Basis& basis,float lr=.006f){
        recon.clear();
        float localLoss=0; int localN=0; int active=0;
        for(int z=0;z<LAT;++z)for(int y=0;y<LAT;++y)for(int x=0;x<LAT;++x){
            int k=li(x,y,z); Vec3 p=coord_to_world(x,y,z); auto q=project(p,cam,basis); if(!q.ok) continue;
            float den=density(h[k]); if(den<.13f) continue; ++active;
            auto c=decode_color(h[k]);
            int rad=std::clamp(int(1+den*4.0f),1,5);
            for(int oy=-rad;oy<=rad;++oy)for(int ox=-rad;ox<=rad;++ox){
                float d2=float(ox*ox+oy*oy); if(d2>rad*rad+.25f)continue;
                float a=den*std::exp(-d2/(2.0f*std::max(1.0f,float(rad*rad)*.35f)));
                recon.add(q.x+ox,q.y+oy,a,c);
            }
            // Local online decoder update using target colour at the projected voxel centre.
            if((unsigned)q.x<(unsigned)FB_W&&(unsigned)q.y<(unsigned)FB_H){
                const RGB tc=target.color[q.y*FB_W+q.x];
                float tv[3]={tc.r/255.0f,tc.g/255.0f,tc.b/255.0f};
                for(int o=0;o<3;++o){
                    float e=c[o]-tv[o]; localLoss+=e*e; ++localN;
                    float grad=e*c[o]*(1-c[o]);
                    bdec[o]-=lr*grad;
                    for(int j=0;j<H;++j) Wdec[o][j]-=lr*grad*h[k][j];
                }
                float brightness=(tv[0]+tv[1]+tv[2])/3.0f;
                float targetDen=std::clamp(brightness*1.4f,0.0f,1.0f);
                float gd=(den-targetDen)*den*(1-den);
                bdens-=lr*.35f*gd; for(int j=0;j<H;++j)Wdens[j]-=lr*.35f*gd*h[k][j];
            }
        }
        stats.active_voxels=active;
        (void)localLoss; (void)localN;
    }

    void residual_and_backproject(const Framebuffer& target,const Camera& cam,const Basis& basis){
        std::fill(fb.begin(),fb.end(),HVec{}); std::fill(fbCount.begin(),fbCount.end(),0.0f);
        double mse=0,energy=0; int n=0;
        for(int y=0;y<FB_H;++y)for(int x=0;x<FB_W;++x){
            int pi=y*FB_W+x; auto rhat=recon.get(pi);
            float tv[3]={target.color[pi].r/255.0f,target.color[pi].g/255.0f,target.color[pi].b/255.0f};
            float er[3]={tv[0]-rhat[0],tv[1]-rhat[1],tv[2]-rhat[2]};
            mse+=er[0]*er[0]+er[1]*er[1]+er[2]*er[2]; n+=3;
            if(((x|y)&1)==0){
                float z=target.depth[pi]; if(std::isfinite(z)&&z<14.0f){
                    Vec3 p=unproject(x,y,z,cam,basis); int k=li(world_to_coord(p.x),world_to_coord(p.y),world_to_coord(p.z));
                    for(int c=0;c<3;++c){ fb[k][c]+=er[c]; }
                    fbCount[k]+=1.0f;
                }
            }
        }
        for(int k=0;k<LAT_VOX;++k) if(fbCount[k]>0){
            float inv=1.0f/fbCount[k];
            for(int c=0;c<3;++c){fb[k][c]=std::clamp(fb[k][c]*inv,-1.0f,1.0f);energy+=fb[k][c]*fb[k][c];}
        }
        stats.reconstruction_mse=n?float(mse/n):0;
        stats.residual_rms=std::sqrt(stats.reconstruction_mse);
        stats.feedback_energy=std::sqrt(float(energy)/(LAT_VOX*3));
    }

    void update_stats(){
        double s=0; for(const auto& v:h)for(float x:v)s+=double(x)*x;
        stats.latent_rms=std::sqrt(float(s)/(LAT_VOX*H));
    }

    CycleReceipt closed_loop_update(const Framebuffer& src,const Camera& cam,const Basis& basis,std::uint64_t frame){
        const auto t0=std::chrono::steady_clock::now();

        NeuralField3D baseline=*this;
        baseline.encode(src,cam,basis);
        baseline.recurrent_step(2);
        baseline.decode_and_learn(src,cam,basis,0.0f);
        baseline.residual_and_backproject(src,cam,basis);
        baseline.update_stats();

        NeuralField3D candidate=baseline;
        candidate.decode_and_learn(src,cam,basis,0.006f);
        candidate.decode_and_learn(src,cam,basis,0.0f);
        candidate.residual_and_backproject(src,cam,basis);
        candidate.recurrent_step(1);
        candidate.update_stats();

        const bool finite=std::isfinite(candidate.stats.reconstruction_mse)
            && std::isfinite(candidate.stats.feedback_energy)
            && std::isfinite(candidate.stats.latent_rms);
        constexpr float kMseSlack=1.0e-7f;
        constexpr float kFeedbackLimit=0.50f;
        const bool non_regressing=candidate.stats.reconstruction_mse
            <= baseline.stats.reconstruction_mse+kMseSlack;
        const bool bounded_feedback=candidate.stats.feedback_energy<=kFeedbackLimit;
        const bool commit=finite&&non_regressing&&bounded_feedback;

        const float baseline_mse=baseline.stats.reconstruction_mse;
        const float candidate_mse=candidate.stats.reconstruction_mse;
        if(commit){
            *this=std::move(candidate);
        } else {
            *this=std::move(baseline);
        }

        stats.step_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-t0).count();
        return CycleReceipt{frame,baseline_mse,candidate_mse,stats.reconstruction_mse,
                            stats.feedback_energy,stats.latent_rms,stats.active_voxels,
                            commit,stats.step_ms};
    }

};

struct RenderContext { Camera cam; Basis basis; };

static void draw_line_3d(Framebuffer& fb,Vec3 a,Vec3 b,RGB c,const Camera& cam,const Basis& basis){
    for(int i=0;i<=180;++i){float t=float(i)/180.0f;Vec3 p=a*(1-t)+b*t;auto q=project(p,cam,basis);if(q.ok)splat(fb,q.x,q.y,q.z,c,1);}
}

static RenderContext render_scene(Framebuffer& fb,float t,uint64_t frame,const NeuralField3D& ann){
    fb.clear(t);
    Camera cam;
    cam.pos={std::sin(t*.18f)*1.8f,std::sin(t*.11f)*.65f,-8.2f+std::cos(t*.15f)*.45f};
    cam.target={0,0,0}; Basis basis=camera_basis(cam);
    static const std::vector<Vertex> torus=make_torus();
    float ax=.28f*std::sin(t*.31f),ay=t*.48f,az=.14f*std::cos(t*.23f);
    float pulse=.5f+.5f*std::sin(t*2.0f);
    float globalFeedback=std::clamp(ann.stats.feedback_energy,0.0f,.35f);

    for(const auto& v:torus){
        Vec3 p=rotate_xyz(v.p,ax,ay,az), n=normalize(rotate_xyz(v.n,ax,ay,az));
        auto hv=ann.sample(p);
        float local=std::tanh(hv[0]);
        float phase=std::atan2(p.z,p.x)*5.0f+p.y*4.0f-t*3.1f;
        float procedural=.035f*std::sin(phase);
        float neuralDisp=.055f*local + .025f*globalFeedback*std::tanh(hv[4]);
        p=p+n*(procedural+neuralDisp);
        auto q=project(p,cam,basis); if(!q.ok)continue;
        float tint=.55f*std::tanh(hv[1])+.45f*std::tanh(hv[2]);
        RGB c=shade(v.base,n,p,cam,pulse,tint); splat(fb,q.x,q.y,q.z,c,q.z<6.5f?2:1);
    }

    draw_line_3d(fb,{-3,0,0},{3,0,0},{220,80,80},cam,basis);
    draw_line_3d(fb,{0,-2.6f,0},{0,2.6f,0},{80,220,110},cam,basis);
    draw_line_3d(fb,{0,0,-3},{0,0,3},{80,130,240},cam,basis);

    // Inward feedback stream whose contraction is modulated by the ANN residual energy.
    float contraction=1.0f-.45f*std::clamp(globalFeedback,0.0f,.35f);
    for(int i=0;i<1300;++i){
        float a=2*PI*(i%260)/260.0f+t*.55f,s=float(i)/1299.0f;
        float rr=(3.7f*(1-s)+.10f)*contraction;
        Vec3 p{rr*std::cos(a),1.5f*(s-.5f)+.32f*std::sin(3*a+t),rr*std::sin(a)};
        p=rotate_xyz(p,ax*.5f,ay*.35f,az*.5f); auto q=project(p,cam,basis);
        if(q.ok)splat(fb,q.x,q.y,q.z,{uint8_t(80+150*s),uint8_t(150+80*s),245},1);
    }

    // Latent voxel probes: active neural voxels appear as small points in the 3D world.
    for(int z=0;z<LAT;++z)for(int y=0;y<LAT;++y)for(int x=0;x<LAT;++x){
        int k=li(x,y,z); float den=ann.density(ann.h[k]); if(den<.67f)continue;
        Vec3 p=coord_to_world(x,y,z); auto q=project(p,cam,basis); if(!q.ok)continue;
        auto cc=ann.decode_color(ann.h[k]);
        RGB c{uint8_t(255*cc[0]),uint8_t(255*cc[1]),uint8_t(255*cc[2])}; splat(fb,q.x,q.y,q.z,c,1);
    }

    // Core marker.
    for(int i=0;i<420;++i){
        float a=2*PI*i/420.0f,b=std::acos(1.0f-2.0f*((i+.5f)/420.0f));
        Vec3 p{.24f*std::sin(b)*std::cos(a+t),.24f*std::cos(b),.24f*std::sin(b)*std::sin(a+t)};
        auto q=project(p,cam,basis); if(q.ok)splat(fb,q.x,q.y,q.z,{250,215,125},2);
    }
    (void)frame; return {cam,basis};
}

static std::pair<int,int> terminal_size(){
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO csbi{}; if(GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE),&csbi))
        return {csbi.srWindow.Right-csbi.srWindow.Left+1,csbi.srWindow.Bottom-csbi.srWindow.Top+1};
#else
    winsize ws{}; if(ioctl(STDOUT_FILENO,TIOCGWINSZ,&ws)==0&&ws.ws_col>0&&ws.ws_row>0)return {ws.ws_col,ws.ws_row};
#endif
    return {120,40};
}
static void enable_vt(){
#ifdef _WIN32
    HANDLE h=GetStdHandle(STD_OUTPUT_HANDLE);DWORD mode=0;if(GetConsoleMode(h,&mode))SetConsoleMode(h,mode|ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}
static void terminal_enter(){enable_vt();std::cout<<"\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H"<<std::flush;}
static void terminal_leave(){std::cout<<"\x1b[0m\x1b[?25h\x1b[?1049l"<<std::flush;}
static std::atomic<bool> running{true};
static void signal_handler(int){running=false;}

static std::string frame_to_ansi(const Framebuffer& fb,const ANNStats& st,double fps,uint64_t frame){
    auto [cols,rows]=terminal_size(); int outW=std::clamp(cols,40,FB_W),outRows=std::clamp(rows-3,8,FB_H/2),outHpx=outRows*2;
    std::ostringstream ss; ss<<"\x1b[H";
    ss<<"80K 3D GRAPHICS ↔ 3D ANN  |  "<<FB_W<<'x'<<FB_H<<"="<<FB_PIXELS<<" px  |  latent "<<LAT<<'^'<<3<<'x'<<H
      <<"  |  frame "<<frame<<"  |  "<<std::fixed<<std::setprecision(1)<<fps<<" FPS\x1b[K\n";
    ss<<"loss="<<std::setprecision(5)<<st.reconstruction_mse<<"  residual="<<st.residual_rms
      <<"  latent="<<st.latent_rms<<"  fb="<<st.feedback_energy<<"  active="<<st.active_voxels
      <<"  ANN="<<std::setprecision(2)<<st.step_ms<<" ms  | Ctrl+C exits\x1b[K\n";
    RGB lastFg{255,255,255},lastBg{0,0,0};bool first=true;
    for(int oy=0;oy<outRows;++oy){
        int y0=std::min(FB_H-1,int((2*oy+.5f)*FB_H/outHpx)); int y1=std::min(FB_H-1,int((2*oy+1.5f)*FB_H/outHpx));
        for(int ox=0;ox<outW;++ox){
            int x=std::min(FB_W-1,int((ox+.5f)*FB_W/outW)); RGB fg=fb.color[y0*FB_W+x],bg=fb.color[y1*FB_W+x];
            if(first||fg.r!=lastFg.r||fg.g!=lastFg.g||fg.b!=lastFg.b){ss<<"\x1b[38;2;"<<int(fg.r)<<';'<<int(fg.g)<<';'<<int(fg.b)<<'m';lastFg=fg;}
            if(first||bg.r!=lastBg.r||bg.g!=lastBg.g||bg.b!=lastBg.b){ss<<"\x1b[48;2;"<<int(bg.r)<<';'<<int(bg.g)<<';'<<int(bg.b)<<'m';lastBg=bg;}
            first=false;ss<<"▀";
        }
        ss<<"\x1b[0m\x1b[K\n";first=true;
    }
    ss<<"\x1b[0m\x1b[J";return ss.str();
}

struct Args {
    bool headless=false;
    bool self_test=false;
    bool json=false;
    std::uint64_t maxFrames=0;
};

static void usage(const char* argv0){
    std::cout<<"Usage: "<<argv0
             <<" [--headless] [--frames N] [--json] [--self-test]\n";
}

static Args parse_args(int argc,char** argv){
    Args a;
    for(int i=1;i<argc;++i){
        std::string s=argv[i];
        if(s=="--headless") a.headless=true;
        else if(s=="--self-test"){a.self_test=true;a.headless=true;}
        else if(s=="--json") a.json=true;
        else if(s=="--frames"){
            if(i+1>=argc) throw std::invalid_argument("--frames requires a value");
            a.maxFrames=std::strtoull(argv[++i],nullptr,10);
            if(a.maxFrames==0) throw std::invalid_argument("--frames must be greater than zero");
        } else if(s=="--help"||s=="-h"){usage(argv[0]);std::exit(0);}
        else throw std::invalid_argument("unknown argument: "+s);
    }
    return a;
}

// Exact-integer adapter from the already-rendered 80K RGB framebuffer.
// Fixed spatial sampling into a 16^3 forcing field; no back-edge from QSOL readout.
static qsol3d::Field qsol_frame_input(const Framebuffer& fb){
    qsol3d::Field input{};
    for(int k=0;k<qsol3d::kVoxels;++k){
        const auto& px=fb.color[(static_cast<std::size_t>(k)*FB_PIXELS)/qsol3d::kVoxels];
        input.cells[k]=qsol3d::excitation_from_rgb(px.r,px.b);
    }
    return input;
}

static bool receipt_valid(const CycleReceipt& r){
    return std::isfinite(r.baseline_mse)
        && std::isfinite(r.candidate_mse)
        && std::isfinite(r.authoritative_mse)
        && std::isfinite(r.feedback_energy)
        && std::isfinite(r.latent_rms)
        && r.feedback_energy<=0.50f
        && r.active_voxels>0
        && (!r.parameter_commit||r.candidate_mse<=r.baseline_mse+1.0e-7f);
}

static std::string receipt_json(const CycleReceipt& r,std::uint64_t frames,
                                const qsol3d::Field& qsol_state,
                                const qsol3d::Observation& qsol_readout){
    std::ostringstream os;
    os<<std::fixed<<std::setprecision(8)
      <<"{\"engine\":\"DrMoagi-80K-Pixel-Inward-ANN\","
      <<"\"framebuffer_width\":"<<FB_W<<','
      <<"\"framebuffer_height\":"<<FB_H<<','
      <<"\"framebuffer_pixels\":"<<FB_PIXELS<<','
      <<"\"latent_edge\":"<<LAT<<','
      <<"\"latent_voxels\":"<<LAT_VOX<<','
      <<"\"hidden_channels\":"<<H<<','
      <<"\"frames\":"<<frames<<','
      <<"\"frame\":"<<r.frame<<','
      <<"\"baseline_mse\":"<<r.baseline_mse<<','
      <<"\"candidate_mse\":"<<r.candidate_mse<<','
      <<"\"authoritative_mse\":"<<r.authoritative_mse<<','
      <<"\"feedback_energy\":"<<r.feedback_energy<<','
      <<"\"latent_rms\":"<<r.latent_rms<<','
      <<"\"active_voxels\":"<<r.active_voxels<<','
      <<"\"parameter_commit\":"<<(r.parameter_commit?"true":"false")<<','
      <<"\"step_ms\":"<<r.step_ms<<','
      <<"\"qsol_voxels\":"<<qsol3d::kVoxels<<','
      <<"\"qsol_observer_units\":"<<qsol3d::kObserverUnits<<','
      <<"\"qsol_contraction_bound\":0.9375,"
      <<"\"qsol_state_hash\":\""<<qsol3d::state_hash(qsol_state)<<"\","
      <<"\"qsol_render_hash\":\""<<qsol_readout.render_hash<<"\","
      <<"\"qsol_camera_dx\":"<<qsol_readout.camera_dx<<','
      <<"\"qsol_camera_dy\":"<<qsol_readout.camera_dy<<'}';
    return os.str();
}

static bool self_test(){
    static_assert(FB_PIXELS==80000,"exact 80K-pixel framebuffer invariant");
    static_assert(LAT_VOX==4096,"16^3 latent voxel invariant");

    Camera cam;
    const Basis basis=camera_basis(cam);
    const Vec3 p{0.5f,0.25f,0.75f};
    const auto q=project(p,cam,basis);
    if(!q.ok) return false;
    const Vec3 p2=unproject(q.x,q.y,q.z,cam,basis);
    const float roundtrip=std::sqrt(dot(p2-p,p2-p));
    if(!(roundtrip<0.06f)) return false;

    Framebuffer fb;
    NeuralField3D ann;
    qsol3d::Field qsol_state{};
    int commits=0;
    CycleReceipt last{};
    constexpr std::uint64_t kFrames=18;
    for(std::uint64_t i=0;i<kFrames;++i){
        const float t=static_cast<float>(i)*0.025f;
        const RenderContext rc=render_scene(fb,t,i,ann);
        last=ann.closed_loop_update(fb,rc.cam,rc.basis,i);
        const auto excitation=qsol_frame_input(fb);
        qsol_state=qsol3d::advance(qsol_state,excitation);
        const auto state_before_observe=qsol3d::state_hash(qsol_state);
        const auto readout=qsol3d::observe(qsol_state,excitation,static_cast<std::uint32_t>(i));
        if(qsol3d::state_hash(qsol_state)!=state_before_observe ||
           qsol3d::observe(qsol_state,excitation,static_cast<std::uint32_t>(i)).render_hash
               !=readout.render_hash ||
           std::abs(readout.camera_dx)>6 || std::abs(readout.camera_dy)>6) return false;
        if(!receipt_valid(last)) return false;
        commits+=last.parameter_commit?1:0;
    }
    if(commits==0) return false;
    std::cout<<"SELF_TEST PASS pixels="<<FB_PIXELS
             <<" latent_voxels="<<LAT_VOX
             <<" commits="<<commits
             <<" final_mse="<<last.authoritative_mse
             <<" feedback="<<last.feedback_energy<<'\n';
    return true;
}

} // namespace inward3d

int main(int argc,char** argv){
    using namespace inward3d;
    try{
        Args args=parse_args(argc,argv);
        if(args.self_test) return self_test()?0:2;

        std::signal(SIGINT,signal_handler);
#ifdef SIGTERM
        std::signal(SIGTERM,signal_handler);
#endif
        if(!args.headless){terminal_enter();std::atexit(terminal_leave);}

        Framebuffer fb;
        NeuralField3D ann;
        qsol3d::Field qsol_state{};
        qsol3d::Observation qsol_readout{};
        using clock=std::chrono::steady_clock;
        auto t0=clock::now(),last=t0;
        double fps=0;
        std::uint64_t frame=0;
        CycleReceipt lastReceipt{};
        while(running && (args.maxFrames==0 || frame<args.maxFrames)){
            const auto begin=clock::now();
            const float t=args.headless ? static_cast<float>(frame)*0.025f
                : std::chrono::duration<float>(begin-t0).count();
            const RenderContext rc=render_scene(fb,t,frame,ann);
            lastReceipt=ann.closed_loop_update(fb,rc.cam,rc.basis,frame);
            const auto excitation=qsol_frame_input(fb);
            qsol_state=qsol3d::advance(qsol_state,excitation);
            // Readout is pure: not supplied to ANN, renderer, or evolution logic.
            qsol_readout=qsol3d::observe(qsol_state,excitation,static_cast<std::uint32_t>(frame));

            const auto now=clock::now();
            const double dt=std::chrono::duration<double>(now-last).count();
            last=now;
            const double inst=dt>1e-6?1.0/dt:0.0;
            fps=frame==0?inst:.90*fps+.10*inst;
            if(args.headless){
                if(!args.json && (frame%10==0 || (args.maxFrames && frame+1==args.maxFrames)))
                    std::cout<<"frame="<<frame
                             <<" baseline_mse="<<lastReceipt.baseline_mse
                             <<" candidate_mse="<<lastReceipt.candidate_mse
                             <<" authoritative_mse="<<lastReceipt.authoritative_mse
                             <<" commit="<<(lastReceipt.parameter_commit?"yes":"no")
                             <<" feedback="<<ann.stats.feedback_energy
                             <<" latent="<<ann.stats.latent_rms
                             <<" active="<<ann.stats.active_voxels
                             <<" ann_ms="<<ann.stats.step_ms<<"\n";
            } else {
                std::cout<<frame_to_ansi(fb,ann.stats,fps,frame)<<std::flush;
            }
            ++frame;
            if(!args.headless){
                const auto elapsed=clock::now()-begin;
                const auto target=std::chrono::milliseconds(33);
                if(elapsed<target) std::this_thread::sleep_for(target-elapsed);
            }
        }
        if(!args.headless) terminal_leave();
        if(args.json) std::cout<<receipt_json(lastReceipt,frame,qsol_state,qsol_readout)<<'\n';
        return receipt_valid(lastReceipt)||frame==0?0:3;
    } catch(const std::exception& e){
        std::cerr<<"fatal: "<<e.what()<<'\n';
        return 1;
    }
}
