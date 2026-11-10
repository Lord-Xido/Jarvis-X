// JARVIS-X: reproducible 3D permeation swarm CPU research prototype.
// No trained weights, no GPU acceleration, and no 10^18-voxel materialization.
// Copyright (c) 2026. See repository LICENSE.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

struct Tensor4D {
    int d=0,h=0,w=0,c=0;
    std::vector<float> data;
    Tensor4D()=default;
    Tensor4D(int D,int H,int W,int C,float v=0):d(D),h(H),w(W),c(C) {
        if(D<=0||H<=0||W<=0||C<=0) throw std::invalid_argument("positive dimensions required");
        data.assign(static_cast<size_t>(D)*H*W*C,v);
    }
    float& at(int z,int y,int x,int ch) {return data[((static_cast<size_t>(z)*h+y)*w+x)*c+ch];}
    const float& at(int z,int y,int x,int ch) const {return data[((static_cast<size_t>(z)*h+y)*w+x)*c+ch];}
};
static float sigmoid(float x) {return x>=0?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));}
static float he(std::mt19937& g,int fanin) {
    std::normal_distribution<float> n(0.f,std::sqrt(2.f/static_cast<float>(fanin)));
    return n(g);
}

struct Conv3D {
    int ic,oc;
    std::vector<float> w,b;
    Conv3D(int i,int o,std::mt19937& g):ic(i),oc(o),w(static_cast<size_t>(27)*i*o),b(o,0.f) {
        for(auto& v:w) v=he(g,27*i);
    }
    void forward(const Tensor4D& in,Tensor4D& out) const {
        if(in.c!=ic || in.d%2 || in.h%2 || in.w%2) throw std::invalid_argument("Conv3D input shape");
        out=Tensor4D(in.d/2,in.h/2,in.w/2,oc);
        #pragma omp parallel for collapse(3) schedule(static)
        for(int z=0;z<out.d;++z) for(int y=0;y<out.h;++y) for(int x=0;x<out.w;++x)
            for(int o=0;o<oc;++o) {
                float s=b[o];
                for(int kz=0;kz<3;++kz) for(int ky=0;ky<3;++ky) for(int kx=0;kx<3;++kx) {
                    int iz=2*z+kz-1,iy=2*y+ky-1,ix=2*x+kx-1;
                    if(iz<0||iy<0||ix<0||iz>=in.d||iy>=in.h||ix>=in.w) continue;
                    size_t k=static_cast<size_t>((kz*3+ky)*3+kx)*ic*oc;
                    for(int i=0;i<ic;++i) s+=in.at(iz,iy,ix,i)*w[k+i*oc+o];
                }
                out.at(z,y,x,o)=std::max(0.f,s);
            }
    }
};

// Genuine gather-form transpose of the stride-2 centered 3-tap mapping, no scatter races.
struct Deconv3D {
    int ic,oc;
    std::vector<float> w,b;
    Deconv3D(int i,int o,std::mt19937& g):ic(i),oc(o),w(static_cast<size_t>(27)*i*o),b(o,0.f) {
        for(auto& v:w) v=he(g,27*i);
    }
    void forward(const Tensor4D& in,Tensor4D& out) const {
        if(in.c!=ic) throw std::invalid_argument("Deconv3D input channels");
        out=Tensor4D(in.d*2,in.h*2,in.w*2,oc);
        #pragma omp parallel for collapse(3) schedule(static)
        for(int z=0;z<out.d;++z) for(int y=0;y<out.h;++y) for(int x=0;x<out.w;++x)
            for(int o=0;o<oc;++o) {
                float s=b[o];
                for(int kz=0;kz<3;++kz) {
                    if((z+1-kz)%2) continue;
                    int iz=(z+1-kz)/2; if(iz<0||iz>=in.d) continue;
                    for(int ky=0;ky<3;++ky) {
                        if((y+1-ky)%2) continue;
                        int iy=(y+1-ky)/2; if(iy<0||iy>=in.h) continue;
                        for(int kx=0;kx<3;++kx) {
                            if((x+1-kx)%2) continue;
                            int ix=(x+1-kx)/2; if(ix<0||ix>=in.w) continue;
                            size_t k=static_cast<size_t>((kz*3+ky)*3+kx)*ic*oc;
                            for(int i=0;i<ic;++i) s+=in.at(iz,iy,ix,i)*w[k+i*oc+o];
                        }
                    }
                }
                out.at(z,y,x,o)=s;
            }
    }
};

struct Dense {
    int ic,oc;
    std::vector<float> w,b;
    Dense(int i,int o,std::mt19937& g):ic(i),oc(o),w(static_cast<size_t>(i)*o),b(o,0.f) {
        for(auto& v:w) v=he(g,i);
    }
    std::vector<float> forward(const std::vector<float>& in) const {
        if(static_cast<int>(in.size())!=ic) throw std::invalid_argument("Dense input size");
        std::vector<float> out(oc);
        #pragma omp parallel for schedule(static)
        for(int o=0;o<oc;++o) {
            float s=b[o];
            for(int i=0;i<ic;++i) s+=in[i]*w[static_cast<size_t>(i)*oc+o];
            out[o]=s;
        }
        return out;
    }
};

// Explicit 6-neighbour zero-flux scheme. It conserves sum (up to FP roundoff).
static void diffuse(Tensor4D& v,float alpha) {
    if(v.c!=1||!std::isfinite(alpha)||alpha<0.f||alpha>1.f/6.f)
        throw std::invalid_argument("diffusion requires one channel and 0 <= alpha <= 1/6");
    const Tensor4D a=v;
    #pragma omp parallel for collapse(3) schedule(static)
    for(int z=0;z<v.d;++z) for(int y=0;y<v.h;++y) for(int x=0;x<v.w;++x) {
        float q=a.at(z,y,x,0);
        float lap=a.at(std::min(z+1,v.d-1),y,x,0)+a.at(std::max(z-1,0),y,x,0)
                 +a.at(z,std::min(y+1,v.h-1),x,0)+a.at(z,std::max(y-1,0),x,0)
                 +a.at(z,y,std::min(x+1,v.w-1),0)+a.at(z,y,std::max(x-1,0),0)-6.f*q;
        v.at(z,y,x,0)=q+alpha*lap;
    }
}

struct PermeationVAE {
    std::mt19937 init_gen,noise_gen;
    Conv3D conv1,conv2;
    Dense mu_layer,logvar_layer,project;
    Deconv3D deconv1,deconv2;
    explicit PermeationVAE(uint32_t seed)
        :init_gen(seed),noise_gen(seed^0x9e3779b9U),conv1(1,16,init_gen),conv2(16,32,init_gen),
         mu_layer(2048,16,init_gen),logvar_layer(2048,16,init_gen),project(16,2048,init_gen),
         deconv1(32,16,init_gen),deconv2(16,1,init_gen) {}
    void step(const Tensor4D& input,Tensor4D& output,float drift,float diffusion) {
        if(input.d!=16||input.h!=16||input.w!=16||input.c!=1)
            throw std::invalid_argument("reference engine accepts 16x16x16x1 only");
        Tensor4D a,b;
        conv1.forward(input,a);conv2.forward(a,b);
        auto mu=mu_layer.forward(b.data), logvar=logvar_layer.forward(b.data);
        std::normal_distribution<float> normal(0.f,1.f);
        std::vector<float> latent(16);
        for(int i=0;i<16;++i) {
            float lv=std::clamp(logvar[i],-8.f,4.f);
            latent[i]=mu[i]+std::exp(.5f*lv)*normal(noise_gen)+drift*normal(noise_gen);
        }
        auto d=project.forward(latent);
        Tensor4D field(4,4,4,32);
        field.data=std::move(d);
        Tensor4D intermediate;
        deconv1.forward(field,intermediate);
        deconv2.forward(intermediate,output);
        for(auto& q:output.data) q=sigmoid(q);
        diffuse(output,diffusion);
    }
};

struct SwarmNode {
    PermeationVAE model;
    Tensor4D state;
    explicit SwarmNode(uint32_t id,uint32_t seed):model(seed+id*1664525U),state(16,16,16,1) {
        std::mt19937 g(seed^id^0xa5a5a5a5U);
        std::uniform_real_distribution<float> r(.01f,.21f);
        for(auto& x:state.data) x=r(g);
    }
};

static Tensor4D cross_target() {
    Tensor4D t(16,16,16,1);
    for(int z=0;z<16;++z) for(int y=0;y<16;++y) for(int x=0;x<16;++x)
        if((std::abs(x-8)<2&&std::abs(y-8)<2)||
           (std::abs(x-8)<2&&std::abs(z-8)<2)||
           (std::abs(y-8)<2&&std::abs(z-8)<2)) t.at(z,y,x,0)=1.f;
    return t;
}

struct Metrics {
    double mse=0,mean=0,minimum=1,maximum=0;
    uint64_t checksum=14695981039346656037ULL;
};

class MegaSwarm {
    int side;
    std::vector<SwarmNode> nodes;
    Tensor4D target;
    int step_number=0;
public:
    explicit MegaSwarm(int s,uint32_t seed=42):side(s),target(cross_target()) {
        if(s<1||s>8) throw std::invalid_argument("swarm side must be 1..8");
        nodes.reserve(static_cast<size_t>(s)*s*s);
        for(int i=0;i<s*s*s;++i) nodes.emplace_back(static_cast<uint32_t>(i),seed);
    }
    int count() const {return static_cast<int>(nodes.size());}
    int step() const {return step_number;}
    void advance() {
        ++step_number;
        float drift=.15f+.1f*std::sin(step_number*.1f);
        float diffusion=.12f+.03f*std::cos(step_number*.2f); // always below 1/6
        // Model instances and RNGs are owned by individual nodes; no data races.
        #pragma omp parallel for schedule(static)
        for(int i=0;i<count();++i) {
            Tensor4D output;
            nodes[i].model.step(nodes[i].state,output,drift,diffusion);
            nodes[i].state=std::move(output);
        }
        // True synchronous six-neighbour coupling, from an immutable snapshot.
        // This exchanges *voxel fields*, not the internal 16-value latent vectors.
        std::vector<Tensor4D> snapshot;
        snapshot.reserve(nodes.size());
        for(const auto& n:nodes) snapshot.push_back(n.state);
        constexpr float coupling=.10f, target_pull=.05f;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<count();++i) {
            int x=i%side,y=(i/side)%side,z=i/(side*side);
            int neighbors[6],n=0;
            if(x>0)neighbors[n++]=i-1;
            if(x+1<side)neighbors[n++]=i+1;
            if(y>0)neighbors[n++]=i-side;
            if(y+1<side)neighbors[n++]=i+side;
            if(z>0)neighbors[n++]=i-side*side;
            if(z+1<side)neighbors[n++]=i+side*side;
            for(size_t k=0;k<target.data.size();++k) {
                float q=snapshot[i].data[k],avg=0.f;
                for(int j=0;j<n;++j) avg+=snapshot[neighbors[j]].data[k];
                avg=n?avg/n:q;
                float mixed=(1.f-coupling)*q+coupling*avg;
                nodes[i].state.data[k]=(1.f-target_pull)*mixed+target_pull*target.data[k];
            }
        }
    }
    Metrics metrics() const {
        Metrics m; double sum=0,err=0;
        const double divisor=static_cast<double>(nodes.size())*target.data.size();
        for(const auto& node:nodes) for(size_t k=0;k<target.data.size();++k) {
            float v=node.state.data[k];
            if(!std::isfinite(v))throw std::runtime_error("nonfinite voxel");
            m.minimum=std::min(m.minimum,static_cast<double>(v));
            m.maximum=std::max(m.maximum,static_cast<double>(v));
            sum+=v;
            double delta=static_cast<double>(v)-target.data[k];err+=delta*delta;
            uint32_t bits;std::memcpy(&bits,&v,sizeof(bits));
            for(int b=0;b<4;++b) {m.checksum^=(bits>>(8*b))&255U;m.checksum*=1099511628211ULL;}
        }
        m.mean=sum/divisor;m.mse=err/divisor;
        return m;
    }
};

static void require(bool yes,const char* message) {if(!yes)throw std::runtime_error(message);}
static int self_test() {
    Tensor4D v(7,7,7,1);v.at(3,3,3,0)=1.f;
    diffuse(v,.15f);
    double sum=std::accumulate(v.data.begin(),v.data.end(),0.0);
    require(std::abs(sum-1.0)<1e-5,"diffusion mass conservation");
    require(v.at(3,3,3,0)>0.f&&v.at(3,3,3,0)<1.f,"diffusion spreads impulse");
    Tensor4D flat(4,4,4,1,.37f);diffuse(flat,.15f);
    for(float q:flat.data)require(std::abs(q-.37f)<1e-6f,"constant-field invariant");
    bool rejected=false;try{diffuse(flat,.20f);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unstable time step rejected");
    // One-hot transpose-kernel maps (0,0,0) to the matching output voxel.
    std::mt19937 kernel_seed(77);
    Deconv3D tr(1,1,kernel_seed);
    std::fill(tr.w.begin(),tr.w.end(),0.f);
    tr.w[(1*3+1)*3+1]=1.f;
    Tensor4D tr_in(2,2,2,1),tr_out;
    tr_in.at(0,0,0,0)=1.f;
    tr.forward(tr_in,tr_out);
    require(tr_out.at(0,0,0,0)==1.f && tr_out.at(1,0,0,0)==0.f,"transpose-conv kernel coordinates");
    PermeationVAE a(17),b(17);
    Tensor4D input(16,16,16,1,.5f),oa,ob;
    a.step(input,oa,.15f,.13f);b.step(input,ob,.15f,.13f);
    require(oa.d==16&&oa.h==16&&oa.w==16&&oa.c==1,"output shape");
    require(oa.data==ob.data,"single-engine seeded determinism");
    for(float q:oa.data)require(q>=0.f&&q<=1.f&&std::isfinite(q),"bounded finite output");
    omp_set_num_threads(1);
    MegaSwarm s1(2,112);s1.advance();auto m1=s1.metrics();
    omp_set_num_threads(std::min(4,omp_get_max_threads()));
    MegaSwarm s2(2,112);s2.advance();auto m2=s2.metrics();
    require(m1.checksum==m2.checksum,"thread-count checksum parity");
    require(m1.mse>=0&&m1.mse<=1,"finite bounded target MSE");
    std::cout<<"PASS diffusion-mass constant-field stability-seal shape bounded-output seeded-replay thread-parity\n";
    std::cout<<"GOLDEN checksum=0x"<<std::hex<<m1.checksum<<std::dec<<" mse="<<std::fixed<<std::setprecision(8)<<m1.mse<<"\n";
    return 0;
}
static int parse_int(const char* s,int low,int high) {
    std::string v=s;size_t end=0;long n=std::stol(v,&end,10);
    if(end!=v.size()||n<low||n>high)throw std::invalid_argument("argument out of range: "+v);
    return static_cast<int>(n);
}
int main(int argc,char** argv) {
    try {
        int side=2,steps=5,threads=std::min(4,omp_get_max_threads());
        uint32_t seed=42;bool test=false;
        for(int i=1;i<argc;++i) {
            std::string s=argv[i];
            if(s=="--self-test")test=true;
            else if((s=="--swarm"||s=="--steps"||s=="--threads"||s=="--seed")&&i+1<argc) {
                int n=parse_int(argv[++i],s=="--seed"?0:1,s=="--swarm"?8:s=="--steps"?150:s=="--threads"?256:1000000);
                if(s=="--swarm")side=n;else if(s=="--steps")steps=n;else if(s=="--threads")threads=n;else seed=static_cast<uint32_t>(n);
            } else throw std::invalid_argument("usage: [--self-test] [--swarm 1..8] [--steps 1..150] [--threads 1..256] [--seed 0..1000000]");
        }
        if(test)return self_test();
        omp_set_num_threads(threads);
        MegaSwarm swarm(side,seed);
        std::cout<<"JX3D research reference | untrained stochastic VAE-style CPU model | target blend is NOT watermark security\n";
        std::cout<<"requested_virtual_domain=1000000^3 (symbolic only) actual_nodes="<<swarm.count()
                 <<" voxels_per_node=4096 allocated_state_voxels="<<static_cast<uint64_t>(swarm.count())*4096<<"\n";
        for(int i=0;i<steps;++i) {
            auto begin=std::chrono::steady_clock::now();
            swarm.advance();
            auto m=swarm.metrics();
            double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            std::cout<<"TRACE step="<<swarm.step()<<" target_mse="<<std::fixed<<std::setprecision(8)<<m.mse
                     <<" mean="<<m.mean<<" min="<<m.minimum<<" max="<<m.maximum
                     <<" fnv64=0x"<<std::hex<<m.checksum<<std::dec
                     <<" elapsed_ms="<<std::fixed<<std::setprecision(3)<<elapsed<<"\n";
        }
        std::cout<<"DONE: finite-volume proxy execution completed; no model training, security lock, or global convergence certified\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}
}
