#include "gcl_common.hpp"
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Weights=std::array<float,gcl::PARAMS>;
// Exact binary16-representable coefficients for CPU/FP16-CUDA comparison.
Weights make_weights(uint32_t seed=42){
    Weights w{};
    uint32_t state=seed;
    auto init=[&](int base,int inputs,int outputs){
        for(int o=0;o<outputs;++o)
            for(int i=0;i<inputs;++i){
                state=1664525u*state+1013904223u;
                w[base+o*inputs+i]=float(int((state>>16)&255u)-128)/1024.f;
            }
        // Biases are zero, deterministic and half-representable.
    };
    init(gcl::O1,8,64);init(gcl::O2,64,32);init(gcl::O3,32,16);init(gcl::O4,16,6);
    return w;
}
struct HostAccess {const Weights* w; float operator()(int i) const{return (*w)[i];}};
std::vector<uint32_t> render_cpu(int w,int h,float t,const Weights& weights){
    if(w<2||h<2||w>7680||h>4320)throw std::invalid_argument("dimensions must be 2..7680 x 2..4320");
    std::vector<uint32_t> out(size_t(w)*size_t(h));
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){
        gcl::Sample s[9];
        for(int j=0;j<3;++j)for(int i=0;i<3;++i)s[j*3+i]=gcl::generate(x+i-1,y+j-1,w,h,t);
        float edge,contrast;gcl::RGB blur;
        gcl::neighborhood(s,edge,contrast,blur);
        const float f[8]={float(x)/float(w-1),float(y)/float(h-1),s[4].rgb.r,s[4].rgb.g,s[4].rgb.b,contrast,edge,t};
        float ctrl[6];gcl::infer(HostAccess{&weights},f,ctrl);
        out[size_t(y)*size_t(w)+size_t(x)]=gcl::pack(gcl::convert(s[4].rgb,blur,ctrl,x,y));
    }
    return out;
}
uint64_t fnv(const std::vector<uint32_t>& v){
    uint64_t h=14695981039346656037ull;
    for(uint32_t p:v)for(int i=0;i<4;++i){h^=(p>>(8*i))&255u;h*=1099511628211ull;}
    return h;
}
#ifdef GCL_HAS_CUDA
bool cuda_render(int w,int h,float t,const Weights& wgt,std::vector<uint32_t>& out,std::string& error);
#endif
void self_test(const Weights& w){
    if(gcl::PARAMS!=3286||gcl::O4!=3184)throw std::runtime_error("ANN parameter layout");
    if(gcl::pack({0.f,1.f,.5f})!=(3u<<30|512u<<20|1023u<<10))throw std::runtime_error("packing layout");
    auto a=render_cpu(19,17,.125f,w),b=render_cpu(19,17,.125f,w);
    if(a!=b||a.size()!=323||a.front()==a.back())throw std::runtime_error("deterministic CPU test");
    const auto c=render_cpu(32,18,.125f,w);
    std::cout<<"SELF_TEST=PASS hash19x17=0x"<<std::hex<<fnv(a)<<" hash32x18=0x"<<fnv(c)<<std::dec<<"\n";
    // Optional GPU parity is separately gated on CUDA hardware; arithmetic may differ a few LSB.
}
int main(int argc,char** argv){
    try{
        int w=64,h=36;float t=.125f;bool test=false,full=false,run_cuda=false;
        std::string path;
        for(int i=1;i<argc;++i){
            std::string a=argv[i];
            if(a=="--self-test")test=true;
            else if(a=="--full")full=true;
            else if(a=="--cuda")run_cuda=true;
            else if(a=="--width"&&i+1<argc)w=std::stoi(argv[++i]);
            else if(a=="--height"&&i+1<argc)h=std::stoi(argv[++i]);
            else if(a=="--output"&&i+1<argc)path=argv[++i];
            else if(a=="--time"&&i+1<argc)t=std::stof(argv[++i]);
            else throw std::invalid_argument("unknown or incomplete option: "+a);
        }
        auto weights=make_weights();
        if(test){self_test(weights);return 0;}
        if(full){w=7680;h=4320;}
        if(w<2||h<2||w>7680||h>4320)throw std::invalid_argument("invalid dimensions");
        std::vector<uint32_t> frame;
        auto start=std::chrono::steady_clock::now();
        if(run_cuda){
#ifdef GCL_HAS_CUDA
            std::string error;
            if(!cuda_render(w,h,t,weights,frame,error))throw std::runtime_error("CUDA: "+error);
#else
            throw std::runtime_error("CUDA backend not built (configure with NVIDIA nvcc)");
#endif
        } else frame=render_cpu(w,h,t,weights);
        const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(!path.empty()){
            std::ofstream os(path,std::ios::binary);
            if(!os)throw std::runtime_error("cannot open output");
            for(uint32_t p:frame)for(int k=0;k<4;++k)os.put(char((p>>(8*k))&255u));
            if(!os)throw std::runtime_error("output write failure");
        }
        std::cout<<"backend="<<(run_cuda?"cuda":"cpu")<<" size="<<w<<"x"<<h<<" pixels="<<frame.size()
                 <<" output_bytes="<<frame.size()*4<<" elapsed_ms="<<std::fixed<<std::setprecision(3)<<elapsed
                 <<" fnv64=0x"<<std::hex<<fnv(frame)<<std::dec<<"\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
