#include "engine.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace jx64 {
AutoEncoding3DEngine::AutoEncoding3DEngine():R0(N,0),R1(N,0),R2(N,0),R3(N,0),R4(N,0),seed_field(N,0),scratch(N,0){}
std::size_t AutoEncoding3DEngine::idx(int x,int y,int z){return static_cast<std::size_t>(x+W*y+W*H*z);}
float AutoEncoding3DEngine::activate(float x){
    if(x>3.0f)return 1.0f;
    if(x< -3.0f)return -1.0f;
    const float x2=x*x;
    return x*(27.0f+x2)/(27.0f+9.0f*x2);
}
std::uint32_t AutoEncoding3DEngine::normalized_stride(std::uint32_t stride){return (stride|1u)&MASK;}
std::uint32_t AutoEncoding3DEngine::inverse_stride(std::uint32_t stride){
    const std::uint32_t s=normalized_stride(stride);
    // Newton inversion in Z/(2^32), then mask to 2^18.
    std::uint32_t x=s;
    for(int k=0;k<5;k++)x*=2u-s*x;
    return x&MASK;
}
std::uint64_t AutoEncoding3DEngine::hash_field(const Field& v){
    // FNV-1a over IEEE-754 float bits written in canonical little-endian order.
    std::uint64_t h=14695981039346656037ULL;
    for(float f:v){
        std::uint32_t u=std::bit_cast<std::uint32_t>(f);
        for(int i=0;i<4;i++){h^=(u>>(i*8))&0xffu;h*=1099511628211ULL;}
    }
    return h;
}
void AutoEncoding3DEngine::generate_seed(std::uint64_t seed){
    for(int i=0;i<N;i++){
       seed^=seed<<13; seed^=seed>>7; seed^=seed<<17;
       seed_field[i]=static_cast<float>(seed&0xffffULL)/32768.0f-1.0f;
    }
    R0=seed_field;
}
void AutoEncoding3DEngine::encode(){
    for(int i=0;i<N;i++) R1[i]=activate(R0[i]*2.5f+0.5f);
}
void AutoEncoding3DEngine::permute(std::uint32_t stride){
    const auto s=normalized_stride(stride);
    for(std::uint32_t i=0;i<static_cast<std::uint32_t>(N);i++) R2[i]=R1[(std::uint64_t(i)*s)&MASK];
}
void AutoEncoding3DEngine::undo_permute(std::uint32_t stride,const Field& in, Field& out)const{
    if(in.size()!=N)throw std::invalid_argument("inverse permutation: source size");
    out.resize(N);
    const auto inv=inverse_stride(stride);
    for(std::uint32_t j=0;j<static_cast<std::uint32_t>(N);j++)out[j]=in[(std::uint64_t(j)*inv)&MASK];
}
void AutoEncoding3DEngine::feedback(int iterations,float gain){
    if(iterations<0||iterations>1000||!std::isfinite(gain)||gain<0.0f||gain>1.0f)
        throw std::invalid_argument("invalid feedback configuration");
    std::fill(R3.begin(),R3.end(),0.0f);
    if(iterations==0){R3=R2;return;}
    for(int t=0;t<iterations;t++){
       for(int i=0;i<N;i++) R3[i]=activate(R2[i]+gain*R3[i]);
    }
}
static inline int clamp64(int p){return p<0?0:(p>63?63:p);}
void AutoEncoding3DEngine::propagate(){
    // Boundary is well defined and written on every invocation (replicate/clamp).
    for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
       const auto c=idx(x,y,z);
       const float s=R3[idx(clamp64(x+1),y,z)]+R3[idx(clamp64(x-1),y,z)]
                    +R3[idx(x,clamp64(y+1),z)]+R3[idx(x,clamp64(y-1),z)]
                    +R3[idx(x,y,clamp64(z+1))]+R3[idx(x,y,clamp64(z-1))];
       R4[c]=0.40f*R3[c]+0.10f*s;
    }
}
void AutoEncoding3DEngine::decode_reference(Field& out) const{
    out.resize(N);
    for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
      float sum=0.0f;
      for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)
        sum+=R4[idx(clamp64(x+dx),clamp64(y+dy),clamp64(z+dz))];
      out[idx(x,y,z)]=activate(sum/27.0f);
    }
}
void AutoEncoding3DEngine::decode_separable(Field& out){
    // The 27-point uniform kernel factors as (1,1,1) in each axis.
    // Floating-point association differs from the literal 27-term implementation.
    out.resize(N);
    for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
      scratch[idx(x,y,z)]=(R4[idx(clamp64(x-1),y,z)]+R4[idx(x,y,z)]+R4[idx(clamp64(x+1),y,z)])/3.0f;
    }
    for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
      R2[idx(x,y,z)]=(scratch[idx(x,clamp64(y-1),z)]+scratch[idx(x,y,z)]+scratch[idx(x,clamp64(y+1),z)])/3.0f;
    }
    for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
      const float sum=R2[idx(x,y,clamp64(z-1))]+R2[idx(x,y,z)]+R2[idx(x,y,clamp64(z+1))];
      out[idx(x,y,z)]=activate(sum/3.0f);
    }
}
Metrics AutoEncoding3DEngine::execute(Config cfg,const std::string& decoder){
    generate_seed(cfg.seed);encode();permute(cfg.stride);feedback(cfg.iterations,cfg.feedback_gain);propagate();
    if(decoder=="reference") decode_reference(R0);
    else if(decoder=="separable")decode_separable(R0);
    else throw std::invalid_argument("decoder must be reference or separable");
    Metrics m{};m.decoder=decoder;
    for(int i=0;i<N;i++){
       const double es=double(R0[i])-double(seed_field[i]);
       const double ee=double(R0[i])-double(R1[i]);
       m.mse_to_seed+=es*es;
       m.mse_to_encoded+=ee*ee;
    }
    m.mse_to_seed/=N;m.mse_to_encoded/=N;
    for(int i=0;i<32;i++)m.checksum32+=R0[i];
    m.input_hash=hash_field(seed_field);m.output_hash=hash_field(R0);
    return m;
}
}
