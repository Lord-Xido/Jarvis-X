#pragma once
// JARVIS-X Omega3: fused 8K GCL-ANN reference math. SDR only; no training or 3D voxels.
#include <cstdint>
#include <cmath>
#ifdef __CUDACC__
#define GCL_HD __host__ __device__
#else
#define GCL_HD
#endif
namespace gcl {
constexpr int INPUT=8, L1=64, L2=32, L3=16, OUTPUT=6;
constexpr int O1=0, O2=64*8+64, O3=O2+32*64+32, O4=O3+16*32+16;
constexpr int PARAMS=O4+6*16+6; // 3286
constexpr int BLOCK=16, HALO=1, TILE=18;
struct RGB {float r,g,b;};
struct Sample {RGB rgb; float y;}; // 16-byte shared-memory record
GCL_HD inline float clamp01(float x) {return fminf(1.f,fmaxf(0.f,x));}
GCL_HD inline int clampi(int x,int lo,int hi){return x<lo?lo:(x>hi?hi:x);}
GCL_HD inline RGB plus(RGB a,RGB b){return {a.r+b.r,a.g+b.g,a.b+b.b};}
GCL_HD inline RGB scale(RGB a,float k){return {a.r*k,a.g*k,a.b*k};}
GCL_HD inline float lum(RGB c){return .2126f*c.r+.7152f*c.g+.0722f*c.b;}
GCL_HD inline Sample generate(int px,int py,int width,int height,float t){
    const int x=clampi(px,0,width-1), y=clampi(py,0,height-1);
    const float u=float(x)/float(width-1), v=float(y)/float(height-1);
    RGB c={.5f+.5f*sinf(u*20.f+t),.5f+.5f*sinf(v*20.f-t),
           .5f+.5f*sinf((u+v)*15.f)};
    if (((x/120+y/120)&1)==0) c=scale(c,.7f);
    return {c,lum(c)};
}
// Sobel gradients and 3x3 Gaussian-smoothed source, s row-major [3][3].
GCL_HD inline void neighborhood(const Sample s[9],float &edge,float &contrast,RGB &blur){
    const float gx=-s[0].y+s[2].y-2.f*s[3].y+2.f*s[5].y-s[6].y+s[8].y;
    const float gy=-s[0].y-2.f*s[1].y-s[2].y+s[6].y+2.f*s[7].y+s[8].y;
    edge=clamp01(sqrtf(gx*gx+gy*gy)/5.65685425f);
    RGB avg={0,0,0}; float lsum=0;
    const int k[9]={1,2,1,2,4,2,1,2,1};
    for(int i=0;i<9;++i){avg=plus(avg,scale(s[i].rgb,float(k[i])));lsum+=s[i].y*float(k[i]);}
    blur=scale(avg,1.f/16.f);
    contrast=clamp01(fabsf(s[4].y-lsum/16.f));
}
GCL_HD inline float relu(float x){return x>0.f?x:0.f;}
GCL_HD inline float sigmoid(float x){return 1.f/(1.f+expf(-x));}
// Host accessor reads FP32 weights; CUDA accessor reads FP16 constant-memory weights.
template<class Accessor> GCL_HD inline void infer(Accessor get,const float f[8],float ctrl[6]){
    float h1[64], h2[32], h3[16];
    for(int o=0;o<64;++o){float s=get(O1+64*8+o);for(int i=0;i<8;++i)s+=get(O1+o*8+i)*f[i];h1[o]=relu(s);}
    for(int o=0;o<32;++o){float s=get(O2+32*64+o);for(int i=0;i<64;++i)s+=get(O2+o*64+i)*h1[i];h2[o]=relu(s);}
    for(int o=0;o<16;++o){float s=get(O3+16*32+o);for(int i=0;i<32;++i)s+=get(O3+o*32+i)*h2[i];h3[o]=tanhf(s);}
    for(int o=0;o<6;++o){float s=get(O4+6*16+o);for(int i=0;i<16;++i)s+=get(O4+o*16+i)*h3[i];ctrl[o]=sigmoid(s);}
}
GCL_HD inline float srgb_linear(float x){x=clamp01(x);return x<=.04045f?x/12.92f:powf((x+.055f)/1.055f,2.4f);}
GCL_HD inline float linear_srgb(float x){x=clamp01(x);return x<=.0031308f?12.92f*x:1.055f*powf(x,1.f/2.4f)-.055f;}
GCL_HD inline RGB convert(RGB src,RGB blur,const float ctrl[6],int x,int y){
    // Signed unsharp residual in nonlinear source code-value space; creative SDR look.
    const float sharp=ctrl[3];
    src={clamp01(src.r+sharp*(src.r-blur.r)),clamp01(src.g+sharp*(src.g-blur.g)),clamp01(src.b+sharp*(src.b-blur.b))};
    const float exponent=2.2f/(1.6f+ctrl[4]);
    float r=powf(clamp01(srgb_linear(src.r)*(.4f+1.2f*ctrl[0])),exponent);
    float g=powf(clamp01(srgb_linear(src.g)*(.4f+1.2f*ctrl[1])),exponent);
    float b=powf(clamp01(srgb_linear(src.b)*(.4f+1.2f*ctrl[2])),exponent);
    const float r2=.6274f*r+.3293f*g+.0433f*b;
    const float g2=.0691f*r+.9195f*g+.0114f*b;
    const float b2=.0164f*r+.0880f*g+.8956f*b;
    RGB out={linear_srgb(r2),linear_srgb(g2),linear_srgb(b2)};
    // 4x4 centered Bayer offsets: bounded, spatially ordered, amplitude <= 0.5/1023.
    const int bayer[16]={0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5};
    const float d=(float(bayer[((y&3)<<2)|(x&3)])-7.5f)/(16.f*1023.f)*ctrl[5];
    return {clamp01(out.r+d),clamp01(out.g+d),clamp01(out.b+d)};
}
GCL_HD inline uint32_t q10(float c){return uint32_t(floorf(clamp01(c)*1023.f+.5f));}
GCL_HD inline uint32_t pack(RGB c){return q10(c.r)|(q10(c.g)<<10)|(q10(c.b)<<20)|(3u<<30);}
GCL_HD inline RGB unpack(uint32_t p){return {float(p&1023u)/1023.f,float((p>>10)&1023u)/1023.f,float((p>>20)&1023u)/1023.f};}
} // namespace gcl
