// JX64 AVX2/FMA - deterministic, boundary-safe CPU reference and optimized engines
// SPDX-License-Identifier: MIT
// Build with CMake or: g++ -std=c++20 -O3 -mavx2 -mfma -fno-fast-math jx64_avx2_fma.cpp -o jx64avx2
#include <immintrin.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

using u64=std::uint64_t;
constexpr int W=64,H=64,D=64,N=W*H*D, PLANE=W*H;
static_assert((N&(N-1))==0);

template <typename T> struct AlignedArray {
    T* p=nullptr;
    explicit AlignedArray(std::size_t n=N):p(static_cast<T*>(_mm_malloc(sizeof(T)*n,64))) {
        if (!p) throw std::bad_alloc{};
        std::fill_n(p,n,T{});
    }
    ~AlignedArray(){_mm_free(p);}
    AlignedArray(const AlignedArray&)=delete;
    AlignedArray& operator=(const AlignedArray&)=delete;
    AlignedArray(AlignedArray&&)=delete;
    AlignedArray& operator=(AlignedArray&&)=delete;
    T* data() {return p;}
    const T* data() const {return p;}
    T& operator[](std::size_t i){return p[i];}
    const T& operator[](std::size_t i)const{return p[i];}
    void fill(T v){std::fill_n(p,N,v);}
};

inline int flat(int x,int y,int z){return x+W*(y+H*z);}
inline int clipped(int v,int high){return std::min(high,std::max(0,v));}
inline float activation_scalar(float x){
    // Deliberately rational tanh *approximation*, not exact std::tanh.
    if (x>=3.0f) return 1.0f;
    if (x<=-3.0f) return -1.0f;
    const float q=x*x;
    return x*(27.0f+q)/(27.0f+9.0f*q);
}
inline __m256 activation_avx(__m256 x){
    const __m256 one=_mm256_set1_ps(1), mn=_mm256_set1_ps(-1);
    const __m256 q=_mm256_mul_ps(x,x);
    const __m256 num=_mm256_mul_ps(x,_mm256_add_ps(_mm256_set1_ps(27),q));
    const __m256 den=_mm256_fmadd_ps(_mm256_set1_ps(9),q,_mm256_set1_ps(27));
    __m256 out=_mm256_div_ps(num,den);
    out=_mm256_blendv_ps(out,one,_mm256_cmp_ps(x,_mm256_set1_ps(3),_CMP_GE_OQ));
    return _mm256_blendv_ps(out,mn,_mm256_cmp_ps(x,_mm256_set1_ps(-3),_CMP_LE_OQ));
}

struct Metrics {double mse=0, mean_abs=0, sum1024=0, shannon_bits=0;u64 input_hash=0,output_hash=0;};
inline u64 hash64(const float* values){ // FNV-1a over IEEE-754 bit patterns
    u64 h=14695981039346656037ULL;
    const auto* v=reinterpret_cast<const std::uint8_t*>(values);
    for(std::size_t i=0;i<std::size_t(N)*sizeof(float);i++) {h^=v[i];h*=1099511628211ULL;}
    return h;
}

struct Engine {
    AlignedArray<float> original, decoded, encoded, permuted, feedback_field, scratch, scratch2;
    float gain=.5f;
    Engine()=default;
    void seed(u64 state){
        for(int i=0;i<N;i++){
            state^=state<<13;state^=state>>7;state^=state<<17;
            original[i]=float(state&0xffffu)/32768.0f-1.0f;
        }
        std::memcpy(decoded.data(),original.data(),N*sizeof(float));
        encoded.fill(0);permuted.fill(0);feedback_field.fill(0);scratch.fill(0);scratch2.fill(0);
    }
    void encode(bool avx){
        if(!avx){for(int i=0;i<N;i++)encoded[i]=activation_scalar(original[i]*2.5f+.5f);return;}
        const __m256 w=_mm256_set1_ps(2.5f),b=_mm256_set1_ps(.5f);
        for(int i=0;i<N;i+=8){
            __m256 v=_mm256_load_ps(original.data()+i);
            _mm256_store_ps(encoded.data()+i,activation_avx(_mm256_fmadd_ps(v,w,b)));
        }
    }
    void permute(std::uint32_t stride,bool avx){
        const std::uint32_t s=stride|1u;
        if(!avx){for(int i=0;i<N;i++)permuted[i]=encoded[(std::uint32_t(u64(i)*s))&(N-1)];return;}
        alignas(32) std::int32_t positions[8];
        for(int i=0;i<N;i+=8){
            for(int j=0;j<8;j++)positions[j]=std::int32_t((u64(i+j)*s)&(N-1));
            __m256i ix=_mm256_load_si256(reinterpret_cast<const __m256i*>(positions));
            __m256 v=_mm256_i32gather_ps(encoded.data(),ix,4);
            _mm256_store_ps(permuted.data()+i,v);
        }
    }
    void feedback(int iterations,bool avx){
        if(iterations<0||iterations>128)throw std::invalid_argument("iterations must be 0..128");
        feedback_field.fill(0);
        if(!avx){
            for(int t=0;t<iterations;t++)
                for(int i=0;i<N;i++)feedback_field[i]=activation_scalar(permuted[i]+gain*feedback_field[i]);
            return;
        }
        const __m256 g=_mm256_set1_ps(gain);
        for(int t=0;t<iterations;t++)for(int i=0;i<N;i+=8){
            __m256 mem=_mm256_load_ps(feedback_field.data()+i);
            __m256 value=_mm256_load_ps(permuted.data()+i);
            _mm256_store_ps(feedback_field.data()+i,activation_avx(_mm256_fmadd_ps(g,mem,value)));
        }
    }
    // 6-neighbor diffusion: boundary replication (clamped coordinates).
    void diffuse(bool avx){
        const float* src=feedback_field.data();float* dst=permuted.data();
        auto point=[&](int x,int y,int z){
            const int xp=std::min(x+1,W-1),xm=std::max(x-1,0);
            const int yp=std::min(y+1,H-1),ym=std::max(y-1,0);
            const int zp=std::min(z+1,D-1),zm=std::max(z-1,0);
            const int c=flat(x,y,z);
            dst[c]=.4f*src[c]+.1f*(src[flat(xp,y,z)]+src[flat(xm,y,z)] +
               src[flat(x,yp,z)]+src[flat(x,ym,z)] +src[flat(x,y,zp)]+src[flat(x,y,zm)]);
        };
        for(int z=0;z<D;z++)for(int y=0;y<H;y++){
            if(!avx){for(int x=0;x<W;x++)point(x,y,z);continue;}
            point(0,y,z);
            const int yp=std::min(y+1,H-1),ym=std::max(y-1,0);
            const int zp=std::min(z+1,D-1),zm=std::max(z-1,0);
            const __m256 w0=_mm256_set1_ps(.4f),w1=_mm256_set1_ps(.1f);
            for(int x=1;x<=56;x+=8){ // 1..56 only, 7 safe SIMD blocks
                int c=flat(x,y,z);
                const __m256 vc=_mm256_loadu_ps(src+c);
                const __m256 xp=_mm256_loadu_ps(src+c+1),xm=_mm256_loadu_ps(src+c-1);
                const __m256 yplus=_mm256_loadu_ps(src+flat(x,yp,z));
                const __m256 yminus=_mm256_loadu_ps(src+flat(x,ym,z));
                const __m256 zplus=_mm256_loadu_ps(src+flat(x,y,zp));
                const __m256 zminus=_mm256_loadu_ps(src+flat(x,y,zm));
                const __m256 n=_mm256_add_ps(_mm256_add_ps(xp,xm),
                  _mm256_add_ps(_mm256_add_ps(yplus,yminus),_mm256_add_ps(zplus,zminus)));
                _mm256_storeu_ps(dst+c,_mm256_add_ps(_mm256_mul_ps(w0,vc),_mm256_mul_ps(w1,n)));
            }
            for(int x=57;x<W;x++)point(x,y,z);
        }
    }
    void decode27(bool avx){
        const float* src=permuted.data();float* dst=decoded.data();
        const __m256 inv27=_mm256_set1_ps(1.0f/27.0f);
        for(int z=0;z<D;z++)for(int y=0;y<H;y++){
            if(avx){
                for(int x=1;x<=56;x+=8){
                    __m256 s=_mm256_setzero_ps();
                    for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++){
                        const int zz=clipped(z+dz,D-1),yy=clipped(y+dy,H-1);
                        const int c=flat(x,yy,zz);
                        s=_mm256_add_ps(s,_mm256_loadu_ps(src+c-1));
                        s=_mm256_add_ps(s,_mm256_loadu_ps(src+c));
                        s=_mm256_add_ps(s,_mm256_loadu_ps(src+c+1));
                    }
                    _mm256_storeu_ps(dst+flat(x,y,z),activation_avx(_mm256_mul_ps(s,inv27)));
                }
            }
            for(int x=0;x<W;x++){
                if(avx&&x>=1&&x<=56)continue;
                float s=0;
                for(int dz=-1;dz<=1;dz++)for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)
                    s+=src[flat(clipped(x+dx,W-1),clipped(y+dy,H-1),clipped(z+dz,D-1))];
                dst[flat(x,y,z)]=activation_scalar(s/27.0f);
            }
        }
    }
    void decode_separable(){
        // Three 1D mean passes, clamp boundary, then one activation. Math-equivalent to 27-point mean,
        // not bitwise identical due to floating-point addition/rounding order.
        for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
            scratch[flat(x,y,z)]=(permuted[flat(clipped(x-1,W-1),y,z)] +
              permuted[flat(x,y,z)] +permuted[flat(clipped(x+1,W-1),y,z)])*(1.0f/3.0f);
        }
        for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
            scratch2[flat(x,y,z)]=(scratch[flat(x,clipped(y-1,H-1),z)]+
              scratch[flat(x,y,z)]+scratch[flat(x,clipped(y+1,H-1),z)])*(1.0f/3.0f);
        }
        for(int z=0;z<D;z++)for(int y=0;y<H;y++)for(int x=0;x<W;x++){
            const float mean=(scratch2[flat(x,y,clipped(z-1,D-1))]+scratch2[flat(x,y,z)]+
              scratch2[flat(x,y,clipped(z+1,D-1))])*(1.0f/3.0f);
            decoded[flat(x,y,z)]=activation_scalar(mean);
        }
    }
    void execute(u64 state=0xDEADBEEF12345678ULL,std::uint32_t stride=73,int iterations=8,
                 const std::string& mode="simd27"){
        if(mode!="scalar27"&&mode!="simd27"&&mode!="separable")throw std::invalid_argument("mode");
        const bool avx=mode!="scalar27";
        seed(state);encode(avx);permute(stride,avx);feedback(iterations,avx);diffuse(avx);
        if(mode=="separable")decode_separable();else decode27(avx);
    }
    Metrics metrics()const{
        Metrics m;std::array<std::uint32_t,256> bins{};
        for(int i=0;i<N;i++){
            const double e=double(original[i])-double(decoded[i]);
            m.mse+=e*e;m.mean_abs+=std::abs(e);
            if(i<1024)m.sum1024+=decoded[i];
            const float q=std::max(-1.0f,std::min(1.0f,permuted[i]));
            const int bin=clipped(int((q+1.0f)*127.999f),255);
            bins[bin]++;
        }
        m.mse/=N;m.mean_abs/=N;
        for(const auto cnt:bins)if(cnt){double p=double(cnt)/N;m.shannon_bits-=p*std::log2(p);}
        m.input_hash=hash64(original.data());m.output_hash=hash64(decoded.data());
        return m;
    }
};

int main(int argc,char**argv){
    try{
        std::string mode=argc>1?argv[1]:"simd27";
        const int repeats=argc>2?std::stoi(argv[2]):5;
        if(repeats<1||repeats>1000)throw std::invalid_argument("repeats 1..1000");
        Engine e;std::vector<double> ms;ms.reserve(repeats);
        for(int t=0;t<repeats;t++){
            const auto begin=std::chrono::steady_clock::now();
            e.execute(0xDEADBEEF12345678ULL,73,8,mode);
            const auto end=std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double,std::milli>(end-begin).count());
        }
        auto m=e.metrics();std::sort(ms.begin(),ms.end());
        std::cout<<std::setprecision(12)<<"mode="<<mode<<" median_ms="<<ms[ms.size()/2]
          <<" mse="<<m.mse<<" mae="<<m.mean_abs<<" entropy_bits="<<m.shannon_bits
          <<" checksum1024="<<m.sum1024<<" input_fnv64="<<std::hex<<m.input_hash
          <<" output_fnv64="<<m.output_hash<<std::dec<<'\n';
    }catch(const std::exception&ex){std::cerr<<"ERROR: "<<ex.what()<<'\n';return 1;}
    return 0;
}
