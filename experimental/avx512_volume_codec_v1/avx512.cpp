// Copyright (c) 2026 Dr Matladi Maxwell Moagi. See repository LICENSE.
// No global -mavx512f: target functions are entered only after runtime dispatch.
#include "backend.hpp"
#include <stdexcept>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(JX_DISABLE_AVX512)
#define JX_HAS_AVX512 1
#include <immintrin.h>
#define JX_AVX512 __attribute__((target("avx512f"),noinline))
#else
#define JX_HAS_AVX512 0
#endif
namespace jx::volume {
bool avx512_compiled() noexcept {return JX_HAS_AVX512!=0;}
namespace detail {
#if JX_HAS_AVX512
JX_AVX512 void pool_avx512(const float* input,float* output,std::size_t d,std::size_t h,std::size_t w){
    const __m512i even=_mm512_setr_epi32(0,2,4,6,8,10,12,14,0,0,0,0,0,0,0,0);
    const __m512i odd=_mm512_setr_epi32(1,3,5,7,9,11,13,15,0,0,0,0,0,0,0,0);
    for(std::size_t z=0;z<d/2;++z)for(std::size_t y=0;y<h/2;++y){
        std::size_t x=0;
        for(;x+8<=w/2;x+=8){
            // Eight independent 2x2x2 blocks per ZMM double accumulator.
            __m512d sum=_mm512_setzero_pd();
            for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy){
                const __m512 row=_mm512_loadu_ps(input+(2*z+dz)*h*w+(2*y+dy)*w+2*x);
                const __m256 a=_mm512_castps512_ps256(_mm512_permutexvar_ps(even,row));
                const __m256 b=_mm512_castps512_ps256(_mm512_permutexvar_ps(odd,row));
                sum=_mm512_add_pd(sum,_mm512_cvtps_pd(a));
                sum=_mm512_add_pd(sum,_mm512_cvtps_pd(b));
            }
            _mm256_storeu_ps(output+(z*(h/2)+y)*(w/2)+x,_mm512_cvtpd_ps(_mm512_mul_pd(sum,_mm512_set1_pd(0.125))));
        }
        for(;x<w/2;++x){double sum=0;
            for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy)for(std::size_t dx=0;dx<2;++dx)
                sum+=static_cast<double>(input[(2*z+dz)*h*w+(2*y+dy)*w+2*x+dx]);
            output[(z*(h/2)+y)*(w/2)+x]=static_cast<float>(sum*0.125);
        }
    }
}
JX_AVX512 void expand_avx512(const float* means,float* output,std::size_t d,std::size_t h,std::size_t w){
    const __m512i lo=_mm512_setr_epi32(0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7);
    const __m512i hi=_mm512_setr_epi32(8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15);
    for(std::size_t z=0;z<d/2;++z)for(std::size_t y=0;y<h/2;++y){std::size_t x=0;
        for(;x+16<=w/2;x+=16){const __m512 values=_mm512_loadu_ps(means+(z*(h/2)+y)*(w/2)+x);
            const __m512 a=_mm512_permutexvar_ps(lo,values),b=_mm512_permutexvar_ps(hi,values);
            for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy){float* row=output+(2*z+dz)*h*w+(2*y+dy)*w+2*x;
                _mm512_storeu_ps(row,a);_mm512_storeu_ps(row+16,b);}
        }
        for(;x<w/2;++x){const float value=means[(z*(h/2)+y)*(w/2)+x];
            for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy)for(std::size_t dx=0;dx<2;++dx)
                output[(2*z+dz)*h*w+(2*y+dy)*w+2*x+dx]=value;
        }
    }
}
JX_AVX512 double mse_avx512(const float* reference,const float* output,std::size_t count){
    __m512d sum=_mm512_setzero_pd();std::size_t i=0;
    for(;i+8<=count;i+=8){const __m512d a=_mm512_cvtps_pd(_mm256_loadu_ps(reference+i));
        const __m512d b=_mm512_cvtps_pd(_mm256_loadu_ps(output+i));const __m512d difference=_mm512_sub_pd(a,b);
        sum=_mm512_fmadd_pd(difference,difference,sum);
    }
    alignas(64) double lanes[8];_mm512_storeu_pd(lanes,sum);double total=0;for(double lane:lanes)total+=lane;
    for(;i<count;++i){const double difference=static_cast<double>(reference[i])-output[i];total+=difference*difference;}
    return count?total/static_cast<double>(count):0;
}
#else
void pool_avx512(const float*,float*,std::size_t,std::size_t,std::size_t){throw std::runtime_error("AVX-512 backend not compiled");}
void expand_avx512(const float*,float*,std::size_t,std::size_t,std::size_t){throw std::runtime_error("AVX-512 backend not compiled");}
double mse_avx512(const float*,const float*,std::size_t){throw std::runtime_error("AVX-512 backend not compiled");}
#endif
}
}
