// Copyright (c) 2026 Dr Matladi Maxwell Moagi. See repository LICENSE.
#include "backend.hpp"
#include <stdexcept>

namespace jx::volume {
bool avx512_supported() noexcept {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return avx512_compiled() && __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}
Backend resolve(Backend requested){
    if(requested==Backend::Auto)return avx512_supported()?Backend::AVX512:Backend::Scalar;
    if(requested==Backend::AVX512 && !avx512_supported())throw std::runtime_error("AVX-512F backend unavailable on this CPU/build");
    if(requested!=Backend::Scalar && requested!=Backend::AVX512)throw std::invalid_argument("unknown backend");
    return requested;
}
const char* backend_name(Backend backend) noexcept {return backend==Backend::AVX512?"avx512f":backend==Backend::Scalar?"scalar":"auto";}
namespace detail {
void pool_scalar(const float* input,float* output,std::size_t d,std::size_t h,std::size_t w){
    for(std::size_t z=0;z<d/2;++z)for(std::size_t y=0;y<h/2;++y)for(std::size_t x=0;x<w/2;++x){
        double sum=0;
        for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy)for(std::size_t dx=0;dx<2;++dx)
            sum+=static_cast<double>(input[(2*z+dz)*h*w+(2*y+dy)*w+2*x+dx]);
        output[(z*(h/2)+y)*(w/2)+x]=static_cast<float>(sum*0.125);
    }
}
void expand_scalar(const float* means,float* output,std::size_t d,std::size_t h,std::size_t w){
    for(std::size_t z=0;z<d/2;++z)for(std::size_t y=0;y<h/2;++y)for(std::size_t x=0;x<w/2;++x){
        const float mean=means[(z*(h/2)+y)*(w/2)+x];
        for(std::size_t dz=0;dz<2;++dz)for(std::size_t dy=0;dy<2;++dy)for(std::size_t dx=0;dx<2;++dx)
            output[(2*z+dz)*h*w+(2*y+dy)*w+2*x+dx]=mean;
    }
}
double mse_scalar(const float* reference,const float* output,std::size_t count){
    double sum=0;for(std::size_t i=0;i<count;++i){const double diff=static_cast<double>(reference[i])-output[i];sum+=diff*diff;}
    return count?sum/static_cast<double>(count):0;
}
}
}
