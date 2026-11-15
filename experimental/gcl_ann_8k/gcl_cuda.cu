// Optional CUDA 16x16+halo fused Sobel -> FP16-weights ANN -> SDR RGB10A2 renderer.
#include "gcl_common.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <array>
#include <string>
#include <vector>

__constant__ __half c_net[gcl::PARAMS]; // ~6.4 KiB; broadcast-friendly weights.
struct DeviceAccess {
    __device__ float operator()(int index) const{return __half2float(c_net[index]);}
};
__global__ void fused_8k(int width,int height,float t,uint32_t* __restrict__ out){
    __shared__ gcl::Sample tile[gcl::TILE*gcl::TILE];
    const int tx=threadIdx.x,ty=threadIdx.y,thread=ty*16+tx;
    const int x=blockIdx.x*16+tx,y=blockIdx.y*16+ty;
    for(int n=thread;n<gcl::TILE*gcl::TILE;n+=256){
        const int sx=n%gcl::TILE,sy=n/gcl::TILE;
        tile[n]=gcl::generate(int(blockIdx.x)*16+sx-1,int(blockIdx.y)*16+sy-1,width,height,t);
    }
    __syncthreads();
    if(x>=width||y>=height)return;
    gcl::Sample s[9];
    for(int j=0;j<3;++j)for(int i=0;i<3;++i)s[j*3+i]=tile[(ty+j)*gcl::TILE+tx+i];
    float edge,contrast;gcl::RGB blur;
    gcl::neighborhood(s,edge,contrast,blur);
    const float f[8]={float(x)/float(width-1),float(y)/float(height-1),s[4].rgb.r,s[4].rgb.g,s[4].rgb.b,contrast,edge,t};
    float ctrl[6];gcl::infer(DeviceAccess{},f,ctrl);
    out[size_t(y)*size_t(width)+size_t(x)]=gcl::pack(gcl::convert(s[4].rgb,blur,ctrl,x,y));
}
bool cuda_render(int w,int h,float t,const std::array<float,gcl::PARAMS>& weights,
                 std::vector<uint32_t>& out,std::string& error){
    int count=0;
    auto rc=cudaGetDeviceCount(&count);
    if(rc!=cudaSuccess||count<1){error="no CUDA device";return false;}
    std::array<__half,gcl::PARAMS> half_weights;
    for(int i=0;i<gcl::PARAMS;++i)half_weights[size_t(i)]=__float2half_rn(weights[size_t(i)]);
    rc=cudaMemcpyToSymbol(c_net,half_weights.data(),sizeof(half_weights));
    if(rc!=cudaSuccess){error=cudaGetErrorString(rc);return false;}
    uint32_t* dev=nullptr;
    const size_t bytes=size_t(w)*size_t(h)*sizeof(uint32_t);
    rc=cudaMalloc(reinterpret_cast<void**>(&dev),bytes);
    if(rc!=cudaSuccess){error=cudaGetErrorString(rc);return false;}
    const dim3 blocks((w+15)/16,(h+15)/16),threads(16,16);
    fused_8k<<<blocks,threads>>>(w,h,t,dev);
    rc=cudaGetLastError();
    if(rc==cudaSuccess)rc=cudaDeviceSynchronize();
    if(rc==cudaSuccess){out.resize(size_t(w)*size_t(h));rc=cudaMemcpy(out.data(),dev,bytes,cudaMemcpyDeviceToHost);}
    cudaFree(dev);
    if(rc!=cudaSuccess){error=cudaGetErrorString(rc);return false;}
    return true;
}
