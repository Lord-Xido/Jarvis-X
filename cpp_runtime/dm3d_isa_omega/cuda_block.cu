// Optional NVIDIA CUDA parity backend for v1 2x2x2 block-average encoding.
// This file is never compiled unless CMake finds a CUDA compiler.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void dm3d_avg8_kernel(const unsigned char* __restrict__ src,
                                  unsigned char* __restrict__ dst){
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=4096)return;
    unsigned x=i%16,y=(i/16)%16,z=i/256;
    unsigned p=(2*z*32+2*y)*32+2*x;
    unsigned s=unsigned(src[p])+src[p+1]+src[p+32]+src[p+33]
              +src[p+1024]+src[p+1025]+src[p+1056]+src[p+1057];
    dst[i]=static_cast<unsigned char>((s+4)/8);
}
extern "C" int dm3d_cuda_available(){
    int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0;
}
extern "C" int dm3d_cuda_block_encode(const std::uint8_t* host_in,std::uint8_t* host_out){
    if(!dm3d_cuda_available()||!host_in||!host_out)return 1;
    unsigned char* device_in=nullptr;unsigned char* device_out=nullptr;
    int status=0;
    if(cudaMalloc(reinterpret_cast<void**>(&device_in),32768)!=cudaSuccess)return 2;
    if(cudaMalloc(reinterpret_cast<void**>(&device_out),4096)!=cudaSuccess){cudaFree(device_in);return 3;}
    if(cudaMemcpy(device_in,host_in,32768,cudaMemcpyHostToDevice)!=cudaSuccess)status=4;
    if(!status){dm3d_avg8_kernel<<<16,256>>>(device_in,device_out);
        if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess)status=5;}
    if(!status&&cudaMemcpy(host_out,device_out,4096,cudaMemcpyDeviceToHost)!=cudaSuccess)status=6;
    cudaFree(device_in);cudaFree(device_out);
    return status;
}
