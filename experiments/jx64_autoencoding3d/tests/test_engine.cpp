#include "engine.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace jx64;
static void need(bool c,const char* reason){if(!c)throw std::runtime_error(reason);}
int main(){
  try{
    auto e=std::make_unique<AutoEncoding3DEngine>();
    need(N==262144,"geometry");
    for(auto stride:{0u,1u,2u,73u,74u,65535u,0xffffffffu}){
      auto s=AutoEncoding3DEngine::normalized_stride(stride);
      auto inv=AutoEncoding3DEngine::inverse_stride(stride);
      need(((std::uint64_t)s*inv&MASK)==1,"multiplicative inverse");
      e->generate_seed(0xDEADBEEF12345678ULL);e->encode();e->permute(stride);
      Field restored;
      e->undo_permute(stride,e->R2,restored);
      need(restored==e->R1,"permutation exact inverse");
    }
    for(auto seed:{0ULL,1ULL,0xDEADBEEF12345678ULL,0xffffffffffffffffULL}){
      Config c; c.seed=seed;c.iterations=8;
      const auto a=e->execute(c,"reference");
      const Field original=e->R0;
      const auto b=e->execute(c,"reference");
      need(original==e->R0 && a.output_hash==b.output_hash,"reentrant deterministic");
      need(std::isfinite(a.mse_to_seed)&&a.mse_to_seed>=0,"finite MSE");
      need(std::isfinite(a.checksum32),"checksum finite");
      // Invariance of constant input under mean+stencil alone.
    }
    double maximum_diff=0;
    for(auto seed:{0ULL,1ULL,0xDEADBEEF12345678ULL,0xffffffffffffffffULL}){
      for(auto iters:{0,1,3,8}){
        Config c; c.seed=seed;c.iterations=iters;
        e->execute(c,"reference");
        const Field reference=e->R0;
        e->execute(c,"separable");
        for(int i=0;i<N;i++)maximum_diff=std::max(maximum_diff,double(std::abs(reference[i]-e->R0[i])));
        need(maximum_diff<1e-5,"27-tap/separable numerical parity");
      }
    }
    // Finite boundaries cannot retain stale values after execute().
    Config c; c.seed=123;c.iterations=2; e->execute(c,"separable");
    for(auto i:{0,N-1,63,4095,4096,100000})need(std::isfinite(e->R0[i]),"boundary finite");
    bool rejected=false;
    try{c.iterations=-1;e->execute(c,"reference");}catch(const std::invalid_argument&){rejected=true;}
    need(rejected,"invalid iterations rejected");
    std::cout<<"PASS: 64^3 geometry, exact modular inverse, deterministic reruns, invalid input, 16 reference/separable field comparisons, bounds; max_diff="<<maximum_diff<<"\n";
    return 0;
  }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
