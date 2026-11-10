#define main engine_sample_main
#include "jx64_avx2_fma.cpp"
#undef main
#include <iostream>
#include <limits>

void require(bool x,const char*m){if(!x)throw std::runtime_error(m);}
int main(){
    try {
        Engine scalar,simd,sep;
        require((reinterpret_cast<std::uintptr_t>(scalar.original.data()) % 64)==0,"original not 64-byte aligned");
        require((reinterpret_cast<std::uintptr_t>(scalar.encoded.data()) % 64)==0,"encoded not 64-byte aligned");
        // An odd multiplier is a bijection modulo 2^18: every index appears exactly once.
        for(const std::uint32_t s0 : {0u,1u,73u,256u,65535u}) {
            std::vector<std::uint8_t> seen(N,0);
            const std::uint32_t s=s0|1u;
            for(int i=0;i<N;i++) {
                std::uint32_t dest=std::uint32_t(u64(i)*s)&(N-1);
                require(seen[dest]++==0,"permutation collision");
            }
            for(auto value:seen)require(value==1,"incomplete permutation");
        }
        int tested=0;
        double worst_simd=0,worst_sep=0;
        const std::array<u64,5> seeds={0,1,0xDEADBEEF12345678ULL,0xffffffffffffffffULL,101};
        const std::array<std::uint32_t,5> strides={0,1,73,256,65535};
        const std::array<int,4> iterations={0,1,3,8};
        for(int caseid=0;caseid<20;caseid++){
            const u64 seed=seeds[caseid%5];
            const std::uint32_t stride=strides[(caseid/2)%5];
            const int n=iterations[caseid%4];
            scalar.execute(seed,stride,n,"scalar27");
            simd.execute(seed,stride,n,"simd27");
            sep.execute(seed,stride,n,"separable");
            for(int i=0;i<N;i++){
                require(std::isfinite(scalar.decoded[i])&&std::isfinite(simd.decoded[i])&&std::isfinite(sep.decoded[i]),"nonfinite");
                worst_simd=std::max(worst_simd,std::abs(double(scalar.decoded[i])-simd.decoded[i]));
                worst_sep=std::max(worst_sep,std::abs(double(scalar.decoded[i])-sep.decoded[i]));
                require(scalar.original[i]==simd.original[i]&&scalar.original[i]==sep.original[i],"input changed");
            }
            require(worst_simd<3e-6,"scalar vs SIMD diverged");
            require(worst_sep<3e-6,"scalar vs separable diverged");
            // Ensure initial seeded state remains intact and distinct from decoder output
            require(scalar.metrics().input_hash==simd.metrics().input_hash,"input hash differs");
            auto before=simd.metrics().output_hash;
            simd.execute(seed,stride,n,"simd27");
            require(before==simd.metrics().output_hash,"repeat run not deterministic");
            tested++;
        }
        Engine impulses;
        impulses.seed(12345);
        impulses.permuted.fill(0);
        impulses.permuted[flat(0,0,0)]=1;
        impulses.decode27(true);
        require(impulses.decoded[flat(0,0,0)]>0,"origin decoding failed");
        require(impulses.decoded[flat(63,63,63)]==0,"unexpected distant activation");
        Engine uniform;
        uniform.feedback_field.fill(.75f);
        uniform.diffuse(true);
        for(int i=0;i<N;i++)require(std::abs(uniform.permuted[i]-.75f)<1e-6,"constant field not preserved at boundary");
        uniform.decode27(true);
        for(int i=0;i<N;i++)require(std::abs(uniform.decoded[i]-activation_scalar(.75f))<1e-6,"uniform decode boundary");
        std::cout<<"PASS tested_cases="<<tested<<" compared_voxels="<<tested*N
          <<" max_abs_simd="<<worst_simd<<" max_abs_separable="<<worst_sep
          <<" boundary_uniform=PASS repeatability=PASS\n";
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
