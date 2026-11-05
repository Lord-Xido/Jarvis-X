// C++ in-process baseline for the exact v4 encoder; CLI/file-load excluded.
// This intentionally includes, but does not edit, the already verified v4 code.
#define main jx_ae3d_original_cli_main
#include "../src/jx_ae3d.cpp"
#undef main
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <numeric>

int main(int argc,char** argv){
    if(argc != 5){ std::cerr << "usage: bench_cpp_v4 MODEL VOLUME REPEATS THREADS_UNUSED\n";return 2; }
    try {
        const auto model=jx::load_model(argv[1]);
        const auto volume=jx::load_volume(argv[2]);
        const auto blocks=jx::blocks(volume);
        const int repeats=std::atoi(argv[3]);
        if(repeats<3 || repeats>1000) return 2;
        volatile unsigned long long sink=0;
        auto one=[&](){
            auto start=std::chrono::steady_clock::now();
            for(const auto& block:blocks){
                auto result=model.encode_q(block);
                sink+=static_cast<unsigned>(static_cast<unsigned char>(result[0]));
            }
            return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        };
        (void)one();
        std::vector<double> observations;
        for(int i=0;i<repeats;++i) observations.push_back(one());
        std::sort(observations.begin(),observations.end());
        std::cout << "{\"backend\":\"v4_cpp\",\"blocks\":"<<blocks.size()
                  <<",\"median_kernel_seconds\":"<<std::setprecision(12)<<observations[observations.size()/2]
                  <<",\"sink\":"<<sink<<"}\n";
    }catch(const std::exception& ex){std::cerr<<"benchmark error: "<<ex.what()<<'\n';return 1;}
}
