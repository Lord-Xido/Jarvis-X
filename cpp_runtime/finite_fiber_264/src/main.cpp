#include "dm3d/fiber264.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace dm3d::fiber;

int main(int argc,char** argv) {
    try {
        int steps=16;
        double lambda=0.73;
        bool exhaustive=false;
        for(int i=1;i<argc;++i){
            std::string arg(argv[i]);
            if(arg=="--exhaustive")exhaustive=true;
            else if(arg=="--steps" && i+1<argc)steps=std::stoi(argv[++i]);
            else if(arg=="--lambda" && i+1<argc)lambda=std::stod(argv[++i]);
            else if(arg=="--help"){
                std::cout<<"Usage: dm3d-fiber264 [--steps N] [--lambda 0.73] [--exhaustive]\n";
                return 0;
            }else throw std::invalid_argument("unknown or incomplete argument: "+arg);
        }
        if(steps<1||steps>100000)throw std::invalid_argument("steps outside [1,100000]");
        State x=from_index(752001);
        State g=control_shift(42,17);
        std::cout<<"DM3D FINITE 3D FIBER |X|="<<STATE_COUNT
                 <<" |image(C)|=1 |Fix(C)|=1 |C^-1(v*)|="<<STATE_COUNT
                 <<" translation_order="<<translation_order(g)<<"\n";
        for(int n=0;n<steps;++n){
            if(!ctr_invariant(x,g))throw std::runtime_error("CTR invariant failed");
            x=add(x,g);
            if(n<5||n+1==steps)
                std::cout<<"cycle="<<n+1<<" hidden_index="<<to_index(x)
                         <<" visible_index="<<to_index(collapse(x))<<"\n";
        }
        // Inward refinement is a *different*, continuous process from finite translation.
        const auto target=embed512(x);
        Latent512 initial{};
        const auto result=refine(initial,target,lambda,100,1e-8);
        std::cout<<std::setprecision(12)
                 <<"inward_iterations="<<result.iterations
                 <<" final_step_delta="<<result.final_delta
                 <<" latent_mse="<<result.final_mse
                 <<" state_decoded="<<(decode512(result.latent)==x?"true":"false")
                 <<" converged="<<(result.converged?"true":"false")<<"\n";
        if(!result.converged || decode512(result.latent)!=x)
            throw std::runtime_error("inward convergence or decoding did not pass");
        if(exhaustive) {
            std::uint64_t visited=0;
            for(std::uint64_t i=0;i<STATE_COUNT;++i) {
                const State v=from_index(i);
                if(to_index(v)!=i || collapse(v)!=V_STAR)
                    throw std::runtime_error("exhaustive bijection/collapse failure");
                ++visited;
            }
            if(visited!=STATE_COUNT)throw std::runtime_error("incomplete enumeration");
            std::cout<<"EXHAUSTIVE PASS states="<<visited<<"\n";
        }
        std::cout<<"CTR PASS\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"CTR FAIL: "<<e.what()<<"\n";
        return 1;
    }
}
