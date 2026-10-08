// Integration: finite hidden-fiber observation -> 512D -> 8^3 SSA tensor -> Conv3D -> residual energy.
#include "dm3d/fiber264.hpp"
#include "dm3d_ssa.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace f=dm3d::fiber;

static dm3d::Value to_ssa(const f::State& x){
    const auto latent=f::embed512(x);
    dm3d::Value v=dm3d::filled(dm3d::Shape::tensor(8,8,8),0.0f);
    for(std::size_t i=0;i<f::LATENT_DIM;++i)v.data[i]=float(latent[i]);
    return v;
}
int main(){
    try {
        dm3d::Graph graph{
            {
                {dm3d::Op::Input},
                {dm3d::Op::Conv3D,0,-1,0.8f,0.2f},
                {dm3d::Op::Multiply,1,1},
                {dm3d::Op::Mean,2}
            },3
        };
        const auto zero=to_ssa(f::V_STAR);
        const auto nonzero=f::from_index(f::STATE_COUNT-1);
        const auto input=to_ssa(nonzero);
        if(input.data.size()!=512 || input.shape!=dm3d::Shape::tensor(8,8,8))
            throw std::runtime_error("SSA tensor adapter shape violation");
        const auto a=dm3d::execute(graph,zero),b=dm3d::execute(graph,input);
        if(a.shape!=dm3d::Shape::scalar() || b.shape!=dm3d::Shape::scalar()
            || a.data.at(0)!=0 || !std::isfinite(b.data.at(0)) || b.data.at(0)<=1e-6f)
            throw std::runtime_error("SSA convolution did not distinguish hidden observations");
        if(f::collapse(nonzero)!=f::collapse(f::V_STAR))
            throw std::runtime_error("SSA operator affected constant collapse");
        if(f::decode512(f::embed512(nonzero))!=nonzero)
            throw std::runtime_error("512D observation adapter lost group state");
        std::cout<<"PASS finite fiber -> 512D -> SSA Tensor3D[8,8,8] -> Conv3D -> Mean"
                 <<" scalar_energy="<<b.data.at(0)<<"\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"FAIL fiber/SSA integration "<<e.what()<<"\n";return 1;
    }
}
