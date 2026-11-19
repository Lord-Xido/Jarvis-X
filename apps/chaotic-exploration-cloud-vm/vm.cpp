#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// JARVIS X - bounded, deterministic 3D bytecode VM (CPU reference).
// This is an application-level virtual machine, not a hardware hypervisor.
namespace {
enum class Op : uint8_t { INGEST=0x10, LORENZ=0x20, ATTEND_GRU=0x30,
    FOLD_3D=0x40, PLAN=0x50, VERIFY=0x60, HALT=0xff };
constexpr std::array<Op,7> program = {Op::INGEST,Op::LORENZ,Op::ATTEND_GRU,
    Op::FOLD_3D,Op::PLAN,Op::VERIFY,Op::HALT};
struct Config {int side=12,steps=8,seed=42,horizon=8; double alpha=.09,beta=.12,chaos=.03,epsilon=.0001;};
int integer(const std::string &s, int low, int high) {
    size_t end=0; long long n=std::stoll(s,&end);
    if(end!=s.size() || n<low || n>high) throw std::invalid_argument("integer out of range");
    return static_cast<int>(n);
}
double real(const std::string &s,double low,double high) {
    size_t end=0;double x=std::stod(s,&end);
    if(end!=s.size() || !std::isfinite(x) || x<low || x>high) throw std::invalid_argument("float out of range");
    return x;
}
uint64_t splitmix64(uint64_t &x) {
    uint64_t z=(x+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
double sigmoid(double x){return 1.0/(1.0+std::exp(-x));}
struct Engine {
    Config cfg;
    std::vector<double> grid,previous,temporary;
    std::array<double,3> q{.1,0.,0.};
    double rms=0,linf=0,field_energy=0,selected_cost=0;
    int path=0,iterations=0;
    explicit Engine(Config c):cfg(c),grid(static_cast<size_t>(c.side)*c.side*c.side),
     previous(grid.size()),temporary(grid.size()){}
    size_t idx(int x,int y,int z)const{
        int n=cfg.side;
        x=(x+n)%n; y=(y+n)%n; z=(z+n)%n;
        return (static_cast<size_t>(z)*n+y)*n+x;
    }
    void ingest(){
        uint64_t seed=static_cast<uint64_t>(cfg.seed);
        for(double &v:grid) v=2.0*(splitmix64(seed)>>11)*(1.0/9007199254740992.0)-1.0;
    }
    void lorenz(){
        constexpr double dt=.005;
        auto [x,y,z]=q;
        q[0]+=dt*10*(y-x);
        q[1]+=dt*(x*(28-z)-y);
        q[2]+=dt*(x*y-(8.0/3.0)*z);
        for(double &v:grid)v=std::tanh(v+cfg.chaos*q[0]);
    }
    void attend_gru(){
        // Scalar attention context and scalar gated recurrent update; NOT a trained Transformer/GRU.
        double weighted=0,denom=0;
        for(double v:grid){double w=std::exp(v);weighted+=w*v;denom+=w;}
        const double context=weighted/denom;
        for(double &v:grid){const double gate=sigmoid(v+context);
            v=(1-gate)*v+gate*std::tanh(.75*v+.25*context);}
    }
    void fold(){
        const int n=cfg.side;
        double mean=0;for(double v:grid)mean+=v;mean/=grid.size();
        for(int z=0;z<n;++z)for(int y=0;y<n;++y)for(int x=0;x<n;++x){
            size_t i=idx(x,y,z);double v=grid[i];
            double lap=grid[idx(x-1,y,z)]+grid[idx(x+1,y,z)]
                +grid[idx(x,y-1,z)]+grid[idx(x,y+1,z)]
                +grid[idx(x,y,z-1)]+grid[idx(x,y,z+1)]-6*v;
            temporary[i]=v+cfg.alpha*lap-cfg.beta*(v-mean);
        }
        grid.swap(temporary);
    }
    void plan(){
        const int n=cfg.side,center=n/2;
        constexpr int dirs[4][3]={{1,0,0},{0,1,0},{0,0,1},{1,1,1}};
        selected_cost=std::numeric_limits<double>::infinity();
        for(int d=0;d<4;++d){double cost=0;
            for(int t=0;t<cfg.horizon;++t){
                int x=(center+dirs[d][0]*t)%n,y=(center+dirs[d][1]*t)%n,z=(center+dirs[d][2]*t)%n;
                double v=grid[idx(x,y,z)];
                cost+=v*v+.01*std::abs(v);
            }
            cost/=cfg.horizon;
            if(cost<selected_cost){selected_cost=cost;path=d;}
        }
    }
    void verify(){
        double sq=0,energy=0;linf=0;
        for(size_t i=0;i<grid.size();++i){
            if(!std::isfinite(grid[i]))throw std::runtime_error("non-finite state");
            double diff=grid[i]-previous[i];sq+=diff*diff;
            linf=std::max(linf,std::abs(diff));energy+=grid[i]*grid[i];
        }
        rms=std::sqrt(sq/grid.size());field_energy=energy/grid.size();
        if(!std::isfinite(rms)||!std::isfinite(selected_cost))throw std::runtime_error("non-finite metric");
    }
    void execute(){
        bool initialized=false;
        for(int step=0;step<cfg.steps;++step){
            previous=grid;
            for(Op op:program){switch(op){
                case Op::INGEST: if(!initialized){ingest();initialized=true;}break;
                case Op::LORENZ:lorenz();break;
                case Op::ATTEND_GRU:attend_gru();break;
                case Op::FOLD_3D:fold();break;
                case Op::PLAN:plan();break;
                case Op::VERIFY:verify();break;
                case Op::HALT:break;
            }}
            iterations=step+1;
            if(rms<cfg.epsilon)break;
        }
    }
    // FNV-1a of deterministic 16-bit quantized values; non-cryptographic fingerprint.
    uint64_t fingerprint()const{
        uint64_t h=14695981039346656037ULL;
        for(double v:grid){int32_t qv=static_cast<int32_t>(std::round(v*10000));
            uint32_t u=static_cast<uint32_t>(qv);
            for(int k=0;k<4;++k){h^=(u>>(8*k))&255u;h*=1099511628211ULL;}}
        return h;
    }
};
}
int main(int argc,char**argv){
    try{
        Config cfg;
        if((argc-1)%2)throw std::invalid_argument("expected --key value pairs");
        for(int i=1;i<argc;i+=2){std::string k=argv[i],v=argv[i+1];
            if(k=="--side")cfg.side=integer(v,4,32);
            else if(k=="--steps")cfg.steps=integer(v,1,100);
            else if(k=="--seed")cfg.seed=integer(v,0,2147483647);
            else if(k=="--horizon")cfg.horizon=integer(v,1,32);
            else if(k=="--alpha")cfg.alpha=real(v,0,.166666666666);
            else if(k=="--beta")cfg.beta=real(v,0,.5);
            else if(k=="--chaos")cfg.chaos=real(v,0,.2);
            else if(k=="--epsilon")cfg.epsilon=real(v,0,1);
            else throw std::invalid_argument("unknown option: "+k);
        }
        const auto start=std::chrono::steady_clock::now();
        Engine engine(cfg);engine.execute();
        const auto stop=std::chrono::steady_clock::now();
        const double ms=std::chrono::duration<double,std::milli>(stop-start).count();
        std::cout<<std::fixed<<std::setprecision(10)
            <<"{\"engine\":\"jarvisx-chaotic-3d-bytecode-cpu-v1\",\"execution\":\"native\","
            <<"\"verified_finite\":true,\"side\":"<<cfg.side<<",\"voxels\":"<<engine.grid.size()
            <<",\"iterations\":"<<engine.iterations<<",\"residual_rms\":"<<engine.rms
            <<",\"residual_linf\":"<<engine.linf<<",\"field_energy\":"<<engine.field_energy
            <<",\"path_id\":"<<engine.path<<",\"path_cost\":"<<engine.selected_cost
            <<",\"lorenz\":["<<engine.q[0]<<","<<engine.q[1]<<","<<engine.q[2]<<"]"
            <<",\"fingerprint64\":\""<<std::hex<<std::setw(16)<<std::setfill('0')<<engine.fingerprint()
            <<std::dec<<"\",\"runtime_ms\":"<<ms
            <<",\"opcodes\":[\"INGEST\",\"LORENZ\",\"ATTEND_GRU\",\"FOLD_3D\",\"PLAN\",\"VERIFY\",\"HALT\"]}\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<"VM ERROR: "<<e.what()<<'\n';return 2;}
}
