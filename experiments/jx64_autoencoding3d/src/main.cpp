#include "engine.hpp"
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>

int main(int argc,char**argv){
    try{
      jx64::Config cfg;std::string mode="separable";int repeats=1;
      for(int i=1;i<argc;i++){
        const std::string k=argv[i];
        if(i+1>=argc)throw std::invalid_argument("expected argument after "+k);
        const std::string v=argv[++i];
        if(k=="--seed")cfg.seed=std::stoull(v,nullptr,0);
        else if(k=="--stride")cfg.stride=static_cast<std::uint32_t>(std::stoull(v,nullptr,0));
        else if(k=="--iterations")cfg.iterations=std::stoi(v);
        else if(k=="--gain")cfg.feedback_gain=std::stof(v);
        else if(k=="--decoder")mode=v;
        else if(k=="--repeat")repeats=std::stoi(v);
        else throw std::invalid_argument("unknown option: "+k);
      }
      if(repeats<1||repeats>100)throw std::invalid_argument("repeat must be in 1..100");
      auto engine=std::make_unique<jx64::AutoEncoding3DEngine>();
      jx64::Metrics result{};
      auto start=std::chrono::steady_clock::now();
      for(int k=0;k<repeats;k++)result=engine->execute(cfg,mode);
      auto end=std::chrono::steady_clock::now();
      const double ms=std::chrono::duration<double,std::milli>(end-start).count()/repeats;
      std::cout<<std::setprecision(12)
        <<"geometry=64x64x64 voxels="<<jx64::N<<" decoder="<<mode<<" iterations="<<cfg.iterations<<" stride="<<jx64::AutoEncoding3DEngine::normalized_stride(cfg.stride)<<"\n"
        <<"elapsed_ms_per_run="<<ms<<" checksum32="<<result.checksum32<<"\n"
        <<"mse_to_original_seed="<<result.mse_to_seed<<" mse_to_encoded_activations="<<result.mse_to_encoded<<"\n"
        <<"seed_hash_fnv64="<<std::hex<<result.input_hash<<" output_hash_fnv64="<<result.output_hash<<std::dec<<"\n";
      return 0;
    }catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 2;}
}
