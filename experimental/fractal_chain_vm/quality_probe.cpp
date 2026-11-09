#define JX_FRACTAL_CHAIN_NO_MAIN
#include "main_fractal_runtime.cpp"
#include <limits>
#include <sstream>

static double mse(const std::vector<float>& a,const std::vector<float>& b){
 if(a.size()!=b.size())throw std::runtime_error("shape mismatch");
 double sum=0;for(size_t i=0;i<a.size();++i){double d=double(a[i])-b[i];sum+=d*d;}
 return sum/a.size();
}
int main(int argc,char**argv){
 try {
  std::string output=argc>1?argv[1]:"quality.csv";
  constexpr unsigned dim=16,iterations=12,levels=5;
  jarvisx::Runtime reference(dim,1,iterations);
  reference.mandelbulb();
  reference.permeate(4,0.12f);
  const auto target=reference.volume;
  std::vector<float> constant(target.size(),0.5f);
  const double baseline=mse(target,constant);
  std::ofstream csv(output);
  if(!csv)throw std::runtime_error("cannot write quality report");
  csv<<"levels,reference_mse,constant_0_5_baseline_mse,relative_to_baseline,internal_final_mse,output_hash\n";
  bool any_better=false;
  for(unsigned n=1;n<=levels;++n){
   jarvisx::Runtime r(dim,n,iterations);r.run();
   double err=mse(target,r.decoded);
   if(!std::isfinite(err)||!std::isfinite(r.mse_value)||r.trace.size()!=size_t(n)*10+1)
    throw std::runtime_error("invalid quality metrics or trace");
   if(err<baseline)any_better=true;
   csv<<n<<','<<std::setprecision(12)<<err<<','<<baseline<<','
      <<(baseline>0?err/baseline:std::numeric_limits<double>::quiet_NaN())<<','
      <<r.mse_value<<','<<jarvisx::hash(r.decoded)<<'\n';
   std::cout<<"levels="<<n<<" fixed_reference_mse="<<err
            <<" constant_baseline_mse="<<baseline<<" internal_mse="<<r.mse_value<<'\n';
  }
  std::cout<<"Quality probe completed. Any level beats constant baseline: "<<any_better
           <<". No monotonic improvement asserted.\n";
  return csv?0:1;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
