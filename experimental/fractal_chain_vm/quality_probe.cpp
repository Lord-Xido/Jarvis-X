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
  double target_mean=0;
  for(float x:target)target_mean+=x;
  target_mean/=target.size();
  std::vector<float> mean_prediction(target.size(),float(target_mean));
  const double mean_baseline=mse(target,mean_prediction);
  if(!std::isfinite(mean_baseline)||mean_baseline<=0)
   throw std::runtime_error("invalid reference variance");
  std::ofstream csv(output);
  if(!csv)throw std::runtime_error("cannot write quality report");
  csv<<"levels,reference_mse,constant_0_5_baseline_mse,mean_baseline_mse,relative_to_0_5,relative_to_mean,internal_final_mse,output_hash,decoded_stddev,psnr_db\\n";
  bool any_better=false;
  for(unsigned n=1;n<=levels;++n){
   jarvisx::Runtime r(dim,n,iterations);r.run();
   double err=mse(target,r.decoded);
   if(!std::isfinite(err)||!std::isfinite(r.mse_value)||r.trace.size()!=size_t(n)*10+1)
    throw std::runtime_error("invalid quality metrics or trace");
   if(err<mean_baseline)any_better=true;
   double decoded_mean=0;
   for(float v:r.decoded)decoded_mean+=v;
   decoded_mean/=r.decoded.size();
   double decoded_var=0;
   for(float v:r.decoded){double delta=v-decoded_mean;decoded_var+=delta*delta;}
   decoded_var/=r.decoded.size();
   double psnr=err>0?10*std::log10(1/err):std::numeric_limits<double>::infinity();
   csv<<n<<','<<std::setprecision(12)<<err<<','<<baseline<<','<<mean_baseline<<','
      <<(baseline>0?err/baseline:std::numeric_limits<double>::quiet_NaN())<<','
      <<err/mean_baseline<<','<<r.mse_value<<','<<jarvisx::hash(r.decoded)<<','
      <<std::sqrt(decoded_var)<<','<<psnr<<'\n';
   std::cout<<std::setprecision(12)<<"QUALITY levels="<<n<<" fixed_mse="<<err
            <<" mean_baseline_mse="<<mean_baseline<<" ratio="<<err/mean_baseline
            <<" decoded_stddev="<<std::sqrt(decoded_var)<<" psnr_db="<<psnr<<'\n';
  }
  std::cout<<"Quality probe completed. Any level beats optimal constant-mean baseline: "<<any_better
           <<". No monotonic improvement asserted.\n";
  return csv?0:1;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
