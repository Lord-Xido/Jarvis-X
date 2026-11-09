#define JX_FRACTAL_CHAIN_NO_MAIN
#include "main_fractal_runtime.cpp"
#include "spatial_codec.hpp"
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>

static double sqerr(const std::vector<float>& a,const std::vector<float>& b) {
 if(a.size()!=b.size()||a.empty())throw std::invalid_argument("invalid MSE arrays");
 double sum=0;for(size_t i=0;i<a.size();++i){double d=double(a[i])-b[i];sum+=d*d;}return sum/a.size();
}
int main(int argc,char**argv) {
 try{
  std::string prefix=argc>1?argv[1]:"spatial_quality";
  constexpr unsigned D=16,NITER=12;
  constexpr double lambda=0.02;
  jarvisx::Runtime reference(D,1,NITER);
  reference.mandelbulb();reference.permeate(4,0.12f);
  const auto source=reference.volume;
  const size_t raw_bytes=source.size()*sizeof(float);
  double mean=0;for(float x:source)mean+=x;mean/=source.size();
  const double constant_mse=sqerr(source,std::vector<float>(source.size(),float(mean)));
  if(!(constant_mse>0))throw std::runtime_error("reference unexpectedly constant");
  jarvisx::Runtime legacy(D,5,NITER);legacy.run();
  const double legacy_mse=sqerr(source,legacy.decoded);
  jarvisx::Runtime integrated(D,5,NITER);
  integrated.codec=jarvisx::Runtime::Codec::SpatialU8;
  integrated.rd_lambda=lambda;
  integrated.run();
  const double integrated_mse=sqerr(source,integrated.decoded);
  if(!std::isfinite(integrated_mse)||integrated.trace.size()!=51)
   throw std::runtime_error("invalid integrated VM spatial trace or metric");
  std::cout<<"VM_SPATIAL levels=5 mse="<<std::setprecision(12)<<integrated_mse
    <<" legacy_mse="<<legacy_mse<<" selected_block="<<integrated.rd_block
    <<" encoded_bytes="<<integrated.rd_bytes<<'\n';
  if(!(integrated_mse<legacy_mse))
   throw std::runtime_error("integrated spatial VM failed to beat legacy projection");
  std::ofstream csv(prefix+".csv");
  if(!csv)throw std::runtime_error("cannot create spatial-quality CSV");
  csv<<"block,payload_bytes,serialized_bytes,raw_to_encoded_ratio,mse,ratio_to_mean_baseline,objective,legacy_mse\n";
  double best_objective=std::numeric_limits<double>::infinity();
  double best_mse=std::numeric_limits<double>::infinity();
  unsigned best_block=0;
  jarvisx::SpatialCodec::Payload winner;
  for(unsigned B:{2u,4u,8u}){
   auto encoded=jarvisx::SpatialCodec::encode(source,D,B);
   auto decoded=jarvisx::SpatialCodec::decode(encoded);
   const double mse=sqerr(source,decoded);
   size_t encoded_bytes=jarvisx::SpatialCodec::serialized_size(encoded);
   const double ratio=double(raw_bytes)/encoded_bytes;
   const double objective=mse+lambda*encoded_bytes/double(raw_bytes);
   if(!std::isfinite(mse)||!std::isfinite(objective))throw std::runtime_error("invalid RD metric");
   csv<<B<<','<<encoded.bytes.size()<<','<<encoded_bytes<<','<<std::setprecision(12)<<ratio
      <<','<<mse<<','<<mse/constant_mse<<','<<objective<<','<<legacy_mse<<'\n';
   std::cout<<std::setprecision(12)<<"SPATIAL block="<<B<<" bytes="<<encoded_bytes
      <<" ratio="<<ratio<<" mse="<<mse<<" mean_baseline_mse="<<constant_mse
      <<" legacy_5level_mse="<<legacy_mse<<" objective="<<objective<<'\n';
   if(objective<best_objective){best_objective=objective;best_mse=mse;best_block=B;winner=encoded;}
  }
  if(!csv)throw std::runtime_error("CSV write failed");
  const std::string file=prefix+".jx3b";
  jarvisx::SpatialCodec::save(winner,file);
  auto readback=jarvisx::SpatialCodec::load(file);
  if(readback.bytes!=winner.bytes||readback.block!=winner.block||readback.dim!=winner.dim)
   throw std::runtime_error("spatial payload round-trip mismatch");
  if(sqerr(source,jarvisx::SpatialCodec::decode(readback))!=best_mse)
   throw std::runtime_error("decoded reconstruction round-trip mismatch");
  // Corruption-detection regression. FNV-32 is an accidental-error checksum,
  // NOT an authentication or tamper-resistance primitive.
  std::ifstream archive(file,std::ios::binary);
  std::vector<char> corrupt((std::istreambuf_iterator<char>(archive)),{});
  if(corrupt.size()<25)throw std::runtime_error("truncated archive");
  corrupt[24]=char(uint8_t(corrupt[24])^1u);
  const std::string corrupted=prefix+".corrupted";
  {std::ofstream out(corrupted,std::ios::binary);out.write(corrupt.data(),std::streamsize(corrupt.size()));}
  bool rejected=false;
  try {jarvisx::SpatialCodec::load(corrupted);}catch(const std::runtime_error&){rejected=true;}
  if(!rejected)throw std::runtime_error("tampered byte was not detected");
  if(!(best_mse<constant_mse && best_mse<legacy_mse))
   throw std::runtime_error("spatial block codec did not beat reference baselines");
  std::cout<<"SPATIAL_BEST block="<<best_block<<" mse="<<best_mse
   <<" mean_baseline_mse="<<constant_mse<<" legacy_mse="<<legacy_mse
   <<" improved_over_constant="<<(best_mse<constant_mse)
   <<" improved_over_legacy="<<(best_mse<legacy_mse)
   <<" checksum_reject=1\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"SPATIAL_ERROR: "<<e.what()<<'\n';return 1;}
}
