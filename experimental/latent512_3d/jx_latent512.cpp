// JARVIS-X Ω³: auditable reference 64^3 -> 512-float 3D autoencoder.
// C++17, no external dependencies. Research prototype: fixed box-pool encoder
// plus trainable 27-tap trilinear decoder; NOT a deep neural network.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace jxl512 {
constexpr int SIDE=64, COARSE=8, BLOCK=8, LATENT=512, TAPS=27;
constexpr size_t VOXELS=size_t(SIDE)*SIDE*SIDE, ARCHIVE_BYTES=2304;
using Volume=std::vector<uint8_t>;
using Z=std::array<float,LATENT>;
using Features=std::vector<std::array<float,TAPS>>;
struct Model { std::array<double,TAPS> w{}; double bias=0; };
struct Archive {Z z{}; Model model{};};
struct Step {double before=0, after=0; unsigned accepted=0;};
constexpr uint64_t FNV_OFFSET=14695981039346656037ULL;
uint64_t fnv(const uint8_t* p,size_t n) {uint64_t h=FNV_OFFSET; for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;} return h;}
uint64_t hash(const Volume& v){return fnv(v.data(),v.size());}
int idx3(int x,int y,int z,int side){return (z*side+y)*side+x;}
int clampi(int x,int low,int high){return std::max(low,std::min(high,x));}
Model identity(){Model m; m.w[13]=1.; return m;}
Z encode(const Volume& v) {
  if(v.size()!=VOXELS) throw std::invalid_argument("input must be exactly 64^3 uint8 voxels");
  Z z{};
  for(int cz=0;cz<COARSE;++cz) for(int cy=0;cy<COARSE;++cy) for(int cx=0;cx<COARSE;++cx){
    uint32_t sum=0;
    for(int dz=0;dz<BLOCK;++dz)for(int dy=0;dy<BLOCK;++dy)for(int dx=0;dx<BLOCK;++dx)
      sum+=v[size_t(idx3(cx*BLOCK+dx,cy*BLOCK+dy,cz*BLOCK+dz,SIDE))];
    z[size_t(idx3(cx,cy,cz,COARSE))]=float(double(sum)/(BLOCK*BLOCK*BLOCK*255.0));
  }
  return z;
}
float sample(const Z& z,double x,double y,double zz){
  const int x0=int(std::floor(x)),y0=int(std::floor(y)),z0=int(std::floor(zz));
  const double tx=x-x0,ty=y-y0,tz=zz-z0;
  double s=0;
  for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
    const double w=(dx?tx:1-tx)*(dy?ty:1-ty)*(dz?tz:1-tz);
    s+=w*z[size_t(idx3(clampi(x0+dx,0,7),clampi(y0+dy,0,7),clampi(z0+dz,0,7),COARSE))];
  }
  return float(s);
}
Features features(const Z& z){
  Features f(VOXELS);
  for(int zz=0;zz<SIDE;++zz)for(int y=0;y<SIDE;++y)for(int x=0;x<SIDE;++x){
    const double xx=(x+0.5)/BLOCK-0.5, yy=(y+0.5)/BLOCK-0.5, zz0=(zz+0.5)/BLOCK-0.5;
    auto& row=f[size_t(idx3(x,y,zz,SIDE))]; int k=0;
    for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)
      row[size_t(k++)]=sample(z,xx+dx,yy+dy,zz0+dz);
  }
  return f;
}
double predict(const std::array<float,TAPS>& f,const Model& m){double y=m.bias;for(int k=0;k<TAPS;++k)y+=m.w[size_t(k)]*f[size_t(k)];return y;}
Volume decode(const Features& f,const Model& m){
  if(f.size()!=VOXELS)throw std::invalid_argument("feature count mismatch");
  Volume v(VOXELS);
  for(size_t i=0;i<VOXELS;++i){double x=predict(f[i],m);if(!std::isfinite(x))throw std::runtime_error("nonfinite reconstruction");
    v[i]=uint8_t(std::lround(std::max(0.,std::min(1.,x))*255.));}
  return v;
}
double mse(const Volume& a,const Volume& b){if(a.size()!=VOXELS||b.size()!=VOXELS)throw std::invalid_argument("MSE dimension mismatch");double s=0;for(size_t i=0;i<VOXELS;++i){double e=int(a[i])-int(b[i]);s+=e*e;}return s/VOXELS;}
double objective(const Features& f,const Volume& target,const Model& m,std::array<double,TAPS>* gradient=nullptr,double* gb=nullptr){
  if(f.size()!=VOXELS||target.size()!=VOXELS)throw std::invalid_argument("training dimension mismatch");
  if(gradient)gradient->fill(0);
  if(gb)*gb=0;
  double s=0;
  for(size_t i=0;i<VOXELS;++i){double e=predict(f[i],m)-double(target[i])/255.;s+=e*e;
    if(gradient){for(int k=0;k<TAPS;++k)(*gradient)[size_t(k)]+=2*e*f[i][size_t(k)];if(gb)*gb+=2*e;}}
  if(gradient){for(double& x:*gradient)x/=VOXELS;if(gb)*gb/=VOXELS;}
  return s/VOXELS;
}
Step optimise(const Features& f,const Volume& v,Model& m,unsigned epochs){
  Step step;step.before=objective(f,v,m);double best=step.before;
  for(unsigned iter=0;iter<epochs;++iter){std::array<double,TAPS> g{};double gb=0;objective(f,v,m,&g,&gb);
    double scale=1.;bool applied=false;
    for(unsigned trial=0;trial<20;++trial){Model candidate=m;
      for(int k=0;k<TAPS;++k)candidate.w[size_t(k)]-=scale*g[size_t(k)];
      candidate.bias-=scale*gb;
      const double loss=objective(f,v,candidate);
      if(std::isfinite(loss)&&loss+1e-15<best){m=candidate;best=loss;++step.accepted;applied=true;break;}
      scale*=0.5;
    }
    if(!applied)break;
  }
  step.after=best;return step;
}
void write_u(std::vector<uint8_t>& out,uint64_t x,unsigned n){for(unsigned i=0;i<n;++i)out.push_back(uint8_t(x>>(8*i)));}
uint64_t read_u(const std::vector<uint8_t>& v,size_t& p,unsigned n){if(p+n>v.size())throw std::runtime_error("truncated archive");uint64_t x=0;for(unsigned i=0;i<n;++i)x|=uint64_t(v[p++])<<(8*i);return x;}
void write_f32(std::vector<uint8_t>& b,float x){uint32_t bits;static_assert(sizeof(bits)==sizeof(x),"float32 required");std::memcpy(&bits,&x,4);write_u(b,bits,4);}
void write_f64(std::vector<uint8_t>& b,double x){uint64_t bits;static_assert(sizeof(bits)==sizeof(x),"float64 required");std::memcpy(&bits,&x,8);write_u(b,bits,8);}
float read_f32(const std::vector<uint8_t>& b,size_t& p){uint32_t x=uint32_t(read_u(b,p,4));float f;std::memcpy(&f,&x,4);if(!std::isfinite(f))throw std::runtime_error("nonfinite latent");return f;}
double read_f64(const std::vector<uint8_t>& b,size_t& p){uint64_t x=read_u(b,p,8);double f;std::memcpy(&f,&x,8);if(!std::isfinite(f))throw std::runtime_error("nonfinite model");return f;}
std::vector<uint8_t> pack(const Archive& a){
  std::vector<uint8_t> b; b.reserve(ARCHIVE_BYTES);
  for(char c:std::string("JXL51201"))b.push_back(uint8_t(c));
  write_u(b,SIDE,4);write_u(b,LATENT,4);write_u(b,TAPS,4);write_u(b,512*4+27*8+8,4);
  for(float x:a.z){if(!std::isfinite(x)||x<0||x>1)throw std::invalid_argument("invalid latent");write_f32(b,x);}
  for(double x:a.model.w){if(!std::isfinite(x))throw std::invalid_argument("invalid weight");write_f64(b,x);}
  if(!std::isfinite(a.model.bias))throw std::invalid_argument("invalid bias");
  write_f64(b,a.model.bias);
  write_u(b,fnv(b.data(),b.size()),8);if(b.size()!=ARCHIVE_BYTES)throw std::logic_error("archive size");return b;
}
Archive unpack(const std::vector<uint8_t>& b){
  if(b.size()!=ARCHIVE_BYTES)throw std::runtime_error("archive must be exactly 2304 bytes");
  if(std::string(b.begin(),b.begin()+8)!="JXL51201")throw std::runtime_error("invalid magic");
  size_t p=8;
  if(read_u(b,p,4)!=SIDE||read_u(b,p,4)!=LATENT||read_u(b,p,4)!=TAPS||read_u(b,p,4)!=2272)throw std::runtime_error("invalid archive dimensions");
  Archive a;for(float& x:a.z){x=read_f32(b,p);if(x<0||x>1)throw std::runtime_error("invalid latent range");}
  for(double& x:a.model.w)x=read_f64(b,p);
  a.model.bias=read_f64(b,p);
  if(read_u(b,p,8)!=fnv(b.data(),b.size()-8)||p!=b.size())throw std::runtime_error("checksum mismatch");
  return a;
}
Volume synth(unsigned seed){
  Volume v(VOXELS);
  for(int z=0;z<SIDE;++z)for(int y=0;y<SIDE;++y)for(int x=0;x<SIDE;++x){
    const double a=0.44+0.17*std::sin((x+seed)*0.079)+0.13*std::cos((y+2*seed)*0.065)+0.10*std::sin((z+seed)*0.098);
    const double r=std::sqrt((x-28.)*(x-28.)+(y-36.)*(y-36.)+(z-30.)*(z-30.));
    const double value=std::max(0.,std::min(1.,a+0.15*std::exp(-r*r/270.)));
    v[size_t(idx3(x,y,z,SIDE))]=uint8_t(std::lround(255*value));
  }
  return v;
}
std::vector<uint8_t> read_file(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open "+path);return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};}
void write_file(const std::string& path,const std::vector<uint8_t>& b){std::ofstream f(path,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("cannot write "+path);f.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));if(!f)throw std::runtime_error("write failed "+path);}
void require(bool yes,const std::string& what){if(!yes)throw std::runtime_error("selftest: "+what);}
void selftest(){
  for(uint8_t c:{uint8_t(0),uint8_t(177),uint8_t(255)}){
    Volume original(VOXELS,c);const Z z=encode(original);auto f=features(z);require(decode(f,identity())==original,"constant roundtrip");
  }
  Volume input=synth(11);const Z z=encode(input);const auto f=features(z);Model m=identity();
  auto before=decode(f,m);const double first=mse(input,before);
  Step s=optimise(f,input,m,6);
  require(s.after<=s.before+1e-14,"training worsened objective");require(s.accepted>0,"optimizer accepted no step");
  auto bytes=pack({z,m});require(bytes.size()==ARCHIVE_BYTES,"archive size");
  const Archive a=unpack(bytes);const auto output=decode(features(a.z),a.model);
  require(output==decode(f,m),"serialize roundtrip");require(std::isfinite(mse(input,output)),"invalid MSE");
  auto altered=bytes;altered[72]^=1;bool rejected=false;try{(void)unpack(altered);}catch(const std::runtime_error&){rejected=true;}require(rejected,"corruption rejection");
  altered=bytes;altered.pop_back();rejected=false;try{(void)unpack(altered);}catch(const std::runtime_error&){rejected=true;}require(rejected,"truncation rejection");
  Model repeat=identity();auto s2=optimise(f,input,repeat,6);require(pack({z,repeat})==bytes&&s2.accepted==s.accepted,"same-build reproducibility");
  Volume heldout=synth(29);auto z2=encode(heldout);const auto f2=features(z2);
  std::cout<<std::setprecision(10)<<"SELFTEST PASS\ntrain_mse_before="<<first<<" train_mse_after="<<mse(input,output)
           <<" heldout_mse="<<mse(heldout,decode(f2,m))<<" iterations_accepted="<<s.accepted<<" bytes="<<bytes.size()<<"\n";
}
void demo(){
  const Volume input=synth(11); std::cout<<"TRACE 00 op=0x01 LOAD input_fnv64="<<std::hex<<hash(input)<<std::dec<<" bytes="<<input.size()<<"\n";
  const Z z=encode(input);auto f=features(z);Model m=identity();
  std::cout<<"TRACE 01 op=0x10 ENCODE latent_count="<<LATENT<<" latent_bytes="<<LATENT*4<<"\n";
  const double before=mse(input,decode(f,m));const Step s=optimise(f,input,m,6);
  std::cout<<std::setprecision(10)<<"TRACE 02 op=0x11 TRAIN normalized_objective_before="<<s.before<<" after="<<s.after<<" accepted="<<s.accepted<<"\n";
  const auto archive=pack({z,m});std::cout<<"TRACE 03 op=0x20 PACK archive_bytes="<<archive.size()<<" fnv64="<<std::hex<<fnv(archive.data(),archive.size())<<std::dec<<"\n";
  const auto restored=unpack(archive);std::cout<<"TRACE 04 op=0x21 UNPACK checksum=OK\n";
  const Volume output=decode(features(restored.z),restored.model);
  std::cout<<"TRACE 05 op=0x30 DECODE output_fnv64="<<std::hex<<hash(output)<<std::dec<<" bytes="<<output.size()<<"\n";
  const double error=mse(input,output);
  std::cout<<"TRACE 06 op=0x31 EVAL mse_uint8_before="<<before<<" after="<<error
           <<" psnr_db="<<(error==0?std::numeric_limits<double>::infinity():10*std::log10(255*255/error))<<"\n";
  std::cout<<"TRACE 07 op=0xff HALT nominal_uint8_to_f32_latent_ratio=128 actual_uint8_to_archive_ratio="<<double(VOXELS)/archive.size()<<"\n";
}
} // namespace jxl512
int main(int argc,char** argv){
  try{using namespace jxl512;
    const std::string command=argc>1?argv[1]:"demo";
    if(command=="selftest"&&argc==2){selftest();return 0;}
    if(command=="demo"&&argc==2){demo();return 0;}
    if((command=="encode"||command=="train")&&argc==4){
      Volume input=read_file(argv[2]);Z z=encode(input);Model m=identity();
      if(command=="train"){auto f=features(z);const Step s=optimise(f,input,m,6);std::cout<<"training normalized MSE: "<<s.before<<" -> "<<s.after<<" accepted="<<s.accepted<<"\n";}
      auto a=pack({z,m});write_file(argv[3],a);std::cout<<"wrote "<<a.size()<<" bytes\n";return 0;
    }
    if(command=="decode"&&argc==4){Archive a=unpack(read_file(argv[2]));auto v=decode(features(a.z),a.model);write_file(argv[3],v);std::cout<<"wrote "<<v.size()<<" voxels\n";return 0;}
    std::cerr<<"usage: jxl512 demo | selftest | encode input.raw output.jxl512 | train input.raw output.jxl512 | decode input.jxl512 output.raw\n";return 2;
  }catch(const std::exception& e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
