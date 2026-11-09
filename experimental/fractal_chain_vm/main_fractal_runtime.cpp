#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jarvisx {
enum Op:uint8_t { HALT=0xff, MANDELBULB=0x20, ENCODE=0x21, DECODE=0x22, PERMEATE=0x23, RECOIL=0x24, EVOLVE=0x25, BLEND=0x26, MSE=0x27, JUMP_IF=0x31 };
struct Instr { uint8_t op,a,b,c; float f; uint32_t target; };
struct Trace { unsigned level,pc,op; uint64_t ns,in_hash,out_hash; float mse,power; };
static uint64_t hash(const std::vector<float>& v) {
 uint64_t h=14695981039346656037ULL; for(float x:v){uint32_t b;std::memcpy(&b,&x,4);for(unsigned k=0;k<4;++k){h^=(b>>(8*k))&255;h*=1099511628211ULL;}}return h;
}
static uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
static void u32(std::ostream& f,uint32_t v){for(unsigned i=0;i<4;++i)f.put(char(v>>(8*i)));}
struct Runtime {
 unsigned D,levels,iterations,level=0,pc=0; uint64_t cycles=0;
 float power=8,mse_value=0,latent_mean=0;
 std::vector<float> volume,decoded,latent,prior_latent,velocity;
 std::vector<Trace> trace;
 std::vector<Instr> program={
  {MANDELBULB,0,0,0,8,0},{PERMEATE,0,0,4,0.12f,0},
  {ENCODE,0,0,128,0,0},{RECOIL,1,0,0,0.35f,0},
  {DECODE,0,0,0,0,0},{RECOIL,0,0,0,0,0},
  {MSE,0,0,0,0,0},{EVOLVE,0,0,0,0,0},
  {BLEND,0,0,0,0.5f,0},{JUMP_IF,0,0,0,0,0},{HALT,0,0,0,0,0}
 };
 Runtime(unsigned d,unsigned n,unsigned it):D(d),levels(n),iterations(it) {
  if(D<4||D>256||D%4||levels<1||levels>100||iterations<1||iterations>1024)
   throw std::invalid_argument("requires D divisible by 4 in [4,256], levels [1,100], iterations [1,1024]");
 }
 size_t count()const{return size_t(D)*D*D;}
 size_t at(unsigned z,unsigned y,unsigned x)const{return (size_t(z)*D+y)*D+x;}
 uint64_t state_hash()const{return hash(volume) ^ (hash(decoded)*0x9e3779b97f4a7c15ULL) ^ bits(power);}
 void mandelbulb() {
  volume.assign(count(),0);
  for(unsigned z=0;z<D;++z)for(unsigned y=0;y<D;++y)for(unsigned x=0;x<D;++x){
   float cx=(float(x)-D/2.f)/(D/2.f)*1.5f,cy=(float(y)-D/2.f)/(D/2.f)*1.5f,cz=(float(z)-D/2.f)/(D/2.f)*1.5f;
   float a=0,b=0,c=0,val=1;
   for(unsigned it=0;it<iterations;++it) {
    float r=std::sqrt(a*a+b*b+c*c);
    if(!std::isfinite(r)||r>2){val=float(it)/iterations;break;}
    if(r==0){a=cx;b=cy;c=cz;continue;}
    float th=std::acos(std::clamp(c/r,-1.f,1.f))*power;
    float ph=std::atan2(b,a)*power;
    float rn=std::pow(r,power);
    a=rn*std::sin(th)*std::cos(ph)+cx;
    b=rn*std::sin(th)*std::sin(ph)+cy;
    c=rn*std::cos(th)+cz;
    if(!std::isfinite(a)||!std::isfinite(b)||!std::isfinite(c)||a*a+b*b+c*c>4){val=float(it)/iterations;break;}
   }
   volume[at(z,y,x)]=val;
  }
 }
 void permeate(unsigned steps,float dt){
  for(unsigned st=0;st<steps;++st){
   auto old=volume;
   for(unsigned z=1;z+1<D;++z)for(unsigned y=1;y+1<D;++y)for(unsigned x=1;x+1<D;++x){
    size_t i=at(z,y,x),p=size_t(D)*D;float c=old[i];
    float gx=old[i+1]-old[i-1],gy=old[i+D]-old[i-D],gz=old[i+p]-old[i-p];
    float permeability=1/(1+8*std::sqrt(gx*gx+gy*gy+gz*gz));
    float flux=permeability*(old[i+1]+old[i-1]+old[i+D]+old[i-D]+old[i+p]+old[i-p]-6*c);
    volume[i]=std::clamp(c+dt*flux,0.f,1.f);
   }
  }
 }
 static float weight(size_t i,size_t j){
  uint64_t h=uint64_t(i)*1000003ULL+uint64_t(j)*9176ULL+8000ULL;
  h^=h>>33;h*=0xff51afd7ed558ccdULL;h^=h>>33;
  return float((double(h>>11)*(1.0/9007199254740992.0)*2-1)*0.02);
 }
 void encode(unsigned width) {
  unsigned P=D/4;std::vector<float> pooled;pooled.reserve(size_t(P)*P*P);
  for(unsigned z=0;z<P;++z)for(unsigned y=0;y<P;++y)for(unsigned x=0;x<P;++x){
   float m=0;for(unsigned dz=0;dz<4;++dz)for(unsigned dy=0;dy<4;++dy)for(unsigned dx=0;dx<4;++dx)
    m=std::max(m,volume[at(z*4+dz,y*4+dy,x*4+dx)]);
   pooled.push_back(m);
  }
  prior_latent=latent;latent.assign(width,0);
  for(size_t i=0;i<width;++i){double a=0;for(size_t j=0;j<pooled.size();++j)a+=pooled[j]*weight(i,j);latent[i]=std::max(0.f,float(a));}
  latent_mean=0;for(float v:latent)latent_mean+=v;latent_mean/=width;
 }
 void recoil_latent(float k){
  if(prior_latent.size()!=latent.size())return;
  latent_mean=0;for(size_t i=0;i<latent.size();++i){latent[i]+=k*(prior_latent[i]-latent[i]);latent_mean+=latent[i];}latent_mean/=latent.size();
 }
 void decode() {
  unsigned P=D/4;std::vector<float> pooled(size_t(P)*P*P);
  for(size_t j=0;j<pooled.size();++j){double a=0;for(size_t i=0;i<latent.size();++i)a+=latent[i]*weight(i,j);pooled[j]=float(1/(1+std::exp(-a)));}
  decoded.resize(count());
  for(unsigned z=0;z<D;++z)for(unsigned y=0;y<D;++y)for(unsigned x=0;x<D;++x)
   decoded[at(z,y,x)]=pooled[(size_t(z/4)*P+y/4)*P+x/4];
 }
 void recoil_voxel(){
  if(velocity.size()!=decoded.size())velocity.assign(decoded.size(),0);
  for(size_t i=0;i<decoded.size();++i){
   float delta=decoded[i]-volume[i];velocity[i]+=(-6*delta-velocity[i])*0.02f;
   decoded[i]=std::clamp(decoded[i]+velocity[i]*0.02f,0.f,1.f);
  }
 }
 void mse(){double a=0;for(size_t i=0;i<count();++i){double v=double(decoded[i])-volume[i];a+=v*v;}mse_value=float(a/count());}
 void evolve(){power=std::clamp(8.f+2*latent_mean,2.f,16.f);}
 void blend(float alpha){
  ++level;if(level==levels)return;
  auto base=decoded;mandelbulb();
  for(size_t i=0;i<count();++i)volume[i]=std::clamp(alpha*base[i]+(1-alpha)*volume[i],0.f,1.f);
 }
 void run(){
  bool halted=false;
  while(!halted){
   if(pc>=program.size()||cycles>uint64_t(levels)*11+16)throw std::runtime_error("invalid PC or cycle limit");
   unsigned oldpc=pc,oldlevel=level;auto in=program[pc];uint64_t before=state_hash();
   auto t0=std::chrono::steady_clock::now();
   switch(in.op){
    case MANDELBULB:if(level==0)mandelbulb();break;
    case PERMEATE:permeate(in.c,in.f);break;
    case ENCODE:encode(in.c);break;
    case RECOIL:if(in.a==1)recoil_latent(in.f);else recoil_voxel();break;
    case DECODE:decode();break;
    case MSE:mse();break;
    case EVOLVE:evolve();break;
    case BLEND:blend(in.f);break;
    case JUMP_IF:if(level<levels){if(in.target>=program.size())throw std::runtime_error("bad jump target");pc=in.target;}else ++pc;break;
    case HALT:halted=true;break;
    default:throw std::runtime_error("unsupported bytecode");
   }
   if(in.op!=JUMP_IF)++pc;
   auto t1=std::chrono::steady_clock::now();
   trace.push_back({oldlevel,oldpc,in.op,uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count()),before,state_hash(),mse_value,power});
   ++cycles;
  }
 }
 bool save(const std::string& prefix)const{
  std::ofstream v(prefix+".jx3v",std::ios::binary),b(prefix+".jxbc",std::ios::binary),t(prefix+".csv");
  if(!v||!b||!t)return false;
  v.write("JX3V",4);u32(v,1);u32(v,D);u32(v,D);u32(v,D);u32(v,uint32_t(decoded.size()));
  for(float x:decoded)u32(v,bits(x));
  b.write("JXBC",4);u32(b,1);u32(b,uint32_t(program.size()));
  for(auto in:program){b.put(char(in.op));b.put(char(in.a));b.put(char(in.b));b.put(char(in.c));u32(b,bits(in.f));u32(b,in.target);}
  t<<"level,pc,opcode,duration_ns,mse,power,input_hash,output_hash\n"<<std::setprecision(9);
  for(auto r:trace)t<<r.level<<','<<r.pc<<','<<r.op<<','<<r.ns<<','<<r.mse<<','<<r.power<<','<<r.in_hash<<','<<r.out_hash<<'\n';
  return bool(v)&&bool(b)&&bool(t);
 }
};
}
int main(int argc,char**argv){
 try{
  unsigned d=32,l=5,n=32;std::string prefix="fractal_chain";
  for(int i=1;i<argc;++i){
   std::string key=argv[i];
   if(i+1>=argc)throw std::invalid_argument("missing option value");
   std::string value=argv[++i];
   if(key=="--dim")d=std::stoul(value);
   else if(key=="--levels")l=std::stoul(value);
   else if(key=="--iterations")n=std::stoul(value);
   else if(key=="--prefix")prefix=value;
   else throw std::invalid_argument("unknown option");
  }
  jarvisx::Runtime r(d,l,n);r.run();
  if(!r.save(prefix))throw std::runtime_error("file write failed");
  std::cout<<"JX3D_CHAIN_OK levels="<<r.level<<" cycles="<<r.cycles<<" mse="<<r.mse_value
    <<" trace="<<r.trace.size()<<" volume_hash="<<r.state_hash()<<'\n';
  return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}