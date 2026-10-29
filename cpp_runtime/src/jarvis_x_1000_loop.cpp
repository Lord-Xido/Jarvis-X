// JARVIS X Unified 3D Engine v2 -- C++17, standalone, CPU reference.
// Combines original lossless reversible block codec, geometric emitter,
// and a TRAINABLE sparse-patch 3D convolutional autoencoder.
// Geometric coordinates are lossy visualisations; latents reconstruct via decoder.
#define main jarvis_legacy_main
// JARVIS X 3D sparse reversible auto-codec, C++17.
// A functional geometric codec, not a pretrained neural network.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
static constexpr uint32_t SIDE=1000000;
static void put32(ostream& o,uint32_t x){for(int i=0;i<4;i++)o.put(char(x>>(8*i)));}
static uint32_t get32(istream& in){uint32_t v=0;for(int i=0;i<4;i++){int c=in.get();if(c<0)throw runtime_error("truncated input");v|=uint32_t(c)<<(8*i);}return v;}
static void put64(ostream& o,uint64_t x){for(int i=0;i<8;i++)o.put(char(x>>(8*i)));}
static uint64_t get64(istream& in){uint64_t v=0;for(int i=0;i<8;i++){int c=in.get();if(c<0)throw runtime_error("truncated input");v|=uint64_t(c)<<(8*i);}return v;}
static uint64_t hash64(const vector<uint8_t>& a){uint64_t h=14695981039346656037ull;for(auto c:a){h^=c;h*=1099511628211ull;}return h;}
struct Point{uint32_t x,y,z;};
// Bijective mapping from 1D byte index to the 10^18-cell virtual 3D cube.
static Point position(uint64_t i){return {uint32_t(i%SIDE),uint32_t((i/SIDE)%SIDE),uint32_t(i/(uint64_t(SIDE)*SIDE))};}
static uint64_t index(Point p){return uint64_t(p.x)+uint64_t(SIDE)*p.y+uint64_t(SIDE)*SIDE*p.z;}
// Reversible integer lifting on groups of eight samples: 3D 2x2x2 Haar-like transform.
// Arithmetic is modulo 256, guaranteeing exact reconstruction of arbitrary byte streams.
static void transform(uint8_t* a,bool inverse){
 if(!inverse){for(int axis=0;axis<3;axis++)for(int j=0;j<8;j++)if(!(j&(1<<axis))){auto k=j|(1<<axis);a[k]=uint8_t(a[k]-a[j]);a[j]=uint8_t(a[j]+a[k]);}}
 else{for(int axis=2;axis>=0;axis--)for(int j=0;j<8;j++)if(!(j&(1<<axis))){auto k=j|(1<<axis);a[j]=uint8_t(a[j]-a[k]);a[k]=uint8_t(a[k]+a[j]);}}
}
static vector<uint8_t> readfile(const string& path){ifstream f(path,ios::binary);if(!f)throw runtime_error("cannot open "+path);return vector<uint8_t>(istreambuf_iterator<char>(f),{});}
static void writefile(const string& path,const vector<uint8_t>& v){ofstream f(path,ios::binary);if(!f)throw runtime_error("cannot write "+path);if(!v.empty())f.write((const char*)v.data(),v.size());if(!f)throw runtime_error("write failed");}
static void encode(const string& src,const string& dst){auto data=readfile(src);if(data.size()>uint64_t(SIDE)*SIDE*SIDE)throw runtime_error("virtual cube overflow");ofstream o(dst,ios::binary);if(!o)throw runtime_error("cannot open output");o.write("JX3D",4);put32(o,1);put64(o,data.size());put64(o,hash64(data));uint64_t blocks=(data.size()+7)/8;put64(o,blocks);uint64_t raw=0,packed=0;
 for(uint64_t b=0;b<blocks;b++){uint8_t v[8]={};size_t n=min(uint64_t(8),uint64_t(data.size())-8*b);copy_n(data.data()+8*b,n,v);uint8_t t[8];copy_n(v,8,t);transform(t,false);int z=0;for(auto c:t)z+=(c==0);bool use=z>=3;o.put(use?1:0);if(use){uint8_t mask=0;for(int j=0;j<8;j++)if(t[j])mask|=uint8_t(1<<j);o.put(char(mask));for(int j=0;j<8;j++)if(mask&(1<<j))o.put(char(t[j]));packed++;}else{o.write((char*)v,8);raw++;}}
 if(!o)throw runtime_error("encode write failed");
 cout<<"encoded "<<data.size()<<" bytes; blocks="<<blocks<<" transformed="<<packed<<" raw="<<raw<<"\n";
}
static void decode(const string& src,const string& dst){ifstream in(src,ios::binary);if(!in)throw runtime_error("cannot open input");char magic[4];if(!in.read(magic,4)||string(magic,4)!="JX3D")throw runtime_error("bad magic");if(get32(in)!=1)throw runtime_error("unsupported version");uint64_t size=get64(in),expected=get64(in),blocks=get64(in);if(size>uint64_t(SIDE)*SIDE*SIDE||blocks!=(size+7)/8)throw runtime_error("invalid header");if(size>uint64_t(vector<uint8_t>().max_size()))throw runtime_error("output too large");vector<uint8_t> out;out.reserve(size);for(uint64_t b=0;b<blocks;b++){int flag=in.get();if(flag<0||flag>1)throw runtime_error("invalid/truncated block");uint8_t v[8]={};if(flag){int mask=in.get();if(mask<0)throw runtime_error("truncated mask");for(int j=0;j<8;j++)if(mask&(1<<j)){int c=in.get();if(c<0)throw runtime_error("truncated payload");v[j]=uint8_t(c);}transform(v,true);}else{if(!in.read((char*)v,8))throw runtime_error("truncated raw block");}size_t n=min(uint64_t(8),size-8*b);out.insert(out.end(),v,v+n);}if(in.peek()!=EOF)throw runtime_error("unexpected trailing data");if(hash64(out)!=expected)throw runtime_error("integrity check failed");writefile(dst,out);cout<<"decoded and verified "<<out.size()<<" bytes\n";}
static void selftest(){for(int n=0;n<256;n++){uint8_t a[8],b[8];for(int j=0;j<8;j++)a[j]=uint8_t(n*29+j*37);copy_n(a,8,b);transform(b,false);transform(b,true);if(!equal(a,a+8,b))throw runtime_error("transform failed");}for(uint64_t i: {0ull,1ull,999999ull,1000000ull,999999999999999999ull}){if(index(position(i))!=i)throw runtime_error("coordinate mapping failed");}string in="jx3d_test_input.bin",enc="jx3d_test_encoded.jx3d",out="jx3d_test_output.bin";vector<uint8_t> v(131073);uint32_t s=123456789;for(size_t i=0;i<v.size();i++){s^=s<<13;s^=s>>17;s^=s<<5;v[i]=(i%31<22)?uint8_t(i%4):uint8_t(s);}writefile(in,v);encode(in,enc);decode(enc,out);if(readfile(out)!=v)throw runtime_error("roundtrip failed");remove(in.c_str());remove(enc.c_str());remove(out.c_str());cout<<"PASS: 3D coordinate bijection, reversible transform, file roundtrip, integrity\n";}
int main(int argc,char** argv){try{if(argc==2&&string(argv[1])=="selftest"){selftest();return 0;}if(argc==4){string mode=argv[1];if(mode=="encode"){encode(argv[2],argv[3]);return 0;}if(mode=="decode"){decode(argv[2],argv[3]);return 0;}}cerr<<"Usage: jarvis_x_3d selftest | encode INPUT OUTPUT.jx3d | decode INPUT.jx3d OUTPUT\n";return 2;}catch(const exception& e){cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}

#undef main
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace emission {
constexpr double PI=3.14159265358979323846; constexpr uint32_t SIDE=1000000;
using V4=std::array<double,4>;
uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
double uni(uint64_t s){return double(mix(s)>>11)*0x1.0p-53;}
uint64_t morton(uint32_t x,uint32_t y,uint32_t z){uint64_t v=0;for(int b=0;b<20;b++){v|=uint64_t((x>>b)&1)<<(3*b);v|=uint64_t((y>>b)&1)<<(3*b+1);v|=uint64_t((z>>b)&1)<<(3*b+2);}return v;}
std::array<uint32_t,3> unmorton(uint64_t v){std::array<uint32_t,3> a{};for(int b=0;b<20;b++)for(int k=0;k<3;k++)a[k]|=uint32_t((v>>(3*b+k))&1)<<b;return a;}
uint32_t coord(double v){return uint32_t(std::llround(std::clamp((v+10)/20.,0.,1.)*(SIDE-1)));}
V4 encode(V4 x){V4 z{};double sum=x[0]+x[1]+x[2]+x[3];for(int i=0;i<4;i++)z[i]=.8*x[i]+.05*sum;return z;}
V4 decode(V4 z){V4 x{};double sum=z[0]+z[1]+z[2]+z[3];for(int i=0;i<4;i++)x[i]=1.25*z[i]-.0625*sum;return x;}
void check(){for(auto a: {std::array<uint32_t,3>{0,0,0},{1,2,3},{SIDE-1,SIDE-1,SIDE-1},{123456,654321,111111}})if(unmorton(morton(a[0],a[1],a[2]))!=a)throw std::runtime_error("Morton inverse");for(int j=0;j<1000;j++){V4 x{uni(j*4),uni(j*4+1),uni(j*4+2),uni(j*4+3)};auto y=decode(encode(x));for(int k=0;k<4;k++)if(std::abs(x[k]-y[k])>1e-12)throw std::runtime_error("linear inverse");}}
int run(int argc,char**argv){if(argc!=5)throw std::runtime_error("emit TICKS VOXELS SAMPLES OUTPUT.csv");int ticks=std::stoi(argv[1]),vox=std::stoi(argv[2]),samples=std::stoi(argv[3]);if(ticks<=0||vox<=0||samples<=0||uint64_t(ticks)*vox*samples>10000000ULL)throw std::runtime_error("rows must be 1..10000000");check();std::ofstream out(argv[4]);if(!out)throw std::runtime_error("output open failed");out<<"tick,voxel,sample,gx,gy,gz,morton,px,py,pz,trust,error,input0,input1,input2,input3,latent0,latent1,latent2,latent3\n"<<std::setprecision(17);std::vector<double> trust(vox,1.);double maxerr=0;auto start=std::chrono::steady_clock::now();for(int k=0;k<ticks;k++)for(int n=0;n<vox;n++)for(int j=0;j<samples;j++){uint64_t seed=mix(uint64_t(k+1)*0x9e3779b97f4a7c15ULL^uint64_t(n+1)*0xbf58476d1ce4e5b9ULL^uint64_t(j+1)*0x94d049bb133111ebULL);V4 x{uni(seed),uni(seed+1),uni(seed+2),uni(seed+3)};auto z=encode(x),xr=decode(z);double err=0,dist=0;for(int t=0;t<4;t++){err=std::max(err,std::abs(x[t]-xr[t]));dist+=(x[t]-.5)*(x[t]-.5);}maxerr=std::max(err,maxerr);double s=2*PI*z[0],t=2*PI*z[1],alpha=PI/2*z[2],rho=z[3];double px=10*rho*std::sin(2*alpha)*std::cos(s-t),py=10*rho*std::sin(2*alpha)*std::sin(s-t),pz=10*rho*std::cos(2*alpha);auto gx=coord(px),gy=coord(py),gz=coord(pz);trust[n]=std::clamp(trust[n]*.999+.001/(1+std::sqrt(dist)/2+10*std::exp(-k/2000.)),0.,1.);out<<k<<','<<n<<','<<j<<','<<gx<<','<<gy<<','<<gz<<','<<morton(gx,gy,gz)<<','<<px<<','<<py<<','<<pz<<','<<trust[n]<<','<<err;for(double v:x)out<<','<<v;for(double v:z)out<<','<<v;out<<'\n';}out.flush();if(!out)throw std::runtime_error("write failure");double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::cout<<"PASS: "<<uint64_t(ticks)*vox*samples<<" rows, "<<seconds<<" s, "<<uint64_t(ticks)*vox*samples/seconds<<" rows/sec; max reconstruction error "<<maxerr<<"\n";return 0;}
}


#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <sstream>
#include <limits>
#include <utility>

namespace neural {
namespace fs=std::filesystem;
constexpr int N=8, H=16, L=3;
constexpr int K=0, W1=4, B1=W1+H*N, W2=B1+H, B2=W2+L*H;
constexpr int W3=B2+L, B3=W3+H*L, W4=B3+H, B4=W4+N*H;
constexpr int PARAMS=B4+N;
static_assert(PARAMS==399,"unexpected trainable parameter count");
constexpr size_t MAX_BYTES=32u*1024u*1024u; // bounded-reference neural loader
constexpr double PI2=3.141592653589793238462643383279502884;

uint64_t mixer(uint64_t x){return emission::mix(x);}
void ensure(bool p,const std::string& s){if(!p)throw std::runtime_error(s);}
int pos_int(const std::string& s,int limit){
 if(s.empty()||!std::all_of(s.begin(),s.end(),[](unsigned char c){return std::isdigit(c)!=0;}))throw std::invalid_argument("expected positive integer: "+s);
 unsigned long long n=std::stoull(s); if(n<1||n>static_cast<unsigned long long>(limit))throw std::out_of_range("integer outside range 1.."+std::to_string(limit));return int(n);
}
std::vector<uint8_t> load(const std::string& path){
 fs::path p(path);ensure(fs::is_regular_file(p),"not an input file: "+path);
 ensure(fs::file_size(p)<=MAX_BYTES,"neural reference input exceeds 32 MiB safety cap");
 return ::readfile(path);
}
void real64(std::ostream& o,double v){uint64_t u=0;static_assert(sizeof u==sizeof v,"double must be 64-bit");std::memcpy(&u,&v,8);::put64(o,u);}
double un64(std::istream& in){uint64_t u=::get64(in);double v;std::memcpy(&v,&u,8);ensure(std::isfinite(v)&&std::abs(v)<1e6,"non-finite or unreasonable parameter");return v;}
void fl32(std::ostream& o,float v){uint32_t u;static_assert(sizeof v==sizeof u,"float must be 32-bit");std::memcpy(&u,&v,4);::put32(o,u);}
float un32(std::istream& in){uint32_t u=::get32(in);float v;std::memcpy(&v,&u,4);ensure(std::isfinite(v)&&std::abs(v)<=1.01f,"invalid latent value");return v;}
uint64_t fingerprint(const std::vector<double>& a){
 uint64_t h=14695981039346656037ULL;
 for(double value:a){uint64_t bits;std::memcpy(&bits,&value,8);for(int j=0;j<8;j++){h^=uint8_t(bits>>(j*8));h*=1099511628211ULL;}}
 return h;
}
double sigmoid(double z){return z>=0 ? 1.0/(1.0+std::exp(-z)) : std::exp(z)/(1.0+std::exp(z));}
struct Example{std::array<double,N> x{};int valid=N;};
Example example(const std::vector<uint8_t>& bytes,size_t block){
 Example q;size_t off=block*N;
 q.valid=static_cast<int>(std::min(size_t(N),bytes.size()-off));
 for(int i=0;i<q.valid;i++)q.x[i]=bytes[off+i]/255.;
 return q;
}
struct State{
 std::array<double,N> u{},y{};
 std::array<double,H> h1{},h3{};
 std::array<double,L> z{};
};
struct Model{
 std::vector<double> p,m,v;
 uint64_t step=0;
 Model(uint64_t seed=42):p(PARAMS,0),m(PARAMS,0),v(PARAMS,0){
  std::mt19937_64 rng(seed);
  p[K]=0.85;p[K+1]=p[K+2]=p[K+3]=0.05;
  auto init=[&](int start,int rows,int cols){double range=std::sqrt(6.0/(rows+cols));std::uniform_real_distribution<double> dist(-range,range);for(int j=0;j<rows*cols;j++)p[start+j]=dist(rng);};
  init(W1,H,N);init(W2,L,H);init(W3,H,L);init(W4,N,H);
 }
 State forward(const Example& a) const{
  State f;
  // 3D 2x2x2 patch: one trainable centre and three axis-neighbour kernels.
  // XOR toggles the x/y/z coordinate bit within the patch.
  for(int i=0;i<N;i++){
   f.u[i]=p[K]*a.x[i]+p[K+1]*a.x[i^1]+p[K+2]*a.x[i^2]+p[K+3]*a.x[i^4];
  }
  for(int j=0;j<H;j++){
   double s=p[B1+j];for(int i=0;i<N;i++)s+=p[W1+j*N+i]*f.u[i];f.h1[j]=std::tanh(s);
  }
  for(int j=0;j<L;j++){
   double s=p[B2+j];for(int i=0;i<H;i++)s+=p[W2+j*H+i]*f.h1[i];f.z[j]=std::tanh(s);
  }
  return decode(f.z,f);
 }
 State decode(const std::array<double,L>& z,State f={}) const{
  f.z=z;
  for(int j=0;j<H;j++){
   double s=p[B3+j];for(int i=0;i<L;i++)s+=p[W3+j*L+i]*z[i];f.h3[j]=std::tanh(s);
  }
  for(int j=0;j<N;j++){
   double s=p[B4+j];for(int i=0;i<H;i++)s+=p[W4+j*H+i]*f.h3[i];f.y[j]=sigmoid(s);
  }
  return f;
 }
 double loss(const Example& a)const{
  auto f=forward(a);double ss=0;for(int i=0;i<a.valid;i++)ss+=(f.y[i]-a.x[i])*(f.y[i]-a.x[i]);return ss/a.valid;
 }
 double grad(const Example& a,std::vector<double>& g)const{
  auto f=forward(a);
  std::array<double,N> d4{},du{};
  std::array<double,H> d3{},d1{};
  std::array<double,L> d2{};
  double loss=0.;
  for(int j=0;j<a.valid;j++){
   double diff=f.y[j]-a.x[j];loss+=diff*diff;
   d4[j]=2.*diff/a.valid*f.y[j]*(1.-f.y[j]);
   g[B4+j]+=d4[j];
   for(int i=0;i<H;i++)g[W4+j*H+i]+=d4[j]*f.h3[i];
  }
  for(int i=0;i<H;i++){
   double s=0;for(int j=0;j<N;j++)s+=d4[j]*p[W4+j*H+i];
   d3[i]=s*(1.-f.h3[i]*f.h3[i]);g[B3+i]+=d3[i];
   for(int j=0;j<L;j++)g[W3+i*L+j]+=d3[i]*f.z[j];
  }
  for(int i=0;i<L;i++){
   double s=0;for(int j=0;j<H;j++)s+=d3[j]*p[W3+j*L+i];
   d2[i]=s*(1.-f.z[i]*f.z[i]);g[B2+i]+=d2[i];
   for(int j=0;j<H;j++)g[W2+i*H+j]+=d2[i]*f.h1[j];
  }
  for(int i=0;i<H;i++){
   double s=0;for(int j=0;j<L;j++)s+=d2[j]*p[W2+j*H+i];
   d1[i]=s*(1.-f.h1[i]*f.h1[i]);g[B1+i]+=d1[i];
   for(int j=0;j<N;j++)g[W1+i*N+j]+=d1[i]*f.u[j];
  }
  for(int i=0;i<N;i++){
   double s=0;for(int j=0;j<H;j++)s+=d1[j]*p[W1+j*N+i];du[i]=s;
   g[K]+=s*a.x[i];g[K+1]+=s*a.x[i^1];g[K+2]+=s*a.x[i^2];g[K+3]+=s*a.x[i^4];
  }
  return loss/a.valid;
 }
 void adam(std::vector<double>& g,int count,double lr){
  ensure(count>0,"empty minibatch");
  double norm=0.;for(double v0:g){double t=v0/count;norm+=t*t;}
  double scale=std::min(1.,5./(std::sqrt(norm)+1e-12));
  step++;
  double b1corr=1.-std::pow(0.9,static_cast<double>(step));
  double b2corr=1.-std::pow(0.999,static_cast<double>(step));
  for(int i=0;i<PARAMS;i++){
   double d=g[i]*scale/count;
   m[i]=.9*m[i]+.1*d;v[i]=.999*v[i]+.001*d*d;
   p[i]-=lr*(m[i]/b1corr/(std::sqrt(v[i]/b2corr)+1e-8)+1e-5*p[i]);
   ensure(std::isfinite(p[i])&&std::abs(p[i])<1e6,"unstable training parameters");
  }
  std::fill(g.begin(),g.end(),0.);
 }
};
void save_model(const Model& a,const std::string& path){
 std::ofstream out(path,std::ios::binary|std::ios::trunc);ensure(bool(out),"cannot write model");
 out.write("JXNM",4);::put32(out,1);::put32(out,PARAMS);::put64(out,a.step);::put64(out,fingerprint(a.p));
 for(const auto* v:{&a.p,&a.m,&a.v})for(double t:*v)real64(out,t);
 ensure(bool(out),"model write failed");
}
Model read_model(const std::string& path){
 std::ifstream in(path,std::ios::binary);ensure(bool(in),"cannot read model: "+path);
 char magic[4];ensure(bool(in.read(magic,4))&&std::string(magic,4)=="JXNM","invalid model magic");
 ensure(::get32(in)==1&&::get32(in)==PARAMS,"incompatible model architecture");
 Model a; a.step=::get64(in);ensure(a.step<1000000000ULL,"invalid optimiser step");uint64_t hash=::get64(in);
 for(auto* v:{&a.p,&a.m,&a.v})for(double& t:*v)t=un64(in);
 ensure(in.peek()==EOF,"unexpected bytes in model");ensure(fingerprint(a.p)==hash,"model checksum mismatch");
 return a;
}
size_t nblocks(const std::vector<uint8_t>& a){return (a.size()+N-1)/N;}
struct TrainResult{double initial,final_loss,train_loss;int best_epoch,train_count,val_count;};
TrainResult train(Model& model,const std::vector<uint8_t>& bytes,int epochs,size_t limit,bool verbose){
 ensure(!bytes.empty(),"training input must contain data");
 size_t count=std::min(nblocks(bytes),limit);ensure(count>=10,"training requires at least ten 2x2x2 patches (80 bytes)");
 std::vector<size_t> training,validation;
 for(size_t i=0;i<count;i++)(((mixer(i+42)%5)==0)?validation:training).push_back(i);
 ensure(!training.empty()&&!validation.empty(),"training/validation partition empty");
 auto mean=[&](const std::vector<size_t>& ids){double ss=0;for(size_t i:ids)ss+=model.loss(example(bytes,i));return ss/ids.size();};
 double initial=mean(validation),best=initial,best_train=mean(training);
 Model best_model=model;int best_epoch=0;
 std::mt19937 rng(42);
 for(int e=1;e<=epochs;e++){
  std::shuffle(training.begin(),training.end(),rng);
  std::vector<double> g(PARAMS,0.);int batch=0;
  for(size_t idx:training){model.grad(example(bytes,idx),g);batch++;
   if(batch==32){model.adam(g,batch,0.006);batch=0;}}
  if(batch)model.adam(g,batch,0.006);
  double val=mean(validation),tr=mean(training);
  if(val<best){best=val;best_model=model;best_epoch=e;best_train=tr;}
  if(verbose&&(e==1||e==epochs||e%10==0))std::cout<<"epoch="<<e<<" train_mse="<<std::setprecision(8)<<tr<<" val_mse="<<val<<"\n";
 }
 model=best_model;
 return {initial,best,best_train,best_epoch,int(training.size()),int(validation.size())};
}
void latent_encode(const std::string& input,const Model& model,const std::string& path){
 auto bytes=load(input);
 std::ofstream out(path,std::ios::binary|std::ios::trunc);ensure(bool(out),"cannot write latent stream");
 out.write("JXLT",4);::put32(out,1);::put64(out,bytes.size());::put64(out,nblocks(bytes));::put64(out,fingerprint(model.p));
 for(size_t i=0;i<nblocks(bytes);i++){
  auto z=model.forward(example(bytes,i)).z;
  for(double value:z)fl32(out,float(value));
 }
 ensure(bool(out),"latent write failed");
}
struct LatentFile{uint64_t nbytes;std::vector<std::array<double,L>> z;};
LatentFile load_latent(const std::string& path,const Model& model){
 std::ifstream in(path,std::ios::binary);ensure(bool(in),"cannot open latent file");char magic[4];
 ensure(bool(in.read(magic,4))&&std::string(magic,4)=="JXLT","invalid latent magic");
 ensure(::get32(in)==1,"invalid latent version");uint64_t nb=::get64(in),blocks=::get64(in),hash=::get64(in);
 ensure(hash==fingerprint(model.p),"latent/model fingerprint mismatch");
 ensure(nb<=MAX_BYTES&&blocks==(nb+7)/8,"invalid latent length or safety cap");
 LatentFile result{nb,{}};result.z.resize(static_cast<size_t>(blocks));
 for(auto& item:result.z)for(double& t:item)t=un32(in);
 ensure(in.peek()==EOF,"trailing latent data");return result;
}
uint8_t byte(double v){return uint8_t(std::clamp(std::lround(v*255.0),0L,255L));}
std::vector<uint8_t> reconstruct(const LatentFile& lat,const Model& model){
 std::vector<uint8_t> out(static_cast<size_t>(lat.nbytes));
 for(size_t b=0;b<lat.z.size();b++){
  auto d=model.decode(lat.z[b]);for(int i=0;i<N&&b*8+i<out.size();i++)out[b*8+i]=byte(d.y[i]);
 }
 return out;
}
void latent_decode(const std::string& latfile,const Model& model,const std::string& output){
 auto lat=load_latent(latfile,model);::writefile(output,reconstruct(lat,model));
}
uint32_t quant(double x){return uint32_t(std::llround(std::clamp((x+10.)/20.,0.,1.)*(emission::SIDE-1)));}
struct Geo{double px,py,pz,hx,hy,hz;uint32_t gx,gy,gz;uint64_t morton;};
Geo geometry(const std::array<double,L>& z){
 // Volumetric learned latent geometry, separate from Hopf sphere visualisation.
 double px=10*z[0],py=10*z[1],pz=10*z[2];
 double s=PI2*(z[0]+1),t=PI2*(z[1]+1),alpha=(PI2/4)*(z[2]+1);
 double hx=10*std::sin(2*alpha)*std::cos(s-t);
 double hy=10*std::sin(2*alpha)*std::sin(s-t);
 double hz=10*std::cos(2*alpha);
 uint32_t gx=quant(px),gy=quant(py),gz=quant(pz);
 return {px,py,pz,hx,hy,hz,gx,gy,gz,emission::morton(gx,gy,gz)};
}
void geometry_csv(const std::string& latfile,const Model& model,const std::string& path){
 auto lat=load_latent(latfile,model);
 std::ofstream out(path,std::ios::trunc);ensure(bool(out),"cannot write geometry CSV");
 out<<"block,source_x,source_y,source_z,latent0,latent1,latent2,px,py,pz,hopf_x,hopf_y,hopf_z,gx,gy,gz,morton\n"<<std::setprecision(17);
 for(size_t i=0;i<lat.z.size();i++){
  auto g=geometry(lat.z[i]);auto src=::position(i);out<<i<<','<<src.x<<','<<src.y<<','<<src.z<<','<<lat.z[i][0]<<','<<lat.z[i][1]<<','<<lat.z[i][2]<<','
  <<g.px<<','<<g.py<<','<<g.pz<<','<<g.hx<<','<<g.hy<<','<<g.hz<<','
  <<g.gx<<','<<g.gy<<','<<g.gz<<','<<g.morton<<'\n';
 }
 ensure(bool(out),"geometry CSV write failed");
}
struct Metrics{uint64_t count;double mae,mse,psnr;};
Metrics compare(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){
 ensure(a.size()==b.size(),"reconstructed length mismatch");
 long double s=0,sa=0;for(size_t i=0;i<a.size();i++){long double d=int(a[i])-int(b[i]);s+=d*d;sa+=std::abs(d);}
 if(a.empty())return {0,0,0,std::numeric_limits<double>::infinity()};
 double mse=double(s/a.size()),mae=double(sa/a.size());
 return {a.size(),mae,mse,mse==0?std::numeric_limits<double>::infinity():10*std::log10(255.*255./mse)};
}
void write_report(const std::string& path,const TrainResult& tr,const Metrics& met,const fs::path& outdir,double seconds){
 std::ofstream o(path);ensure(bool(o),"cannot write report");o<<std::setprecision(12);
 o<<"{\n  \"system\": \"JARVIS X Unified 3D Engine v2\",\n"
  <<"  \"scope\": \"lossless binary codec plus trained lossy neural latent codec\",\n"
  <<"  \"virtual_axis_length\": 1000000,\n  \"virtual_cells\": \"1000000000000000000\",\n"
  <<"  \"trained_3d_patch_convolution\": true,\n  \"latent_dimensions\": 3,\n"
  <<"  \"parameters\": "<<PARAMS<<",\n  \"training_blocks\": "<<tr.train_count<<",\n"
  <<"  \"validation_blocks\": "<<tr.val_count<<",\n  \"best_epoch\": "<<tr.best_epoch<<",\n"
  <<"  \"initial_validation_mse_normalized\": "<<tr.initial<<",\n"
  <<"  \"final_validation_mse_normalized\": "<<tr.final_loss<<",\n"
  <<"  \"bytes_evaluated\": "<<met.count<<",\n  \"neural_mae_byte_units\": "<<met.mae<<",\n"
  <<"  \"neural_mse_byte_units\": "<<met.mse<<",\n  \"neural_psnr_db\": ";
 if(std::isfinite(met.psnr))o<<met.psnr;else o<<"null";
 o<<",\n  \"lossless_roundtrip_pass\": true,\n  \"elapsed_seconds\": "<<seconds<<",\n"
  <<"  \"outputs\": {\n    \"checkpoint\": \"model.jxnm\",\n    \"latent_stream\": \"latents.jxlt\",\n"
  <<"    \"neural_reconstruction\": \"neural_reconstruction.bin\",\n"
  <<"    \"geometric_emissions\": \"geometry.csv\",\n    \"lossless_archive\": \"original.jx3d\",\n"
  <<"    \"lossless_restoration\": \"lossless_restored.bin\"\n  }\n}\n";
 ensure(bool(o),"report write failed");
 (void)outdir;
}
void pipeline(const std::string& input,const std::string& output,int epochs,int limit){
 auto start=std::chrono::steady_clock::now();
 auto bytes=load(input);
 fs::create_directories(output);fs::path dir(output);
 ensure(fs::absolute(input).lexically_normal()!=fs::absolute(dir/"neural_reconstruction.bin").lexically_normal(),"input/output alias");
 Model model;auto tr=train(model,bytes,epochs,size_t(limit),true);
 save_model(model,(dir/"model.jxnm").string());
 auto restored_model=read_model((dir/"model.jxnm").string());
 latent_encode(input,restored_model,(dir/"latents.jxlt").string());
 latent_decode((dir/"latents.jxlt").string(),restored_model,(dir/"neural_reconstruction.bin").string());
 geometry_csv((dir/"latents.jxlt").string(),restored_model,(dir/"geometry.csv").string());
 // Original lossless codec is intentionally independent from lossy ANN.
 ::encode(input,(dir/"original.jx3d").string());
 ::decode((dir/"original.jx3d").string(),(dir/"lossless_restored.bin").string());
 ensure(::readfile((dir/"lossless_restored.bin").string())==bytes,"lossless round-trip failed");
 auto met=compare(bytes,::readfile((dir/"neural_reconstruction.bin").string()));
 double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 write_report((dir/"report.json").string(),tr,met,dir,seconds);
 std::cout<<std::fixed<<std::setprecision(8)
   <<"PIPELINE PASS: neural train -> latent encode -> latent decode -> geometry -> lossless encode/decode -> integrity\n"
   <<"initial validation MSE="<<tr.initial<<" final validation MSE="<<tr.final_loss<<" best_epoch="<<tr.best_epoch<<"\n"
   <<"neural reconstruction MAE="<<met.mae<<" bytes; MSE="<<met.mse<<" bytes^2; PSNR="<<met.psnr<<" dB\n"
   <<"lossless reconstruction: EXACT; wall time="<<seconds<<" seconds\n"
   <<"artifacts in "<<dir<<"\n";
}
void gradient_check(){
 Model model(99);Example a;for(int i=0;i<N;i++)a.x[i]=(i+1)/10.;
 std::vector<double> grads(PARAMS,0);model.grad(a,grads);const double eps=1e-5;
 for(int i:{K,K+1,K+2,K+3,W1,W1+12,B1,W2,W2+10,B2,W3,W3+5,B3,W4,W4+5,B4}){
  double old=model.p[i];model.p[i]=old+eps;double plus=model.loss(a);
  model.p[i]=old-eps;double minus=model.loss(a);model.p[i]=old;
  double numerical=(plus-minus)/(2*eps);
  double err=std::abs(grads[i]-numerical);
  ensure(err<1e-7,"backprop gradient check failed at parameter "+std::to_string(i));
 }
}
void selftest(){
 ::selftest();emission::check();gradient_check();
 std::vector<uint8_t> synthetic(8*512);
 for(size_t b=0;b<512;b++)for(int i=0;i<8;i++){
  int x=i&1,y=(i>>1)&1,z=(i>>2)&1;
  int t=int((b*19)%256);synthetic[b*8+i]=uint8_t((t+x*11+y*23+z*37)%256);
 }
 Model model; auto tr=train(model,synthetic,35,512,false);
 ensure(tr.final_loss<tr.initial,"validation reconstruction loss failed to improve");
 fs::path base=fs::temp_directory_path()/("jx_full_test_"+std::to_string(mixer(42)));
 fs::create_directories(base);
 try{
  auto file=(base/"input.bin").string(),mod=(base/"model.jxnm").string(),lat=(base/"lat.jxlt").string();
  ::writefile(file,synthetic);save_model(model,mod);
  auto m2=read_model(mod);ensure(fingerprint(m2.p)==fingerprint(model.p),"model persist error");
  latent_encode(file,m2,lat);auto l=load_latent(lat,m2);
  ensure(l.z.size()==512,"latent block count failed");
  auto rec=reconstruct(l,m2);ensure(rec.size()==synthetic.size(),"reconstruction length failed");
  auto g=geometry(l.z[0]);ensure(emission::unmorton(g.morton)==std::array<uint32_t,3>{g.gx,g.gy,g.gz},"3D Morton inverse failed");
  geometry_csv(lat,m2,(base/"geometry.csv").string());
  std::ifstream check((base/"geometry.csv").string());size_t lines=0;std::string s;while(std::getline(check,s))lines++;
  ensure(lines==513,"geometric record count failed");
  // Tampered models are not accepted.
  {std::fstream corrupt(mod,std::ios::binary|std::ios::in|std::ios::out);corrupt.seekp(32);char q=static_cast<char>(0xff);corrupt.write(&q,1);}
  bool rejected=false;try{read_model(mod);}catch(const std::exception&){rejected=true;}
  ensure(rejected,"corrupted model not rejected");
  fs::remove_all(base);
 }catch(...){fs::remove_all(base);throw;}
 std::cout<<"SELFTEST PASS: lossless codec; 3D Morton; Hopf; analytic inverse; finite-difference 3D-conv/ANN gradients; "
          <<"validation improvement; checkpoint; latent stream; neural decoding; CSV; corruption detection\n";
 std::cout<<"validation normalized MSE: "<<tr.initial<<" -> "<<tr.final_loss<<"\n";
}
void usage(){
 std::cerr<<"JARVIS X Unified 3D Engine v2\n"
 <<"  selftest\n"
 <<"  pipeline INPUT OUTPUT_DIR [EPOCHS=45] [MAX_TRAIN_BLOCKS=8192]\n"
 <<"  train INPUT MODEL.jxnm [EPOCHS=45] [MAX_TRAIN_BLOCKS=8192]\n"
 <<"  finetune INPUT EXISTING_MODEL.jxnm OUTPUT_MODEL.jxnm [EPOCHS=20] [MAX_TRAIN_BLOCKS=8192]\n"
 <<"  latent-encode INPUT MODEL.jxnm LATENTS.jxlt\n"
 <<"  latent-decode LATENTS.jxlt MODEL.jxnm RECONSTRUCTED.bin\n"
 <<"  geometry LATENTS.jxlt MODEL.jxnm POINTS.csv\n"
 <<"  encode INPUT OUTPUT.jx3d      # legacy lossless, exact\n"
 <<"  decode INPUT.jx3d OUTPUT      # legacy lossless, exact\n"
 <<"  emit TICKS VOXELS SAMPLES OUTPUT.csv    # 4D analytic demonstrator\n";
}
int dispatch(int argc,char**argv){
 if(argc<2){usage();return 2;}std::string cmd=argv[1];
 if(cmd=="selftest"&&argc==2){selftest();return 0;}
 if((cmd=="pipeline"||cmd=="train")&&(argc>=4&&argc<=6)){
  int epochs=argc>=5?pos_int(argv[4],1000):45;
  int limit=argc>=6?pos_int(argv[5],1000000):8192;
  if(cmd=="pipeline")pipeline(argv[2],argv[3],epochs,limit);
  else{
   auto bytes=load(argv[2]);Model m;auto t=train(m,bytes,epochs,limit,true);
   save_model(m,argv[3]);std::cout<<"TRAIN PASS: validation MSE "<<t.initial<<" -> "<<t.final_loss<<"; best epoch="<<t.best_epoch<<"\n";
  }return 0;
 }
 if(cmd=="finetune"&&argc>=5&&argc<=7){auto bytes=load(argv[2]);Model m=read_model(argv[3]);int epochs=argc>=6?pos_int(argv[5],1000):20;int limit=argc>=7?pos_int(argv[6],1000000):8192;auto t=train(m,bytes,epochs,limit,true);save_model(m,argv[4]);std::cout<<"FINETUNE PASS: validation MSE "<<t.initial<<" -> "<<t.final_loss<<"; best epoch="<<t.best_epoch<<"\n";return 0;}
 if(cmd=="latent-encode"&&argc==5){latent_encode(argv[2],read_model(argv[3]),argv[4]);std::cout<<"LATENT ENCODE PASS\n";return 0;}
 if(cmd=="latent-decode"&&argc==5){latent_decode(argv[2],read_model(argv[3]),argv[4]);std::cout<<"LATENT DECODE PASS (lossy)\n";return 0;}
 if(cmd=="geometry"&&argc==5){geometry_csv(argv[2],read_model(argv[3]),argv[4]);std::cout<<"3D GEOMETRY PASS\n";return 0;}
 if((cmd=="encode"||cmd=="decode")&&argc==4){if(cmd=="encode")::encode(argv[2],argv[3]);else ::decode(argv[2],argv[3]);return 0;}
 if(cmd=="emit"&&argc==6)return emission::run(argc-1,argv+1);
 usage();return 2;
}
} // namespace neural

// ===== JARVIS X 1000-iteration recursive 3D verification module =====
// This extension uses the ACTUAL trained model from the v2 engine above.
// No parameters are altered: this tests feedback dynamics, not retraining.
#include <unordered_map>
#include <unordered_set>
#include <ctime>

namespace recursive3d {
namespace fs=std::filesystem;
using A8=std::array<double,8>;
using B8=std::array<uint8_t,8>;
static constexpr int STEPS=1000;
static constexpr int SNAP[]={0,1,2,5,10,20,50,100,250,500,750,1000};
struct Diagnostics {
 double max_hopf_radius_error=0, max_grid_quant_error=0, max_axis_value=0;
 uint64_t coordinate_checks=0, lifting_checks=0, analytic_checks=0;
 int first_continuous_fixed=-1, first_quant_recurrence=-1, quant_period=-1;
 double initial_continuous_step_rmse=0, final_continuous_step_rmse=0;
 double final_continuous_origin_rmse=0, final_quant_origin_rmse=0;
 int final_changed_bytes=0, max_changed_bytes=0;
 double initial_jacobian_inf=0, final_jacobian_inf=0, conservative_global_lipschitz=0;
 int snapshot_rows=0;
 double seconds=0;
};

neural::Example ex(const A8& x){neural::Example a; a.x=x;a.valid=8;return a;}
A8 f(const neural::Model& m,const A8& x){return m.forward(ex(x)).y;}
B8 to_bytes(const A8& a){B8 b{};for(int d=0;d<8;d++)b[d]=neural::byte(a[d]);return b;}
A8 from_bytes(const B8& b){A8 a{};for(int d=0;d<8;d++)a[d]=double(b[d])/255.;return a;}

double sampled_jacobian_inf(const neural::Model& m, const A8& x){
 const double eps=1e-5;
 A8 row_sum{};
 for(int col=0;col<8;col++){
  auto plus=x,minus=x;plus[col]+=eps;minus[col]-=eps;
  auto yp=f(m,plus),ym=f(m,minus);
  for(int row=0;row<8;row++)row_sum[row]+=std::abs((yp[row]-ym[row])/(2*eps));
 }
 return *std::max_element(row_sum.begin(),row_sum.end());
}
double infinity_rows(const std::vector<double>& p,int start,int rows,int columns){
 double biggest=0;
 for(int i=0;i<rows;i++){double s=0;for(int j=0;j<columns;j++)s+=std::abs(p[start+i*columns+j]);biggest=std::max(biggest,s);}
 return biggest;
}
double global_bound(const neural::Model& m){
 const auto& p=m.p;
 double conv=0;for(int i=0;i<4;i++)conv+=std::abs(p[i]);
 return 0.25*conv*
   infinity_rows(p,neural::W1,neural::H,neural::N)*
   infinity_rows(p,neural::W2,neural::L,neural::H)*
   infinity_rows(p,neural::W3,neural::H,neural::L)*
   infinity_rows(p,neural::W4,neural::N,neural::H);
}
uint64_t hash_bytes(const std::vector<B8>& v){
 uint64_t h=14695981039346656037ULL;
 for(const auto& a:v)for(auto b:a){h^=b;h*=1099511628211ULL;}
 return h;
}
void check_geometric(const neural::Geo& g, Diagnostics& q){
 const double radius=std::sqrt(g.hx*g.hx+g.hy*g.hy+g.hz*g.hz);
 q.max_hopf_radius_error=std::max(q.max_hopf_radius_error,std::abs(radius-10));
 const auto coords=emission::unmorton(g.morton);
 neural::ensure(coords==std::array<uint32_t,3>{g.gx,g.gy,g.gz},"Morton inversion failed");
 neural::ensure(g.gx<SIDE&&g.gy<SIDE&&g.gz<SIDE,"quantised grid out of bounds");
 const double scale=20./(SIDE-1);
 const double recons[]={-10+g.gx*scale,-10+g.gy*scale,-10+g.gz*scale};
 const double p[]={g.px,g.py,g.pz};
 for(int d=0;d<3;d++){
  q.max_axis_value=std::max(q.max_axis_value,std::abs(p[d]));
  q.max_grid_quant_error=std::max(q.max_grid_quant_error,std::abs(recons[d]-p[d]));
 }
 neural::ensure(std::isfinite(radius)&&std::abs(radius-10)<1e-10,"Hopf 3D radius invariant failed");
 neural::ensure(q.max_grid_quant_error<=10./(SIDE-1)+1e-10,"3D quantisation error bound failed");
 neural::ensure(q.max_axis_value<=10.00000001,"latent point escaped virtual cube");
 q.coordinate_checks++;
}
void check_lifting(const B8& a, Diagnostics& d){
 auto b=a;
 ::transform(b.data(),false);
 ::transform(b.data(),true);
 neural::ensure(a==b,"3D integer lifting was not bijective");
 d.lifting_checks++;
}
void check_analytic(Diagnostics& d){
 for(int k=0;k<STEPS;k++)for(int j=0;j<64;j++){
  emission::V4 x{emission::uni(uint64_t(k)*257+j*4),emission::uni(uint64_t(k)*257+j*4+1),
                 emission::uni(uint64_t(k)*257+j*4+2),emission::uni(uint64_t(k)*257+j*4+3)};
  const auto y=emission::decode(emission::encode(x));
  for(int t=0;t<4;t++)neural::ensure(std::abs(y[t]-x[t])<1e-12,"analytic 4D inverse failed");
  d.analytic_checks++;
 }
}

Diagnostics run(const fs::path& input,const fs::path& modelpath,const fs::path& output,size_t samples){
 neural::ensure(samples>=1&&samples<=16384,"samples must be in 1..16384");
 fs::create_directories(output);
 auto bytes=neural::load(input.string());
 auto model=neural::read_model(modelpath.string());
 neural::ensure(bytes.size()>=samples*8,"input has fewer complete patches than requested");
 // Verify *full source* via actual archived codec before recursive neural experiment.
 ::encode(input.string(),(output/"source.jx3d").string());
 ::decode((output/"source.jx3d").string(),(output/"restored_source.bin").string());
 neural::ensure(::readfile((output/"restored_source.bin").string())==bytes,"full file exact restoration failed");

 const size_t complete=bytes.size()/8;
 std::vector<A8> state(samples),origin(samples);
 std::vector<B8> quant(samples),qorigin(samples);
 std::vector<size_t> source_block(samples);
 for(size_t i=0;i<samples;i++){
  source_block[i]=(i*complete)/samples;
  auto e=neural::example(bytes,source_block[i]);
  state[i]=origin[i]=e.x;
  quant[i]=qorigin[i]=to_bytes(state[i]);
  auto src=position(source_block[i]);
  neural::ensure(index(src)==source_block[i],"virtual 3D index bijection failed");
 }
 Diagnostics d;
 d.conservative_global_lipschitz=global_bound(model);
 d.initial_jacobian_inf=sampled_jacobian_inf(model,origin[0]);
 std::ofstream metrics(output/"iteration_metrics.csv"),geo(output/"geometry_snapshots.csv");
 neural::ensure(bool(metrics)&&bool(geo),"cannot open verification CSV files");
 metrics<<"iteration,continuous_step_rmse,continuous_origin_rmse,continuous_max_residual,quant_step_changed_bytes,quant_origin_changed_bytes,quant_origin_rmse,latent_mean_radius,xyz_rms_delta,changed_morton_addresses,hopf_max_radius_error,grid_max_quant_error\n"<<std::setprecision(17);
 geo<<"iteration,sample,source_block,source_x,source_y,source_z,px,py,pz,hopf_x,hopf_y,hopf_z,gx,gy,gz,morton,z0,z1,z2\n"<<std::setprecision(17);
 std::unordered_map<uint64_t,std::vector<int>> hashes;
 std::vector<std::vector<B8>> history;
 history.reserve(STEPS+1);
 std::vector<std::array<double,3>> previous_xyz(samples);
 std::vector<uint64_t> previous_codes(samples);
 auto start=std::chrono::steady_clock::now();
 for(int k=0;k<=STEPS;k++){
  std::vector<A8> next(samples);
  std::vector<B8> next_quant(samples);
  long double residual2=0,origin2=0,byteorigin2=0,latent_radius=0,spatial2=0;
  double max_residual=0;
  size_t changed_quant=0,changed_from_origin=0,changed_addr=0;
  const bool snapshot=std::find(std::begin(SNAP),std::end(SNAP),k)!=std::end(SNAP);
  for(size_t n=0;n<samples;n++){
   const auto forward=model.forward(ex(state[n]));
   const auto z=forward.z;
   auto geom=neural::geometry(z);
   check_geometric(geom,d);
   auto qstep=model.forward(ex(from_bytes(quant[n]))).y;
   next_quant[n]=to_bytes(qstep);
   check_lifting(quant[n],d);
   auto raw=position(source_block[n]);
   if(snapshot && n<256){
    geo<<k<<','<<n<<','<<source_block[n]<<','<<raw.x<<','<<raw.y<<','<<raw.z<<','
       <<geom.px<<','<<geom.py<<','<<geom.pz<<','<<geom.hx<<','<<geom.hy<<','<<geom.hz<<','
       <<geom.gx<<','<<geom.gy<<','<<geom.gz<<','<<geom.morton<<','<<z[0]<<','<<z[1]<<','<<z[2]<<'\n';
    d.snapshot_rows++;
   }
   for(int j=0;j<8;j++){
    next[n][j]=forward.y[j];
    const double diff=next[n][j]-state[n][j];
    residual2+=diff*diff;max_residual=std::max(max_residual,std::abs(diff));
    const double drift=state[n][j]-origin[n][j];origin2+=drift*drift;
    const double qdiff=double(quant[n][j])-qorigin[n][j];byteorigin2+=qdiff*qdiff;
    changed_quant+=quant[n][j]!=next_quant[n][j];
    changed_from_origin+=quant[n][j]!=qorigin[n][j];
    neural::ensure(std::isfinite(next[n][j])&&next[n][j]>=0&&next[n][j]<=1,"neural decoder non-finite/out of bounds");
   }
   latent_radius+=std::sqrt(geom.px*geom.px+geom.py*geom.py+geom.pz*geom.pz);
   if(k>0){
    const auto xyz=std::array<double,3>{geom.px,geom.py,geom.pz};
    for(int t=0;t<3;t++){double dx=xyz[t]-previous_xyz[n][t];spatial2+=dx*dx;}
    changed_addr+=(previous_codes[n]!=geom.morton);
   }
   previous_xyz[n]={geom.px,geom.py,geom.pz};
   previous_codes[n]=geom.morton;
  }
  const double rmse=std::sqrt(double(residual2/(samples*8)));
  const double origin_rmse=std::sqrt(double(origin2/(samples*8)));
  const double quant_origin_rmse=std::sqrt(double(byteorigin2/(samples*8)))/255.;
  const double xyz_rmse=k?std::sqrt(double(spatial2/(samples*3))):0.;
  if(k==0)d.initial_continuous_step_rmse=rmse;
  if(k==STEPS){d.final_continuous_step_rmse=rmse;d.final_continuous_origin_rmse=origin_rmse;d.final_quant_origin_rmse=quant_origin_rmse;d.final_changed_bytes=int(changed_quant);}
  d.max_changed_bytes=std::max(d.max_changed_bytes,int(changed_quant));
  if(d.first_continuous_fixed<0&&rmse<1e-10)d.first_continuous_fixed=k;
  metrics<<k<<','<<rmse<<','<<origin_rmse<<','<<max_residual<<','<<changed_quant<<','<<changed_from_origin<<','<<quant_origin_rmse<<','
         <<double(latent_radius/samples)<<','<<xyz_rmse<<','<<changed_addr<<','<<d.max_hopf_radius_error<<','<<d.max_grid_quant_error<<'\n';
  // Exact full-state recurrence test: the hash is only an index; equality is checked on the original byte state.
  auto h=hash_bytes(quant);
  if(d.first_quant_recurrence<0){
   auto it=hashes.find(h);
   if(it!=hashes.end()){
    for(int prev:it->second)if(history[size_t(prev)]==quant){d.first_quant_recurrence=k;d.quant_period=k-prev;break;}
   }
   hashes[h].push_back(k);
  }
  history.push_back(quant);
  if(k<STEPS){state.swap(next);quant.swap(next_quant);}
 }
 metrics.flush();geo.flush();
 neural::ensure(bool(metrics)&&bool(geo),"verification output write failed");
 metrics.close();geo.close();
 d.final_jacobian_inf=sampled_jacobian_inf(model,state[0]);
 check_analytic(d);
 d.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 neural::ensure(d.coordinate_checks==(STEPS+1)*samples,"wrong 3D geometry check count");
 neural::ensure(d.lifting_checks==(STEPS+1)*samples,"wrong reversible 3D lift check count");
 // Independent on-disk CSV physical record verification.
 auto count_rows=[](const fs::path& p){std::ifstream f(p);size_t n=0;std::string line;while(std::getline(f,line))n++;return n;};
 neural::ensure(count_rows(output/"iteration_metrics.csv")==STEPS+2,"incorrect iteration CSV row count");
 neural::ensure(count_rows(output/"geometry_snapshots.csv")==size_t(d.snapshot_rows+1),"incorrect snapshot CSV row count");
 std::ofstream report(output/"verification_report.json");
 neural::ensure(bool(report),"cannot write verification report");
 report<<std::setprecision(17);
 report<<"{\n"
       <<"  \"system\": \"JARVIS X v2 + recursive 3D verification\",\n"
       <<"  \"claim\": \"1000 feedback transitions on actual trained 3D autoencoder\",\n"
       <<"  \"iterations_executed\": "<<STEPS<<",\n"
       <<"  \"metrics_rows\": "<<STEPS+1<<",\n"
       <<"  \"sampled_patches\": "<<samples<<",\n"
       <<"  \"source_bytes\": "<<bytes.size()<<",\n"
       <<"  \"sampled_bytes_per_iteration\": "<<8*samples<<",\n"
       <<"  \"neural_parameters\": "<<neural::PARAMS<<",\n"
       <<"  \"training_during_feedback\": false,\n"
       <<"  \"continuous_fixed_point_threshold_rmse\": 1e-10,\n"
       <<"  \"first_iteration_below_threshold\": "<<d.first_continuous_fixed<<",\n"
       <<"  \"initial_continuous_step_rmse\": "<<d.initial_continuous_step_rmse<<",\n"
       <<"  \"final_continuous_step_rmse\": "<<d.final_continuous_step_rmse<<",\n"
       <<"  \"final_continuous_origin_rmse\": "<<d.final_continuous_origin_rmse<<",\n"
       <<"  \"final_quantized_origin_rmse\": "<<d.final_quant_origin_rmse<<",\n"
       <<"  \"final_quantized_bytes_changing_per_transition\": "<<d.final_changed_bytes<<",\n"
       <<"  \"first_quantized_full_state_recurrence_iteration\": "<<d.first_quant_recurrence<<",\n"
       <<"  \"detected_quantized_full_state_period\": "<<d.quant_period<<",\n"
       <<"  \"max_changed_bytes_in_one_transition\": "<<d.max_changed_bytes<<",\n"
       <<"  \"max_hopf_radius_error\": "<<d.max_hopf_radius_error<<",\n"
       <<"  \"max_3d_grid_quantization_error\": "<<d.max_grid_quant_error<<",\n"
       <<"  \"max_abs_cartesian_coordinate\": "<<d.max_axis_value<<",\n"
       <<"  \"morton_roundtrip_checks\": "<<d.coordinate_checks<<",\n"
       <<"  \"3d_lifting_roundtrip_checks\": "<<d.lifting_checks<<",\n"
       <<"  \"analytic_4d_codec_roundtrip_checks\": "<<d.analytic_checks<<",\n"
       <<"  \"global_jacobian_infinity_norm_upper_bound\": "<<d.conservative_global_lipschitz<<",\n"
       <<"  \"sampled_initial_jacobian_infinity_norm\": "<<d.initial_jacobian_inf<<",\n"
       <<"  \"sampled_final_jacobian_infinity_norm\": "<<d.final_jacobian_inf<<",\n"
       <<"  \"3d_snapshot_rows\": "<<d.snapshot_rows<<",\n"
       <<"  \"exact_full_source_file_binary_roundtrip\": true,\n"
       <<"  \"elapsed_seconds\": "<<d.seconds<<"\n} \n";
 neural::ensure(bool(report),"failed to write report");
 std::cout<<std::setprecision(12)
  <<"1000_ITERATION_3D_VERIFICATION PASS\n"
  <<"steps="<<STEPS<<" patches="<<samples<<" geometric coordinate checks="<<d.coordinate_checks<<"\n"
  <<"continuous feedback step RMSE: "<<d.initial_continuous_step_rmse<<" -> "<<d.final_continuous_step_rmse<<"\n"
  <<"continuous fixed-point threshold first met at step="<<d.first_continuous_fixed<<"\n"
  <<"quantized full-state recurrence at step="<<d.first_quant_recurrence<<" period="<<d.quant_period<<"\n"
  <<"quantized changed bytes final transition="<<d.final_changed_bytes<<"\n"
  <<"Hopf norm maximum error="<<d.max_hopf_radius_error<<" Morton tests="<<d.coordinate_checks<<"\n"
  <<"full source file exact binary roundtrip PASS\n"
  <<"elapsed_seconds="<<d.seconds<<"\n";
 return d;
}
} // namespace recursive3d

int main(int argc,char** argv){
 try{
  if(argc==2&&std::string(argv[1])=="selftest"){neural::selftest();return 0;}
  if(argc>=2&&std::string(argv[1])=="loop1000"){
   if(argc<5||argc>6)throw std::runtime_error("Usage: jarvis_x_1000_loop loop1000 INPUT MODEL OUTPUT_DIR [SAMPLED_PATCHES=1024]");
   size_t samples=argc==6?size_t(neural::pos_int(argv[5],16384)):1024;
   recursive3d::run(argv[2],argv[3],argv[4],samples);
   return 0;
  }
  return neural::dispatch(argc,argv); // all v2 commands remain available
 } catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
