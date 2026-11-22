// Dr Moagi 3D codec v2: portable C++17 reference encoder/decoder.
// Uses the SAME frozen Q14 3D basis as moagi_core.js (basis.bin).
// Build: g++ -std=c++17 -O2 -Wall -Wextra -pedantic moagi_3d_runtime.cpp -o moagi3d
// Usage: ./moagi3d basis.bin encode input.raw output.dm3d
//        ./moagi3d basis.bin decode input.dm3d output.raw
//        ./moagi3d basis.bin selftest
// Geometric model, not a trained ANN or a hardware throughput claim.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dm3d {
constexpr int NX=14,NY=8,NZ=7,N=NX*NY*NZ,K=16;
constexpr int64_t DENOM=16384LL*1024LL;
using Raw=std::array<uint8_t,N>;
using Latent=std::array<int32_t,K>;
using FloatLatent=std::array<double,K>;
using Matrix=std::array<std::array<double,K>,K>;
using Bytes=std::vector<uint8_t>;

Bytes read_file(const std::string& file) {
 std::ifstream f(file,std::ios::binary);
 if(!f)throw std::runtime_error("cannot open "+file);
 return Bytes(std::istreambuf_iterator<char>(f),{});
}
void write_file(const std::string& file,const Bytes& bytes){
 std::ofstream f(file,std::ios::binary); if(!f)throw std::runtime_error("cannot write "+file);
 f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
 if(!f)throw std::runtime_error("short write");
}
uint32_t checksum(const Raw& x){uint32_t h=2166136261u;for(auto b:x){h^=b;h*=16777619u;}return h;}
void put16(Bytes& b,uint16_t x){b.push_back(uint8_t(x));b.push_back(uint8_t(x>>8));}
void put32(Bytes& b,uint32_t x){for(int i=0;i<4;i++)b.push_back(uint8_t(x>>(8*i)));}
uint16_t get16(const Bytes& b,size_t p){if(p+2>b.size())throw std::runtime_error("truncated uint16");return uint16_t(uint16_t(b[p])|(uint16_t(b[p+1])<<8));}
uint32_t get32(const Bytes& b,size_t p){if(p+4>b.size())throw std::runtime_error("truncated uint32");return uint32_t(b[p])|(uint32_t(b[p+1])<<8)|(uint32_t(b[p+2])<<16)|(uint32_t(b[p+3])<<24);}
int64_t floor_div(int64_t n,int64_t d) {return n>=0?n/d:-((-n+d-1)/d);}
Bytes rle(const Raw& bytes){
 Bytes out;size_t i=0;
 while(i<N){
  const bool zero=bytes[i]==0;size_t j=i;
  while(j<N&&(bytes[j]==0)==zero&&j-i<255)++j;
  out.push_back(zero?0:1);out.push_back(static_cast<uint8_t>(j-i));
  if(!zero)for(size_t k=i;k<j;k++)out.push_back(bytes[k]);
  i=j;
 }return out;
}
Raw unrle(const Bytes& bytes){
 Raw out{};size_t i=0,o=0;
 while(i<bytes.size()){
  if(i+2>bytes.size())throw std::runtime_error("truncated RLE header");
  auto mode=bytes[i++],count=bytes[i++];
  if(!count||o+count>N)throw std::runtime_error("RLE length overflow");
  if(mode==0){o+=count;}
  else if(mode==1){if(i+count>bytes.size())throw std::runtime_error("truncated RLE bytes");for(int j=0;j<count;j++)out[o++]=bytes[i++];}
  else throw std::runtime_error("bad RLE opcode");
 }
 if(o!=N)throw std::runtime_error("bad RLE total length");
 return out;
}
struct Engine {
 std::array<int16_t,N*K> bi{};
 std::array<double,N*K> b{};
 Matrix l{};
 explicit Engine(const std::string& path){
  auto bytes=read_file(path);
  if(bytes.size()!=2*N*K)throw std::runtime_error("basis.bin must contain 25088 bytes");
  for(int i=0;i<N*K;i++){
   bi[i]=static_cast<int16_t>(get16(bytes,2*i));b[i]=double(bi[i])/16384.0;
  }
  Matrix gram{};
  for(int i=0;i<N;i++)for(int k=0;k<K;k++)for(int j=0;j<K;j++)gram[k][j]+=b[i*K+k]*b[i*K+j];
  for(int k=0;k<K;k++)gram[k][k]+=.025;
  for(int i=0;i<K;i++)for(int j=0;j<=i;j++){
   double v=gram[i][j];for(int p=0;p<j;p++)v-=l[i][p]*l[j][p];
   if(i==j){if(!(v>0))throw std::runtime_error("non-SPD Gram matrix");l[i][j]=std::sqrt(v);}
   else l[i][j]=v/l[j][j];
  }
 }
 Latent encode(const Raw& x)const{
  FloatLatent rhs{},y{},z{};
  for(int i=0;i<N;i++)for(int k=0;k<K;k++)rhs[k]+=b[i*K+k]*x[i];
  for(int i=0;i<K;i++){double v=rhs[i];for(int j=0;j<i;j++)v-=l[i][j]*y[j];y[i]=v/l[i][i];}
  for(int i=K-1;i>=0;i--){double v=y[i];for(int j=i+1;j<K;j++)v-=l[j][i]*z[j];z[i]=v/l[i][i];}
  Latent q{};for(int k=0;k<K;k++){
   const auto v=std::llround(z[k]*1024.0);
   if(v<std::numeric_limits<int32_t>::min()||v>std::numeric_limits<int32_t>::max())throw std::runtime_error("latent overflow");
   q[k]=static_cast<int32_t>(v);
  }return q;
 }
 Raw predict(const Latent& q)const{
  Raw out{};for(int i=0;i<N;i++){
   int64_t sum=0;for(int k=0;k<K;k++)sum+=int64_t(bi[i*K+k])*int64_t(q[k]);
   const auto v=floor_div(sum+DENOM/2,DENOM);
   out[i]=static_cast<uint8_t>(std::clamp<int64_t>(v,0,255));
  }return out;
 }
 Bytes pack(const Raw& raw,const Latent& q)const{
  const auto predicted=predict(q);Raw residual{};
  for(int i=0;i<N;i++)residual[i]=raw[i]^predicted[i];
  const auto encoded=rle(residual); const bool useRle=encoded.size()<N;
  Bytes out;out.reserve(16+4*K+N);
  for(const auto c:{'D','M','3','D'})out.push_back(uint8_t(c));
  for(const uint8_t v:{uint8_t(2),uint8_t(NX),uint8_t(NY),uint8_t(NZ),uint8_t(K),uint8_t(useRle)})out.push_back(v);
  put16(out,static_cast<uint16_t>(useRle?encoded.size():N));put32(out,checksum(raw));
  for(auto x:q)put32(out,static_cast<uint32_t>(x));
  if(useRle)out.insert(out.end(),encoded.begin(),encoded.end());
  else out.insert(out.end(),residual.begin(),residual.end());
  return out;
 }
 Raw unpack(const Bytes& bytes)const{
  if(bytes.size()<16+4*K||bytes[0]!='D'||bytes[1]!='M'||bytes[2]!='3'||bytes[3]!='D'||bytes[4]!=2)throw std::runtime_error("invalid DM3D version/signature");
  if(bytes[5]!=NX||bytes[6]!=NY||bytes[7]!=NZ||bytes[8]!=K)throw std::runtime_error("unsupported dimensions");
  const auto mode=bytes[9]; const auto length=get16(bytes,10); const auto hash=get32(bytes,12);
  if(bytes.size()!=size_t(16+4*K+length))throw std::runtime_error("invalid packet length");
  Latent q{};for(int k=0;k<K;k++)q[k]=static_cast<int32_t>(get32(bytes,16+4*k));
  Raw residual{};
  if(mode==0){if(length!=N)throw std::runtime_error("raw residual is not 784 bytes");std::copy(bytes.begin()+16+4*K,bytes.end(),residual.begin());}
  else if(mode==1)residual=unrle(Bytes(bytes.begin()+16+4*K,bytes.end()));
  else throw std::runtime_error("unsupported residual mode");
  const auto prediction=predict(q);Raw out{};for(int i=0;i<N;i++)out[i]=prediction[i]^residual[i];
  if(checksum(out)!=hash)throw std::runtime_error("checksum mismatch");
  return out;
 }
};
Raw as_raw(const Bytes& bytes){if(bytes.size()!=N)throw std::runtime_error("raw input must be exactly 784 bytes");Raw a{};std::copy(bytes.begin(),bytes.end(),a.begin());return a;}
Bytes as_bytes(const Raw& r){return Bytes(r.begin(),r.end());}
}
int main(int argc,char** argv){
 try{
  if(argc<3){std::cerr<<"Usage: moagi3d basis.bin selftest | encode input.raw output.dm3d | decode input.dm3d output.raw\n";return 2;}
  const dm3d::Engine engine(argv[1]);const std::string mode=argv[2];
  if(mode=="selftest"){
   dm3d::Raw in{};for(int i=0;i<dm3d::N;i++)in[i]=uint8_t((i*37+(i/19)*11)%256);
   const auto encoded=engine.pack(in,engine.encode(in)); const auto restored=engine.unpack(encoded);
   if(in!=restored)throw std::runtime_error("selftest byte failure");
   std::cout<<"PASS 784/784 bytes, "<<encoded.size()<<"-byte package, ratio "<<std::fixed<<std::setprecision(6)<<double(dm3d::N)/encoded.size()<<"x\n";return 0;
  }
  if(argc!=5)throw std::runtime_error("encode/decode require input and output file paths");
  if(mode=="encode"){
   const auto input=dm3d::as_raw(dm3d::read_file(argv[3]));
   const auto encoded=engine.pack(input,engine.encode(input));
   if(engine.unpack(encoded)!=input)throw std::runtime_error("roundtrip mismatch");
   dm3d::write_file(argv[4],encoded);
   std::cout<<"PASS encoded: "<<dm3d::N<<" source bytes -> "<<encoded.size()<<" packed bytes, exact verified\n";
  }else if(mode=="decode"){
   const auto decoded=engine.unpack(dm3d::read_file(argv[3]));
   dm3d::write_file(argv[4],dm3d::as_bytes(decoded));
   std::cout<<"PASS decoded: 784 verified bytes\n";
  }else throw std::runtime_error("unknown command");
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}