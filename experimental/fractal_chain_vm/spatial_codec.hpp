#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Reference-only deterministic 8-bit 3D block-mean transform. This is NOT an ANN.
namespace jarvisx {
struct SpatialCodec {
 struct Payload { unsigned dim=0, block=0; std::vector<uint8_t> bytes; };
 static constexpr unsigned header_bytes=24;
 static unsigned blocks(unsigned D,unsigned B) {
  if(D<4 || D>256 || B==0 || D%B)throw std::invalid_argument("invalid block shape");
  unsigned n=D/B; return n*n*n;
 }
 static size_t index(unsigned z,unsigned y,unsigned x,unsigned D) {
  return (size_t(z)*D+y)*D+x;
 }
 static Payload encode(const std::vector<float>& v,unsigned D,unsigned B) {
  if(v.size()!=size_t(D)*D*D)throw std::invalid_argument("invalid 3D input");
  unsigned N=D/B;Payload p; p.dim=D;p.block=B;p.bytes.reserve(blocks(D,B));
  for(unsigned z=0;z<N;++z)for(unsigned y=0;y<N;++y)for(unsigned x=0;x<N;++x){
   double sum=0;
   for(unsigned dz=0;dz<B;++dz)for(unsigned dy=0;dy<B;++dy)for(unsigned dx=0;dx<B;++dx){
    float val=v[index(z*B+dz,y*B+dy,x*B+dx,D)];
    if(!std::isfinite(val)||val<0||val>1)throw std::invalid_argument("voxel outside [0,1]");
    sum+=val;
   }
   p.bytes.push_back(uint8_t(std::lround((sum/(B*B*B))*255)));
  }
  return p;
 }
 static std::vector<float> decode(const Payload& p) {
  unsigned D=p.dim,B=p.block,N=D/B;
  if(p.bytes.size()!=blocks(D,B))throw std::invalid_argument("invalid encoded payload length");
  std::vector<float> v(size_t(D)*D*D);
  for(unsigned z=0;z<D;++z)for(unsigned y=0;y<D;++y)for(unsigned x=0;x<D;++x)
   v[index(z,y,x,D)]=p.bytes[index(z/B,y/B,x/B,N)]/255.f;
  return v;
 }
 static uint32_t checksum(const std::vector<uint8_t>& b) {
  uint32_t h=2166136261u;for(uint8_t x:b){h^=x;h*=16777619u;}return h;
 }
 static void put32(std::ostream& out,uint32_t x) {
  for(unsigned k=0;k<4;++k)out.put(char(x>>(k*8)));
 }
 static uint32_t get32(std::istream& in) {
  uint32_t v=0;for(unsigned k=0;k<4;++k){int c=in.get();if(c==EOF)throw std::runtime_error("truncated spatial header");v|=uint32_t(uint8_t(c))<<(k*8);}return v;
 }
 static size_t serialized_size(const Payload& p){return header_bytes+p.bytes.size();}
 static void save(const Payload& p,const std::string& path) {
  if(p.bytes.size()!=blocks(p.dim,p.block))throw std::invalid_argument("invalid payload");
  std::ofstream out(path,std::ios::binary);if(!out)throw std::runtime_error("cannot create file");
  out.write("JX3B",4);put32(out,1);put32(out,p.dim);put32(out,p.block);
  put32(out,uint32_t(p.bytes.size()));put32(out,checksum(p.bytes));
  out.write(reinterpret_cast<const char*>(p.bytes.data()),std::streamsize(p.bytes.size()));
  if(!out)throw std::runtime_error("spatial file write failed");
 }
 static Payload load(const std::string& path) {
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  if(!in)throw std::runtime_error("cannot open spatial file");
  auto size=in.tellg();if(size<0||size>std::streamoff(1<<25))throw std::runtime_error("invalid spatial file size");
  in.seekg(0);char magic[4];in.read(magic,4);
  if(!in||std::string(magic,4)!="JX3B"||get32(in)!=1)throw std::runtime_error("bad spatial format");
  Payload p;p.dim=get32(in);p.block=get32(in);
  uint32_t length=get32(in),expected_hash=get32(in);
  if(length!=blocks(p.dim,p.block) || size!=std::streamoff(header_bytes+length))
   throw std::runtime_error("invalid spatial length");
  p.bytes.resize(length);
  in.read(reinterpret_cast<char*>(p.bytes.data()),length);
  if(!in || checksum(p.bytes)!=expected_hash)throw std::runtime_error("corrupt spatial payload");
  return p;
 }
};
}
