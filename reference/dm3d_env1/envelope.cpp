// DM3D-ENV1: standalone, generic public C++17 envelope conformance reference.
// Not confidential DM-PCS software, encryption, or a production parser.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace dm3d {
constexpr std::size_t HEADER_SIZE = 84, MAX_PAYLOAD = 64*64*64;
constexpr std::array<std::uint8_t,8> MAGIC = {'D','M','3','D','E','N','V','1'};
constexpr const char* PROFILE = "DM3D-CUBE-U8-V1";
constexpr std::uint32_t K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
constexpr std::uint32_t rotr(std::uint32_t x,int n){return (x>>n)|(x<<(32-n));}
void put32(std::vector<std::uint8_t>& b,std::uint32_t v){
  for(int i=3;i>=0;--i)b.push_back(static_cast<std::uint8_t>(v>>(i*8)));
}
void put64(std::vector<std::uint8_t>& b,std::uint64_t v){
  for(int i=7;i>=0;--i)b.push_back(static_cast<std::uint8_t>(v>>(i*8)));
}
std::uint32_t u32(const std::uint8_t* p){
  return (std::uint32_t(p[0])<<24)|(std::uint32_t(p[1])<<16)|(std::uint32_t(p[2])<<8)|p[3];
}
std::uint64_t u64(const std::uint8_t* p){return (std::uint64_t(u32(p))<<32)|u32(p+4);}
std::array<std::uint8_t,32> sha256(const std::vector<std::uint8_t>& data){
  std::uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  std::vector<std::uint8_t> b(data);
  b.push_back(0x80);
  while(b.size()%64!=56)b.push_back(0);
  put64(b,std::uint64_t(data.size())*8);
  for(std::size_t offset=0;offset<b.size();offset+=64){
    std::uint32_t w[64];
    for(int i=0;i<16;++i)w[i]=u32(b.data()+offset+4*i);
    for(int i=16;i<64;++i){
      const std::uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
      const std::uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
      w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    std::uint32_t a=h[0],bb=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],q=h[7];
    for(int i=0;i<64;++i){
      const std::uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
      const std::uint32_t choose=(e&f)^((~e)&g);
      const std::uint32_t temp1=q+s1+choose+K[i]+w[i];
      const std::uint32_t s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
      const std::uint32_t majority=(a&bb)^(a&c)^(bb&c);
      const std::uint32_t temp2=s0+majority;
      q=g;g=f;f=e;e=d+temp1;d=c;c=bb;bb=a;a=temp1+temp2;
    }
    h[0]+=a;h[1]+=bb;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=q;
  }
  std::array<std::uint8_t,32> out{};
  for(int i=0;i<8;++i)for(int j=0;j<4;++j)
    out[i*4+j]=static_cast<std::uint8_t>(h[i]>>(24-j*8));
  return out;
}
std::string hex(const std::array<std::uint8_t,32>& a){
  const char chars[]="0123456789abcdef";std::string s;s.reserve(64);
  for(auto v:a){s.push_back(chars[v>>4]);s.push_back(chars[v&15]);}
  return s;
}
void require_side(unsigned side){
  if(side<2||side>64||(side&(side-1)))throw std::runtime_error("side must be power of 2 in [2,64]");
}
std::vector<std::uint8_t> pack(const std::vector<std::uint8_t>& data,unsigned side){
  require_side(side);
  if(data.size()!=std::size_t(side)*side*side)throw std::runtime_error("payload must be side^3 bytes");
  std::vector<std::uint8_t> b;b.reserve(HEADER_SIZE+data.size());
  b.insert(b.end(),MAGIC.begin(),MAGIC.end());
  b.push_back(0);b.push_back(1);b.push_back(0);b.push_back(0);
  for(int i=0;i<16;i++)b.push_back(i<15?std::uint8_t(PROFILE[i]):0);
  b.push_back(1);b.push_back(1);b.push_back(0);b.push_back(0);
  for(int i=0;i<3;i++)put32(b,side);
  put64(b,data.size());
  const auto dg=sha256(data);b.insert(b.end(),dg.begin(),dg.end());
  if(b.size()!=HEADER_SIZE)throw std::runtime_error("internal header length mismatch");
  b.insert(b.end(),data.begin(),data.end());return b;
}
struct Decoded{unsigned side;std::vector<std::uint8_t> payload;};
Decoded unpack(const std::vector<std::uint8_t>& b){
  if(b.size()<HEADER_SIZE||b.size()>HEADER_SIZE+MAX_PAYLOAD)
    throw std::runtime_error("invalid envelope size");
  if(!std::equal(MAGIC.begin(),MAGIC.end(),b.begin())||
    b[8]!=0||b[9]!=1||b[10]!=0||b[11]!=0)
    throw std::runtime_error("invalid magic/version/flags");
  for(int i=0;i<16;i++)
    if(b[12+i]!=(i<15?std::uint8_t(PROFILE[i]):0))
      throw std::runtime_error("invalid profile");
  if(b[28]!=1||b[29]!=1||b[30]!=0||b[31]!=0)
    throw std::runtime_error("invalid layout/boundary/reserved");
  const auto dx=u32(b.data()+32),dy=u32(b.data()+36),dz=u32(b.data()+40);
  require_side(dx);
  const auto n=u64(b.data()+44);
  if(dy!=dx||dz!=dx||n!=std::uint64_t(dx)*dx*dx||b.size()!=HEADER_SIZE+n)
    throw std::runtime_error("geometry/payload length mismatch");
  std::vector<std::uint8_t> data(b.begin()+HEADER_SIZE,b.end());
  const auto dg=sha256(data);
  if(!std::equal(dg.begin(),dg.end(),b.begin()+52))
    throw std::runtime_error("SHA-256 payload mismatch");
  return {dx,std::move(data)};
}
std::uint32_t row_index(unsigned x,unsigned y,unsigned z,unsigned side){
  require_side(side);
  if(x>=side||y>=side||z>=side)throw std::runtime_error("coordinate outside cube");
  return x+side*y+side*side*z;
}
std::uint32_t morton_index(unsigned x,unsigned y,unsigned z,unsigned side){
  row_index(x,y,z,side);
  std::uint32_t m=0;
  for(unsigned i=0;(1U<<i)<side;++i){
    m|=((x>>i)&1U)<<(3*i);
    m|=((y>>i)&1U)<<(3*i+1);
    m|=((z>>i)&1U)<<(3*i+2);
  }
  return m;
}
std::vector<std::uint8_t> read_file(const std::string& path){
  std::ifstream in(path,std::ios::binary|std::ios::ate);
  if(!in)throw std::runtime_error("cannot open input");
  const auto n=in.tellg();
  if(n<0||std::uint64_t(n)>HEADER_SIZE+MAX_PAYLOAD)
    throw std::runtime_error("file exceeds reference limit");
  in.seekg(0);
  return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
void write_file(const std::string& path,const std::vector<std::uint8_t>& b){
  std::ofstream out(path,std::ios::binary);
  if(!out)throw std::runtime_error("cannot open output");
  out.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size()));
  if(!out)throw std::runtime_error("cannot write output");
}
}
int main(int argc,char** argv){
  try{
    const std::string action=argc>1?argv[1]:"";
    if(action=="pack"&&argc==5){
      const int side=std::stoi(argv[3]);
      if(side<0)throw std::runtime_error("invalid side");
      dm3d::write_file(argv[4],dm3d::pack(dm3d::read_file(argv[2]),unsigned(side)));
    }else if(action=="unpack"&&argc==4){
      const auto decoded=dm3d::unpack(dm3d::read_file(argv[2]));
      dm3d::write_file(argv[3],decoded.payload);
    }else if(action=="inspect"&&argc==3){
      const auto decoded=dm3d::unpack(dm3d::read_file(argv[2]));
      std::cout<<"{\"format\":\"DM3D-ENV1\",\"payload_bytes\":"<<decoded.payload.size()
               <<",\"sha256\":\""<<dm3d::hex(dm3d::sha256(decoded.payload))
               <<"\",\"side\":"<<decoded.side<<"}\n";
    }else if(action=="index"&&argc==6){
      const auto side=std::stoul(argv[2]),x=std::stoul(argv[3]),
                 y=std::stoul(argv[4]),z=std::stoul(argv[5]);
      std::cout<<dm3d::row_index(x,y,z,side)<<" "
               <<dm3d::morton_index(x,y,z,side)<<"\n";
    }else{
      std::cerr<<"usage: envelope pack INPUT SIDE OUTPUT | unpack INPUT OUTPUT | inspect INPUT | index SIDE X Y Z\n";
      return 2;
    }
    return 0;
  }catch(const std::exception& e){
    std::cerr<<"DM3D-ENV1 rejected: "<<e.what()<<"\n";
    return 1;
  }
}
