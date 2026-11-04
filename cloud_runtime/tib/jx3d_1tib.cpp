// JARVIS-X 1 TiB SPARSE 3D STREAMING REFERENCE
// C++17, no external dependencies. Deterministic synthetic input, causal XOR3D
// predictor, adaptive ZERO/CONSTANT/RLE/RAW bytecode, bounded-memory streaming.
// This is NOT a trained ANN or an authenticated encryption scheme.
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __unix__
#include <sys/resource.h>
#endif

namespace jx {
constexpr size_t BX=128, BY=128, BZ=64;
constexpr size_t BLOCK=BX*BY*BZ;
constexpr uint64_t VX=8192, VY=8192, VZ=16384;
constexpr uint64_t GX=VX/BX, GY=VY/BY, GZ=VZ/BZ;
constexpr uint64_t NTILES=GX*GY*GZ;
constexpr uint64_t VBYTES=VX*VY*VZ;
constexpr size_t HSIZE=32;
constexpr uint8_t VERSION=2, OP_XOR3D=0x31;
constexpr uint8_t RAW=0, RLE=1, ZERO=2, CONSTANT=3, DIRECT=4;
static_assert(BLOCK==1048576, "one tile must be 1 MiB");
static_assert(GX==64 && GY==64 && GZ==256, "tile grid");
static_assert(VBYTES == (1ULL<<40) && NTILES == (1ULL<<20), "1TiB virtual geometry");

uint32_t crc32(const uint8_t* p, size_t n) {
  static const auto table=[](){
    std::array<uint32_t,256> t{};
    for(uint32_t i=0;i<256;i++){
      uint32_t v=i;
      for(int j=0;j<8;j++)v=(v>>1)^(0xEDB88320U & (0U-(v&1U)));
      t[i]=v;
    }
    return t;
  }();
  uint32_t c=0xFFFFFFFFU;
  for(size_t i=0;i<n;i++)c=table[(c^p[i])&255U]^(c>>8);
  return c^0xFFFFFFFFU;
}
void p16(std::array<uint8_t,HSIZE>& h,size_t i,uint16_t v){
  h[i]=uint8_t(v);h[i+1]=uint8_t(v>>8);
}
void p32(std::array<uint8_t,HSIZE>& h,size_t i,uint32_t v){
  for(int j=0;j<4;j++) h[i+j]=uint8_t(v>>(8*j));
}
uint16_t g16(const std::array<uint8_t,HSIZE>& h,size_t i){
  return uint16_t(uint16_t(h[i])|(uint16_t(h[i+1])<<8));
}
uint32_t g32(const std::array<uint8_t,HSIZE>& h,size_t i){
  uint32_t v=0;for(int j=0;j<4;j++)v|=uint32_t(h[i+j])<<(8*j);return v;
}
struct Record {std::array<uint8_t,HSIZE> h{};std::vector<uint8_t> payload;};
uint64_t id_of(const Record& r){
  return uint64_t(g16(r.h,8)) + GX*uint64_t(g16(r.h,10)) + GX*GY*uint64_t(g16(r.h,12));
}
uint8_t predict(const std::vector<uint8_t>& b,size_t i){
  const size_t x=i%128,y=(i/128)%128,z=i/16384;
  return (x?b[i-1]:0) ^ (y?b[i-128]:0) ^ (z?b[i-16384]:0);
}
void residual_encode(const std::vector<uint8_t>& raw,std::vector<uint8_t>& residual){
  for(size_t i=0;i<BLOCK;i++)residual[i]=raw[i]^predict(raw,i);
}
void residual_decode(const std::vector<uint8_t>& residual,std::vector<uint8_t>& raw){
  std::fill(raw.begin(),raw.end(),0);
  for(size_t i=0;i<BLOCK;i++)raw[i]=residual[i]^predict(raw,i);
}
std::vector<uint8_t> rle_encode(const std::vector<uint8_t>& b){
  std::vector<uint8_t> r;
  r.reserve(BLOCK/2);
  for(size_t i=0;i<BLOCK;){
    size_t e=i+1;
    while(e<BLOCK && e-i<255 && b[e]==b[i])++e;
    r.push_back(uint8_t(e-i));r.push_back(b[i]);i=e;
  }
  return r;
}
void rle_decode(const std::vector<uint8_t>& b,std::vector<uint8_t>& out){
  if(b.empty() || (b.size()%2))throw std::runtime_error("RLE payload length invalid");
  size_t at=0;
  for(size_t i=0;i<b.size();i+=2){
    uint8_t n=b[i];
    if(n==0 || n>BLOCK-at)throw std::runtime_error("RLE decompression bound violated");
    std::fill(out.begin()+static_cast<std::ptrdiff_t>(at),
              out.begin()+static_cast<std::ptrdiff_t>(at+n),b[i+1]);
    at+=n;
  }
  if(at!=BLOCK)throw std::runtime_error("RLE decoded size invalid");
}
// Strongly deterministic generator for tests. Not representative of real media.
void generate(std::vector<uint8_t>& raw,uint64_t tile){
  const uint64_t cx=tile%GX,cy=(tile/GX)%GY,cz=tile/(GX*GY);
  const uint32_t kind=uint32_t(tile%4);
  uint32_t state=uint32_t(tile*2654435761ULL+0x9e3779b9ULL);
  for(size_t i=0;i<BLOCK;i++){
    uint8_t v=0;
    if(kind==1)v=uint8_t((tile*37)&255);
    else if(kind==2){
      const uint64_t x=cx*BX+i%BX,y=cy*BY+(i/BX)%BY,z=cz*BZ+i/(BX*BY);
      v=uint8_t((x/8+y/8+z/4+(x^y^z)/32)&255);
    } else if(kind==3){
      state^=state<<13;state^=state>>17;state^=state<<5;v=uint8_t(state);
    }
    raw[i]=v;
  }
}
Record encode(const std::vector<uint8_t>& raw,uint64_t tile,std::vector<uint8_t>& residual,uint32_t valid_bytes=uint32_t(BLOCK)){
  if(raw.size()!=BLOCK || tile>=NTILES || valid_bytes==0 || valid_bytes>BLOCK)throw std::runtime_error("invalid raw tile");
  Record r;
  const bool zero=std::all_of(raw.begin(),raw.end(),[](uint8_t x){return x==0;});
  const bool constant=std::all_of(raw.begin(),raw.end(),[&](uint8_t x){return x==raw[0];});
  uint8_t fmt=RAW;
  if(zero){fmt=ZERO;}
  else if(constant){fmt=CONSTANT;r.payload={raw[0]};}
  else {
    residual_encode(raw,residual);
    auto packed=rle_encode(residual);
    if(packed.size()<BLOCK){fmt=RLE;r.payload=std::move(packed);}
    else {fmt=RAW;r.payload=residual;}
    if(r.payload.size()>=valid_bytes){
      fmt=DIRECT;
      r.payload.assign(raw.begin(),raw.begin()+valid_bytes);
    }
  }
  auto& h=r.h;
  h[0]='J';h[1]='X';h[2]='T';h[3]='B';h[4]=VERSION;h[5]=OP_XOR3D;h[6]=fmt;
  p16(h,8,uint16_t(tile%GX));p16(h,10,uint16_t((tile/GX)%GY));p16(h,12,uint16_t(tile/(GX*GY)));
  p32(h,16,valid_bytes);p32(h,20,uint32_t(r.payload.size()));
  p32(h,24,crc32(raw.data(),BLOCK));p32(h,28,crc32(r.payload.data(),r.payload.size()));
  return r;
}
void validate_header(const std::array<uint8_t,HSIZE>& h){
  if(std::memcmp(h.data(),"JXTB",4)!=0 || h[4]!=VERSION || h[5]!=OP_XOR3D ||
     h[7] || h[14] || h[15] || g16(h,8)>=GX || g16(h,10)>=GY || g16(h,12)>=GZ ||
     (g32(h,16)==0 || g32(h,16)>BLOCK) || g32(h,20)>BLOCK || h[6]>DIRECT)
    throw std::runtime_error("invalid record header");
  const uint32_t n=g32(h,20);
  switch(h[6]){
    case ZERO: if(n!=0)throw std::runtime_error("zero tile nonempty");break;
    case CONSTANT: if(n!=1)throw std::runtime_error("constant tile invalid");break;
    case RAW: if(n!=BLOCK)throw std::runtime_error("raw tile invalid length");break;
    case DIRECT: if(n!=g32(h,16))throw std::runtime_error("direct tile length invalid");break;
    case RLE: if(n==0 || (n&1))throw std::runtime_error("RLE tile invalid length");break;
    default:throw std::runtime_error("unknown mode");
  }
}
void decode(const Record& r,std::vector<uint8_t>& residual,std::vector<uint8_t>& out){
  validate_header(r.h);
  if(r.payload.size()!=g32(r.h,20))throw std::runtime_error("mismatched payload length");
  if(crc32(r.payload.data(),r.payload.size())!=g32(r.h,28))throw std::runtime_error("CRC32 payload mismatch");
  switch(r.h[6]){
    case ZERO:std::fill(out.begin(),out.end(),0);break;
    case CONSTANT:std::fill(out.begin(),out.end(),r.payload[0]);break;
    case RAW:residual=r.payload;residual_decode(residual,out);break;
    case DIRECT:std::fill(out.begin(),out.end(),0);std::copy(r.payload.begin(),r.payload.end(),out.begin());break;
    case RLE:rle_decode(r.payload,residual);residual_decode(residual,out);break;
    default:throw std::runtime_error("unknown opcode");
  }
  if(crc32(out.data(),out.size())!=g32(r.h,24))throw std::runtime_error("CRC32 raw mismatch");
}
// A complete read is mandatory. Truncated records are not accepted.
bool read_record(std::istream& in,Record& r){
  r=Record{};
  int c=in.peek();
  if(c==std::char_traits<char>::eof()){
    if(in.eof())return false;
    throw std::runtime_error("stream read failed");
  }
  in.read(reinterpret_cast<char*>(r.h.data()),HSIZE);
  if(in.gcount()!=static_cast<std::streamsize>(HSIZE))throw std::runtime_error("truncated header");
  validate_header(r.h);
  const auto n=g32(r.h,20);
  r.payload.resize(n);
  if(n){
    in.read(reinterpret_cast<char*>(r.payload.data()),n);
    if(in.gcount()!=static_cast<std::streamsize>(n))throw std::runtime_error("truncated payload");
  }
  return true;
}
void write_record(std::ostream& out,const Record& r){
  out.write(reinterpret_cast<const char*>(r.h.data()),HSIZE);
  if(!r.payload.empty())out.write(reinterpret_cast<const char*>(r.payload.data()),static_cast<std::streamsize>(r.payload.size()));
  if(!out)throw std::runtime_error("failed to write encoded record");
}
struct Stats {uint64_t tiles=0,source=0,encoded=0,zero=0,constant=0,raw=0,rle=0,direct=0;};
void count(Stats& s,const Record& r){
  ++s.tiles;s.source+=g32(r.h,16);s.encoded+=HSIZE+r.payload.size();
  switch(r.h[6]){case ZERO:++s.zero;break;case CONSTANT:++s.constant;break;case RAW:++s.raw;break;case RLE:++s.rle;break;case DIRECT:++s.direct;break;}
}
uint64_t checked_number(const std::string& s){
  if(s.empty() || s[0]=='-')throw std::runtime_error("negative or empty integer");
  size_t end=0;const auto n=std::stoull(s,&end);
  if(end!=s.size())throw std::runtime_error("invalid numeric input");
  return n;
}
struct Options {
  uint64_t start=0,tiles=64;
  std::string output,verify;
  bool resume=false,overwrite=false,corrupt=false;
};
Options parse(int argc,char** argv){
  Options o;
  for(int i=1;i<argc;i++){
    const std::string a=argv[i];
    auto next=[&](){if(i+1==argc)throw std::runtime_error("argument missing after "+a);return std::string(argv[++i]);};
    if(a=="--start")o.start=checked_number(next());
    else if(a=="--tiles")o.tiles=checked_number(next());
    else if(a=="--out")o.output=next();
    else if(a=="--verify")o.verify=next();
    else if(a=="--resume")o.resume=true;
    else if(a=="--overwrite")o.overwrite=true;
    else if(a=="--corrupt-test")o.corrupt=true;
    else if(a=="--help"){
      std::cout<<"jx3d_1tib --start 0 --tiles 64 [--out stream.jxtb] [--overwrite | --resume] [--corrupt-test]\n"
               <<"jx3d_1tib --verify stream.jxtb\n"
               <<"Virtual domain: 8192 x 8192 x 16384 byte voxels (1 TiB)\n"
               <<"Tile: 128 x 128 x 64 (1 MiB); 1,048,576 tiles total\n";
      std::exit(0);
    }
    else throw std::runtime_error("unknown parameter: "+a);
  }
  if(!o.verify.empty()){
    if(!o.output.empty() || o.resume || o.overwrite)throw std::runtime_error("--verify cannot be combined with writing options");
  } else {
    if(o.tiles==0 || o.start>=NTILES || o.tiles>NTILES-o.start)throw std::runtime_error("tile range exceeds virtual 1 TiB geometry");
    if(o.resume && o.overwrite)throw std::runtime_error("--resume and --overwrite conflict");
    if((o.resume || o.overwrite) && o.output.empty())throw std::runtime_error("--resume/--overwrite requires --out");
  }
  return o;
}
// Verify all existing records before appending. This is O(prior stream size) but
// avoids relying on untrusted checkpoints; a production system would use a signed manifest/index.
Stats scan(std::istream& in,std::vector<uint8_t>& raw,std::vector<uint8_t>& residual,
           std::vector<uint8_t>& decoded,uint64_t& start,std::ostream* log){
  Stats s;
  Record r;
  uint64_t expected=NTILES;
  while(read_record(in,r)){
    const uint64_t id=id_of(r);
    if(expected==NTILES){start=id;expected=id;}
    if(id!=expected)throw std::runtime_error("missing, reordered or duplicated tile");
    if(id>=NTILES)throw std::runtime_error("tile outside virtual volume");
    decode(r,residual,decoded);
    generate(raw,id);
    if(decoded!=raw)throw std::runtime_error("tile contents differ from deterministic reference");
    count(s,r);
    expected++;
  }
  if(log)*log<<"verified_existing_tiles="<<s.tiles<<"\n";
  return s;
}
void test_corruption(const Record& original,std::vector<uint8_t>& residual,std::vector<uint8_t>& decoded){
  Record damaged=original;
  if(damaged.payload.empty())damaged.h[24]^=1;
  else damaged.payload[damaged.payload.size()/2]^=1;
  bool rejected=false;
  try{decode(damaged,residual,decoded);}catch(const std::exception&){rejected=true;}
  if(!rejected)throw std::runtime_error("corrupted record incorrectly accepted");
}
void report(const Stats& s,double secs){
  std::cout<<std::fixed<<std::setprecision(4)
           <<"tiles="<<s.tiles<<" raw_bytes="<<s.source<<" encoded_bytes="<<s.encoded<<"\n"
           <<"formats_zero="<<s.zero<<" constant="<<s.constant<<" rle="<<s.rle<<" raw="<<s.raw<<" direct="<<s.direct<<"\n"
           <<"storage_reduction_percent="<<(s.source?(100.0*(1-double(s.encoded)/double(s.source))):0.0)<<"\n"
           <<"elapsed_s="<<secs<<" input_MiB_per_s="<<(secs?double(s.tiles)/secs:0.0)<<"\n";
#ifdef __unix__
  rusage u{};
  if(getrusage(RUSAGE_SELF,&u)==0)std::cout<<"maxrss_kib="<<u.ru_maxrss<<"\n";
#endif
  std::cout<<"verified_reconstruction=PASS\n";
}
} // namespace jx

#ifndef JXTB_NO_MAIN
int main(int argc,char** argv){
  try {
    const auto o=jx::parse(argc,argv);
    std::vector<uint8_t> raw(jx::BLOCK),decoded(jx::BLOCK),residual(jx::BLOCK);
    const auto t0=std::chrono::steady_clock::now();
    jx::Stats stats;
    bool corruption_checked=false;
    if(!o.verify.empty()){
      std::ifstream in(o.verify,std::ios::binary);
      if(!in)throw std::runtime_error("cannot open encoded file");
      uint64_t first=jx::NTILES;
      stats=jx::scan(in,raw,residual,decoded,first,nullptr);
      if(stats.tiles==0)throw std::runtime_error("empty stream");
      std::cout<<"verification_start_tile="<<first<<"\n";
    } else {
      uint64_t existing=0;
      if(!o.output.empty()){
        std::ifstream in(o.output,std::ios::binary);
        if(o.resume){
          if(!in)throw std::runtime_error("resume requires existing output file");
          uint64_t first=jx::NTILES;
          auto scanned=jx::scan(in,raw,residual,decoded,first,&std::cout);
          if(scanned.tiles==0 || first!=o.start || scanned.tiles>o.tiles)
            throw std::runtime_error("resume stream does not match selected tile range");
          existing=scanned.tiles;
          stats=scanned;
        } else if(in && !o.overwrite)throw std::runtime_error("refusing to overwrite existing output; pass --overwrite");
      }
      std::ofstream out;
      if(!o.output.empty()){
        out.open(o.output,std::ios::binary|(o.resume?std::ios::app:std::ios::trunc));
        if(!out)throw std::runtime_error("cannot open output for writing");
      }
      for(uint64_t tile=o.start+existing;tile<o.start+o.tiles;tile++){
        jx::generate(raw,tile);
        auto r=jx::encode(raw,tile,residual);
        jx::decode(r,residual,decoded);
        if(raw!=decoded)throw std::runtime_error("round-trip mismatch");
        if(o.corrupt && !corruption_checked){jx::test_corruption(r,residual,decoded);corruption_checked=true;}
        if(out.is_open())jx::write_record(out,r);
        jx::count(stats,r);
      }
      if(out.is_open()){out.flush();if(!out)throw std::runtime_error("failed to finalize encoded file");}
      if(o.corrupt && !corruption_checked){
        // An entirely resumed file may require testing its last validated record;
        // here we use a deterministically generated representative tile.
        jx::generate(raw,o.start);
        auto r=jx::encode(raw,o.start,residual);
        jx::test_corruption(r,residual,decoded);
      }
      std::cout<<"start_tile="<<o.start<<" corruption_test="<<(o.corrupt?"PASS":"NOT_RUN")<<"\n";
    }
    const auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    jx::report(stats,sec);
    return 0;
  } catch(const std::exception& e){
    std::cerr<<"ERROR: "<<e.what()<<"\n";
    return 1;
  }
}

#endif // JXTB_NO_MAIN