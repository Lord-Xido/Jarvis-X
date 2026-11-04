// JARVIS-X JX3D v1: sparse 3D bytecode ANN streaming reference runtime.
// C++17, no external dependencies. Educational/research reference, not production security.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace jx3d {
constexpr std::uint64_t AXIS = 1'000'000'000'000ULL; // 1000 GB (decimal) addressable positions / axis
constexpr std::size_t SIDE = 16;
constexpr std::size_t VOXELS = SIDE * SIDE * SIDE;
constexpr std::uint64_t TILE_AXIS = AXIS / SIDE;
constexpr std::size_t HIDDEN = 8;
constexpr std::size_t INPUTS = 3; // causal neighbor voxels x-1, y-1, z-1
constexpr std::size_t PARAMS = INPUTS * HIDDEN + HIDDEN + HIDDEN + 1;
constexpr std::int64_t SCALE = 65536;

using Tile = std::array<std::uint8_t, VOXELS>;
using Params = std::array<std::int32_t, PARAMS>;
struct Coord { std::uint64_t x, y, z; };
struct Record { Coord p; Tile vox{}; };
struct Statistics { std::uint64_t tiles=0, zeros=0, raw=0, rle=0, payload_bytes=0; };

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void put8(std::ostream& o, std::uint8_t v) { o.put(static_cast<char>(v)); if (!o) fail("write failure"); }
void put32(std::ostream& o, std::uint32_t v) { for (int i=0;i<4;++i) put8(o,static_cast<std::uint8_t>(v>>(i*8))); }
void put64(std::ostream& o, std::uint64_t v) { for (int i=0;i<8;++i) put8(o,static_cast<std::uint8_t>(v>>(i*8))); }
std::uint8_t get8(std::istream& in) { int v=in.get(); if(v==EOF) fail("truncated input"); return static_cast<std::uint8_t>(v); }
std::uint32_t get32(std::istream& in) { std::uint32_t r=0; for(int i=0;i<4;++i) r|=static_cast<std::uint32_t>(get8(in))<<(8*i); return r; }
std::uint64_t get64(std::istream& in) { std::uint64_t r=0; for(int i=0;i<8;++i) r|=static_cast<std::uint64_t>(get8(in))<<(8*i); return r; }
std::int32_t as_signed(std::uint32_t u) { // standard portable signed int32 decode
    if (u <= static_cast<std::uint32_t>(INT32_MAX)) return static_cast<std::int32_t>(u);
    return static_cast<std::int32_t>(static_cast<std::int64_t>(u)-4'294'967'296LL);
}
void check_coord(const Coord& c) {
    if(c.x>=TILE_AXIS || c.y>=TILE_AXIS || c.z>=TILE_AXIS) fail("tile coordinate outside 1000 GB cube");
}
std::size_t idx(std::size_t x,std::size_t y,std::size_t z) { return (z*SIDE+y)*SIDE+x; }
std::uint32_t crc32(const Tile& t) { // corruption detection, NOT authentication
    std::uint32_t c=0xffffffffu;
    for(auto b:t) { c^=b; for(int j=0;j<8;++j) c=(c>>1)^(0xedb88320u & (0u-(c&1u))); }
    return ~c;
}
void write_raw_record(std::ostream& o,const Record& r) {
    check_coord(r.p); put64(o,r.p.x); put64(o,r.p.y); put64(o,r.p.z);
    o.write(reinterpret_cast<const char*>(r.vox.data()), static_cast<std::streamsize>(VOXELS));
    if(!o) fail("raw record write failure");
}
bool read_raw_record(std::istream& in, Record& r) {
    int lead=in.peek(); if(lead==EOF) { if(in.eof())return false; fail("raw stream read error"); }
    r.p={get64(in),get64(in),get64(in)}; check_coord(r.p);
    in.read(reinterpret_cast<char*>(r.vox.data()),static_cast<std::streamsize>(VOXELS));
    if(!in) fail("truncated raw tile");
    return true;
}

struct ANN {
    Params p{};
    // Parameters encoded Q16.16, inference uses integer arithmetic throughout.
    std::uint8_t predict(std::uint8_t left,std::uint8_t up,std::uint8_t back) const {
        const std::array<std::int64_t,3> x={
            static_cast<std::int64_t>(left)*SCALE/255,
            static_cast<std::int64_t>(up)*SCALE/255,
            static_cast<std::int64_t>(back)*SCALE/255};
        std::array<std::int64_t,HIDDEN> h{};
        for(std::size_t k=0;k<HIDDEN;++k) {
            std::int64_t a=p[INPUTS*HIDDEN+k];
            for(std::size_t j=0;j<INPUTS;++j) a+=static_cast<std::int64_t>(p[k*INPUTS+j])*x[j]/SCALE;
            h[k]=std::clamp<std::int64_t>(a,0,2*SCALE);
        }
        std::int64_t out=p[PARAMS-1];
        for(std::size_t k=0;k<HIDDEN;++k)
            out+=static_cast<std::int64_t>(p[INPUTS*HIDDEN+HIDDEN+k])*h[k]/SCALE;
        out=std::clamp<std::int64_t>(out,0,SCALE);
        return static_cast<std::uint8_t>((out*255+SCALE/2)/SCALE);
    }
};

// Bytecode has distinct encode/decode programs; the switch dispatches actual operations.
enum class Op : std::uint8_t { ANN_XOR_ENCODE=0x20, RLE_PACK=0x30, SELECT_CODEC=0x31,
                               RLE_UNPACK=0x32, ANN_XOR_DECODE=0x40, VERIFY_CRC=0x50 };
enum class Codec : std::uint8_t { ZERO=0, RAW=1, ANN_XOR_RLE=2 };
struct Frame { Tile raw{}; Tile residual{}; std::vector<std::uint8_t> payload; Codec codec=Codec::RAW; std::uint32_t expected_crc=0; };

// Each tile is independently reversible. Causal context is reconstructed in scan order.
void ann_xor_encode(Frame& f,const ANN& ann) {
    for(std::size_t z=0;z<SIDE;++z)
      for(std::size_t y=0;y<SIDE;++y)
        for(std::size_t x=0;x<SIDE;++x) {
          std::size_t i=idx(x,y,z);
          auto a=x?f.raw[idx(x-1,y,z)]:0;
          auto b=y?f.raw[idx(x,y-1,z)]:0;
          auto c=z?f.raw[idx(x,y,z-1)]:0;
          f.residual[i]=f.raw[i]^ann.predict(a,b,c);
        }
}
void ann_xor_decode(Frame& f,const ANN& ann) {
    for(std::size_t z=0;z<SIDE;++z)
      for(std::size_t y=0;y<SIDE;++y)
        for(std::size_t x=0;x<SIDE;++x) {
          std::size_t i=idx(x,y,z);
          auto a=x?f.raw[idx(x-1,y,z)]:0;
          auto b=y?f.raw[idx(x,y-1,z)]:0;
          auto c=z?f.raw[idx(x,y,z-1)]:0;
          f.raw[i]=f.residual[i]^ann.predict(a,b,c);
        }
}
std::vector<std::uint8_t> rle_pack(const Tile& tile) {
    std::vector<std::uint8_t> o;
    o.reserve(VOXELS/2);
    for(std::size_t i=0;i<VOXELS;) {
        std::size_t n=1;
        while(i+n<VOXELS && n<255 && tile[i+n]==tile[i]) ++n;
        o.push_back(static_cast<std::uint8_t>(n)); o.push_back(tile[i]); i+=n;
    }
    return o;
}
Tile rle_unpack(const std::vector<std::uint8_t>& payload) {
    if(payload.size()%2) fail("invalid RLE payload length");
    Tile t{}; std::size_t pos=0;
    for(std::size_t i=0;i<payload.size();i+=2) {
        std::uint8_t count=payload[i]; if(!count||pos+count>VOXELS) fail("invalid RLE count");
        std::fill_n(t.begin()+static_cast<std::ptrdiff_t>(pos),count,payload[i+1]);pos+=count;
    }
    if(pos!=VOXELS) fail("incomplete RLE decoded tile");
    return t;
}
void run_encode_vm(Frame& f,const ANN& model) {
    constexpr std::array<Op,3> PROGRAM={Op::ANN_XOR_ENCODE,Op::RLE_PACK,Op::SELECT_CODEC};
    for(Op op:PROGRAM) switch(op) {
        case Op::ANN_XOR_ENCODE: ann_xor_encode(f,model); break;
        case Op::RLE_PACK: f.payload=rle_pack(f.residual); break;
        case Op::SELECT_CODEC:
          if(std::all_of(f.raw.begin(),f.raw.end(),[](auto b){return b==0;})) {
             f.codec=Codec::ZERO;f.payload.clear();
          } else if(f.payload.size()<VOXELS) f.codec=Codec::ANN_XOR_RLE;
          else {f.codec=Codec::RAW;f.payload.assign(f.raw.begin(),f.raw.end());}
          break;
        default: fail("illegal encoder opcode");
    }
}
void run_decode_vm(Frame& f,const ANN& model,const std::array<Op,3>& program) {
    for(Op op:program) switch(op) {
        case Op::RLE_UNPACK:
          if(f.codec==Codec::RAW) {
              if(f.payload.size()!=VOXELS) fail("invalid raw payload length");
              std::copy(f.payload.begin(),f.payload.end(),f.raw.begin());
          } else if(f.codec==Codec::ZERO) {
              if(!f.payload.empty()) fail("invalid zero payload length");
              f.raw.fill(0);
          } else if(f.codec==Codec::ANN_XOR_RLE) f.residual=rle_unpack(f.payload);
          else fail("unknown codec");
          break;
        case Op::ANN_XOR_DECODE:
          if(f.codec==Codec::ANN_XOR_RLE) ann_xor_decode(f,model);
          break;
        case Op::VERIFY_CRC:
          if(crc32(f.raw)!=f.expected_crc) fail("CRC32 mismatch: damaged tile or incompatible decoder");
          break;
        default: fail("illegal decoder opcode");
    }
}

void write_model(std::ostream& o,const ANN& m) {
    for(auto p:m.p) put32(o,static_cast<std::uint32_t>(p));
}
ANN read_model(std::istream& in) {
    ANN ann;for(auto& p:ann.p)p=as_signed(get32(in));return ann;
}
void save_model(const std::string& path,const ANN& ann) {
    std::ofstream o(path,std::ios::binary);if(!o)fail("cannot write model: "+path);
    o.write("JXANN1",6);write_model(o,ann);
}
ANN load_model(const std::string& path) {
    std::ifstream in(path,std::ios::binary);if(!in)fail("cannot open model: "+path);
    char hdr[6];in.read(hdr,6);if(!in||std::string(hdr,6)!="JXANN1")fail("invalid model header");
    ANN ann=read_model(in);if(in.peek()!=EOF)fail("model trailing data");return ann;
}
ANN default_model() {
    ANN m;
    // True 3->8->1 ReLU ANN, fixed deterministic initial weights. Train to improve predictions.
    // Several positive hidden units approximate an average of 3 spatial neighbors.
    for(std::size_t k=0;k<HIDDEN;++k) {
        for(std::size_t j=0;j<INPUTS;++j)
            m.p[k*INPUTS+j]=static_cast<std::int32_t>((k<3 && k==j ? 1.0 : (k>=3 ? 0.11:0.0))*SCALE);
        m.p[INPUTS*HIDDEN+k]=static_cast<std::int32_t>((k>=3?0.02:0.0)*SCALE);
        m.p[INPUTS*HIDDEN+HIDDEN+k]=static_cast<std::int32_t>((k<3?0.23:0.02)*SCALE);
    }
    m.p[PARAMS-1]=static_cast<std::int32_t>(0.03*SCALE);
    return m;
}

struct Example { std::array<double,INPUTS> in; double target; };
ANN train(const std::string& tile_file) {
    std::ifstream in(tile_file,std::ios::binary);if(!in)fail("cannot open training data");
    std::vector<Example> examples;examples.reserve(12000);
    Record r;
    while(examples.size()<12000 && read_raw_record(in,r)) {
        for(std::size_t z=0;z<SIDE && examples.size()<12000;++z)
          for(std::size_t y=0;y<SIDE && examples.size()<12000;++y)
            for(std::size_t x=0;x<SIDE && examples.size()<12000;++x) {
              std::size_t i=idx(x,y,z);
              if(i%7!=0)continue;
              examples.push_back({{
                (x?r.vox[idx(x-1,y,z)]:0)/255.0,
                (y?r.vox[idx(x,y-1,z)]:0)/255.0,
                (z?r.vox[idx(x,y,z-1)]:0)/255.0},r.vox[i]/255.0});
            }
    }
    if(examples.empty())fail("no training examples");
    std::array<std::array<double,INPUTS>,HIDDEN> w{};
    std::array<double,HIDDEN> b{},v{};
    double outb=0.03;
    for(std::size_t k=0;k<HIDDEN;++k) {
        for(std::size_t j=0;j<INPUTS;++j) w[k][j]=(k<3 && k==j)?1.0:(k>=3?0.11:0.0);
        b[k]=k>=3?0.02:0.0; v[k]=k<3?0.23:0.02;
    }
    // Bounded sequential SGD; deterministic order; float training need not be bit-identical across hardware.
    for(int epoch=0;epoch<12;++epoch) {
      double eta=0.012/(1+epoch*0.2);
      for(const auto& e:examples) {
        std::array<double,HIDDEN> pre{},h{},oldv=v;
        for(std::size_t k=0;k<HIDDEN;++k) {
          pre[k]=b[k];for(std::size_t j=0;j<INPUTS;++j)pre[k]+=w[k][j]*e.in[j];
          h[k]=std::max(0.0,pre[k]);
        }
        double predicted=outb;
        for(std::size_t k=0;k<HIDDEN;++k)predicted+=v[k]*h[k];
        double grad=std::clamp(2*(predicted-e.target),-2.0,2.0);
        for(std::size_t k=0;k<HIDDEN;++k) {
          v[k]=std::clamp(v[k]-eta*grad*h[k],-3.0,3.0);
          double hidden_grad=grad*oldv[k]*(pre[k]>0?1.0:0.0);
          for(std::size_t j=0;j<INPUTS;++j)
            w[k][j]=std::clamp(w[k][j]-eta*hidden_grad*e.in[j],-3.0,3.0);
          b[k]=std::clamp(b[k]-eta*hidden_grad,-3.0,3.0);
        }
        outb=std::clamp(outb-eta*grad,-3.0,3.0);
      }
    }
    ANN result;
    auto quant=[](double d)->std::int32_t {return static_cast<std::int32_t>(std::llround(std::clamp(d,-3.0,3.0)*SCALE));};
    for(std::size_t k=0;k<HIDDEN;++k) {
        for(std::size_t j=0;j<INPUTS;++j)result.p[k*INPUTS+j]=quant(w[k][j]);
        result.p[INPUTS*HIDDEN+k]=quant(b[k]);
        result.p[INPUTS*HIDDEN+HIDDEN+k]=quant(v[k]);
    }
    result.p[PARAMS-1]=quant(outb);
    std::cerr<<"trained examples="<<examples.size()<<" epochs=12 parameters="<<PARAMS<<"\n";
    return result;
}
constexpr std::array<Op,3> DECODE_BYTECODE={Op::RLE_UNPACK,Op::ANN_XOR_DECODE,Op::VERIFY_CRC};
struct StreamHeader { ANN model; std::array<Op,3> bytecode; };
void write_header(std::ostream& o,const ANN& ann) {
    o.write("JX3Dv1",6);put8(o,static_cast<std::uint8_t>(SIDE));
    put64(o,AXIS);put64(o,AXIS);put64(o,AXIS);
    put32(o,static_cast<std::uint32_t>(PARAMS));write_model(o,ann);
    put8(o,static_cast<std::uint8_t>(DECODE_BYTECODE.size()));
    for(auto op:DECODE_BYTECODE)put8(o,static_cast<std::uint8_t>(op));
}
StreamHeader read_header(std::istream& in) {
    char hdr[6];in.read(hdr,6);if(!in||std::string(hdr,6)!="JX3Dv1")fail("invalid JX3D magic/version");
    if(get8(in)!=SIDE)fail("unsupported tile side");
    for(int i=0;i<3;++i)if(get64(in)!=AXIS)fail("unsupported axis extent");
    if(get32(in)!=PARAMS)fail("unsupported ANN parameter count");
    StreamHeader h;h.model=read_model(in);
    if(get8(in)!=DECODE_BYTECODE.size())fail("unsupported bytecode length");
    for(std::size_t i=0;i<DECODE_BYTECODE.size();++i) {
        auto opcode=static_cast<Op>(get8(in));
        if(opcode!=DECODE_BYTECODE[i])fail("unsupported or unsafe bytecode program");
        h.bytecode[i]=opcode;
    }
    return h;
}
void stats_line(const Statistics& s) {
    std::cout<<"tiles="<<s.tiles<<" zero="<<s.zeros<<" raw="<<s.raw
      <<" ann_rle="<<s.rle<<" payload_bytes="<<s.payload_bytes<<"\n";
}
void encode(const std::string& src,const std::string& dst,const ANN& ann) {
    std::ifstream in(src,std::ios::binary);if(!in)fail("cannot open source tiles");
    std::ofstream o(dst,std::ios::binary);if(!o)fail("cannot open encoded output");
    write_header(o,ann);Record r; Statistics s;
    while(read_raw_record(in,r)) {
        Frame f;f.raw=r.vox;run_encode_vm(f,ann);
        put8(o,'T');put64(o,r.p.x);put64(o,r.p.y);put64(o,r.p.z);
        put8(o,static_cast<std::uint8_t>(f.codec));
        put32(o,static_cast<std::uint32_t>(f.payload.size()));put32(o,crc32(r.vox));
        if(!f.payload.empty())o.write(reinterpret_cast<const char*>(f.payload.data()),static_cast<std::streamsize>(f.payload.size()));
        if(!o)fail("encoded tile write error");
        ++s.tiles;s.payload_bytes+=f.payload.size();
        if(f.codec==Codec::ZERO)++s.zeros;else if(f.codec==Codec::RAW)++s.raw;else ++s.rle;
    }
    put8(o,'E');put64(o,s.tiles);o.flush();if(!o)fail("failed final encoded output");
    stats_line(s);
}
void decode(const std::string& src,const std::string& dst) {
    std::ifstream in(src,std::ios::binary);if(!in)fail("cannot open encoded input");
    std::ofstream o(dst,std::ios::binary);if(!o)fail("cannot open decoded output");
    StreamHeader header=read_header(in);Statistics s;
    for(;;) {
        std::uint8_t marker=get8(in);
        if(marker=='E') { if(get64(in)!=s.tiles)fail("tile count footer mismatch");break; }
        if(marker!='T')fail("invalid tile marker");
        Record r;r.p={get64(in),get64(in),get64(in)};check_coord(r.p);
        Frame f;f.codec=static_cast<Codec>(get8(in));
        auto n=get32(in);f.expected_crc=get32(in);
        if(n>2*VOXELS)fail("payload length exceeds tile maximum");
        f.payload.resize(n);
        if(n)in.read(reinterpret_cast<char*>(f.payload.data()),n);
        if(!in)fail("truncated tile payload");
        run_decode_vm(f,header.model,header.bytecode);r.vox=f.raw;write_raw_record(o,r);
        ++s.tiles;s.payload_bytes+=n;
        if(f.codec==Codec::ZERO)++s.zeros;else if(f.codec==Codec::RAW)++s.raw;else ++s.rle;
    }
    if(in.peek()!=EOF)fail("trailing stream data");
    o.flush();
    if(!o)fail("decoded output write failed");
    stats_line(s);
}
std::uint32_t lcg(std::uint32_t& s) {s=s*1664525u+1013904223u;return s;}
void generate(const std::string& path,std::size_t count) {
    if(count>100000)fail("reference generator maximum is 100000 tiles");
    std::ofstream o(path,std::ios::binary);if(!o)fail("cannot open generated output");
    for(std::size_t i=0;i<count;++i) {
        Record r;
        r.p={ (1000ULL+i*1315423911ULL)%TILE_AXIS,
              (2000ULL+i*2654435761ULL)%TILE_AXIS,
              (3000ULL+i*362437ULL)%TILE_AXIS};
        std::uint32_t state=static_cast<std::uint32_t>(i+42);
        for(std::size_t z=0;z<SIDE;++z)
          for(std::size_t y=0;y<SIDE;++y)
            for(std::size_t x=0;x<SIDE;++x) {
                std::uint8_t value=0;
                switch(i%5) {
                    case 0: value=0;break;
                    case 1: value=static_cast<std::uint8_t>(i%256);break;
                    case 2: value=static_cast<std::uint8_t>((x*5+y*3+z*2+i)%256);break;
                    case 3: value=static_cast<std::uint8_t>((x/4+y/4+z/4)%2?220:20);break;
                    case 4: value=static_cast<std::uint8_t>(lcg(state)>>24);break;
                }
                r.vox[idx(x,y,z)]=value;
            }
        write_raw_record(o,r);
    }
    std::cout<<"generated tiles="<<count<<" raw_voxel_bytes="<<count*VOXELS<<"\n";
}
void verify(const std::string& a,const std::string& b) {
    std::ifstream in1(a,std::ios::binary),in2(b,std::ios::binary);
    if(!in1||!in2)fail("cannot open verification files");
    Record r1,r2;std::uint64_t tiles=0;
    for(;;) {
        bool x=read_raw_record(in1,r1),y=read_raw_record(in2,r2);
        if(x!=y)fail("tile count mismatch");
        if(!x)break;
        if(r1.p.x!=r2.p.x||r1.p.y!=r2.p.y||r1.p.z!=r2.p.z||r1.vox!=r2.vox)
            fail("round-trip mismatch at tile "+std::to_string(tiles));
        ++tiles;
    }
    std::cout<<"PASS byte-exact reconstruction: tiles="<<tiles<<" voxels="<<tiles*VOXELS<<"\n";
}
void usage() {
    std::cout<<"JX3D v1 | 3D sparse ANN bytecode streaming\n"
      <<"  jx3d generate <tiles.bin> <tile_count>\n"
      <<"  jx3d train    <tiles.bin> <weights.bin>\n"
      <<"  jx3d encode   <tiles.bin> <stream.jx3d> [weights.bin]\n"
      <<"  jx3d decode   <stream.jx3d> <recovered.bin>\n"
      <<"  jx3d verify   <tiles.bin> <recovered.bin>\n"
      <<"Raw tiles are repeated [u64 tile_x,y,z LE][4096 raw bytes].\n"
      <<"Virtual axis positions: 0..999999999999, tile side=16.\n";
}
} // namespace jx3d
int main(int argc,char** argv) {
    try {
      using namespace jx3d;
      if(argc<2) {usage();return 1;}
      std::string op=argv[1];
      if(op=="generate" && argc==4) {
        std::string s=argv[3];std::size_t pos=0;
        auto count=std::stoull(s,&pos);if(pos!=s.size()||count>100000)fail("invalid tile count");
        generate(argv[2],static_cast<std::size_t>(count));
      } else if(op=="train" && argc==4) save_model(argv[3],train(argv[2]));
      else if(op=="encode" && (argc==4||argc==5)) encode(argv[2],argv[3],argc==5?load_model(argv[4]):default_model());
      else if(op=="decode" && argc==4) decode(argv[2],argv[3]);
      else if(op=="verify" && argc==4) verify(argv[2],argv[3]);
      else {usage();return 1;}
      return 0;
    } catch(const std::exception& ex) {std::cerr<<"ERROR: "<<ex.what()<<"\n";return 2;}
}