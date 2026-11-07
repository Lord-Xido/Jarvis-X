// Jarvis-X / isolated DM3D-GEOM1 reference (C++17).
// Real 3D trilinear brick geometry; optional exact XOR residual; verified inward feedback.
// Not a neural network, not a general-purpose compressor, and not an authoritative state format.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dm3dgeom {
using Byte = std::uint8_t;
using Bytes = std::vector<Byte>;
static constexpr std::array<Byte,8> MAGIC{{'D','M','3','D','T','R','I','1'}};
static constexpr unsigned VERSION = 1;
static constexpr std::size_t HEADER = 26;

void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void valid(unsigned side, unsigned brick) {
    require(side >= 2 && side <= 64 && brick >= 2 && brick <= 8 && side % brick == 0,
            "side must be 2..64, brick 2..8, side divisible by brick");
}
std::size_t index3(unsigned x, unsigned y, unsigned z, unsigned side) {
    return static_cast<std::size_t>(x) + side * (static_cast<std::size_t>(y) + side * static_cast<std::size_t>(z));
}
std::size_t voxels(unsigned side) { return static_cast<std::size_t>(side)*side*side; }
std::size_t brick_count(unsigned side, unsigned brick) { auto b=side/brick; return static_cast<std::size_t>(b)*b*b; }

struct Packet { unsigned side = 0, brick = 0; Bytes corners, residual; bool lossless = false; };

// The corner encoding is a genuine spatial map: eight samples for every 3D brick.
Bytes geometry_decode(unsigned side, unsigned brick, const Bytes& corners) {
    valid(side, brick);
    require(corners.size() == brick_count(side,brick)*8, "corner count mismatch");
    Bytes output(voxels(side),0);
    const unsigned groups = side / brick, e = brick - 1;
    const std::uint64_t denom = static_cast<std::uint64_t>(e)*e*e;
    for(unsigned bz=0;bz<groups;++bz) for(unsigned by=0;by<groups;++by)
    for(unsigned bx=0;bx<groups;++bx) {
        const std::size_t b = static_cast<std::size_t>(bx) + groups*(static_cast<std::size_t>(by)+groups*bz);
        for(unsigned z=0;z<brick;++z) for(unsigned y=0;y<brick;++y) for(unsigned x=0;x<brick;++x) {
            std::uint64_t sum=0;
            for(unsigned k=0;k<8;++k) {
                const unsigned wx=(k&1U)?x:e-x;
                const unsigned wy=(k&2U)?y:e-y;
                const unsigned wz=(k&4U)?z:e-z;
                sum += static_cast<std::uint64_t>(corners[8*b+k])*wx*wy*wz;
            }
            const unsigned value=static_cast<unsigned>((sum+denom/2)/denom);
            output[index3(bx*brick+x,by*brick+y,bz*brick+z,side)] = static_cast<Byte>(value);
        }
    }
    return output;
}
Packet encode(unsigned side,unsigned brick,const Bytes& input,bool lossless) {
    valid(side,brick); require(input.size()==voxels(side),"input size mismatch");
    Packet packet; packet.side=side; packet.brick=brick; packet.lossless=lossless;
    const unsigned groups=side/brick;
    packet.corners.reserve(brick_count(side,brick)*8);
    for(unsigned bz=0;bz<groups;++bz) for(unsigned by=0;by<groups;++by)
    for(unsigned bx=0;bx<groups;++bx) for(unsigned k=0;k<8;++k) {
        const unsigned x=bx*brick+((k&1U)?brick-1:0);
        const unsigned y=by*brick+((k&2U)?brick-1:0);
        const unsigned z=bz*brick+((k&4U)?brick-1:0);
        packet.corners.push_back(input[index3(x,y,z,side)]);
    }
    if(lossless) {
        const Bytes approximation=geometry_decode(side,brick,packet.corners);
        packet.residual.resize(input.size());
        for(std::size_t i=0;i<input.size();++i) packet.residual[i]=input[i]^approximation[i];
    }
    return packet;
}
Bytes decode(const Packet& packet) {
    valid(packet.side,packet.brick);
    require(packet.lossless == !packet.residual.empty(), "residual flag mismatch");
    Bytes output=geometry_decode(packet.side,packet.brick,packet.corners);
    if(packet.lossless) {
        require(packet.residual.size()==output.size(),"residual count mismatch");
        for(std::size_t i=0;i<output.size();++i) output[i]^=packet.residual[i];
    }
    return output;
}
void put16(Bytes& b,unsigned value){b.push_back(static_cast<Byte>(value>>8));b.push_back(static_cast<Byte>(value));}
void put32(Bytes& b,std::uint32_t value){for(int shift=24;shift>=0;shift-=8)b.push_back(static_cast<Byte>(value>>shift));}
unsigned get16(const Bytes& b,std::size_t off){return (static_cast<unsigned>(b.at(off))<<8) | b.at(off+1);}
std::uint32_t get32(const Bytes& b,std::size_t off){return (static_cast<std::uint32_t>(b.at(off))<<24)|(static_cast<std::uint32_t>(b.at(off+1))<<16)|(static_cast<std::uint32_t>(b.at(off+2))<<8)|b.at(off+3);}
std::uint32_t crc_update(std::uint32_t crc,Byte c){
    crc^=c; for(int k=0;k<8;++k)crc=(crc>>1)^((crc&1U)?0xEDB88320U:0U); return crc;
}
std::uint32_t checksum(const Bytes& b){
    std::uint32_t crc=0xFFFFFFFFU;
    // Exclude checksum field bytes at offsets [22,26); include the header and payload.
    for(std::size_t i=0;i<b.size();++i) if(i<22 || i>=HEADER)crc=crc_update(crc,b[i]);
    return ~crc;
}
Bytes serialize(const Packet& packet){
    valid(packet.side,packet.brick);
    require(packet.corners.size()==brick_count(packet.side,packet.brick)*8,"corner count mismatch");
    require(packet.residual.size()==(packet.lossless?voxels(packet.side):0),"residual count mismatch");
    Bytes b; b.reserve(HEADER+packet.corners.size()+packet.residual.size());
    b.insert(b.end(),MAGIC.begin(),MAGIC.end());put16(b,VERSION);put16(b,packet.side);
    b.push_back(static_cast<Byte>(packet.brick));b.push_back(static_cast<Byte>(packet.lossless?1:0));
    put32(b,static_cast<std::uint32_t>(packet.corners.size()));
    put32(b,static_cast<std::uint32_t>(packet.residual.size()));put32(b,0);
    b.insert(b.end(),packet.corners.begin(),packet.corners.end());
    b.insert(b.end(),packet.residual.begin(),packet.residual.end());
    const std::uint32_t crc=checksum(b);
    for(int k=0;k<4;++k)b[22+static_cast<std::size_t>(k)]=static_cast<Byte>(crc>>(24-8*k));
    return b;
}
Packet parse(const Bytes& b){
    require(b.size()>=HEADER,"truncated header");
    require(std::equal(MAGIC.begin(),MAGIC.end(),b.begin()),"wrong magic");
    require(get16(b,8)==VERSION,"wrong version");
    const unsigned side=get16(b,10),brick=b[12],flags=b[13];
    valid(side,brick);require(flags<=1,"invalid flags");
    const std::size_t nc=get32(b,14),nr=get32(b,18);
    require(nc==brick_count(side,brick)*8,"invalid corner length");
    require(nr==(flags?voxels(side):0),"invalid residual length");
    require(nc+nr+HEADER==b.size(),"trailing or truncated payload");
    require(get32(b,22)==checksum(b),"CRC32 mismatch");
    Packet p; p.side=side;p.brick=brick;p.lossless=flags!=0;
    p.corners.assign(b.begin()+static_cast<std::ptrdiff_t>(HEADER),b.begin()+static_cast<std::ptrdiff_t>(HEADER+nc));
    p.residual.assign(b.begin()+static_cast<std::ptrdiff_t>(HEADER+nc),b.end());
    return p;
}
Bytes synthetic(unsigned side) {
    Bytes b(voxels(side));
    for(unsigned z=0;z<side;++z)for(unsigned y=0;y<side;++y)for(unsigned x=0;x<side;++x) {
        const unsigned value= (17*x + 11*y + 7*z + ((x*y+z*z)%19)*5 + ((x+2*y+3*z)%7)*9) % 256;
        b[index3(x,y,z,side)]=static_cast<Byte>(value);
    }
    return b;
}
double mse(const Bytes& a,const Bytes& b){
    require(a.size()==b.size(),"mse length mismatch");
    double sum=0;
    for(std::size_t i=0;i<a.size();++i){double d=double(a[i])-b[i];sum+=d*d;}
    return sum/static_cast<double>(a.size());
}
struct Pass { unsigned index;double mse_initial,mse_previous;bool fixed,period_two; };
std::vector<Pass> feedback(unsigned side,unsigned brick,const Bytes& original,unsigned passes) {
    require(passes>=1 && passes<=100,"passes must be 1..100");
    Bytes previous=original, before_previous;
    std::vector<Pass> out;
    for(unsigned t=1;t<=passes;++t){
        const Packet packet=encode(side,brick,previous,false);
        const Bytes current=decode(packet);
        const bool fixed=current==previous;
        const bool period_two=t>=2 && current==before_previous && !fixed;
        out.push_back({t,mse(original,current),mse(previous,current),fixed,period_two});
        before_previous=std::move(previous);
        previous=current;
    }
    return out;
}
Bytes file_read(const std::string& path) {
    std::ifstream f(path,std::ios::binary);require(bool(f),"cannot open input file");
    return Bytes(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>());
}
void file_write(const std::string& path,const Bytes& data) {
    std::ofstream f(path,std::ios::binary);require(bool(f),"cannot open output file");
    f.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));
    require(bool(f),"write error");
}
void self_test(){
    unsigned checks=0;
    auto check=[&](bool ok,const char* why){require(ok,why);++checks;};
    const unsigned n=8,brick=4;
    Bytes affine(voxels(n));
    for(unsigned z=0;z<n;++z)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x)
        affine[index3(x,y,z,n)]=static_cast<Byte>(11+2*x+3*y+4*z);
    const Packet a=encode(n,brick,affine,false);
    check(decode(a)==affine,"multiaffine geometric reconstruction");
    const Bytes nonlinear=synthetic(n);
    const Packet exact=encode(n,brick,nonlinear,true);
    check(decode(exact)==nonlinear,"residual XOR exact roundtrip");
    const Bytes bytes=serialize(exact);
    check(decode(parse(bytes))==nonlinear,"serialized roundtrip");
    check(bytes.size()==HEADER+brick_count(n,brick)*8+voxels(n),"wire size");
    Bytes corrupt=bytes;corrupt.back()^=1;
    try { (void)parse(corrupt);require(false,"corrupted packet accepted"); } catch(const std::runtime_error& e) {
        check(std::string(e.what())=="CRC32 mismatch","corruption rejection");
    }
    Bytes appended=bytes;appended.push_back(0);
    try{(void)parse(appended);require(false,"trailing byte accepted");}catch(const std::runtime_error& e){
        check(std::string(e.what())=="trailing or truncated payload","trailing data rejection");
    }
    const Bytes first=decode(encode(n,brick,nonlinear,false));
    const Bytes second=decode(encode(n,brick,first,false));
    check(first==second,"genuine geometry-dependent fixed-point idempotence");
    const auto passes=feedback(n,brick,nonlinear,6);
    check(passes.size()==6 && passes[1].fixed,"feedback fixed point");
    check(!passes[1].period_two,"no false oscillation classification");
    // Geometry sensitivity: a distinct tensor with equal flattened values is NOT a valid geometry profile.
    check(index3(1,2,3,8)==209,"row-major 3D mapping");
    // Explicit varying-side input & arbitrary byte patterns.
    for(unsigned side:{4U,8U,16U}) {
        Bytes input(voxels(side));
        for(std::size_t i=0;i<input.size();++i) input[i]=static_cast<Byte>((i*73U+i/7U)%256U);
        const Packet p=encode(side,4,input,true);
        check(decode(parse(serialize(p)))==input,"cross-size exact bytes");
    }
    std::cout<<"SELF_TEST_OK checks="<<checks<<"\n";
}
void run(unsigned side,unsigned brick,unsigned passes,bool lossless,const std::string& input_path,const std::string& prefix){
    valid(side,brick);
    const Bytes input=input_path.empty()?synthetic(side):file_read(input_path);
    require(input.size()==voxels(side),"input file must contain exactly side^3 raw U8 voxels");
    const auto t0=std::chrono::steady_clock::now();
    const Packet packet=encode(side,brick,input,lossless);
    const Bytes wire=serialize(packet);
    const Packet recovered_packet=parse(wire);
    const Bytes recovered=decode(recovered_packet);
    const Bytes approximation=geometry_decode(side,brick,packet.corners);
    require(recovered==(lossless?input:approximation),"roundtrip invariant failed");
    const auto passes_data=feedback(side,brick,input,passes);
    const double milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
    const double error=mse(input,approximation);
    if(!prefix.empty()){
        file_write(prefix+".dm3dtri",wire);
        file_write(prefix+".recovered.u8",recovered);
    }
    std::cout<<std::fixed<<std::setprecision(9);
    std::cout<<"{\n \"schema\":\"dm3d-geometric-loop-receipt-v1\",\n \"domain\":\"3d-u8-brick\","
        <<"\n \"side\":"<<side<<",\n \"brick\":"<<brick
        <<",\n \"voxels\":"<<input.size()<<",\n \"wire_bytes\":"<<wire.size()
        <<",\n \"latent_bytes\":"<<packet.corners.size()
        <<",\n \"residual_bytes\":"<<packet.residual.size()
        <<",\n \"lossless\":"<<(lossless?"true":"false")
        <<",\n \"byte_identical_recovery\":"<<(recovered==input?"true":"false")
        <<",\n \"first_pass_mse\":"<<error<<",\n \"elapsed_ms\":"<<milliseconds<<",\n \"passes\":[\n";
    for(std::size_t i=0;i<passes_data.size();++i){ const auto& p=passes_data[i];
        std::cout<<"  {\"iteration\":"<<p.index<<",\"mse_to_input\":"<<p.mse_initial
            <<",\"mse_to_previous\":"<<p.mse_previous<<",\"fixed_point\":"
            <<(p.fixed?"true":"false")<<",\"period_two\":"<<(p.period_two?"true":"false")<<"}"
            <<(i+1<passes_data.size()?",":"")<<"\n";
    }
    std::cout<<" ],\n \"sampled_voxels\":[\n";
    const unsigned stride=std::max(1U,side/8);
    bool first=true;
    for(unsigned z=0;z<side;z+=stride)for(unsigned y=0;y<side;y+=stride)for(unsigned x=0;x<side;x+=stride){
        const std::size_t id=index3(x,y,z,side);
        if(!first)std::cout<<",\n";
        first=false;
        std::cout<<"  ["<<x<<","<<y<<","<<z<<","<<unsigned(input[id])<<","<<unsigned(approximation[id])
            <<","<<unsigned(recovered[id])<<"]";
    }
    std::cout<<"\n ]\n}\n";
}
} // namespace dm3dgeom
int main(int argc,char** argv){
    try {
        unsigned side=8,brick=4,passes=6;bool lossless=true,selftest=false;
        std::string input_path,prefix;
        for(int i=1;i<argc;++i){
            const std::string a=argv[i];
            auto val=[&](){if(++i>=argc)throw std::runtime_error("missing argument");return std::string(argv[i]);};
            auto num=[&](){const std::string s=val();std::size_t used=0;const auto n=std::stoul(s,&used);
                if(used!=s.size()||n>1000000)throw std::runtime_error("invalid number");
                return static_cast<unsigned>(n);};
            if(a=="--self-test")selftest=true;
            else if(a=="--side")side=num(); else if(a=="--brick")brick=num();
            else if(a=="--passes")passes=num();else if(a=="--lossy")lossless=false;
            else if(a=="--input")input_path=val();else if(a=="--out-prefix")prefix=val();
            else throw std::runtime_error("unknown argument: "+a);
        }
        if(selftest)dm3dgeom::self_test();
        else dm3dgeom::run(side,brick,passes,lossless,input_path,prefix);
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}