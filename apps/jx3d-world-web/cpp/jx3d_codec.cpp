// JX3D v1 native C++20 codec; interoperates with Python codec.py.
// Dependencies: zlib + OpenSSL libcrypto. C++ executable is offline CLI.
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <openssl/evp.h>
#include <zlib.h>

namespace jx {
using Bytes=std::vector<std::uint8_t>;
constexpr std::size_t kHeader=52;
constexpr std::size_t kMaxPacket=1500000;

void check_dimension(unsigned d) {
    if(d < 16 || d > 64 || (d & 1))
        throw std::runtime_error("dimension must be even, 16..64");
}

Bytes read_file(const std::string& name) {
    std::ifstream f(name,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open: "+name);
    return Bytes(std::istreambuf_iterator<char>(f),{});
}
void write_file(const std::string& name,const Bytes& b) {
    std::ofstream f(name,std::ios::binary);
    if(!f || !f.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size())))
        throw std::runtime_error("cannot write: "+name);
}
std::array<std::uint8_t,32> sha256(const Bytes& in) {
    std::array<std::uint8_t,32> out{};
    unsigned int len=0;
    if(EVP_Digest(in.data(),in.size(),out.data(),&len,EVP_sha256(),nullptr)!=1 || len!=32)
        throw std::runtime_error("SHA-256 failure");
    return out;
}
void put16(Bytes& b,unsigned v){b.push_back((v>>8)&255);b.push_back(v&255);}
void put32(Bytes& b,std::uint32_t v){for(int s=24;s>=0;s-=8)b.push_back((v>>s)&255);}
unsigned get16(const Bytes& b,std::size_t i){return (unsigned(b.at(i))<<8)|b.at(i+1);}
std::uint32_t get32(const Bytes& b,std::size_t i){return (std::uint32_t(b.at(i))<<24)|(std::uint32_t(b.at(i+1))<<16)|(std::uint32_t(b.at(i+2))<<8)|b.at(i+3);}

Bytes coarse(const Bytes& data,unsigned d){
    unsigned h=d/2;
    Bytes c(h*h*h);
    for(unsigned z=0;z<h;++z)for(unsigned y=0;y<h;++y)for(unsigned x=0;x<h;++x){
        unsigned sum=0;
        for(unsigned dz=0;dz<2;++dz)for(unsigned dy=0;dy<2;++dy){
            auto r=((2*z+dz)*d+(2*y+dy))*d+2*x;
            sum+=data[r]+data[r+1];
        }
        c[(z*h+y)*h+x]=static_cast<std::uint8_t>((sum+4)/8);
    }
    return c;
}
Bytes predict(const Bytes& c,unsigned d){
    unsigned h=d/2;
    Bytes data(d*d*d);
    for(unsigned z=0;z<d;++z)for(unsigned y=0;y<d;++y)for(unsigned x=0;x<d;++x)
        data[(z*d+y)*d+x]=c[((z/2*h)+(y/2))*h+x/2];
    return data;
}

Bytes encode(const Bytes& input,unsigned d,bool preview){
    check_dimension(d);
    if(input.size()!=std::size_t(d)*d*d)throw std::runtime_error("raw input length is not dim^3");
    Bytes c=coarse(input,d),base=predict(c,d);
    Bytes payload=c;
    Bytes decoded=preview?base:input;
    if(!preview){
        payload.reserve(c.size()+input.size());
        for(std::size_t i=0;i<input.size();++i)payload.push_back(input[i]^base[i]);
    }
    uLongf zlen=compressBound(static_cast<uLong>(payload.size()));
    Bytes zipped(zlen);
    auto result=compress2(zipped.data(),&zlen,payload.data(),static_cast<uLong>(payload.size()),6);
    if(result!=Z_OK)throw std::runtime_error("zlib compression failure");
    zipped.resize(zlen);
    Bytes output;
    output.reserve(kHeader+zipped.size());
    output.insert(output.end(),{'J','X','3','D',1,static_cast<std::uint8_t>(preview?1:0)});
    for(int i=0;i<3;++i)put16(output,d);
    put32(output,static_cast<std::uint32_t>(c.size()));
    put32(output,static_cast<std::uint32_t>(zipped.size()));
    const auto hash=sha256(decoded);
    output.insert(output.end(),hash.begin(),hash.end());
    if(output.size()!=kHeader)throw std::runtime_error("invalid internal header length");
    output.insert(output.end(),zipped.begin(),zipped.end());
    return output;
}
struct Result{Bytes volume;unsigned dim;bool preview;};
Result decode(const Bytes& packet){
    if(packet.size()<kHeader || packet.size()>kMaxPacket)throw std::runtime_error("invalid packet size");
    if(packet[0]!='J'||packet[1]!='X'||packet[2]!='3'||packet[3]!='D'||packet[4]!=1||packet[5]>1)
        throw std::runtime_error("unsupported header");
    const bool preview=packet[5]==1;
    unsigned d=get16(packet,6);
    check_dimension(d);
    if(get16(packet,8)!=d || get16(packet,10)!=d)throw std::runtime_error("unequal dimensions");
    std::size_t csz=std::size_t(d/2)*(d/2)*(d/2);
    std::size_t plen=csz+(preview?0:std::size_t(d)*d*d);
    if(get32(packet,12)!=csz || get32(packet,16)!=packet.size()-kHeader)
        throw std::runtime_error("invalid payload lengths");
    Bytes payload(plen);
    z_stream z{};
    z.next_in=const_cast<Bytef*>(packet.data()+kHeader);
    z.avail_in=static_cast<uInt>(packet.size()-kHeader);
    z.next_out=payload.data();z.avail_out=static_cast<uInt>(plen);
    if(inflateInit(&z)!=Z_OK)throw std::runtime_error("inflateInit failure");
    int status=inflate(&z,Z_FINISH);
    bool okay=status==Z_STREAM_END && z.total_out==plen && z.avail_in==0;
    inflateEnd(&z);
    if(!okay)throw std::runtime_error("corrupt, trailing, or oversized zlib stream");
    Bytes c(payload.begin(),payload.begin()+csz);
    Bytes v=predict(c,d);
    if(!preview)for(std::size_t i=0;i<v.size();++i)v[i]^=payload[csz+i];
    const auto hash=sha256(v);
    for(unsigned i=0;i<32;++i)if(packet[20+i]!=hash[i])throw std::runtime_error("SHA-256 mismatch");
    return {std::move(v),d,preview};
}
}

int main(int argc,char** argv){
    try{
        if(argc<3)throw std::runtime_error("usage: jx3d encode input.raw DIM lossless|preview output.jx3d | decode input.jx3d output.raw | verify input.jx3d");
        std::string op=argv[1];
        if(op=="encode"){
            if(argc!=6)throw std::runtime_error("encode requires raw path, dimension, mode, output path");
            unsigned dim=static_cast<unsigned>(std::stoul(argv[3]));
            std::string mode=argv[4];
            if(mode!="lossless"&&mode!="preview")throw std::runtime_error("invalid mode");
            auto out=jx::encode(jx::read_file(argv[2]),dim,mode=="preview");
            jx::write_file(argv[5],out);
            std::cout<<"encoded "<<out.size()<<" bytes\n";
        }else if(op=="decode"||op=="verify"){
            if(argc!=(op=="decode"?4:3))throw std::runtime_error("incorrect decode/verify arguments");
            auto result=jx::decode(jx::read_file(argv[2]));
            if(op=="decode")jx::write_file(argv[3],result.volume);
            std::cout<<"verified "<<result.dim<<"^3 "<<(result.preview?"preview":"lossless")
                     <<" ("<<result.volume.size()<<" reconstructed bytes)\n";
        }else throw std::runtime_error("unknown operation");
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
