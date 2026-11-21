// JX-PROM-100 v1 — independently compiled C++17 PNG Pixel ROM verifier and loader.
// Read-only non-Turing-complete decoder; no arbitrary code execution from PNG.
#include <png.h>
#include <zlib.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace prom {
using Byte=std::uint8_t;
using Bytes=std::vector<Byte>;
constexpr unsigned WIDTH=7856, HEIGHT=4496;
constexpr unsigned BANK_COUNT=100, TILE_W=768, TILE_H=432, GAP=16;
constexpr unsigned PATCH_W=128, PATCH_H=32;
constexpr std::size_t BANK_BYTES=PATCH_W*PATCH_H*3;
constexpr std::size_t HEADER=128, PAYLOAD=10240, TOTAL=100*PAYLOAD;
static_assert(BANK_BYTES==12288 && TOTAL==1024000 && BANK_BYTES-HEADER-PAYLOAD==1920,"format constants");

std::array<Byte,32> sha256(const Byte* data,std::size_t len){
    std::array<Byte,32> digest{};
    unsigned out=0;
    if(EVP_Digest(data,len,digest.data(),&out,EVP_sha256(),nullptr)!=1 || out!=digest.size())
        throw std::runtime_error("OpenSSL SHA256 failed");
    return digest;
}
std::uint32_t get32(const Byte* data){
    return std::uint32_t(data[0]) | std::uint32_t(data[1])<<8 |
           std::uint32_t(data[2])<<16 | std::uint32_t(data[3])<<24;
}
std::uint32_t crc(const Byte* data,std::size_t len){
    // zlib is platform-stable, including at all documented payload lengths.
    return static_cast<std::uint32_t>(crc32_z(0L,data,len));
}
std::string hex(const std::array<Byte,32>& digest){
    std::string out;
    const char* digits="0123456789abcdef";
    for(Byte v:digest){out+=digits[v>>4];out+=digits[v&15];}
    return out;
}
void write_file(const std::string& path,const Bytes& bytes){
    std::ofstream out(path,std::ios::binary);
    if(!out || !out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("cannot write verified frame");
}
Bytes read_png_rgb(const std::string& path){
    png_image image{};
    image.version=PNG_IMAGE_VERSION;
    if(!png_image_begin_read_from_file(&image,path.c_str()))
        throw std::runtime_error(std::string("PNG open failed: ")+image.message);
    if(image.width!=WIDTH || image.height!=HEIGHT){
        png_image_free(&image);
        throw std::runtime_error("unexpected canvas dimensions");
    }
    image.format=PNG_FORMAT_RGB;
    Bytes pixels(PNG_IMAGE_SIZE(image));
    if(!png_image_finish_read(&image,nullptr,pixels.data(),0,nullptr)){
        std::string err=image.message;
        png_image_free(&image);
        throw std::runtime_error("PNG decode failed: "+err);
    }
    png_image_free(&image);
    if(pixels.size()!=static_cast<std::size_t>(WIDTH)*HEIGHT*3)
        throw std::runtime_error("unexpected RGB array size");
    return pixels;
}
std::array<Byte,BANK_BYTES> extract_bank(const Bytes& pixels,unsigned i){
    const unsigned row=i/10,col=i%10;
    const unsigned x0=GAP+col*(TILE_W+GAP)+632;
    const unsigned y0=GAP+row*(TILE_H+GAP)+392;
    std::array<Byte,BANK_BYTES> bank{};
    for(unsigned y=0;y<PATCH_H;y++){
        const std::size_t src=(static_cast<std::size_t>(y0+y)*WIDTH+x0)*3;
        const std::size_t dst=static_cast<std::size_t>(y)*PATCH_W*3;
        std::copy_n(pixels.data()+src,static_cast<std::size_t>(PATCH_W)*3,bank.data()+dst);
    }
    return bank;
}
Bytes decode(const Bytes& pixels){
    Bytes frame(TOTAL,0);
    std::array<Byte,32> expected_full{};
    for(unsigned i=0;i<BANK_COUNT;i++){
        auto bank=extract_bank(pixels,i);
        const Byte* b=bank.data();
        if(std::memcmp(b,"JXPRB001",8)!=0)throw std::runtime_error("bank "+std::to_string(i)+": magic invalid");
        if(get32(b+8)!=i || get32(b+12)!=BANK_COUNT || get32(b+16)!=i*PAYLOAD ||
           get32(b+20)!=PAYLOAD || get32(b+24)!=TOTAL || get32(b+28)!=1 || get32(b+36)!=0)
            throw std::runtime_error("bank "+std::to_string(i)+": address / version mismatch");
        std::array<Byte,HEADER> h{};
        std::copy_n(b,HEADER,h.data());
        std::fill(h.begin()+104,h.begin()+108,0);
        if(crc(h.data(),h.size())!=get32(b+104))
            throw std::runtime_error("bank "+std::to_string(i)+": header CRC failure");
        if(i==0)std::copy_n(b+40,32,expected_full.data());
        else if(!std::equal(expected_full.begin(),expected_full.end(),b+40))
            throw std::runtime_error("bank "+std::to_string(i)+": frame digest disagree");
        if(crc(b+HEADER,PAYLOAD)!=get32(b+32))
            throw std::runtime_error("bank "+std::to_string(i)+": payload CRC failure");
        auto digest=sha256(b+HEADER,PAYLOAD);
        if(!std::equal(digest.begin(),digest.end(),b+72))
            throw std::runtime_error("bank "+std::to_string(i)+": payload SHA256 failure");
        if(std::any_of(b+HEADER+PAYLOAD,b+BANK_BYTES,[](Byte v){return v!=0;}))
            throw std::runtime_error("bank "+std::to_string(i)+": nonzero reserved padding");
        std::copy_n(b+HEADER,PAYLOAD,frame.begin()+static_cast<std::size_t>(i)*PAYLOAD);
    }
    if(sha256(frame.data(),frame.size())!=expected_full)
        throw std::runtime_error("reassembled 100-bank frame SHA256 mismatch");
    if(std::memcmp(frame.data(),"JXSF1T12",8)!=0)
        throw std::runtime_error("underlying bytecode frame magic invalid");
    if(crc(frame.data(),frame.size()-4)!=get32(frame.data()+frame.size()-4))
        throw std::runtime_error("underlying bytecode frame CRC failure");
    if(std::memcmp(frame.data()+4096,"JXEX",4)!=0 ||
       std::memcmp(frame.data()+5120,"JXOP",4)!=0)
        throw std::runtime_error("underlying VM execution slot signatures invalid");
    return frame;
}
void inspect(const Bytes& frame){
    const std::uint32_t count=frame[4100];
    if(count!=8)throw std::runtime_error("unsupported VM opcode count");
    std::cout<<"JX-PROM-100 PASS banks=100 frame_bytes="<<frame.size()
             <<" sha256="<<hex(sha256(frame.data(),frame.size()))<<"\n";
    std::cout<<"Bytecode JXEX offset=4096, JXOP offset=5120 (reserved)\n";
    constexpr std::array<Byte,8> expected{{0x11,0x20,0x20,0x20,0x21,0x30,0x40,0xff}};
    for(unsigned i=0;i<count;i++){
        std::size_t at=4112+i*4;
        if(frame[at]!=expected[i])throw std::runtime_error("unknown/disallowed bytecode opcode");
        std::cout<<"ROM["<<i<<"] opcode=0x"<<std::hex<<std::setw(2)<<std::setfill('0')
                 <<unsigned(frame[at])<<std::dec<<" src="<<unsigned(frame[at+1])
                 <<" dst="<<unsigned(frame[at+2])<<" aux="<<unsigned(frame[at+3])<<"\n";
    }
}
}
int main(int argc,char**argv){
    try{
        if(argc<3 || (std::string(argv[1])!="--decode" && std::string(argv[1])!="--inspect")){
            std::cerr<<"Usage: jx_pixelrom --inspect ROM.png | --decode ROM.png recovered.jxsf\n";
            return 2;
        }
        const std::string mode=argv[1];
        if((mode=="--decode"&&argc!=4)||(mode=="--inspect"&&argc!=3))
            throw std::runtime_error("wrong argument count");
        auto pixels=prom::read_png_rgb(argv[2]);
        auto frame=prom::decode(pixels);
        prom::inspect(frame);
        if(mode=="--decode"){
            prom::write_file(argv[3],frame);
            std::cout<<"RECOVERED FRAME: "<<argv[3]<<"\n";
        }
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"PIXEL ROM REJECTED: "<<e.what()<<"\n";
        return 1;
    }
}
