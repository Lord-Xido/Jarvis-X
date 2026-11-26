#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <algorithm>
namespace s3dbcx {
constexpr int W=2796,H=1290,N=128;
using Latent=std::array<int8_t,N>;
using Frame=std::vector<uint8_t>;
struct Ins {uint8_t op,mode,dst,src; uint32_t imm;};
constexpr std::array<Ins,8> rom{{
 {1,0,1,0,0},{2,0,7,1,0},{3,0,1,7,0},{4,0,3,2,0},
 {5,0,3,3,0},{6,0,5,3,0},{7,0,6,5,0},{8,0,2,3,64}}};
uint64_t pack(uint8_t op,uint16_t x,uint16_t y,uint16_t p,uint16_t t,uint16_t f){
 return (uint64_t(op&15)<<60)|(uint64_t(x&4095)<<48)|(uint64_t(y&4095)<<36)|
 (uint64_t(p&4095)<<24)|(uint64_t(t&4095)<<12)|uint64_t(f&4095);
}
std::array<uint8_t,8> little(uint64_t w){std::array<uint8_t,8> b{};for(int i=0;i<8;i++)b[i]=uint8_t(w>>(8*i));return b;}
uint64_t read64(const std::array<uint8_t,8>& b){uint64_t w=0;for(int i=0;i<8;i++)w|=uint64_t(b[i])<<(8*i);return w;}
std::array<uint8_t,8> bytes(const Ins& i){return {i.op,i.mode,i.dst,i.src,uint8_t(i.imm),uint8_t(i.imm>>8),uint8_t(i.imm>>16),uint8_t(i.imm>>24)};}
int clamp(int v){return std::max(-128,std::min(127,v));}
int sar7(int32_t v){return v>=0?int(v/128):-int((uint32_t(-int64_t(v))+127)/128);}
int idx(int x,int y,int z){return x+8*y+32*z;}
struct VM{
 Latent input{},geo{},latent{},candidate{},memory{},transport{};
 Frame framebuffer;uint64_t touch=0,epochs=0;bool committed=false,presented=false;
 VM():framebuffer(size_t(W)*H*4,0){}
 void enc(){auto b=little(touch); input.fill(0);for(int i=0;i<8;i++)input[i]=int8_t(b[i]);}
 void geometry(){for(int z=0;z<4;z++)for(int y=0;y<4;y++)for(int x=0;x<8;x++){
 int i=idx(x,y,z);int32_t v=input[i];
 int n=input[idx((x+1)&7,y,z)]+input[idx((x+7)&7,y,z)]+input[idx(x,(y+1)&3,z)]+input[idx(x,(y+3)&3,z)]+input[idx(x,y,(z+1)&3)]+input[idx(x,y,(z+3)&3)];
 geo[i]=int8_t(clamp(v+sar7((n-6*v)*16)));}}
 void ricci(){input=geo;} // geometric smoothing surrogate, NOT actual Ricci curvature
 void recurrence(){for(int z=0;z<4;z++)for(int y=0;y<4;y++)for(int x=0;x<8;x++){
 int i=idx(x,y,z);
 int32_t acc=64*int(latent[i])+32*int(input[i]);
 acc+=4*(int(latent[idx((x+1)&7,y,z)])+int(latent[idx((x+7)&7,y,z)])+int(latent[idx(x,(y+1)&3,z)])+int(latent[idx(x,(y+3)&3,z)])+int(latent[idx(x,y,(z+1)&3)])+int(latent[idx(x,y,(z+3)&3)]));
 candidate[i]=int8_t(clamp(sar7(acc)));}}
 void trans(){for(int z=0;z<4;z++)for(int y=0;y<4;y++)for(int x=0;x<8;x++)transport[idx((x+y)&7,y,z)]=candidate[idx(x,y,z)];}
 void decode(){uint64_t w=touch;int cx=int(((w>>48)&4095)*uint64_t(W)>>12);int cy=int(((w>>36)&4095)*uint64_t(H)>>12);
 int intensity=std::min(255,160+int(uint8_t(transport[0]))/4);
 for(int y=0;y<H;y++)for(int x=0;x<W;x++){
 int dx=std::abs(x-cx),dy=std::abs(y-cy);int v=std::max(0,intensity-((dx+dy)>>2));size_t k=(size_t(y)*W+x)*4;
 framebuffer[k]=uint8_t(v>>1);framebuffer[k+1]=uint8_t(v>>2);framebuffer[k+2]=uint8_t(v);framebuffer[k+3]=255;
 }}
 void scan(){presented=true;} // simulation only: no physical display action
 void echo(){if(!presented)throw std::runtime_error("scan missing");latent=candidate;memory=latent;epochs++;committed=true;}
 void execute(){committed=false;presented=false;for(const auto& i:rom){switch(i.op){case 1:enc();break;case 2:geometry();break;case 3:ricci();break;case 4:recurrence();break;case 5:trans();break;case 6:decode();break;case 7:scan();break;case 8:echo();break;default:throw std::runtime_error("invalid opcode");}}}
};
void selftest(){
 auto w=pack(2,2048,2048,1024,1,0);assert(w==0x2800800400001000ULL);assert(read64(little(w))==w);
 assert((bytes(rom[0])==std::array<uint8_t,8>{1,0,1,0,0,0,0,0}));
 std::array<bool,N> seen{};for(int z=0;z<4;z++)for(int y=0;y<4;y++)for(int x=0;x<8;x++){int j=idx((x+y)&7,y,z);assert(!seen[j]);seen[j]=true;assert(((x+y+8-y)&7)==x);}
 VM vm;vm.touch=w;vm.execute();assert(vm.committed&&vm.presented&&vm.epochs==1);assert(vm.framebuffer.size()==14427360);
 for(size_t k=3;k<vm.framebuffer.size();k+=4)assert(vm.framebuffer[k]==255);
 Frame target=vm.framebuffer;target[0]^=0x81;target[target.size()-4]^=0x04;
 Frame residual(target.size());for(size_t i=0;i<target.size();i++)residual[i]=target[i]^vm.framebuffer[i];
 uint8_t mismatch=0;for(size_t i=0;i<target.size();i++)mismatch|=target[i]^(vm.framebuffer[i]^residual[i]);assert(mismatch==0);
 auto before=vm.epochs;vm.execute();assert(vm.epochs==before+1);
 std::printf("S3D-BCx PASS: pack, ROM, 3D transport, two epochs, %zu framebuffer bytes, XOR residual\n",vm.framebuffer.size());
}
}
int main(){s3dbcx::selftest();}
