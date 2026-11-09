// Dr Moagi Ω³ — 1000 MB 3D bitwise execution-trace engine
// C++17; streaming, lossless, replay-verifiable, bounded-memory.
// Build: g++ -std=c++17 -O3 -Wall -Wextra -pedantic src/moagi_bittrace.cpp -lcrypto -o moagi_bittrace
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <openssl/sha.h>

namespace fs = std::filesystem;
using u8 = std::uint8_t; using u32 = std::uint32_t; using u64 = std::uint64_t;
using Digest = std::array<u8, SHA256_DIGEST_LENGTH>;
constexpr char MAGIC[8] = {'M','3','B','T','R','C','1','\0'};
constexpr u32 VERSION = 1;
using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string& msg) {throw std::runtime_error(msg);}
void check(bool cond, const std::string& msg) {if (!cond) fail(msg);}
std::string hex(const Digest& d) {static constexpr char t[] = "0123456789abcdef"; std::string s; s.reserve(64); for (u8 b:d){s.push_back(t[b>>4]);s.push_back(t[b&15]);}return s;}
Digest sha(const u8* bytes, size_t n) {Digest d{}; SHA256(bytes,n,d.data());return d;}
Digest sha(const std::vector<u8>& b) {return sha(b.data(),b.size());}
void append32(std::vector<u8>& dst, u32 n) {for(int i=0;i<4;++i)dst.push_back(u8(n>>(8*i)));}
void append64(std::vector<u8>& dst, u64 n) {for(int i=0;i<8;++i)dst.push_back(u8(n>>(8*i)));}
void write32(std::ostream& f, u32 n) {u8 b[4];for(int i=0;i<4;++i)b[i]=u8(n>>(8*i)); f.write((char*)b,4);check(bool(f),"write32 failed");}
u32 read32(std::istream& f) {u8 b[4];f.read((char*)b,4);check(bool(f),"Truncated archive integer");return u32(b[0])|(u32(b[1])<<8)|(u32(b[2])<<16)|(u32(b[3])<<24);}
void put(std::ostream& f, const u8* p, size_t n) {f.write((const char*)p,static_cast<std::streamsize>(n));check(bool(f),"Binary write failed");}
void take(std::istream& f,u8* p,size_t n) {f.read((char*)p,static_cast<std::streamsize>(n));check(bool(f),"Binary read failed");}

u32 crc32(const std::vector<u8>& b) { // Table-driven IEEE CRC32, same bytes as bitwise reference.
 static const auto table = []{std::array<u32,256> t{};for(u32 j=0;j<256;++j){u32 x=j;for(int k=0;k<8;++k)x=(x>>1)^((x&1)?0xedb88320u:0u);t[j]=x;}return t;}();
 u32 crc=0xffffffffu;for(u8 x:b)crc=(crc>>8)^table[(crc^x)&255u];return ~crc;
}
u32 pop8(u8 x) {return u32(__builtin_popcount(unsigned(x)));}
struct Shape {u32 x=1000,y=1000,z=1000; u64 plane()const{return u64(x)*y;} u64 voxels()const{return plane()*z;} void valid()const{check(x&&y&&z,"Shape dimensions must be positive");check(plane()<=20'000'000,"Plane too large for configured bounded-memory runner"); check(voxels()<=u64(std::numeric_limits<std::streamoff>::max()),"Volume offset overflow");}};

std::vector<u8> rle(const std::vector<u8>& in){std::vector<u8> out;out.reserve(std::min(in.size()*2,in.size()+size_t(2048)));for(size_t i=0;i<in.size();){size_t n=1;while(i+n<in.size()&&in[i+n]==in[i]&&n<255)++n;out.push_back(u8(n));out.push_back(in[i]);i+=n;}return out;}
std::vector<u8> unrle(const std::vector<u8>& in,size_t expected){check((in.size()&1)==0,"RLE payload odd-sized");std::vector<u8> out;out.reserve(expected);for(size_t i=0;i<in.size();i+=2){u8 n=in[i];check(n>0,"RLE zero run");check(size_t(n)<=expected-out.size(),"RLE overflow");out.insert(out.end(),size_t(n),in[i+1]);}check(out.size()==expected,"RLE decoded length mismatch");return out;}
std::vector<u8> predict_x(const std::vector<u8>& in,Shape sh){std::vector<u8> pred(in.size());for(u32 y=0;y<sh.y;++y){size_t r=size_t(y)*sh.x;u8 prev=0;for(u32 x=0;x<sh.x;++x){auto cur=in[r+x];pred[r+x]=u8(cur^prev);prev=cur;}}return pred;}
std::vector<u8> invert_x(const std::vector<u8>& in,Shape sh){std::vector<u8> out(in.size());for(u32 y=0;y<sh.y;++y){size_t r=size_t(y)*sh.x;u8 prev=0;for(u32 x=0;x<sh.x;++x){u8 cur=u8(in[r+x]^prev);out[r+x]=cur;prev=cur;}}return out;}
enum : u8 {RAW=0,RLE_RAW=1,RLE_XOR_X=2};
const char* codec_name(u8 c) {switch(c){case RAW:return "raw";case RLE_RAW:return "rle";case RLE_XOR_X:return "xor_x+rle";default:return "invalid";}}
struct Encoded {u8 mode=0;std::vector<u8> bytes;};
Encoded encode(const std::vector<u8>& in,Shape sh){Encoded best{RAW,in};auto r=rle(in);if(r.size()<best.bytes.size())best={RLE_RAW,std::move(r)};auto p=predict_x(in,sh);auto q=rle(p);if(q.size()<best.bytes.size())best={RLE_XOR_X,std::move(q)};return best;}
std::vector<u8> decode(const Encoded& c,Shape sh){size_t n=size_t(sh.plane());switch(c.mode){case RAW:check(c.bytes.size()==n,"Raw payload length mismatch");return c.bytes;case RLE_RAW:return unrle(c.bytes,n);case RLE_XOR_X:return invert_x(unrle(c.bytes,n),sh);default:fail("Unknown codec opcode");}}

struct Stat {u64 changed_bytes=0,changed_bits=0,in_bits=0,out_bits=0;std::array<u64,8> flips{};int first_x=-1,first_y=-1;u8 first_before=0,first_after=0;};
Stat compare(const std::vector<u8>& a,const std::vector<u8>& b,Shape sh){
 check(a.size()==b.size(),"Compare size mismatch");Stat s;
 constexpr u64 ONES=0x0101010101010101ULL;
 size_t i=0, n=a.size();
 for(;i+8<=n;i+=8){
   u64 aa,bb;std::memcpy(&aa,a.data()+i,8);std::memcpy(&bb,b.data()+i,8);u64 v=aa^bb;
   s.in_bits+=u64(__builtin_popcountll(aa));s.out_bits+=u64(__builtin_popcountll(bb));
   if(v){
     s.changed_bits+=u64(__builtin_popcountll(v));
     u64 flags=v; flags|=flags>>1;flags|=flags>>2;flags|=flags>>4;
     s.changed_bytes+=u64(__builtin_popcountll(flags&ONES));
     for(int bit=0;bit<8;++bit)s.flips[bit]+=u64(__builtin_popcountll((v>>bit)&ONES));
     if(s.first_x<0){for(size_t j=0;j<8;++j)if(a[i+j]!=b[i+j]){size_t p=i+j;s.first_x=int(p%sh.x);s.first_y=int(p/sh.x);s.first_before=a[p];s.first_after=b[p];break;}}
   }
 }
 for(;i<n;++i){u8 v=u8(a[i]^b[i]);s.in_bits+=pop8(a[i]);s.out_bits+=pop8(b[i]);if(v){++s.changed_bytes;s.changed_bits+=pop8(v);if(s.first_x<0){s.first_x=int(i%sh.x);s.first_y=int(i/sh.x);s.first_before=a[i];s.first_after=b[i];}for(int bit=0;bit<8;++bit)s.flips[bit]+=(v>>bit)&1u;}}
 return s;
}

void read_plane(std::ifstream& f,u32 z,Shape sh,std::vector<u8>& into){into.resize(size_t(sh.plane()));auto pos=std::streamoff(sh.plane()*u64(z));f.clear();f.seekg(pos,std::ios::beg);check(bool(f),"Plane seek failed");take(f,into.data(),into.size());}
// 0: reference, 1: optimized. Both evaluate the same integer formula.
void spatial_ref(const std::vector<u8>& prev,const std::vector<u8>& cur,const std::vector<u8>& next,Shape sh,u32 gain,std::vector<u8>& out){
 for(u32 y=0;y<sh.y;++y)for(u32 x=0;x<sh.x;++x){size_t p=size_t(y)*sh.x+x;
 auto left=cur[size_t(y)*sh.x+(x+sh.x-1)%sh.x];auto right=cur[size_t(y)*sh.x+(x+1)%sh.x];
 auto up=cur[size_t((y+sh.y-1)%sh.y)*sh.x+x];auto down=cur[size_t((y+1)%sh.y)*sh.x+x];
 unsigned v=2u*cur[p]+left+right+up+down+prev[p]+next[p];unsigned s=(v+4)/8;
 out[p]=u8(((256u-gain)*s+gain*unsigned(cur[p])+128u)>>8);
 }
}
void spatial_fast(const std::vector<u8>& prev,const std::vector<u8>& cur,const std::vector<u8>& next,Shape sh,u32 gain,std::vector<u8>& out){
 const size_t xlen=sh.x;
 for(u32 y=0;y<sh.y;++y){size_t row=size_t(y)*xlen;size_t above=size_t(y?y-1:sh.y-1)*xlen;size_t below=size_t(y+1==sh.y?0:y+1)*xlen;
 for(size_t x=0;x<xlen;++x){size_t p=row+x; unsigned v=2u*cur[p]+cur[row+(x?x-1:xlen-1)]+cur[row+(x+1==xlen?0:x+1)]+cur[above+x]+cur[below+x]+prev[p]+next[p];unsigned s=(v+4u)>>3;
 out[p]=u8(((256u-gain)*s+gain*unsigned(cur[p])+128u)>>8);
 }}
}
void calculate(const std::vector<u8>& prev,const std::vector<u8>& cur,const std::vector<u8>& next,Shape sh,u32 gain,std::vector<u8>& out,bool fast){out.resize(cur.size());if(fast)spatial_fast(prev,cur,next,sh,gain,out);else spatial_ref(prev,cur,next,sh,gain,out);}

struct Chain {Digest value{};explicit Chain(Shape sh,u32 gain) {std::vector<u8> pre{'M','3','B','T','-','C','H','A','I','N','-','2'};append32(pre,sh.x);append32(pre,sh.y);append32(pre,sh.z);append32(pre,gain);value=sha(pre);} 
 void update(u32 t,u32 z,bool fast,const Digest& input,const Digest& output,const Stat& st,u8 codec,u32 bytes){std::vector<u8> m; m.reserve(32+8+64+8*12);m.insert(m.end(),value.begin(),value.end());append32(m,t);append32(m,z);append32(m,fast?1u:0u);m.insert(m.end(),input.begin(),input.end());m.insert(m.end(),output.begin(),output.end());append64(m,st.changed_bytes);append64(m,st.changed_bits);append64(m,st.in_bits);append64(m,st.out_bits);for(auto x:st.flips)append64(m,x);append32(m,codec);append32(m,bytes);value=sha(m);}
};

void header(std::ostream& f,Shape sh,u32 gain){f.write(MAGIC,8);write32(f,VERSION);write32(f,sh.x);write32(f,sh.y);write32(f,sh.z);write32(f,gain);}
std::pair<Shape,u32> parse_header(std::istream& f){char magic[8];f.read(magic,8);check(bool(f)&&std::memcmp(magic,MAGIC,8)==0,"Invalid archive magic");check(read32(f)==VERSION,"Archive version mismatch");Shape sh;sh.x=read32(f);sh.y=read32(f);sh.z=read32(f);u32 gain=read32(f);sh.valid();check(gain<=256,"Invalid archive gain");return {sh,gain};}
void write_archive_plane(std::ostream& f,u32 z,const Encoded& en,const std::vector<u8>& raw){write32(f,z);write32(f,en.mode);write32(f,u32(raw.size()));write32(f,u32(en.bytes.size()));write32(f,crc32(raw));Digest h=sha(raw);put(f,h.data(),h.size());put(f,en.bytes.data(),en.bytes.size());}
struct ArchivePlane{u32 z;Encoded codec;std::vector<u8> raw;};
ArchivePlane read_archive_plane(std::istream& f,Shape sh,u32 expected_z){ArchivePlane p;p.z=read32(f);check(p.z==expected_z,"Archive plane order mismatch");u32 mode=read32(f),raw_size=read32(f),payload_size=read32(f),crc=read32(f);check(mode<=RLE_XOR_X,"Invalid archive codec");check(raw_size==sh.plane(),"Wrong archive plane dimensions");check(payload_size<=sh.plane()*2,"Oversized archive plane payload");Digest h{};take(f,h.data(),h.size());p.codec.mode=u8(mode);p.codec.bytes.resize(payload_size);take(f,p.codec.bytes.data(),payload_size);p.raw=decode(p.codec,sh);check(crc32(p.raw)==crc,"Archive CRC mismatch");check(sha(p.raw)==h,"Archive SHA-256 mismatch");return p;}
void decode_archive(const std::string& source,const std::string& target){std::ifstream f(source,std::ios::binary);check(bool(f),"Archive not found");auto [sh,gain]=parse_header(f);std::ofstream out(target+".partial",std::ios::binary|std::ios::trunc);check(bool(out),"Cannot open decode output");try{for(u32 z=0;z<sh.z;++z){auto p=read_archive_plane(f,sh,z);put(out,p.raw.data(),p.raw.size());}check(f.peek()==std::char_traits<char>::eof(),"Archive trailing bytes");out.close();check(bool(out),"Decode close failed");fs::rename(target+".partial",target);}catch(...){out.close();fs::remove(target+".partial");throw;}std::cout<<"DECODE_OK shape="<<sh.x<<"x"<<sh.y<<"x"<<sh.z<<" bytes="<<sh.voxels()<<" gain="<<gain<<"\n";}

struct Config{Shape sh;u32 gain=64;u32 steps=1;bool fast=true;std::string input,output,trace,archive;};
struct GlobalStats{u64 flips=0,changes=0,one_in=0,one_out=0;std::array<u64,8> planes{};u64 archive_bytes=0;u64 elapsed_ms=0;};
void check_file_size(const std::string& file,Shape sh){check(fs::is_regular_file(file),"Input must be a regular file: "+file);check(fs::file_size(file)==sh.voxels(),"File size does not match declared volume");}
void ensure_paths(const Config& c){check(!c.input.empty()&&!c.output.empty()&&!c.trace.empty(),"Need input, output, trace");check(c.steps>=1&&c.steps<=64,"Steps must be 1..64");check(c.gain<=256,"Gain q8 must be 0..256");c.sh.valid();auto abs=[](const std::string& s){return fs::absolute(s).lexically_normal().string();};check(abs(c.input)!=abs(c.output),"Output cannot overwrite input");check(abs(c.trace)!=abs(c.output)&&abs(c.trace)!=abs(c.input),"Trace path clashes with data path");if(!c.archive.empty())check(abs(c.archive)!=abs(c.input)&&abs(c.archive)!=abs(c.output)&&abs(c.archive)!=abs(c.trace),"Archive path collision");}
void write_csv_header(std::ofstream& f){f<<"iteration,z,kernel_id,plane_voxels,changed_voxels,changed_bits,input_1bits,output_1bits,bit0,bit1,bit2,bit3,bit4,bit5,bit6,bit7,codec,encoded_bytes,crc32_output,sha_input,sha_output,sha_chain,first_x,first_y,first_before,first_after,elapsed_us,decision\n";}
void csv_row(std::ofstream& f,u32 t,u32 z,Shape sh,bool fast,const Stat& st,const Encoded& en,u32 crc,const Digest& before,const Digest& after,const Digest& chain,u64 us){f<<t<<","<<z<<","<<(fast?"M6_FAST":"M6_REF")<<","<<sh.plane()<<","<<st.changed_bytes<<","<<st.changed_bits<<","<<st.in_bits<<","<<st.out_bits;for(auto n:st.flips)f<<","<<n;f<<","<<codec_name(en.mode)<<","<<en.bytes.size()<<","<<crc<<","<<hex(before)<<","<<hex(after)<<","<<hex(chain)<<","<<st.first_x<<","<<st.first_y<<","<<unsigned(st.first_before)<<","<<unsigned(st.first_after)<<","<<us<<",COMMIT\n";check(bool(f),"Trace write failed");}
GlobalStats execute_step(const std::string& inpath,const std::string& outpath,std::ofstream& trace,std::ofstream* archive,Shape sh,u32 gain,u32 t,bool fast,Chain& chain){
 std::ifstream input(inpath,std::ios::binary);check(bool(input),"Input open failed");std::ofstream output(outpath,std::ios::binary|std::ios::trunc);check(bool(output),"Output open failed");
 if(archive)header(*archive,sh,gain);
 std::vector<u8> prev,cur,next,out;
 read_plane(input,sh.z-1,sh,prev);read_plane(input,0,sh,cur);read_plane(input,sh.z>1?1:0,sh,next);
 GlobalStats g;auto tstart=Clock::now();
 for(u32 z=0;z<sh.z;++z){auto begin=Clock::now();calculate(prev,cur,next,sh,gain,out,fast);auto st=compare(cur,out,sh);auto en=encode(out,sh);auto recovered=decode(en,sh);check(recovered==out,"Codec failed round trip at plane "+std::to_string(z));
 Digest a=sha(cur),b=sha(out);chain.update(t,z,fast,a,b,st,en.mode,u32(en.bytes.size()));
 put(output,out.data(),out.size());if(archive){write_archive_plane(*archive,z,en,out);g.archive_bytes+=en.bytes.size()+52;}
 auto elapsed=u64(std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-begin).count());csv_row(trace,t,z,sh,fast,st,en,crc32(out),a,b,chain.value,elapsed);
 g.flips+=st.changed_bits;g.changes+=st.changed_bytes;g.one_in+=st.in_bits;g.one_out+=st.out_bits;for(int bit=0;bit<8;++bit)g.planes[bit]+=st.flips[bit];
 prev.swap(cur);cur.swap(next);if(z+1<sh.z)read_plane(input,(z+2)%sh.z,sh,next);
 }
 output.flush();check(bool(output),"Output flush failed");output.close();check(bool(output),"Output close failed");g.elapsed_ms=u64(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-tstart).count());return g;
}
void run(const Config& c){ensure_paths(c);check_file_size(c.input,c.sh);std::ofstream trace(c.trace+".partial",std::ios::trunc);check(bool(trace),"Cannot open trace");write_csv_header(trace);
 std::string part=c.output+".partial",a=c.output+".stageA",b=c.output+".stageB",archive_part=c.archive.empty()?"":c.archive+".partial";
 std::ofstream archive;Chain chain(c.sh,c.gain);GlobalStats sum;auto all_start=Clock::now();
 try{
  if(!c.archive.empty()){archive.open(archive_part,std::ios::binary|std::ios::trunc);check(bool(archive),"Cannot open archive");}
  for(u32 t=0;t<c.steps;++t){std::string src=t?((t&1)?a:b):c.input;std::string dst=(t+1==c.steps)?part:((t&1)?b:a);
   auto g=execute_step(src,dst,trace,(t+1==c.steps&&!c.archive.empty())?&archive:nullptr,c.sh,c.gain,t,c.fast,chain);
   sum.flips+=g.flips;sum.changes+=g.changes;sum.one_in+=g.one_in;sum.one_out+=g.one_out;sum.archive_bytes+=g.archive_bytes;for(int q=0;q<8;++q)sum.planes[q]+=g.planes[q];
   std::cout<<"ITERATION="<<t<<" planes="<<c.sh.z<<" bits_traced="<<(c.sh.voxels()*8)<<" flipped_bits="<<g.flips<<" changed_voxels="<<g.changes<<" elapsed_ms="<<g.elapsed_ms<<" chain_sha256="<<hex(chain.value)<<"\n";
  }
  if(archive.is_open()){archive.close();check(bool(archive),"Archive close failed");}
  trace.close();check(bool(trace),"Trace close failed");
  fs::rename(part,c.output);if(!c.archive.empty())fs::rename(archive_part,c.archive);fs::rename(c.trace+".partial",c.trace);
  fs::remove(a);fs::remove(b);
  sum.elapsed_ms=u64(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-all_start).count());
  std::cout<<"COMMITTED volume_bytes="<<c.sh.voxels()<<" bits_per_step="<<(c.sh.voxels()*8)<<" iterations="<<c.steps<<" total_bit_comparisons="<<(c.sh.voxels()*8*c.steps)<<" total_flipped_bits="<<sum.flips<<" total_changed_voxels="<<sum.changes<<" total_ms="<<sum.elapsed_ms<<" archive_bytes="<<sum.archive_bytes<<" final_chain_sha256="<<hex(chain.value)<<"\n";
 }catch(...){archive.close();trace.close();fs::remove(part);fs::remove(archive_part);fs::remove(c.trace+".partial");fs::remove(a);fs::remove(b);throw;}
}
void generate(const std::string& output,Shape sh,const std::string& pattern){sh.valid();check(pattern=="checker"||pattern=="constant"||pattern=="impulse"||pattern=="ramp","Unknown synthetic pattern");std::ofstream o(output+".partial",std::ios::binary|std::ios::trunc);check(bool(o),"Cannot create source");std::vector<u8> plane(size_t(sh.plane()),0);
 try{for(u32 z=0;z<sh.z;++z){if(pattern=="checker"){for(u32 y=0;y<sh.y;++y)for(u32 x=0;x<sh.x;++x)plane[size_t(y)*sh.x+x]=((x+y+z)&1)?255:0;}else if(pattern=="constant")std::fill(plane.begin(),plane.end(),128);else if(pattern=="impulse"){std::fill(plane.begin(),plane.end(),0);if(z==sh.z/2)plane[size_t(sh.y/2)*sh.x+sh.x/2]=255;}else if(pattern=="ramp"){for(u32 y=0;y<sh.y;++y)for(u32 x=0;x<sh.x;++x)plane[size_t(y)*sh.x+x]=u8((u64(x)*17+y*31+z*13)&255u);}put(o,plane.data(),plane.size());}o.close();check(bool(o),"Generate close failed");fs::rename(output+".partial",output);}catch(...){o.close();fs::remove(output+".partial");throw;}
 std::cout<<"GENERATED bytes="<<sh.voxels()<<" bits="<<(sh.voxels()*8)<<" shape="<<sh.x<<"x"<<sh.y<<"x"<<sh.z<<" pattern="<<pattern<<"\n";}
std::vector<std::string> split(const std::string& s,char delim){std::stringstream ss(s);std::string q;std::vector<std::string> r;while(std::getline(ss,q,delim))r.push_back(q);return r;}
Digest from_hex(const std::string& s){check(s.size()==64,"Incorrect SHA-256 hex length");Digest d{};for(int i=0;i<32;++i){auto charhex=[](char c)->int{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;fail("Invalid SHA-256 hex character");};d[i]=u8(charhex(s[2*i])*16+charhex(s[2*i+1]));}return d;}
u64 number(const std::string& s){size_t end=0;u64 val=std::stoull(s,&end);check(end==s.size(),"Invalid integer: "+s);return val;}
void verify_trace(const std::string& source,const std::string& result,const std::string& tracepath,Shape sh,u32 gain,bool fast){sh.valid();check_file_size(source,sh);check_file_size(result,sh);std::ifstream before(source,std::ios::binary),after(result,std::ios::binary),tr(tracepath);check(bool(tr),"Trace not found");std::string line;check(bool(std::getline(tr,line)),"Empty trace");check(line.find("iteration,z,kernel_id")==0,"Invalid trace header");std::vector<u8> prev,cur,next,expect,actual;read_plane(before,sh.z-1,sh,prev);read_plane(before,0,sh,cur);read_plane(before,sh.z>1?1:0,sh,next);Chain chain(sh,gain);
 u64 total_changes=0,total_flips=0;
 for(u32 z=0;z<sh.z;++z){check(bool(std::getline(tr,line)),"Missing plane trace");auto v=split(line,',');check(v.size()==28,"Trace row field count mismatch");check(number(v[0])==0&&number(v[1])==z,"Expected single-step trace in ascending z order");check(v[2]==(fast?"M6_FAST":"M6_REF"),"Kernel label mismatch");check(number(v[3])==sh.plane(),"Plane size mismatch");calculate(prev,cur,next,sh,gain,expect,fast);read_plane(after,z,sh,actual);check(actual==expect,"Replayed spatial update disagrees at plane "+std::to_string(z));auto st=compare(cur,actual,sh);auto en=encode(actual,sh);Digest d0=sha(cur),d1=sha(actual);chain.update(0,z,fast,d0,d1,st,en.mode,u32(en.bytes.size()));check(number(v[4])==st.changed_bytes&&number(v[5])==st.changed_bits,"Bit changes mismatch");check(number(v[6])==st.in_bits&&number(v[7])==st.out_bits,"One bits mismatch");for(int j=0;j<8;++j)check(number(v[8+j])==st.flips[j],"Bit plane flip mismatch");check(v[16]==codec_name(en.mode)&&number(v[17])==en.bytes.size(),"Codec record mismatch");check(number(v[18])==crc32(actual),"CRC field mismatch");check(from_hex(v[19])==d0&&from_hex(v[20])==d1,"SHA state mismatch");check(from_hex(v[21])==chain.value,"Provenance hash chain mismatch");check(std::stoi(v[22])==st.first_x&&std::stoi(v[23])==st.first_y&&number(v[24])==st.first_before&&number(v[25])==st.first_after,"First-change coordinate mismatch");check(v[27]=="COMMIT","Trace not committed");total_flips+=st.changed_bits;total_changes+=st.changed_bytes;
 prev.swap(cur);cur.swap(next);if(z+1<sh.z)read_plane(before,(z+2)%sh.z,sh,next);
 }
 check(!std::getline(tr,line),"Trailing trace lines");std::cout<<"REPLAY_VERIFIED bits="<<(sh.voxels()*8)<<" changed_bits="<<total_flips<<" changed_voxels="<<total_changes<<" chain_sha256="<<hex(chain.value)<<"\n";}
void export_events(const std::string& before_path,const std::string& after_path,const std::string& csv_path,Shape sh,u64 limit){
 sh.valid();check(limit>0&&limit<=1000000,"Bit-event limit must be 1..1000000");check_file_size(before_path,sh);check_file_size(after_path,sh);
 std::ifstream a(before_path,std::ios::binary),b(after_path,std::ios::binary);check(bool(a)&&bool(b),"Cannot open bit-event sources");
 std::ofstream f(csv_path+".partial",std::ios::trunc);check(bool(f),"Cannot open event output");
 f<<"z,y,x,bit,linear_byte_offset,before_bit,after_bit,before_byte,after_byte,xor_mask\n";
 u64 events=0;std::vector<u8> p,q;
 try{
  for(u32 z=0;z<sh.z&&events<limit;++z){read_plane(a,z,sh,p);read_plane(b,z,sh,q);
   for(size_t i=0;i<p.size()&&events<limit;++i){u8 diff=u8(p[i]^q[i]);if(!diff)continue;
    for(int bit=0;bit<8&&events<limit;++bit)if((diff>>bit)&1u){
     f<<z<<","<<(i/sh.x)<<","<<(i%sh.x)<<","<<bit<<","<<(u64(z)*sh.plane()+i)<<","<<unsigned((p[i]>>bit)&1u)<<","<<unsigned((q[i]>>bit)&1u)<<","<<unsigned(p[i])<<","<<unsigned(q[i])<<","<<unsigned(diff)<<"\n";
     ++events;
    }
   }
  }
  f.close();check(bool(f),"Bit-event output close failed");fs::rename(csv_path+".partial",csv_path);
 }catch(...){f.close();fs::remove(csv_path+".partial");throw;}
 std::cout<<"BIT_EVENTS_EXPORTED count="<<events<<" source_bytes="<<sh.voxels()<<" output="<<csv_path<<"\n";
}
void selftest(){for(u32 y=1;y<=8;++y){Shape sh{u32(7+y),u32(5+y),3};sh.valid();std::vector<u8> a(size_t(sh.plane()));for(size_t i=0;i<a.size();++i)a[i]=u8((i*37+y*43)&255);for(auto v:a){(void)v;}auto e=encode(a,sh);check(decode(e,sh)==a,"Codec selftest failure");auto pr=predict_x(a,sh);check(invert_x(pr,sh)==a,"Predictor inversion failure");auto next=a,prev=a,out1=a,out2=a;for(u32 g:{0u,64u,128u,256u}){calculate(prev,a,next,sh,g,out1,false);calculate(prev,a,next,sh,g,out2,true);check(out1==out2,"Kernel comparison failure");}auto st=compare(a,out1,sh);check(std::accumulate(st.flips.begin(),st.flips.end(),u64(0))==st.changed_bits,"Bitplane count mismatch");}
 for(u32 i=0;i<256;++i){Shape sh{256,1,1};std::vector<u8> x(256,u8(i));auto e=encode(x,sh);check(decode(e,sh)==x,"Exhaustive byte codec failure");}
 // Archive and replay independent from the single-plane codec test.
 fs::path dir=fs::temp_directory_path()/"moagi_bittrace_selftest";fs::create_directories(dir);auto s=[&](const char* name){return (dir/name).string();};
 try{Shape sh{17,11,5};generate(s("in.raw"),sh,"ramp");Config c; c.sh=sh;c.gain=64;c.input=s("in.raw");c.output=s("out.raw");c.trace=s("trace.csv");c.archive=s("archive.m3bt");run(c);verify_trace(c.input,c.output,c.trace,sh,64,true);decode_archive(c.archive,s("decoded.raw"));check(fs::file_size(s("decoded.raw"))==sh.voxels(),"Decoded file size mismatch");std::ifstream f1(s("out.raw"),std::ios::binary),f2(s("decoded.raw"),std::ios::binary);std::vector<u8> b1((std::istreambuf_iterator<char>(f1)),{}),b2((std::istreambuf_iterator<char>(f2)),{});check(b1==b2,"Archive decode mismatch");
 // Trace tamper rejection: alter a recorded chain hash, require replay failure.
 {std::ifstream tr_in(s("trace.csv"));std::string h,ln;std::getline(tr_in,h);std::getline(tr_in,ln);auto col=split(ln,',');check(col.size()==28,"Unexpected trace format in test");col[21][0]=(col[21][0]=='a'?'b':'a');std::ofstream tr_bad(s("trace_bad.csv"));tr_bad<<h<<"\n";for(size_t j=0;j<col.size();++j){if(j)tr_bad<<",";tr_bad<<col[j];}tr_bad<<"\n";while(std::getline(tr_in,ln))tr_bad<<ln<<"\n";tr_bad.close();bool detected_trace=false;try{verify_trace(c.input,c.output,s("trace_bad.csv"),sh,64,true);}catch(...){detected_trace=true;}check(detected_trace,"Altered provenance trace accepted");}
 export_events(c.input,c.output,s("events.csv"),sh,64);
 // Provenance tamper rejection.
 std::fstream corrupt(s("archive.m3bt"),std::ios::in|std::ios::out|std::ios::binary);check(bool(corrupt),"Corruption test file open failed");corrupt.seekg(81);char x=0;corrupt.read(&x,1);corrupt.seekp(81);x^=0x55;corrupt.write(&x,1);corrupt.close();bool detected=false;try{decode_archive(s("archive.m3bt"),s("bad.raw"));}catch(...){detected=true;}check(detected,"Corrupt archive accepted");
 }catch(...){fs::remove_all(dir);throw;}fs::remove_all(dir);
 std::cout<<"SELFTEST_OK exhaustive_byte_values=256 spatial_kernels=pass codec=pass stream_replay=pass tamper_rejection=pass\n";
}
std::string required(int& i,int argc,char**argv){check(i<argc,"Missing CLI argument");return argv[i++];}
void usage(){std::cout<<"Dr Moagi 1000MB 3D bit-trace engine\n"
 "  moagi_bittrace generate <raw> <x> <y> <z> <checker|constant|impulse|ramp>\n"
 "  moagi_bittrace run <input.raw> <output.raw> <trace.csv> <x> <y> <z> <gain_q8:0..256> [steps:1..64] [fast|ref] [archive.m3bt]\n"
 "  moagi_bittrace replay <input.raw> <output.raw> <trace.csv> <x> <y> <z> <gain_q8> [fast|ref]  (single-step)\n"
 "  moagi_bittrace decode <archive.m3bt> <output.raw>\n"
 "  moagi_bittrace events <before.raw> <after.raw> <events.csv> <x> <y> <z> <event_limit>\n"
 "  moagi_bittrace selftest\n";}
int main(int argc,char** argv){try{if(argc<2){usage();return 2;}std::string cmd=argv[1];int i=2;
 if(cmd=="generate"){auto path=required(i,argc,argv);Shape sh{u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv)))};auto pattern=required(i,argc,argv);check(i==argc,"Trailing generate arguments");generate(path,sh,pattern);}
 else if(cmd=="run"){Config c;c.input=required(i,argc,argv);c.output=required(i,argc,argv);c.trace=required(i,argc,argv);c.sh={u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv)))};c.gain=u32(number(required(i,argc,argv)));if(i<argc)c.steps=u32(number(required(i,argc,argv)));if(i<argc){auto kernel=required(i,argc,argv);check(kernel=="fast"||kernel=="ref","Kernel must be fast/ref");c.fast=(kernel=="fast");}if(i<argc)c.archive=required(i,argc,argv);check(i==argc,"Trailing run arguments");run(c);}
 else if(cmd=="replay"){std::string input=required(i,argc,argv),output=required(i,argc,argv),trace=required(i,argc,argv);Shape sh{u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv)))};u32 g=u32(number(required(i,argc,argv)));std::string kernel=i<argc?required(i,argc,argv):"fast";check(kernel=="fast"||kernel=="ref","Invalid kernel");check(i==argc,"Trailing replay arguments");verify_trace(input,output,trace,sh,g,kernel=="fast");}
 else if(cmd=="decode"){auto src=required(i,argc,argv),dst=required(i,argc,argv);check(i==argc,"Trailing decode args");decode_archive(src,dst);}
 else if(cmd=="events"){auto src=required(i,argc,argv),dst=required(i,argc,argv),out=required(i,argc,argv);Shape sh{u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv))),u32(number(required(i,argc,argv)))};auto limit=number(required(i,argc,argv));check(i==argc,"Trailing events args");export_events(src,dst,out,sh,limit);}
 else if(cmd=="selftest"){check(i==argc,"Trailing selftest args");selftest();}
 else{usage();return 2;}
 return 0;}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}