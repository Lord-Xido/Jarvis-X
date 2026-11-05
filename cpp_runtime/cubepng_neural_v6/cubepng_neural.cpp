// JARVIS-X CubePNG NEURAL STREAM v6 — lossless streaming 3D Morton predictive codec.
// C++20, zlib, libpng, OpenSSL. Reference implementation; trained quantized 3-8-1 MLP; NOT a VQ-VAE.
// Build: g++ -O3 -std=c++20 -Wall -Wextra -Wpedantic cubepng_neural.cpp -o cubepng_neural $(pkg-config --cflags --libs libpng zlib openssl)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>  // mkstemp, close: exclusive owner-only staging files (POSIX)
#include <png.h>
#include <zlib.h>
#include <openssl/evp.h>

namespace jx {
using U8=std::uint8_t; using U32=std::uint32_t; using U64=std::uint64_t;
using Bytes=std::vector<U8>;
constexpr std::size_t C=32768;
constexpr std::array<U8,8> MAGIC={'J','X','3','S','T','R','3','\0'};
constexpr U32 VERSION=3;
void req(bool c,const std::string& what){if(!c) throw std::runtime_error(what);}
void put32(std::ostream& os,U32 n){for(int i=0;i<4;++i)os.put(char((n>>(8*i))&255));}
void put64(std::ostream& os,U64 n){for(int i=0;i<8;++i)os.put(char((n>>(8*i))&255));}
U32 get32(std::istream& is){U32 r=0;for(int i=0;i<4;++i){int c=is.get();req(c!=EOF,"Truncated u32");r|=U32(c)<<(8*i);}return r;}
U64 get64(std::istream& is){U64 r=0;for(int i=0;i<8;++i){int c=is.get();req(c!=EOF,"Truncated u64");r|=U64(c)<<(8*i);}return r;}
void writeAll(std::ostream& os,const U8* p,std::size_t n){if(n)os.write(reinterpret_cast<const char*>(p),std::streamsize(n));req(bool(os),"Output write failure");}
void readAll(std::istream& is,U8* p,std::size_t n){if(n)is.read(reinterpret_cast<char*>(p),std::streamsize(n));req(bool(is),"Truncated or unreadable input");}
struct Sha {
 EVP_MD_CTX* c=EVP_MD_CTX_new();
 Sha(){req(c!=nullptr&&EVP_DigestInit_ex(c,EVP_sha256(),nullptr)==1,"SHA init failed");}
 Sha(const Sha&)=delete;Sha& operator=(const Sha&)=delete;
 ~Sha(){EVP_MD_CTX_free(c);}
 void update(const U8* p,std::size_t n){req(EVP_DigestUpdate(c,p,n)==1,"SHA update failed");}
 std::array<U8,32> finish(){std::array<U8,32> out{};unsigned n=0;req(EVP_DigestFinal_ex(c,out.data(),&n)==1&&n==32,"SHA finalize failed");return out;}
};
std::string hex(const std::array<U8,32>& d){static const char* H="0123456789abcdef";std::string s;for(U8 x:d){s+=H[x>>4];s+=H[x&15];}return s;}
struct XYZ {int x,y,z;};
struct Mapping {
 std::array<XYZ,C> xyz{};std::array<std::uint16_t,C> idx{};
 Mapping(){for(std::size_t m=0;m<C;++m){XYZ v{0,0,0};for(int i=0;i<5;++i){v.x|=int((m>>(3*i))&1)<<i;v.y|=int((m>>(3*i+1))&1)<<i;v.z|=int((m>>(3*i+2))&1)<<i;}xyz[m]=v;idx[m]=std::uint16_t((v.z*32+v.y)*32+v.x);}}
};
const Mapping& mapping(){static const Mapping m;return m;}
struct Weights{std::int16_t a=0,b=0,c=0,bias=0;};
#include "neural_mlp.hpp"
U8 prediction(U8 k,const std::array<U8,C>& voxel,XYZ p,Weights w,const NeuralWeights* nn=nullptr){
 const int a=p.x?voxel[std::size_t((p.z*32+p.y)*32+p.x-1)]:0;
 const int b=p.y?voxel[std::size_t((p.z*32+p.y-1)*32+p.x)]:0;
 const int c=p.z?voxel[std::size_t(((p.z-1)*32+p.y)*32+p.x)]:0;
 switch(k){
 case 0:return 0;case 1:return U8(a);case 2:return U8(b);case 3:return U8(c);
 case 4:return U8((a+b+c)/3);
 case 5:return U8(a+b+c-std::min({a,b,c})-std::max({a,b,c}));
 case 6:return U8(a^b^c);
 case 7:{int d=a+b-c,da=std::abs(d-a),db=std::abs(d-b),dc=std::abs(d-c);return U8(da<=db&&da<=dc?a:(db<=dc?b:c));}
 case 9:{req(nn!=nullptr,"Missing neural weights");return neuralPredict(a,b,c,*nn);}
 case 8:{const int linear=int(w.a)*a+int(w.b)*b+int(w.c)*c;
         // Integer arithmetic is identical on encoder and decoder on all platforms.
         const int v=(linear>=0?(linear+128)/256:-((-linear+128)/256))+int(w.bias);
         return U8(std::clamp(v,0,255));}
 default:throw std::runtime_error("Unknown predictor");
 }
}
// Causal 3D linear predictor: fit a 4-parameter, ridge-regularized least
// squares model to voxels that precede each sample in Morton order. 4x4 normal
// system solved with pivoted elimination, then parameters quantized to int16.
Weights learn(const U8* src,std::size_t n){
 std::array<U8,C> voxel{};double s[4][5]{};
 const auto& m=mapping();
 for(std::size_t i=0;i<n;++i){const XYZ p=m.xyz[i];
  const double a=p.x?voxel[std::size_t((p.z*32+p.y)*32+p.x-1)]/255.0:0;
  const double b=p.y?voxel[std::size_t((p.z*32+p.y-1)*32+p.x)]/255.0:0;
  const double c=p.z?voxel[std::size_t(((p.z-1)*32+p.y)*32+p.x)]/255.0:0;
  if(i%4==0){const double v[4]={a,b,c,1.0};const double y=src[i]/255.0;
   for(int j=0;j<4;++j){for(int k=0;k<4;++k)s[j][k]+=v[j]*v[k];s[j][4]+=v[j]*y;}}
  voxel[m.idx[i]]=src[i];
 }
 for(int j=0;j<4;++j)s[j][j]+=(j==3?1.0:3.0); // regularization
 for(int j=0;j<4;++j){int pivot=j;for(int i=j+1;i<4;++i)if(std::abs(s[i][j])>std::abs(s[pivot][j]))pivot=i;
  for(int k=j;k<=4;++k)std::swap(s[j][k],s[pivot][k]);
  if(std::abs(s[j][j])<1e-10)return {};
  const double inv=1.0/s[j][j];for(int k=j;k<=4;++k)s[j][k]*=inv;
  for(int i=0;i<4;++i)if(i!=j){const double t=s[i][j];for(int k=j;k<=4;++k)s[i][k]-=t*s[j][k];}
 }
 auto q=[](double d){return std::int16_t(std::clamp(std::lround(d),-2048L,2048L));};
 return {q(s[0][4]*256.0),q(s[1][4]*256.0),q(s[2][4]*256.0),q(s[3][4]*255.0)};
}
Bytes residualize(const U8* src,std::size_t n,U8 kernel,Weights w,const NeuralWeights* nn=nullptr,SpatialContext ctx={}){
 if(kernel==9 && ctx.pixels){req(nn!=nullptr&&ctx.rowBytes>=4,"Missing pixel MLP context");
  Bytes res(n);for(std::size_t i=0;i<n;++i){const auto x=pixelNeighbors(src,i,ctx);
   res[i]=src[i]^neuralPredict(x[0],x[1],x[2],*nn);}return res;
 }
 std::array<U8,C> voxel{};Bytes result(n);const auto& m=mapping();
 for(std::size_t i=0;i<n;++i){result[i]=src[i]^prediction(kernel,voxel,m.xyz[i],w,nn);voxel[m.idx[i]]=src[i];}
 return result;
}
Bytes restore(const U8* src,std::size_t n,U8 kernel,Weights w,const NeuralWeights* nn=nullptr,SpatialContext ctx={}){
 if(kernel==9 && ctx.pixels){req(nn!=nullptr&&ctx.rowBytes>=4,"Missing pixel MLP context");
  Bytes out(n);for(std::size_t i=0;i<n;++i){const auto x=pixelNeighbors(out.data(),i,ctx);
   out[i]=src[i]^neuralPredict(x[0],x[1],x[2],*nn);}return out;
 }
 std::array<U8,C> voxel{};Bytes result(n);const auto& m=mapping();
 for(std::size_t i=0;i<n;++i){const U8 v=src[i]^prediction(kernel,voxel,m.xyz[i],w,nn);voxel[m.idx[i]]=result[i]=v;}
 return result;
}
Bytes zpack(const Bytes& b,int level){uLongf size=compressBound(uLong(b.size()));Bytes out(std::size_t(size),0);
 req(compress2(out.data(),&size,b.data(),uLong(b.size()),level)==Z_OK,"zlib compression failed");out.resize(std::size_t(size));return out;}
Bytes zunpack(const Bytes& b,std::size_t expected){req(expected>0&&expected<=C,"Bad raw chunk length");
 Bytes out(expected);uLongf size=uLongf(expected);
 req(uncompress(out.data(),&size,b.data(),uLong(b.size()))==Z_OK&&size==expected,"Corrupt zlib chunk");return out;}
struct Record{U8 kernel=0;U32 raw=0,crc=0;Weights w{};NeuralWeights nn{};Bytes packed;};
// Additional learned neural model occupies 50 bytes when selected (kernel=9).
Record makeRecord(const U8* src,std::size_t n,int level,const std::string& strategy,SpatialContext ctx={}){
 req(n>0&&n<=C,"Invalid input chunk size");Record best;best.raw=U32(n);
 auto attempt=[&](U8 kernel,Weights w,const NeuralWeights* nn=nullptr){Bytes res=residualize(src,n,kernel,w,nn,ctx);Bytes packed=zpack(res,level);
  if(best.packed.empty()||packed.size()+(kernel==9?50u:0u)<best.packed.size()+(best.kernel==9?50u:0u)){
   best.kernel=kernel;best.w=w;if(nn)best.nn=*nn;best.crc=U32(crc32(0,res.data(),uInt(res.size())));best.packed=std::move(packed);}}
 ;
 if(strategy=="fixed") attempt(0,{});
 else if(strategy=="learned")attempt(8,learn(src,n));
 else if(strategy=="adaptive")for(U8 k=0;k<=7;++k)attempt(k,{});
 else if(strategy=="linear-hybrid"){for(U8 k=0;k<=7;++k)attempt(k,{});attempt(8,learn(src,n));}
 else if(strategy=="neural"){const auto nn=learnNeural(src,n,ctx);attempt(9,{},&nn);}
 else if(strategy=="hybrid"){for(U8 k=0;k<=7;++k)attempt(k,{});attempt(8,learn(src,n));const auto nn=learnNeural(src,n,ctx);attempt(9,{},&nn);}
 else throw std::runtime_error("Unknown strategy");
 return best;
}
void put16(std::ostream& o,std::int16_t n){const auto u=std::uint16_t(n);o.put(char(u&255));o.put(char((u>>8)&255));}
std::int16_t get16(std::istream& in){int a=in.get(),b=in.get();req(a!=EOF&&b!=EOF,"Truncated weight");
 const unsigned u=unsigned(a|(b<<8));return std::int16_t(u<32768?int(u):int(u)-65536);}
void putNeural(std::ostream& o,const NeuralWeights& nn){
 for(auto x:nn.w1)o.put(char(x));
 for(auto x:nn.b1)put16(o,x);
 for(auto x:nn.w2)o.put(char(x));
 put16(o,nn.b2);
}
NeuralWeights getNeural(std::istream& is){NeuralWeights n{};
 for(auto& x:n.w1){const int b=is.get();req(b!=EOF,"Truncated neural w1");x=std::int8_t(b<128?b:b-256);}
 for(auto& x:n.b1)x=get16(is);
 for(auto& x:n.w2){const int b=is.get();req(b!=EOF,"Truncated neural w2");x=std::int8_t(b<128?b:b-256);}
 n.b2=get16(is);return n;
}
// Detect accidental record-metadata corruption, including NN weights that
// happen not to affect a particular output. CRC32 is not an authentication MAC.
U32 modelCRC(const Record& r){
 std::array<U8,59> buf{};std::size_t i=0;
 buf[i++]=r.kernel;
 const auto put=[&](std::int16_t x){const auto u=std::uint16_t(x);buf[i++]=U8(u&255);buf[i++]=U8(u>>8);};
 put(r.w.a);put(r.w.b);put(r.w.c);put(r.w.bias);
 if(r.kernel==9){for(auto x:r.nn.w1)buf[i++]=U8(x);
  for(auto x:r.nn.b1)put(x);
  for(auto x:r.nn.w2)buf[i++]=U8(x);
  put(r.nn.b2);
 }
 return U32(crc32(0,buf.data(),uInt(i)));
}
void putRecord(std::ostream& o,const Record& r){o.put(char(r.kernel));put32(o,r.raw);put32(o,U32(r.packed.size()));put32(o,r.crc);put32(o,modelCRC(r));
 put16(o,r.w.a);put16(o,r.w.b);put16(o,r.w.c);put16(o,r.w.bias);if(r.kernel==9)putNeural(o,r.nn);writeAll(o,r.packed.data(),r.packed.size());}
Record getRecord(std::istream& is,std::size_t expected){Record r;int id=is.get();req(id!=EOF&&id>=0&&id<=9,"Truncated/unknown kernel");r.kernel=U8(id);
 r.raw=get32(is);const U32 packed=get32(is);r.crc=get32(is);const U32 metaCRC=get32(is);
 r.w={get16(is),get16(is),get16(is),get16(is)};if(r.kernel==9)r.nn=getNeural(is);
 req(modelCRC(r)==metaCRC,"Model metadata CRC32 mismatch");
 req(r.raw==expected,"Wrong chunk length");req(packed>0&&packed<=compressBound(uLong(C)),"Bad compressed size");
 r.packed.resize(packed);readAll(is,r.packed.data(),packed);
 if(r.kernel!=8)req(r.w.a==0&&r.w.b==0&&r.w.c==0&&r.w.bias==0,"Unexpected parameters for fixed kernel");
 return r;
}
struct Header{U32 mode=0,w=0,h=0;U64 raw=0,chunks=0;std::array<U8,32> digest{};};
void writeHeader(std::ostream& o,const Header& h){writeAll(o,MAGIC.data(),8);put32(o,VERSION);put32(o,h.mode);
 put32(o,h.w);put32(o,h.h);put64(o,h.raw);writeAll(o,h.digest.data(),32);put64(o,h.chunks);}
Header readHeader(std::istream& in,U64 maxBytes){std::array<U8,8> magic{};readAll(in,magic.data(),8);req(magic==MAGIC,"Not a JX3STR3 archive");
 req(get32(in)==VERSION,"Unsupported archive version");Header h;h.mode=get32(in);h.w=get32(in);h.h=get32(in);h.raw=get64(in);
 readAll(in,h.digest.data(),32);h.chunks=get64(in);
 req(h.mode<=1&&h.raw>0&&h.raw<=maxBytes,"Invalid archive size/mode");
 req(h.chunks==(h.raw+C-1)/C,"Inconsistent chunk count");
 if(h.mode==1)req(h.w>0&&h.h>0&&h.raw%4==0&&U64(h.w)<=h.raw/4&&U64(h.w)*h.h==h.raw/4,"Invalid PNG pixel dimensions");
 else req(h.w==0&&h.h==0,"File-mode dimensions must be zero");
 return h;
}
class Encoder {
 std::fstream out_;Bytes hold_;Sha sha_;U64 raw_=0,count_=0,max_;U32 mode_,w_,h_;int level_;
 std::string strategy_;std::array<U64,10> hist_{};
public:
 Encoder(const std::string& path,U32 mode,U32 w,U32 h,int level,std::string strategy,U64 maxBytes):
  out_(path,std::ios::in|std::ios::out|std::ios::binary|std::ios::trunc),max_(maxBytes),mode_(mode),w_(w),h_(h),level_(level),strategy_(std::move(strategy)){
  req(bool(out_),"Cannot create archive output");req(level>=0&&level<=9,"Invalid compression level");
  writeHeader(out_,Header{mode,w,h,0,0,{}});hold_.reserve(C);
 }
 void feed(const U8* p,std::size_t n){req(U64(n)<=max_-raw_,"Input exceeds --max-mib limit");sha_.update(p,n);raw_+=n;
  while(n){const std::size_t add=std::min(C-hold_.size(),n);hold_.insert(hold_.end(),p,p+add);p+=add;n-=add;
   if(hold_.size()==C)flush();}}
 void flush(){if(hold_.empty())return;const SpatialContext ctx{mode_==1,std::size_t(w_)*4,count_*C};auto r=makeRecord(hold_.data(),hold_.size(),level_,strategy_,ctx);putRecord(out_,r);hist_[r.kernel]++;count_++;hold_.clear();}
 Header finish(){flush();req(raw_>0,"Empty input");if(mode_==1)req(U64(w_)*h_*4==raw_,"Pixel byte count mismatch");
  Header h;h.mode=mode_;h.w=w_;h.h=h_;h.raw=raw_;h.chunks=count_;h.digest=sha_.finish();
  out_.seekp(0);writeHeader(out_,h);out_.flush();req(bool(out_),"Failed to finalize archive");out_.close();
  std::cout<<"ENCODE OK: mode="<<(mode_?"pixels":"file")<<" raw="<<raw_<<" chunks="<<count_<<" strategy="<<strategy_<<" sha256="<<hex(h.digest)<<"\n";
  std::cout<<"KERNEL HISTOGRAM:";for(int k=0;k<10;++k)std::cout<<" "<<k<<":"<<hist_[k];std::cout<<"\n";
  return h;}
};
class Decoder {
 std::ifstream in_;Header header_;U64 max_;
public:
 Decoder(const std::string& path,U64 maxBytes):in_(path,std::ios::binary),max_(maxBytes){req(bool(in_),"Cannot open archive");header_=readHeader(in_,max_);}
 const Header& header()const{return header_;}
 template<typename F>void run(F sink){Sha sha;U64 total=0;
  for(U64 i=0;i<header_.chunks;++i){const std::size_t n=std::size_t(std::min<U64>(C,header_.raw-total));
   auto rec=getRecord(in_,n);auto res=zunpack(rec.packed,n);
   req(U32(crc32(0,res.data(),uInt(n)))==rec.crc,"Residual CRC32 mismatch");
   const SpatialContext ctx{header_.mode==1,std::size_t(header_.w)*4,total};
   auto bytes=restore(res.data(),n,rec.kernel,rec.w,rec.kernel==9?&rec.nn:nullptr,ctx);sha.update(bytes.data(),n);sink(bytes.data(),n);total+=n;}
  req(total==header_.raw,"Length mismatch");req(in_.peek()==EOF,"Trailing data in archive");
  req(sha.finish()==header_.digest,"SHA256 mismatch after reconstruction");}
};
// Streaming libpng row reader/writer. Interlaced and 16-bit PNGs use file-exact
// mode; pixel mode intentionally normalizes to 8-bit RGBA and omits metadata.
class PngReader {
 FILE* f_=nullptr;png_structp png_=nullptr;png_infop info_=nullptr;U32 w_=0,h_=0;std::size_t rb_=0;
public:
 explicit PngReader(const std::string& path){f_=std::fopen(path.c_str(),"rb");req(f_!=nullptr,"Unable to open PNG");
  png_=png_create_read_struct(PNG_LIBPNG_VER_STRING,nullptr,nullptr,nullptr);req(png_!=nullptr,"png_create_read_struct failed");
  info_=png_create_info_struct(png_);req(info_!=nullptr,"png_create_info_struct failed");
  if(setjmp(png_jmpbuf(png_)))throw std::runtime_error("libpng reader initialization failed");
  png_init_io(png_,f_);png_read_info(png_,info_);
  w_=png_get_image_width(png_,info_);h_=png_get_image_height(png_,info_);
  req(w_>0&&h_>0,"Zero PNG dimensions");req(png_get_bit_depth(png_,info_)!=16,"16-bit PNG: use file mode");
  req(png_get_interlace_type(png_,info_)==PNG_INTERLACE_NONE,"Interlaced PNG: use file mode");
  const int color=png_get_color_type(png_,info_);
  if(color==PNG_COLOR_TYPE_PALETTE)png_set_palette_to_rgb(png_);
  if(color==PNG_COLOR_TYPE_GRAY&&png_get_bit_depth(png_,info_)<8)png_set_expand_gray_1_2_4_to_8(png_);
  const bool hasTrns=png_get_valid(png_,info_,PNG_INFO_tRNS)!=0;
  if(hasTrns)png_set_tRNS_to_alpha(png_);
  if(color==PNG_COLOR_TYPE_GRAY||color==PNG_COLOR_TYPE_GRAY_ALPHA)png_set_gray_to_rgb(png_);
  if(!(color & PNG_COLOR_MASK_ALPHA)&&!hasTrns)png_set_filler(png_,0xff,PNG_FILLER_AFTER);
  png_read_update_info(png_,info_);rb_=png_get_rowbytes(png_,info_);
  req(U64(w_)*4==rb_,"RGBA conversion did not produce 4 channels");
 }
 PngReader(const PngReader&)=delete;PngReader& operator=(const PngReader&)=delete;
 ~PngReader(){if(png_)png_destroy_read_struct(&png_,info_?&info_:nullptr,nullptr);if(f_)std::fclose(f_);}
 U32 width()const{return w_;}U32 height()const{return h_;}std::size_t rowBytes()const{return rb_;}
 void row(U8* out){if(setjmp(png_jmpbuf(png_)))throw std::runtime_error("libpng row read failed");png_read_row(png_,out,nullptr);}
 void finish(){if(setjmp(png_jmpbuf(png_)))throw std::runtime_error("libpng end read failed");png_read_end(png_,info_);}
};
class PngWriter {
 FILE* f_=nullptr;png_structp png_=nullptr;png_infop info_=nullptr;
 U32 h_;std::size_t rowBytes_;Bytes row_;std::size_t used_=0;U32 rows_=0;
public:
 PngWriter(const std::string& path,U32 w,U32 h):h_(h),rowBytes_(std::size_t(w)*4),row_(rowBytes_){
  req(w>0&&h>0,"Zero image dimensions");f_=std::fopen(path.c_str(),"wb");req(f_!=nullptr,"Cannot write PNG");
  png_=png_create_write_struct(PNG_LIBPNG_VER_STRING,nullptr,nullptr,nullptr);req(png_!=nullptr,"PNG writer init failed");
  info_=png_create_info_struct(png_);req(info_!=nullptr,"PNG writer info failed");
  if(setjmp(png_jmpbuf(png_)))throw std::runtime_error("PNG header write failed");
  png_init_io(png_,f_);png_set_IHDR(png_,info_,w,h,8,PNG_COLOR_TYPE_RGBA,PNG_INTERLACE_NONE,PNG_COMPRESSION_TYPE_BASE,PNG_FILTER_TYPE_BASE);
  png_write_info(png_,info_);
 }
 PngWriter(const PngWriter&)=delete;PngWriter& operator=(const PngWriter&)=delete;
 ~PngWriter(){if(png_)png_destroy_write_struct(&png_,info_?&info_:nullptr);if(f_)std::fclose(f_);}
 void feed(const U8* p,std::size_t n){while(n){const std::size_t take=std::min(n,rowBytes_-used_);std::memcpy(row_.data()+used_,p,take);
  p+=take;n-=take;used_+=take;if(used_==rowBytes_){req(rows_<h_,"Too many decoded pixel rows");
   writeRow();used_=0;++rows_;}}}
 void writeRow(){if(setjmp(png_jmpbuf(png_)))throw std::runtime_error("PNG row write failed");
  png_write_row(png_,row_.data());}
 void finish(){req(used_==0&&rows_==h_,"Incomplete decoded pixel rows");
  if(setjmp(png_jmpbuf(png_)))throw std::runtime_error("PNG finalize failed");
  png_write_end(png_,info_);}
};
U64 limitMiB(const std::string& s){const U64 v=std::stoull(s);req(v>=1&&v<=16384,"--max-mib must be 1..16384");return v*1024*1024;}
void verifyPNGFile(std::istream& is){std::array<U8,33> h{};readAll(is,h.data(),h.size());
 const U8 pngSig[8]={137,80,78,71,13,10,26,10};
 req(std::equal(h.begin(),h.begin()+8,pngSig)&&std::memcmp(h.data()+12,"IHDR",4)==0,"Missing PNG signature/IHDR");
 is.clear();is.seekg(0);req(bool(is),"Cannot rewind PNG");}
struct Config{std::string strategy="hybrid";std::string mode="file";int level=6;U64 maxBytes=1024ull*1024*1024;};
// Exclusive, unpredictable, owner-only staging path. The file is created with mode 0600.
std::string stagePath(const std::string& output){
 std::string pattern=output+".partial.XXXXXX";
 std::vector<char> buf(pattern.begin(),pattern.end());buf.push_back('\0');
 const int fd=::mkstemp(buf.data());req(fd>=0,"Cannot create exclusive staging file");
 ::close(fd);return std::string(buf.data());
}
// No-overwrite atomic publish (same-directory hard link); avoids replacing existing output.
void safeMove(const std::string& temp,const std::string& final){
 std::filesystem::create_hard_link(temp,final); // fails if target exists
 std::filesystem::remove(temp);
}
void encodeFile(const std::string& input,const std::string& output,const Config& opt){
 req(std::filesystem::absolute(input)!=std::filesystem::absolute(output),"Input and output paths coincide");
 req(!std::filesystem::exists(output),"Output already exists; refusing to overwrite");
 const std::string tmp=stagePath(output);
 try{
  if(opt.mode=="pixels"){
   PngReader png(input);req(U64(png.width())*png.height()<=opt.maxBytes/4,"Image exceeds memory/byte limit");
   Encoder enc(tmp,1,png.width(),png.height(),opt.level,opt.strategy,opt.maxBytes);
   Bytes row(png.rowBytes());for(U32 y=0;y<png.height();++y){png.row(row.data());enc.feed(row.data(),row.size());}
   png.finish();(void)enc.finish();
  }else{
   std::ifstream f(input,std::ios::binary);req(bool(f),"Cannot open input PNG");verifyPNGFile(f);
   Encoder enc(tmp,0,0,0,opt.level,opt.strategy,opt.maxBytes);
   std::array<U8,65536> buffer{};while(f){f.read(reinterpret_cast<char*>(buffer.data()),buffer.size());
    const std::streamsize n=f.gcount();if(n>0)enc.feed(buffer.data(),std::size_t(n));}
   req(f.eof(),"PNG read error");(void)enc.finish();
  }
  safeMove(tmp,output);
 }catch(...){std::filesystem::remove(tmp);throw;}
}
void decodeFile(const std::string& source,const std::string& output,U64 maxBytes){
 req(std::filesystem::absolute(source)!=std::filesystem::absolute(output),"Input and output paths coincide");
 req(!std::filesystem::exists(output),"Output already exists; refusing to overwrite");
 const std::string tmp=stagePath(output);
 try{
  Decoder decoder(source,maxBytes);const auto& h=decoder.header();
  if(h.mode==1){PngWriter png(tmp,h.w,h.h);decoder.run([&](const U8* p,std::size_t n){png.feed(p,n);});png.finish();}
  else{std::ofstream out(tmp,std::ios::binary);req(bool(out),"Cannot create decoded output");
   decoder.run([&](const U8* p,std::size_t n){writeAll(out,p,n);});out.close();req(bool(out),"Cannot finalize decoded file");
   std::ifstream check(tmp,std::ios::binary);verifyPNGFile(check);}
  safeMove(tmp,output);std::cout<<"DECODE OK: raw="<<h.raw<<" sha256="<<hex(h.digest)<<"\n";
 }catch(...){std::filesystem::remove(tmp);throw;}
}
void verifyFile(const std::string& source,const std::string& archive,U64 maxBytes){
 Decoder decoder(archive,maxBytes);const auto& h=decoder.header();
 if(h.mode==0){std::ifstream ref(source,std::ios::binary);req(bool(ref),"Cannot read reference PNG");
  decoder.run([&](const U8* p,std::size_t n){Bytes b(n);readAll(ref,b.data(),n);req(std::equal(b.begin(),b.end(),p),"PNG file bytes differ");});
  req(ref.peek()==EOF,"Reference has trailing bytes");}
 else{PngReader ref(source);req(ref.width()==h.w&&ref.height()==h.h,"Image dimensions differ");
  Bytes row(ref.rowBytes());std::size_t used=0;U32 rows=0;
  decoder.run([&](const U8* p,std::size_t n){while(n){if(used==0){req(rows<h.h,"Reference too short");ref.row(row.data());++rows;}
   const std::size_t take=std::min(n,row.size()-used);req(std::memcmp(row.data()+used,p,take)==0,"RGBA pixels differ");
   used+=take;n-=take;p+=take;if(used==row.size())used=0;}});
  req(used==0&&rows==h.h,"Reference PNG rows incomplete");ref.finish();}
 std::cout<<"VERIFY PASSED: exact "<<(h.mode?"decoded RGBA pixels":"PNG file bytes")<<" sha256="<<hex(h.digest)<<"\n";
}
void selfTest(){std::mt19937 r(12345);
 for(std::size_t n:{std::size_t(1),std::size_t(7),std::size_t(31),std::size_t(1023),C-1,C}){
  Bytes b(n);for(auto& c:b)c=U8(r()&255);const Weights w=learn(b.data(),n);
  const auto nn=learnNeural(b.data(),n);
  for(U8 k=0;k<=9;++k){const auto z=residualize(b.data(),n,k,w,k==9?&nn:nullptr);
   req(restore(z.data(),n,k,w,k==9?&nn:nullptr)==b,"Predictor mismatch");}
  for(const auto& strategy:{"fixed","adaptive","learned","linear-hybrid","neural","hybrid"}){
   auto rec=makeRecord(b.data(),n,3,strategy);auto res=zunpack(rec.packed,n);
   req(U32(crc32(0,res.data(),uInt(n)))==rec.crc,"CRC mismatch");
   req(restore(res.data(),n,rec.kernel,rec.w,rec.kernel==9?&rec.nn:nullptr)==b,"Strategy mismatch");}
 }
 Bytes structured(C);const auto& m=mapping();
 for(std::size_t i=0;i<C;++i){const XYZ p=m.xyz[i];structured[i]=U8(180+(p.x+2*p.y+p.z)%17);}
 const auto samples=trainingSamples(structured.data(),structured.size());FloatMLP start;FloatMLP trained=start;
 trained.train(samples);req(trained.loss(samples)<start.loss(samples),"Training did not improve MSE");
 const auto nn=trained.quantize();auto res=residualize(structured.data(),C,9,{},&nn);
 req(restore(res.data(),C,9,{},&nn)==structured,"Structured neural reconstruction failed");
 const SpatialContext px{true,257u*4,0};const auto nnp=learnNeural(structured.data(),C,px);
 const auto resp=residualize(structured.data(),C,9,{},&nnp,px);
 req(restore(resp.data(),C,9,{},&nnp,px)==structured,"Pixel-spatial neural reconstruction failed");
 std::cout<<"SELF-TEST PASSED: 10 kernels, quantized MLP trained by SGD, exact residual, metadata CRC, partial chunks\n";
}
void help(){std::cout<<"CubePNG NEURAL STREAM v6\n"
 <<"  cubepng_neural encode input.png output.jx6 [--mode file|pixels] [--strategy fixed|adaptive|learned|linear-hybrid|neural|hybrid] [--level 0..9] [--max-mib N]\n"
 <<"  cubepng_neural decode input.jx6 output.png [--max-mib N]\n"
 <<"  cubepng_neural verify original.png input.jx6 [--max-mib N]\n"
 <<"  cubepng_neural self-test\n";}
} // jx
int main(int argc,char** argv){using namespace jx;
 try{if(argc==2&&std::string(argv[1])=="self-test"){selfTest();return 0;}if(argc<4){help();return 2;}
  const std::string cmd=argv[1];Config opt;
  for(int i=4;i<argc;++i){const std::string k=argv[i];req(i+1<argc,"Missing value: "+k);const std::string v=argv[++i];
   if(k=="--mode")opt.mode=v;else if(k=="--strategy")opt.strategy=v;
   else if(k=="--level")opt.level=std::stoi(v);else if(k=="--max-mib")opt.maxBytes=limitMiB(v);
   else throw std::runtime_error("Unknown option: "+k);}
  req(opt.mode=="file"||opt.mode=="pixels","Unknown mode");
  const auto start=std::chrono::steady_clock::now();
  if(cmd=="encode")encodeFile(argv[2],argv[3],opt);
  else if(cmd=="decode")decodeFile(argv[2],argv[3],opt.maxBytes);
  else if(cmd=="verify")verifyFile(argv[2],argv[3],opt.maxBytes);
  else throw std::runtime_error("Unknown command");
  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::cout<<"TIME_MS="<<std::fixed<<std::setprecision(2)<<ms<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}