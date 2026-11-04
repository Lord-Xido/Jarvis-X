// JARVIS-X 3D ANN Tile Autoencoder v1.0 (C++20, standard library only)
// 1000^3 logical byte voxels, 10^3 tiles, shared trainable 3D convolution weights.
// Lossy: learned 4x5x5x5 uint8 latent. Lossless: latent plus exact XOR residual.
// This is an ANN, not cryptography; no claimed speedup or compression guarantee.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <map>
#include <utility>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace jx {
constexpr int DIM=1000, SIDE=10, GRID=DIM/SIDE, COARSE=SIDE/2;
constexpr int VOX=SIDE*SIDE*SIDE, CHANNELS=4, KERNEL=27;
constexpr int LATENT=CHANNELS*COARSE*COARSE*COARSE;
constexpr std::uint64_t LOGICAL_BYTES=1000000000ULL;
constexpr std::uint32_t MAX_TILES=1000000U;
using Tile=std::array<std::uint8_t,VOX>;
using Features=std::array<float,LATENT>;
using Code=std::array<std::uint8_t,LATENT>;
constexpr std::array<char,8> CODE_MAGIC={'J','X','A','E','3','D','0','3'};
constexpr std::array<char,8> MODEL_MAGIC={'J','X','M','O','D','0','0','3'};

int clampi(int v,int lo,int hi) {return std::clamp(v,lo,hi);}
int vid(int x,int y,int z) {return (z*SIDE+y)*SIDE+x;}
int cid(int x,int y,int z) {return (z*COARSE+y)*COARSE+x;}
int fid(int ch,int x,int y,int z) {return ch*125+cid(x,y,z);}
int kid(int dx,int dy,int dz) {return ((dz+1)*3+(dy+1))*3+(dx+1);}
int tile_id(int x,int y,int z) {
    if(x<0||y<0||z<0||x>=DIM||y>=DIM||z>=DIM) throw std::runtime_error("coordinates out of range");
    return ((z/SIDE)*GRID+(y/SIDE))*GRID+x/SIDE;
}
int local_id(int x,int y,int z) {return vid(x%SIDE,y%SIDE,z%SIDE);}
float sigmoid(float a) {
    if(a>=0.f) {const float t=std::exp(-a);return 1.f/(1.f+t);}
    const float t=std::exp(a);return t/(1.f+t);
}
float voxel(const Tile& a,int x,int y,int z) {
    return a[vid(clampi(x,0,9),clampi(y,0,9),clampi(z,0,9))]/255.f;
}
struct Model {
    std::array<float,CHANNELS*KERNEL> encoder{},decoder{};
    std::array<float,CHANNELS> ebias{};
    float dbias=0;
};
Model init_model() {
    std::mt19937 rng(20261104U);
    std::uniform_real_distribution<float> unif(-0.12f,0.12f);
    Model m;
    for(float& w:m.encoder) w=unif(rng);
    for(float& w:m.decoder) w=unif(rng);
    for(float& b:m.ebias) b=unif(rng);
    return m;
}
Features encode(const Model& m,const Tile& a) {
    Features h{};
    for(int c=0;c<CHANNELS;++c)
      for(int z=0;z<COARSE;++z) for(int y=0;y<COARSE;++y) for(int x=0;x<COARSE;++x) {
        float sum=m.ebias[c];
        for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
          sum+=m.encoder[c*KERNEL+kid(dx,dy,dz)]*voxel(a,2*x+dx,2*y+dy,2*z+dz);
        h[fid(c,x,y,z)]=std::tanh(sum);
      }
    return h;
}
std::array<float,VOX> decode_float(const Model& m,const Features& h) {
    std::array<float,VOX> v{};
    for(int z=0;z<SIDE;++z) for(int y=0;y<SIDE;++y) for(int x=0;x<SIDE;++x) {
        float sum=m.dbias;
        for(int c=0;c<CHANNELS;++c)
          for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
             int xx=clampi(x/2+dx,0,4), yy=clampi(y/2+dy,0,4), zz=clampi(z/2+dz,0,4);
             sum+=m.decoder[c*KERNEL+kid(dx,dy,dz)]*h[fid(c,xx,yy,zz)];
          }
        v[vid(x,y,z)]=sigmoid(sum);
    }
    return v;
}
Code quantize(const Features& h) {
    Code q{};
    for(std::size_t i=0;i<q.size();++i)
      q[i]=static_cast<std::uint8_t>(std::lround(127.5f*(std::clamp(h[i],-1.f,1.f)+1.f)));
    return q;
}
Features dequantize(const Code& q) {
    Features h{};
    for(std::size_t i=0;i<q.size();++i) h[i]=static_cast<float>(q[i])/127.5f-1.f;
    return h;
}
Tile predict(const Model& m,const Code& q) {
    const auto f=decode_float(m,dequantize(q));Tile a{};
    for(std::size_t i=0;i<a.size();++i)
      a[i]=static_cast<std::uint8_t>(std::clamp(std::lround(f[i]*255.f),0L,255L));
    return a;
}
float mse(const Model& m,const Tile& a) {
    const auto v=decode_float(m,encode(m,a));float s=0;
    for(int i=0;i<VOX;++i) {const float d=v[i]-a[i]/255.f;s+=d*d;}
    return s/static_cast<float>(VOX);
}
// Exact analytic backpropagation through 3D convolution, tanh and sigmoid.
float train_step(Model& m,const Tile& a,float rate) {
    const Features h=encode(m,a);
    const auto p=decode_float(m,h);
    std::array<float,CHANNELS*KERNEL> dw{},ew{};
    std::array<float,CHANNELS> eb{};
    Features dh{};float db=0,loss=0;
    for(int z=0;z<SIDE;++z) for(int y=0;y<SIDE;++y) for(int x=0;x<SIDE;++x) {
        const int j=vid(x,y,z);const float err=p[j]-a[j]/255.f;
        loss+=err*err;
        const float delta=(2.f/VOX)*err*p[j]*(1.f-p[j]);db+=delta;
        for(int c=0;c<CHANNELS;++c)
          for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            const int k=c*KERNEL+kid(dx,dy,dz);
            const int t=fid(c,clampi(x/2+dx,0,4),clampi(y/2+dy,0,4),clampi(z/2+dz,0,4));
            dw[k]+=delta*h[t];dh[t]+=delta*m.decoder[k];
          }
    }
    for(int c=0;c<CHANNELS;++c)
      for(int z=0;z<COARSE;++z) for(int y=0;y<COARSE;++y) for(int x=0;x<COARSE;++x) {
        const int j=fid(c,x,y,z);const float delta=dh[j]*(1.f-h[j]*h[j]);
        eb[c]+=delta;
        for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx)
          ew[c*KERNEL+kid(dx,dy,dz)]+=delta*voxel(a,2*x+dx,2*y+dy,2*z+dz);
      }
    auto update=[rate](float& w,float g){w-=rate*std::clamp(g,-1.f,1.f);};
    for(std::size_t i=0;i<dw.size();++i) {update(m.decoder[i],dw[i]);update(m.encoder[i],ew[i]);}
    for(int c=0;c<CHANNELS;++c) update(m.ebias[c],eb[c]);
    update(m.dbias,db);
    return loss/VOX;
}
Tile synthetic(std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int type=seed%5U;
    std::uniform_real_distribution<float> uni(0.f,1.f);
    const float cx=2.f+6.f*uni(rng),cy=2.f+6.f*uni(rng),cz=2.f+6.f*uni(rng);
    const float radius=1.5f+4.f*uni(rng), phase=6.28f*uni(rng);
    Tile out{};
    for(int z=0;z<10;++z) for(int y=0;y<10;++y) for(int x=0;x<10;++x) {
        const float dx=x-cx,dy=y-cy,dz=z-cz;
        const float r=std::sqrt(dx*dx+dy*dy+dz*dz);
        float v=0;
        switch(type) {
          case 0: v=(r<radius)?1.f:0.f;break;
          case 1: v=std::exp(-0.5f*r*r/(radius*radius));break;
          case 2: v=0.5f+0.45f*std::sin(0.35f*x+0.25f*y+0.42f*z+phase);break;
          case 3: v=(dx+0.6f*dy-0.4f*dz>0.f)?0.85f:0.15f;break;
          default:v=(x+y+z)/27.f;break;
        }
        out[vid(x,y,z)]=static_cast<std::uint8_t>(std::clamp(std::lround(255.f*v),0L,255L));
    }
    return out;
}
// Strict little-endian binary I/O, independent of host endianness.
void put(std::ostream& o,std::uint64_t v,unsigned n) {
    for(unsigned i=0;i<n;++i) o.put(static_cast<char>((v>>(8*i))&255U));
    if(!o) throw std::runtime_error("write failed");
}
std::uint64_t get(std::istream& in,unsigned n) {
    std::uint64_t v=0;
    for(unsigned i=0;i<n;++i) {
        const int c=in.get();if(c==std::char_traits<char>::eof()) throw std::runtime_error("truncated stream");
        v|=std::uint64_t(static_cast<unsigned char>(c))<<(8*i);
    }
    return v;
}
std::uint32_t bits(float f) {std::uint32_t u;std::memcpy(&u,&f,4);return u;}
float real(std::uint32_t u) {float f;std::memcpy(&f,&u,4);return f;}
std::uint64_t fingerprint(const Model& m) {
    std::uint64_t h=14695981039346656037ULL;
    auto mix=[&](float v) {const auto u=bits(v);for(unsigned j=0;j<4;++j){h^=(u>>(8*j))&255U;h*=1099511628211ULL;}};
    for(float f:m.encoder) mix(f);
    for(float f:m.decoder) mix(f);
    for(float f:m.ebias) mix(f);
    mix(m.dbias);
    return h;
}
void save_model(const Model& m,const std::string& path) {
    std::ofstream o(path,std::ios::binary|std::ios::trunc);
    if(!o) throw std::runtime_error("cannot write model");
    o.write(MODEL_MAGIC.data(),8);put(o,1,4);put(o,CHANNELS,4);
    for(float f:m.encoder) put(o,bits(f),4);
    for(float f:m.decoder) put(o,bits(f),4);
    for(float f:m.ebias) put(o,bits(f),4);
    put(o,bits(m.dbias),4);put(o,fingerprint(m),8);
}
Model load_model(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in) throw std::runtime_error("cannot open model");
    char mag[8];in.read(mag,8);
    if(!in||!std::equal(mag,mag+8,MODEL_MAGIC.begin())||get(in,4)!=1||get(in,4)!=CHANNELS)
      throw std::runtime_error("unsupported model format");
    Model m;
    for(float& f:m.encoder) f=real(static_cast<std::uint32_t>(get(in,4)));
    for(float& f:m.decoder) f=real(static_cast<std::uint32_t>(get(in,4)));
    for(float& f:m.ebias) f=real(static_cast<std::uint32_t>(get(in,4)));
    m.dbias=real(static_cast<std::uint32_t>(get(in,4)));
    if(get(in,8)!=fingerprint(m)) throw std::runtime_error("model fingerprint mismatch");
    for(float f:m.encoder) if(!std::isfinite(f)) throw std::runtime_error("invalid model weight");
    for(float f:m.decoder) if(!std::isfinite(f)) throw std::runtime_error("invalid model weight");
    for(float f:m.ebias) if(!std::isfinite(f)) throw std::runtime_error("invalid model bias");
    if(!std::isfinite(m.dbias)) throw std::runtime_error("invalid decoder bias");
    if(in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing model data");
    return m;
}
std::uint32_t crc32(const std::uint8_t* a,std::size_t n) {
    std::uint32_t c=0xFFFFFFFFU;
    for(std::size_t i=0;i<n;++i) {
        c^=a[i];for(int j=0;j<8;++j) c=(c>>1)^(0xEDB88320U&static_cast<std::uint32_t>(-(c&1U)));
    }
    return ~c;
}
// 0,len denotes a zero run; 1,len,bytes denotes a nonzero literal run.
std::vector<std::uint8_t> rle_encode(const std::vector<std::uint8_t>& src) {
    std::vector<std::uint8_t> out;std::size_t i=0;
    while(i<src.size()) {
        const bool zero=src[i]==0;std::size_t n=1;
        while(i+n<src.size()&&n<255&&(src[i+n]==0)==zero) ++n;
        out.push_back(zero?0:1);out.push_back(static_cast<std::uint8_t>(n));
        if(!zero) out.insert(out.end(),src.begin()+i,src.begin()+i+n);
        i+=n;
    }
    return out;
}
std::vector<std::uint8_t> rle_decode(const std::vector<std::uint8_t>& src,std::size_t expected) {
    std::vector<std::uint8_t> out;std::size_t i=0;
    while(i<src.size()) {
        if(i+2>src.size()) throw std::runtime_error("truncated RLE");
        const int kind=src[i++],n=src[i++];
        if((kind!=0&&kind!=1)||n==0||out.size()+n>expected) throw std::runtime_error("invalid RLE run");
        if(kind==0) out.insert(out.end(),n,0);
        else {
          if(i+n>src.size()) throw std::runtime_error("invalid RLE literal");
          out.insert(out.end(),src.begin()+i,src.begin()+i+n);i+=n;
        }
    }
    if(out.size()!=expected) throw std::runtime_error("RLE length mismatch");
    return out;
}
enum class Mode:std::uint32_t { LOSSLESS=1,LOSSY=2 };
struct Record {
    std::uint32_t index=0;std::uint16_t valid=0;
    Code q{};std::uint8_t codec=2;std::vector<std::uint8_t> data;
    std::uint32_t checksum=0;
};
bool is_zero(const Tile& a,int valid) {
    for(int i=0;i<valid;++i) if(a[i]!=0) return false;
    return true;
}
Record make_record(const Model& m,const Tile& original,std::uint32_t index,int valid,Mode mode) {
    if(valid<1||valid>VOX) throw std::runtime_error("invalid tile length");
    Record r;r.index=index;r.valid=static_cast<std::uint16_t>(valid);
    r.q=quantize(encode(m,original));
    const Tile predicted=predict(m,r.q);
    if(mode==Mode::LOSSLESS) {
        std::vector<std::uint8_t> residual(valid);
        for(int i=0;i<valid;++i) residual[i]=original[i]^predicted[i];
        auto packed=rle_encode(residual);
        if(packed.size()<residual.size()) {r.codec=1;r.data=std::move(packed);}
        else {r.codec=0;r.data=std::move(residual);}
        r.checksum=crc32(original.data(),valid);
    } else r.checksum=crc32(predicted.data(),valid);
    return r;
}
Tile restore(const Model& m,const Record& r,Mode mode) {
    if(r.valid==0||r.valid>VOX) throw std::runtime_error("invalid tile valid length");
    Tile decoded=predict(m,r.q);
    if(mode==Mode::LOSSLESS) {
        std::vector<std::uint8_t> residual;
        if(r.codec==0&&r.data.size()==r.valid) residual=r.data;
        else if(r.codec==1) residual=rle_decode(r.data,r.valid);
        else throw std::runtime_error("invalid lossless residual codec");
        for(int i=0;i<r.valid;++i) decoded[i]^=residual[i];
    } else if(r.codec!=2||!r.data.empty()) throw std::runtime_error("invalid lossy record");
    if(crc32(decoded.data(),r.valid)!=r.checksum) throw std::runtime_error("tile CRC mismatch");
    return decoded;
}
void write_record(std::ostream& out,const Record& r) {
    if(r.data.size()>65535U) throw std::runtime_error("record payload too large");
    put(out,r.index,4);put(out,r.valid,2);
    out.write(reinterpret_cast<const char*>(r.q.data()),LATENT);
    put(out,r.codec,1);put(out,r.data.size(),2);put(out,r.checksum,4);
    if(!r.data.empty()) out.write(reinterpret_cast<const char*>(r.data.data()),static_cast<std::streamsize>(r.data.size()));
    if(!out) throw std::runtime_error("record write failed");
}
Record read_record(std::istream& in) {
    Record r;r.index=static_cast<std::uint32_t>(get(in,4));r.valid=static_cast<std::uint16_t>(get(in,2));
    in.read(reinterpret_cast<char*>(r.q.data()),LATENT);
    if(!in) throw std::runtime_error("truncated code");
    r.codec=static_cast<std::uint8_t>(get(in,1));
    const auto n=static_cast<std::size_t>(get(in,2));r.checksum=static_cast<std::uint32_t>(get(in,4));
    if(n>VOX+VOX/2) throw std::runtime_error("oversized record payload");
    r.data.resize(n);if(n) in.read(reinterpret_cast<char*>(r.data.data()),static_cast<std::streamsize>(n));
    if(!in) throw std::runtime_error("truncated record payload");
    return r;
}
struct Header {
    Mode mode=Mode::LOSSLESS;std::uint64_t count=0,fingerprint=0;
    std::uint32_t records=0;
};
void write_header(std::ostream& o,const Header& h) {
    o.write(CODE_MAGIC.data(),8);put(o,1,4);put(o,DIM,4);put(o,SIDE,4);
    put(o,static_cast<std::uint32_t>(h.mode),4);put(o,h.count,8);
    put(o,h.fingerprint,8);put(o,h.records,4);
}
Header read_header(std::istream& in,const Model& m) {
    char mag[8];in.read(mag,8);
    if(!in||!std::equal(mag,mag+8,CODE_MAGIC.begin())) throw std::runtime_error("bad bitstream magic");
    if(get(in,4)!=1||get(in,4)!=DIM||get(in,4)!=SIDE) throw std::runtime_error("wrong bitstream dimensions/version");
    const auto mm=get(in,4);
    if(mm!=1&&mm!=2) throw std::runtime_error("unsupported mode");
    Header h;h.mode=static_cast<Mode>(mm);h.count=get(in,8);h.fingerprint=get(in,8);
    h.records=static_cast<std::uint32_t>(get(in,4));
    if(h.count>LOGICAL_BYTES||h.fingerprint!=fingerprint(m)) throw std::runtime_error("bad size or model mismatch");
    const auto tiles=(h.count+VOX-1)/VOX;
    if(h.records>tiles) throw std::runtime_error("too many tile records");
    return h;
}
class AtomicFile {
    std::string outname,tempname;bool finished=false;
public:
    std::fstream out;
    explicit AtomicFile(const std::string& name):outname(name),tempname(name+".partial") {
        if(std::filesystem::exists(outname)||std::filesystem::exists(tempname))
          throw std::runtime_error("output/partial already exists");
        out.open(tempname,std::ios::binary|std::ios::in|std::ios::out|std::ios::trunc);
        if(!out) throw std::runtime_error("cannot create output");
    }
    void commit() {
        out.flush();if(!out) throw std::runtime_error("flush failed");
        out.close();if(!out) throw std::runtime_error("close failed");
        std::filesystem::rename(tempname,outname);finished=true;
    }
    ~AtomicFile() {if(!finished){out.close();std::error_code ec;std::filesystem::remove(tempname,ec);}}
};
void patch_records(std::ostream& o,std::uint32_t n) {
    o.seekp(40,std::ios::beg);if(!o) throw std::runtime_error("seek failed");
    put(o,n,4);o.seekp(0,std::ios::end);
}
void pack(const Model& m,const std::string& src,const std::string& dst,Mode mode) {
    const auto n=std::filesystem::file_size(src);
    if(n>LOGICAL_BYTES) throw std::runtime_error("max 1000^3 byte voxels");
    std::ifstream in(src,std::ios::binary);if(!in) throw std::runtime_error("cannot open source");
    AtomicFile out(dst);Header h;h.mode=mode;h.count=n;h.fingerprint=fingerprint(m);write_header(out.out,h);
    const std::uint32_t tiles=static_cast<std::uint32_t>((n+VOX-1)/VOX);
    for(std::uint32_t id=0;id<tiles;++id) {
        Tile a{};const auto len=static_cast<int>(std::min<std::uint64_t>(VOX,n-std::uint64_t(id)*VOX));
        in.read(reinterpret_cast<char*>(a.data()),len);
        if(in.gcount()!=len) throw std::runtime_error("source truncated during encoding");
        if(is_zero(a,len)) continue;
        write_record(out.out,make_record(m,a,id,len,mode));++h.records;
    }
    patch_records(out.out,h.records);out.commit();
    std::cout<<"pack: input="<<n<<" bytes, tiles="<<tiles<<", nonzero records="<<h.records<<"\n";
}
void unpack(const Model& m,const std::string& src,const std::string& dst) {
    std::ifstream in(src,std::ios::binary);if(!in) throw std::runtime_error("cannot open packed source");
    const Header h=read_header(in,m);AtomicFile out(dst);
    std::uint32_t seen=0;Record pending{};bool have=false;
    const std::uint32_t tiles=static_cast<std::uint32_t>((h.count+VOX-1)/VOX);
    const Tile zero{};
    for(std::uint32_t id=0;id<tiles;++id) {
        if(!have&&seen<h.records) {pending=read_record(in);++seen;have=true;}
        const auto valid=static_cast<int>(std::min<std::uint64_t>(VOX,h.count-std::uint64_t(id)*VOX));
        if(have&&pending.index<id) throw std::runtime_error("unsorted record indexes");
        if(have&&pending.index==id) {
            if(pending.valid!=valid) throw std::runtime_error("tile length mismatch");
            const Tile decoded=restore(m,pending,h.mode);
            out.out.write(reinterpret_cast<const char*>(decoded.data()),valid);have=false;
        } else out.out.write(reinterpret_cast<const char*>(zero.data()),valid);
        if(!out.out) throw std::runtime_error("decoded write failed");
    }
    if(have||seen!=h.records||in.peek()!=std::char_traits<char>::eof())
      throw std::runtime_error("extra or out-of-range record bytes");
    out.commit();std::cout<<"unpack: output="<<h.count<<" bytes, records="<<h.records<<"\n";
}
std::uint8_t probe(const Model& m,const std::string& src,int x,int y,int z) {
    std::ifstream in(src,std::ios::binary);if(!in) throw std::runtime_error("cannot open packed source");
    const Header h=read_header(in,m);const auto target=static_cast<std::uint32_t>(tile_id(x,y,z));
    const auto tiles=(h.count+VOX-1)/VOX;
    if(target>=tiles) throw std::runtime_error("coordinate beyond raw volume");
    std::uint32_t last=0;
    for(std::uint32_t i=0;i<h.records;++i) {
        Record r=read_record(in);
        if(r.index>=tiles||(i>0&&r.index<=last)) throw std::runtime_error("unsorted/out-of-range record");
        last=r.index;
        if(r.index==target) {
            const auto globalbyte=std::uint64_t(target)*VOX+local_id(x,y,z);
            if(globalbyte>=h.count) throw std::runtime_error("voxel is beyond raw length");
            return restore(m,r,h.mode)[local_id(x,y,z)];
        }
        if(r.index>target) break;
    }
    return 0;
}
void sparse_demo(const Model& m,const std::string& dst) {
    AtomicFile out(dst);Header h;h.count=LOGICAL_BYTES;h.mode=Mode::LOSSLESS;
    h.fingerprint=fingerprint(m);h.records=1;write_header(out.out,h);
    Tile a{};for(int i=0;i<VOX;++i) a[i]=static_cast<std::uint8_t>((7*i+19)%256);
    write_record(out.out,make_record(m,a,MAX_TILES-1,VOX,h.mode));out.commit();
    std::cout<<"sparse-demo: logical volume=1,000,000,000 voxels; stored tiles=1\n";
}
void selftest() {
    static_assert(GRID*GRID*GRID==MAX_TILES);
    if(tile_id(999,999,999)!=999999||local_id(999,999,999)!=999)
      throw std::runtime_error("3D indexing failed");
    Model m=init_model();
    const Tile a=synthetic(13);const float before=mse(m,a);
    for(int i=0;i<15;++i) train_step(m,a,0.12f);
    if(!(mse(m,a)<before)) throw std::runtime_error("gradient training failed to reduce MSE");
    Model plus=m,minus=m,updated=m;
    plus.encoder[0]+=0.005f;minus.encoder[0]-=0.005f;
    const float numerical=(mse(plus,a)-mse(minus,a))/0.01f;
    train_step(updated,a,0.02f);
    const float analytic=(m.encoder[0]-updated.encoder[0])/0.02f;
    if(!std::isfinite(numerical)||std::abs(numerical-analytic)>0.002f+0.1f*std::abs(numerical))
      throw std::runtime_error("analytical gradient does not match finite difference");
    for(int t=0;t<64;++t) {
        Tile original=synthetic(1000+t);
        if(t==1) for(int i=0;i<VOX;++i) original[i]=static_cast<std::uint8_t>(i*73+9);
        if(t==2) original.fill(0);
        const Record r=make_record(m,original,t,VOX,Mode::LOSSLESS);
        if(restore(m,r,Mode::LOSSLESS)!=original) throw std::runtime_error("lossless tile inverse failed");
        const Record q=make_record(m,original,t,VOX,Mode::LOSSY);
        (void)restore(m,q,Mode::LOSSY);
    }
    std::vector<std::uint8_t> v(1000,0);for(int i=600;i<700;++i) v[i]=static_cast<std::uint8_t>(i);
    if(rle_decode(rle_encode(v),v.size())!=v) throw std::runtime_error("RLE failed");
    std::cout<<"SELFTEST PASS: coordinates, training, 64 lossless and 64 lossy tiles, RLE\n";
    std::cout<<"training MSE "<<before<<" -> "<<mse(m,a)
             <<"; gradient analytic="<<analytic<<" finite difference="<<numerical<<"\n";
}
int argument(const char* s,int min,int max) {
    const std::string v(s);if(v.empty()||v.find_first_not_of("0123456789")!=std::string::npos)
      throw std::runtime_error("expected positive integer argument");
    const auto n=std::stoull(v);if(n<static_cast<unsigned>(min)||n>static_cast<unsigned>(max))
      throw std::runtime_error("integer outside permitted range");
    return static_cast<int>(n);
}
void evaluate(const Model& m,int count) {
    double continuous=0,quantized=0;
    for(int j=0;j<count;++j) {
        const Tile a=synthetic(557711U+static_cast<std::uint32_t>(j)*101U);
        continuous+=mse(m,a);
        const Tile b=predict(m,quantize(encode(m,a)));
        for(int i=0;i<VOX;++i) {
            const double d=(static_cast<int>(b[i])-static_cast<int>(a[i]))/255.0;
            quantized+=d*d;
        }
    }
    continuous/=count;quantized/=(static_cast<double>(count)*VOX);
    const double psnr=quantized>0?10*std::log10(1.0/quantized):std::numeric_limits<double>::infinity();
    std::cout<<std::setprecision(9)<<"validation tiles="<<count<<" continuous_MSE="<<continuous
             <<" quantized_MSE="<<quantized<<" quantized_PSNR_dB="<<psnr<<"\n";
}
void train(const std::string& dest,int steps) {
    Model m=init_model();
    float running=0;
    for(int i=0;i<steps;++i) {
        Tile tile=synthetic(static_cast<std::uint32_t>(20261104U+static_cast<std::uint32_t>(i)*37U));
        running+=train_step(m,tile,0.2f);
        if((i+1)%100==0) {std::cout<<"step "<<i+1<<" avg MSE="<<running/100<<"\n";running=0;}
    }
    save_model(m,dest);
    std::cout<<"saved model; weights="<<CHANNELS*KERNEL*2+CHANNELS+1
             <<"; latent bytes/tile="<<LATENT<<"; fingerprint="<<std::hex<<fingerprint(m)<<std::dec<<"\n";
}

// =====================================================================
// JARVIS-X PERMEATE v1.1 — direct-face Chebyshev wavefront, real JXA records.
// For the dossier's 100^3 tile lattice, NOT 10^15 physical tiles.
// The source is deterministic synthetic data; zero_mod deliberately injects
// zero tiles to exercise elision. This mode is a codec test, not real-data I/O.
// =====================================================================
struct PermeateStats {
    std::uint64_t shells_touched=0; // actually counts distinct tile visits
    std::uint64_t materialized_tiles=0;
    std::uint64_t bytes_written=0;  // actual bytes including JXA header
    std::uint64_t macs=0;           // analytical multiply-accumulate count
    std::uint64_t elided_zero_tiles=0;
    double elapsed_s=0.0;
};

inline std::uint64_t pack_coord(std::uint32_t tx,std::uint32_t ty,std::uint32_t tz) {
    if(tx>=GRID||ty>=GRID||tz>=GRID) throw std::runtime_error("pack_coord out of range");
    static_assert(GRID <= (1<<17));
    return (std::uint64_t(tz)<<34) | (std::uint64_t(ty)<<17) | tx;
}

// Emit the six cube faces as three disjoint pairs; clipping does not change
// shell membership. Each cell is visited once, with O(number of face cells).
template<class Fn>
void for_each_shell_tile(std::uint32_t cx,std::uint32_t cy,std::uint32_t cz,
                         std::uint32_t r,Fn&& fn) {
    if(cx>=GRID||cy>=GRID||cz>=GRID||r>=GRID)
        throw std::runtime_error("invalid shell centre/radius");
    if(r==0) {fn(cx,cy,cz);return;}
    const std::int64_t x0=std::int64_t(cx)-r, x1=std::int64_t(cx)+r;
    const std::int64_t y0=std::int64_t(cy)-r, y1=std::int64_t(cy)+r;
    const std::int64_t z0=std::int64_t(cz)-r, z1=std::int64_t(cz)+r;
    auto inside=[](std::int64_t v){return v>=0&&v<GRID;};
    auto iterate=[](std::int64_t lo,std::int64_t hi,auto&& act) {
        for(std::int64_t k=std::max<std::int64_t>(lo,0);
            k<=std::min<std::int64_t>(hi,GRID-1);++k) act(k);
    };
    auto emit=[&](std::int64_t x,std::int64_t y,std::int64_t z) {
        fn(std::uint32_t(x),std::uint32_t(y),std::uint32_t(z));
    };
    for(const auto z : {z0,z1}) if(inside(z))
        iterate(y0,y1,[&](auto y){iterate(x0,x1,[&](auto x){emit(x,y,z);});});
    for(const auto y : {y0,y1}) if(inside(y))
        iterate(z0+1,z1-1,[&](auto z){iterate(x0,x1,[&](auto x){emit(x,y,z);});});
    for(const auto x : {x0,x1}) if(inside(x))
        iterate(z0+1,z1-1,[&](auto z){iterate(y0+1,y1-1,[&](auto y){emit(x,y,z);});});
}

inline std::uint64_t cube_cardinality(std::uint32_t cx,std::uint32_t cy,
                                      std::uint32_t cz,std::int64_t r) {
    if(r<0) return 0;
    auto extent=[&](std::int64_t c){
        return std::uint64_t(std::min<std::int64_t>(GRID-1,c+r)
                              -std::max<std::int64_t>(0,c-r)+1);
    };
    return extent(cx)*extent(cy)*extent(cz);
}
inline std::uint64_t shell_cardinality(std::uint32_t cx,std::uint32_t cy,
                                       std::uint32_t cz,std::uint32_t r) {
    if(cx>=GRID||cy>=GRID||cz>=GRID||r>=GRID)
        throw std::runtime_error("invalid shell centre/radius");
    return cube_cardinality(cx,cy,cz,r)-
           cube_cardinality(cx,cy,cz,std::int64_t(r)-1);
}

struct SparseStore {
    std::map<std::uint32_t,Record> entries;
    const Record& upsert(std::uint32_t id,Record r) {
        if(id>=MAX_TILES || r.index!=id) throw std::runtime_error("bad record tile id");
        return entries.insert_or_assign(id,std::move(r)).first->second;
    }
};

inline std::uint32_t wavefront_seed(std::uint32_t id) {
    const std::uint64_t mixed=(std::uint64_t(id)*0x9E3779B97F4A7C15ULL)>>32;
    return std::uint32_t(mixed)^id;
}

PermeateStats permeate(const Model& m,SparseStore& store,
                        std::uint32_t cx,std::uint32_t cy,std::uint32_t cz,
                        std::uint32_t radius,const std::string& output,
                        bool lossless=true,bool verbose=true,
                        std::uint32_t zero_mod=0) {
    using clock=std::chrono::steady_clock;
    auto t0=clock::now();
    if(cx>=GRID||cy>=GRID||cz>=GRID||radius>=GRID)
        throw std::runtime_error("invalid permeate centre/radius");
    if(!store.entries.empty())
        throw std::runtime_error("store must be empty to create new stream");
    const Mode mode=lossless?Mode::LOSSLESS:Mode::LOSSY;
    constexpr std::uint64_t enc_macs=std::uint64_t(LATENT)*KERNEL;
    constexpr std::uint64_t dec_macs=std::uint64_t(VOX)*CHANNELS*KERNEL;
    // make_record encodes+decodes; restore decodes again to verify CRC/identity.
    constexpr std::uint64_t verified_macs=enc_macs+2*dec_macs;
    PermeateStats st;
    for(std::uint32_t r=0;r<=radius;++r) {
        std::uint64_t current=0;
        for_each_shell_tile(cx,cy,cz,r,[&](std::uint32_t tx,std::uint32_t ty,std::uint32_t tz){
            ++current;
            ++st.shells_touched;
            const std::uint32_t id=std::uint32_t(tile_id(int(tx)*SIDE,int(ty)*SIDE,int(tz)*SIDE));
            Tile x=synthetic(wavefront_seed(id));
            if(zero_mod!=0 && id%zero_mod==0) x.fill(0);
            if(is_zero(x,VOX)) {++st.elided_zero_tiles;return;}
            Record rec=make_record(m,x,id,VOX,mode);
            const Record& stored=store.upsert(id,std::move(rec));
            const Tile recovered=restore(m,stored,mode);
            if(lossless && recovered!=x)
                throw std::runtime_error("permeate byte identity failure: tile " + std::to_string(id));
            ++st.materialized_tiles;
            st.macs+=verified_macs;
        });
        if(current!=shell_cardinality(cx,cy,cz,r))
            throw std::runtime_error("shell cardinality mismatch at r="+std::to_string(r));
        if(verbose&&(r%10==0||r==radius))
            std::cout<<"  [r="<<r<<"] shell="<<current
                     <<" visited="<<st.shells_touched
                     <<" records="<<st.materialized_tiles<<" MACs="<<st.macs<<"\n";
    }
    if(store.entries.size()>MAX_TILES)
        throw std::runtime_error("more records than physical tile address space");
    Header h;
    h.mode=mode;
    h.count=LOGICAL_BYTES;
    h.fingerprint=fingerprint(m);
    h.records=std::uint32_t(store.entries.size());
    AtomicFile out(output);
    write_header(out.out,h);
    for(const auto& [id,rec]:store.entries) {
        if(rec.index!=id) throw std::runtime_error("record index mismatch");
        write_record(out.out,rec);
    }
    out.commit();
    st.bytes_written=std::filesystem::file_size(output);
    st.elapsed_s=std::chrono::duration<double>(clock::now()-t0).count();
    return st;
}
void print_permeate_stats(const PermeateStats& st) {
    std::cout<<"[PERMEATE] visited="<<st.shells_touched
             <<" records="<<st.materialized_tiles
             <<" zero-elided="<<st.elided_zero_tiles
             <<" bytes="<<st.bytes_written
             <<" analytical_MACs="<<st.macs
             <<" elapsed="<<std::fixed<<std::setprecision(3)<<st.elapsed_s<<"s\n";
}
void verify_permeate_file(const Model& m,const std::string& file) {
    std::ifstream in(file,std::ios::binary);
    if(!in) throw std::runtime_error("cannot open PERMEATE output");
    const Header h=read_header(in,m);
    if(h.count!=LOGICAL_BYTES) throw std::runtime_error("wrong logical extent");
    std::uint32_t prior=0;
    for(std::uint32_t i=0;i<h.records;++i) {
        const Record r=read_record(in);
        if(r.index>=MAX_TILES || (i>0 && r.index<=prior) || r.valid!=VOX)
            throw std::runtime_error("bad record order/index/length");
        (void) restore(m,r,h.mode); // tests CRC32 on each decoded tile
        prior=r.index;
    }
    if(in.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("trailing bytes in PERMEATE output");
    std::cout<<"PERMEATE VERIFY PASS: mode="
             <<(h.mode==Mode::LOSSLESS?"lossless":"lossy")
             <<" records="<<h.records
             <<" bytes="<<std::filesystem::file_size(file)
             <<" logical_voxels="<<h.count<<"\n";
}

void permeate_selftest(const Model& m,const std::string& output) {
    std::vector<std::uint8_t> seen(MAX_TILES,0);
    std::uint64_t count=0;
    for(std::uint32_t r=0;r<GRID;++r) {
        std::uint64_t local=0;
        for_each_shell_tile(0,0,0,r,[&](auto x,auto y,auto z) {
            const auto id=std::uint32_t(tile_id(int(x)*SIDE,int(y)*SIDE,int(z)*SIDE));
            if(seen[id]++) throw std::runtime_error("duplicate tile across shells");
            ++local;++count;
        });
        if(local!=shell_cardinality(0,0,0,r)) throw std::runtime_error("wrong shell size");
    }
    if(count!=MAX_TILES) throw std::runtime_error("shells do not cover all tiles");
    if(pack_coord(0,0,0)==pack_coord(1,0,0)||
       pack_coord(1,0,0)==pack_coord(0,1,0)) throw std::runtime_error("coord key collision");
    SparseStore store;
    const auto stats=permeate(m,store,0,0,0,2,output,true,false,3);
    if(stats.shells_touched!=27 || stats.elided_zero_tiles!=9 ||
       stats.materialized_tiles!=18 || stats.bytes_written!=std::filesystem::file_size(output))
        throw std::runtime_error("permeate sparse statistic failure");
    verify_permeate_file(m,output);
    const Tile expected=synthetic(wavefront_seed(1));
    if(probe(m,output,10,0,0)!=expected[0] || probe(m,output,0,0,0)!=0 ||
       probe(m,output,990,990,990)!=0)
        throw std::runtime_error("permeate serialized probe mismatch");
    std::cout<<"PERMEATE SELFTEST PASS: 1,000,000 distinct shell visits, "
             <<"27 near-origin tiles, 9 zero-elided, 18 verified records\n";
    print_permeate_stats(stats);
}

} // namespace jx

int main(int argc,char** argv) {
  try {
    const std::string cmd=argc>1?argv[1]:"";
    if(cmd=="permeate-verify"&&argc==4) {
        jx::verify_permeate_file(jx::load_model(argv[2]),argv[3]);return 0;
    }
    if(cmd=="permeate-selftest"&&argc==4) {
        jx::permeate_selftest(jx::load_model(argv[2]),argv[3]);return 0;
    }
    if(cmd=="permeate"&&(argc==9||argc==10)) {
        const auto m=jx::load_model(argv[2]);
        const auto cx=jx::argument(argv[4],0,jx::GRID-1);
        const auto cy=jx::argument(argv[5],0,jx::GRID-1);
        const auto cz=jx::argument(argv[6],0,jx::GRID-1);
        const auto r=jx::argument(argv[7],0,jx::GRID-1);
        const std::string mode=argv[8];
        if(mode!="lossless"&&mode!="lossy") throw std::runtime_error("invalid mode");
        const auto zero_mod=argc==10?jx::argument(argv[9],1,1000000000):0;
        jx::SparseStore store;
        const auto st=jx::permeate(m,store,cx,cy,cz,r,argv[3],mode=="lossless",true,zero_mod);
        jx::print_permeate_stats(st);return 0;
    }
    if(cmd=="selftest"&&argc==2) {jx::selftest();return 0;}
    if(cmd=="train"&&(argc==3||argc==4)) {
        jx::train(argv[2],argc==4?jx::argument(argv[3],1,1000000):500);return 0;
    }
    if(cmd=="evaluate"&&(argc==3||argc==4)) {
        jx::evaluate(jx::load_model(argv[2]),argc==4?jx::argument(argv[3],1,100000):100);return 0;
    }
    if(cmd=="pack"&&argc==6) {
        const auto mode=std::string(argv[5])=="lossless"?jx::Mode::LOSSLESS:jx::Mode::LOSSY;
        if(std::string(argv[5])!="lossless"&&std::string(argv[5])!="lossy")
          throw std::runtime_error("mode must be lossless or lossy");
        jx::pack(jx::load_model(argv[2]),argv[3],argv[4],mode);return 0;
    }
    if(cmd=="unpack"&&argc==5) {jx::unpack(jx::load_model(argv[2]),argv[3],argv[4]);return 0;}
    if(cmd=="sparse-demo"&&argc==4) {jx::sparse_demo(jx::load_model(argv[2]),argv[3]);return 0;}
    if(cmd=="probe"&&argc==7) {
        const auto m=jx::load_model(argv[2]);
        const int x=jx::argument(argv[4],0,999),y=jx::argument(argv[5],0,999),z=jx::argument(argv[6],0,999);
        std::cout<<"voxel["<<x<<","<<y<<","<<z<<"]="<<int(jx::probe(m,argv[3],x,y,z))<<"\n";return 0;
    }
    std::cerr<<"usage:\n"
        <<" jarvisx selftest\n jarvisx train MODEL [steps]\n"
        <<" jarvisx permeate-selftest MODEL OUT.jxa\n"
        <<" jarvisx permeate-verify MODEL STREAM.jxa\n"
        <<" jarvisx permeate MODEL OUT.jxa CX CY CZ R lossless|lossy [ZERO_MOD]\n"
        <<" jarvisx evaluate MODEL [tiles]\n"
        <<" jarvisx pack MODEL RAW PACKED lossless|lossy\n"
        <<" jarvisx unpack MODEL PACKED RAW\n"
        <<" jarvisx sparse-demo MODEL PACKED\n"
        <<" jarvisx probe MODEL PACKED X Y Z\n";
    return 2;
  } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
