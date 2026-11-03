// Dr Moagi Mandelbulb 3D Recursive ANN Multimodal Quine
// C++20 standalone demonstration. Standard library only.
// Build: g++ -O2 -std=c++20 -Wall -Wextra -pedantic dm_mandelquine.cpp -o dm_mandelquine
// Quine: ./dm_mandelquine quine > replica.cpp  (cmp dm_mandelquine.cpp replica.cpp)
// NOT a trained LLM or production multimedia codec; lossless roundtrip uses residual side-data.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dm {
using Byte = std::uint8_t;
using Bytes = std::vector<Byte>;
namespace fs = std::filesystem;
constexpr float PI = 3.14159265358979323846f;
constexpr std::size_t LIMIT = 8u * 1024u * 1024u;
constexpr int H = 12, Z = 4, FI = 6, DI = Z + 5;

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
Byte quantize(float x) { return Byte(std::clamp(int(std::lround((std::clamp(x,-1.f,1.f)+1.f)*127.5f)),0,255)); }
float norm(Byte b) { return (float(b)/127.5f)-1.f; }

Bytes read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    require(bool(f), "Cannot open: " + p.string());
    auto n = f.tellg(); require(n >= 0 && std::uint64_t(n) <= LIMIT + 65536, "Input too large (limit 8 MiB payload)");
    f.seekg(0); Bytes b(static_cast<std::size_t>(n));
    if (!b.empty()) f.read(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
    require(bool(f) || b.empty(), "Read failed"); return b;
}
void write_file(const fs::path& p, const Bytes& b) {
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc); require(bool(f), "Cannot write: " + p.string());
    if (!b.empty()) f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    require(bool(f), "Write failed: " + p.string());
}
void u32(std::ostream& o, std::uint32_t x) { for (int i=0;i<4;i++) o.put(char((x>>(8*i))&255u)); }
void u64(std::ostream& o, std::uint64_t x) { for (int i=0;i<8;i++) o.put(char((x>>(8*i))&255u)); }
std::uint32_t r32(std::istream& s) { std::uint32_t n=0; for(int i=0;i<4;i++){int c=s.get();require(c!=EOF,"Truncated archive");n|=std::uint32_t(c)<<(8*i);} return n; }
std::uint64_t r64(std::istream& s) { std::uint64_t n=0; for(int i=0;i<8;i++){int c=s.get();require(c!=EOF,"Truncated archive");n|=std::uint64_t(c)<<(8*i);} return n; }
void f32(std::ostream& s, float x) { static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559); u32(s,std::bit_cast<std::uint32_t>(x)); }
float rf32(std::istream& s) { return std::bit_cast<float>(r32(s)); }
void putbytes(std::ostream& s, const Bytes& v) { if(!v.empty())s.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size())); }
Bytes getbytes(std::istream& s,std::size_t n) { require(n<=LIMIT+65536,"Invalid field length"); Bytes b(n);if(n)s.read(reinterpret_cast<char*>(b.data()),std::streamsize(n));require(bool(s)||n==0,"Truncated archive data");return b; }
std::uint32_t le32(const Bytes& d,std::size_t i) {require(i+4<=d.size(),"Bad WAV header");return std::uint32_t(d[i])|(std::uint32_t(d[i+1])<<8)|(std::uint32_t(d[i+2])<<16)|(std::uint32_t(d[i+3])<<24);}
std::uint16_t le16(const Bytes& d,std::size_t i) {require(i+2<=d.size(),"Bad WAV header");return std::uint16_t(d[i])|(std::uint16_t(d[i+1])<<8);}

struct Media {
    std::string kind;
    Bytes prefix, data, suffix;
    std::uint32_t width=0,height=0,channels=0,sample_rate=0;
};
Media parse_media(const std::string& kind,const Bytes& input) {
    Media m; m.kind=kind;
    if(kind=="text"||kind=="bytes") {m.data=input;return m;}
    if(kind=="image") {
        // Strict P6 PPM (RGB 8-bit) with ASCII header and optional comments.
        require(input.size()>12 && input[0]=='P' && input[1]=='6', "Image must be P6 PPM");
        std::size_t p=0;
        auto token=[&](){
            while(p<input.size()) {
                if(input[p]=='#') {while(p<input.size() && input[p]!='\n') ++p;}
                else if(input[p]==' '||input[p]=='\n'||input[p]=='\r'||input[p]=='\t') ++p;
                else break;
            }
            std::string s;
            while(p<input.size() && input[p]!='#' && input[p]!=' ' && input[p]!='\n' && input[p]!='\r' && input[p]!='\t') s+=char(input[p++]);
            require(!s.empty() && s.size()<20,"Bad PPM header");return s;
        };
        require(token()=="P6","Not P6 PPM");
        m.width=std::stoul(token());m.height=std::stoul(token());
        require(token()=="255","Only 8-bit PPM supported");
        require(p<input.size() && (input[p]==' '||input[p]=='\n'||input[p]=='\r'||input[p]=='\t'),"Missing PPM separator");
        // PPM is specified to use one whitespace separator before binary pixels.
        if(input[p]=='\r' && p+1<input.size() && input[p+1]=='\n')p+=2;else ++p;
        require(m.width>0 && m.height>0 && std::uint64_t(m.width)*m.height*3<=LIMIT,"PPM dimensions invalid");
        require(input.size()==p+std::uint64_t(m.width)*m.height*3,"PPM raster size mismatch");
        m.channels=3; m.prefix.assign(input.begin(),input.begin()+std::ptrdiff_t(p));m.data.assign(input.begin()+std::ptrdiff_t(p),input.end());return m;
    }
    if(kind=="audio") {
        require(input.size()>=44 && std::memcmp(input.data(),"RIFF",4)==0 && std::memcmp(input.data()+8,"WAVE",4)==0,"Audio must be RIFF/WAVE");
        std::size_t pos=12,dataoff=0,ndata=0;bool fmt=false,gotdata=false;
        while(pos+8<=input.size()) {
            const auto chunk_size=le32(input,pos+4);
            const auto begin=pos+8;
            require(chunk_size<=input.size()-begin,"WAV chunk beyond file");
            const std::string name(reinterpret_cast<const char*>(input.data()+pos),4);
            if(name=="fmt ") {
                require(chunk_size>=16,"Invalid PCM format");
                require(le16(input,begin)==1 && le16(input,begin+14)==16,"Only PCM16 WAV supported");
                m.channels=le16(input,begin+2);m.sample_rate=le32(input,begin+4);
                require(m.channels>0 && m.channels<=8 && m.sample_rate>0,"Bad WAV channels/sample rate");fmt=true;
            } else if(name=="data" && !gotdata) {dataoff=begin;ndata=chunk_size;gotdata=true;}
            pos=begin+chunk_size+(chunk_size&1u);
        }
        require(fmt&&gotdata&&ndata>0&&ndata<=LIMIT&&ndata%(2*m.channels)==0,"Unsupported WAV payload");
        m.prefix.assign(input.begin(),input.begin()+std::ptrdiff_t(dataoff));
        m.data.assign(input.begin()+std::ptrdiff_t(dataoff),input.begin()+std::ptrdiff_t(dataoff+ndata));
        m.suffix.assign(input.begin()+std::ptrdiff_t(dataoff+ndata),input.end());return m;
    }
    throw std::runtime_error("Kind must be text/image/audio/bytes");
}
Bytes combine(const Media& m, const Bytes& data) {
    Bytes out;out.reserve(m.prefix.size()+data.size()+m.suffix.size());
    out.insert(out.end(),m.prefix.begin(),m.prefix.end());out.insert(out.end(),data.begin(),data.end());out.insert(out.end(),m.suffix.begin(),m.suffix.end());return out;
}
std::uint32_t cube_side(std::size_t n) {
    std::uint32_t s=1;while(std::uint64_t(s)*s*s<n)++s;return s;
}
struct Point { float x,y,z; };
Point position(std::size_t i,std::uint32_t s) {
    const float scale=s>1?2.f/float(s-1):0.f;
    auto x=i%s,y=(i/s)%s,z=i/(std::size_t(s)*s);
    return {float(x)*scale-1.f,float(y)*scale-1.f,float(z)*scale-1.f};
}
// Mandelbulb exponent-eight escape-time scalar, mapped to [-1,1].
float mandelbulb(const Point& pt) {
    float zx=0,zy=0,zz=0;int k=0;
    for(;k<7;k++) {
        const float r=std::sqrt(zx*zx+zy*zy+zz*zz);
        if(r>2.f)break;
        const float theta=std::acos(std::clamp(zz/std::max(r,1e-8f),-1.f,1.f))*8.f;
        const float phi=std::atan2(zy,zx)*8.f;
        const float rp=std::pow(r,8.f);
        const float st=std::sin(theta);
        zx=rp*st*std::cos(phi)+pt.x;
        zy=rp*st*std::sin(phi)+pt.y;
        zz=rp*std::cos(theta)+pt.z;
    }
    return k==7?-1.f:1.f-2.f*float(k)/7.f;
}
using Lat = std::array<float,Z>;
struct Cell {Point p{};float bulb=0,neigh=0;};
std::vector<Cell> cells_for(const Bytes& b,std::uint32_t s) {
    std::vector<Cell> c(b.size());const std::size_t ss=std::size_t(s)*s;
    for(std::size_t i=0;i<b.size();i++) {
        c[i].p=position(i,s);c[i].bulb=mandelbulb(c[i].p);
        const std::size_t x=i%s,y=(i/s)%s,z=i/ss;
        float sum=0;int count=0;
        auto a=[&](std::size_t j){if(j<b.size()){sum+=norm(b[j]);++count;}};
        if(x) a(i-1);
        if(x+1<s) a(i+1);
        if(y) a(i-s);
        if(y+1<s) a(i+s);
        if(z) a(i-ss);
        if(z+1<s) a(i+ss);
        c[i].neigh=count?sum/count:0.f;
    }
    return c;
}
std::array<float,FI> features(Byte b,const Cell& c) {
    return {norm(b),c.bulb,c.p.x,c.p.y,c.p.z,c.neigh};
}

struct Network {
    std::array<float,H*FI> ew{};std::array<float,H> eb{};
    std::array<float,Z*H> zw{};std::array<float,Z> zb{};
    std::array<float,H*DI> dw{};std::array<float,H> db{};
    std::array<float,H> ow{};float ob=0;
    explicit Network(bool initialize=true) {
        if(initialize){std::mt19937 rng(0xD0A613u);std::uniform_real_distribution<float> dist(-.3f,.3f);
            for(auto& x:ew) x=dist(rng);
            for(auto& x:zw) x=dist(rng);
            for(auto& x:dw) x=dist(rng);
            for(auto& x:ow) x=dist(rng);
        }
    }
    struct Cache {std::array<float,H> eh{},dh{};Lat z{};std::array<float,DI> din{};float y=0;};
    Cache forward(const std::array<float,FI>& x) const {
        Cache c;
        for(int j=0;j<H;j++){float v=eb[j];for(int k=0;k<FI;k++)v+=ew[j*FI+k]*x[k];c.eh[j]=std::tanh(v);}
        for(int k=0;k<Z;k++){float v=zb[k];for(int j=0;j<H;j++)v+=zw[k*H+j]*c.eh[j];c.z[k]=std::tanh(v);}
        for(int k=0;k<Z;k++)c.din[k]=c.z[k];
        for(int k=0;k<5;k++)c.din[Z+k]=x[k+1];
        for(int j=0;j<H;j++){float v=db[j];for(int k=0;k<DI;k++)v+=dw[j*DI+k]*c.din[k];c.dh[j]=std::tanh(v);}
        float v=ob;for(int j=0;j<H;j++)v+=ow[j]*c.dh[j];c.y=std::tanh(v);
        return c;
    }
    float decode(const Lat& z,const Cell& cell) const {
        std::array<float,DI> x{};
        for(int k=0;k<Z;k++)x[k]=z[k];
        x[Z]=cell.bulb;x[Z+1]=cell.p.x;x[Z+2]=cell.p.y;x[Z+3]=cell.p.z;x[Z+4]=cell.neigh;
        float o=ob;
        for(int j=0;j<H;j++){float a=db[j];for(int k=0;k<DI;k++)a+=dw[j*DI+k]*x[k];o+=ow[j]*std::tanh(a);}
        return std::tanh(o);
    }
    float train_one(const std::array<float,FI>& x,float lr) {
        const auto c=forward(x);const float err=c.y-x[0];
        const float dout=err*(1.f-c.y*c.y);
        std::array<float,H> dhgrad{},ehgrad{};Lat dzgrad{};
        for(int j=0;j<H;j++)dhgrad[j]=dout*ow[j]*(1.f-c.dh[j]*c.dh[j]);
        for(int k=0;k<Z;k++){
            float v=0;for(int j=0;j<H;j++)v+=dhgrad[j]*dw[j*DI+k];
            dzgrad[k]=v*(1.f-c.z[k]*c.z[k]);
        }
        for(int j=0;j<H;j++){
            float v=0;for(int k=0;k<Z;k++)v+=dzgrad[k]*zw[k*H+j];
            ehgrad[j]=v*(1.f-c.eh[j]*c.eh[j]);
        }
        for(int j=0;j<H;j++){ow[j]-=lr*dout*c.dh[j];for(int k=0;k<DI;k++)dw[j*DI+k]-=lr*dhgrad[j]*c.din[k];db[j]-=lr*dhgrad[j];}
        ob-=lr*dout;
        for(int k=0;k<Z;k++){for(int j=0;j<H;j++)zw[k*H+j]-=lr*dzgrad[k]*c.eh[j];zb[k]-=lr*dzgrad[k];}
        for(int j=0;j<H;j++){for(int k=0;k<FI;k++)ew[j*FI+k]-=lr*ehgrad[j]*x[k];eb[j]-=lr*ehgrad[j];}
        return err*err;
    }
    void serialize(std::ostream& s) const {
        auto wr=[&](const auto& a){for(float x:a)f32(s,x);};wr(ew);wr(eb);wr(zw);wr(zb);wr(dw);wr(db);wr(ow);f32(s,ob);
    }
    void deserialize(std::istream& s) {
        auto rd=[&](auto& a){for(float& x:a){x=rf32(s);require(std::isfinite(x),"Invalid model weights");}};
        rd(ew);rd(eb);rd(zw);rd(zb);rd(dw);rd(db);rd(ow);ob=rf32(s);require(std::isfinite(ob),"Invalid model bias");
    }
};

// Spatial feedback R for z^(t+1) = .68*z^t + .32*tanh(z^0 + .18*N(z^t) + .08*phi).
void recursive_refine(std::vector<Lat>& z,const std::vector<Cell>& c,std::uint32_t s,int rounds=4) {
    const auto base=z;std::vector<Lat> next(z.size());const std::size_t ss=std::size_t(s)*s;
    for(int t=0;t<rounds;t++) {
        for(std::size_t i=0;i<z.size();i++) {
            Lat avg{};int count=0;const std::size_t x=i%s,y=(i/s)%s,h=i/ss;
            auto a=[&](std::size_t j){if(j<z.size()){for(int k=0;k<Z;k++)avg[k]+=z[j][k];++count;}};
            if(x) a(i-1);
        if(x+1<s) a(i+1);
            if(y) a(i-s);
        if(y+1<s) a(i+s);
            if(h) a(i-ss);
            if(h+1<s) a(i+ss);
            for(int k=0;k<Z;k++){
                float v=base[i][k]+(count?.18f*avg[k]/count:0.f)+.08f*c[i].bulb;
                next[i][k]=.68f*z[i][k]+.32f*std::tanh(v);
            }
        }
        z.swap(next);
    }
}

struct Archive {
    Media media;std::uint32_t side=0,epochs=0;Network net{false};
    std::vector<Cell> geom;
    std::vector<Lat> latent;
    Bytes prediction,residual;
};
Archive encode(const Media& m,int epochs) {
    require(!m.data.empty()&&m.data.size()<=LIMIT,"Payload must contain 1..8 MiB");
    Archive a;a.media=m;a.side=cube_side(m.data.size());a.epochs=std::uint32_t(epochs);a.net=Network(true);
    a.geom=cells_for(m.data,a.side);
    std::mt19937 rng(0xABCD918u);
    std::uniform_int_distribution<std::size_t> pick(0,m.data.size()-1);
    const std::size_t steps=std::min<std::size_t>(6000,m.data.size()*2);
    float mse=0;
    for(int ep=0;ep<epochs;ep++){
        double loss=0;
        for(std::size_t j=0;j<steps;j++) {
            const auto i=pick(rng);loss+=a.net.train_one(features(m.data[i],a.geom[i]),.018f);
        }
        mse=float(loss/double(steps));
    }
    a.latent.resize(m.data.size());a.prediction.resize(m.data.size());a.residual.resize(m.data.size());
    for(std::size_t i=0;i<m.data.size();i++)a.latent[i]=a.net.forward(features(m.data[i],a.geom[i])).z;
    recursive_refine(a.latent,a.geom,a.side);
    for(std::size_t i=0;i<m.data.size();i++){
        a.prediction[i]=quantize(a.net.decode(a.latent[i],a.geom[i]));
        a.residual[i]=Byte(m.data[i]^a.prediction[i]);
    }
    std::cout<<"Trained epoch MSE (pre-refinement): "<<std::fixed<<std::setprecision(6)<<mse<<"\n";
    return a;
}
// Store explicit predicted byte plus exact XOR residual. No compression claims.
void save(const Archive& a,const fs::path& path) {
    if(path.has_parent_path())fs::create_directories(path.parent_path());
    std::ofstream s(path,std::ios::binary|std::ios::trunc);require(bool(s),"Cannot create archive");
    s.write("DMQ3D002",8);
    const auto& m=a.media;
    const auto kind=m.kind=="text"?1u:m.kind=="image"?2u:m.kind=="audio"?3u:4u;
    u32(s,kind);u32(s,a.side);u32(s,a.epochs);
    u32(s,m.width);u32(s,m.height);u32(s,m.channels);u32(s,m.sample_rate);
    u64(s,a.latent.size());u64(s,m.prefix.size());u64(s,m.suffix.size());
    putbytes(s,m.prefix);putbytes(s,m.suffix);a.net.serialize(s);
    for(const auto& z:a.latent)for(float v:z)f32(s,v);
    for(const auto& c:a.geom)f32(s,c.neigh);
    putbytes(s,a.prediction);putbytes(s,a.residual);require(bool(s),"Archive write failed");
}
Archive load(const fs::path& path) {
    std::ifstream s(path,std::ios::binary);require(bool(s),"Cannot read archive");
    char sig[8];s.read(sig,8);require(bool(s)&&std::memcmp(sig,"DMQ3D002",8)==0,"Wrong archive signature");
    Archive a;const auto kind=r32(s);require(kind>=1&&kind<=4,"Wrong modality");
    a.media.kind=kind==1?"text":kind==2?"image":kind==3?"audio":"bytes";
    a.side=r32(s);a.epochs=r32(s);
    a.media.width=r32(s);a.media.height=r32(s);a.media.channels=r32(s);a.media.sample_rate=r32(s);
    const auto n=r64(s),p=r64(s),q=r64(s);
    require(n>0&&n<=LIMIT&&p<=65536&&q<=65536&&a.side==cube_side(std::size_t(n)),"Invalid archive shape");
    a.media.prefix=getbytes(s,std::size_t(p));a.media.suffix=getbytes(s,std::size_t(q));a.net.deserialize(s);
    a.latent.resize(std::size_t(n));
    for(auto& z:a.latent)for(auto& v:z){v=rf32(s);require(std::isfinite(v)&&std::abs(v)<=2.f,"Corrupt latent field");}
    a.geom.resize(std::size_t(n));
    for(auto& c:a.geom){c.neigh=rf32(s);require(std::isfinite(c.neigh) && std::abs(c.neigh)<=1.001f,"Corrupt neighborhood feature");}
    a.prediction=getbytes(s,std::size_t(n));a.residual=getbytes(s,std::size_t(n));
    require(s.peek()==EOF,"Archive trailing data not expected");
    // Regenerate coordinates and fractal; preserve archived local-neighborhood features.
    for(std::size_t i=0;i<n;i++){a.geom[i].p=position(i,a.side);a.geom[i].bulb=mandelbulb(a.geom[i].p);}
    return a;
}
Bytes restore(const Archive& a) {
    Bytes b(a.residual.size());for(std::size_t i=0;i<b.size();i++)b[i]=Byte(a.prediction[i]^a.residual[i]);return b;
}
// Generative decoding is separate from the exact residual reconstruction path.
Bytes generate(const Archive& a,std::uint32_t seed,float strength) {
    require(strength>=0.f&&strength<=2.f,"Strength must be between 0 and 2");
    const auto n=a.latent.size();Bytes out(n);std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-1.f,1.f);
    for(std::size_t i=0;i<n;i++) {
        auto z=a.latent[i];const auto& c=a.geom[i];
        for(int k=0;k<Z;k++)z[k]=std::tanh(z[k]+strength*(.35f*c.bulb+.18f*jitter(rng)));
        const auto value=a.net.decode(z,c);
        out[i]=quantize(value);
    }
    if(a.media.kind=="image") {
        for(std::size_t i=0;i<n;i++){
            const float f=a.geom[i].bulb;
            const int rgb=int(i%3);
            const float wave=20.f*std::sin(.12f*float(i/3)+(rgb+1)*2.f+f*6.f);
            out[i]=Byte(std::clamp(int(out[i])+int(strength*(wave+35*f)),0,255));
        }
    } else if(a.media.kind=="audio") {
        // Stable 16-bit oscillator modulated by neural predictions; no random PCM-byte noise.
        require(a.media.sample_rate>0&&n%2==0,"Bad audio generation metadata");
        float phase=0;
        for(std::size_t i=0;i<n/2;i++){
            const float mod=float(out[2*i])/255.f;
            const float frequency=180.f+260.f*mod;
            phase+=2.f*PI*frequency/float(a.media.sample_rate);
            const float env=(.15f+.35f*mod)*(.8f+.2f*a.geom[2*i].bulb);
            const auto pcm=std::int16_t(std::lround(std::clamp(env*std::sin(phase),-1.f,1.f)*30000.f));
            const auto bits=std::uint16_t(pcm);
            out[2*i]=Byte(bits&255u);out[2*i+1]=Byte(bits>>8);
        }
    } else if(a.media.kind=="text") {
        // Procedural lexicon selection driven by ANN bytes: not linguistic understanding.
        constexpr std::array<const char*,20> words={"latent","geometry","recursive","field","signal","encode","decode","fractal","neural","phase","memory","vector","kernel","matrix","system","feedback","wave","form","pattern","state"};
        std::string sentence;
        for(std::size_t i=0;i<n;i+=8){
            sentence+=words[(std::size_t(out[i])+std::size_t(seed%20))%words.size()];
            sentence+=((i/8)%9==8)?".\n":" ";
        }
        out.assign(sentence.begin(),sentence.end());
    }
    return out;
}
void inspect(const Archive& a) {
    const auto n=a.latent.size();
    std::cout<<"mode="<<a.media.kind<<" shape="<<a.side<<"x"<<a.side<<"x"<<a.side
             <<" occupied="<<n<<" epochs="<<a.epochs<<"\n";
    if(a.media.kind=="image")std::cout<<"PPM="<<a.media.width<<"x"<<a.media.height<<" RGB\n";
    if(a.media.kind=="audio")std::cout<<"PCM16 rate="<<a.media.sample_rate<<" channels="<<a.media.channels<<"\n";
    double mse=0;std::size_t nonzero=0,predictor_mismatch=0;
    for(std::size_t i=0;i<n;i++){
        const int error=int(a.prediction[i])-int(Byte(a.prediction[i]^a.residual[i]));
        mse+=double(error)*error;nonzero+=a.residual[i]!=0;
        predictor_mismatch+=dm::quantize(a.net.decode(a.latent[i],a.geom[i]))!=a.prediction[i];
    }
    std::cout<<"ANN pre-residual RMSE="<<std::sqrt(mse/double(n))<<" bytes"
             <<" residual_nonzero="<<nonzero<<"/"<<n
             <<" decoder_replay_mismatch="<<predictor_mismatch<<"\n";
    std::cout<<"Exact restoration guaranteed by prediction XOR residual, not by ANN alone.\n";
}
void demo(const fs::path& dir) {
    fs::create_directories(dir);
    const std::string t="Dr Moagi recursive Mandelbulb geometry. Encode decode iterate.\nMultimodal field synthesis.\n";
    Bytes text(t.begin(),t.end());write_file(dir/"input.txt",text);
    Bytes ppm;const std::string hdr="P6\n48 48\n255\n";ppm.assign(hdr.begin(),hdr.end());
    for(int y=0;y<48;y++)for(int x=0;x<48;x++){
        auto f=mandelbulb({2.f*x/47-1.f,2.f*y/47-1.f,.2f});
        ppm.push_back(Byte(std::clamp(int((f+1.f)*125.f),0,255)));
        ppm.push_back(Byte(x*255/47));ppm.push_back(Byte(y*255/47));
    }
    write_file(dir/"input.ppm",ppm);
    Bytes wav(44,0);const std::uint32_t rate=8000,samples=2400,nbytes=samples*2;
    auto write16=[&](int j,std::uint16_t v){wav[std::size_t(j)]=Byte(v);wav[std::size_t(j+1)]=Byte(v>>8);};
    auto write32=[&](int j,std::uint32_t v){for(int k=0;k<4;k++)wav[std::size_t(j+k)]=Byte(v>>(8*k));};
    std::memcpy(wav.data(),"RIFF",4);write32(4,36+nbytes);std::memcpy(wav.data()+8,"WAVEfmt ",8);
    write32(16,16);write16(20,1);write16(22,1);write32(24,rate);write32(28,rate*2);write16(32,2);write16(34,16);
    std::memcpy(wav.data()+36,"data",4);write32(40,nbytes);
    for(std::uint32_t i=0;i<samples;i++){
        auto v=std::int16_t(std::lround(11000*std::sin(2*PI*440.f*float(i)/float(rate))));
        auto bits=std::uint16_t(v);wav.push_back(Byte(bits));wav.push_back(Byte(bits>>8));
    }
    write_file(dir/"input.wav",wav);
    for(const auto& entry:std::array<std::pair<std::string,std::string>,3>{{{"text","txt"},{"image","ppm"},{"audio","wav"}}}) {
        const auto kind=entry.first,extension=entry.second;
        const auto name=dir/("input."+extension);
        auto original=read_file(name);auto m=parse_media(kind,original);
        auto a=encode(m,5);
        save(a,dir/(kind+".dmq"));
        auto loaded=load(dir/(kind+".dmq"));
        auto restored=combine(loaded.media,restore(loaded));
        require(restored==original,"Roundtrip mismatch on "+kind);
        write_file(dir/(kind+"_restored."+extension),restored);
        write_file(dir/(kind+"_generated."+extension),combine(loaded.media,generate(loaded,123u,.8f)));
        std::cout<<"PASS exact "<<kind<<" round-trip ("<<original.size()<<" bytes). ";inspect(loaded);
    }
    std::cout<<"Demo outputs in: "<<dir<<"\n";
}
} // namespace dm

static void emit_quine() {
    // No filesystem read: the raw literal reconstructs this entire translation unit.
    const std::string source_template = R"DMQ(// Dr Moagi Mandelbulb 3D Recursive ANN Multimodal Quine
// C++20 standalone demonstration. Standard library only.
// Build: g++ -O2 -std=c++20 -Wall -Wextra -pedantic dm_mandelquine.cpp -o dm_mandelquine
// Quine: ./dm_mandelquine quine > replica.cpp  (cmp dm_mandelquine.cpp replica.cpp)
// NOT a trained LLM or production multimedia codec; lossless roundtrip uses residual side-data.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dm {
using Byte = std::uint8_t;
using Bytes = std::vector<Byte>;
namespace fs = std::filesystem;
constexpr float PI = 3.14159265358979323846f;
constexpr std::size_t LIMIT = 8u * 1024u * 1024u;
constexpr int H = 12, Z = 4, FI = 6, DI = Z + 5;

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
Byte quantize(float x) { return Byte(std::clamp(int(std::lround((std::clamp(x,-1.f,1.f)+1.f)*127.5f)),0,255)); }
float norm(Byte b) { return (float(b)/127.5f)-1.f; }

Bytes read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    require(bool(f), "Cannot open: " + p.string());
    auto n = f.tellg(); require(n >= 0 && std::uint64_t(n) <= LIMIT + 65536, "Input too large (limit 8 MiB payload)");
    f.seekg(0); Bytes b(static_cast<std::size_t>(n));
    if (!b.empty()) f.read(reinterpret_cast<char*>(b.data()), std::streamsize(b.size()));
    require(bool(f) || b.empty(), "Read failed"); return b;
}
void write_file(const fs::path& p, const Bytes& b) {
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc); require(bool(f), "Cannot write: " + p.string());
    if (!b.empty()) f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    require(bool(f), "Write failed: " + p.string());
}
void u32(std::ostream& o, std::uint32_t x) { for (int i=0;i<4;i++) o.put(char((x>>(8*i))&255u)); }
void u64(std::ostream& o, std::uint64_t x) { for (int i=0;i<8;i++) o.put(char((x>>(8*i))&255u)); }
std::uint32_t r32(std::istream& s) { std::uint32_t n=0; for(int i=0;i<4;i++){int c=s.get();require(c!=EOF,"Truncated archive");n|=std::uint32_t(c)<<(8*i);} return n; }
std::uint64_t r64(std::istream& s) { std::uint64_t n=0; for(int i=0;i<8;i++){int c=s.get();require(c!=EOF,"Truncated archive");n|=std::uint64_t(c)<<(8*i);} return n; }
void f32(std::ostream& s, float x) { static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559); u32(s,std::bit_cast<std::uint32_t>(x)); }
float rf32(std::istream& s) { return std::bit_cast<float>(r32(s)); }
void putbytes(std::ostream& s, const Bytes& v) { if(!v.empty())s.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size())); }
Bytes getbytes(std::istream& s,std::size_t n) { require(n<=LIMIT+65536,"Invalid field length"); Bytes b(n);if(n)s.read(reinterpret_cast<char*>(b.data()),std::streamsize(n));require(bool(s)||n==0,"Truncated archive data");return b; }
std::uint32_t le32(const Bytes& d,std::size_t i) {require(i+4<=d.size(),"Bad WAV header");return std::uint32_t(d[i])|(std::uint32_t(d[i+1])<<8)|(std::uint32_t(d[i+2])<<16)|(std::uint32_t(d[i+3])<<24);}
std::uint16_t le16(const Bytes& d,std::size_t i) {require(i+2<=d.size(),"Bad WAV header");return std::uint16_t(d[i])|(std::uint16_t(d[i+1])<<8);}

struct Media {
    std::string kind;
    Bytes prefix, data, suffix;
    std::uint32_t width=0,height=0,channels=0,sample_rate=0;
};
Media parse_media(const std::string& kind,const Bytes& input) {
    Media m; m.kind=kind;
    if(kind=="text"||kind=="bytes") {m.data=input;return m;}
    if(kind=="image") {
        // Strict P6 PPM (RGB 8-bit) with ASCII header and optional comments.
        require(input.size()>12 && input[0]=='P' && input[1]=='6', "Image must be P6 PPM");
        std::size_t p=0;
        auto token=[&](){
            while(p<input.size()) {
                if(input[p]=='#') {while(p<input.size() && input[p]!='\n') ++p;}
                else if(input[p]==' '||input[p]=='\n'||input[p]=='\r'||input[p]=='\t') ++p;
                else break;
            }
            std::string s;
            while(p<input.size() && input[p]!='#' && input[p]!=' ' && input[p]!='\n' && input[p]!='\r' && input[p]!='\t') s+=char(input[p++]);
            require(!s.empty() && s.size()<20,"Bad PPM header");return s;
        };
        require(token()=="P6","Not P6 PPM");
        m.width=std::stoul(token());m.height=std::stoul(token());
        require(token()=="255","Only 8-bit PPM supported");
        require(p<input.size() && (input[p]==' '||input[p]=='\n'||input[p]=='\r'||input[p]=='\t'),"Missing PPM separator");
        // PPM is specified to use one whitespace separator before binary pixels.
        if(input[p]=='\r' && p+1<input.size() && input[p+1]=='\n')p+=2;else ++p;
        require(m.width>0 && m.height>0 && std::uint64_t(m.width)*m.height*3<=LIMIT,"PPM dimensions invalid");
        require(input.size()==p+std::uint64_t(m.width)*m.height*3,"PPM raster size mismatch");
        m.channels=3; m.prefix.assign(input.begin(),input.begin()+std::ptrdiff_t(p));m.data.assign(input.begin()+std::ptrdiff_t(p),input.end());return m;
    }
    if(kind=="audio") {
        require(input.size()>=44 && std::memcmp(input.data(),"RIFF",4)==0 && std::memcmp(input.data()+8,"WAVE",4)==0,"Audio must be RIFF/WAVE");
        std::size_t pos=12,dataoff=0,ndata=0;bool fmt=false,gotdata=false;
        while(pos+8<=input.size()) {
            const auto chunk_size=le32(input,pos+4);
            const auto begin=pos+8;
            require(chunk_size<=input.size()-begin,"WAV chunk beyond file");
            const std::string name(reinterpret_cast<const char*>(input.data()+pos),4);
            if(name=="fmt ") {
                require(chunk_size>=16,"Invalid PCM format");
                require(le16(input,begin)==1 && le16(input,begin+14)==16,"Only PCM16 WAV supported");
                m.channels=le16(input,begin+2);m.sample_rate=le32(input,begin+4);
                require(m.channels>0 && m.channels<=8 && m.sample_rate>0,"Bad WAV channels/sample rate");fmt=true;
            } else if(name=="data" && !gotdata) {dataoff=begin;ndata=chunk_size;gotdata=true;}
            pos=begin+chunk_size+(chunk_size&1u);
        }
        require(fmt&&gotdata&&ndata>0&&ndata<=LIMIT&&ndata%(2*m.channels)==0,"Unsupported WAV payload");
        m.prefix.assign(input.begin(),input.begin()+std::ptrdiff_t(dataoff));
        m.data.assign(input.begin()+std::ptrdiff_t(dataoff),input.begin()+std::ptrdiff_t(dataoff+ndata));
        m.suffix.assign(input.begin()+std::ptrdiff_t(dataoff+ndata),input.end());return m;
    }
    throw std::runtime_error("Kind must be text/image/audio/bytes");
}
Bytes combine(const Media& m, const Bytes& data) {
    Bytes out;out.reserve(m.prefix.size()+data.size()+m.suffix.size());
    out.insert(out.end(),m.prefix.begin(),m.prefix.end());out.insert(out.end(),data.begin(),data.end());out.insert(out.end(),m.suffix.begin(),m.suffix.end());return out;
}
std::uint32_t cube_side(std::size_t n) {
    std::uint32_t s=1;while(std::uint64_t(s)*s*s<n)++s;return s;
}
struct Point { float x,y,z; };
Point position(std::size_t i,std::uint32_t s) {
    const float scale=s>1?2.f/float(s-1):0.f;
    auto x=i%s,y=(i/s)%s,z=i/(std::size_t(s)*s);
    return {float(x)*scale-1.f,float(y)*scale-1.f,float(z)*scale-1.f};
}
// Mandelbulb exponent-eight escape-time scalar, mapped to [-1,1].
float mandelbulb(const Point& pt) {
    float zx=0,zy=0,zz=0;int k=0;
    for(;k<7;k++) {
        const float r=std::sqrt(zx*zx+zy*zy+zz*zz);
        if(r>2.f)break;
        const float theta=std::acos(std::clamp(zz/std::max(r,1e-8f),-1.f,1.f))*8.f;
        const float phi=std::atan2(zy,zx)*8.f;
        const float rp=std::pow(r,8.f);
        const float st=std::sin(theta);
        zx=rp*st*std::cos(phi)+pt.x;
        zy=rp*st*std::sin(phi)+pt.y;
        zz=rp*std::cos(theta)+pt.z;
    }
    return k==7?-1.f:1.f-2.f*float(k)/7.f;
}
using Lat = std::array<float,Z>;
struct Cell {Point p{};float bulb=0,neigh=0;};
std::vector<Cell> cells_for(const Bytes& b,std::uint32_t s) {
    std::vector<Cell> c(b.size());const std::size_t ss=std::size_t(s)*s;
    for(std::size_t i=0;i<b.size();i++) {
        c[i].p=position(i,s);c[i].bulb=mandelbulb(c[i].p);
        const std::size_t x=i%s,y=(i/s)%s,z=i/ss;
        float sum=0;int count=0;
        auto a=[&](std::size_t j){if(j<b.size()){sum+=norm(b[j]);++count;}};
        if(x) a(i-1);
        if(x+1<s) a(i+1);
        if(y) a(i-s);
        if(y+1<s) a(i+s);
        if(z) a(i-ss);
        if(z+1<s) a(i+ss);
        c[i].neigh=count?sum/count:0.f;
    }
    return c;
}
std::array<float,FI> features(Byte b,const Cell& c) {
    return {norm(b),c.bulb,c.p.x,c.p.y,c.p.z,c.neigh};
}

struct Network {
    std::array<float,H*FI> ew{};std::array<float,H> eb{};
    std::array<float,Z*H> zw{};std::array<float,Z> zb{};
    std::array<float,H*DI> dw{};std::array<float,H> db{};
    std::array<float,H> ow{};float ob=0;
    explicit Network(bool initialize=true) {
        if(initialize){std::mt19937 rng(0xD0A613u);std::uniform_real_distribution<float> dist(-.3f,.3f);
            for(auto& x:ew) x=dist(rng);
            for(auto& x:zw) x=dist(rng);
            for(auto& x:dw) x=dist(rng);
            for(auto& x:ow) x=dist(rng);
        }
    }
    struct Cache {std::array<float,H> eh{},dh{};Lat z{};std::array<float,DI> din{};float y=0;};
    Cache forward(const std::array<float,FI>& x) const {
        Cache c;
        for(int j=0;j<H;j++){float v=eb[j];for(int k=0;k<FI;k++)v+=ew[j*FI+k]*x[k];c.eh[j]=std::tanh(v);}
        for(int k=0;k<Z;k++){float v=zb[k];for(int j=0;j<H;j++)v+=zw[k*H+j]*c.eh[j];c.z[k]=std::tanh(v);}
        for(int k=0;k<Z;k++)c.din[k]=c.z[k];
        for(int k=0;k<5;k++)c.din[Z+k]=x[k+1];
        for(int j=0;j<H;j++){float v=db[j];for(int k=0;k<DI;k++)v+=dw[j*DI+k]*c.din[k];c.dh[j]=std::tanh(v);}
        float v=ob;for(int j=0;j<H;j++)v+=ow[j]*c.dh[j];c.y=std::tanh(v);
        return c;
    }
    float decode(const Lat& z,const Cell& cell) const {
        std::array<float,DI> x{};
        for(int k=0;k<Z;k++)x[k]=z[k];
        x[Z]=cell.bulb;x[Z+1]=cell.p.x;x[Z+2]=cell.p.y;x[Z+3]=cell.p.z;x[Z+4]=cell.neigh;
        float o=ob;
        for(int j=0;j<H;j++){float a=db[j];for(int k=0;k<DI;k++)a+=dw[j*DI+k]*x[k];o+=ow[j]*std::tanh(a);}
        return std::tanh(o);
    }
    float train_one(const std::array<float,FI>& x,float lr) {
        const auto c=forward(x);const float err=c.y-x[0];
        const float dout=err*(1.f-c.y*c.y);
        std::array<float,H> dhgrad{},ehgrad{};Lat dzgrad{};
        for(int j=0;j<H;j++)dhgrad[j]=dout*ow[j]*(1.f-c.dh[j]*c.dh[j]);
        for(int k=0;k<Z;k++){
            float v=0;for(int j=0;j<H;j++)v+=dhgrad[j]*dw[j*DI+k];
            dzgrad[k]=v*(1.f-c.z[k]*c.z[k]);
        }
        for(int j=0;j<H;j++){
            float v=0;for(int k=0;k<Z;k++)v+=dzgrad[k]*zw[k*H+j];
            ehgrad[j]=v*(1.f-c.eh[j]*c.eh[j]);
        }
        for(int j=0;j<H;j++){ow[j]-=lr*dout*c.dh[j];for(int k=0;k<DI;k++)dw[j*DI+k]-=lr*dhgrad[j]*c.din[k];db[j]-=lr*dhgrad[j];}
        ob-=lr*dout;
        for(int k=0;k<Z;k++){for(int j=0;j<H;j++)zw[k*H+j]-=lr*dzgrad[k]*c.eh[j];zb[k]-=lr*dzgrad[k];}
        for(int j=0;j<H;j++){for(int k=0;k<FI;k++)ew[j*FI+k]-=lr*ehgrad[j]*x[k];eb[j]-=lr*ehgrad[j];}
        return err*err;
    }
    void serialize(std::ostream& s) const {
        auto wr=[&](const auto& a){for(float x:a)f32(s,x);};wr(ew);wr(eb);wr(zw);wr(zb);wr(dw);wr(db);wr(ow);f32(s,ob);
    }
    void deserialize(std::istream& s) {
        auto rd=[&](auto& a){for(float& x:a){x=rf32(s);require(std::isfinite(x),"Invalid model weights");}};
        rd(ew);rd(eb);rd(zw);rd(zb);rd(dw);rd(db);rd(ow);ob=rf32(s);require(std::isfinite(ob),"Invalid model bias");
    }
};

// Spatial feedback R for z^(t+1) = .68*z^t + .32*tanh(z^0 + .18*N(z^t) + .08*phi).
void recursive_refine(std::vector<Lat>& z,const std::vector<Cell>& c,std::uint32_t s,int rounds=4) {
    const auto base=z;std::vector<Lat> next(z.size());const std::size_t ss=std::size_t(s)*s;
    for(int t=0;t<rounds;t++) {
        for(std::size_t i=0;i<z.size();i++) {
            Lat avg{};int count=0;const std::size_t x=i%s,y=(i/s)%s,h=i/ss;
            auto a=[&](std::size_t j){if(j<z.size()){for(int k=0;k<Z;k++)avg[k]+=z[j][k];++count;}};
            if(x) a(i-1);
        if(x+1<s) a(i+1);
            if(y) a(i-s);
        if(y+1<s) a(i+s);
            if(h) a(i-ss);
            if(h+1<s) a(i+ss);
            for(int k=0;k<Z;k++){
                float v=base[i][k]+(count?.18f*avg[k]/count:0.f)+.08f*c[i].bulb;
                next[i][k]=.68f*z[i][k]+.32f*std::tanh(v);
            }
        }
        z.swap(next);
    }
}

struct Archive {
    Media media;std::uint32_t side=0,epochs=0;Network net{false};
    std::vector<Cell> geom;
    std::vector<Lat> latent;
    Bytes prediction,residual;
};
Archive encode(const Media& m,int epochs) {
    require(!m.data.empty()&&m.data.size()<=LIMIT,"Payload must contain 1..8 MiB");
    Archive a;a.media=m;a.side=cube_side(m.data.size());a.epochs=std::uint32_t(epochs);a.net=Network(true);
    a.geom=cells_for(m.data,a.side);
    std::mt19937 rng(0xABCD918u);
    std::uniform_int_distribution<std::size_t> pick(0,m.data.size()-1);
    const std::size_t steps=std::min<std::size_t>(6000,m.data.size()*2);
    float mse=0;
    for(int ep=0;ep<epochs;ep++){
        double loss=0;
        for(std::size_t j=0;j<steps;j++) {
            const auto i=pick(rng);loss+=a.net.train_one(features(m.data[i],a.geom[i]),.018f);
        }
        mse=float(loss/double(steps));
    }
    a.latent.resize(m.data.size());a.prediction.resize(m.data.size());a.residual.resize(m.data.size());
    for(std::size_t i=0;i<m.data.size();i++)a.latent[i]=a.net.forward(features(m.data[i],a.geom[i])).z;
    recursive_refine(a.latent,a.geom,a.side);
    for(std::size_t i=0;i<m.data.size();i++){
        a.prediction[i]=quantize(a.net.decode(a.latent[i],a.geom[i]));
        a.residual[i]=Byte(m.data[i]^a.prediction[i]);
    }
    std::cout<<"Trained epoch MSE (pre-refinement): "<<std::fixed<<std::setprecision(6)<<mse<<"\n";
    return a;
}
// Store explicit predicted byte plus exact XOR residual. No compression claims.
void save(const Archive& a,const fs::path& path) {
    if(path.has_parent_path())fs::create_directories(path.parent_path());
    std::ofstream s(path,std::ios::binary|std::ios::trunc);require(bool(s),"Cannot create archive");
    s.write("DMQ3D002",8);
    const auto& m=a.media;
    const auto kind=m.kind=="text"?1u:m.kind=="image"?2u:m.kind=="audio"?3u:4u;
    u32(s,kind);u32(s,a.side);u32(s,a.epochs);
    u32(s,m.width);u32(s,m.height);u32(s,m.channels);u32(s,m.sample_rate);
    u64(s,a.latent.size());u64(s,m.prefix.size());u64(s,m.suffix.size());
    putbytes(s,m.prefix);putbytes(s,m.suffix);a.net.serialize(s);
    for(const auto& z:a.latent)for(float v:z)f32(s,v);
    for(const auto& c:a.geom)f32(s,c.neigh);
    putbytes(s,a.prediction);putbytes(s,a.residual);require(bool(s),"Archive write failed");
}
Archive load(const fs::path& path) {
    std::ifstream s(path,std::ios::binary);require(bool(s),"Cannot read archive");
    char sig[8];s.read(sig,8);require(bool(s)&&std::memcmp(sig,"DMQ3D002",8)==0,"Wrong archive signature");
    Archive a;const auto kind=r32(s);require(kind>=1&&kind<=4,"Wrong modality");
    a.media.kind=kind==1?"text":kind==2?"image":kind==3?"audio":"bytes";
    a.side=r32(s);a.epochs=r32(s);
    a.media.width=r32(s);a.media.height=r32(s);a.media.channels=r32(s);a.media.sample_rate=r32(s);
    const auto n=r64(s),p=r64(s),q=r64(s);
    require(n>0&&n<=LIMIT&&p<=65536&&q<=65536&&a.side==cube_side(std::size_t(n)),"Invalid archive shape");
    a.media.prefix=getbytes(s,std::size_t(p));a.media.suffix=getbytes(s,std::size_t(q));a.net.deserialize(s);
    a.latent.resize(std::size_t(n));
    for(auto& z:a.latent)for(auto& v:z){v=rf32(s);require(std::isfinite(v)&&std::abs(v)<=2.f,"Corrupt latent field");}
    a.geom.resize(std::size_t(n));
    for(auto& c:a.geom){c.neigh=rf32(s);require(std::isfinite(c.neigh) && std::abs(c.neigh)<=1.001f,"Corrupt neighborhood feature");}
    a.prediction=getbytes(s,std::size_t(n));a.residual=getbytes(s,std::size_t(n));
    require(s.peek()==EOF,"Archive trailing data not expected");
    // Regenerate coordinates and fractal; preserve archived local-neighborhood features.
    for(std::size_t i=0;i<n;i++){a.geom[i].p=position(i,a.side);a.geom[i].bulb=mandelbulb(a.geom[i].p);}
    return a;
}
Bytes restore(const Archive& a) {
    Bytes b(a.residual.size());for(std::size_t i=0;i<b.size();i++)b[i]=Byte(a.prediction[i]^a.residual[i]);return b;
}
// Generative decoding is separate from the exact residual reconstruction path.
Bytes generate(const Archive& a,std::uint32_t seed,float strength) {
    require(strength>=0.f&&strength<=2.f,"Strength must be between 0 and 2");
    const auto n=a.latent.size();Bytes out(n);std::mt19937 rng(seed);
    std::uniform_real_distribution<float> jitter(-1.f,1.f);
    for(std::size_t i=0;i<n;i++) {
        auto z=a.latent[i];const auto& c=a.geom[i];
        for(int k=0;k<Z;k++)z[k]=std::tanh(z[k]+strength*(.35f*c.bulb+.18f*jitter(rng)));
        const auto value=a.net.decode(z,c);
        out[i]=quantize(value);
    }
    if(a.media.kind=="image") {
        for(std::size_t i=0;i<n;i++){
            const float f=a.geom[i].bulb;
            const int rgb=int(i%3);
            const float wave=20.f*std::sin(.12f*float(i/3)+(rgb+1)*2.f+f*6.f);
            out[i]=Byte(std::clamp(int(out[i])+int(strength*(wave+35*f)),0,255));
        }
    } else if(a.media.kind=="audio") {
        // Stable 16-bit oscillator modulated by neural predictions; no random PCM-byte noise.
        require(a.media.sample_rate>0&&n%2==0,"Bad audio generation metadata");
        float phase=0;
        for(std::size_t i=0;i<n/2;i++){
            const float mod=float(out[2*i])/255.f;
            const float frequency=180.f+260.f*mod;
            phase+=2.f*PI*frequency/float(a.media.sample_rate);
            const float env=(.15f+.35f*mod)*(.8f+.2f*a.geom[2*i].bulb);
            const auto pcm=std::int16_t(std::lround(std::clamp(env*std::sin(phase),-1.f,1.f)*30000.f));
            const auto bits=std::uint16_t(pcm);
            out[2*i]=Byte(bits&255u);out[2*i+1]=Byte(bits>>8);
        }
    } else if(a.media.kind=="text") {
        // Procedural lexicon selection driven by ANN bytes: not linguistic understanding.
        constexpr std::array<const char*,20> words={"latent","geometry","recursive","field","signal","encode","decode","fractal","neural","phase","memory","vector","kernel","matrix","system","feedback","wave","form","pattern","state"};
        std::string sentence;
        for(std::size_t i=0;i<n;i+=8){
            sentence+=words[(std::size_t(out[i])+std::size_t(seed%20))%words.size()];
            sentence+=((i/8)%9==8)?".\n":" ";
        }
        out.assign(sentence.begin(),sentence.end());
    }
    return out;
}
void inspect(const Archive& a) {
    const auto n=a.latent.size();
    std::cout<<"mode="<<a.media.kind<<" shape="<<a.side<<"x"<<a.side<<"x"<<a.side
             <<" occupied="<<n<<" epochs="<<a.epochs<<"\n";
    if(a.media.kind=="image")std::cout<<"PPM="<<a.media.width<<"x"<<a.media.height<<" RGB\n";
    if(a.media.kind=="audio")std::cout<<"PCM16 rate="<<a.media.sample_rate<<" channels="<<a.media.channels<<"\n";
    double mse=0;std::size_t nonzero=0,predictor_mismatch=0;
    for(std::size_t i=0;i<n;i++){
        const int error=int(a.prediction[i])-int(Byte(a.prediction[i]^a.residual[i]));
        mse+=double(error)*error;nonzero+=a.residual[i]!=0;
        predictor_mismatch+=dm::quantize(a.net.decode(a.latent[i],a.geom[i]))!=a.prediction[i];
    }
    std::cout<<"ANN pre-residual RMSE="<<std::sqrt(mse/double(n))<<" bytes"
             <<" residual_nonzero="<<nonzero<<"/"<<n
             <<" decoder_replay_mismatch="<<predictor_mismatch<<"\n";
    std::cout<<"Exact restoration guaranteed by prediction XOR residual, not by ANN alone.\n";
}
void demo(const fs::path& dir) {
    fs::create_directories(dir);
    const std::string t="Dr Moagi recursive Mandelbulb geometry. Encode decode iterate.\nMultimodal field synthesis.\n";
    Bytes text(t.begin(),t.end());write_file(dir/"input.txt",text);
    Bytes ppm;const std::string hdr="P6\n48 48\n255\n";ppm.assign(hdr.begin(),hdr.end());
    for(int y=0;y<48;y++)for(int x=0;x<48;x++){
        auto f=mandelbulb({2.f*x/47-1.f,2.f*y/47-1.f,.2f});
        ppm.push_back(Byte(std::clamp(int((f+1.f)*125.f),0,255)));
        ppm.push_back(Byte(x*255/47));ppm.push_back(Byte(y*255/47));
    }
    write_file(dir/"input.ppm",ppm);
    Bytes wav(44,0);const std::uint32_t rate=8000,samples=2400,nbytes=samples*2;
    auto write16=[&](int j,std::uint16_t v){wav[std::size_t(j)]=Byte(v);wav[std::size_t(j+1)]=Byte(v>>8);};
    auto write32=[&](int j,std::uint32_t v){for(int k=0;k<4;k++)wav[std::size_t(j+k)]=Byte(v>>(8*k));};
    std::memcpy(wav.data(),"RIFF",4);write32(4,36+nbytes);std::memcpy(wav.data()+8,"WAVEfmt ",8);
    write32(16,16);write16(20,1);write16(22,1);write32(24,rate);write32(28,rate*2);write16(32,2);write16(34,16);
    std::memcpy(wav.data()+36,"data",4);write32(40,nbytes);
    for(std::uint32_t i=0;i<samples;i++){
        auto v=std::int16_t(std::lround(11000*std::sin(2*PI*440.f*float(i)/float(rate))));
        auto bits=std::uint16_t(v);wav.push_back(Byte(bits));wav.push_back(Byte(bits>>8));
    }
    write_file(dir/"input.wav",wav);
    for(const auto& entry:std::array<std::pair<std::string,std::string>,3>{{{"text","txt"},{"image","ppm"},{"audio","wav"}}}) {
        const auto kind=entry.first,extension=entry.second;
        const auto name=dir/("input."+extension);
        auto original=read_file(name);auto m=parse_media(kind,original);
        auto a=encode(m,5);
        save(a,dir/(kind+".dmq"));
        auto loaded=load(dir/(kind+".dmq"));
        auto restored=combine(loaded.media,restore(loaded));
        require(restored==original,"Roundtrip mismatch on "+kind);
        write_file(dir/(kind+"_restored."+extension),restored);
        write_file(dir/(kind+"_generated."+extension),combine(loaded.media,generate(loaded,123u,.8f)));
        std::cout<<"PASS exact "<<kind<<" round-trip ("<<original.size()<<" bytes). ";inspect(loaded);
    }
    std::cout<<"Demo outputs in: "<<dir<<"\n";
}
} // namespace dm

static void emit_quine() {
    // No filesystem read: the raw literal reconstructs this entire translation unit.
    const std::string source_template = __SELF_LITERAL_6C8E__;
    std::string result=source_template;
    const auto at=result.find("__SELF_LITERAL_6C8E__");
    if(at==std::string::npos)throw std::runtime_error("Quine marker missing");
    result.replace(at,std::strlen("__SELF_LITERAL_6C8E__"),"R\"DMQ("+source_template+")DMQ\"");
    std::cout<<result;
}
int main(int argc,char** argv) {
    try {
        if(argc<2) {
            std::cout<<"Usage:\n  "<<argv[0]<<" demo [folder]\n  "<<argv[0]<<" encode <text|image|audio|bytes> <input> <archive> [epochs]\n"
                <<"  "<<argv[0]<<" decode <archive> <output>\n  "<<argv[0]<<" generate <archive> <output> [seed] [strength]\n"
                <<"  "<<argv[0]<<" inspect <archive>\n  "<<argv[0]<<" quine > replica.cpp\n";return 0;
        }
        const std::string cmd=argv[1];
        if(cmd=="quine") {emit_quine();return 0;}
        if(cmd=="demo") {dm::demo(argc>=3?argv[2]:"dmq_demo");return 0;}
        if(cmd=="encode"&&argc>=5) {
            int ep=argc>=6?std::stoi(argv[5]):6;dm::require(ep>0&&ep<=40,"epochs must be 1..40");
            auto input=dm::read_file(argv[3]);auto a=dm::encode(dm::parse_media(argv[2],input),ep);
            dm::save(a,argv[4]);dm::inspect(a);return 0;
        }
        if(cmd=="decode"&&argc>=4) {
            auto a=dm::load(argv[2]);dm::write_file(argv[3],dm::combine(a.media,dm::restore(a)));return 0;
        }
        if(cmd=="generate"&&argc>=4) {
            auto a=dm::load(argv[2]);const auto seed=argc>=5?std::uint32_t(std::stoul(argv[4])):123u;
            const auto strength=argc>=6?std::stof(argv[5]):.8f;
            // Synthetic text has a new payload length; only text/bytes have empty headers.
            dm::write_file(argv[3],dm::combine(a.media,dm::generate(a,seed,strength)));return 0;
        }
        if(cmd=="inspect"&&argc>=3) {dm::inspect(dm::load(argv[2]));return 0;}
        throw std::runtime_error("Invalid command or argument count");
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
)DMQ";
    std::string result=source_template;
    const auto at=result.find("__SELF_LITERAL_6C8E__");
    if(at==std::string::npos)throw std::runtime_error("Quine marker missing");
    result.replace(at,std::strlen("__SELF_LITERAL_6C8E__"),"R\"DMQ("+source_template+")DMQ\"");
    std::cout<<result;
}
int main(int argc,char** argv) {
    try {
        if(argc<2) {
            std::cout<<"Usage:\n  "<<argv[0]<<" demo [folder]\n  "<<argv[0]<<" encode <text|image|audio|bytes> <input> <archive> [epochs]\n"
                <<"  "<<argv[0]<<" decode <archive> <output>\n  "<<argv[0]<<" generate <archive> <output> [seed] [strength]\n"
                <<"  "<<argv[0]<<" inspect <archive>\n  "<<argv[0]<<" quine > replica.cpp\n";return 0;
        }
        const std::string cmd=argv[1];
        if(cmd=="quine") {emit_quine();return 0;}
        if(cmd=="demo") {dm::demo(argc>=3?argv[2]:"dmq_demo");return 0;}
        if(cmd=="encode"&&argc>=5) {
            int ep=argc>=6?std::stoi(argv[5]):6;dm::require(ep>0&&ep<=40,"epochs must be 1..40");
            auto input=dm::read_file(argv[3]);auto a=dm::encode(dm::parse_media(argv[2],input),ep);
            dm::save(a,argv[4]);dm::inspect(a);return 0;
        }
        if(cmd=="decode"&&argc>=4) {
            auto a=dm::load(argv[2]);dm::write_file(argv[3],dm::combine(a.media,dm::restore(a)));return 0;
        }
        if(cmd=="generate"&&argc>=4) {
            auto a=dm::load(argv[2]);const auto seed=argc>=5?std::uint32_t(std::stoul(argv[4])):123u;
            const auto strength=argc>=6?std::stof(argv[5]):.8f;
            // Synthetic text has a new payload length; only text/bytes have empty headers.
            dm::write_file(argv[3],dm::combine(a.media,dm::generate(a,seed,strength)));return 0;
        }
        if(cmd=="inspect"&&argc>=3) {dm::inspect(dm::load(argv[2]));return 0;}
        throw std::runtime_error("Invalid command or argument count");
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
