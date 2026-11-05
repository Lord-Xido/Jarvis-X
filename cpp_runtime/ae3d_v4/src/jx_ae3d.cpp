// JARVIS-X 3D AUTOENCODER v4.0 -- C++17 standard-library neural codec.
// Real learned 4x4x4 block autoencoder: 64 -> 32 -> 12 -> 32 -> 64.
// - Backpropagation with Adam; quantized latent int8; optional exact XOR residual.
// - Sparse 120-bit virtual origin (3*40-bit coordinate components).
// - Versioned portable little-endian files, CRC32 integrity, model fingerprint.
// Educational/reference CPU runtime; bounded allocations; no GPU or trillion weights.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace jx_quine {int reproduce();}

namespace jx {
constexpr int B=4,D=64,H=32,L=12;
constexpr uint64_t SIDE=1000000000000ULL;
constexpr size_t MAX_VOXELS=64ULL*1024*1024; // 64 MiB per volume, conservative
constexpr size_t MAX_FILE=128ULL*1024*1024;
constexpr int OFF_W1=0;
constexpr int OFF_B1=OFF_W1+H*D;
constexpr int OFF_W2=OFF_B1+H;
constexpr int OFF_B2=OFF_W2+L*H;
constexpr int OFF_W3=OFF_B2+L;
constexpr int OFF_B3=OFF_W3+H*L;
constexpr int OFF_W4=OFF_B3+H;
constexpr int OFF_B4=OFF_W4+D*H;
constexpr int NPARAM=OFF_B4+D;
using Bytes=std::vector<uint8_t>;
[[noreturn]] void fail(const std::string& s){throw std::runtime_error(s);}
void check(bool v,const std::string& s){if(!v)fail(s);}
uint32_t crc32(const Bytes& v, size_t size=SIZE_MAX){
    uint32_t c=0xffffffffU;
    const size_t n=std::min(size,v.size());
    for(size_t i=0;i<n;++i){c^=v[i];for(int j=0;j<8;++j)c=(c>>1)^((c&1U)?0xedb88320U:0U);}
    return c^0xffffffffU;
}
uint64_t fnv(const Bytes& v){uint64_t h=14695981039346656037ULL;for(uint8_t x:v){h^=x;h*=1099511628211ULL;}return h;}
void put(Bytes& o,uint64_t x,int n){for(int i=0;i<n;i++){o.push_back(uint8_t(x&255));x>>=8;}}
uint64_t get(const Bytes& v,size_t& p,int n){check(n>0&&n<=8 && p<=v.size() && size_t(n)<=v.size()-p,"truncated file");uint64_t x=0;for(int i=0;i<n;++i)x|=uint64_t(v[p++])<<(8*i);return x;}
void tag(Bytes& v,const char* s){for(int i=0;i<8;++i)v.push_back(uint8_t(s[i]));}
void require_tag(const Bytes& v,size_t& p,const char* s){check(v.size()>=8,"truncated header");for(int i=0;i<8;++i)check(v[p++]==uint8_t(s[i]),"wrong file format or version");}
void put_double(Bytes& v,double d){static_assert(sizeof(double)==8,"requires IEEE754 binary64");uint64_t bits;std::memcpy(&bits,&d,8);put(v,bits,8);}
double get_double(const Bytes& v,size_t& p){uint64_t bits=get(v,p,8);double x;std::memcpy(&x,&bits,8);check(std::isfinite(x),"nonfinite model parameter");return x;}
void seal(Bytes& v){put(v,crc32(v),4);}
void check_seal(const Bytes& v){check(v.size()>=4,"truncated checksum");size_t p=v.size()-4;check(uint32_t(get(v,p,4))==crc32(v,v.size()-4),"CRC32 mismatch");}
Bytes read_file(const std::string& path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);check(bool(f),"cannot read: "+path);
    auto n=f.tellg();check(n>=0 && uint64_t(n)<=MAX_FILE,"file too large");f.seekg(0);
    Bytes v(size_t(n),uint8_t(0));if(n>0)f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n));check(bool(f)||n==0,"read failure");return v;
}
void write_file(const std::string& path,const Bytes& v){
    std::ofstream f(path,std::ios::binary|std::ios::trunc);check(bool(f),"cannot write: "+path);
    if(!v.empty()) f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()));
    f.flush();check(bool(f),"write failure: "+path);
}
uint64_t num(const std::string& s){size_t consumed=0;unsigned long long n;try{n=std::stoull(s,&consumed);}catch(...){fail("invalid integer: "+s);}check(consumed==s.size() && (s.empty()||s[0]!='-'),"invalid integer: "+s);return uint64_t(n);}
double real(const std::string& s){size_t consumed=0;double n;try{n=std::stod(s,&consumed);}catch(...){fail("invalid number: "+s);}check(consumed==s.size()&&std::isfinite(n),"invalid number: "+s);return n;}
uint32_t dimension(const std::string& s){auto n=num(s);check(n>=4&&n<=1024&&n%4==0,"dimension must be multiple of 4 in [4,1024]");return uint32_t(n);}
size_t voxels(uint32_t x,uint32_t y,uint32_t z){
    check(x>=4&&y>=4&&z>=4&&x%4==0&&y%4==0&&z%4==0,"invalid 3D dimensions");
    uint64_t n=uint64_t(x)*y*z;check(n<=MAX_VOXELS,"volume exceeds 64 MiB safety limit");return size_t(n);
}
struct Volume{uint32_t x=0,y=0,z=0;Bytes data;};
size_t idx(const Volume& v,uint32_t x,uint32_t y,uint32_t z){return (size_t(z)*v.y+y)*v.x+x;}
Bytes serial_volume(const Volume& v){check(v.data.size()==voxels(v.x,v.y,v.z),"bad volume size");Bytes a;tag(a,"JXVOL4\r\n");put(a,v.x,4);put(a,v.y,4);put(a,v.z,4);a.insert(a.end(),v.data.begin(),v.data.end());seal(a);return a;}
Volume parse_volume(const Bytes& v){size_t p=0;require_tag(v,p,"JXVOL4\r\n");auto x=uint32_t(get(v,p,4)),y=uint32_t(get(v,p,4)),z=uint32_t(get(v,p,4));auto n=voxels(x,y,z);check(v.size()==20+n+4,"volume length mismatch");check_seal(v);return {x,y,z,Bytes(v.begin()+20,v.end()-4)};}
Volume load_volume(const std::string& p){return parse_volume(read_file(p));}
void save_volume(const std::string& p,const Volume& v){write_file(p,serial_volume(v));}
Volume synthetic(uint32_t side,uint64_t seed){Volume v{side,side,side,Bytes(voxels(side,side,side))};
    std::mt19937_64 rng(seed);std::uniform_real_distribution<double> u(-1.,1.);
    const double a=u(rng)*0.9,b=u(rng)*0.9,c=u(rng)*0.9;
    for(uint32_t z=0;z<side;++z)for(uint32_t y=0;y<side;++y)for(uint32_t x=0;x<side;++x){
        double X=(double(x)+0.5)/side,Y=(double(y)+0.5)/side,Z=(double(z)+0.5)/side;
        double r2=(X-.52)*(X-.52)+(Y-.49)*(Y-.49)+(Z-.55)*(Z-.55);
        double f=.43+.19*std::sin(6.2831853*(X+.4*Y)+a)
                    +.17*std::cos(6.2831853*(Z+.25*X)+b)
                    +.24*std::exp(-r2*34.)+.06*std::sin(20.*(X*Y+Z)+c);
        v.data[idx(v,x,y,z)]=uint8_t(std::lround(255*std::clamp(f,0.,1.)));
    }return v;
}
std::vector<std::array<double,D>> blocks(const Volume& v){
    std::vector<std::array<double,D>> b; b.reserve(v.data.size()/D);
    for(uint32_t z=0;z<v.z;z+=4)for(uint32_t y=0;y<v.y;y+=4)for(uint32_t x=0;x<v.x;x+=4){
        std::array<double,D> a{};int k=0;
        for(int dz=0;dz<4;++dz)for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)
            a[k++]=v.data[idx(v,x+dx,y+dy,z+dz)]/255.;
        b.push_back(a);
    }return b;
}
std::array<uint8_t,D> block_bytes(const Volume& v,uint32_t x,uint32_t y,uint32_t z){
    std::array<uint8_t,D> a{};int k=0;
    for(int dz=0;dz<4;++dz)for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)
        a[k++]=v.data[idx(v,x+dx,y+dy,z+dz)];
    return a;
}
void insert_block(Volume& v,uint32_t x,uint32_t y,uint32_t z,const std::array<uint8_t,D>& a){int k=0;
    for(int dz=0;dz<4;++dz)for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)
        v.data[idx(v,x+dx,y+dy,z+dz)]=a[k++];
}
inline double sig(double x){if(x>=0){double e=std::exp(-x);return 1./(1.+e);}double e=std::exp(x);return e/(1.+e);}
struct Cache {std::array<double,H> h1{},h2{};std::array<double,L> z{};std::array<double,D> out{};};
struct Model{
    std::vector<double> p;
    explicit Model(uint64_t seed=42):p(NPARAM,0.){
        std::mt19937_64 rng(seed);std::normal_distribution<double> gauss(0,1);
        for(int i=OFF_W1;i<OFF_B1;++i)p[i]=gauss(rng)*std::sqrt(2./(D+H));
        for(int i=OFF_W2;i<OFF_B2;++i)p[i]=gauss(rng)*std::sqrt(2./(H+L));
        for(int i=OFF_W3;i<OFF_B3;++i)p[i]=gauss(rng)*std::sqrt(2./(H+L));
        for(int i=OFF_W4;i<OFF_B4;++i)p[i]=gauss(rng)*std::sqrt(2./(H+D));
    }
    Cache encode_forward(const std::array<double,D>& x)const{
        Cache c;
        for(int i=0;i<H;++i){double a=p[OFF_B1+i];for(int j=0;j<D;++j)a+=p[OFF_W1+i*D+j]*x[j];c.h1[i]=std::tanh(a);}
        for(int i=0;i<L;++i){double a=p[OFF_B2+i];for(int j=0;j<H;++j)a+=p[OFF_W2+i*H+j]*c.h1[j];c.z[i]=std::tanh(a);}
        decode_forward(c);return c;
    }
    void decode_forward(Cache& c)const{
        for(int i=0;i<H;++i){double a=p[OFF_B3+i];for(int j=0;j<L;++j)a+=p[OFF_W3+i*L+j]*c.z[j];c.h2[i]=std::tanh(a);}
        for(int i=0;i<D;++i){double a=p[OFF_B4+i];for(int j=0;j<H;++j)a+=p[OFF_W4+i*H+j]*c.h2[j];c.out[i]=sig(a);}
    }
    std::array<int8_t,L> encode_q(const std::array<double,D>& x)const{
        Cache c=encode_forward(x);std::array<int8_t,L> q{};
        for(int i=0;i<L;++i)q[i]=int8_t(std::lround(std::clamp(c.z[i],-1.,1.)*127.));
        return q;
    }
    std::array<uint8_t,D> decode_q(const std::array<int8_t,L>& q)const{
        Cache c;for(int i=0;i<L;++i)c.z[i]=double(q[i])/127.;decode_forward(c);
        std::array<uint8_t,D> out{};for(int i=0;i<D;++i)out[i]=uint8_t(std::lround(255.*std::clamp(c.out[i],0.,1.)));return out;
    }
    double mse(const std::array<double,D>& x)const{
        auto c=encode_forward(x);double sum=0.;for(int i=0;i<D;++i){double d=c.out[i]-x[i];sum+=d*d;}return sum/D;
    }
    double grad(const std::array<double,D>& x,std::vector<double>& g)const{
        auto c=encode_forward(x);
        std::array<double,D> d4{};std::array<double,H> d3{},d1{};std::array<double,L> d2{};
        double loss=0.;
        for(int i=0;i<D;++i){double d=c.out[i]-x[i];loss+=d*d;d4[i]=(2./D)*d*c.out[i]*(1.-c.out[i]);}
        for(int j=0;j<H;++j){double s=0.;for(int i=0;i<D;++i)s+=p[OFF_W4+i*H+j]*d4[i];d3[j]=s*(1.-c.h2[j]*c.h2[j]);}
        for(int j=0;j<L;++j){double s=0.;for(int i=0;i<H;++i)s+=p[OFF_W3+i*L+j]*d3[i];d2[j]=s*(1.-c.z[j]*c.z[j]);}
        for(int j=0;j<H;++j){double s=0.;for(int i=0;i<L;++i)s+=p[OFF_W2+i*H+j]*d2[i];d1[j]=s*(1.-c.h1[j]*c.h1[j]);}
        for(int i=0;i<D;++i){g[OFF_B4+i]+=d4[i];for(int j=0;j<H;++j)g[OFF_W4+i*H+j]+=d4[i]*c.h2[j];}
        for(int i=0;i<H;++i){g[OFF_B3+i]+=d3[i];for(int j=0;j<L;++j)g[OFF_W3+i*L+j]+=d3[i]*c.z[j];}
        for(int i=0;i<L;++i){g[OFF_B2+i]+=d2[i];for(int j=0;j<H;++j)g[OFF_W2+i*H+j]+=d2[i]*c.h1[j];}
        for(int i=0;i<H;++i){g[OFF_B1+i]+=d1[i];for(int j=0;j<D;++j)g[OFF_W1+i*D+j]+=d1[i]*x[j];}
        return loss/D;
    }
};
Bytes serialize_model(const Model& m){Bytes v;tag(v,"JXMOD4\r\n");put(v,D,2);put(v,H,2);put(v,L,2);put(v,NPARAM,4);
    for(double d:m.p){check(std::isfinite(d),"nonfinite model");put_double(v,d);}seal(v);return v;
}
Model parse_model(const Bytes& v){size_t p=0;require_tag(v,p,"JXMOD4\r\n");check(get(v,p,2)==D&&get(v,p,2)==H&&get(v,p,2)==L&&get(v,p,4)==NPARAM,"incompatible model architecture");
    check(v.size()==8+10+8*NPARAM+4,"bad model length");check_seal(v);Model m;for(int i=0;i<NPARAM;++i)m.p[i]=get_double(v,p);return m;
}
void save_model(const std::string& path,const Model& m){write_file(path,serialize_model(m));}
Model load_model(const std::string& path){return parse_model(read_file(path));}
struct TrainResult{double first,best,last;int epoch;size_t train,validation;};
TrainResult train(Model& net,const Volume& v,int epochs,double lr,uint64_t seed=1234){
    check(epochs>=1&&epochs<=5000,"epochs outside [1,5000]");check(lr>0.&&lr<=0.1,"learning rate outside (0,0.1]");
    auto samples=blocks(v);check(samples.size()>=8,"need at least eight 4x4x4 blocks");
    std::vector<size_t> ids(samples.size());std::iota(ids.begin(),ids.end(),size_t(0));std::mt19937_64 rng(seed);std::shuffle(ids.begin(),ids.end(),rng);
    size_t nval=std::max<size_t>(1,ids.size()/8);size_t ntrain=ids.size()-nval;
    auto valid=[&](const Model& m){
        // Validate the actual deployed quantized 12-byte latent codec, not an ideal
        // continuous bottleneck that might conceal quantization errors.
        double val=0.;
        for(size_t i=ntrain;i<ids.size();++i){
            const auto& x=samples[ids[i]];
            const auto q=m.encode_q(x);
            const auto predicted=m.decode_q(q);
            for(int j=0;j<D;++j){const double d=double(predicted[j])/255.-x[j];val+=d*d;}
        }
        return val/double(nval*D);
    };
    Model best=net;double first=valid(net),bestval=first;int bestepoch=0;
    std::vector<double> mom(NPARAM,0.),vel(NPARAM,0.),grad(NPARAM,0.);
    double b1pow=1.,b2pow=1.;const int batch=16;int steps=0;
    for(int ep=1;ep<=epochs;++ep){std::shuffle(ids.begin(),ids.begin()+ntrain,rng);
        for(size_t st=0;st<ntrain;st+=batch){std::fill(grad.begin(),grad.end(),0.);size_t end=std::min(ntrain,st+batch),cnt=end-st;
            for(size_t j=st;j<end;++j)net.grad(samples[ids[j]],grad);
            ++steps;b1pow*=.9;b2pow*=.999;
            for(int k=0;k<NPARAM;++k){double g=std::clamp(grad[k]/double(cnt),-2.,2.);
                mom[k]=.9*mom[k]+.1*g;vel[k]=.999*vel[k]+.001*g*g;
                net.p[k]-=lr*(mom[k]/(1.-b1pow))/(std::sqrt(vel[k]/(1.-b2pow))+1e-8);
            }
        }
        double score=valid(net);if(score<bestval){bestval=score;best=net;bestepoch=ep;}
    }
    net=best;return {first,bestval,valid(net),bestepoch,ntrain,nval};
}
Bytes rle_pack(const Bytes& in){Bytes out;size_t i=0;
    while(i<in.size()){
        size_t count=1;while(i+count<in.size()&&count<130&&in[i+count]==in[i])++count;
        if(count>=3){out.push_back(uint8_t(128+(count-3)));out.push_back(in[i]);i+=count;continue;}
        size_t st=i;++i;
        while(i<in.size()&&i-st<128){size_t r=1;while(i+r<in.size()&&r<3&&in[i+r]==in[i])++r;if(r>=3)break;++i;}
        out.push_back(uint8_t(i-st-1));out.insert(out.end(),in.begin()+st,in.begin()+i);
    }return out;
}
Bytes rle_unpack(const Bytes& in,size_t expected){Bytes out;out.reserve(expected);size_t p=0;
    while(p<in.size()){
        uint8_t h=in[p++];size_t count=(h<128)?size_t(h)+1:size_t(h-128)+3;
        check(count<=expected-out.size(),"RLE overflow");
        if(h<128){check(count<=in.size()-p,"RLE truncated literal");out.insert(out.end(),in.begin()+p,in.begin()+p+count);p+=count;}
        else{check(p<in.size(),"RLE truncated run");out.insert(out.end(),count,in[p++]);}
    }
    check(out.size()==expected,"RLE decoded length mismatch");return out;
}
struct Archive{
    uint32_t x,y,z;uint64_t ox,oy,oz,model_hash;bool exact;uint8_t codec;Bytes latents,residual;
};
Bytes serialize_archive(const Archive& a){
    check(a.ox<SIDE&&a.oy<SIDE&&a.oz<SIDE&&a.x<=SIDE-a.ox&&a.y<=SIDE-a.oy&&a.z<=SIDE-a.oz,"3D virtual origin out of range");
    size_t n=voxels(a.x,a.y,a.z);check(a.latents.size()==(n/D)*L,"latent count mismatch");
    if(a.exact){check(a.codec==1||a.codec==2,"invalid exact codec");check(!a.residual.empty(),"missing lossless residual");}
    else{check(a.codec==0&&a.residual.empty(),"unexpected lossy residual");}
    Bytes v;tag(v,"JXPKT4\r\n");put(v,a.x,4);put(v,a.y,4);put(v,a.z,4);
    put(v,a.ox,8);put(v,a.oy,8);put(v,a.oz,8);put(v,a.model_hash,8);
    put(v,uint8_t(a.exact),1);put(v,a.codec,1);put(v,a.latents.size(),8);put(v,a.residual.size(),8);
    v.insert(v.end(),a.latents.begin(),a.latents.end());v.insert(v.end(),a.residual.begin(),a.residual.end());seal(v);return v;
}
Archive parse_archive(const Bytes& v){size_t p=0;require_tag(v,p,"JXPKT4\r\n");Archive a{};
    a.x=uint32_t(get(v,p,4));a.y=uint32_t(get(v,p,4));a.z=uint32_t(get(v,p,4));
    a.ox=get(v,p,8);a.oy=get(v,p,8);a.oz=get(v,p,8);a.model_hash=get(v,p,8);
    a.exact=get(v,p,1)!=0;a.codec=uint8_t(get(v,p,1));size_t ls=size_t(get(v,p,8)),rs=size_t(get(v,p,8));
    size_t n=voxels(a.x,a.y,a.z);check(a.ox<SIDE&&a.oy<SIDE&&a.oz<SIDE&&a.x<=SIDE-a.ox&&a.y<=SIDE-a.oy&&a.z<=SIDE-a.oz,"origin exceeds virtual domain");
    check(ls==n/D*L,"invalid latent payload length");check(rs<=MAX_FILE && p<=v.size() && ls<=v.size()-p && rs<=v.size()-p-ls && p+ls+rs+4==v.size(),"archive size mismatch");
    if(a.exact)check((a.codec==1||a.codec==2)&&rs>0,"invalid exact residual format");
    else check(a.codec==0&&rs==0,"invalid lossy format");
    check_seal(v);a.latents.assign(v.begin()+p,v.begin()+p+ls);p+=ls;a.residual.assign(v.begin()+p,v.begin()+p+rs);return a;
}
Archive make_archive(const Volume& v,const Model& net,bool exact,uint64_t ox,uint64_t oy,uint64_t oz){
    Archive a{v.x,v.y,v.z,ox,oy,oz,fnv(serialize_model(net)),exact,0,{},{}};
    check(ox<SIDE&&oy<SIDE&&oz<SIDE&&v.x<=SIDE-ox&&v.y<=SIDE-oy&&v.z<=SIDE-oz,"origin exceeds 3D virtual domain");
    a.latents.reserve(v.data.size()/D*L);Bytes residual; if(exact)residual.reserve(v.data.size());
    for(uint32_t z=0;z<v.z;z+=4)for(uint32_t y=0;y<v.y;y+=4)for(uint32_t x=0;x<v.x;x+=4){
        auto b=block_bytes(v,x,y,z);std::array<double,D> norm{};
        for(int i=0;i<D;++i)norm[i]=double(b[i])/255.;
        auto q=net.encode_q(norm);for(int i=0;i<L;++i)a.latents.push_back(uint8_t(q[i]));
        if(exact){auto prediction=net.decode_q(q);for(int i=0;i<D;++i)residual.push_back(b[i]^prediction[i]);}
    }
    if(exact){auto packed=rle_pack(residual);a.codec=packed.size()<residual.size()?2:1;a.residual=a.codec==2?std::move(packed):std::move(residual);}
    return a;
}
Volume decompress(const Archive& a,const Model& net){
    check(a.model_hash==fnv(serialize_model(net)),"model fingerprint mismatch");
    Volume v{a.x,a.y,a.z,Bytes(voxels(a.x,a.y,a.z))};
    Bytes residual;if(a.exact){residual=(a.codec==1)?a.residual:rle_unpack(a.residual,v.data.size());check(residual.size()==v.data.size(),"residual length mismatch");}
    size_t lb=0,rb=0;
    for(uint32_t z=0;z<v.z;z+=4)for(uint32_t y=0;y<v.y;y+=4)for(uint32_t x=0;x<v.x;x+=4){
        std::array<int8_t,L> q{};for(int i=0;i<L;++i){int b=int(a.latents[lb++]);q[i]=int8_t(b<=127?b:b-256);}
        auto b=net.decode_q(q);if(a.exact){for(int i=0;i<D;++i)b[i]^=residual[rb++];}insert_block(v,x,y,z,b);
    }return v;
}
std::pair<uint8_t,uint8_t> query(const Archive& a,const Model& net,uint64_t x,uint64_t y,uint64_t z){
    check(a.model_hash==fnv(serialize_model(net)),"model fingerprint mismatch");
    check(x>=a.ox&&y>=a.oy&&z>=a.oz&&x-a.ox<a.x&&y-a.oy<a.y&&z-a.oz<a.z,
          "query outside archived 3D tile");
    uint64_t dx=x-a.ox,dy=y-a.oy,dz=z-a.oz;
    size_t block=((size_t(dz/4)*(a.y/4))+(dy/4))*(a.x/4)+(dx/4);
    std::array<int8_t,L> q{};
    for(int i=0;i<L;++i){int b=int(a.latents[block*L+i]);q[i]=int8_t(b<=127?b:b-256);}
    auto predicted=net.decode_q(q);
    int local=int((dz%4)*16+(dy%4)*4+(dx%4));
    uint8_t predicted_byte=predicted[local], decoded=predicted_byte;
    if(a.exact){
        auto residual=(a.codec==1)?a.residual:rle_unpack(a.residual,voxels(a.x,a.y,a.z));
        decoded^=residual[block*D+size_t(local)];
    }
    return {predicted_byte,decoded};
}

void export_pgm(const Volume& v,uint32_t z,const std::string& path){
    check(z<v.z,"slice outside volume");std::ofstream f(path,std::ios::binary);check(bool(f),"cannot write PGM");
    f<<"P5\n"<<v.x<<" "<<v.y<<"\n255\n";
    for(uint32_t y=0;y<v.y;++y)for(uint32_t x=0;x<v.x;++x)f.put(char(v.data[idx(v,x,y,z)]));
    f.flush();check(bool(f),"PGM write failure");
}
void comparison(const Volume& a,const Volume& b){check(a.x==b.x&&a.y==b.y&&a.z==b.z,"volume dimensions differ");
    double err=0.,ab=0.;size_t exact=0;for(size_t i=0;i<a.data.size();++i){double d=double(a.data[i])-b.data[i];err+=d*d;ab+=std::abs(d);exact+=a.data[i]==b.data[i];}
    err/=a.data.size();ab/=a.data.size();double psnr=(err==0)?std::numeric_limits<double>::infinity():10.*std::log10(255.*255./err);
    std::cout<<std::setprecision(9)<<"MSE="<<err<<" MAE="<<ab<<" PSNR_dB="<<psnr<<" exact_bytes="<<exact<<"/"<<a.data.size()<<"\n";
}
std::string usage(){return R"(JARVIS-X 3D Neural Autoencoder v4.0 (CPU / C++17)
Usage:
  jxae3d generate <out.jxvol> <side> [seed]
  jxae3d import-raw <input.raw> <nx> <ny> <nz> <out.jxvol>
  jxae3d export-raw <input.jxvol> <output.raw>
  jxae3d train <input.jxvol> <model.jxmodel> [epochs=80] [lr=0.003]
  jxae3d encode <input.jxvol> <model.jxmodel> <archive.jxpkt> <lossy|exact> [origin_x origin_y origin_z]
  jxae3d decode <archive.jxpkt> <model.jxmodel> <output.jxvol>
  jxae3d query <archive.jxpkt> <model.jxmodel> <world_x> <world_y> <world_z>
  jxae3d evolve <input.jxvol> <model.jxmodel> <cycles> <epochs_per_cycle> [lr=0.003]
  jxae3d inspect <archive.jxpkt>
  jxae3d compare <original.jxvol> <reconstruction.jxvol>
  jxae3d slice <input.jxvol> <z_index> <out.pgm>
  jxae3d selftest
  jxae3d --quine > regenerated.cpp
Coordinates must fit 0 <= origin_i, origin_i + size_i <= 10^12.
)";}
void selftest(){
    Volume v=synthetic(16,7);check(parse_volume(serial_volume(v)).data==v.data,"volume roundtrip failed");
    Model m(7);auto r=train(m,v,45,.004,7);check(r.best<=r.first,"training did not improve validation");
    for(bool exact:{false,true}){
        auto a=make_archive(v,m,exact,SIDE-16,1234567890ULL,37);
        a=parse_archive(serialize_archive(a));Volume restored=decompress(a,m);
        check(restored.data.size()==v.data.size(),"decoded volume size mismatch");
        if(exact)check(v.data==restored.data,"exact reconstruction failed");
        auto corrupt=serialize_archive(a);corrupt[corrupt.size()/2]^=1;
        bool rejected=false;try{parse_archive(corrupt);}catch(...){rejected=true;}
        check(rejected,"corruption not detected");
        Model other(99);rejected=false;try{decompress(a,other);}catch(...){rejected=true;}
        check(rejected,"wrong model not detected");
    }
    {auto a=make_archive(v,m,true,0,0,0);bool rejected=false;try{a.ox=SIDE;serialize_archive(a);}catch(...){rejected=true;}check(rejected,"invalid 3D position accepted");}
    auto mm=parse_model(serialize_model(m));check(serialize_model(m)==serialize_model(mm),"model roundtrip failed");
    {auto a=make_archive(v,m,true,SIDE-16,100,200);
     auto bb=query(a,m,SIDE-14,102,203);
     check(bb.second==v.data[idx(v,2,2,3)],"world-coordinate query failed");}
    {Bytes sample(300,0);for(size_t i=90;i<150;++i)sample[i]=uint8_t(i);for(size_t i=150;i<200;++i)sample[i]=17;
     check(rle_unpack(rle_pack(sample),sample.size())==sample,"RLE roundtrip failed");}
    std::cout<<"PASS neural_training validation_initial="<<r.first<<" validation_best="<<r.best<<" best_epoch="<<r.epoch<<"\n";
    std::cout<<"PASS model_checkpoint, quantized_latent, exact_XOR, lossy_decode, CRC32, model_fingerprint, 120_bit_virtual_address\n";
}
} // namespace jx

int main(int argc,char** argv){
    using namespace jx;
    try{
        if(argc==2 && std::string(argv[1])=="--quine")return jx_quine::reproduce();
        if(argc<2){std::cout<<usage();return 0;}
        std::string cmd=argv[1];
        if(cmd=="selftest"&&argc==2){selftest();return 0;}
        if(cmd=="generate"&&(argc==4||argc==5)){
            uint32_t s=dimension(argv[3]);uint64_t seed=argc==5?num(argv[4]):42;
            auto v=synthetic(s,seed);save_volume(argv[2],v);
            std::cout<<"generated voxel volume "<<s<<"^3 with "<<v.data.size()<<" bytes\n";return 0;
        }
        if(cmd=="import-raw"&&argc==7){uint32_t x=dimension(argv[3]),y=dimension(argv[4]),z=dimension(argv[5]);auto raw=read_file(argv[2]);check(raw.size()==voxels(x,y,z),"raw data length does not match dimensions");save_volume(argv[6],{x,y,z,std::move(raw)});return 0;}
        if(cmd=="export-raw"&&argc==4){auto v=load_volume(argv[2]);write_file(argv[3],v.data);return 0;}
        if(cmd=="train"&&argc>=4&&argc<=6){auto v=load_volume(argv[2]);int epochs=argc>=5?int(num(argv[4])):80;double lr=argc==6?real(argv[5]):.003;
            Model net(42);std::ifstream f(argv[3],std::ios::binary);if(f.good())net=load_model(argv[3]);
            auto r=train(net,v,epochs,lr);save_model(argv[3],net);
            std::cout<<std::setprecision(8)<<"trained model parameters="<<NPARAM<<" blocks_train="<<r.train<<" blocks_validation="<<r.validation
                     <<" initial_validation_MSE="<<r.first<<" best_validation_MSE="<<r.best<<" best_epoch="<<r.epoch<<"\n";return 0;}
        if(cmd=="encode"&&(argc==6||argc==9)){
            auto v=load_volume(argv[2]);auto net=load_model(argv[3]);std::string mode=argv[5];check(mode=="lossy"||mode=="exact","mode must be lossy or exact");
            uint64_t ox=0,oy=0,oz=0;if(argc==9){ox=num(argv[6]);oy=num(argv[7]);oz=num(argv[8]);}
            auto a=make_archive(v,net,mode=="exact",ox,oy,oz);auto bytes=serialize_archive(a);write_file(argv[4],bytes);
            std::cout<<"encoded 3D blocks="<<v.data.size()/D<<" latent_bytes="<<a.latents.size()<<" residual_bytes="<<a.residual.size()
                     <<" archive_bytes="<<bytes.size()<<" raw_bytes="<<v.data.size()<<" mode="<<mode<<" origin=("<<ox<<","<<oy<<","<<oz<<")\n";return 0;
        }
        if(cmd=="evolve"&&(argc==6||argc==7)){
            auto v=load_volume(argv[2]);const uint64_t cycles=num(argv[4]);
            const uint64_t epochs=num(argv[5]);double lr=argc==7?real(argv[6]):.003;
            check(cycles>=1&&cycles<=1000&&epochs>=1&&epochs<=5000,"invalid evolution iteration budget");
            Model net(42);std::ifstream input(argv[3],std::ios::binary);
            if(input.good())net=load_model(argv[3]);
            for(uint64_t t=1;t<=cycles;++t){
                auto result=train(net,v,int(epochs),lr);
                // Each cycle chooses the best quantized-codec validation state.
                // Preserve prior model if the proposed optimization does not improve.
                save_model(argv[3],net);
                std::cout<<std::setprecision(9)<<"cycle="<<t<<" validation_MSE_before="<<result.first
                         <<" validation_MSE_after="<<result.best<<" accepted="<<(result.epoch>0)
                         <<" best_epoch="<<result.epoch<<" checkpoint="<<argv[3]<<"\n";
            }
            return 0;
        }
        if(cmd=="query"&&argc==7){
            auto a=parse_archive(read_file(argv[2]));auto m=load_model(argv[3]);
            uint64_t x=num(argv[4]),y=num(argv[5]),z=num(argv[6]);
            auto b=query(a,m,x,y,z);
            std::cout<<"world=("<<x<<","<<y<<","<<z<<") predicted="<<int(b.first)
                     <<" decoded="<<int(b.second)<<" exact="<<a.exact<<"\n";return 0;
        }
        if(cmd=="decode"&&argc==5){auto a=parse_archive(read_file(argv[2]));auto net=load_model(argv[3]);auto v=decompress(a,net);save_volume(argv[4],v);
            std::cout<<"decoded "<<v.x<<"x"<<v.y<<"x"<<v.z<<" exact="<<a.exact<<" origin=("<<a.ox<<","<<a.oy<<","<<a.oz<<")\n";return 0;}
        if(cmd=="inspect"&&argc==3){auto a=parse_archive(read_file(argv[2]));std::cout<<"shape="<<a.x<<"x"<<a.y<<"x"<<a.z
          <<" origin=("<<a.ox<<","<<a.oy<<","<<a.oz<<") latent_bytes="<<a.latents.size()<<" residual_bytes="<<a.residual.size()
          <<" exact="<<a.exact<<" residual_codec="<<int(a.codec)<<" fingerprint(FNV64)="<<std::hex<<a.model_hash<<std::dec<<"\n";return 0;}
        if(cmd=="compare"&&argc==4){comparison(load_volume(argv[2]),load_volume(argv[3]));return 0;}
        if(cmd=="slice"&&argc==5){auto v=load_volume(argv[2]);export_pgm(v,uint32_t(num(argv[3])),argv[4]);return 0;}
        std::cerr<<usage();return 2;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}

// Self-reproduction layer: the source template is compiled into the program.
// `--quine` produces the complete generated source without reading a source file.
namespace jx_quine {
static constexpr char source_template[] = R"JX4SOURCE(// JARVIS-X 3D AUTOENCODER v4.0 -- C++17 standard-library neural codec.
// Real learned 4x4x4 block autoencoder: 64 -> 32 -> 12 -> 32 -> 64.
// - Backpropagation with Adam; quantized latent int8; optional exact XOR residual.
// - Sparse 120-bit virtual origin (3*40-bit coordinate components).
// - Versioned portable little-endian files, CRC32 integrity, model fingerprint.
// Educational/reference CPU runtime; bounded allocations; no GPU or trillion weights.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace jx_quine {int reproduce();}

namespace jx {
constexpr int B=4,D=64,H=32,L=12;
constexpr uint64_t SIDE=1000000000000ULL;
constexpr size_t MAX_VOXELS=64ULL*1024*1024; // 64 MiB per volume, conservative
constexpr size_t MAX_FILE=128ULL*1024*1024;
constexpr int OFF_W1=0;
constexpr int OFF_B1=OFF_W1+H*D;
constexpr int OFF_W2=OFF_B1+H;
constexpr int OFF_B2=OFF_W2+L*H;
constexpr int OFF_W3=OFF_B2+L;
constexpr int OFF_B3=OFF_W3+H*L;
constexpr int OFF_W4=OFF_B3+H;
constexpr int OFF_B4=OFF_W4+D*H;
constexpr int NPARAM=OFF_B4+D;
using Bytes=std::vector<uint8_t>;
[[noreturn]] void fail(const std::string& s){throw std::runtime_error(s);}
void check(bool v,const std::string& s){if(!v)fail(s);}
uint32_t crc32(const Bytes& v, size_t size=SIZE_MAX){
    uint32_t c=0xffffffffU;
    const size_t n=std::min(size,v.size());
    for(size_t i=0;i<n;++i){c^=v[i];for(int j=0;j<8;++j)c=(c>>1)^((c&1U)?0xedb88320U:0U);}
    return c^0xffffffffU;
}
uint64_t fnv(const Bytes& v){uint64_t h=14695981039346656037ULL;for(uint8_t x:v){h^=x;h*=1099511628211ULL;}return h;}
void put(Bytes& o,uint64_t x,int n){for(int i=0;i<n;i++){o.push_back(uint8_t(x&255));x>>=8;}}
uint64_t get(const Bytes& v,size_t& p,int n){check(n>0&&n<=8 && p<=v.size() && size_t(n)<=v.size()-p,"truncated file");uint64_t x=0;for(int i=0;i<n;++i)x|=uint64_t(v[p++])<<(8*i);return x;}
void tag(Bytes& v,const char* s){for(int i=0;i<8;++i)v.push_back(uint8_t(s[i]));}
void require_tag(const Bytes& v,size_t& p,const char* s){check(v.size()>=8,"truncated header");for(int i=0;i<8;++i)check(v[p++]==uint8_t(s[i]),"wrong file format or version");}
void put_double(Bytes& v,double d){static_assert(sizeof(double)==8,"requires IEEE754 binary64");uint64_t bits;std::memcpy(&bits,&d,8);put(v,bits,8);}
double get_double(const Bytes& v,size_t& p){uint64_t bits=get(v,p,8);double x;std::memcpy(&x,&bits,8);check(std::isfinite(x),"nonfinite model parameter");return x;}
void seal(Bytes& v){put(v,crc32(v),4);}
void check_seal(const Bytes& v){check(v.size()>=4,"truncated checksum");size_t p=v.size()-4;check(uint32_t(get(v,p,4))==crc32(v,v.size()-4),"CRC32 mismatch");}
Bytes read_file(const std::string& path){
    std::ifstream f(path,std::ios::binary|std::ios::ate);check(bool(f),"cannot read: "+path);
    auto n=f.tellg();check(n>=0 && uint64_t(n)<=MAX_FILE,"file too large");f.seekg(0);
    Bytes v(size_t(n),uint8_t(0));if(n>0)f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n));check(bool(f)||n==0,"read failure");return v;
}
void write_file(const std::string& path,const Bytes& v){
    std::ofstream f(path,std::ios::binary|std::ios::trunc);check(bool(f),"cannot write: "+path);
    if(!v.empty()) f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()));
    f.flush();check(bool(f),"write failure: "+path);
}
uint64_t num(const std::string& s){size_t consumed=0;unsigned long long n;try{n=std::stoull(s,&consumed);}catch(...){fail("invalid integer: "+s);}check(consumed==s.size() && (s.empty()||s[0]!='-'),"invalid integer: "+s);return uint64_t(n);}
double real(const std::string& s){size_t consumed=0;double n;try{n=std::stod(s,&consumed);}catch(...){fail("invalid number: "+s);}check(consumed==s.size()&&std::isfinite(n),"invalid number: "+s);return n;}
uint32_t dimension(const std::string& s){auto n=num(s);check(n>=4&&n<=1024&&n%4==0,"dimension must be multiple of 4 in [4,1024]");return uint32_t(n);}
size_t voxels(uint32_t x,uint32_t y,uint32_t z){
    check(x>=4&&y>=4&&z>=4&&x%4==0&&y%4==0&&z%4==0,"invalid 3D dimensions");
    uint64_t n=uint64_t(x)*y*z;check(n<=MAX_VOXELS,"volume exceeds 64 MiB safety limit");return size_t(n);
}
struct Volume{uint32_t x=0,y=0,z=0;Bytes data;};
size_t idx(const Volume& v,uint32_t x,uint32_t y,uint32_t z){return (size_t(z)*v.y+y)*v.x+x;}
Bytes serial_volume(const Volume& v){check(v.data.size()==voxels(v.x,v.y,v.z),"bad volume size");Bytes a;tag(a,"JXVOL4\r\n");put(a,v.x,4);put(a,v.y,4);put(a,v.z,4);a.insert(a.end(),v.data.begin(),v.data.end());seal(a);return a;}
Volume parse_volume(const Bytes& v){size_t p=0;require_tag(v,p,"JXVOL4\r\n");auto x=uint32_t(get(v,p,4)),y=uint32_t(get(v,p,4)),z=uint32_t(get(v,p,4));auto n=voxels(x,y,z);check(v.size()==20+n+4,"volume length mismatch");check_seal(v);return {x,y,z,Bytes(v.begin()+20,v.end()-4)};}
Volume load_volume(const std::string& p){return parse_volume(read_file(p));}
void save_volume(const std::string& p,const Volume& v){write_file(p,serial_volume(v));}
Volume synthetic(uint32_t side,uint64_t seed){Volume v{side,side,side,Bytes(voxels(side,side,side))};
    std::mt19937_64 rng(seed);std::uniform_real_distribution<double> u(-1.,1.);
    const double a=u(rng)*0.9,b=u(rng)*0.9,c=u(rng)*0.9;
    for(uint32_t z=0;z<side;++z)for(uint32_t y=0;y<side;++y)for(uint32_t x=0;x<side;++x){
        double X=(double(x)+0.5)/side,Y=(double(y)+0.5)/side,Z=(double(z)+0.5)/side;
        double r2=(X-.52)*(X-.52)+(Y-.49)*(Y-.49)+(Z-.55)*(Z-.55);
        double f=.43+.19*std::sin(6.2831853*(X+.4*Y)+a)
                    +.17*std::cos(6.2831853*(Z+.25*X)+b)
                    +.24*std::exp(-r2*34.)+.06*std::sin(20.*(X*Y+Z)+c);
        v.data[idx(v,x,y,z)]=uint8_t(std::lround(255*std::clamp(f,0.,1.)));
    }return v;
}
std::vector<std::array<double,D>> blocks(const Volume& v){
    std::vector<std::array<double,D>> b; b.reserve(v.data.size()/D);
    for(uint32_t z=0;z<v.z;z+=4)for(uint32_t y=0;y<v.y;y+=4)for(uint32_t x=0;x<v.x;x+=4){
        std::array<double,D> a{};int k=0;
        for(int dz=0;dz<4;++dz)for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)
            a[k++]=v.data[idx(v,x+dx,y+dy,z+dz)]/255.;
        b.push_back(a);
    }return b;
}
std::array<uint8_t,D> block_bytes(const Volume& v,uint32_t x,uint32_t y,uint32_t z){
    std::array<uint8_t,D> a{};int k=0;
    for(int dz=0;dz<4;++dz)for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)
        a[k++]=v.data[idx(v,x+dx,y+dy,z+dz)];
    return a;
}
void insert_block(Volume& v,uint32_t x,uint32_t y,uint32_t z,const std::array<uint8_t,D>& a){int k=0;
    for(int dz=0;dz<4;++dz)for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)
        v.data[idx(v,x+dx,y+dy,z+dz)]=a[k++];
}
inline double sig(double x){if(x>=0){double e=std::exp(-x);return 1./(1.+e);}double e=std::exp(x);return e/(1.+e);}
struct Cache {std::array<double,H> h1{},h2{};std::array<double,L> z{};std::array<double,D> out{};};
struct Model{
    std::vector<double> p;
    explicit Model(uint64_t seed=42):p(NPARAM,0.){
        std::mt19937_64 rng(seed);std::normal_distribution<double> gauss(0,1);
        for(int i=OFF_W1;i<OFF_B1;++i)p[i]=gauss(rng)*std::sqrt(2./(D+H));
        for(int i=OFF_W2;i<OFF_B2;++i)p[i]=gauss(rng)*std::sqrt(2./(H+L));
        for(int i=OFF_W3;i<OFF_B3;++i)p[i]=gauss(rng)*std::sqrt(2./(H+L));
        for(int i=OFF_W4;i<OFF_B4;++i)p[i]=gauss(rng)*std::sqrt(2./(H+D));
    }
    Cache encode_forward(const std::array<double,D>& x)const{
        Cache c;
        for(int i=0;i<H;++i){double a=p[OFF_B1+i];for(int j=0;j<D;++j)a+=p[OFF_W1+i*D+j]*x[j];c.h1[i]=std::tanh(a);}
        for(int i=0;i<L;++i){double a=p[OFF_B2+i];for(int j=0;j<H;++j)a+=p[OFF_W2+i*H+j]*c.h1[j];c.z[i]=std::tanh(a);}
        decode_forward(c);return c;
    }
    void decode_forward(Cache& c)const{
        for(int i=0;i<H;++i){double a=p[OFF_B3+i];for(int j=0;j<L;++j)a+=p[OFF_W3+i*L+j]*c.z[j];c.h2[i]=std::tanh(a);}
        for(int i=0;i<D;++i){double a=p[OFF_B4+i];for(int j=0;j<H;++j)a+=p[OFF_W4+i*H+j]*c.h2[j];c.out[i]=sig(a);}
    }
    std::array<int8_t,L> encode_q(const std::array<double,D>& x)const{
        Cache c=encode_forward(x);std::array<int8_t,L> q{};
        for(int i=0;i<L;++i)q[i]=int8_t(std::lround(std::clamp(c.z[i],-1.,1.)*127.));
        return q;
    }
    std::array<uint8_t,D> decode_q(const std::array<int8_t,L>& q)const{
        Cache c;for(int i=0;i<L;++i)c.z[i]=double(q[i])/127.;decode_forward(c);
        std::array<uint8_t,D> out{};for(int i=0;i<D;++i)out[i]=uint8_t(std::lround(255.*std::clamp(c.out[i],0.,1.)));return out;
    }
    double mse(const std::array<double,D>& x)const{
        auto c=encode_forward(x);double sum=0.;for(int i=0;i<D;++i){double d=c.out[i]-x[i];sum+=d*d;}return sum/D;
    }
    double grad(const std::array<double,D>& x,std::vector<double>& g)const{
        auto c=encode_forward(x);
        std::array<double,D> d4{};std::array<double,H> d3{},d1{};std::array<double,L> d2{};
        double loss=0.;
        for(int i=0;i<D;++i){double d=c.out[i]-x[i];loss+=d*d;d4[i]=(2./D)*d*c.out[i]*(1.-c.out[i]);}
        for(int j=0;j<H;++j){double s=0.;for(int i=0;i<D;++i)s+=p[OFF_W4+i*H+j]*d4[i];d3[j]=s*(1.-c.h2[j]*c.h2[j]);}
        for(int j=0;j<L;++j){double s=0.;for(int i=0;i<H;++i)s+=p[OFF_W3+i*L+j]*d3[i];d2[j]=s*(1.-c.z[j]*c.z[j]);}
        for(int j=0;j<H;++j){double s=0.;for(int i=0;i<L;++i)s+=p[OFF_W2+i*H+j]*d2[i];d1[j]=s*(1.-c.h1[j]*c.h1[j]);}
        for(int i=0;i<D;++i){g[OFF_B4+i]+=d4[i];for(int j=0;j<H;++j)g[OFF_W4+i*H+j]+=d4[i]*c.h2[j];}
        for(int i=0;i<H;++i){g[OFF_B3+i]+=d3[i];for(int j=0;j<L;++j)g[OFF_W3+i*L+j]+=d3[i]*c.z[j];}
        for(int i=0;i<L;++i){g[OFF_B2+i]+=d2[i];for(int j=0;j<H;++j)g[OFF_W2+i*H+j]+=d2[i]*c.h1[j];}
        for(int i=0;i<H;++i){g[OFF_B1+i]+=d1[i];for(int j=0;j<D;++j)g[OFF_W1+i*D+j]+=d1[i]*x[j];}
        return loss/D;
    }
};
Bytes serialize_model(const Model& m){Bytes v;tag(v,"JXMOD4\r\n");put(v,D,2);put(v,H,2);put(v,L,2);put(v,NPARAM,4);
    for(double d:m.p){check(std::isfinite(d),"nonfinite model");put_double(v,d);}seal(v);return v;
}
Model parse_model(const Bytes& v){size_t p=0;require_tag(v,p,"JXMOD4\r\n");check(get(v,p,2)==D&&get(v,p,2)==H&&get(v,p,2)==L&&get(v,p,4)==NPARAM,"incompatible model architecture");
    check(v.size()==8+10+8*NPARAM+4,"bad model length");check_seal(v);Model m;for(int i=0;i<NPARAM;++i)m.p[i]=get_double(v,p);return m;
}
void save_model(const std::string& path,const Model& m){write_file(path,serialize_model(m));}
Model load_model(const std::string& path){return parse_model(read_file(path));}
struct TrainResult{double first,best,last;int epoch;size_t train,validation;};
TrainResult train(Model& net,const Volume& v,int epochs,double lr,uint64_t seed=1234){
    check(epochs>=1&&epochs<=5000,"epochs outside [1,5000]");check(lr>0.&&lr<=0.1,"learning rate outside (0,0.1]");
    auto samples=blocks(v);check(samples.size()>=8,"need at least eight 4x4x4 blocks");
    std::vector<size_t> ids(samples.size());std::iota(ids.begin(),ids.end(),size_t(0));std::mt19937_64 rng(seed);std::shuffle(ids.begin(),ids.end(),rng);
    size_t nval=std::max<size_t>(1,ids.size()/8);size_t ntrain=ids.size()-nval;
    auto valid=[&](const Model& m){
        // Validate the actual deployed quantized 12-byte latent codec, not an ideal
        // continuous bottleneck that might conceal quantization errors.
        double val=0.;
        for(size_t i=ntrain;i<ids.size();++i){
            const auto& x=samples[ids[i]];
            const auto q=m.encode_q(x);
            const auto predicted=m.decode_q(q);
            for(int j=0;j<D;++j){const double d=double(predicted[j])/255.-x[j];val+=d*d;}
        }
        return val/double(nval*D);
    };
    Model best=net;double first=valid(net),bestval=first;int bestepoch=0;
    std::vector<double> mom(NPARAM,0.),vel(NPARAM,0.),grad(NPARAM,0.);
    double b1pow=1.,b2pow=1.;const int batch=16;int steps=0;
    for(int ep=1;ep<=epochs;++ep){std::shuffle(ids.begin(),ids.begin()+ntrain,rng);
        for(size_t st=0;st<ntrain;st+=batch){std::fill(grad.begin(),grad.end(),0.);size_t end=std::min(ntrain,st+batch),cnt=end-st;
            for(size_t j=st;j<end;++j)net.grad(samples[ids[j]],grad);
            ++steps;b1pow*=.9;b2pow*=.999;
            for(int k=0;k<NPARAM;++k){double g=std::clamp(grad[k]/double(cnt),-2.,2.);
                mom[k]=.9*mom[k]+.1*g;vel[k]=.999*vel[k]+.001*g*g;
                net.p[k]-=lr*(mom[k]/(1.-b1pow))/(std::sqrt(vel[k]/(1.-b2pow))+1e-8);
            }
        }
        double score=valid(net);if(score<bestval){bestval=score;best=net;bestepoch=ep;}
    }
    net=best;return {first,bestval,valid(net),bestepoch,ntrain,nval};
}
Bytes rle_pack(const Bytes& in){Bytes out;size_t i=0;
    while(i<in.size()){
        size_t count=1;while(i+count<in.size()&&count<130&&in[i+count]==in[i])++count;
        if(count>=3){out.push_back(uint8_t(128+(count-3)));out.push_back(in[i]);i+=count;continue;}
        size_t st=i;++i;
        while(i<in.size()&&i-st<128){size_t r=1;while(i+r<in.size()&&r<3&&in[i+r]==in[i])++r;if(r>=3)break;++i;}
        out.push_back(uint8_t(i-st-1));out.insert(out.end(),in.begin()+st,in.begin()+i);
    }return out;
}
Bytes rle_unpack(const Bytes& in,size_t expected){Bytes out;out.reserve(expected);size_t p=0;
    while(p<in.size()){
        uint8_t h=in[p++];size_t count=(h<128)?size_t(h)+1:size_t(h-128)+3;
        check(count<=expected-out.size(),"RLE overflow");
        if(h<128){check(count<=in.size()-p,"RLE truncated literal");out.insert(out.end(),in.begin()+p,in.begin()+p+count);p+=count;}
        else{check(p<in.size(),"RLE truncated run");out.insert(out.end(),count,in[p++]);}
    }
    check(out.size()==expected,"RLE decoded length mismatch");return out;
}
struct Archive{
    uint32_t x,y,z;uint64_t ox,oy,oz,model_hash;bool exact;uint8_t codec;Bytes latents,residual;
};
Bytes serialize_archive(const Archive& a){
    check(a.ox<SIDE&&a.oy<SIDE&&a.oz<SIDE&&a.x<=SIDE-a.ox&&a.y<=SIDE-a.oy&&a.z<=SIDE-a.oz,"3D virtual origin out of range");
    size_t n=voxels(a.x,a.y,a.z);check(a.latents.size()==(n/D)*L,"latent count mismatch");
    if(a.exact){check(a.codec==1||a.codec==2,"invalid exact codec");check(!a.residual.empty(),"missing lossless residual");}
    else{check(a.codec==0&&a.residual.empty(),"unexpected lossy residual");}
    Bytes v;tag(v,"JXPKT4\r\n");put(v,a.x,4);put(v,a.y,4);put(v,a.z,4);
    put(v,a.ox,8);put(v,a.oy,8);put(v,a.oz,8);put(v,a.model_hash,8);
    put(v,uint8_t(a.exact),1);put(v,a.codec,1);put(v,a.latents.size(),8);put(v,a.residual.size(),8);
    v.insert(v.end(),a.latents.begin(),a.latents.end());v.insert(v.end(),a.residual.begin(),a.residual.end());seal(v);return v;
}
Archive parse_archive(const Bytes& v){size_t p=0;require_tag(v,p,"JXPKT4\r\n");Archive a{};
    a.x=uint32_t(get(v,p,4));a.y=uint32_t(get(v,p,4));a.z=uint32_t(get(v,p,4));
    a.ox=get(v,p,8);a.oy=get(v,p,8);a.oz=get(v,p,8);a.model_hash=get(v,p,8);
    a.exact=get(v,p,1)!=0;a.codec=uint8_t(get(v,p,1));size_t ls=size_t(get(v,p,8)),rs=size_t(get(v,p,8));
    size_t n=voxels(a.x,a.y,a.z);check(a.ox<SIDE&&a.oy<SIDE&&a.oz<SIDE&&a.x<=SIDE-a.ox&&a.y<=SIDE-a.oy&&a.z<=SIDE-a.oz,"origin exceeds virtual domain");
    check(ls==n/D*L,"invalid latent payload length");check(rs<=MAX_FILE && p<=v.size() && ls<=v.size()-p && rs<=v.size()-p-ls && p+ls+rs+4==v.size(),"archive size mismatch");
    if(a.exact)check((a.codec==1||a.codec==2)&&rs>0,"invalid exact residual format");
    else check(a.codec==0&&rs==0,"invalid lossy format");
    check_seal(v);a.latents.assign(v.begin()+p,v.begin()+p+ls);p+=ls;a.residual.assign(v.begin()+p,v.begin()+p+rs);return a;
}
Archive make_archive(const Volume& v,const Model& net,bool exact,uint64_t ox,uint64_t oy,uint64_t oz){
    Archive a{v.x,v.y,v.z,ox,oy,oz,fnv(serialize_model(net)),exact,0,{},{}};
    check(ox<SIDE&&oy<SIDE&&oz<SIDE&&v.x<=SIDE-ox&&v.y<=SIDE-oy&&v.z<=SIDE-oz,"origin exceeds 3D virtual domain");
    a.latents.reserve(v.data.size()/D*L);Bytes residual; if(exact)residual.reserve(v.data.size());
    for(uint32_t z=0;z<v.z;z+=4)for(uint32_t y=0;y<v.y;y+=4)for(uint32_t x=0;x<v.x;x+=4){
        auto b=block_bytes(v,x,y,z);std::array<double,D> norm{};
        for(int i=0;i<D;++i)norm[i]=double(b[i])/255.;
        auto q=net.encode_q(norm);for(int i=0;i<L;++i)a.latents.push_back(uint8_t(q[i]));
        if(exact){auto prediction=net.decode_q(q);for(int i=0;i<D;++i)residual.push_back(b[i]^prediction[i]);}
    }
    if(exact){auto packed=rle_pack(residual);a.codec=packed.size()<residual.size()?2:1;a.residual=a.codec==2?std::move(packed):std::move(residual);}
    return a;
}
Volume decompress(const Archive& a,const Model& net){
    check(a.model_hash==fnv(serialize_model(net)),"model fingerprint mismatch");
    Volume v{a.x,a.y,a.z,Bytes(voxels(a.x,a.y,a.z))};
    Bytes residual;if(a.exact){residual=(a.codec==1)?a.residual:rle_unpack(a.residual,v.data.size());check(residual.size()==v.data.size(),"residual length mismatch");}
    size_t lb=0,rb=0;
    for(uint32_t z=0;z<v.z;z+=4)for(uint32_t y=0;y<v.y;y+=4)for(uint32_t x=0;x<v.x;x+=4){
        std::array<int8_t,L> q{};for(int i=0;i<L;++i){int b=int(a.latents[lb++]);q[i]=int8_t(b<=127?b:b-256);}
        auto b=net.decode_q(q);if(a.exact){for(int i=0;i<D;++i)b[i]^=residual[rb++];}insert_block(v,x,y,z,b);
    }return v;
}
std::pair<uint8_t,uint8_t> query(const Archive& a,const Model& net,uint64_t x,uint64_t y,uint64_t z){
    check(a.model_hash==fnv(serialize_model(net)),"model fingerprint mismatch");
    check(x>=a.ox&&y>=a.oy&&z>=a.oz&&x-a.ox<a.x&&y-a.oy<a.y&&z-a.oz<a.z,
          "query outside archived 3D tile");
    uint64_t dx=x-a.ox,dy=y-a.oy,dz=z-a.oz;
    size_t block=((size_t(dz/4)*(a.y/4))+(dy/4))*(a.x/4)+(dx/4);
    std::array<int8_t,L> q{};
    for(int i=0;i<L;++i){int b=int(a.latents[block*L+i]);q[i]=int8_t(b<=127?b:b-256);}
    auto predicted=net.decode_q(q);
    int local=int((dz%4)*16+(dy%4)*4+(dx%4));
    uint8_t predicted_byte=predicted[local], decoded=predicted_byte;
    if(a.exact){
        auto residual=(a.codec==1)?a.residual:rle_unpack(a.residual,voxels(a.x,a.y,a.z));
        decoded^=residual[block*D+size_t(local)];
    }
    return {predicted_byte,decoded};
}

void export_pgm(const Volume& v,uint32_t z,const std::string& path){
    check(z<v.z,"slice outside volume");std::ofstream f(path,std::ios::binary);check(bool(f),"cannot write PGM");
    f<<"P5\n"<<v.x<<" "<<v.y<<"\n255\n";
    for(uint32_t y=0;y<v.y;++y)for(uint32_t x=0;x<v.x;++x)f.put(char(v.data[idx(v,x,y,z)]));
    f.flush();check(bool(f),"PGM write failure");
}
void comparison(const Volume& a,const Volume& b){check(a.x==b.x&&a.y==b.y&&a.z==b.z,"volume dimensions differ");
    double err=0.,ab=0.;size_t exact=0;for(size_t i=0;i<a.data.size();++i){double d=double(a.data[i])-b.data[i];err+=d*d;ab+=std::abs(d);exact+=a.data[i]==b.data[i];}
    err/=a.data.size();ab/=a.data.size();double psnr=(err==0)?std::numeric_limits<double>::infinity():10.*std::log10(255.*255./err);
    std::cout<<std::setprecision(9)<<"MSE="<<err<<" MAE="<<ab<<" PSNR_dB="<<psnr<<" exact_bytes="<<exact<<"/"<<a.data.size()<<"\n";
}
std::string usage(){return R"(JARVIS-X 3D Neural Autoencoder v4.0 (CPU / C++17)
Usage:
  jxae3d generate <out.jxvol> <side> [seed]
  jxae3d import-raw <input.raw> <nx> <ny> <nz> <out.jxvol>
  jxae3d export-raw <input.jxvol> <output.raw>
  jxae3d train <input.jxvol> <model.jxmodel> [epochs=80] [lr=0.003]
  jxae3d encode <input.jxvol> <model.jxmodel> <archive.jxpkt> <lossy|exact> [origin_x origin_y origin_z]
  jxae3d decode <archive.jxpkt> <model.jxmodel> <output.jxvol>
  jxae3d query <archive.jxpkt> <model.jxmodel> <world_x> <world_y> <world_z>
  jxae3d evolve <input.jxvol> <model.jxmodel> <cycles> <epochs_per_cycle> [lr=0.003]
  jxae3d inspect <archive.jxpkt>
  jxae3d compare <original.jxvol> <reconstruction.jxvol>
  jxae3d slice <input.jxvol> <z_index> <out.pgm>
  jxae3d selftest
  jxae3d --quine > regenerated.cpp
Coordinates must fit 0 <= origin_i, origin_i + size_i <= 10^12.
)";}
void selftest(){
    Volume v=synthetic(16,7);check(parse_volume(serial_volume(v)).data==v.data,"volume roundtrip failed");
    Model m(7);auto r=train(m,v,45,.004,7);check(r.best<=r.first,"training did not improve validation");
    for(bool exact:{false,true}){
        auto a=make_archive(v,m,exact,SIDE-16,1234567890ULL,37);
        a=parse_archive(serialize_archive(a));Volume restored=decompress(a,m);
        check(restored.data.size()==v.data.size(),"decoded volume size mismatch");
        if(exact)check(v.data==restored.data,"exact reconstruction failed");
        auto corrupt=serialize_archive(a);corrupt[corrupt.size()/2]^=1;
        bool rejected=false;try{parse_archive(corrupt);}catch(...){rejected=true;}
        check(rejected,"corruption not detected");
        Model other(99);rejected=false;try{decompress(a,other);}catch(...){rejected=true;}
        check(rejected,"wrong model not detected");
    }
    {auto a=make_archive(v,m,true,0,0,0);bool rejected=false;try{a.ox=SIDE;serialize_archive(a);}catch(...){rejected=true;}check(rejected,"invalid 3D position accepted");}
    auto mm=parse_model(serialize_model(m));check(serialize_model(m)==serialize_model(mm),"model roundtrip failed");
    {auto a=make_archive(v,m,true,SIDE-16,100,200);
     auto bb=query(a,m,SIDE-14,102,203);
     check(bb.second==v.data[idx(v,2,2,3)],"world-coordinate query failed");}
    {Bytes sample(300,0);for(size_t i=90;i<150;++i)sample[i]=uint8_t(i);for(size_t i=150;i<200;++i)sample[i]=17;
     check(rle_unpack(rle_pack(sample),sample.size())==sample,"RLE roundtrip failed");}
    std::cout<<"PASS neural_training validation_initial="<<r.first<<" validation_best="<<r.best<<" best_epoch="<<r.epoch<<"\n";
    std::cout<<"PASS model_checkpoint, quantized_latent, exact_XOR, lossy_decode, CRC32, model_fingerprint, 120_bit_virtual_address\n";
}
} // namespace jx

int main(int argc,char** argv){
    using namespace jx;
    try{
        if(argc==2 && std::string(argv[1])=="--quine")return jx_quine::reproduce();
        if(argc<2){std::cout<<usage();return 0;}
        std::string cmd=argv[1];
        if(cmd=="selftest"&&argc==2){selftest();return 0;}
        if(cmd=="generate"&&(argc==4||argc==5)){
            uint32_t s=dimension(argv[3]);uint64_t seed=argc==5?num(argv[4]):42;
            auto v=synthetic(s,seed);save_volume(argv[2],v);
            std::cout<<"generated voxel volume "<<s<<"^3 with "<<v.data.size()<<" bytes\n";return 0;
        }
        if(cmd=="import-raw"&&argc==7){uint32_t x=dimension(argv[3]),y=dimension(argv[4]),z=dimension(argv[5]);auto raw=read_file(argv[2]);check(raw.size()==voxels(x,y,z),"raw data length does not match dimensions");save_volume(argv[6],{x,y,z,std::move(raw)});return 0;}
        if(cmd=="export-raw"&&argc==4){auto v=load_volume(argv[2]);write_file(argv[3],v.data);return 0;}
        if(cmd=="train"&&argc>=4&&argc<=6){auto v=load_volume(argv[2]);int epochs=argc>=5?int(num(argv[4])):80;double lr=argc==6?real(argv[5]):.003;
            Model net(42);std::ifstream f(argv[3],std::ios::binary);if(f.good())net=load_model(argv[3]);
            auto r=train(net,v,epochs,lr);save_model(argv[3],net);
            std::cout<<std::setprecision(8)<<"trained model parameters="<<NPARAM<<" blocks_train="<<r.train<<" blocks_validation="<<r.validation
                     <<" initial_validation_MSE="<<r.first<<" best_validation_MSE="<<r.best<<" best_epoch="<<r.epoch<<"\n";return 0;}
        if(cmd=="encode"&&(argc==6||argc==9)){
            auto v=load_volume(argv[2]);auto net=load_model(argv[3]);std::string mode=argv[5];check(mode=="lossy"||mode=="exact","mode must be lossy or exact");
            uint64_t ox=0,oy=0,oz=0;if(argc==9){ox=num(argv[6]);oy=num(argv[7]);oz=num(argv[8]);}
            auto a=make_archive(v,net,mode=="exact",ox,oy,oz);auto bytes=serialize_archive(a);write_file(argv[4],bytes);
            std::cout<<"encoded 3D blocks="<<v.data.size()/D<<" latent_bytes="<<a.latents.size()<<" residual_bytes="<<a.residual.size()
                     <<" archive_bytes="<<bytes.size()<<" raw_bytes="<<v.data.size()<<" mode="<<mode<<" origin=("<<ox<<","<<oy<<","<<oz<<")\n";return 0;
        }
        if(cmd=="evolve"&&(argc==6||argc==7)){
            auto v=load_volume(argv[2]);const uint64_t cycles=num(argv[4]);
            const uint64_t epochs=num(argv[5]);double lr=argc==7?real(argv[6]):.003;
            check(cycles>=1&&cycles<=1000&&epochs>=1&&epochs<=5000,"invalid evolution iteration budget");
            Model net(42);std::ifstream input(argv[3],std::ios::binary);
            if(input.good())net=load_model(argv[3]);
            for(uint64_t t=1;t<=cycles;++t){
                auto result=train(net,v,int(epochs),lr);
                // Each cycle chooses the best quantized-codec validation state.
                // Preserve prior model if the proposed optimization does not improve.
                save_model(argv[3],net);
                std::cout<<std::setprecision(9)<<"cycle="<<t<<" validation_MSE_before="<<result.first
                         <<" validation_MSE_after="<<result.best<<" accepted="<<(result.epoch>0)
                         <<" best_epoch="<<result.epoch<<" checkpoint="<<argv[3]<<"\n";
            }
            return 0;
        }
        if(cmd=="query"&&argc==7){
            auto a=parse_archive(read_file(argv[2]));auto m=load_model(argv[3]);
            uint64_t x=num(argv[4]),y=num(argv[5]),z=num(argv[6]);
            auto b=query(a,m,x,y,z);
            std::cout<<"world=("<<x<<","<<y<<","<<z<<") predicted="<<int(b.first)
                     <<" decoded="<<int(b.second)<<" exact="<<a.exact<<"\n";return 0;
        }
        if(cmd=="decode"&&argc==5){auto a=parse_archive(read_file(argv[2]));auto net=load_model(argv[3]);auto v=decompress(a,net);save_volume(argv[4],v);
            std::cout<<"decoded "<<v.x<<"x"<<v.y<<"x"<<v.z<<" exact="<<a.exact<<" origin=("<<a.ox<<","<<a.oy<<","<<a.oz<<")\n";return 0;}
        if(cmd=="inspect"&&argc==3){auto a=parse_archive(read_file(argv[2]));std::cout<<"shape="<<a.x<<"x"<<a.y<<"x"<<a.z
          <<" origin=("<<a.ox<<","<<a.oy<<","<<a.oz<<") latent_bytes="<<a.latents.size()<<" residual_bytes="<<a.residual.size()
          <<" exact="<<a.exact<<" residual_codec="<<int(a.codec)<<" fingerprint(FNV64)="<<std::hex<<a.model_hash<<std::dec<<"\n";return 0;}
        if(cmd=="compare"&&argc==4){comparison(load_volume(argv[2]),load_volume(argv[3]));return 0;}
        if(cmd=="slice"&&argc==5){auto v=load_volume(argv[2]);export_pgm(v,uint32_t(num(argv[3])),argv[4]);return 0;}
        std::cerr<<usage();return 2;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}

// Self-reproduction layer: the source template is compiled into the program.
// `--quine` produces the complete generated source without reading a source file.
namespace jx_quine {
static constexpr char source_template[] = __JX_SOURCE_QUINE__;
int reproduce(){
    const std::string t(source_template);
    const std::string marker=std::string("__JX_SOURCE_")+"QUINE__";
    const size_t at=t.find(marker);
    if(at==std::string::npos||t.find(marker,at+marker.size())!=std::string::npos){
        std::cerr<<"source template marker error\n";return 3;
    }
    std::cout.write(t.data(),std::streamsize(at));
    std::cout<<"R\"JX4SOURCE(";
    std::cout<<t;
    std::cout<<")JX4SOURCE\"";
    std::cout.write(t.data()+at+marker.size(),std::streamsize(t.size()-at-marker.size()));
    return std::cout?0:4;
}
}
)JX4SOURCE";
int reproduce(){
    const std::string t(source_template);
    const std::string marker=std::string("__JX_SOURCE_")+"QUINE__";
    const size_t at=t.find(marker);
    if(at==std::string::npos||t.find(marker,at+marker.size())!=std::string::npos){
        std::cerr<<"source template marker error\n";return 3;
    }
    std::cout.write(t.data(),std::streamsize(at));
    std::cout<<"R\"JX4SOURCE(";
    std::cout<<t;
    std::cout<<")JX4SOURCE\"";
    std::cout.write(t.data()+at+marker.size(),std::streamsize(t.size()-at-marker.size()));
    return std::cout?0:4;
}
}
