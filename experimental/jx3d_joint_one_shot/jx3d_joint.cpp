// JARVIS X | Jointly trainable 3D autoencoder + verifiable residual codec.
// Reproducible C++17 CPU research experiment. Requires system libzstd.
// The "one-second" goal is measured, not a hard cross-hardware guarantee.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <zstd.h>

namespace jx {
using Clock = std::chrono::steady_clock;
constexpr int N=16, H=N/2, V=N*N*N, C=H*H*H, P=8, K=27;
constexpr int DW=P*(K+1), EW=P, W=EW+DW;
constexpr double PI=3.141592653589793238462643383279502884;
using Volume=std::array<std::uint8_t,V>;
using Coarse=std::array<std::uint8_t,C>;
using Field=std::array<float,V>;
using Latent=std::array<float,C>;
using Grad=std::array<double,W>;
using Vec=std::vector<std::uint8_t>;
int at(int x,int y,int z,int side){return (z*side+y)*side+x;}
int periodic(int x,int side){return (x+side)%side;}
int phase(int ox,int oy,int oz){return ox+2*oy+4*oz;}
float limit(float v){return std::clamp(v,0.0f,1.0f);}
std::uint8_t as_byte(float v){return static_cast<std::uint8_t>(std::lround(255.0f*limit(v)));}
void require(bool x, const char* msg){if(!x)throw std::runtime_error(msg);}
struct PRNG {
    std::uint64_t state;
    explicit PRNG(std::uint64_t s):state(s){}
    std::uint64_t next(){state+=0x9e3779b97f4a7c15ULL;auto z=state;
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
    double u(){return static_cast<double>(next()>>11)*(1.0/9007199254740992.0);}
};
Volume smooth(std::uint64_t seed){
    PRNG rng(seed); std::array<double,6> p{};for(double& a:p)a=2*PI*rng.u();
    Volume v{};
    for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){
        const double a=2*PI/N;
        const double f=.5+.13*std::sin(a*x+p[0])+.11*std::cos(a*y+p[1])+
             .08*std::sin(a*z+p[2])+.07*std::sin(a*(x+y+z)+p[3])+
             .025*std::cos(2*a*(x-y)+p[4])+.022*std::sin(a*(2*y+z)+p[5]);
        v[static_cast<std::size_t>(at(x,y,z,N))]=as_byte(static_cast<float>(f));
    }
    return v;
}
Volume random_data(std::uint64_t seed){PRNG p(seed);Volume v{};for(auto& x:v)x=static_cast<std::uint8_t>(p.next()&255U);return v;}
Volume blocks(std::uint64_t seed){PRNG p(seed);Volume v{};const int off=static_cast<int>(p.next()%5U);
    for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){
        const bool cube=(x>=3+off&&x<10+off&&y>=3&&y<13&&z>=4&&z<12);
        const bool sphere=(x-8)*(x-8)+(y-8)*(y-8)+(z-8)*(z-8)<25;
        v[static_cast<std::size_t>(at(x,y,z,N))]=static_cast<std::uint8_t>(cube?230:(sphere?100:17));
    }
    return v;
}
struct Model{std::array<float,W> w{};};
std::array<float,P> mixing(const Model& m){
    double mx=-1e100;for(int i=0;i<P;++i)mx=std::max(mx,static_cast<double>(m.w[static_cast<std::size_t>(i)]));
    std::array<float,P> p{};double s=0;
    for(int i=0;i<P;++i){p[static_cast<std::size_t>(i)]=static_cast<float>(std::exp(static_cast<double>(m.w[static_cast<std::size_t>(i)])-mx));s+=p[static_cast<std::size_t>(i)];}
    for(auto& a:p)a=static_cast<float>(a/s);
    return p;
}
Coarse encode(const Volume& x,const Model& m){
    Coarse z{};const auto alpha=mixing(m);
    for(int cz=0;cz<H;++cz)for(int cy=0;cy<H;++cy)for(int cx=0;cx<H;++cx){
        float sum=0;
        for(int oz=0;oz<2;++oz)for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
            const int p=phase(ox,oy,oz);const int i=at(2*cx+ox,2*cy+oy,2*cz+oz,N);
            sum+=alpha[static_cast<std::size_t>(p)]*static_cast<float>(x[static_cast<std::size_t>(i)])/255.0f;
        }
        z[static_cast<std::size_t>(at(cx,cy,cz,H))]=as_byte(sum);
    }
    return z;
}
Coarse mean_encode(const Volume& x){
    Coarse z{};for(int cz=0;cz<H;++cz)for(int cy=0;cy<H;++cy)for(int cx=0;cx<H;++cx){
        int sum=0;for(int oz=0;oz<2;++oz)for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox)
            sum+=x[static_cast<std::size_t>(at(2*cx+ox,2*cy+oy,2*cz+oz,N))];
        z[static_cast<std::size_t>(at(cx,cy,cz,H))]=static_cast<std::uint8_t>((sum+4)/8);
    }
    return z;
}
std::array<std::array<int,K>,C> neighbors(){
    std::array<std::array<int,K>,C> r{};
    for(int z=0;z<H;++z)for(int y=0;y<H;++y)for(int x=0;x<H;++x){
        int k=0;for(int dz=-1;dz<=1;++dz)for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)
            r[static_cast<std::size_t>(at(x,y,z,H))][static_cast<std::size_t>(k++)]=
                at(periodic(x+dx,H),periodic(y+dy,H),periodic(z+dz,H),H);
    }
    return r;
}
struct Context{
    const std::array<std::array<int,K>,C> neighbor=neighbors();
};
Field decode_neural(const Coarse& z,const Model& m,const Context& ctx){
    Field out{};
    for(int cz=0;cz<H;++cz)for(int cy=0;cy<H;++cy)for(int cx=0;cx<H;++cx){
        const int c=at(cx,cy,cz,H);const float base=static_cast<float>(z[static_cast<std::size_t>(c)])/255.0f;
        for(int oz=0;oz<2;++oz)for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
            const int p=phase(ox,oy,oz), q=EW+p*(K+1);
            float d=m.w[static_cast<std::size_t>(q+K)];
            for(int k=0;k<K;++k){const int j=ctx.neighbor[static_cast<std::size_t>(c)][static_cast<std::size_t>(k)];
                d+=m.w[static_cast<std::size_t>(q+k)]*(static_cast<float>(static_cast<int>(z[static_cast<std::size_t>(j)])-static_cast<int>(z[static_cast<std::size_t>(c)]))/255.0f);
            }
            out[static_cast<std::size_t>(at(2*cx+ox,2*cy+oy,2*cz+oz,N))]=limit(base+0.32f*std::tanh(d));
        }
    }
    return out;
}
Field decode_nearest(const Coarse& z){
    Field out{};for(int iz=0;iz<N;++iz)for(int iy=0;iy<N;++iy)for(int ix=0;ix<N;++ix)
        out[static_cast<std::size_t>(at(ix,iy,iz,N))]=
            static_cast<float>(z[static_cast<std::size_t>(at(ix/2,iy/2,iz/2,H))])/255.0f;
    return out;
}
Field decode_trilinear(const Coarse& z){
    Field out{};for(int iz=0;iz<N;++iz)for(int iy=0;iy<N;++iy)for(int ix=0;ix<N;++ix){
        // Coarse cell centers are at physical coordinates 2c+0.5.
        const float px=(static_cast<float>(ix)-0.5f)*0.5f;
        const float py=(static_cast<float>(iy)-0.5f)*0.5f;
        const float pz=(static_cast<float>(iz)-0.5f)*0.5f;
        const int x0=static_cast<int>(std::floor(px)),y0=static_cast<int>(std::floor(py)),z0=static_cast<int>(std::floor(pz));
        const float fx=px-static_cast<float>(x0),fy=py-static_cast<float>(y0),fz=pz-static_cast<float>(z0);
        float sum=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
            const float wt=(dx?fx:1-fx)*(dy?fy:1-fy)*(dz?fz:1-fz);
            const auto val=z[static_cast<std::size_t>(at(periodic(x0+dx,H),periodic(y0+dy,H),periodic(z0+dz,H),H))];
            sum+=wt*static_cast<float>(val)/255.0f;
        }
        out[static_cast<std::size_t>(at(ix,iy,iz,N))]=sum;
    }
    return out;
}
double rmse(const Field& prediction,const Volume& x){
    double sum=0;for(int i=0;i<V;++i){const double d=static_cast<double>(prediction[static_cast<std::size_t>(i)])-static_cast<double>(x[static_cast<std::size_t>(i)])/255.0;sum+=d*d;}
    return std::sqrt(sum/static_cast<double>(V));
}
// Surrogate derivative of 8-bit quantizer is identity (straight-through estimator).
// Each neural decoder output sends derivative through its 3x3x3 neighborhood to
// all participating encoder latents, then to the eight shared encoder logits.
Grad gradient(const std::vector<Volume>& training,const Model& m,const Context& ctx){
    Grad g{};const auto alpha=mixing(m);
    for(const Volume& x:training){
        const Coarse z=encode(x,m);
        std::array<double,C> dz{};
        for(int cz=0;cz<H;++cz)for(int cy=0;cy<H;++cy)for(int cx=0;cx<H;++cx){
            const int c=at(cx,cy,cz,H);
            const double base=static_cast<double>(z[static_cast<std::size_t>(c)])/255.0;
            for(int oz=0;oz<2;++oz)for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
                const int p=phase(ox,oy,oz),q=EW+p*(K+1),i=at(2*cx+ox,2*cy+oy,2*cz+oz,N);
                double a=m.w[static_cast<std::size_t>(q+K)];
                for(int k=0;k<K;++k){const int j=ctx.neighbor[static_cast<std::size_t>(c)][static_cast<std::size_t>(k)];
                    a+=m.w[static_cast<std::size_t>(q+k)]*static_cast<double>(static_cast<int>(z[static_cast<std::size_t>(j)])-static_cast<int>(z[static_cast<std::size_t>(c)]))/255.0;
                }
                const double t=std::tanh(a),raw=base+0.32*t,pred=std::clamp(raw,0.0,1.0);
                const double error=pred-static_cast<double>(x[static_cast<std::size_t>(i)])/255.0;
                const double d=raw>0 && raw<1? 2.0*error:0.0;
                const double du=d*0.32*(1-t*t);
                g[static_cast<std::size_t>(q+K)]+=du;
                dz[static_cast<std::size_t>(c)]+=d;
                for(int k=0;k<K;++k){const int j=ctx.neighbor[static_cast<std::size_t>(c)][static_cast<std::size_t>(k)];
                    const double feature=static_cast<double>(static_cast<int>(z[static_cast<std::size_t>(j)])-static_cast<int>(z[static_cast<std::size_t>(c)]))/255.0;
                    g[static_cast<std::size_t>(q+k)]+=du*feature;
                    const double correction=du*static_cast<double>(m.w[static_cast<std::size_t>(q+k)]);
                    dz[static_cast<std::size_t>(j)]+=correction;
                    dz[static_cast<std::size_t>(c)]-=correction;
                }
            }
        }
        for(int cz=0;cz<H;++cz)for(int cy=0;cy<H;++cy)for(int cx=0;cx<H;++cx){
            const int c=at(cx,cy,cz,H);double pre=0;
            for(int oz=0;oz<2;++oz)for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
                const int p=phase(ox,oy,oz),i=at(2*cx+ox,2*cy+oy,2*cz+oz,N);
                pre+=static_cast<double>(alpha[static_cast<std::size_t>(p)])*static_cast<double>(x[static_cast<std::size_t>(i)])/255.0;
            }
            for(int oz=0;oz<2;++oz)for(int oy=0;oy<2;++oy)for(int ox=0;ox<2;++ox){
                const int p=phase(ox,oy,oz),i=at(2*cx+ox,2*cy+oy,2*cz+oz,N);
                const double source=static_cast<double>(x[static_cast<std::size_t>(i)])/255.0;
                g[static_cast<std::size_t>(p)]+=dz[static_cast<std::size_t>(c)]*static_cast<double>(alpha[static_cast<std::size_t>(p)])*(source-pre);
            }
        }
    }
    const double scale=1.0/(static_cast<double>(V)*static_cast<double>(training.size()));
    for(auto& v:g)v*=scale;
    return g;
}
struct Adam{
    std::array<double,W> m{},v{};int step=0;
    void update(Model& model,const Grad& g,double lr){
        ++step;const double c1=1-std::pow(.9,step),c2=1-std::pow(.999,step);
        for(int i=0;i<W;++i){const auto k=static_cast<std::size_t>(i);
            const double grad=g[k]+1e-7*model.w[k];m[k]=.9*m[k]+.1*grad;v[k]=.999*v[k]+.001*grad*grad;
            const double upd=lr*(m[k]/c1)/(std::sqrt(v[k]/c2)+1e-8);
            const double after=static_cast<double>(model.w[k])-upd;
            require(std::isfinite(after)&&std::abs(after)<=30.0,"unstable parameters");
            model.w[k]=static_cast<float>(after);
        }
    }
};
struct Feedback{Field value{};double fixedpoint_linf=0;};
Feedback fold(const Field& start,const Field& target,int count){
    Feedback a{start,0};double initial=0;
    for(int i=0;i<V;++i)initial=std::max(initial,std::abs(static_cast<double>(start[static_cast<std::size_t>(i)])-target[static_cast<std::size_t>(i)]));
    for(int t=0;t<count;++t)for(int i=0;i<V;++i){auto& v=a.value[static_cast<std::size_t>(i)];
        v=0.5f*v+0.5f*target[static_cast<std::size_t>(i)];require(std::isfinite(v)&&v>=0&&v<=1,"unstable feedback");
    }
    for(int i=0;i<V;++i)a.fixedpoint_linf=std::max(a.fixedpoint_linf,std::abs(static_cast<double>(a.value[static_cast<std::size_t>(i)])-target[static_cast<std::size_t>(i)]));
    require(a.fixedpoint_linf<=initial*std::pow(0.5,count)+1e-6,"feedback contraction violated");return a;
}
void put16(Vec& x,std::uint16_t v){x.push_back(static_cast<std::uint8_t>(v));x.push_back(static_cast<std::uint8_t>(v>>8));}
void put32(Vec& x,std::uint32_t v){for(int k=0;k<4;++k)x.push_back(static_cast<std::uint8_t>(v>>(8*k)));}
std::uint16_t get16(const Vec& x,std::size_t& pos){require(pos+2<=x.size(),"short packet u16");const auto v=static_cast<std::uint16_t>(static_cast<unsigned>(x[pos])|(static_cast<unsigned>(x[pos+1])<<8));pos+=2;return v;}
std::uint32_t get32(const Vec& x,std::size_t& pos){require(pos+4<=x.size(),"short packet u32");std::uint32_t v=0;for(int k=0;k<4;++k)v|=static_cast<std::uint32_t>(x[pos++])<<(8*k);return v;}
std::uint32_t crc(const Volume& v){std::uint32_t value=~0U;for(auto b:v){value^=b;for(int j=0;j<8;++j)value=(value>>1)^(0xEDB88320U & (0U-(value&1U)));}return ~value;}
Vec pack_zstd(const Vec& raw){Vec compressed(ZSTD_compressBound(raw.size()));const auto n=ZSTD_compress(compressed.data(),compressed.size(),raw.data(),raw.size(),3);
    require(!ZSTD_isError(n),"ZSTD_compress failed");compressed.resize(n);return compressed;}
Vec unpack_zstd(const Vec& compressed,std::size_t bytes){Vec raw(bytes);const auto n=ZSTD_decompress(raw.data(),raw.size(),compressed.data(),compressed.size());
    require(!ZSTD_isError(n)&&n==bytes,"ZSTD_decompress length/corruption error");return raw;}
Vec payload_bytes(const Volume& v){return Vec(v.begin(),v.end());}
Vec packet(std::uint8_t method,const Volume& source,const Vec& payload){
    require(payload.size()<=static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()),"payload too large");
    Vec out;out.reserve(20+payload.size());for(char x:std::string("JX3D"))out.push_back(static_cast<std::uint8_t>(x));
    out.push_back(1);out.push_back(method);put16(out,0);put32(out,V);put32(out,static_cast<std::uint32_t>(payload.size()));put32(out,crc(source));
    out.insert(out.end(),payload.begin(),payload.end());return out;
}
Vec model_bytes(const Model& m){Vec out;out.reserve(4*W);
    for(const float v:m.w){require(std::isfinite(v)&&std::abs(v)<=30,"invalid model parameters");
        std::uint32_t bits=0;static_assert(sizeof(float)==4,"binary32 required");std::memcpy(&bits,&v,4);put32(out,bits);}
    return out;
}
Model model_from(const Vec& payload,std::size_t& off){
    Model m;for(float& v:m.w){std::uint32_t bits=get32(payload,off);std::memcpy(&v,&bits,4);
        require(std::isfinite(v)&&std::abs(v)<=30,"invalid serialized weight");}
    return m;
}
// All residuals are signed 16-bit differences from *quantized* predictions.
Vec residual_bytes(const Volume& x,const Field& approximate){
    Vec raw;raw.reserve(V*2);for(int i=0;i<V;++i){
        const int d=static_cast<int>(x[static_cast<std::size_t>(i)])-static_cast<int>(as_byte(approximate[static_cast<std::size_t>(i)]));
        put16(raw,static_cast<std::uint16_t>(static_cast<std::int16_t>(d)));
    }return raw;
}
Volume apply_residual(const Field& prediction,const Vec& uncompressed){
    require(uncompressed.size()==V*2,"invalid residual count");Volume v{};std::size_t p=0;
    for(int i=0;i<V;++i){const auto bits=get16(uncompressed,p);
        const int signed_d=bits>=32768U?static_cast<int>(bits)-65536:static_cast<int>(bits);
        const int x=static_cast<int>(as_byte(prediction[static_cast<std::size_t>(i)]))+signed_d;
        require(x>=0&&x<=255,"residual out of bounds");v[static_cast<std::size_t>(i)]=static_cast<std::uint8_t>(x);
    }
    return v;
}
Vec neural_packet(const Volume& x,const Model& m,const Context& ctx){
    const Coarse z=encode(x,m);const Field approx=decode_neural(z,m,ctx);
    Vec payload=model_bytes(m);payload.insert(payload.end(),z.begin(),z.end());
    const Vec encoded=pack_zstd(residual_bytes(x,approx));payload.insert(payload.end(),encoded.begin(),encoded.end());
    return packet(2,x,payload);
}
Vec mean_residual_packet(const Volume& x){
    const Coarse z=mean_encode(x);const Field baseline=decode_nearest(z);
    Vec payload(z.begin(),z.end());const Vec encoded=pack_zstd(residual_bytes(x,baseline));
    payload.insert(payload.end(),encoded.begin(),encoded.end());return packet(3,x,payload);
}
Vec zstd_packet(const Volume& x){return packet(1,x,pack_zstd(payload_bytes(x)));}
Vec raw_packet(const Volume& x){return packet(0,x,payload_bytes(x));}
Volume recover(const Vec& src,const Context& ctx){
    require(src.size()>=20,"truncated packet");std::size_t pos=0;
    for(const char c:std::string("JX3D"))require(src[pos++]==static_cast<std::uint8_t>(c),"invalid packet magic");
    require(src[pos++]==1,"unsupported packet version");const std::uint8_t method=src[pos++];require(get16(src,pos)==0,"unknown packet flags");
    require(get32(src,pos)==V,"unsupported dimension");const auto bytes=get32(src,pos);const auto checksum=get32(src,pos);
    require(bytes==src.size()-pos,"mismatched packet payload size");
    Vec payload(src.begin()+static_cast<std::ptrdiff_t>(pos),src.end());Volume result{};
    if(method==0){require(payload.size()==V,"invalid raw length");std::copy(payload.begin(),payload.end(),result.begin());}
    else if(method==1){const Vec raw=unpack_zstd(payload,V);std::copy(raw.begin(),raw.end(),result.begin());}
    else if(method==2){require(payload.size()>4*W+C,"short neural payload");std::size_t off=0;
        const Model model=model_from(payload,off);Coarse z{};
        for(auto& q:z) q=payload[off++];
        const Vec encoded(payload.begin()+static_cast<std::ptrdiff_t>(off),payload.end());
        result=apply_residual(decode_neural(z,model,ctx),unpack_zstd(encoded,V*2));
    }else if(method==3){require(payload.size()>C,"short mean residual payload");Coarse z{};std::copy_n(payload.begin(),C,z.begin());
        const Vec encoded(payload.begin()+C,payload.end());result=apply_residual(decode_nearest(z),unpack_zstd(encoded,V*2));
    }else throw std::runtime_error("invalid packet method");
    require(crc(result)==checksum,"CRC32 mismatch");return result;
}
struct Measure{std::string name;double nearest=0,trilinear=0,untrained=0,trained=0,feedback=0;std::size_t raw=0,zstd=0,mean=0,neural=0,chosen=0;std::string chosen_name;};
Measure evaluate(const std::string& name,const Volume& x,const Model& initial,const Model& trained,const Context& ctx,bool check_adversarial){
    Measure r;r.name=name;
    const Coarse initial_z=encode(x,initial),z=encode(x,trained),mean_z=mean_encode(x);
    const Field near=decode_nearest(mean_z),tri=decode_trilinear(mean_z);
    const Field before=decode_neural(initial_z,initial,ctx),after=decode_neural(z,trained,ctx);
    const Feedback feedback=fold(before,after,6);
    r.nearest=rmse(near,x);r.trilinear=rmse(tri,x);r.untrained=rmse(before,x);r.trained=rmse(after,x);r.feedback=rmse(feedback.value,x);
    const std::array<Vec,4> candidates={raw_packet(x),zstd_packet(x),neural_packet(x,trained,ctx),mean_residual_packet(x)};
    const std::array<std::string,4> names={"raw","zstd3_raw","neural_zstd3_residual","mean_zstd3_residual"};
    r.raw=candidates[0].size();r.zstd=candidates[1].size();r.neural=candidates[2].size();r.mean=candidates[3].size();
    for(const auto& c:candidates)require(recover(c,ctx)==x,"codec packet failed exact roundtrip");
    std::size_t best=0;for(std::size_t i=1;i<candidates.size();++i)if(candidates[i].size()<candidates[best].size())best=i;
    r.chosen=candidates[best].size();r.chosen_name=names[best];
    if(check_adversarial){
        Vec bad=candidates[best];bad.back()^=0x44U;
        bool rejected=false;try{static_cast<void>(recover(bad,ctx));}catch(const std::exception&){rejected=true;}
        require(rejected,"tampered packet accepted");
        bad=candidates[best];bad.pop_back();rejected=false;
        try{static_cast<void>(recover(bad,ctx));}catch(const std::exception&){rejected=true;}
        require(rejected,"truncated packet accepted");
    }
    return r;
}
struct Args{int budget_ms=1000,epochs=85;std::uint64_t seed=20261120ULL;std::string output,decode_file;bool selftest=false;};
int positive(const std::string& value,int max){
    require(!value.empty()&&value[0]!='-',"negative argument");std::size_t n=0;
    const long long v=std::stoll(value,&n);require(n==value.size()&&v>=1&&v<=max,"invalid numeric argument");return static_cast<int>(v);
}
Args parse(int argc,char** argv){Args a;for(int i=1;i<argc;++i){const std::string key=argv[i];if(key=="--self-test"){a.selftest=true;continue;}
    require(i+1<argc,"missing option value");const std::string v=argv[++i];
    if(key=="--budget-ms")a.budget_ms=positive(v,60000);
    else if(key=="--epochs")a.epochs=positive(v,1000);
    else if(key=="--seed"){require(!v.empty()&&v[0]!='-',"invalid seed");std::size_t n=0;a.seed=std::stoull(v,&n);require(n==v.size(),"invalid seed");}
    else if(key=="--out")a.output=v;
    else if(key=="--decode")a.decode_file=v;
    else throw std::runtime_error("unknown option: "+key);
    }return a;
}
struct Results {int epochs=0;double ms=0,weights_encoder_change=0,weights_decoder_change=0;std::vector<Measure> rows;bool under_budget=false;};
Results experiment(const Args& a){
    const auto start=Clock::now();const auto elapsed=[&](){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    Context ctx;
    std::vector<Volume> training;training.reserve(6);
    for(std::uint64_t i=0;i<6;++i)training.push_back(smooth(a.seed+7919*i));
    Model model{},initial=model;Adam opt;int epochs=0;
    for(;epochs<a.epochs;++epochs){if(elapsed()>a.budget_ms*0.65)break;
        const Grad grad=gradient(training,model,ctx);opt.update(model,grad,0.032);
    }
    Results results;results.epochs=epochs;
    for(int i=0;i<EW;++i)results.weights_encoder_change+=std::abs(static_cast<double>(model.w[static_cast<std::size_t>(i)]-initial.w[static_cast<std::size_t>(i)]));
    for(int i=EW;i<W;++i)results.weights_decoder_change+=std::abs(static_cast<double>(model.w[static_cast<std::size_t>(i)]-initial.w[static_cast<std::size_t>(i)]));
    const std::array<std::string,4> labels={"smooth_holdout_A","smooth_holdout_B","OOD_blocks","OOD_random"};
    const std::array<Volume,4> tests={smooth(a.seed+900001),smooth(a.seed+900002),blocks(a.seed+900003),random_data(a.seed+900004)};
    for(int k=0;k<4;++k)results.rows.push_back(evaluate(labels[static_cast<std::size_t>(k)],tests[static_cast<std::size_t>(k)],initial,model,ctx,k==0));
    if(!a.output.empty()){
        const auto original=payload_bytes(tests[0]);
        const std::array<Vec,4> saved_candidates={raw_packet(tests[0]),zstd_packet(tests[0]),neural_packet(tests[0],model,ctx),mean_residual_packet(tests[0])};
        const auto best=std::min_element(saved_candidates.begin(),saved_candidates.end(),[](const Vec& left,const Vec& right){return left.size()<right.size();});
        require(best!=saved_candidates.end(),"no packet selected");
        require(recover(*best,ctx)==tests[0],"selected packet failed verification");
        std::ofstream original_file(a.output+"_original.raw",std::ios::binary);
        std::ofstream packet_file(a.output+"_encoded.jx3d",std::ios::binary);
        require(static_cast<bool>(original_file)&&static_cast<bool>(packet_file),"cannot create packet exports");
        original_file.write(reinterpret_cast<const char*>(original.data()),static_cast<std::streamsize>(original.size()));
        packet_file.write(reinterpret_cast<const char*>(best->data()),static_cast<std::streamsize>(best->size()));
        require(static_cast<bool>(original_file)&&static_cast<bool>(packet_file),"failed writing packet exports");
        std::ofstream summary(a.output+"_metrics.csv");require(static_cast<bool>(summary),"cannot create metrics CSV");
        summary<<"dataset,nearest_rmse,trilinear_rmse,untrained_rmse,joint_trained_rmse,feedback6_rmse,raw_packet,zstd_raw_packet,mean_residual_packet,learned_residual_packet,selected_packet,selected_method\n"<<std::setprecision(12);
        for(const auto& r:results.rows)summary<<r.name<<','<<r.nearest<<','<<r.trilinear<<','<<r.untrained<<','<<r.trained<<','<<r.feedback<<','<<r.raw<<','<<r.zstd<<','<<r.mean<<','<<r.neural<<','<<r.chosen<<','<<r.chosen_name<<'\n';
        require(static_cast<bool>(summary),"CSV export failed");
        std::ofstream volume(a.output+"_heldout.csv");require(static_cast<bool>(volume),"cannot create heldout CSV");
        const auto lat=encode(tests[0],model);
        const auto out=decode_neural(lat,model,ctx);
        volume<<"x,y,z,source,learned_prediction\n"<<std::setprecision(9);
        for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){const int i=at(x,y,z,N);
            volume<<x<<','<<y<<','<<z<<','<<static_cast<int>(tests[0][static_cast<std::size_t>(i)])<<','<<out[static_cast<std::size_t>(i)]<<'\n';}
        require(static_cast<bool>(volume),"heldout export failed");
    }
    require(results.weights_encoder_change>0.0001,"joint encoder failed to update");
    require(results.weights_decoder_change>0.001,"joint decoder failed to update");
    results.ms=elapsed();results.under_budget=results.ms<=a.budget_ms;return results;
}
void display(const Results& r,const Args& a){
    std::cout<<std::fixed<<std::setprecision(7)<<"JX3D_JOINT_END_TO_END\n"
        <<"geometry=16x16x16 latent=8x8x8 parameters="<<W<<" train_fields=6 epochs="<<r.epochs<<'\n'
        <<"encoder_weight_L1_change="<<r.weights_encoder_change<<" decoder_weight_L1_change="<<r.weights_decoder_change<<'\n';
    for(const auto& m:r.rows){
        std::cout<<"DATASET "<<m.name<<" nearest_RMSE="<<m.nearest<<" trilinear_RMSE="<<m.trilinear
         <<" initial_RMSE="<<m.untrained<<" joint_RMSE="<<m.trained<<" feedback_RMSE="<<m.feedback<<'\n'
         <<"CODEC "<<m.name<<" raw_packet="<<m.raw<<" zstd3_raw_packet="<<m.zstd
         <<" mean_residual_packet="<<m.mean<<" neural_residual_packet="<<m.neural
         <<" selected="<<m.chosen_name<<":"<<m.chosen<<" exact_voxels=4096/4096\n";
    }
    std::cout<<"elapsed_ms="<<r.ms<<" budget_ms="<<a.budget_ms<<" within_budget="<<(r.under_budget?"YES":"NO")<<'\n';
}
void standalone_decode(const std::string& input,const std::string& output){
    require(!output.empty(),"--decode requires --out PATH");
    std::ifstream f(input,std::ios::binary);require(static_cast<bool>(f),"cannot read input packet");
    const Vec raw((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>());
    require(raw.size()<=100000,"packet too large");
    const Volume voxels=recover(raw,Context{});
    std::ofstream out(output,std::ios::binary);require(static_cast<bool>(out),"cannot create decoded file");
    out.write(reinterpret_cast<const char*>(voxels.data()),static_cast<std::streamsize>(voxels.size()));
    require(static_cast<bool>(out),"failed writing decoded file");
    std::cout<<"DECODE_PASS exact_packet_crc32_verified voxels=4096 bytes=4096\n";
}
void self_test(){
    Context ctx;Model m{};
    for(const std::uint64_t seed:{1ULL,7ULL,42ULL})for(const Volume& v:{smooth(seed),blocks(seed),random_data(seed)}){
        require(recover(raw_packet(v),ctx)==v,"raw failure");
        require(recover(zstd_packet(v),ctx)==v,"zstd failure");
        require(recover(mean_residual_packet(v),ctx)==v,"mean failure");
        require(recover(neural_packet(v,m,ctx),ctx)==v,"neural failure");
    }
    Args a;a.epochs=40;a.budget_ms=20000;
    const auto r=experiment(a);
    require(r.rows[0].trained<r.rows[0].untrained,"joint train does not improve smooth holdout");
    require(r.rows[1].trained<r.rows[1].untrained,"joint train does not improve second holdout");
    std::cout<<"SELF_TEST_PASS geometry, both trainable branches, exact four-way roundtrip, CRC, corruption, feedback, heldout learning\n";
}
}
int main(int argc,char** argv){try{const auto a=jx::parse(argc,argv);if(a.selftest){jx::self_test();return 0;}
    if(!a.decode_file.empty()){jx::standalone_decode(a.decode_file,a.output);return 0;}
    const auto r=jx::experiment(a);jx::display(r,a);return r.under_budget?0:2;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}