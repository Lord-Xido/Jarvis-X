#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

namespace dm3d_ir {
static constexpr double PI=3.14159265358979323846;
struct Volume { int N{}; std::vector<float> v; explicit Volume(int n=0):N(n),v((size_t)n*n*n,0.0f){} size_t idx(int x,int y,int z)const{return((size_t)z*N+y)*N+x;} };
inline int wrap(int i,int N){i%=N;return i<0?i+N:i;}
enum class Op:uint8_t{DMA,CONV3D,ENCODE,DECODE,MATMUL,ATTENTION,FFT,REDUCE,FIX_POINT};
inline const char* name(Op o){switch(o){case Op::DMA:return"DMA";case Op::CONV3D:return"CONV3D";case Op::ENCODE:return"ENCODE";case Op::DECODE:return"DECODE";case Op::MATMUL:return"MATMUL";case Op::ATTENTION:return"ATTENTION";case Op::FFT:return"FFT";case Op::REDUCE:return"REDUCE";case Op::FIX_POINT:return"FIX_POINT";}return"?";}
struct Node{Op op{Op::DMA};float p0{0.5f},p1{0.2f};int iters{1};};
using Graph=std::vector<Node>;

inline Volume input(int N){Volume a(N);for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){float fx=float(x)/N,fy=float(y)/N,fz=float(z)/N;a.v[a.idx(x,y,z)]=0.48f*std::sin(float(2*PI)*fx)+0.27f*std::cos(float(4*PI)*fy)+0.19f*std::sin(float(6*PI)*fz+float(PI)*fx)+0.06f*std::cos(float(2*PI)*(fx+fy+fz));}return a;}
inline Volume conv3d(const Volume&i,float a,float b){Volume o(i.N);int N=i.N;for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){float c=i.v[i.idx(x,y,z)];float s=i.v[i.idx(wrap(x-1,N),y,z)]+i.v[i.idx(wrap(x+1,N),y,z)]+i.v[i.idx(x,wrap(y-1,N),z)]+i.v[i.idx(x,wrap(y+1,N),z)]+i.v[i.idx(x,y,wrap(z-1,N))]+i.v[i.idx(x,y,wrap(z+1,N))];o.v[o.idx(x,y,z)]=a*c+b*(s/6.0f)+0.04f*std::tanh(c);}return o;}
inline Volume encode(const Volume&i,float m){Volume o(i.N);int N=i.N;m=std::clamp(m,0.0f,1.0f);for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){float s=0;for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)s+=i.v[i.idx(wrap(x+dx,N),wrap(y+dy,N),wrap(z+dz,N))];float c=i.v[i.idx(x,y,z)];o.v[o.idx(x,y,z)]=(1-m)*c+m*s/8.0f;}return o;}
inline Volume decode(const Volume&i,float sh){Volume o(i.N);int N=i.N;sh=std::clamp(sh,0.0f,1.5f);for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){float c=i.v[i.idx(x,y,z)];float s=i.v[i.idx(wrap(x-1,N),y,z)]+i.v[i.idx(wrap(x+1,N),y,z)]+i.v[i.idx(x,wrap(y-1,N),z)]+i.v[i.idx(x,wrap(y+1,N),z)]+i.v[i.idx(x,y,wrap(z-1,N))]+i.v[i.idx(x,y,wrap(z+1,N))];o.v[o.idx(x,y,z)]=c-0.12f*sh*(s-6*c);}return o;}
inline Volume matmul(const Volume&i,float a,float b){Volume o(i.N);int N=i.N;for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x)o.v[o.idx(x,y,z)]=0.5f*b*i.v[i.idx(wrap(x-1,N),y,z)]+a*i.v[i.idx(x,y,z)]+0.5f*b*i.v[i.idx(wrap(x+1,N),y,z)];return o;}
inline Volume attention(const Volume&i,float t){Volume o(i.N);int N=i.N;t=std::clamp(t,0.05f,2.0f);for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){float q=i.v[i.idx(x,y,z)];std::array<float,7>v{q,i.v[i.idx(wrap(x-1,N),y,z)],i.v[i.idx(wrap(x+1,N),y,z)],i.v[i.idx(x,wrap(y-1,N),z)],i.v[i.idx(x,wrap(y+1,N),z)],i.v[i.idx(x,y,wrap(z-1,N))],i.v[i.idx(x,y,wrap(z+1,N))]};std::array<float,7>s{};float m=-std::numeric_limits<float>::infinity();for(int j=0;j<7;++j){s[j]=t*q*v[j];m=std::max(m,s[j]);}float d=0,n=0;for(int j=0;j<7;++j){float w=std::exp(s[j]-m);d+=w;n+=w*v[j];}o.v[o.idx(x,y,z)]=n/std::max(d,1e-12f);}return o;}
inline Volume fft(const Volume&i,float h){int N=i.N;Volume o(N);h=std::clamp(h,0.0f,1.0f);std::vector<std::complex<double>>f(N);for(int z=0;z<N;++z)for(int y=0;y<N;++y){for(int k=0;k<N;++k){std::complex<double>s(0,0);for(int x=0;x<N;++x){double a=-2*PI*k*x/N;s+=double(i.v[i.idx(x,y,z)])*std::complex<double>(std::cos(a),std::sin(a));}if(k>N/4&&k<3*N/4)s*=h;f[k]=s;}for(int x=0;x<N;++x){std::complex<double>s(0,0);for(int k=0;k<N;++k){double a=2*PI*k*x/N;s+=f[k]*std::complex<double>(std::cos(a),std::sin(a));}o.v[o.idx(x,y,z)]=float(s.real()/N);}}return o;}
inline Volume reduce(const Volume&i,float b){b=std::clamp(b,0.0f,1.0f);double s=0;for(float q:i.v)s+=q;float m=float(s/i.v.size());Volume o(i.N);for(size_t j=0;j<i.v.size();++j)o.v[j]=(1-b)*i.v[j]+b*m;return o;}
inline Volume fixpoint(const Volume&i,float l,int it){l=std::clamp(l,0.0f,1.0f);it=std::clamp(it,1,4);Volume c=i;int N=i.N;for(int t=0;t<it;++t){Volume n(N);for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){float q=c.v[c.idx(x,y,z)];float s=c.v[c.idx(wrap(x-1,N),y,z)]+c.v[c.idx(wrap(x+1,N),y,z)]+c.v[c.idx(x,wrap(y-1,N),z)]+c.v[c.idx(x,wrap(y+1,N),z)]+c.v[c.idx(x,y,wrap(z-1,N))]+c.v[c.idx(x,y,wrap(z+1,N))];n.v[n.idx(x,y,z)]=(1-l)*q+l*s/6.0f;}c=std::move(n);}return c;}
inline Volume apply(const Volume&i,const Node&n){switch(n.op){case Op::DMA:return i;case Op::CONV3D:return conv3d(i,n.p0,n.p1);case Op::ENCODE:return encode(i,n.p0);case Op::DECODE:return decode(i,n.p0);case Op::MATMUL:return matmul(i,n.p0,n.p1);case Op::ATTENTION:return attention(i,n.p0);case Op::FFT:return fft(i,n.p0);case Op::REDUCE:return reduce(i,n.p0);case Op::FIX_POINT:return fixpoint(i,n.p0,n.iters);}return i;}
inline Volume execute(const Volume&i,const Graph&g){Volume c=i;for(const auto&n:g)c=apply(c,n);return c;}
inline Graph target(){return{{Op::CONV3D,.84f,.14f,1},{Op::ATTENTION,.55f,0.0f,1},{Op::FIX_POINT,.10f,0.0f,2}};}
inline double maxerr(const Volume&a,const Volume&b){double e=0;for(size_t i=0;i<a.v.size();++i)e=std::max(e,double(std::abs(a.v[i]-b.v[i])));return e;}
inline double mse(const Volume&a,const Volume&b){double e=0;for(size_t i=0;i<a.v.size();++i){double d=double(a.v[i])-b.v[i];e+=d*d;}return e/a.v.size();}
inline double checksum(const Volume&a){double s=0;for(float q:a.v)s+=q;return s;}
} // namespace dm3d_ir
