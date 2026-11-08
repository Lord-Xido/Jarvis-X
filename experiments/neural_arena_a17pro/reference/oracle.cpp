#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using i16=std::int16_t; using u32=std::uint32_t;
struct Spec { const char* name; int x,y,z, pattern, iterations; };
// Keep in lockstep with Swift and the independent NumPy oracle.
constexpr Spec cases[]={
  {"singleton",1,1,1,0,3},
  {"thin_z",1,9,5,0,3},
  {"edge_odd",5,7,3,0,3},
  {"tile_impulse",8,8,4,1,3},
  {"cross_tile_impulse",9,10,5,1,3},
  {"mixed_sign",17,15,9,0,3},
  {"full_random",64,64,32,0,3},
  {"full_constant",64,64,32,2,3},
  {"full_checker",64,64,32,3,3}
};
int round_div(int n, int d){return n>=0?(n+d/2)/d:-((-n+d/2)/d);}
int idx(int x,int y,int z,int nx,int ny){return x+nx*(y+ny*z);}
std::vector<i16> make(const Spec& s){
  std::vector<i16> a(size_t(s.x)*s.y*s.z*3);
  u32 seed=0x0A17C0DEu;
  for(int z=0;z<s.z;z++)for(int y=0;y<s.y;y++)for(int x=0;x<s.x;x++)for(int c=0;c<3;c++){
    seed ^= seed<<13; seed ^= seed>>17; seed ^= seed<<5;
    int v=int(seed%8193u)-4096;
    if(s.pattern==1) v=(x==s.x/2&&y==s.y/2&&z==s.z/2)?(c==1?-4096:4096):0;
    if(s.pattern==2) v=-4096;
    if(s.pattern==3) v=((x+y+z+c)%2)?4096:-4096;
    a[size_t(idx(x,y,z,s.x,s.y))*3+c]=i16(v);
  }
  return a;
}
std::vector<i16> step(const std::vector<i16>& a,const Spec&s){
  std::vector<i16>b(a.size());const int xy=s.x*s.y;
  for(int z=0;z<s.z;z++)for(int y=0;y<s.y;y++)for(int x=0;x<s.x;x++){
    const int p=idx(x,y,z,s.x,s.y);
    for(int c=0;c<3;c++){
      int sm=0,n=0;
      if(x>0){sm+=a[3*(p-1)+c];n++;}if(x+1<s.x){sm+=a[3*(p+1)+c];n++;}
      if(y>0){sm+=a[3*(p-s.x)+c];n++;}if(y+1<s.y){sm+=a[3*(p+s.x)+c];n++;}
      if(z>0){sm+=a[3*(p-xy)+c];n++;}if(z+1<s.z){sm+=a[3*(p+xy)+c];n++;}
      int avg=n?round_div(sm,n):int(a[3*p+c]);
      int val=round_div(17*int(a[3*p+c])+3*avg,20);
      b[3*p+c]=i16(val<-4096?-4096:(val>4096?4096:val));
    }
  }return b;
}
int main(int argc,char**argv){
  if(argc!=2){std::cerr<<"usage: oracle OUTPUT_DIR\n";return 2;}
  for(const auto&s:cases){auto a=make(s);for(int t=1;t<=s.iterations;t++){
    a=step(a,s);std::string path=std::string(argv[1])+"/"+s.name+"_t"+std::to_string(t)+".bin";
    std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("open");
    for(i16 v:a){u32 bits=std::uint16_t(v);char ch[2]={char(bits&255),char(bits>>8)};f.write(ch,2);}
    if(!f)throw std::runtime_error("write");
    std::cout<<s.name<<" iteration="<<t<<" bytes="<<a.size()*2<<"\n";
  }}
  return 0;
}
