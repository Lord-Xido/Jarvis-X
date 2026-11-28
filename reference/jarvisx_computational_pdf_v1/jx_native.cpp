// C++17 native bounded codec VM, matching jarvisx.py neural transform.
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using Bytes=std::vector<uint8_t>;
using I8=std::array<int,8>;
int clamp(int v,int a,int b){return std::max(a,std::min(b,v));}
int floor256(int v){return v>=0?v/256:-((-v+255)/256);}
struct ANN {
 I8 w{}; int bias=128*256;
 I8 features(const Bytes& seen,int i){
  int x=i&7,y=(i>>3)&7,z=(i>>6)&7;
  auto get=[&](int d,bool ok){return ok?int(seen.at(i-d))-128:0;};
  int a=get(1,x>0),b=get(8,y>0),c=get(64,z>0);
  return {a,b,c,get(2,x>1),get(16,y>1),get(128,z>1),(a+b+c)/3,clamp(a-b+c,-128,127)};
 }
 int predict(const I8& h){int v=bias;for(int k=0;k<8;k++)v+=w[k]*h[k];return clamp(floor256(v),0,255);}
 void learn(int a,int p,const I8& h){int e=a-p;bias=clamp(bias+16*e,0,255*256);for(int k=0;k<8;k++)w[k]=clamp(w[k]+e*h[k]/256,-512,512);}
};
Bytes transform(const Bytes& in,bool decode){ANN m; Bytes out,seen;for(int i=0;i<int(in.size());i++){auto h=m.features(seen,i);int p=m.predict(h),a=decode?(in[i]^p):in[i];out.push_back(decode?a:(a^p));seen.push_back(a);m.learn(a,p,h);}return out;}
Bytes read(const char* p,size_t max){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("open input");Bytes b;char c;while(f.get(c)){if(b.size()==max)throw std::runtime_error("input limit");b.push_back(uint8_t(c));}if(!f.eof())throw std::runtime_error("read error");return b;}
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("usage: jx_native program.bin tile.bin residual.bin");
 auto code=read(argv[1],48),src=read(argv[2],512);if(code.size()!=48||src.empty())throw std::runtime_error("invalid size");
 Bytes residual,decoded;bool exact=false,commit=false;
 for(int i=0;i<6;i++){uint64_t word=0;for(int j=0;j<8;j++)word=(word<<8)|code[i*8+j];if(word!=(uint64_t(i+1)<<56))throw std::runtime_error("noncanonical program");switch(word>>56){
  case 1: break; // tile is mapped x=i&7, y=(i>>3)&7, z=i>>6
  case 2: residual=transform(src,false);break;
  case 3: decoded=transform(residual,true);break;
  case 4: exact=decoded==src;break;
  case 5: commit=exact;break; // native raw codec gate, no compression objective
  case 6: std::cout<<"bytes="<<src.size()<<" exact="<<exact<<" commit="<<commit<<"\n";break;
 }}
 if(!commit)throw std::runtime_error("rollback");
 std::ofstream f(argv[3],std::ios::binary|std::ios::trunc);f.write(reinterpret_cast<const char*>(residual.data()),residual.size());if(!f)throw std::runtime_error("write error");return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
