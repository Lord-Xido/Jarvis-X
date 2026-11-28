// Bounded 3D residual optimizer; exact integer arithmetic, no GPU/ANN claims.
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
using I = std::int64_t;
constexpr int N=16, V=N*N*N;
constexpr int STEPS=1000000;
using Field=std::array<int,V>;
int id(int x,int y,int z){return ((x+N)%N)*N*N+((y+N)%N)*N+(z+N)%N;}
std::array<int,6> nb(int i){int x=i/(N*N),y=(i/N)%N,z=i%N;return {id(x-1,y,z),id(x+1,y,z),id(x,y-1,z),id(x,y+1,z),id(x,y,z-1),id(x,y,z+1)};}
struct RNG{uint32_t s=42; uint32_t next(){s^=s<<13;s^=s>>17;s^=s<<5;return s;}};
I sq(I x){return x*x;}
int f(const Field& a,int i){int t=10*a[i];for(int j:nb(i))t+=a[j];return t;}
I verify(const Field& a,const Field& y,const Field& r){I loss=0;for(int i=0;i<V;++i){int e=y[i]-f(a,i);if(e!=r[i]||a[i]<0||a[i]>255)throw std::runtime_error("state invariant");loss+=sq(e);}return loss;}
uint64_t mix(uint64_t h,uint64_t x){for(int k=0;k<8;++k){h^=(x>>(8*k))&255u;h*=1099511628211ULL;}return h;}
struct Result{I initial,final;int commits,rejects,faults;uint64_t hash,chain;std::array<int,3> trials{},accepted{};};
Result run(std::ostream* trace){
 RNG rng; Field truth{},y{},a{},r{},w{};for(int i=0;i<V;++i){int x=i/(N*N), yy=(i/N)%N,z=i%N;truth[i]=32+((x/4)*31+(yy/4)*17+(z/4)*11+int(rng.next()%9))%192;}
 for(int i=0;i<V;++i)y[i]=f(truth,i);
 r=y; I loss=verify(a,y,r);Result out{loss,loss,0,0,0,14695981039346656037ULL,14695981039346656037ULL};
 if(trace)*trace<<"iteration,loss,commits,rejections,faults,chain\n";
 for(int k=1;k<=STEPS;++k){
  // Adaptive coarse-to-fine exploration; retain occasional larger proposals.
  int draw=int(rng.next()%100),level=0;
  if(k<=4096)level=draw<40?2:draw<75?1:0;
  else level=draw<2?2:draw<10?1:0;
  int b=1<<level; ++out.trials[level];
  int x=int(rng.next()%N)/b*b,yy=int(rng.next()%N)/b*b,z=int(rng.next()%N)/b*b;
  std::vector<int> block,touched;int lo=-255,hi=255;
  auto add=[&](int j,int v){if(!w[j])touched.push_back(j);w[j]+=v;};
  for(int dx=0;dx<b;++dx)for(int dy=0;dy<b;++dy)for(int dz=0;dz<b;++dz){int i=id(x+dx,yy+dy,z+dz);block.push_back(i);lo=std::max(lo,-a[i]);hi=std::min(hi,255-a[i]);add(i,10);for(int j:nb(i))add(j,1);}
  I dot=0,norm=0;for(int j:touched){dot+=I(r[j])*w[j];norm+=sq(w[j]);}
  // Exact one-dimensional integer quadratic minimizer: test both nearest integers.
  int d0=std::clamp(int(dot/norm),lo,hi),d1=std::clamp(int(dot/norm)+(dot>=0?1:-1),lo,hi);
  auto delta=[&](int d){return norm*sq(d)-2*dot*d;};
  int d=delta(d1)<delta(d0)?d1:d0;I change=delta(d);
  I independent=0;for(int j:touched)independent+=sq(I(r[j])-I(w[j])*d)-sq(r[j]);
  if(change!=independent)throw std::runtime_error("delta invariant");
  bool fault=k%997==0; if(fault)++out.faults;
  bool commit=change<0&&!fault;
  if(commit){for(int i:block)a[i]+=d;for(int j:touched)r[j]-=w[j]*d;loss+=change;++out.commits;++out.accepted[level];if(verify(a,y,r)!=loss)throw std::runtime_error("commit oracle");}
  else ++out.rejects;
  for(int j:touched)w[j]=0;
  out.chain=mix(out.chain,uint64_t(k));out.chain=mix(out.chain,uint64_t(loss));out.chain=mix(out.chain,uint64_t(x*N*N+yy*N+z));out.chain=mix(out.chain,uint64_t(level));out.chain=mix(out.chain,uint64_t(d));out.chain=mix(out.chain,uint64_t(commit));
  if(k%1000==0||k==1){if(verify(a,y,r)!=loss)throw std::runtime_error("periodic oracle");if(trace)*trace<<k<<','<<loss<<','<<out.commits<<','<<out.rejects<<','<<out.faults<<','<<out.chain<<'\n';}
 }
 out.final=verify(a,y,r);for(int i=0;i<V;++i)out.hash=mix(out.hash,uint64_t(a[i]));
 // Full exact recovery of scaled observation using signed residual; separate from inverse accuracy.
 for(int i=0;i<V;++i)if(f(a,i)+r[i]!=y[i])throw std::runtime_error("residual roundtrip");
 if(out.commits+out.rejects!=STEPS||out.final>out.initial||out.faults!=STEPS/997)throw std::runtime_error("accounting");
 return out;
}
int main(int argc,char** argv){try{std::ofstream file(argc>1?argv[1]:"geometric_trace.csv");if(!file)throw std::runtime_error("trace open");auto a=run(&file),b=run(nullptr);file.close();if(!file)throw std::runtime_error("trace write");if(a.chain!=b.chain||a.hash!=b.hash||a.final!=b.final||a.commits!=b.commits||a.rejects!=b.rejects)throw std::runtime_error("replay");std::cout<<"proposals="<<STEPS<<" voxels="<<V<<" initial="<<a.initial<<" final="<<a.final<<" commits="<<a.commits<<" rejects="<<a.rejects<<" fault_vetoes="<<a.faults<<" state_fnv64="<<std::hex<<a.hash<<" trace_fnv64="<<a.chain<<std::dec<<"\n";for(int j=0;j<3;++j)std::cout<<"block_side="<<(1<<j)<<" trials="<<a.trials[j]<<" commits="<<a.accepted[j]<<"\n";std::cout<<"replay=PASS residual=PASS oracle=PASS\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
