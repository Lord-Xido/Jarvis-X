// JARVIS-X RRAET: stationary 3D residual optimization reference, C++17.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using I=std::int64_t; using U=std::uint64_t;
struct RNG { U s=42; U next(){s+=0x9e3779b97f4a7c15ULL; U z=s; z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL; z=(z^(z>>27))*0x94d049bb133111ebULL; return z^(z>>31);} };
void mix(U& h,U v){for(int i=0;i<8;++i){h^=(v>>(8*i))&255;h*=1099511628211ULL;}}
struct Volume {
 int n,N; std::vector<std::array<int,6>> adj;
 explicit Volume(int a):n(a),N(a*a*a),adj(N){for(int x=0;x<n;++x)for(int y=0;y<n;++y)for(int z=0;z<n;++z){int i=idx(x,y,z);adj[i]={idx(x-1,y,z),idx(x+1,y,z),idx(x,y-1,z),idx(x,y+1,z),idx(x,y,z-1),idx(x,y,z+1)};}}
 int idx(int x,int y,int z)const{return ((x+n)%n*n+(y+n)%n)*n+(z+n)%n;}
 std::vector<int> forward(const std::vector<int>& x)const{std::vector<int> y(N);for(int i=0;i<N;++i){y[i]=10*x[i];for(int j:adj[i])y[i]+=x[j];}return y;}
};
I squared(const std::vector<int>& r){I s=0;for(int v:r)s+=I(v)*v;return s;}
int main(int argc,char** argv){
 int n=argc>1?std::stoi(argv[1]):16; U steps=argc>2?std::stoull(argv[2]):1000000; std::string path=argc>3?argv[3]:"rraet_million_trace.csv";
 if(n<4||n>32||n%4||!steps||steps>10000000)throw std::runtime_error("n in [4,32], divisible by 4; steps in [1,10M]");
 Volume v(n); RNG rng; std::vector<int> truth(v.N),x(v.N,0),r,coeff(v.N,0);for(int& a:truth)a=int(rng.next()%256);const auto y=v.forward(truth);r=y; I loss=squared(r),initial=loss; U hash=14695981039346656037ULL,accepted=0,rejected=0,checks=0,first_zero=0; std::array<U,4> proposed{},commits{};
 // The verifier recomputes Hx directly rather than using the incremental residual.
 auto verify=[&](){auto f=v.forward(x);I s=0;for(int i=0;i<v.N;++i){if(y[i]-f[i]!=r[i]||x[i]<0||x[i]>255)throw std::runtime_error("residual/bounds mismatch");s+=I(y[i]-f[i])*(y[i]-f[i]);}if(s!=loss)throw std::runtime_error("loss mismatch");++checks;};
 std::ofstream out(path);if(!out)throw std::runtime_error("cannot open trace");out<<"step,scale,x,y,z,delta,candidate_sse,committed_sse,accepted,chain_fnv64\n";
 for(U k=1;k<=steps;++k){
  // Periodic full-field proposal: fixed-point Jacobi shadow solve of Hx=y.
  // H is strictly diagonally dominant; the projected update contracts by 6/10.
  if(k%250000==0){
   ++proposed[3]; constexpr I S=65536; std::vector<I> a(v.N),b(v.N);std::vector<int> trial(v.N);
   for(int i=0;i<v.N;++i)a[i]=I(x[i])*S;
   for(int sweep=0;sweep<32;++sweep){for(int i=0;i<v.N;++i){I t=I(y[i])*S;for(int j:v.adj[i])t-=a[j];b[i]=std::clamp((t>=0?(t+5)/10:-((-t+5)/10)),I(0),255*S);}a.swap(b);}
   for(int i=0;i<v.N;++i)trial[i]=int((a[i]+S/2)/S);
   auto pred=v.forward(trial);std::vector<int> rr(v.N);for(int i=0;i<v.N;++i)rr[i]=y[i]-pred[i];I cand=squared(rr);bool ok=cand<loss;
   if(ok){x.swap(trial);r.swap(rr);loss=cand;++accepted;++commits[3];verify();if(!loss&&!first_zero)first_zero=k;}else ++rejected;
   verify();mix(hash,k);mix(hash,3);mix(hash,U(cand));mix(hash,U(loss));mix(hash,ok);
   out<<k<<",0,-1,-1,-1,0,"<<cand<<','<<loss<<','<<ok<<','<<std::hex<<hash<<std::dec<<'\n';continue;
  }
  unsigned level=(k%16==0)?2:(k%4==0?1:0);int b=1<<level; ++proposed[level];int a=int(rng.next()%U(n/b))*b,c=int(rng.next()%U(n/b))*b,d=int(rng.next()%U(n/b))*b;
  std::vector<int> block,touched; block.reserve(64);touched.reserve(448);int lo=-255,hi=255;
  auto add=[&](int i,int w){if(!coeff[i])touched.push_back(i);coeff[i]+=w;};
  for(int i=a;i<a+b;++i)for(int j=c;j<c+b;++j)for(int l=d;l<d+b;++l){int p=v.idx(i,j,l);block.push_back(p);lo=std::max(lo,-x[p]);hi=std::min(hi,255-x[p]);add(p,10);for(int q:v.adj[p])add(q,1);}
  I dot=0,norm=0;for(int p:touched){dot+=I(r[p])*coeff[p];norm+=I(coeff[p])*coeff[p];}
  // exact integer quadratic optimum, constrained to a bounded voxel range.
  int delta=int(dot>=0?(dot+norm/2)/norm:-((-dot+norm/2)/norm));delta=std::clamp(delta,lo,hi);
  // Continue making nontrivial proposals even when the best shift is zero.
  if(!delta){int sign=(rng.next()&1)?1:-1;delta=std::clamp(sign,lo,hi);if(!delta)delta=std::clamp(-sign,lo,hi);}
  I change=I(delta)*delta*norm-2*I(delta)*dot;I candidate=loss+change;
  // Directly recompute all affected terms before acceptance; independent of quadratic expansion.
  I direct=0;for(int p:touched){I old=r[p],next=old-I(delta)*coeff[p];direct+=next*next-old*old;}if(direct!=change||candidate<0)throw std::runtime_error("candidate delta mismatch");
  bool ok=candidate<loss;if(ok){for(int p:block)x[p]+=delta;for(int p:touched)r[p]-=delta*coeff[p];loss=candidate;++accepted;++commits[level];verify();if(!loss&&!first_zero)first_zero=k;}else ++rejected;
  for(int p:touched)coeff[p]=0;
  if(k%1000==0||k==steps)verify();
  mix(hash,k);mix(hash,level);mix(hash,a);mix(hash,c);mix(hash,d);mix(hash,U(I(delta)));mix(hash,U(candidate));mix(hash,U(loss));mix(hash,ok);
  out<<k<<','<<b<<','<<a<<','<<c<<','<<d<<','<<delta<<','<<candidate<<','<<loss<<','<<ok<<','<<std::hex<<hash<<std::dec<<'\n';
 }
 out.close();if(!out)throw std::runtime_error("trace write failure");verify();U errors=0,state=14695981039346656037ULL;for(int i=0;i<v.N;++i){errors+=x[i]!=truth[i];mix(state,U(x[i]));}
 // For the noiseless known-source fixture H is strictly diagonally dominant;
 // exact observation residual implies unique exact source recovery.
 std::cout<<"n="<<n<<" voxels="<<v.N<<" proposals="<<steps<<" accepted="<<accepted<<" rejected="<<rejected<<" initial_sse="<<initial<<" final_sse="<<loss<<" first_zero="<<first_zero<<" source_voxel_errors="<<errors<<" full_checks="<<checks<<"\n";
 for(unsigned i=0;i<4;++i)std::cout<<"scale="<<(i==3?0:(1<<i))<<" proposals="<<proposed[i]<<" accepted="<<commits[i]<<"\n";
 std::cout<<"trace_fnv64="<<std::hex<<hash<<" state_fnv64="<<state<<std::dec<<"\n";
 if(steps>=1000000&&(loss||errors))return 2;
}
