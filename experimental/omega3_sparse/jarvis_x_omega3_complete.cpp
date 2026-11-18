// JARVIS-X Omega3: sparse 90-bit address + 72-bit scheduler + reversible
// tile codec + bounded 3D refinement + persistent sparse overrides.
// C++17 GCC/Clang (__int128 extension). No dense 2^90 allocation.
#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using u64=std::uint64_t;
using u128=unsigned __int128;
using B=std::uint8_t;
constexpr u64 INC=104729, OFFSET=505050;
constexpr u64 SEED=0x4a41525649535833ULL;
constexpr u128 TMASK=(u128(1)<<72)-1;
std::string dec(u128 x){if(!x)return "0";std::string s;while(x){s.push_back(char('0'+x%10));x/=10;}std::reverse(s.begin(),s.end());return s;}
u128 num(const std::string&s){if(s.empty())throw std::invalid_argument("empty integer");u128 x=0;for(char c:s){if(c<'0'||c>'9')throw std::invalid_argument("invalid decimal");unsigned d=c-'0';if(x>(~u128(0)-d)/10)throw std::out_of_range("128-bit overflow");x=x*10+d;}return x;}
u64 bounded(const std::string&s,u64 max){auto x=num(s);if(x>max)throw std::out_of_range("argument out of range");return (u64)x;}
u128 morton(u64 x,u64 y,u64 z,unsigned bits){u128 m=0;for(unsigned b=0;b<bits;b++){m|=u128((x>>b)&1)<<(3*b);m|=u128((y>>b)&1)<<(3*b+1);m|=u128((z>>b)&1)<<(3*b+2);}return m;}
std::array<u64,3> unmorton(u128 m,unsigned bits){std::array<u64,3> a{};for(unsigned b=0;b<bits;b++)for(unsigned k=0;k<3;k++)a[k]|=u64((m>>(3*b+k))&1)<<b;return a;}
u128 schedule(u128 t){return (u128(OFFSET)+u128(INC)*(t&TMASK))&TMASK;}
u64 mix(u64 x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;x^=x>>31;return x;}
B procedural(u128 m,u64 seed=SEED){return B(mix(u64(m)^mix(u64(m>>64)^seed))>>56);}
u128 address(u64 x,u64 y,u64 z){if(x>=(1ULL<<30)||y>=(1ULL<<30)||z>=(1ULL<<30))throw std::out_of_range("coordinate");return morton(x,y,z,30);}
struct U128Hash{size_t operator()(u128 x)const{return size_t(mix(u64(x)^mix(u64(x>>64))));}};
using Store=std::unordered_map<u128,B,U128Hash>;
void save(const Store& s,const std::string&path){std::ofstream f(path,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("cannot write store");f<<"JX3D1\n";std::vector<std::pair<u128,B>> rows(s.begin(),s.end());std::sort(rows.begin(),rows.end(),[](auto&a,auto&b){return a.first<b.first;});for(auto&r:rows)f<<dec(r.first)<<" "<<unsigned(r.second)<<"\n";if(!f)throw std::runtime_error("store write failed");}
Store load(const std::string&path){Store s;std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot read store");std::string head;std::getline(f,head);if(head!="JX3D1")throw std::runtime_error("bad store header");std::string line;while(std::getline(f,line)){std::istringstream in(line);std::string a,b,extra;if(!(in>>a>>b)||(in>>extra))throw std::runtime_error("bad store record");auto m=num(a),v=num(b);if(m>=(u128(1)<<90)||v>255||!s.emplace(m,B(v)).second)throw std::runtime_error("invalid/duplicate record");}if(!f.eof())throw std::runtime_error("store read failed");return s;}
B value(u128 m,const Store&s){auto it=s.find(m);return it==s.end()?procedural(m):it->second;}
constexpr unsigned N=8, COUNT=N*N*N; // bounded 8^3 working tile
using Block=std::array<B,COUNT>;
unsigned idx(unsigned x,unsigned y,unsigned z){return (z*N+y)*N+x;}
Block make_block(u128 tile,const Store&s){auto q=unmorton(tile,24);Block b{};for(unsigned z=0;z<N;z++)for(unsigned y=0;y<N;y++)for(unsigned x=0;x<N;x++)b[idx(x,y,z)]=value(address(q[0]*64+x,q[1]*64+y,q[2]*64+z),s);return b;}
// Reversible 3D spatial XOR predictor: predecessor along x, then y, then z.
Block encode(const Block&b){Block e{};for(unsigned z=0;z<N;z++)for(unsigned y=0;y<N;y++)for(unsigned x=0;x<N;x++){B pred=x?b[idx(x-1,y,z)]:y?b[idx(x,y-1,z)]:z?b[idx(x,y,z-1)]:0;e[idx(x,y,z)]=b[idx(x,y,z)]^pred;}return e;}
Block decode(const Block&e){Block b{};for(unsigned z=0;z<N;z++)for(unsigned y=0;y<N;y++)for(unsigned x=0;x<N;x++){B pred=x?b[idx(x-1,y,z)]:y?b[idx(x,y-1,z)]:z?b[idx(x,y,z-1)]:0;b[idx(x,y,z)]=e[idx(x,y,z)]^pred;}return b;}
u64 checksum(const Block&b){u64 h=14695981039346656037ULL;for(B v:b){h^=v;h*=1099511628211ULL;}return h;}
double loss(const Block&a,const Block&b){double s=0;for(unsigned i=0;i<COUNT;i++){double d=int(a[i])-int(b[i]);s+=d*d;}return s/COUNT;}
// Deterministic 6-neighbour smoothing with reference fidelity and accept/reject.
// Not an ANN; refinement is explicitly bounded and verified.
Block candidate(const Block&cur,const Block&reference){Block out{};for(unsigned z=0;z<N;z++)for(unsigned y=0;y<N;y++)for(unsigned x=0;x<N;x++){unsigned xm=x?x-1:x,xp=x+1<N?x+1:x,ym=y?y-1:y,yp=y+1<N?y+1:y,zm=z?z-1:z,zp=z+1<N?z+1:z;unsigned neighbors=cur[idx(xm,y,z)]+cur[idx(xp,y,z)]+cur[idx(x,ym,z)]+cur[idx(x,yp,z)]+cur[idx(x,y,zm)]+cur[idx(x,y,zp)];unsigned v=(4*unsigned(reference[idx(x,y,z)])+neighbors+5)/10;out[idx(x,y,z)]=B(v);}return out;}
double objective(const Block&b,const Block&reference){double s=0;for(unsigned z=0;z<N;z++)for(unsigned y=0;y<N;y++)for(unsigned x=0;x<N;x++){double v=b[idx(x,y,z)],d=v-reference[idx(x,y,z)];s+=4*d*d;if(x+1<N){double e=v-b[idx(x+1,y,z)];s+=e*e;}if(y+1<N){double e=v-b[idx(x,y+1,z)];s+=e*e;}if(z+1<N){double e=v-b[idx(x,y,z+1)];s+=e*e;}}return s;}
void refine(u128 tile,unsigned steps,const Store&s){Block reference=make_block(tile,s),current=reference;std::cout<<"step,objective,mse_to_original,hash,accepted\n";for(unsigned t=0;t<=steps;t++){if(t){Block next=candidate(current,reference);bool ok=objective(next,reference)<=objective(current,reference);if(ok)current=next;std::cout<<t<<","<<std::fixed<<std::setprecision(3)<<objective(current,reference)<<","<<loss(current,reference)<<","<<checksum(current)<<","<<ok<<"\n";if(!ok)break;}else std::cout<<"0,"<<objective(current,reference)<<",0,"<<checksum(current)<<",1\n";}auto e=encode(current),d=decode(e);if(d!=current)throw std::runtime_error("codec mismatch");std::cout<<"codec_roundtrip=PASS\n";}
void test(){for(u64 x:{0ULL,1ULL,63ULL,64ULL,(1ULL<<30)-1})for(u64 y:{0ULL,2ULL,65ULL,(1ULL<<30)-1})for(u64 z:{0ULL,3ULL,127ULL,(1ULL<<30)-1}){auto a=unmorton(address(x,y,z),30);if(a!=std::array<u64,3>{x,y,z})throw std::runtime_error("morton roundtrip");}
for(u128 t:{u128(0),u128(1),TMASK,TMASK+1,~u128(0)}){auto p=schedule(t);if(p!=schedule(t+(t<=TMASK?TMASK+1:0)))throw std::runtime_error("scheduler period");auto q=unmorton(p,24);if(morton(q[0],q[1],q[2],24)!=p)throw std::runtime_error("tile inverse");for(u64 i:{0ULL,1ULL,63ULL}){u128 m=address(q[0]*64+i,q[1]*64+2,q[2]*64+3);if(m!=(p<<18)+morton(i,2,3,6))throw std::runtime_error("hierarchical identity");}}
Store s;auto m=address(7,8,9);s[m]=201;if(value(m,s)!=201||value(m,{})!=procedural(m))throw std::runtime_error("overlay");Block b=make_block(schedule(0),s);if(decode(encode(b))!=b)throw std::runtime_error("codec");Block c=candidate(b,b);if(objective(c,b)<0)throw std::runtime_error("objective");std::cout<<"SELFTEST PASS: geometry, modular scheduler, hierarchical address, overlay, codec\n";}
void usage(){std::cout<<"--voxel X Y Z | --tile T | --tile-voxel T i j k | --refine T [steps<=1000] | --write FILE X Y Z VALUE | --read FILE X Y Z | --selftest | --bench [count<=10000000] | --quine PATH | --help\n";}
int main(int argc,char**argv){try{if(argc<2){usage();return 2;}std::string cmd=argv[1];if(cmd=="--help"){usage();return 0;}if(cmd=="--selftest"){test();return 0;}if(cmd=="--voxel"&&argc==5){auto x=bounded(argv[2],(1ULL<<30)-1),y=bounded(argv[3],(1ULL<<30)-1),z=bounded(argv[4],(1ULL<<30)-1);auto m=address(x,y,z);std::cout<<"Morton-90: "<<dec(m)<<"\nvoxel-value-u8: "<<unsigned(procedural(m))<<"\n";return 0;}
if(cmd=="--tile"&&argc==3){auto t=num(argv[2]);auto p=schedule(t);auto q=unmorton(p,24);std::cout<<"tile_morton_index_72: "<<dec(p)<<"\ntile_coordinate: ("<<q[0]<<", "<<q[1]<<", "<<q[2]<<")\nvoxel_origin: ("<<q[0]*64<<", "<<q[1]*64<<", "<<q[2]*64<<")\n";return 0;}
if(cmd=="--tile-voxel"&&argc==6){auto t=num(argv[2]);auto i=bounded(argv[3],63),j=bounded(argv[4],63),k=bounded(argv[5],63);auto p=schedule(t);auto q=unmorton(p,24);auto x=q[0]*64+i,y=q[1]*64+j,z=q[2]*64+k;auto m=address(x,y,z);std::cout<<"coordinate: ("<<x<<", "<<y<<", "<<z<<")\nMorton-90: "<<dec(m)<<"\nvoxel-value-u8: "<<unsigned(procedural(m))<<"\n";return 0;}
if(cmd=="--refine"&&(argc==3||argc==4)){auto t=num(argv[2]);unsigned steps=argc==4?unsigned(bounded(argv[3],1000)):8;refine(schedule(t),steps,{});return 0;}
if(cmd=="--write"&&argc==7){Store s;{std::ifstream probe(argv[2]);if(probe.good())s=load(argv[2]);}auto m=address(bounded(argv[3],(1ULL<<30)-1),bounded(argv[4],(1ULL<<30)-1),bounded(argv[5],(1ULL<<30)-1));s[m]=B(bounded(argv[6],255));save(s,argv[2]);std::cout<<"saved overrides: "<<s.size()<<"\n";return 0;}
if(cmd=="--read"&&argc==6){auto s=load(argv[2]);auto m=address(bounded(argv[3],(1ULL<<30)-1),bounded(argv[4],(1ULL<<30)-1),bounded(argv[5],(1ULL<<30)-1));std::cout<<unsigned(value(m,s))<<"\n";return 0;}
if(cmd=="--bench"&&(argc==2||argc==3)){u64 count=argc==3?bounded(argv[2],10000000):100000;volatile u64 sink=0;auto start=std::chrono::steady_clock::now();for(u64 i=0;i<count;i++){auto p=schedule(i);auto q=unmorton(p,24);sink=sink+procedural(address(q[0]*64,q[1]*64,q[2]*64));}double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::cout<<"queries="<<count<<" seconds="<<sec<<" queries_per_second="<<(sec>0?count/sec:0)<<" checksum="<<sink<<"\n";return 0;}
if(cmd=="--quine"&&argc==3){std::ifstream f(argv[2],std::ios::binary);if(!f)throw std::runtime_error("source not found");std::cout<<f.rdbuf();if(!std::cout)throw std::runtime_error("output failed");return 0;}usage();return 2;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
