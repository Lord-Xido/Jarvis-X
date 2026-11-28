// MOAGI O36 SVM v1: actual 64-bit interpreter; original codec retained as reference.
// MOAGI Omega36: procedural 3D ANN + recursive reversible byte codec.
// Build: g++ -std=c++17 -O3 -Wall -Wextra moagi_omega36_engine.cpp -lcrypto -o moagi36
// GCC/Clang __uint128_t for 120-bit logical addresses; OpenSSL for SHA-256.
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <limits>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <openssl/sha.h>
using Byte=uint8_t; using Bytes=std::vector<Byte>; using U64=uint64_t; using Wide=__uint128_t;
constexpr U64 L=1000000000000ULL;
void check(bool v,const char* s){if(!v)throw std::runtime_error(s);}
struct P {U64 x,y,z;};
Wide addr(P p,U64 n=L){check(n>0&&n<=L&&p.x<n&&p.y<n&&p.z<n,"coordinate out of bounds");return Wide(p.x)+Wide(n)*p.y+Wide(n)*n*p.z;}
P coord(Wide a,U64 n=L){check(n>0&&n<=L&&a<Wide(n)*n*n,"address out of bounds");U64 x=a%n;a/=n;U64 y=a%n;return {x,y,U64(a/n)};}
std::string decimal(Wide a){std::string s;do{s+=char('0'+a%10);a/=10;}while(a);std::reverse(s.begin(),s.end());return s;}
std::array<Byte,32> digest(const Bytes& b){std::array<Byte,32>d{}; SHA256(b.data(),b.size(),d.data());return d;}
std::string hex(const std::array<Byte,32>& d){std::string s;for(Byte b:d){s+="0123456789abcdef"[b>>4];s+="0123456789abcdef"[b&15];}return s;}
Byte initial(P p,U64 n=L){const char seed[]="MOAGI-O36-v1"; std::array<Byte,sizeof(seed)+15> b{};std::copy(seed,seed+sizeof(seed),b.begin());Wide a=addr(p,n);for(int i=14;i>=0;--i){b[sizeof(seed)+i]=Byte(a);a>>=8;}std::array<Byte,32>d{};SHA256(b.data(),b.size(),d.data());return d[0];}
U64 prev(U64 a,U64 n){return a?a-1:n-1;} U64 next(U64 a,U64 n){return a+1==n?0:a+1;}
std::array<P,6> neighbors(P p,U64 n){return {{{prev(p.x,n),p.y,p.z},{next(p.x,n),p.y,p.z},{p.x,prev(p.y,n),p.z},{p.x,next(p.y,n),p.z},{p.x,p.y,prev(p.z,n)},{p.x,p.y,next(p.z,n)}}};}
Byte fire(Byte a,const std::array<Byte,6>& q){unsigned s=4*a+32;for(unsigned i=0;i<6;++i)s+=(i%2?1:2)*q[i];return Byte(std::min(255u,s/16));}
struct Key{Wide a;unsigned t;bool operator==(const Key& k)const{return a==k.a&&t==k.t;}};
U64 mix(U64 x){x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;x*=0x94d049bb133111ebULL;return x^(x>>31);}
struct Hash{size_t operator()(const Key& k)const{return mix(U64(k.a))^mix(U64(k.a>>64)+k.t);}};
struct Model{U64 n;std::unordered_map<Key,Byte,Hash> cache,patch;explicit Model(U64 n_=L):n(n_){}
 void set(P p,Byte b){patch[{addr(p,n),0}]=b^initial(p,n);cache.clear();}
 Byte get(P p,unsigned t){check(t<=16,"query depth must be <=16");Key k{addr(p,n),t};auto it=cache.find(k);if(it!=cache.end())return it->second;Byte v;
 if(!t){auto r=patch.find(k);v=initial(p,n)^(r==patch.end()?0:r->second);}else{std::array<Byte,6> q{};auto ps=neighbors(p,n);for(int i=0;i<6;++i)q[i]=get(ps[i],t-1);v=fire(get(p,t-1),q);}cache.emplace(k,v);return v;}
};
size_t volume(unsigned n){return size_t(n)*n*n;}
size_t ix(unsigned x,unsigned y,unsigned z,unsigned n){return x+size_t(n)*(y+size_t(n)*z);}
void sidecheck(unsigned n){check(n&&n<=128&&!(n&(n-1)),"side must be power of two in 1..128");}
unsigned levels(unsigned n){unsigned l=0;while(n>1){n/=2;++l;}return l;}
// RLE: 0..127 = zero run of tag+1; 128..255 = literal run of (tag&127)+1.
Bytes rle(const Bytes& in){Bytes out;for(size_t i=0;i<in.size();){size_t j=i;bool lit=in[i]!=0;while(j<in.size()&&j-i<128&&(in[j]!=0)==lit)++j;out.push_back(Byte((lit?128:0)+(j-i-1)));if(lit)out.insert(out.end(),in.begin()+i,in.begin()+j);i=j;}return out;}
Bytes unrle(const Bytes& in,size_t count){Bytes o;o.reserve(count);size_t i=0;while(i<in.size()){Byte a=in[i++];size_t n=(a&127)+1;check(n<=count-o.size(),"RLE overrun");if(a&128){check(n<=in.size()-i,"truncated RLE");o.insert(o.end(),in.begin()+i,in.begin()+i+n);i+=n;}else o.insert(o.end(),n,0);}check(o.size()==count,"RLE length mismatch");return o;}
void u32(Bytes& b,uint32_t a){for(int i=0;i<4;++i)b.push_back(Byte(a>>(8*i)));}
uint32_t rd32(const Bytes& b,size_t& i){check(i<=b.size()&&b.size()-i>=4,"truncated integer");uint32_t a=0;for(int j=0;j<4;++j)a|=uint32_t(b[i++])<<(8*j);return a;}
// Modes: 0 first child; 1 rounded mean; 2 median; 3 first+ANN predictor;
// 4 mean+ANN predictor; 5 mean+blend of coarse ANN and local parent.
Byte pool(std::array<Byte,8> a,unsigned mode){if(mode==0||mode==3)return a[0];if(mode==2){std::sort(a.begin(),a.end());return Byte((unsigned(a[3])+a[4])/2);}unsigned s=0;for(Byte b:a)s+=b;return Byte((s+4)/8);}
Byte pred(const Bytes& p,unsigned n,unsigned x,unsigned y,unsigned z,unsigned mode){Byte a=p[ix(x,y,z,n)];if(mode<3)return a;auto ps=neighbors({x,y,z},n);std::array<Byte,6> q{};for(int i=0;i<6;++i)q[i]=p[ix(ps[i].x,ps[i].y,ps[i].z,n)];Byte b=fire(a,q);return mode==5?Byte((unsigned(a)+b+1)/2):b;}
struct Layer{unsigned mode;Bytes r;};
struct Encoded{Byte root;std::vector<Layer> layers;};
Encoded fold(Bytes x,unsigned n,const std::vector<unsigned>& modes){Encoded e{};check(modes.size()==levels(n),"mode count");for(unsigned m:modes){check(m<6,"invalid mode");unsigned h=n/2;Bytes p(volume(h));for(unsigned z=0;z<h;++z)for(unsigned y=0;y<h;++y)for(unsigned a=0;a<h;++a){std::array<Byte,8> c{};for(unsigned k=0;k<8;++k)c[k]=x[ix(2*a+(k&1),2*y+((k>>1)&1),2*z+(k>>2),n)];p[ix(a,y,z,h)]=pool(c,m);}
 Bytes r(x.size());for(unsigned z=0;z<n;++z)for(unsigned y=0;y<n;++y)for(unsigned a=0;a<n;++a)r[ix(a,y,z,n)]=x[ix(a,y,z,n)]^pred(p,h,a/2,y/2,z/2,m);
 e.layers.push_back({m,rle(r)});x=std::move(p);n=h;}e.root=x.at(0);return e;}
Bytes unfold(const Encoded& e){Bytes p{e.root};unsigned h=1;for(auto it=e.layers.rbegin();it!=e.layers.rend();++it){unsigned n=2*h;Bytes r=unrle(it->r,volume(n)),x(r.size());for(unsigned z=0;z<n;++z)for(unsigned y=0;y<n;++y)for(unsigned a=0;a<n;++a)x[ix(a,y,z,n)]=r[ix(a,y,z,n)]^pred(p,h,a/2,y/2,z/2,it->mode);p=std::move(x);h=n;}return p;}
// Serialized archive: magic, side, root, per-level mode + length + residual RLE, source SHA256.
Bytes archive(const Encoded& e,unsigned n,const std::array<Byte,32>& hash){Bytes b{'O','3','6','1'};u32(b,n);b.push_back(e.root);for(auto& l:e.layers){b.push_back(Byte(l.mode));u32(b,uint32_t(l.r.size()));b.insert(b.end(),l.r.begin(),l.r.end());}b.insert(b.end(),hash.begin(),hash.end());return b;}
Bytes decode(const Bytes& b){check(b.size()>=41&&b[0]=='O'&&b[1]=='3'&&b[2]=='6'&&b[3]=='1',"bad archive");size_t i=4;unsigned n=rd32(b,i);sidecheck(n);Encoded e{b[i++],{}};for(unsigned j=0;j<levels(n);++j){check(i<b.size()-32,"missing level");unsigned m=b[i++];check(m<6,"bad mode");size_t len=rd32(b,i);check(i<=b.size()-32&&len<=b.size()-32-i,"bad residual length");e.layers.push_back({m,Bytes(b.begin()+i,b.begin()+i+len)});i+=len;}check(i==b.size()-32,"trailing data");Bytes x=unfold(e);auto h=digest(x);check(std::equal(h.begin(),h.end(),b.begin()+i),"SHA256 mismatch");return x;}
struct Result{Bytes data;std::vector<unsigned> modes;size_t initial,attempts=0,accepts=0;};
Result optimize(const Bytes& x,unsigned n,unsigned sweeps=2,bool trace=false){sidecheck(n);check(x.size()==volume(n),"wrong raw volume length");auto hash=digest(x);std::vector<unsigned> m(levels(n),0);Bytes a=archive(fold(x,n,m),n,hash);check(decode(a)==x,"baseline decode");Result r{a,m,a.size()};for(unsigned s=0;s<sweeps;++s){bool changed=false;for(size_t k=0;k<m.size();++k){for(unsigned c=0;c<6;++c){if(c==r.modes[k])continue;auto trial=r.modes;trial[k]=c;auto b=archive(fold(x,n,trial),n,hash);++r.attempts;check(decode(b)==x,"candidate decode");if(b.size()<r.data.size()){if(trace)std::cout<<"COMMIT sweep="<<s<<" level="<<k<<" mode="<<c<<" bytes="<<r.data.size()<<"->"<<b.size()<<"\n";r.data=std::move(b);r.modes=std::move(trial);++r.accepts;changed=true;}}}if(!changed)break;}return r;}
Bytes read(const std::string& f){std::ifstream in(f,std::ios::binary);check(bool(in),"cannot open input");return Bytes(std::istreambuf_iterator<char>(in),{});}
void write(const std::string& f,const Bytes& b){std::ofstream out(f,std::ios::binary);check(bool(out),"cannot open output");out.write(reinterpret_cast<const char*>(b.data()),b.size());check(bool(out),"write failed");}
void report(const Result& r,const Bytes& x){std::cout<<"raw_bytes="<<x.size()<<" baseline_archive="<<r.initial<<" optimized_archive="<<r.data.size()<<" attempts="<<r.attempts<<" commits="<<r.accepts<<" exact="<<(decode(r.data)==x)<<"\nmodes=";for(auto m:r.modes)std::cout<<m<<' ';std::cout<<"\nsource_sha256="<<hex(digest(x))<<"\n";}
void legacy_selftest(){for(P p:std::array<P,3>{{{0,0,0},{L-1,L-1,L-1},{123,456,789}}}){P q=coord(addr(p));check(q.x==p.x&&q.y==p.y&&q.z==p.z,"address roundtrip");}for(unsigned v=0;v<256;++v){Byte b=initial({0,0,0});check(Byte(b^(b^v))==v,"XOR roundtrip");}
 unsigned n=7;Model m(n);m.set({0,0,0},255);Bytes d(volume(n));for(unsigned z=0;z<n;++z)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x)d[ix(x,y,z,n)]=x||y||z?initial({x,y,z},n):255;unsigned checks=0;for(unsigned t=0;t<7;++t){Bytes nd(d.size());for(unsigned z=0;z<n;++z)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){size_t i=ix(x,y,z,n);check(m.get({x,y,z},t)==d[i],"dense comparison");++checks;unsigned s=4*d[i]+32+2*d[ix((x+n-1)%n,y,z,n)]+d[ix((x+1)%n,y,z,n)]+2*d[ix(x,(y+n-1)%n,z,n)]+d[ix(x,(y+1)%n,z,n)]+2*d[ix(x,y,(z+n-1)%n,n)]+d[ix(x,y,(z+1)%n,n)];nd[i]=Byte(s/16);}d.swap(nd);}
 Model v;Bytes trace;for(unsigned t=0;t<=8;++t)trace.push_back(v.get({L-1,0,1},t));check(trace==Bytes{138,68,69,58,51,45,39,34,30},"Python trace mismatch");std::cout<<"dense_comparisons="<<checks<<" byte_roundtrips=256\nvirtual_trace=";for(Byte b:trace)std::cout<<unsigned(b)<<' ';std::cout<<"\ncache_entries="<<v.cache.size()<<" trace_sha256="<<hex(digest(trace))<<"\n";
 size_t trials=0,commits=0;for(unsigned j=0;j<1000;++j){unsigned s=1u<<(1+j%3);Bytes x(volume(s));for(size_t i=0;i<x.size();++i){unsigned a=i%s,b=(i/s)%s,c=i/(s*s);x[i]=j%3==0?Byte(mix(i+j*101)):j%3==1?Byte((a/2+b/2+c/2+j)%256):Byte(j); }auto r=optimize(x,s);check(decode(r.data)==x,"codec roundtrip");check(r.data.size()<=r.initial,"CTR regression");auto bad=r.data;bad.back()^=1;bool rejected=false;try{decode(bad);}catch(const std::exception&){rejected=true;}check(rejected,"corruption accepted");trials+=r.attempts;commits+=r.accepts;}
 std::cout<<"recursive_codec_roundtrips=1000 corruption_rejections=1000 candidate_checks="<<trials<<" commits="<<commits<<"\nSELFTEST PASS\n";}
unsigned num(const char* s){auto a=std::stoull(s);check(a<=UINT32_MAX,"integer too large");return unsigned(a);}

namespace svm {
// [opcode:8][flags:8][dst:16][src:16][operand:16]; binary files little endian.
enum Op:unsigned {ADDR=1,COORD=2,SEED=3,LOAD=4,STORE=5,NXM=0x10,NXP,NYM,NYP,NZM,NZP,ANN=0x20,FOLD=0x30,PRED=0x31,XORRES=0x32,RLE=0x33,PUSH=0x34,PACK=0x35,POP=0x40,UNRLE=0x41,RESTORE=0x42,UNPACK=0x43,HASH=0x50,CMP=0x51,MUTATE=0x60,COST=0x61,COMMIT=0x62,ROLLBACK=0x63,LOOP=0x70,HALT=0xff};
U64 ins(unsigned op,unsigned f=0,unsigned d=0,unsigned s=0,unsigned k=0){return U64(op)<<56|U64(f)<<48|U64(d)<<32|U64(s)<<16|k;}
struct I{unsigned op,f,d,s,k; explicit I(U64 w):op(w>>56),f((w>>48)&255),d((w>>32)&65535),s((w>>16)&65535),k(w&65535){}};
using Program=std::vector<U64>;
std::string word(U64 w){std::ostringstream o;o<<std::hex<<std::uppercase<<std::setw(16)<<std::setfill('0')<<w;return o.str();}
const char* name(unsigned op){switch(op){
#define N(x) case x:return #x;
N(ADDR) N(COORD) N(SEED) N(LOAD) N(STORE) N(NXM) N(NXP) N(NYM) N(NYP) N(NZM) N(NZP) N(ANN) N(FOLD) N(PRED) N(XORRES) N(RLE) N(PUSH) N(PACK) N(POP) N(UNRLE) N(RESTORE) N(UNPACK) N(HASH) N(CMP) N(MUTATE) N(COST) N(COMMIT) N(ROLLBACK) N(LOOP) N(HALT)
#undef N
 default:return "UNKNOWN";}}
Program ann_program(){Program p{ins(ADDR),ins(COORD),ins(LOAD,0,0,0,12),ins(SEED,0,0,0,11)};for(unsigned j=0;j<6;++j)p.push_back(ins(NXM+j,0,j+1));p.push_back(ins(ANN,0,0,1));p.push_back(ins(STORE));p.push_back(ins(HALT));return p;}
struct ANNVM{
 U64 n;Program code=ann_program();std::unordered_map<Key,Byte,Hash> cache;size_t steps=0;bool trace=false;explicit ANNVM(U64 side=L):n(side){}
 Byte eval(P p,unsigned t,unsigned depth=0){check(t<=16&&depth<=16,"ANN depth bound");std::array<Byte,16> r{};Wide a=0;bool addressed=false;size_t pc=0,local=0;while(pc<code.size()){check(++local<=64&&++steps<=100000000,"ANN instruction budget");U64 w=code[pc++];I i(w);check(i.d<16&&i.s<16,"register bound");if(trace&&depth==0)std::cout<<"ANN pc="<<pc-1<<" word="<<word(w)<<" "<<name(i.op)<<" t="<<t<<'\n';switch(i.op){
 case ADDR:a=addr(p,n);addressed=true;break;
 case COORD:check(addressed,"no address");p=coord(a,n);break;
 case LOAD:{check(addressed,"no address");auto it=cache.find({a,t});if(it!=cache.end()){r[i.d]=it->second;pc=i.k;}break;}
 case SEED:if(!t){r[i.d]=initial(p,n);pc=i.k;}break;
 case NXM:case NXP:case NYM:case NYP:case NZM:case NZP:check(t>0,"neighbor at t=0");r[i.d]=eval(neighbors(p,n)[i.op-NXM],t-1,depth+1);break;
 case ANN:{check(t>0&&i.s+6<=16,"ANN operands");std::array<Byte,6> q{};std::copy_n(r.begin()+i.s,6,q.begin());r[i.d]=fire(eval(p,t-1,depth+1),q);break;}
 case STORE:check(addressed&&cache.size()<2000000,"cache bound");cache[{a,t}]=r[i.d];break;
 case HALT:return r[i.d];default:throw std::runtime_error("invalid ANN opcode");}}throw std::runtime_error("ANN missing HALT");}
};
// Actual codec program: baseline and candidates use same instruction sequence.
Program codec_program(){return {ins(HASH,1),ins(MUTATE,0,0,0,21),ins(FOLD),ins(PRED),ins(XORRES),ins(RLE),ins(PUSH),ins(LOOP,0,0,0,2),ins(PACK),ins(UNPACK),ins(POP),ins(UNRLE),ins(PRED,1),ins(RESTORE),ins(LOOP,1,0,0,10),ins(HASH),ins(CMP),ins(COST),ins(COMMIT),ins(ROLLBACK),ins(LOOP,2,0,0,1),ins(HALT)};}
Program decoder_program(){return {ins(UNPACK),ins(POP),ins(UNRLE),ins(PRED,1),ins(RESTORE),ins(LOOP,1,0,0,1),ins(HASH),ins(CMP),ins(HALT)};}
// FOLD and POP also handle 1^3 tiles by skipping the empty geometric level.
struct CodecVM{
 Bytes source,cur,parent,predicted,residual,packed,candidate,best; Encoded enc{}; Layer popped{}; std::vector<unsigned> modes,best_modes; std::array<Byte,32> original{},expected{},actual{};
 unsigned side=0,n=0,h=0,sweeps=2;size_t level=0,cursor=0,attempts=0,accepts=0,rejects=0,baseline=0,steps=0,cost=0;bool verified=false,accepted=false,trace=false,dec_only=false,baseline_pending=true;unsigned phase=0;
 explicit CodecVM(Bytes x,unsigned s,unsigned sw=2):source(std::move(x)),side(s),sweeps(sw){sidecheck(side);check(source.size()==volume(side)&&sw<=16,"invalid source/sweeps");best_modes.assign(levels(side),0);}
 explicit CodecVM(Bytes archive):candidate(std::move(archive)),dec_only(true){}
 void require(unsigned p){check(phase==p,"invalid codec instruction order");}
 void run(const Program& p){size_t pc=0;while(pc<p.size()){check(++steps<=100000,"codec instruction budget");U64 w=p[pc++];I i(w);if(trace)std::cout<<"VM pc="<<pc-1<<" word="<<word(w)<<" "<<name(i.op)<<" level="<<level<<" side="<<n<<'\n';switch(i.op){
 case HASH:if(i.f==1){require(0);original=digest(source);phase=1;}else{require(12);actual=digest(cur);phase=13;}break;
 case MUTATE:{require(1);modes=best_modes;if(baseline_pending){baseline_pending=false;}else{bool found=false;size_t total=size_t(sweeps)*modes.size()*6;while(cursor<total){size_t c=cursor++;size_t k=(c/6)%modes.size();unsigned m=c%6;if(m!=best_modes[k]){modes[k]=m;found=true;break;}}if(!found){pc=i.k;break;}++attempts;}cur=source;n=side;level=0;enc={};verified=accepted=false;phase=2;break;}
 case FOLD:{require(2);if(n==1){phase=7;pc+=5;break;}h=n/2;parent.resize(volume(h));unsigned m=modes.at(level);for(unsigned z=0;z<h;++z)for(unsigned y=0;y<h;++y)for(unsigned x=0;x<h;++x){std::array<Byte,8> b{};for(unsigned j=0;j<8;++j)b[j]=cur[ix(2*x+(j&1),2*y+((j>>1)&1),2*z+(j>>2),n)];parent[ix(x,y,z,h)]=pool(b,m);}phase=3;break;}
 case PRED:{require(i.f?10:3);unsigned m=i.f?popped.mode:modes.at(level);predicted.resize(volume(n));for(unsigned z=0;z<n;++z)for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x)predicted[ix(x,y,z,n)]=pred(parent,h,x/2,y/2,z/2,m);phase=i.f?11:4;break;}
 case XORRES:require(4);residual.resize(cur.size());for(size_t j=0;j<cur.size();++j)residual[j]=cur[j]^predicted[j];phase=5;break;
 case RLE:require(5);packed=rle(residual);phase=6;break;
 case PUSH:require(6);enc.layers.push_back({modes.at(level++),std::move(packed)});cur=std::move(parent);n=h;phase=2;break;
 case PACK:require(7);enc.root=cur.at(0);candidate=archive(enc,side,original);phase=8;break;
 case UNPACK:{require(dec_only?0:8);const auto& b=candidate;check(b.size()>=41&&b[0]=='O'&&b[1]=='3'&&b[2]=='6'&&b[3]=='1',"bad archive");size_t j=4;unsigned s=rd32(b,j);sidecheck(s);if(!dec_only)check(s==side,"side mismatch");side=s;enc={b[j++],{}};for(unsigned k=0;k<levels(side);++k){check(j<b.size()-32,"missing mode");unsigned m=b[j++];check(m<6,"bad mode");size_t len=rd32(b,j);check(j<=b.size()-32&&len<=b.size()-32-j,"bad residual length");enc.layers.push_back({m,Bytes(b.begin()+j,b.begin()+j+len)});j+=len;}check(j==b.size()-32,"trailing data");std::copy_n(b.begin()+j,32,expected.begin());cur={enc.root};n=1;level=enc.layers.size();phase=9;break;}
 case POP:require(9);if(!level){phase=12;pc+=4;break;}popped=enc.layers.at(--level);parent=cur;h=n;n*=2;phase=91;break;
 case UNRLE:require(91);residual=unrle(popped.r,volume(n));phase=10;break;
 case RESTORE:require(11);cur.resize(residual.size());for(size_t j=0;j<cur.size();++j)cur[j]=residual[j]^predicted[j];phase=9;break;
 case CMP:require(13);verified=actual==expected&&(dec_only||cur==source);check(verified,"CTR byte/hash verification failed");phase=14;break;
 case COST:require(14);cost=candidate.size();phase=15;break;
 case COMMIT:require(15);accepted=verified&&(best.empty()||cost<best.size());if(accepted){if(best.empty())baseline=cost;else ++accepts;best=candidate;best_modes=modes;if(trace)std::cout<<"COMMIT bytes="<<cost<<" SHA256="<<hex(actual)<<'\n';}phase=16;break;
 case ROLLBACK:require(16);if(!accepted){++rejects;if(trace)std::cout<<"ROLLBACK bytes="<<cost<<" incumbent="<<best.size()<<'\n';}candidate.clear();phase=1;break;
 case LOOP:if(i.f==0){require(2);if(n>1)pc=i.k;else phase=7;}else if(i.f==1){require(9);if(level)pc=i.k;else phase=12;}else if(i.f==2){require(1);pc=i.k;}else throw std::runtime_error("invalid loop flag");break;
 case HALT:check(dec_only?(phase==14&&verified):(phase==1&&!best.empty()),"premature halt");return;
 default:throw std::runtime_error("invalid codec opcode");}}throw std::runtime_error("codec missing HALT");}
};
Bytes binary(const Program& p){Bytes b;for(U64 w:p)for(unsigned j=0;j<8;++j)b.push_back(Byte(w>>(8*j)));return b;}
Program parse(const Bytes& b){check(!b.empty()&&b.size()%8==0&&b.size()<=4096,"invalid bytecode length");Program p;for(size_t j=0;j<b.size();j+=8){U64 w=0;for(unsigned k=0;k<8;++k)w|=U64(b[j+k])<<(8*k);p.push_back(w);}return p;}
// Only canonical instruction streams accepted from files; avoids arbitrary untrusted branches.
Program load_program(const std::string& path,const Program& expected){auto p=parse(read(path));check(p==expected,"unsupported or corrupt bytecode program");return p;}
void listing(const Program& p){for(size_t j=0;j<p.size();++j)std::cout<<std::setw(2)<<j<<"  "<<word(p[j])<<"  "<<name(I(p[j]).op)<<'\n';}
void tests(){legacy_selftest();size_t candidates=0,commits=0,rollbacks=0;for(unsigned j=0;j<1000;++j){unsigned s=1u<<(j%4);Bytes x(volume(s));for(size_t k=0;k<x.size();++k)x[k]=j%3==0?Byte(mix(k+j*101)):j%3==1?Byte((k/8+j)%256):Byte(j);CodecVM v(x,s);v.run(codec_program());check(v.attempts==v.accepts+v.rejects&&v.best.size()<=v.baseline,"transaction invariant");check(decode(v.best)==x,"legacy cross decode");CodecVM d(v.best);d.run(decoder_program());check(d.cur==x,"VM decode mismatch");auto a=archive(fold(x,s,std::vector<unsigned>(levels(s),0)),s,digest(x));CodecVM d2(a);d2.run(decoder_program());check(d2.cur==x,"legacy to VM mismatch");auto bad=v.best;if(bad.size()<41)throw std::runtime_error("invalid archive");bad.back()^=1;bool rejected=false;try{CodecVM b(bad);b.run(decoder_program());}catch(const std::exception&){rejected=true;}check(rejected,"bad digest accepted");candidates+=v.attempts;commits+=v.accepts;rollbacks+=v.rejects;}
 ANNVM a(7);Model ref(7);size_t count=0;for(unsigned t=0;t<7;++t)for(unsigned z=0;z<7;++z)for(unsigned y=0;y<7;++y)for(unsigned x=0;x<7;++x){check(a.eval({x,y,z},t)==ref.get({x,y,z},t),"ANN VM reference mismatch");++count;}ANNVM q;Bytes tr;for(unsigned t=0;t<=8;++t)tr.push_back(q.eval({L-1,0,1},t));check(tr==Bytes{138,68,69,58,51,45,39,34,30},"virtual trace mismatch");check(parse(binary(codec_program()))==codec_program(),"bytecode roundtrip");std::cout<<"VM roundtrips=1000 legacy_cross_decodes=2000 digest_rejections=1000 candidates="<<candidates<<" commits="<<commits<<" rollbacks="<<rollbacks<<" ANN_comparisons="<<count<<"\nvirtual_trace=";for(Byte b:tr)std::cout<<unsigned(b)<<' ';std::cout<<"\nANN_instructions="<<q.steps<<" cached_states="<<q.cache.size()<<"\nSVM SELFTEST PASS\n";}
void demo(){unsigned s=32;Bytes x(volume(s));for(unsigned z=0;z<s;++z)for(unsigned y=0;y<s;++y)for(unsigned a=0;a<s;++a)x[ix(a,y,z,s)]=Byte(40+2*(a/4)+3*(y/4)+4*(z/4));CodecVM v(x,s);v.run(codec_program());std::cout<<"DEMO structured 32^3 raw="<<x.size()<<" baseline="<<v.baseline<<" best="<<v.best.size()<<" candidates="<<v.attempts<<" commits="<<v.accepts<<" rollbacks="<<v.rejects<<" VM_instructions="<<v.steps<<"\n";CodecVM d(v.best);d.run(decoder_program());check(d.cur==x,"demo decode");ANNVM a;Bytes tile(volume(8));for(unsigned z=0;z<8;++z)for(unsigned y=0;y<8;++y)for(unsigned b=0;b<8;++b)tile[ix(b,y,z,8)]=a.eval({(L-4+b)%L,y,z},3);CodecVM v2(tile,8);v2.run(codec_program());std::cout<<"DEMO ANN tile raw="<<tile.size()<<" best="<<v2.best.size()<<" sha256="<<hex(digest(tile))<<" ANN_instructions="<<a.steps<<" codec_instructions="<<v2.steps<<'\n';}
}
int main(int argc,char** argv){using namespace svm;try{std::string c=argc>1?argv[1]:"";if(c=="selftest"&&argc==2)tests();else if(c=="demo"&&argc==2)demo();else if(c=="emit"&&argc==4){std::string type=argv[2];check(type=="ann"||type=="encode"||type=="decode","program type");auto p=type=="ann"?ann_program():type=="encode"?codec_program():decoder_program();write(argv[3],binary(p));listing(p);}else if(c=="query"&&(argc==6||argc==7)){ANNVM v;if(argc==7)v.code=load_program(argv[6],ann_program());P p{std::stoull(argv[2]),std::stoull(argv[3]),std::stoull(argv[4])};v.trace=true;auto b=v.eval(p,num(argv[5]));std::cout<<"address="<<decimal(addr(p))<<" value="<<unsigned(b)<<" cached="<<v.cache.size()<<" instructions="<<v.steps<<'\n';}else if(c=="encode"&&(argc==5||argc==6)){CodecVM v(read(argv[3]),num(argv[2]));v.trace=true;v.run(argc==6?load_program(argv[5],codec_program()):codec_program());write(argv[4],v.best);std::cout<<"raw="<<v.source.size()<<" baseline="<<v.baseline<<" best="<<v.best.size()<<" candidates="<<v.attempts<<" commits="<<v.accepts<<" rollbacks="<<v.rejects<<" instructions="<<v.steps<<'\n';}else if(c=="decode"&&(argc==4||argc==5)){CodecVM v(read(argv[2]));v.trace=true;v.run(argc==5?load_program(argv[4],decoder_program()):decoder_program());write(argv[3],v.cur);std::cout<<"verified bytes="<<v.cur.size()<<" instructions="<<v.steps<<'\n';}else{std::cout<<"Usage: svm selftest | demo | emit ann|encode|decode FILE | query X Y Z T [PROGRAM] | encode SIDE RAW ARCHIVE [PROGRAM] | decode ARCHIVE RAW [PROGRAM]\n";return c.empty()?0:2;}return 0;}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
