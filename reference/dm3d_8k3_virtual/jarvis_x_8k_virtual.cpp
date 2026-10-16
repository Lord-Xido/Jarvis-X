// JARVIS-X 8K^3 virtual tile reference: bit-exact 2-bit geometric codec.
// C++17; bounded finite tiles. No dense 8192^3 memory allocation.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

constexpr int WORLD=8192, TILE=16, TILES=WORLD/TILE, VOXELS=TILE*TILE*TILE;
constexpr int RAW_BYTES=VOXELS*2/8;
using Raw=std::array<uint8_t,RAW_BYTES>;
uint8_t get2(const Raw& a,int i){return (a.at(i>>2)>>(2*(i&3)))&3u;}
void set2(Raw& a,int i,uint8_t v){auto& c=a.at(i>>2);int s=2*(i&3);c=uint8_t((c&~(3u<<s))|((v&3u)<<s));}
uint8_t gb2(const std::vector<uint8_t>& a,int i){return (a.at(i>>2)>>(2*(i&3)))&3u;}
void sb2(std::vector<uint8_t>& a,int i,uint8_t v){int s=2*(i&3);a.at(i>>2)=uint8_t((a.at(i>>2)&~(3u<<s))|((v&3u)<<s));}
uint64_t fnv(const Raw& a){uint64_t h=14695981039346656037ULL;for(uint8_t b:a){h^=b;h*=1099511628211ULL;}return h;}
uint64_t morton(uint16_t x,uint16_t y,uint16_t z){uint64_t m=0;for(int b=0;b<13;++b){m|=uint64_t((x>>b)&1u)<<(3*b);m|=uint64_t((y>>b)&1u)<<(3*b+1);m|=uint64_t((z>>b)&1u)<<(3*b+2);}return m;}
std::array<uint16_t,3> unmorton(uint64_t m){std::array<uint16_t,3> a{};for(int b=0;b<13;++b)for(int j=0;j<3;++j)a[j]|=uint16_t(((m>>(3*b+j))&1u)<<b);return a;}
int offset(int x,int y,int z){return (z*TILE+y)*TILE+x;}
struct Coord{int x,y,z;};
double dune_height(double gx,double gz){
 return 4096.+900.*std::sin((gx-4096.)/1200.)*std::cos((gz-4096.)/900.)+250.*std::sin((gx+gz)/300.);
}
Raw generate(Coord t){
 Raw r{};
 for(int z=0;z<TILE;++z)for(int x=0;x<TILE;++x){
  const int gx=t.x*TILE+x,gz=t.z*TILE+z;const double surf=dune_height(gx,gz);
  for(int y=0;y<TILE;++y){
   const int gy=t.y*TILE+y;uint8_t v=0;
   if(gy<surf-28.)v=1;else if(gy<=surf)v=2;
   const double dd=(gx-4096.)*(gx-4096.)+(gy-3700.)*(gy-3700.)+(gz-3900.)*(gz-3900.);
   if(v==1&&dd<180.*180.)v=3;
   set2(r,offset(x,y,z),v);
  }
 }
 return r;
}
// Predictor blocks: 2-bit majority values + explicit non-matching voxel records.
std::vector<uint8_t> candidate(const Raw& raw,int block){
 const int side=TILE/block,count=side*side*side;
 std::vector<uint8_t> pred((count+3)/4,0),out;
 for(int cz=0;cz<side;++cz)for(int cy=0;cy<side;++cy)for(int cx=0;cx<side;++cx){
  int hist[4]={0,0,0,0};
  for(int dz=0;dz<block;++dz)for(int dy=0;dy<block;++dy)for(int dx=0;dx<block;++dx)
   ++hist[get2(raw,offset(cx*block+dx,cy*block+dy,cz*block+dz))];
  int v=0;for(int k=1;k<4;++k)if(hist[k]>hist[v])v=k;
  sb2(pred,(cz*side+cy)*side+cx,uint8_t(v));
 }
 std::vector<uint8_t> residual;
 for(int z=0;z<TILE;++z)for(int y=0;y<TILE;++y)for(int x=0;x<TILE;++x){
  int ix=offset(x,y,z),coarse=((z/block)*side+y/block)*side+x/block;
  uint8_t truth=get2(raw,ix);
  if(truth!=gb2(pred,coarse)){
   residual.push_back(uint8_t(ix&255));residual.push_back(uint8_t(ix>>8));residual.push_back(truth);
  }
 }
 int corrections=int(residual.size()/3);
 out.reserve(3+pred.size()+residual.size());out.push_back(uint8_t(block));
 out.push_back(uint8_t(corrections&255));out.push_back(uint8_t(corrections>>8));
 out.insert(out.end(),pred.begin(),pred.end());out.insert(out.end(),residual.begin(),residual.end());
 return out;
}
std::vector<uint8_t> encode(const Raw& raw){
 std::vector<uint8_t> best;best.reserve(RAW_BYTES+1);best.push_back(0);
 best.insert(best.end(),raw.begin(),raw.end());
 for(int b:{2,4,8,16}){auto c=candidate(raw,b);if(c.size()<best.size())best=std::move(c);}
 return best;
}
Raw decode(const std::vector<uint8_t>& b){
 if(b.empty())throw std::runtime_error("empty payload");
 Raw out{};
 if(b[0]==0){
  if(b.size()!=RAW_BYTES+1)throw std::runtime_error("incorrect raw payload");
  std::copy(b.begin()+1,b.end(),out.begin());return out;
 }
 int block=b[0];
 if((block!=2&&block!=4&&block!=8&&block!=16)||b.size()<3)throw std::runtime_error("invalid mode");
 int n=int(b[1])+(int(b[2])<<8),side=TILE/block,cells=side*side*side,nb=(cells+3)/4;
 if(n>VOXELS||b.size()!=size_t(3+nb+3*n))throw std::runtime_error("invalid correction count/length");
 std::vector<uint8_t> pred(b.begin()+3,b.begin()+3+nb);
 for(int z=0;z<TILE;++z)for(int y=0;y<TILE;++y)for(int x=0;x<TILE;++x){
  int coarse=((z/block)*side+y/block)*side+x/block;
  set2(out,offset(x,y,z),gb2(pred,coarse));
 }
 std::array<bool,VOXELS> seen{};
 for(int k=0;k<n;++k){
  int at=3+nb+3*k,ix=int(b[at])+(int(b[at+1])<<8),value=int(b[at+2]);
  if(ix>=VOXELS||value>3||seen[ix])throw std::runtime_error("invalid/duplicate correction");
  seen[ix]=true;set2(out,ix,uint8_t(value));
 }
 return out;
}
double entropy(const Raw& a){
 int hist[4]={0,0,0,0};for(int i=0;i<VOXELS;++i)++hist[get2(a,i)];
 double h=0;for(int c:hist)if(c){double p=double(c)/VOXELS;h-=p*std::log2(p);}return h;
}
struct Choice{Coord c;double h;uint64_t key;};
struct Observation{Coord c;double h;uint64_t key;size_t bytes;int mode;uint64_t digest;};
uint32_t tile_id(Coord c){return (uint32_t(c.z)*TILES+uint32_t(c.y))*TILES+uint32_t(c.x);}
uint64_t origin_morton(Coord c){return morton(uint16_t(c.x*TILE),uint16_t(c.y*TILE),uint16_t(c.z*TILE));}
Coord sample(std::mt19937_64& rng){
 int tx=int(rng()%TILES),tz=int(rng()%TILES);
 double h=dune_height(double(tx*TILE+8),double(tz*TILE+8));
 int ty=int(std::floor(h/TILE))+int(rng()%7)-3;
 return {tx,std::clamp(ty,0,TILES-1),tz};
}
void octree_path(std::unordered_set<uint64_t>& nodes,uint64_t key){
 for(int depth=1;depth<=9;++depth){
  uint64_t prefix=key>>(39-3*depth);nodes.insert((uint64_t(depth)<<39)|prefix);
 }
}
void self_test(){
 for(auto a:std::vector<std::array<uint16_t,3>>{{0,0,0},{8191,8191,8191},{123,4096,7777}})
  if(unmorton(morton(a[0],a[1],a[2]))!=a)throw std::runtime_error("Morton mismatch");
 std::mt19937_64 rng(123);
 for(int j=0;j<25;++j){
  Raw a{};for(int i=0;i<VOXELS;++i)set2(a,i,j<4?uint8_t(j):uint8_t(rng()%4));
  if(decode(encode(a))!=a)throw std::runtime_error("codec mismatch");
 }
 std::cout<<"SELFTEST PASS: 39-bit Morton and exact 2-bit tile codec\n";
}
void write16(std::ofstream& f,uint16_t v){
 const unsigned char b[2]={static_cast<unsigned char>(v&255u),static_cast<unsigned char>((v>>8)&255u)};
 f.write(reinterpret_cast<const char*>(b),2);
}
int optint(int argc,char** argv,const std::string& key,int def){
 for(int i=1;i+1<argc;++i)if(key==argv[i]){
  int v=std::stoi(argv[i+1]);
  if(v<1||v>100000)throw std::runtime_error("bad value for "+key);
  return v;
 }
 return def;
}
std::string optstr(int argc,char** argv,const std::string& key,const std::string& def){
 for(int i=1;i+1<argc;++i)if(key==argv[i])return argv[i+1];return def;
}
int main(int argc,char** argv){
 try{
  for(int i=1;i<argc;++i){
   if(std::string(argv[i])=="--self-test"){self_test();return 0;}
   if(std::string(argv[i])=="--help"){
    std::cout<<"Usage: jarvisx_dm3d_8k3_virtual --rounds 3 --budget 64 --candidates 256 --seed 42 --report report.json --tiles raw.bin --encoded packet.jx8k\n";return 0;
   }
  }
  int rounds=optint(argc,argv,"--rounds",3),candidates=optint(argc,argv,"--candidates",256);
  int budget=optint(argc,argv,"--budget",64),seed=optint(argc,argv,"--seed",42);
  if(rounds>16||candidates>4096||budget>256||budget>candidates||rounds*budget>1024)
   throw std::runtime_error("bounded exploration budget exceeded");
  std::string report=optstr(argc,argv,"--report","jx8k_report.json");
  std::string raw_name=optstr(argc,argv,"--tiles","jx8k_raw.bin");
  std::string enc_name=optstr(argc,argv,"--encoded","jx8k_tiles.jx8k");
  std::ofstream raw(raw_name,std::ios::binary),packet(enc_name,std::ios::binary);
  if(!raw||!packet)throw std::runtime_error("cannot open output");
  const std::array<char,8> magic{{'J','X','8','K','T','1',0,0}};
  packet.write(magic.data(),magic.size());size_t bytes=8,raw_bytes=0,encoded_bytes=0,verified=0,rejected=0;
  std::mt19937_64 rng{uint64_t(seed)};
  std::unordered_set<uint32_t> explored;std::unordered_set<uint64_t> octnodes;
  std::vector<Observation> observations;std::array<size_t,5> modes{};
  double total_entropy=0;auto t0=std::chrono::steady_clock::now();
  std::cout<<"VIRTUAL 8192^3: 2^39 addresses | 16^3 2-bit tiles, bounded exploration\n";
  self_test();
  for(int round=0;round<rounds;++round){
   std::vector<Choice> pool;pool.reserve(candidates);std::unordered_set<uint32_t> sampled;
   for(int j=0;j<candidates;++j){
    Coord c=sample(rng);uint32_t id=tile_id(c);
    if(explored.count(id)||!sampled.insert(id).second)continue;
    auto tile=generate(c);pool.push_back({c,entropy(tile),origin_morton(c)});
   }
   std::sort(pool.begin(),pool.end(),[](const Choice& a,const Choice& b){
    if(a.h!=b.h)return a.h>b.h;return a.key<b.key;
   });
   size_t selected=std::min<size_t>(budget,pool.size());
   for(size_t j=0;j<selected;++j){
    const auto& choice=pool[j];Raw original=generate(choice.c);
    auto compressed=encode(original);Raw reconstructed=decode(compressed);
    if(reconstructed!=original||fnv(reconstructed)!=fnv(original)){++rejected;continue;}
    explored.insert(tile_id(choice.c));octree_path(octnodes,choice.key);
    observations.push_back({choice.c,choice.h,choice.key,compressed.size(),int(compressed[0]),fnv(original)});
    raw.write(reinterpret_cast<const char*>(original.data()),original.size());
    if(compressed.size()>65535)throw std::runtime_error("payload exceeds uint16 length");
    write16(packet,uint16_t(choice.c.x));write16(packet,uint16_t(choice.c.y));
    write16(packet,uint16_t(choice.c.z));write16(packet,uint16_t(compressed.size()));
    packet.write(reinterpret_cast<const char*>(compressed.data()),std::streamsize(compressed.size()));
    bytes+=8+compressed.size();raw_bytes+=RAW_BYTES;encoded_bytes+=compressed.size();
    total_entropy+=choice.h;++verified;
    ++modes[compressed[0]==0?0:(compressed[0]==2?1:compressed[0]==4?2:compressed[0]==8?3:4)];
   }
  }
  raw.close();packet.close();if(!raw||!packet)throw std::runtime_error("output write failed");
  double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  std::ofstream f(report);if(!f)throw std::runtime_error("cannot write report");
  f<<std::fixed<<std::setprecision(6)<<"{\n"
   <<" \"world_side\":8192,\"voxel_count\":549755813888,\"voxel_bits\":2,\"voxel_address_bits\":39,\n"
   <<" \"tile_side\":16,\"raw_tile_bytes\":1024,\"possible_tiles\":134217728,\n"
   <<" \"tower\":\"symbolic: b^(b^(b^(b^(b^(s^(s^s)))))) with b=1024 and s=10^24\",\n"
   <<" \"rounds\":"<<rounds<<",\"candidate_budget_per_round\":"<<candidates
   <<",\"explore_budget_per_round\":"<<budget<<",\"seed\":"<<seed<<",\n"
   <<" \"tiles_verified\":"<<verified<<",\"tiles_rejected\":"<<rejected
   <<",\"materialized_raw_bytes\":"<<raw_bytes<<",\"encoded_bytes\":"<<encoded_bytes
   <<",\"packet_file_bytes\":"<<bytes
   <<",\"packet_compression_ratio\":"<<(bytes?double(raw_bytes)/bytes:0)
   <<",\"compression_ratio\":"<<(encoded_bytes?double(raw_bytes)/encoded_bytes:0)
   <<",\"mean_selected_entropy_bits\":"<<(verified?total_entropy/verified:0)
   <<",\"materialized_octree_nodes\":"<<octnodes.size()<<",\"elapsed_seconds\":"<<elapsed<<",\n"
   <<" \"codec_modes\":{\"raw\":"<<modes[0]<<",\"block2\":"<<modes[1]
   <<",\"block4\":"<<modes[2]<<",\"block8\":"<<modes[3]<<",\"block16\":"<<modes[4]<<"},\n"
   <<" \"tile_file\":\""<<raw_name<<"\",\"encoded_file\":\""<<enc_name<<"\",\n"
   <<" \"observations\":[\n";
  for(size_t i=0;i<observations.size();++i){
   const auto& o=observations[i];
   f<<" {\"x\":"<<o.c.x<<",\"y\":"<<o.c.y<<",\"z\":"<<o.c.z
    <<",\"entropy\":"<<o.h<<",\"morton\":"<<o.key
    <<",\"compressed_bytes\":"<<o.bytes<<",\"block_mode\":"<<o.mode
    <<",\"fnv1a_hex\":\""<<std::hex<<std::setw(16)<<std::setfill('0')<<o.digest
    <<std::dec<<std::setfill(' ')<<"\"}"<<(i+1<observations.size()?",":"")<<"\n";
  }
  f<<" ]\n}\n";f.close();
  std::cout<<"VERIFIED "<<verified<<" tiles, rejected "<<rejected<<"; raw "<<raw_bytes
   <<" bytes; serialized "<<bytes<<" bytes; octree paths "<<octnodes.size()
   <<"; duration "<<elapsed<<"s\n";
  return rejected?2:0;
 }catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<"\n";return 1;}
}
