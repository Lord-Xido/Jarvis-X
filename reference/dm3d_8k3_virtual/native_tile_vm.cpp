// JX8K native typed tile consumer — isolated reference profile, never a production DM3D opcode extension.
// C++17, bounded physical memory; transaction commit/rollback affects only a local sandbox state.
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jx8k {
constexpr size_t RAW_SIZE=1024, VALUES=4096, MAX_TILES=1024, MAX_PACKET=2u*1024u*1024u;
constexpr std::array<uint8_t,8> MAGIC{{'J','X','8','K','T','1',0,0}};
constexpr std::array<uint8_t,7> PROGRAM{{0x70,0x71,0x72,0x73,0x74,0x75,0xff}};
using Raw=std::array<uint8_t,RAW_SIZE>;

uint16_t u16(const std::vector<uint8_t>& p,size_t pos){
 if(pos>p.size() || p.size()-pos<2) throw std::runtime_error("truncated uint16");
 return uint16_t(uint16_t(p[pos]) | (uint16_t(p[pos+1])<<8));
}
uint8_t get2(const Raw& r,size_t i){return uint8_t((r.at(i/4)>>(2*(i%4)))&3u);}
void set2(Raw& r,size_t i,uint8_t v){const unsigned s=unsigned(2*(i%4));
 r.at(i/4)=uint8_t((r.at(i/4)& ~(3u<<s)) | ((v&3u)<<s));}
uint64_t morton39(unsigned x,unsigned y,unsigned z){
 if(x>=8192 || y>=8192 || z>=8192)throw std::runtime_error("coordinate outside 8192 cube");
 uint64_t r=0;for(unsigned i=0;i<13;++i){r|=uint64_t((x>>i)&1u)<<(3*i);
 r|=uint64_t((y>>i)&1u)<<(3*i+1);r|=uint64_t((z>>i)&1u)<<(3*i+2);}return r;
}
uint64_t fnv(const Raw& r){uint64_t h=14695981039346656037ULL;
 for(uint8_t b:r){h^=b;h*=1099511628211ULL;}return h;}
std::vector<uint8_t> read_file(const std::string& path,size_t max_bytes){
 std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("cannot open "+path);
 auto length=f.tellg();if(length<0 || uint64_t(length)>max_bytes)throw std::runtime_error("input exceeds byte bound");
 std::vector<uint8_t> b(size_t(length),uint8_t(0));f.seekg(0);
 if(!b.empty())f.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size()));
 if(!f)throw std::runtime_error("failed to read full input");
 return b;
}
void check_program(const std::vector<uint8_t>& p){
 if(p.size()!=56)throw std::runtime_error("exact 7-word JX8K program required");
 for(size_t i=0;i<PROGRAM.size();++i){
  if(p[i*8]!=PROGRAM[i])throw std::runtime_error("unsupported opcode or instruction ordering");
  for(size_t k=1;k<8;++k)if(p[i*8+k]!=0)throw std::runtime_error("nonzero VM operand/reserved bits");
 }
}
std::vector<uint8_t> default_program(){std::vector<uint8_t> p(56,0);
 for(size_t i=0;i<PROGRAM.size();++i)p[i*8]=PROGRAM[i];
 return p;}
Raw decode(const std::vector<uint8_t>& p,size_t start,size_t size){
 if(size==0 || start>p.size() || size>p.size()-start)throw std::runtime_error("bad payload bounds");
 uint8_t mode=p[start];Raw out{};
 if(mode==0){if(size!=RAW_SIZE+1)throw std::runtime_error("raw payload width");
  for(size_t i=0;i<RAW_SIZE;++i)out[i]=p[start+1+i];
  return out;}
 if((mode!=2 && mode!=4 && mode!=8 && mode!=16) || size<3)throw std::runtime_error("invalid block mode");
 unsigned count=u16(p,start+1);unsigned side=16u/mode;
 size_t nblocks=size_t(side)*side*side,nb=(nblocks+3)/4;
 if(count>VALUES || size!=3+nb+size_t(count)*3)throw std::runtime_error("bad corrections length");
 const size_t pred=start+3;
 for(unsigned z=0;z<16;++z)for(unsigned y=0;y<16;++y)for(unsigned x=0;x<16;++x){
  unsigned ci=((z/mode)*side+y/mode)*side+x/mode;
  unsigned shift=(ci%4)*2;
  set2(out,(size_t(z)*16+y)*16+x,uint8_t((p[pred+ci/4]>>shift)&3u));
 }
 std::array<bool,VALUES> seen{};
 size_t ex=pred+nb;
 for(unsigned j=0;j<count;++j){unsigned offset=u16(p,ex+size_t(j)*3);
  unsigned val=p[ex+size_t(j)*3+2];
  if(offset>=VALUES || val>3 || seen[offset])throw std::runtime_error("invalid duplicate/out-of-range correction");
  seen[offset]=true;set2(out,offset,uint8_t(val));
 }
 return out;
}
struct Observation {unsigned x,y,z;uint64_t morton;std::array<unsigned,4> counts{};uint64_t fnv64;};
struct Result {bool sandbox_committed=false;bool rollback=false;bool unchanged=false;
 size_t tiles=0,staged=0,steps=0,bytes=0,state_after=0;std::string reason;
 std::vector<Observation> observations;};
class Sandbox {
 public:
  Sandbox(){Raw marker{};marker[0]=0xE4;state_.emplace(morton39(8191,8191,8191),marker);}
  // This map is deliberately *not* the canonical Jarvis-X runtime state.
  size_t size() const{return state_.size();}
  Result run(const std::vector<uint8_t>& packet,const std::vector<uint8_t>& truth,
             const std::vector<uint8_t>& program,int reject_index=-1){
    check_program(program);
    if(packet.size()>MAX_PACKET || truth.size()>MAX_TILES*RAW_SIZE || truth.empty() || truth.size()%RAW_SIZE)
       throw std::runtime_error("resource limit or invalid independent reference");
    if(reject_index>=0 && size_t(reject_index)>=truth.size()/RAW_SIZE)
       throw std::runtime_error("reject injection outside selected tile count");
    if(packet.size()<MAGIC.size())throw std::runtime_error("truncated format header");
    for(size_t i=0;i<MAGIC.size();++i)if(packet[i]!=MAGIC[i])throw std::runtime_error("incompatible JX8KT1 version");
    std::map<uint64_t,Raw> staged=state_;std::unordered_set<uint64_t> seen;
    Result result;result.tiles=truth.size()/RAW_SIZE;result.bytes=packet.size();
    size_t cursor=MAGIC.size();
    for(size_t i=0;i<result.tiles;++i){
      if(cursor>packet.size() || packet.size()-cursor<8)throw std::runtime_error("truncated tile header");
      unsigned x=u16(packet,cursor),y=u16(packet,cursor+2),z=u16(packet,cursor+4);
      size_t len=u16(packet,cursor+6);cursor+=8;
      if(x>=512 || y>=512 || z>=512 || len==0 || cursor>packet.size() || len>packet.size()-cursor)
         throw std::runtime_error("invalid tile address/packet length");
      uint64_t key=morton39(x*16,y*16,z*16);
      if(!seen.insert(key).second)throw std::runtime_error("duplicate tile coordinate");
      Raw reconstructed=decode(packet,cursor,len);cursor+=len;
      Raw reference{};for(size_t j=0;j<RAW_SIZE;++j)reference[j]=truth[i*RAW_SIZE+j];
      std::array<float,VALUES> tensor{};std::array<unsigned,4> counts{};
      // MAP_FLOAT32 and REPACK_2BIT are inverse for exactly represented states 0..3.
      for(size_t j=0;j<VALUES;++j){auto v=get2(reconstructed,j);tensor[j]=float(v);++counts[v];}
      Raw roundtrip{};
      for(size_t j=0;j<VALUES;++j){auto v=unsigned(tensor[j]);if(v>3 || float(v)!=tensor[j])throw std::runtime_error("non-integral mapped voxel");set2(roundtrip,j,uint8_t(v));}
      if(roundtrip!=reconstructed || roundtrip!=reference)throw std::runtime_error("CTR bitwise mismatch against independent raw bytes");
      staged[key]=roundtrip; // Provisional only; authoritative sandbox not touched.
      result.observations.push_back({x,y,z,key,counts,fnv(reference)});
      ++result.staged;result.steps+=6;
      if(reject_index>=0 && i==size_t(reject_index)){
        result.rollback=true;result.unchanged=true;result.reason="deliberate verification rejection";
        result.state_after=state_.size();return result; // candidate destroyed; rollback exact.
      }
    }
    if(cursor!=packet.size())throw std::runtime_error("trailing packet bytes");
    if(result.staged!=result.tiles)throw std::runtime_error("incomplete transaction");
    state_.swap(staged); // Single local in-memory sandbox promotion: all tiles or none.
    result.sandbox_committed=true;result.steps+=1;result.state_after=state_.size();
    return result;
  }
 private:
  std::map<uint64_t,Raw> state_;
};
std::string json(const Result& r){
 std::ostringstream s;s<<"{\n\"schema\":\"JX8K-NATIVE-TILEVM-1\",\"authority\":\"sandbox_only\",\"production_state_modified\":false,"
 <<"\"commit_eligible\":false,\"sandbox_committed\":"<<(r.sandbox_committed?"true":"false")
 <<",\"rolled_back\":"<<(r.rollback?"true":"false")<<",\"state_unchanged_after_rollback\":"<<(r.unchanged?"true":"false")
 <<",\"tiles_requested\":"<<r.tiles<<",\"tiles_staged\":"<<r.staged
 <<",\"instruction_steps\":"<<r.steps<<",\"packet_bytes\":"<<r.bytes
 <<",\"sandbox_records_after\":"<<r.state_after<<",\"tensor_dtype\":\"float32\",\"voxels_per_tile\":4096,"
 <<"\"fixed_point\":\"not_evaluated\",\"external_world\":\"not_evaluated\",\"observations\":[";
 for(size_t i=0;i<r.observations.size();++i){auto& a=r.observations[i];if(i)s<<",";
 s<<"{\"tile_coord\":["<<a.x<<","<<a.y<<","<<a.z<<"],\"morton39\":"<<a.morton<<",\"counts\":[";
 for(int j=0;j<4;++j){if(j)s<<",";s<<a.counts[size_t(j)];}
 s<<"],\"fnv1a_hex\":\""<<std::hex<<std::setw(16)<<std::setfill('0')<<a.fnv64<<std::dec<<"\"}";
 }s<<"]}\n";return s.str();
}
void save_atomic(const std::string& path,const std::string& data){
 const std::string tmp=path+".tmp";
 {std::ofstream f(tmp,std::ios::binary|std::ios::trunc);
 if(!f)throw std::runtime_error("cannot open receipt output");
 f<<data;
 if(!f){std::remove(tmp.c_str());throw std::runtime_error("receipt write failure");}}
 if(std::rename(tmp.c_str(),path.c_str())!=0){std::remove(tmp.c_str());throw std::runtime_error("receipt atomic rename failed");}
}
void self_test(){
 if(morton39(8191,8191,8191)!=(1ULL<<39)-1)throw std::runtime_error("morton bad");
 auto p=default_program();check_program(p);auto fake=p;fake[2]=1;
 try{check_program(fake);throw std::runtime_error("invalid VM instruction accepted");}catch(const std::runtime_error& e){if(std::string(e.what())=="invalid VM instruction accepted")throw;}
 Raw a{};for(size_t i=0;i<VALUES;++i)set2(a,i,uint8_t(i&3));
 std::vector<uint8_t> enc(1,0);enc.insert(enc.end(),a.begin(),a.end());
 if(decode(enc,0,enc.size())!=a)throw std::runtime_error("raw codec mismatch");
 std::cout<<"NATIVE-TILEVM SELFTEST PASS\n";
}
} // namespace jx8k
int main(int argc,char** argv){
 try{
  std::string packet,raw,program,receipt;int reject=-1;
  for(int i=1;i<argc;++i){std::string a=argv[i];
   if(a=="--self-test"){jx8k::self_test();return 0;}
   if(i+1>=argc)throw std::runtime_error("missing CLI value");
   std::string v=argv[++i];
   if(a=="--encoded")packet=v;else if(a=="--raw")raw=v;
   else if(a=="--program")program=v;else if(a=="--receipt")receipt=v;
   else if(a=="--force-reject-at"){reject=std::stoi(v);if(reject<0 || reject>=1024)throw std::runtime_error("invalid reject index");}
   else throw std::runtime_error("unknown CLI argument: "+a);
  }
  if(packet.empty()||raw.empty())throw std::runtime_error("required: --encoded PATH --raw PATH");
  const auto pb=jx8k::read_file(packet,jx8k::MAX_PACKET);
  const auto rb=jx8k::read_file(raw,jx8k::MAX_TILES*jx8k::RAW_SIZE);
  const auto code=program.empty()?jx8k::default_program():jx8k::read_file(program,128);
  jx8k::Sandbox sandbox;auto result=sandbox.run(pb,rb,code,reject);
  auto serialized=jx8k::json(result);if(!receipt.empty())jx8k::save_atomic(receipt,serialized);
  std::cout<<serialized;return result.rollback?2:0;
 }catch(const std::exception& e){std::cerr<<"native JX8K VM rejected: "<<e.what()<<"\n";return 1;}
}
