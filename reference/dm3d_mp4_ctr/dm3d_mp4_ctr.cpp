// DM3D MP4 -> 3D byte tiles -> bounded VM -> verification -> atomic output.
// C++17, no external dependencies. Processes media bytes, not decoded video frames.
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace dm {
constexpr std::uint64_t AXIS = 10ull * 1024 * 1024;
constexpr std::size_t TILE = 1024;
constexpr std::size_t RAM = 1024 * 1024;
constexpr std::uint64_t MAX_TRANSACTION_BYTES = 128ull * 1024 * 1024;
constexpr std::uint64_t TX = AXIS/8, TY=AXIS/8, TZ=AXIS/16;
constexpr std::array<char,8> MAGIC{{'J','X','M','P','4','T','1','\0'}};
constexpr std::uint64_t MAX_TILES=std::numeric_limits<std::uint32_t>::max();

[[noreturn]] void fail(const std::string& reason){ throw std::runtime_error(reason); }
void require(bool c,const std::string& why){ if(!c)fail(why); }

void write_u8(std::ostream& o,std::uint8_t x){o.put(static_cast<char>(x));require(bool(o),"write failed");}
void write_u16(std::ostream& o,std::uint16_t x){for(int j=0;j<2;++j)write_u8(o,static_cast<std::uint8_t>(x>>(8*j)));}
void write_u32(std::ostream& o,std::uint32_t x){for(int j=0;j<4;++j)write_u8(o,static_cast<std::uint8_t>(x>>(8*j)));}
void write_u64(std::ostream& o,std::uint64_t x){for(int j=0;j<8;++j)write_u8(o,static_cast<std::uint8_t>(x>>(8*j)));}
std::uint8_t read_u8(std::istream& i){int c=i.get();require(c!=EOF,"truncated archive");return static_cast<std::uint8_t>(c);}
std::uint16_t read_u16(std::istream& i){std::uint16_t x=0;for(int j=0;j<2;++j)x|=static_cast<std::uint16_t>(std::uint16_t(read_u8(i))<<(8*j));return x;}
std::uint32_t read_u32(std::istream& i){std::uint32_t x=0;for(int j=0;j<4;++j)x|=std::uint32_t(read_u8(i))<<(8*j);return x;}
std::uint64_t read_u64(std::istream& i){std::uint64_t x=0;for(int j=0;j<8;++j)x|=std::uint64_t(read_u8(i))<<(8*j);return x;}
void read_exact(std::istream& i,std::uint8_t* dst,std::size_t n){if(!n)return;i.read(reinterpret_cast<char*>(dst),static_cast<std::streamsize>(n));require(i.gcount()==static_cast<std::streamsize>(n),"truncated payload");}
std::uint32_t crc_update(std::uint32_t c,const std::uint8_t* p,std::size_t n){
 for(std::size_t k=0;k<n;++k){c^=p[k];for(int j=0;j<8;++j)c=(c>>1)^(0xedb88320u & (0u-(c&1u)));}return c;
}
std::uint32_t crc(const std::uint8_t* p,std::size_t n){return ~crc_update(0xffffffffu,p,n);}
std::vector<std::uint8_t> rle(const std::uint8_t* p,std::size_t n){
 std::vector<std::uint8_t> out;out.reserve(2*n);
 for(std::size_t i=0;i<n;){std::size_t j=i+1;while(j<n && j-i<255 && p[j]==p[i])++j;out.push_back(static_cast<std::uint8_t>(j-i));out.push_back(p[i]);i=j;}
 return out;
}
std::vector<std::uint8_t> un_rle(const std::vector<std::uint8_t>& in,std::size_t n){
 require((in.size()%2)==0,"invalid RLE length");std::vector<std::uint8_t> out;out.reserve(n);
 for(std::size_t k=0;k<in.size();k+=2){const unsigned cnt=in[k];require(cnt>0 && cnt<=n-out.size(),"RLE overflow");out.insert(out.end(),cnt,in[k+1]);}
 require(out.size()==n,"RLE output length mismatch");return out;
}
std::vector<std::uint8_t> predictive(const std::uint8_t* p,std::size_t n){
 std::vector<std::uint8_t> out(n);std::uint8_t prev=0;for(std::size_t k=0;k<n;++k){out[k]=p[k]^prev;prev=p[k];}return out;
}
std::vector<std::uint8_t> invert_pred(const std::vector<std::uint8_t>& p){
 std::vector<std::uint8_t> out(p.size());std::uint8_t prev=0;for(std::size_t k=0;k<p.size();++k){out[k]=p[k]^prev;prev=out[k];}return out;
}
struct Candidate{std::uint8_t mode;std::vector<std::uint8_t> payload;};
Candidate encode(const std::uint8_t* p,std::size_t n){
 std::vector<std::uint8_t> raw(p,p+n);auto xp=predictive(p,n);
 Candidate best{0,raw};
 std::array<Candidate,3> opts{{{1,rle(p,n)},{2,xp},{3,rle(xp.data(),xp.size())}}};
 for(auto& t:opts)if(t.payload.size()<best.payload.size())best=std::move(t);
 return best;
}
std::vector<std::uint8_t> decode(const Candidate& x,std::size_t n){
 if(x.mode==0 || x.mode==2){require(x.payload.size()==n,"raw/predictive length mismatch");return x.mode==0?x.payload:invert_pred(x.payload);}
 if(x.mode==1 || x.mode==3){auto z=un_rle(x.payload,n);return x.mode==1?z:invert_pred(z);}
 fail("unknown codec mode");
}
std::array<std::uint32_t,3> coords(std::uint64_t id){
 require(id<TX*TY*TZ,"coordinate outside sparse 3D domain");
 return {static_cast<std::uint32_t>((id%TX)*8),static_cast<std::uint32_t>(((id/TX)%TY)*8),static_cast<std::uint32_t>((id/(TX*TY))*16)};
}
struct VM{
 std::vector<std::uint8_t> mem;
 std::uint64_t laplace_energy=0, executed=0;
 VM():mem(RAM,0){}
 static std::size_t index(std::size_t x,std::size_t y,std::size_t z){return x+8*(y+8*z);}
 void run(const std::uint8_t* tile,std::size_t n){
  require(n>=1 && n<=TILE,"invalid tile size");
  std::fill(mem.begin(),mem.begin()+2048,0);
  std::copy(tile,tile+n,mem.begin());std::copy(mem.begin(),mem.begin()+TILE,mem.begin()+TILE);
  // Fixed, bounded bytecode: OBSERVE_6 -> XOR -> REFLECT_X -> REFLECT_X -> XOR -> CHECK -> HALT
  constexpr std::array<std::uint8_t,7> PROGRAM{{0x10,0x20,0x30,0x30,0x20,0x40,0xff}};
  for(std::size_t pc=0;pc<PROGRAM.size();++pc){
   ++executed;
   switch(PROGRAM[pc]){
    case 0x10:{ // periodic 6-neighbour Laplacian; observation-only
      for(std::size_t z=0;z<16;++z)for(std::size_t y=0;y<8;++y)for(std::size_t x=0;x<8;++x){
       const std::size_t i=index(x,y,z);
       const int l=int(mem[index((x+1)%8,y,z)])+int(mem[index((x+7)%8,y,z)])+
          int(mem[index(x,(y+1)%8,z)])+int(mem[index(x,(y+7)%8,z)])+
          int(mem[index(x,y,(z+1)%16)])+int(mem[index(x,y,(z+15)%16)])-6*int(mem[i]);
       laplace_energy+=static_cast<std::uint64_t>(l<0?-l:l);
      }break;
    }
    case 0x20:for(std::size_t i=0;i<TILE;++i)mem[i]^=0xa5;break;
    case 0x30:{
     for(std::size_t z=0;z<16;++z)for(std::size_t y=0;y<8;++y)for(std::size_t x=0;x<4;++x){
      std::swap(mem[index(x,y,z)],mem[index(7-x,y,z)]);
     }break;
    }
    case 0x40:require(std::equal(mem.begin(),mem.begin()+TILE,mem.begin()+TILE),"VM CTR invariant failed");break;
    case 0xff:return;
    default:fail("VM illegal instruction");
   }
  }
  fail("VM missing HALT");
 }
};

struct Header{std::uint64_t bytes;std::uint32_t checksum,tiles;};
void write_header(std::ostream& o,const Header& h){
 o.write(MAGIC.data(),MAGIC.size());write_u16(o,1);write_u16(o,0);write_u64(o,h.bytes);write_u32(o,h.checksum);write_u32(o,h.tiles);
}
Header read_header(std::istream& i){
 std::array<char,8> m{};i.read(m.data(),8);require(i.gcount()==8 && m==MAGIC,"invalid archive magic");
 require(read_u16(i)==1,"unsupported format version");require(read_u16(i)==0,"unsupported archive flags");
 Header h{read_u64(i),read_u32(i),read_u32(i)};
 require(h.bytes<=MAX_TRANSACTION_BYTES,"archive exceeds 128 MiB transaction budget");
 require(h.bytes==0 ? h.tiles==0 : ((h.bytes-1)/TILE+1==h.tiles),"header tile-count mismatch");
 return h;
}
void put_record(std::ostream& o,std::uint32_t id,const std::uint8_t* p,std::size_t n){
 const auto c=encode(p,n);require(c.payload.size()<=2*TILE,"internal encoded size error");
 (void)coords(id);write_u32(o,id);write_u32(o,crc(p,n));write_u16(o,static_cast<std::uint16_t>(n));
 write_u16(o,static_cast<std::uint16_t>(c.payload.size()));write_u8(o,c.mode);for(int i=0;i<3;++i)write_u8(o,0);
 o.write(reinterpret_cast<const char*>(c.payload.data()),static_cast<std::streamsize>(c.payload.size()));require(bool(o),"record write error");
}
std::vector<std::uint8_t> get_record(std::istream& i,std::uint32_t expected,std::size_t expected_len){
 const auto id=read_u32(i), checksum=read_u32(i);
 const auto n=read_u16(i), enc=read_u16(i);const auto mode=read_u8(i);
 for(int j=0;j<3;++j)require(read_u8(i)==0,"nonzero reserved byte");
 require(id==expected,"out of order or duplicate tile");(void)coords(id);
 require(n==expected_len && n>0 && n<=TILE && enc<=2*TILE && enc>0,"invalid tile lengths");
 Candidate c{mode,std::vector<std::uint8_t>(enc)};read_exact(i,c.payload.data(),enc);
 auto r=decode(c,n);require(crc(r.data(),n)==checksum,"per-tile CRC mismatch");return r;
}
struct Stage{
 fs::path dir,temp;bool active=true;
 explicit Stage(const fs::path& target){
  // create_directory is atomic: an already-existing stage directory is never reused.
  std::random_device rd;
  for(int i=0;i<64;++i){
   auto name=target.string()+".dm3d-stage-"+std::to_string(rd())+"-"+std::to_string(i);
   std::error_code ec;
   if(fs::create_directory(name,ec)){dir=name;temp=dir/"candidate.bin";return;}
   if(ec && ec!=std::errc::file_exists)fail("cannot create staging directory: "+ec.message());
  }
  fail("unable to acquire unique staging directory");
 }
 ~Stage(){if(active){std::error_code ec;fs::remove_all(dir,ec);}}
 void commit(const fs::path& target){
  require(!fs::exists(target),"refusing to overwrite output");
  fs::rename(temp,target);active=false;
  std::error_code ec;fs::remove(dir,ec);
 }
};
Header pack(const fs::path& in,const fs::path& out){
 require(fs::exists(in)&&fs::is_regular_file(in),"input is not a regular file");require(!fs::exists(out),"refusing to overwrite archive");
 std::ifstream input(in,std::ios::binary);require(bool(input),"cannot open input");
 const std::uint64_t size=fs::file_size(in);require(size<=MAX_TRANSACTION_BYTES,"input exceeds 128 MiB transaction budget");
 const std::uint64_t ntiles=size==0?0:1+(size-1)/TILE;
 require(ntiles<=MAX_TILES,"file exceeds archive tile-count limit");
 Header h{size,0,static_cast<std::uint32_t>(ntiles)};
 Stage stage(out);std::fstream output(stage.temp,std::ios::binary|std::ios::out|std::ios::trunc);require(bool(output),"cannot open staged archive");
 write_header(output,h);std::array<std::uint8_t,TILE> tile{};
 std::uint32_t running=0xffffffffu;std::uint64_t used=0;
 for(std::uint32_t id=0;id<h.tiles;++id){
  const std::size_t n=static_cast<std::size_t>(std::min<std::uint64_t>(TILE,size-used));read_exact(input,tile.data(),n);
  running=crc_update(running,tile.data(),n);put_record(output,id,tile.data(),n);used+=n;
 }
 require(input.peek()==EOF,"source changed while encoding");
 h.checksum=~running;output.seekp(0);write_header(output,h);output.close();require(bool(output),"archive write/close error");
 stage.commit(out);return h;
}
struct Stats{Header h;std::uint64_t vm_steps=0,laplace_energy=0;bool source_matched=false;};
void compare_files(const fs::path& a,const fs::path& b);
Stats unpack(const fs::path& in,const fs::path& out,const fs::path* source=nullptr){
 require(!fs::exists(out),"refusing to overwrite decoded output");std::ifstream input(in,std::ios::binary);require(bool(input),"cannot open archive");
 const Header h=read_header(input);Stage stage(out);std::ofstream output(stage.temp,std::ios::binary|std::ios::trunc);require(bool(output),"cannot stage decoded output");
 VM vm;std::uint64_t total=0;std::uint32_t running=0xffffffffu;
 for(std::uint32_t id=0;id<h.tiles;++id){
  const std::size_t n=static_cast<std::size_t>(std::min<std::uint64_t>(TILE,h.bytes-total));
  auto r=get_record(input,id,n);vm.run(r.data(),r.size());
  running=crc_update(running,r.data(),n);output.write(reinterpret_cast<const char*>(r.data()),static_cast<std::streamsize>(n));
  require(bool(output),"decoded output write failed");total+=n;
 }
 require(total==h.bytes,"output byte-count mismatch");require((~running)==h.checksum,"whole-file CRC mismatch");
 require(input.peek()==EOF,"unexpected trailing archive data");output.close();require(bool(output),"output close failed");
 if(source)compare_files(*source,stage.temp); // compare BEFORE authoritative publication
 stage.commit(out);return {h,vm.executed,vm.laplace_energy,source!=nullptr};
}
void compare_files(const fs::path& a,const fs::path& b){
 require(fs::file_size(a)==fs::file_size(b),"CTR source/output size mismatch");
 std::ifstream x(a,std::ios::binary),y(b,std::ios::binary);require(bool(x)&&bool(y),"CTR cannot open sources");
 std::array<char,65536> p{},q{};
 for(;;){x.read(p.data(),p.size());y.read(q.data(),q.size());auto nx=x.gcount(),ny=y.gcount();require(nx==ny,"CTR read mismatch");require(std::equal(p.begin(),p.begin()+nx,q.begin()),"CTR bytewise mismatch");if(nx==0)break;}
}
void emit(const Stats& s){std::cout<<"{\"status\":\""<<(s.source_matched?"verified_commit":"crc_verified_decode")<<"\",\"format\":\"JXMP4T1\",\"version\":1,\"logical_axis\":"<<AXIS
 <<",\"guest_ram_bytes\":"<<RAM<<",\"source_bytes\":"<<s.h.bytes<<",\"tiles\":"<<s.h.tiles
 <<",\"crc32\":\""<<std::hex<<std::setfill('0')<<std::setw(8)<<s.h.checksum<<std::dec
 <<"\",\"vm_steps\":"<<s.vm_steps<<",\"laplace_energy\":"<<s.laplace_energy<<"}\n";}
void selftest(){
 std::array<std::uint8_t,TILE> a{};VM vm;
 for(int kind=0;kind<4;++kind){for(std::size_t i=0;i<a.size();++i){a[i]=kind==0?0:kind==1?0x7e:kind==2?static_cast<std::uint8_t>(i):static_cast<std::uint8_t>((i*37)^(i>>2));}
  for(std::size_t n:{std::size_t(1),std::size_t(99),TILE}){auto p=encode(a.data(),n);auto q=decode(p,n);require(std::equal(a.begin(),a.begin()+n,q.begin()),"codec selftest mismatch");vm.run(q.data(),n);}}
 require(coords(0)==std::array<std::uint32_t,3>{0,0,0},"coordinate zero test");
 require(coords(TX)==std::array<std::uint32_t,3>{0,8,0},"coordinate Y test");
 require(coords(TX*TY)==std::array<std::uint32_t,3>{0,0,16},"coordinate Z test");
 std::cout<<"SELFTEST PASS (12 reversible tile cases, VM, spatial address mapping)\n";
}
} // namespace dm
int main(int argc,char** argv){
 try{
  if(argc==2 && std::string(argv[1])=="selftest"){dm::selftest();return 0;}
  if(argc<4){std::cerr<<"Usage: dm3d_mp4_ctr selftest | encode INPUT ARCHIVE | decode ARCHIVE OUTPUT | roundtrip INPUT ARCHIVE OUTPUT\n";return 2;}
  const std::string op=argv[1];
  if(op=="encode"&&argc==4){auto h=dm::pack(argv[2],argv[3]);std::cout<<"ENCODE PASS tiles="<<h.tiles<<" bytes="<<h.bytes<<"\n";return 0;}
  if(op=="decode"&&argc==4){dm::emit(dm::unpack(argv[2],argv[3]));return 0;}
  if(op=="roundtrip"&&argc==5){auto a=fs::absolute(argv[2]),b=fs::absolute(argv[3]),c=fs::absolute(argv[4]);
    dm::require(a!=b&&a!=c&&b!=c,"input/archive/output paths must differ");
    dm::require(!fs::exists(b) && !fs::exists(c),"refusing to overwrite outputs");
    dm::pack(a,b);
    try{auto s=dm::unpack(b,c,&a);dm::emit(s);return 0;}
    catch(...){std::error_code ec;fs::remove(b,ec);throw;}}
  dm::fail("unknown operation or argument count");
 }catch(const std::exception& e){std::cerr<<"DM3D ERROR: "<<e.what()<<"\n";return 1;}
}
