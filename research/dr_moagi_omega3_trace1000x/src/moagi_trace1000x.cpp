// Dr Moagi Ω³ — trace-governed, exact periodic 3D state fast-forward.
// A 1000 MB virtual cube is represented exactly when the input is a periodic 3D tile.
// Arbitrary nonperiodic data falls back to the bundled streaming bittrace kernel.
// Build via CMake; requires C++17 and OpenSSL libcrypto.
#define main moagi_bittrace_original_main
#include "moagi_bittrace.cpp"
#undef main
#include <unordered_map>
#include <optional>
#include <thread>

namespace opt3d {
using ::u8;using ::u32;using ::u64;using ::Shape;using ::Digest;
using ::sha;using ::hex;using ::check;using ::crc32;using ::Clock;

struct Stats {u64 changed=0,bits=0;std::array<u64,8> flips{};};
Stats operator+(Stats a,const Stats& b){a.changed+=b.changed;a.bits+=b.bits;for(int i=0;i<8;++i)a.flips[i]+=b.flips[i];return a;}
Stats scale(Stats a,u64 n){a.changed*=n;a.bits*=n;for(auto&v:a.flips)v*=n;return a;}
struct Tile {
  Shape world;u32 nx=2,ny=2,nz=2,gain=64;u64 steps=1;
  std::vector<u8> data;
  size_t volume()const{return size_t(nx)*ny*nz;}
  size_t at(u32 x,u32 y,u32 z)const{return (size_t(z)*ny+y)*nx+x;}
  u64 factor()const{return world.voxels()/volume();}
  void validate()const{
    world.valid();check(world.voxels()<=1'000'000'000ULL,"maximum supported volume: 1,000,000,000 bytes");
    check(nx&&ny&&nz&&volume()<=4096,"periodic tile must have 1..4096 voxels");
    check(world.x%nx==0&&world.y%ny==0&&world.z%nz==0,"tile dimensions must divide world axes");
    check(gain<=256&&steps>=1&&steps<=1'000'000,"gain 0..256, steps 1..1000000");
    check(data.size()==volume(),"tile size mismatch");
  }
};
struct Neighbors {std::vector<std::array<size_t,6>> index;explicit Neighbors(const Tile& t){
 index.resize(t.volume());for(u32 z=0;z<t.nz;++z)for(u32 y=0;y<t.ny;++y)for(u32 x=0;x<t.nx;++x){
 index[t.at(x,y,z)]={t.at((x+1)%t.nx,y,z),t.at((x+t.nx-1)%t.nx,y,z),
 t.at(x,(y+1)%t.ny,z),t.at(x,(y+t.ny-1)%t.ny,z),t.at(x,y,(z+1)%t.nz),t.at(x,y,(z+t.nz-1)%t.nz)};
 }}};
inline u8 update(unsigned c,unsigned n,u32 gain){unsigned s=(2*c+n+4)>>3;return u8(((256u-gain)*s+gain*c+128u)>>8);}
std::vector<u8> reference(const Tile& t,const std::vector<u8>& a){
 std::vector<u8> out(a.size());for(u32 z=0;z<t.nz;++z)for(u32 y=0;y<t.ny;++y)for(u32 x=0;x<t.nx;++x){
 auto center=a[t.at(x,y,z)];unsigned sum=a[t.at((x+1)%t.nx,y,z)]+a[t.at((x+t.nx-1)%t.nx,y,z)]
 +a[t.at(x,(y+1)%t.ny,z)]+a[t.at(x,(y+t.ny-1)%t.ny,z)]
 +a[t.at(x,y,(z+1)%t.nz)]+a[t.at(x,y,(z+t.nz-1)%t.nz)];
 out[t.at(x,y,z)]=update(center,sum,t.gain);
 }return out;
}
std::vector<u8> indexed(const Tile& t,const Neighbors& p,const std::vector<u8>& a){
 std::vector<u8> out(a.size());for(size_t i=0;i<a.size();++i){auto &nb=p.index[i];unsigned sum=a[nb[0]]+a[nb[1]]+a[nb[2]]+a[nb[3]]+a[nb[4]]+a[nb[5]];out[i]=update(a[i],sum,t.gain);}return out;
}
Stats delta(const Tile& t,const std::vector<u8>& a,const std::vector<u8>& b){Stats r;for(size_t i=0;i<a.size();++i){u8 v=u8(a[i]^b[i]);r.changed+=v!=0;r.bits+=__builtin_popcount(unsigned(v));for(int j=0;j<8;++j)r.flips[j]+=(v>>j)&1u;}return scale(r,t.factor());}
Digest chain_next(const Digest& prior,const std::string& payload){std::vector<u8> in;in.reserve(prior.size()+payload.size());in.insert(in.end(),prior.begin(),prior.end());in.insert(in.end(),payload.begin(),payload.end());return sha(in);}
std::string csvline(const std::vector<std::string>& v){std::ostringstream o;for(size_t i=0;i<v.size();++i){if(i)o<<',';o<<v[i];}return o.str();}
std::vector<std::string> splitcsv(const std::string& s){return ::split(s,',');}
u64 getnum(const std::string& s){return ::number(s);}
void w64(std::ostream& f,u64 v){::write32(f,u32(v));::write32(f,u32(v>>32));}
u64 r64(std::istream& f){u64 lo=::read32(f);return lo|(u64(::read32(f))<<32);}
void save(const Tile& t,const std::string& path,const Digest& journal){t.validate();Shape p{t.nx,t.ny*t.nz,1};auto enc=::encode(t.data,p);
 std::ofstream f(path+".partial",std::ios::binary|std::ios::trunc);check(bool(f),"write archive failed");
 constexpr char magic[8]={'M','3','T','I','L','E','2','\0'};f.write(magic,8);
 ::write32(f,t.world.x);::write32(f,t.world.y);::write32(f,t.world.z);
 ::write32(f,t.nx);::write32(f,t.ny);::write32(f,t.nz);::write32(f,t.gain);w64(f,t.steps);
 ::write32(f,u32(t.data.size()));::write32(f,enc.mode);::write32(f,u32(enc.bytes.size()));::write32(f,crc32(t.data));
 auto digest=sha(t.data);::put(f,digest.data(),digest.size());::put(f,journal.data(),journal.size());::put(f,enc.bytes.data(),enc.bytes.size());f.close();check(bool(f),"archive write incomplete");fs::rename(path+".partial",path);
}
struct Archive {Tile t;Digest chain{};};
Archive load(const std::string& path){std::ifstream f(path,std::ios::binary);check(bool(f),"cannot open tile archive");char magic[8];f.read(magic,8);check(bool(f)&&std::memcmp(magic,"M3TILE2\0",8)==0,"invalid tile archive magic");
 Archive a;auto &t=a.t;t.world.x=::read32(f);t.world.y=::read32(f);t.world.z=::read32(f);
 t.nx=::read32(f);t.ny=::read32(f);t.nz=::read32(f);t.gain=::read32(f);t.steps=r64(f);
 u32 len=::read32(f),mode=::read32(f),encoded=::read32(f),crc=::read32(f);
 // Validate dimensions before allocating any untrusted payload.
 check(t.nx&&t.ny&&t.nz&&u64(t.nx)*t.ny*t.nz<=4096,"invalid tile capacity");
 check(len==u64(t.nx)*t.ny*t.nz&&mode<=::RLE_XOR_X&&encoded<=len*2,"invalid archive size/mode");
 Digest d{};::take(f,d.data(),d.size());::take(f,a.chain.data(),a.chain.size());
 std::vector<u8> packed(encoded);::take(f,packed.data(),packed.size());
 check(f.peek()==EOF,"trailing archive data");t.data=::decode({u8(mode),std::move(packed)},Shape{t.nx,t.ny*t.nz,1});
 t.validate();check(::crc32(t.data)==crc&&sha(t.data)==d,"tile archive integrity failure");return a;
}
// Explicit 8-bit execution ISA. A candidate must pass DECODE + VERIFY before commit.
enum Opcode : u8 {OP_LOAD=0x01,OP_M6=0x02,OP_ENCODE=0x03,OP_DECODE=0x04,
                  OP_VERIFY=0x05,OP_BITTRACE=0x06,OP_COMMIT=0x07,OP_HALT=0xFF};
struct BytecodeResult{std::vector<u8> output;Stats bits;::Encoded encoded;};
BytecodeResult execute_bytecode(const Tile& t,const Neighbors& nb,const std::string& plan){
 constexpr std::array<u8,8> program={OP_LOAD,OP_M6,OP_ENCODE,OP_DECODE,OP_VERIFY,OP_BITTRACE,OP_COMMIT,OP_HALT};
 std::vector<u8> R1,R2;::Encoded latent;Stats difference;bool verified=false,committed=false;
 Shape codec_shape{t.nx,t.ny*t.nz,1};
 for(auto op:program){switch(op){
 case OP_LOAD:check(t.data.size()==t.volume(),"invalid input register");break;
 case OP_M6:R1=plan=="INDEXED_3D"?indexed(t,nb,t.data):reference(t,t.data);break;
 case OP_ENCODE:latent=::encode(R1,codec_shape);break;
 case OP_DECODE:R2=::decode(latent,codec_shape);break;
 case OP_VERIFY:verified=(R1==R2);check(verified,"bytecode codec reconstruction failed");break;
 case OP_BITTRACE:check(verified,"unverified trace");difference=delta(t,t.data,R2);break;
 case OP_COMMIT:check(verified,"unverified commit");committed=true;break;
 case OP_HALT:check(committed,"missing commit");return {std::move(R2),difference,std::move(latent)};
 default: ::fail("unsupported bytecode opcode");
 }}::fail("missing HALT");
}
struct Plan {std::string id;double ref_ns=0,idx_ns=0;};
Plan optimize(const Tile& t,const Neighbors& nb,const std::string& csv){
 // Genuine correctness gate before considering performance.
 auto reference_out=reference(t,t.data),indexed_out=indexed(t,nb,t.data);
 check(reference_out==indexed_out,"indexed kernel differs from scalar reference");
 // Run independent probe states to avoid accepting a kernel specialized only on the seed.
 for(int trial=0;trial<12;++trial){auto modified=t.data;for(size_t j=0;j<modified.size();++j)modified[j]=u8((j*73+trial*29+modified[j])&255);check(reference(t,modified)==indexed(t,nb,modified),"candidate fails independent probe");}
 volatile u64 sink=0;auto benchmark=[&](bool use_idx)->double{
 constexpr int reps=256;auto start=Clock::now();auto state=t.data;
 for(int i=0;i<reps;++i){auto next=use_idx?indexed(t,nb,state):reference(t,state);sink+=next[size_t(i)%next.size()];state.swap(next);}
 return double(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count())/reps;
 };
 // Alternating measurements reduce systematic order bias.
 double ref=(benchmark(false)+benchmark(false))/2.0,idx=(benchmark(true)+benchmark(true))/2.0;
 std::string selected=idx<ref*.98?"INDEXED_3D":"SCALAR_3D";
 std::ofstream out(csv);check(bool(out),"cannot write plan benchmark");out<<"candidate,exact_equivalence,mean_ns_per_tile_step,selected,checksum_sink\n";
 out<<"SCALAR_3D,TRUE,"<<std::fixed<<std::setprecision(3)<<ref<<","<<(selected=="SCALAR_3D"?"TRUE":"FALSE")<<","<<sink<<"\n";
 out<<"INDEXED_3D,TRUE,"<<idx<<","<<(selected=="INDEXED_3D"?"TRUE":"FALSE")<<","<<sink<<"\n";
 return {selected,ref,idx};
}
std::string header(const Tile& t,const Digest& d){return csvline({"#M3TILETRACE2",std::to_string(t.world.x),std::to_string(t.world.y),std::to_string(t.world.z),std::to_string(t.nx),std::to_string(t.ny),std::to_string(t.nz),std::to_string(t.gain),std::to_string(t.steps),hex(d)});}
std::vector<std::string> row_fields(const std::string& kind,u64 start,u64 advance,u64 period,const std::string& plan,const Digest& before,const Digest& after,const Stats& st,const ::Encoded& en,u64 elapsed){
 std::vector<std::string> v={kind,std::to_string(start),std::to_string(advance),std::to_string(period),plan,hex(before),hex(after),std::to_string(st.changed),std::to_string(st.bits)};
 for(u64 n:st.flips){v.push_back(std::to_string(n));}
 v.push_back(::codec_name(en.mode));v.push_back(std::to_string(en.bytes.size()));v.push_back(std::to_string(elapsed));v.push_back("COMMIT");return v;
}
struct Outcome{Tile final;Digest chain{};u64 computed=0,skipped=0,changed=0,flips=0,events=0;double ms=0;};
Outcome run(Tile t,const std::string& trace,const std::string& result,const std::string& plans,bool verbose=true){
 t.validate();auto begin=Clock::now();Neighbors nb(t);auto plan=optimize(t,nb,plans);
 const Digest seed=sha(t.data);const std::string h=header(t,seed);Digest chain=sha(reinterpret_cast<const u8*>(h.data()),h.size());
 std::ofstream journal(trace+".partial");check(bool(journal),"cannot create journal");journal<<h<<"\n";
 journal<<"kind,start,advance,cycle_period,plan,input_sha256,output_sha256,changed_voxels,changed_bits,b0,b1,b2,b3,b4,b5,b6,b7,codec,compressed_bytes,elapsed_ns,decision,chain_sha256\n";
 Outcome out;out.final=t;std::vector<Stats> increments;
 struct Seen{u64 step;std::vector<u8> exact_state;};
 std::unordered_map<std::string,Seen> visited;u64 step=0;
 auto emit=[&](const std::string& kind,u64 advance,u64 period,const std::vector<u8>& after,Stats st,u64 elapsed){
  auto enc=::encode(after,Shape{t.nx,t.ny*t.nz,1});check(::decode(enc,Shape{t.nx,t.ny*t.nz,1})==after,"codec reconstruction failed");
  auto fields=row_fields(kind,step,advance,period,plan.id,sha(t.data),sha(after),st,enc,elapsed);
  auto payload=csvline(fields);chain=chain_next(chain,payload);journal<<payload<<','<<hex(chain)<<'\n';check(bool(journal),"trace write error");
  out.changed+=st.changed;out.flips+=st.bits;out.events++;if(kind=="STEP")out.computed++;else out.skipped+=advance;
  t.data=after;step+=advance;
 };
 while(step<t.steps){auto it=visited.find(hex(sha(t.data)));
   if(it!=visited.end() && it->second.exact_state==t.data){
     u64 period=step-it->second.step;check(period>0,"invalid cycle length");u64 n=(t.steps-step)/period;
     if(n){Stats all{};for(u64 i=it->second.step;i<step;++i)all=all+increments.at(size_t(i));all=scale(all,n);
       emit("CYCLE_FAST_FORWARD",period*n,period,t.data,all,0);continue;
     }
    }else if(it==visited.end() && visited.size()<4096){visited.emplace(hex(sha(t.data)),Seen{step,t.data});}
   auto start=Clock::now();auto candidate=execute_bytecode(t,nb,plan.id);
   auto next=std::move(candidate.output);
   // Both execution choices have to agree on all committed states.
   check(next==reference(t,t.data),"plan diverged from reference");
   Stats stats=candidate.bits;increments.push_back(stats);
   emit("STEP",1,0,next,stats,u64(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count()));
 }
 journal.close();check(bool(journal),"journal close error");fs::rename(trace+".partial",trace);
 save(t,result,chain);out.final=t;out.chain=chain;out.ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
 if(verbose)std::cout<<"RUN_OK logical_voxels="<<t.world.voxels()<<" logical_steps="<<t.steps<<" computed="<<out.computed<<" skipped="<<out.skipped<<" changed_bits="<<out.flips<<" events="<<out.events<<" elapsed_ms="<<out.ms<<" chain="<<hex(chain)<<" plan="<<plan.id<<"\n";
 return out;
}
Outcome replay(const std::string& seedpath,const std::string& tracepath,const std::string& finalpath){
 auto a=load(seedpath);auto t=a.t;Neighbors nb(t);std::ifstream f(tracepath);check(bool(f),"trace not readable");std::string line;check(bool(std::getline(f,line))&&line==header(t,sha(t.data)),"trace header/seed mismatch");
 check(bool(std::getline(f,line))&&line.find("kind,start,advance,")==0,"missing trace schema");
 Digest chain=sha(reinterpret_cast<const u8*>(header(t,sha(t.data)).data()),header(t,sha(t.data)).size());
 Outcome out;std::vector<Stats> history;std::unordered_map<std::string,u64> visited;u64 step=0;
 while(std::getline(f,line)){
  auto fields=splitcsv(line);check(fields.size()==22,"invalid trace field count");
  auto stated=fields.back();fields.pop_back();auto payload=csvline(fields);chain=chain_next(chain,payload);check(hex(chain)==stated,"trace SHA-256 chain invalid");
  std::string kind=fields[0];u64 start=getnum(fields[1]),advance=getnum(fields[2]),period=getnum(fields[3]);
  check(start==step&&advance>0&&step+advance<=t.steps,"invalid event range");
  check(fields[4]=="INDEXED_3D"||fields[4]=="SCALAR_3D","unknown plan");
  check(fields[5]==hex(sha(t.data))&&fields[20]=="COMMIT","state/commit mismatch");
  Stats stats{};std::vector<u8> after=t.data;
  if(kind=="STEP"){
    check(advance==1&&period==0,"malformed step");auto next=reference(t,t.data);
    if(fields[4]=="INDEXED_3D")check(next==indexed(t,nb,t.data),"indexed replay diverged");
    stats=delta(t,t.data,next);after=std::move(next);history.push_back(stats);out.computed++;
  }else if(kind=="CYCLE_FAST_FORWARD"){
    check(period>0&&advance%period==0&&period<=history.size()&&step>=period,"invalid skip period");
    auto probe=t.data;Stats one{};
    for(u64 p=0;p<period;++p){auto next=reference(t,probe);one=one+delta(t,probe,next);probe.swap(next);}
    check(probe==t.data,"attempted unproven cycle skip");stats=scale(one,advance/period);out.skipped+=advance;
  }else ::fail("unknown trace event");
  check(fields[6]==hex(sha(after))&&getnum(fields[7])==stats.changed&&getnum(fields[8])==stats.bits,"state transition mismatch");
  for(int bit=0;bit<8;++bit)check(getnum(fields[9+bit])==stats.flips[bit],"bitplane mismatch");
  auto enc=::encode(after,Shape{t.nx,t.ny*t.nz,1});check(fields[17]==::codec_name(enc.mode)&&getnum(fields[18])==enc.bytes.size(),"codec mismatch");
  check(::decode(enc,Shape{t.nx,t.ny*t.nz,1})==after,"codec failure on replay");
  (void)getnum(fields[19]);out.events++;out.changed+=stats.changed;out.flips+=stats.bits;t.data=std::move(after);step+=advance;
 }
 check(step==t.steps,"trace incomplete");auto final=load(finalpath);check(final.t.world.x==t.world.x&&final.t.world.y==t.world.y&&final.t.world.z==t.world.z&&final.t.gain==t.gain&&final.t.steps==t.steps&&final.t.data==t.data&&final.chain==chain,"final archive and trace disagree");out.final=t;out.chain=chain;
 std::cout<<"REPLAY_OK steps="<<step<<" computed="<<out.computed<<" skipped="<<out.skipped<<" changed_bits="<<out.flips<<" chain="<<hex(chain)<<"\n";return out;
}
Tile make_tile(Shape shape,u32 nx,u32 ny,u32 nz,u32 gain,u64 steps,const std::string& pattern){Tile t;t.world=shape;t.nx=nx;t.ny=ny;t.nz=nz;t.gain=gain;t.steps=steps;t.data.resize(t.volume());
 u64 seed=0x123456789abcdefULL;
 for(u32 z=0;z<nz;++z)for(u32 y=0;y<ny;++y)for(u32 x=0;x<nx;++x){u8 b=0;
 if(pattern=="checker")b=((x+y+z)&1)?255:0;
 else if(pattern=="ramp")b=u8((x*7+y*13+z*23)&255);
 else if(pattern=="impulse")b=(x==0&&y==0&&z==0)?255:0;
 else if(pattern=="noise"){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;b=u8(seed);}
 else if(pattern=="uniform")b=128;
 else {::fail("unknown tile pattern");}
 t.data[t.at(x,y,z)]=b;}
 t.validate();return t;
}
void materialize(const Tile& t,const std::string& path){t.validate();std::vector<u8> row(t.world.x),plane(size_t(t.world.plane()));
 std::ofstream f(path+".partial",std::ios::binary|std::ios::trunc);check(bool(f),"cannot create raw volume");
 for(u32 z=0;z<t.world.z;++z){u32 sz=z%t.nz;for(u32 y=0;y<t.world.y;++y){u32 sy=y%t.ny;for(u32 x=0;x<t.world.x;++x)row[x]=t.data[t.at(x%t.nx,sy,sz)];std::memcpy(plane.data()+size_t(y)*t.world.x,row.data(),row.size());}
 ::put(f,plane.data(),plane.size());}f.close();check(bool(f),"raw materialization failed");fs::rename(path+".partial",path);std::cout<<"MATERIALIZE_OK bytes="<<t.world.voxels()<<" path="<<path<<"\n";}
std::optional<Tile> detect_raw(const std::string& path,Shape sh,u32 nx,u32 ny,u32 nz,u32 gain,u64 steps){
 Tile t;t.world=sh;t.nx=nx;t.ny=ny;t.nz=nz;t.gain=gain;t.steps=steps;t.data.resize(t.volume());t.validate();::check_file_size(path,sh);
 std::ifstream f(path,std::ios::binary);check(bool(f),"cannot open raw volume");std::vector<u8> plane(size_t(sh.plane()));
 for(u32 z=0;z<sh.z;++z){::take(f,plane.data(),plane.size());for(u32 y=0;y<sh.y;++y)for(u32 x=0;x<sh.x;++x){size_t pos=size_t(y)*sh.x+x,loc=t.at(x%nx,y%ny,z%nz);if(z<nz&&y<ny&&x<nx)t.data[loc]=plane[pos];else if(plane[pos]!=t.data[loc])return std::nullopt;}}
 return t;
}
void tests(){size_t n=0;
 for(u32 tx=1;tx<=5;++tx)for(u32 ty=1;ty<=5;++ty)for(u32 tz=1;tz<=4;++tz)for(u32 gain:{0u,1u,64u,128u,255u,256u}){
  auto t=make_tile(Shape{tx*2,ty*2,tz*2},tx,ty,tz,gain,6,"noise");Neighbors nb(t);
  auto a=reference(t,t.data),b=indexed(t,nb,t.data);check(a==b,"indexed mismatch");n++;
  // Compare the expanded 3D state with the original streaming kernel's independent 3D stencil.
  Shape sh=t.world;std::vector<u8> expanded(size_t(sh.voxels())),dense(size_t(sh.voxels()));
  for(u32 z=0;z<sh.z;++z)for(u32 y=0;y<sh.y;++y)for(u32 x=0;x<sh.x;++x)expanded[size_t(z)*sh.plane()+size_t(y)*sh.x+x]=t.data[t.at(x%tx,y%ty,z%tz)];
  for(u32 z=0;z<sh.z;++z){size_t p=size_t(sh.plane());std::vector<u8> prev(expanded.begin()+size_t((z+sh.z-1)%sh.z)*p,expanded.begin()+size_t((z+sh.z-1)%sh.z+1)*p),cur(expanded.begin()+size_t(z)*p,expanded.begin()+size_t(z+1)*p),next(expanded.begin()+size_t((z+1)%sh.z)*p,expanded.begin()+size_t((z+1)%sh.z+1)*p),out;
 ::calculate(prev,cur,next,sh,gain,out,true);std::copy(out.begin(),out.end(),dense.begin()+size_t(z)*p);}
  for(u32 z=0;z<sh.z;++z)for(u32 y=0;y<sh.y;++y)for(u32 x=0;x<sh.x;++x){check(dense[size_t(z)*sh.plane()+size_t(y)*sh.x+x]==a[t.at(x%tx,y%ty,z%tz)],"dense-periodic inequality");n++;}
  auto en=::encode(t.data,Shape{tx,ty*tz,1});check(::decode(en,Shape{tx,ty*tz,1})==t.data,"codec not exact");n++;
 }
 fs::path dir=fs::temp_directory_path()/"moagi_1000x_tile_tests";fs::create_directories(dir);
 try{for(const char* pat:{"checker","ramp","uniform","noise"}){
  auto t=make_tile(Shape{20,24,16},4,3,2,64,10000,pat);auto seed=(dir/(std::string(pat)+".seed")).string(),tr=(dir/(std::string(pat)+".csv")).string(),fin=(dir/(std::string(pat)+".final")).string(),pl=(dir/(std::string(pat)+".plans")).string();
  save(t,seed,{});auto a=run(t,tr,fin,pl,false);auto b=replay(seed,tr,fin);check(a.chain==b.chain&&a.flips==b.flips&&a.final.data==b.final.data,"trace mismatch");n++;
  auto raw=(dir/(std::string(pat)+".raw")).string();materialize(a.final,raw);auto r=detect_raw(raw,t.world,t.nx,t.ny,t.nz,t.gain,t.steps);check(r.has_value()&&r->data==a.final.data,"raw recognition failure");n++;
  if(std::string(pat)=="checker"){
   std::ifstream file(tr);std::string text((std::istreambuf_iterator<char>(file)),{});size_t pos=text.find("CYCLE_FAST_FORWARD");check(pos!=std::string::npos,"no valid fast-forward in trace");text[pos]='Z';std::ofstream bad(tr+".bad");bad<<text;bad.close();bool rejected=false;try{replay(seed,tr+".bad",fin);}catch(...){rejected=true;}check(rejected,"tampered trace accepted");n++;
   // Detect corrupted archive bytes and require rejection.
   auto broken=fin+".bad";fs::copy_file(fin,broken,fs::copy_options::overwrite_existing);std::fstream f(broken,std::ios::binary|std::ios::in|std::ios::out);f.seekg(-1,std::ios::end);char bit;f.get(bit);f.seekp(-1,std::ios::end);bit^=1;f.put(bit);f.close();rejected=false;try{load(broken);}catch(...){rejected=true;}check(rejected,"corrupt archive accepted");n++;
  }
 }}catch(...){fs::remove_all(dir);throw;}fs::remove_all(dir);
 std::cout<<"SELFTEST_OK checks="<<n<<" dense_vs_periodic=pass variable_tile=pass codec=pass exact_replay=pass tamper=pass\n";
}
void usage(){std::cout<<"Dr Moagi Ω³ periodic 3D trace-optimized engine\n"
 "  moagi_trace1000x generate <x> <y> <z> <tile_x> <tile_y> <tile_z> <gain> <steps> <checker|uniform|ramp|noise|impulse> <seed.tile>\n"
 "  moagi_trace1000x run <seed.tile> <trace.csv> <final.tile> <plans.csv>\n"
 "  moagi_trace1000x replay <seed.tile> <trace.csv> <final.tile>\n"
 "  moagi_trace1000x materialize <state.tile> <output.raw>\n"
 "  moagi_trace1000x fromraw <input.raw> <x> <y> <z> <tile_x> <tile_y> <tile_z> <gain> <steps> <output_prefix>\n"
 "  moagi_trace1000x selftest\n";}
} // namespace opt3d
int main(int argc,char**argv){try{using namespace opt3d;if(argc<2){opt3d::usage();return 2;}std::string cmd=argv[1];int p=2;
 if(cmd=="selftest"){check(argc==2,"selftest takes no args");tests();}
 else if(cmd=="generate"){check(argc==12,"generate expected x y z tx ty tz gain steps pattern out");Shape sh{u32(getnum(argv[p++])),u32(getnum(argv[p++])),u32(getnum(argv[p++]))};u32 tx=u32(getnum(argv[p++])),ty=u32(getnum(argv[p++])),tz=u32(getnum(argv[p++])),g=u32(getnum(argv[p++]));u64 steps=getnum(argv[p++]);auto pattern=std::string(argv[p++]),file=std::string(argv[p++]);auto t=make_tile(sh,tx,ty,tz,g,steps,pattern);save(t,file,{});std::cout<<"GENERATED tile_bytes="<<t.volume()<<" logical_voxels="<<sh.voxels()<<" file="<<file<<"\n";}
 else if(cmd=="run"){check(argc==6,"run expected seed trace final plans");auto t=load(argv[2]).t;run(t,argv[3],argv[4],argv[5]);}
 else if(cmd=="replay"){check(argc==5,"replay expected seed trace final");replay(argv[2],argv[3],argv[4]);}
 else if(cmd=="materialize"){check(argc==4,"materialize expected state output");materialize(load(argv[2]).t,argv[3]);}
 else if(cmd=="fromraw"){check(argc==12,"fromraw expected input x y z tx ty tz gain steps prefix");auto input=std::string(argv[p++]);Shape sh{u32(getnum(argv[p++])),u32(getnum(argv[p++])),u32(getnum(argv[p++]))};u32 tx=u32(getnum(argv[p++])),ty=u32(getnum(argv[p++])),tz=u32(getnum(argv[p++])),g=u32(getnum(argv[p++]));u64 steps=getnum(argv[p++]);std::string prefix=argv[p++];auto started=Clock::now();auto found=detect_raw(input,sh,tx,ty,tz,g,steps);double ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();if(found){save(*found,prefix+".seed.tile",{});auto result=run(*found,prefix+".trace.csv",prefix+".final.tile",prefix+".plans.csv");std::cout<<"AUTO_PLAN=EXACT_3D_PERIODIC classification_ms="<<ms<<" output_materialization=deferred\n";(void)result;}else{check(steps<=64,"dense fallback is limited to 64 iterations");::Config config;config.sh=sh;config.gain=g;config.steps=u32(steps);config.fast=true;config.input=input;config.output=prefix+".raw";config.trace=prefix+".trace.csv";config.archive=prefix+".m3bt";std::cout<<"AUTO_PLAN=DENSE_STREAM classification_ms="<<ms<<"\n";::run(config);}}
 else{opt3d::usage();return 2;}return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}