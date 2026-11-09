// JARVIS-X Omega^3 fractal cloud reference runtime v2.0
// Deterministic, locally sharded, CPU threaded, no ML and no actual remote cluster execution.
// Licensed for research use. Canonical Dr Moagi v1.0 is unmodified.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using u8=std::uint8_t; using u16=std::uint16_t; using u32=std::uint32_t; using u64=std::uint64_t;
constexpr u32 Q=65535U;
constexpr int D6[6][3]={{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
struct Digit {int x,y,z;};
struct Cell {int x,y,z;u32 path;u16 q;u8 mask;std::array<int32_t,6> nb;};
struct Partition {size_t first,last;std::vector<u32> halo_ids;std::vector<u16> halo_values;};
struct Snapshot {u32 lo,hi,max_err;double rms_codec;u64 digest;};
struct Frame {int step;u64 digest;std::vector<u16> q;};
struct Options {int depth=2,steps=24,workers=4,inject_fault_step=-1;std::string out="output",events="sample",resume="";bool no_frames=false;};
static int num_arg(const std::string& s) {
 size_t pos=0; int n=std::stoi(s,&pos,10);
 if(pos!=s.size()) throw std::runtime_error("invalid numeric argument: "+s);
 return n;
}
static Options arguments(int argc,char**argv){
 Options o;
 for(int i=1;i<argc;i++){
  std::string k=argv[i];
  if(k=="--help"){std::cout<<"Usage: jx3d_cloud --depth 1..4 --steps 0..10000 --workers auto|1..64 --out DIR --events none|sample|full [--inject-fault-step N] [--resume checkpoint.bin] [--no-frames]\n";std::exit(0);}
  if(k=="--no-frames"){o.no_frames=true;continue;}
  if(i+1>=argc)throw std::runtime_error("missing value for "+k);
  std::string v=argv[++i];
  if(k=="--depth")o.depth=num_arg(v);
  else if(k=="--steps")o.steps=num_arg(v);
  else if(k=="--workers")o.workers=(v=="auto"?0:num_arg(v));
  else if(k=="--out")o.out=v;
  else if(k=="--events")o.events=v;
  else if(k=="--resume")o.resume=v;
  else if(k=="--inject-fault-step")o.inject_fault_step=num_arg(v);
  else throw std::runtime_error("unknown option: "+k);
 }
 if(o.depth<1||o.depth>4||o.steps<0||o.steps>10000||o.workers<0||o.workers>64)
  throw std::runtime_error("limits: depth 1..4, steps 0..10000, workers 1..64");
 if(o.inject_fault_step==0||o.inject_fault_step>o.steps||o.inject_fault_step< -1)
  throw std::runtime_error("inject-fault-step must be -1 or an iteration 1..steps");
 if(o.events!="none"&&o.events!="sample"&&o.events!="full")
  throw std::runtime_error("events must be none|sample|full");
 return o;
}
static bool keep(int x,int y,int z){return int(x==1)+int(y==1)+int(z==1)<=1;}
static u64 power(u64 x,int n){u64 a=1;for(int k=0;k<n;k++)a*=x;return a;}
static std::array<Digit,20> alphabet(){
 std::array<Digit,20> lut{};int n=0;
 for(int z=0;z<3;z++)for(int y=0;y<3;y++)for(int x=0;x<3;x++)if(keep(x,y,z))lut[n++]={x,y,z};
 if(n!=20)throw std::runtime_error("alphabet invariant");
 return lut;
}
static u32 address(int x,int y,int z,int depth,const std::array<Digit,20>& lut){
 u32 path=0;
 for(int k=0;k<depth;k++){
  int symbol=-1;
  for(int s=0;s<20;s++)if(lut[s].x==x%3&&lut[s].y==y%3&&lut[s].z==z%3){symbol=s;break;}
  if(symbol<0)throw std::runtime_error("invalid triadic digit");
  path|=u32(symbol)<<(5*k);x/=3;y/=3;z/=3;
 }
 if(x||y||z)throw std::runtime_error("address overflow");
 return path;
}
static void grow(std::vector<Cell>& nodes,int remaining,int total,int x,int y,int z,
                 const std::array<Digit,20>& lut){
 if(remaining==0){
  u32 v=u32((17*x+29*y+43*z)%1009);
  Cell c{x,y,z,address(x,y,z,total,lut),u16((v*Q+504)/1008),0,{-1,-1,-1,-1,-1,-1}};
  nodes.push_back(c);return;
 }
 for(const auto& d:lut)grow(nodes,remaining-1,total,3*x+d.x,3*y+d.y,3*z+d.z,lut);
}
static u64 key(int x,int y,int z){return (u64(u32(x))<<42)|(u64(u32(y))<<21)|u64(u32(z));}
static u64 fnvb(u64 h,u8 v){return (h^u64(v))*1099511628211ULL;}
static u64 hash32(u64 h,u32 v){for(int k=0;k<4;k++)h=fnvb(h,u8(v>>(8*k)));return h;}
static u64 hash64(u64 h,u64 v){for(int k=0;k<8;k++)h=fnvb(h,u8(v>>(8*k)));return h;}
static u32 q8(u32 q){return (q*255U+Q/2U)/Q;}
static u32 d8(u32 q){return (q*Q+127U)/255U;}
static Snapshot snapshot(const std::vector<Cell>& cells,u64 prev,int step){
 u32 lo=Q,hi=0,maxerr=0;long double esq=0;
 u64 h=14695981039346656037ULL;h=hash64(h,prev);h=hash32(h,u32(step));
 for(const auto& c:cells){
  u32 v=c.q;lo=std::min(lo,v);hi=std::max(hi,v);
  u32 recon=d8(q8(v));u32 err=v>recon?v-recon:recon-v;
  maxerr=std::max(maxerr,err);esq+=(long double)err*err;
  h=hash32(h,u32(c.x));h=hash32(h,u32(c.y));h=hash32(h,u32(c.z));h=hash32(h,v);
 }
 return {lo,hi,maxerr,double(std::sqrt(esq/cells.size())/Q),h};
}
static void write_u32(std::ostream&out,u32 v){for(int k=0;k<4;k++)out.put(char((v>>(8*k))&255U));}
static void write_u64(std::ostream&out,u64 v){for(int k=0;k<8;k++)out.put(char((v>>(8*k))&255ULL));}
static void write_u16(std::ostream&out,u16 v){out.put(char(v&255U));out.put(char((v>>8)&255U));}
static std::string hex16(u64 h){std::ostringstream s;s<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return s.str();}
static void emit_csv(std::ostream&out,int t,size_t n,u64 edges,const Snapshot&s,double rms,const char*status){
 out<<t<<','<<n<<','<<edges<<','<<std::fixed<<std::setprecision(8)
 <<double(s.lo)/Q<<','<<double(s.hi)/Q<<','<<double(s.hi-s.lo)/Q<<','
 <<double(s.max_err)/Q<<','<<s.rms_codec<<','<<rms<<','<<hex16(s.digest)<<','<<status<<'\n';
}
static u16 read_u16(std::istream&in){
 int a=in.get(),b=in.get();if(a<0||b<0)throw std::runtime_error("truncated checkpoint u16");
 return u16(u32(a)|(u32(b)<<8));
}
static u32 read_u32(std::istream&in){u32 x=0;for(int k=0;k<4;k++){int b=in.get();if(b<0)throw std::runtime_error("truncated checkpoint u32");x|=u32(b)<<(8*k);}return x;}
static u64 read_u64(std::istream&in){u64 x=0;for(int k=0;k<8;k++){int b=in.get();if(b<0)throw std::runtime_error("truncated checkpoint u64");x|=u64(b)<<(8*k);}return x;}
static void emit_checkpoint(const fs::path&file,const std::vector<Cell>&nodes,int depth,int step,u64 h,u64 parent){
 const fs::path tmp(file.string()+".tmp");
 std::ofstream f(tmp,std::ios::binary);if(!f)throw std::runtime_error("checkpoint create failed");
 f.write("JX3DCHK2",8);write_u32(f,u32(depth));write_u32(f,u32(step));
 write_u64(f,u64(nodes.size()));write_u64(f,h);write_u64(f,parent);
 for(const auto&c:nodes)write_u16(f,c.q);
 f.flush();if(!f)throw std::runtime_error("checkpoint write failed");f.close();
 fs::rename(tmp,file); // Atomic replacement on supported local filesystems; no fsync durability guarantee.
}
static void load_checkpoint(const fs::path&file,std::vector<Cell>&nodes,int depth,int& step,u64&digest,u64&parent){
 std::ifstream f(file,std::ios::binary);if(!f)throw std::runtime_error("checkpoint not found");
 char magic[8];if(!f.read(magic,8)||std::string(magic,8)!="JX3DCHK2")throw std::runtime_error("checkpoint version mismatch");
 u32 d=read_u32(f),t=read_u32(f);u64 n=read_u64(f),h=read_u64(f),prior=read_u64(f);
 if(d!=u32(depth)||n!=nodes.size()||t>10000)throw std::runtime_error("checkpoint shape or step mismatch");
 for(auto&cell:nodes)cell.q=read_u16(f);
 if(f.get()!=EOF)throw std::runtime_error("trailing checkpoint data");
 auto calculated=snapshot(nodes,prior,int(t));
 if(calculated.digest!=h)throw std::runtime_error("checkpoint hash-chain mismatch");
 step=int(t);digest=h;parent=prior;
}
static void emit_codec(const fs::path&file,const std::vector<Cell>&nodes,int depth,int step){
 std::ofstream f(file,std::ios::binary);if(!f)throw std::runtime_error("codec create failed");
 f.write("JX3DQ801",8);write_u32(f,u32(depth));write_u32(f,u32(step));write_u64(f,u64(nodes.size()));
 for(const auto&c:nodes)f.put(char(q8(c.q)));
 if(!f)throw std::runtime_error("codec write failed");
}
static void emit_states(const fs::path&file,const std::vector<Cell>&nodes,const std::vector<int>&owners){
 std::ofstream f(file);if(!f)throw std::runtime_error("state csv create failed");
 f<<"index,x,y,z,path,shard,face_mask,q16,q8,reconstructed_q16,abs_error\n";
 for(size_t i=0;i<nodes.size();i++){
  const auto&c=nodes[i];u32 rec=d8(q8(c.q));
  f<<i<<','<<c.x<<','<<c.y<<','<<c.z<<','<<c.path<<','<<owners[i]<<','<<unsigned(c.mask)<<','
   <<c.q<<','<<q8(c.q)<<','<<rec<<','<<(c.q>rec?c.q-rec:rec-c.q)<<'\n';
 }
}
static void emit_frames(const fs::path&file,const std::vector<Cell>&nodes,
 const std::vector<int>&owners,const std::vector<size_t>&selected,const std::vector<Frame>&frames,int depth){
 std::ofstream f(file);if(!f)throw std::runtime_error("frame json create failed");
 f<<"{\"schema\":\"jx3d.frames.v2\",\"depth\":"<<depth<<",\"side\":"<<power(3,depth)<<",\"total_nodes\":"<<nodes.size()
 <<",\"sampled_nodes\":"<<selected.size()<<",\"coords\":[";
 for(size_t j=0;j<selected.size();j++){
  const auto&i=selected[j];const auto&c=nodes[i];if(j)f<<',';
  f<<'['<<c.x<<','<<c.y<<','<<c.z<<','<<unsigned(c.mask)<<','<<owners[i]<<','<<c.path<<']';
 }
 f<<"],\"frames\":[";
 for(size_t k=0;k<frames.size();k++){
  if(k)f<<',';
  f<<"{\"step\":"<<frames[k].step<<",\"hash\":\""<<hex16(frames[k].digest)<<"\",\"q\":[";
  for(size_t j=0;j<frames[k].q.size();j++){if(j)f<<',';f<<frames[k].q[j];}
  f<<"]}";
 }
 f<<"]}\n";
 if(!f)throw std::runtime_error("frame json write failed");
}
int main(int argc,char**argv){
 try{
  const Options opt=arguments(argc,argv);
  fs::create_directories(opt.out);const fs::path root(opt.out);
  const auto start=std::chrono::steady_clock::now();
  auto lut=alphabet();std::vector<Cell> nodes;nodes.reserve(size_t(power(20,opt.depth)));
  grow(nodes,opt.depth,opt.depth,0,0,0,lut);
  if(nodes.size()!=power(20,opt.depth))throw std::runtime_error("node count invariant");
  // Sparse coordinate-to-index dictionary: no dense 3^3L allocation.
  std::unordered_map<u64,u32> indices;indices.reserve(nodes.size()*2);
  for(size_t i=0;i<nodes.size();i++){
   const auto&c=nodes[i];
   if(!indices.emplace(key(c.x,c.y,c.z),u32(i)).second)throw std::runtime_error("duplicate cell");
  }
  u64 half_edges=0;
  for(auto&c:nodes){
   for(int d=0;d<6;d++){
    int x=c.x+D6[d][0],y=c.y+D6[d][1],z=c.z+D6[d][2];
    if(x<0||y<0||z<0||x>=int(power(3,opt.depth))||y>=int(power(3,opt.depth))||z>=int(power(3,opt.depth)))continue;
    auto it=indices.find(key(x,y,z));if(it!=indices.end()){
     c.mask|=u8(1U<<d);c.nb[d]=int32_t(it->second);half_edges++;
    }
   }
  }
  if(half_edges%2 || half_edges/2!=2*(power(20,opt.depth)-power(8,opt.depth)))
   throw std::runtime_error("edge count invariant");
  for(size_t i=0;i<nodes.size();i++){
   const auto&c=nodes[i];
   for(int d=0;d<6;d++)if(c.nb[d]>=0 && nodes[size_t(c.nb[d])].nb[d^1]!=int32_t(i))
    throw std::runtime_error("reciprocal neighbor mismatch");
  }
  int auto_workers=std::max(1,std::min(32,std::min(int(std::thread::hardware_concurrency()==0?1:std::thread::hardware_concurrency()),int((nodes.size()+1999)/2000))));
  int workers=std::min(int(nodes.size()),opt.workers==0?auto_workers:opt.workers);
  std::vector<Partition> parts(size_t(workers),Partition{});
  std::vector<int> owner(nodes.size(),-1);
  std::vector<std::array<int32_t,6>> ghost_slot(nodes.size());
  u64 remote_links=0;
  for(int w=0;w<workers;w++){
   Partition&p=parts[size_t(w)];
   p.first=nodes.size()*size_t(w)/size_t(workers);
   p.last=nodes.size()*size_t(w+1)/size_t(workers);
   for(size_t i=p.first;i<p.last;i++)owner[i]=w;
  }
  for(int w=0;w<workers;w++){
   Partition&p=parts[size_t(w)];std::map<u32,u32> slotmap;
   for(size_t i=p.first;i<p.last;i++)for(int d=0;d<6;d++){
    int32_t j=nodes[i].nb[d];ghost_slot[i][d]=-1;
    if(j>=0 && owner[size_t(j)]!=w){
     remote_links++;
     auto ins=slotmap.emplace(u32(j),u32(slotmap.size()));
     ghost_slot[i][d]=int32_t(ins.first->second);
    }
   }
   p.halo_ids.resize(slotmap.size());
   // Slots were assigned in encounter order, not sorted-key order.
   for(const auto&kv:slotmap)p.halo_ids[kv.second]=kv.first;
   p.halo_values.resize(slotmap.size());
  }
  u64 halo_copies=0;for(const auto&p:parts)halo_copies+=p.halo_ids.size();
  std::ofstream trace(root/"trace.csv"),quality(root/"quality.csv"),events;
  if(!trace||!quality)throw std::runtime_error("output create failed");
  if(opt.events!="none"){
   events.open(root/"events.csv");if(!events)throw std::runtime_error("event csv create failed");
   events<<"step,index,x,y,z,path,shard,face_mask,old_q16,neighbor_sum,new_q16,xor_delta,changed_bits,q8,decoded_q16,abs_error,hash,status\n";
  }
  trace<<"step,nodes,edges,min,max,range,max_codec_error,rms_codec_error,rms_delta,hash,status\n";
  quality<<"step,changed_cells,changed_bits,rms_delta_exact,edge_energy_q16,codec_max_error_q16,codec_rms_normalized,halo_cells_copied,halo_bytes,status\n";
  std::vector<size_t> selected;
  if(!opt.no_frames){
   size_t n=std::min(size_t(8000),nodes.size());
   for(size_t k=0;k<n;k++)selected.push_back(k*nodes.size()/n);
  }
  std::vector<Frame> frames;
  auto capture=[&](int step,u64 hash){
   if(opt.no_frames)return;
   int interval=std::max(1,(opt.steps+63)/64);
   if(step!=0 && step!=opt.steps && step%interval!=0)return;
   Frame fr;fr.step=step;fr.digest=hash;fr.q.reserve(selected.size());
   for(size_t i:selected)fr.q.push_back(nodes[i].q);
   frames.push_back(std::move(fr));
  };
  int initial_step=0;u64 previous=0,parent=0;Snapshot s;
  if(!opt.resume.empty()){
   load_checkpoint(opt.resume,nodes,opt.depth,initial_step,previous,parent);
   if(initial_step>opt.steps)throw std::runtime_error("checkpoint step exceeds --steps target");
   s=snapshot(nodes,parent,initial_step);
  } else {
   s=snapshot(nodes,0,0);previous=s.digest;
  }
  emit_csv(trace,initial_step,nodes.size(),half_edges/2,s,0.,opt.resume.empty()?"INIT":"RESUME");
  long double initial_energy=0;
  for(size_t i=0;i<nodes.size();i++)for(int d=0;d<6;d++)if(nodes[i].nb[d]>=0&&size_t(nodes[i].nb[d])>i){
   long double df=(long double)nodes[i].q-(long double)nodes[size_t(nodes[i].nb[d])].q;initial_energy+=df*df;
  }
  quality<<initial_step<<",0,0,0.00000000,"<<std::fixed<<std::setprecision(8)<<double(initial_energy)<<','<<s.max_err<<','
         <<s.rms_codec<<",0,0,"<<(opt.resume.empty()?"INIT":"RESUME")<<"\n";
  capture(initial_step,s.digest);
  std::vector<u32> next(nodes.size(),0);
  int last=initial_step;bool rolled_back=false;
  for(int step=initial_step+1;step<=opt.steps;step++){
   // Explicit immutable snapshot -> halo exchange -> partition compute -> candidate gate -> commit.
   for(auto&p:parts)for(size_t k=0;k<p.halo_ids.size();k++)p.halo_values[k]=nodes[p.halo_ids[k]].q;
   std::vector<std::thread> threads;threads.reserve(size_t(workers));
   for(int w=0;w<workers;w++)threads.emplace_back([&,w]{
    const auto&p=parts[size_t(w)];
    for(size_t i=p.first;i<p.last;i++){
     const Cell&c=nodes[i];u64 sum=0;int degree=0;
     for(int d=0;d<6;d++)if(c.mask&(1U<<d)){
      degree++;int32_t j=c.nb[d];
      int32_t slot=ghost_slot[i][d];
      sum+=slot>=0?p.halo_values[size_t(slot)]:nodes[size_t(j)].q;
     }
     next[i]=degree?u32((3ULL*u64(degree)*c.q+sum+2ULL*u64(degree))/(4ULL*u64(degree))):u32(c.q);
    }
   });
   for(auto&t:threads)t.join();
   if(opt.inject_fault_step==step)next[0]=Q+1;
   u32 lo=Q,hi=0;long double error2=0,edge_energy=0;
   u64 changed=0,changed_bits=0;
   bool arithmetic_valid=true;
   for(size_t i=0;i<nodes.size();i++){
    lo=std::min(lo,next[i]);hi=std::max(hi,next[i]);
    const Cell&c=nodes[i];
    int degree=0;u64 sum=0;
    for(int d=0;d<6;d++)if(c.nb[d]>=0){degree++;sum+=nodes[size_t(c.nb[d])].q;}
    const u32 independently=degree?u32((3ULL*u64(degree)*c.q+sum+2ULL*u64(degree))/(4ULL*u64(degree))):u32(c.q);
    if(next[i]!=independently)arithmetic_valid=false;
    if(next[i]!=c.q)changed++;
    changed_bits+=u64(__builtin_popcount(u32(next[i]^c.q)));
    long double delta=(long double)next[i]-(long double)c.q;error2+=delta*delta;
    for(int d=0;d<6;d++)if(c.nb[d]>=0 && size_t(c.nb[d])>i){
     long double dd=(long double)next[i]-(long double)next[size_t(c.nb[d])];edge_energy+=dd*dd;
    }
   }
   // The independent arithmetic gate detects injected corruption; range gate retains v1 behaviour.
   bool accepted=(hi<=Q)&&(hi-lo<=s.hi-s.lo)&&arithmetic_valid;
   if(!accepted){
    std::ofstream rejected(root/"rejections.csv");
    rejected<<"step,index,x,y,z,old_q16,proposed_q16,gate,committed_hash\n";
    for(size_t i=0;i<nodes.size();i++)if(next[i]>Q){
     const auto&c=nodes[i];rejected<<step<<','<<i<<','<<c.x<<','<<c.y<<','<<c.z<<','<<c.q<<','<<next[i]
        <<",OUT_OF_RANGE,"<<hex16(s.digest)<<'\n';
    }
    rejected<<step<<",-1,0,0,0,0,0,"<<(arithmetic_valid?"RANGE_GATE":"ARITHMETIC_GATE")<<','<<hex16(s.digest)<<'\n';
    emit_csv(trace,step,nodes.size(),half_edges/2,s,0.,"ROLLBACK");
    quality<<step<<","<<changed<<","<<changed_bits<<",0.00000000,0.00000000,"<<s.max_err<<","<<s.rms_codec
           <<","<<halo_copies<<","<<(halo_copies*2)<<",ROLLBACK\n";
    rolled_back=true;
    break;
   }
   // Match v1.0 reporting behaviour precisely in trace.csv (rounded RMS).
   double canonical_delta=double(std::llround(std::sqrt(error2/nodes.size())))/Q;
   double exact_delta=double(std::sqrt(error2/nodes.size())/Q);
   std::vector<u16> old;
   if(opt.events!="none"){
    old.reserve(nodes.size());for(const auto&c:nodes)old.push_back(c.q);
   }
   for(size_t i=0;i<nodes.size();i++)nodes[i].q=u16(next[i]);
   parent=previous;s=snapshot(nodes,previous,step);previous=s.digest;
   emit_csv(trace,step,nodes.size(),half_edges/2,s,canonical_delta,"COMMIT");
   quality<<step<<','<<changed<<','<<changed_bits<<','<<std::fixed<<std::setprecision(8)<<exact_delta<<','
     <<double(edge_energy)<<','<<s.max_err<<','<<s.rms_codec<<','<<halo_copies<<','<<halo_copies*2<<",COMMIT\n";
   capture(step,s.digest);
   if(opt.events!="none"){
    for(size_t i=0;i<nodes.size();i++){
     bool sample=(i<8);
     if(!sample){
      for(const auto&p:parts)if(i==p.first||i+1==p.last){sample=true;break;}
     }
     if(opt.events=="sample"&&!sample)continue;
     const auto&c=nodes[i];u64 sum=0;
     for(int d=0;d<6;d++)if(c.nb[d]>=0)sum+=old[size_t(c.nb[d])];
     u32 reconstructed=d8(q8(c.q));
     events<<step<<','<<i<<','<<c.x<<','<<c.y<<','<<c.z<<','<<c.path<<','<<owner[i]<<','<<unsigned(c.mask)
      <<','<<old[i]<<','<<sum<<','<<c.q<<','<<(u32(old[i])^c.q)<<','<<__builtin_popcount(u32(old[i])^c.q)
      <<','<<q8(c.q)<<','<<reconstructed<<','<<(c.q>reconstructed?c.q-reconstructed:reconstructed-c.q)
      <<','<<hex16(s.digest)<<",COMMIT\n";
    }
   }
   last=step;
  }
  emit_checkpoint(root/"checkpoint.bin",nodes,opt.depth,last,s.digest,parent);
  emit_codec(root/"codec_q8.bin",nodes,opt.depth,last);
  emit_states(root/"state.csv",nodes,owner);
  if(!opt.no_frames)emit_frames(root/"frames.json",nodes,owner,selected,frames,opt.depth);
  trace.close();quality.close();if(events.is_open())events.close();
  const auto end=std::chrono::steady_clock::now();
  auto ms=std::chrono::duration<double,std::milli>(end-start).count();
  std::ofstream report(root/"run.json");
  report<<"{\n  \"schema\": \"jx3d.run.v2\",\n  \"depth\": "<<opt.depth
    <<",\n  \"steps_requested\": "<<opt.steps<<",\n  \"resumed_at_step\": "<<initial_step<<",\n  \"steps_committed\": "<<last
    <<",\n  \"workers\": "<<workers<<",\n  \"nodes\": "<<nodes.size()
    <<",\n  \"edges\": "<<half_edges/2
    <<",\n  \"remote_directed_edges\": "<<remote_links
    <<",\n  \"unique_halo_values_per_step\": "<<halo_copies
    <<",\n  \"actual_remote_workers\": false"
    <<",\n  \"mode\": \"local_cpu_sharded_simulation\""
    <<",\n  \"status\": \""<<(rolled_back?"ROLLBACK":"COMMITTED")<<"\""
    <<",\n  \"final_hash\": \""<<hex16(s.digest)<<"\""
    <<",\n  \"elapsed_ms\": "<<std::fixed<<std::setprecision(3)<<ms<<"\n}\n";
  if(!report)throw std::runtime_error("run report write failed");
  std::cout<<"JX3D CLOUD LOCAL v2 status="<<(rolled_back?"ROLLBACK":"COMMITTED")
    <<" cells="<<nodes.size()<<" edges="<<half_edges/2<<" workers="<<workers
    <<" committed="<<last<<" halo_values="<<halo_copies
    <<" hash="<<hex16(s.digest)<<" elapsed_ms="<<ms<<"\n";
  return rolled_back?2:0;
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
