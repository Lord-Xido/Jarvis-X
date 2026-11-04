// JARVIS-X bounded-memory file adapter to the 1 TiB virtual tile codec.
// Supports arbitrary binary input (including PNG, WAV, MP4, GIF, text) without
// confusing file encoding with learned multimodal generation.
#define JXTB_NO_MAIN
#include "jx3d_1tib.cpp"
#include <filesystem>

namespace fs=std::filesystem;

static void require_new(const std::string& name) {
  if(fs::exists(name))throw std::runtime_error("refusing to overwrite existing file: "+name);
}
static void check_same_file(const std::string& a,const std::string& b){
  if(fs::absolute(a).lexically_normal()==fs::absolute(b).lexically_normal())
    throw std::runtime_error("input and output paths must differ");
}

int main(int argc,char** argv){
 try {
  if(argc<3)throw std::runtime_error("usage: jx3d_media encode INPUT STREAM [START_TILE] | decode STREAM OUTPUT | verify STREAM [ORIGINAL]");
  const std::string command=argv[1];
  std::vector<uint8_t> raw(jx::BLOCK),dec(jx::BLOCK),residual(jx::BLOCK);
  jx::Stats stats;
  auto t0=std::chrono::steady_clock::now();
  if(command=="encode"){
    if(argc<4 || argc>5)throw std::runtime_error("encode INPUT STREAM [START_TILE]");
    std::string path=argv[2],dest=argv[3];
    check_same_file(path,dest);require_new(dest);
    uint64_t start=argc==5?jx::checked_number(argv[4]):0;
    if(start>=jx::NTILES)throw std::runtime_error("start tile outside virtual volume");
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("cannot read input file");
    std::ofstream out(dest,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot create encoded file");
    for(uint64_t tile=start;tile<jx::NTILES;tile++){
      std::fill(raw.begin(),raw.end(),0);
      in.read(reinterpret_cast<char*>(raw.data()),jx::BLOCK);
      auto read=in.gcount();
      if(read<0)throw std::runtime_error("file read error");
      if(read==0){if(in.bad())throw std::runtime_error("read failed");break;}
      auto rec=jx::encode(raw,tile,residual,static_cast<uint32_t>(read));
      jx::decode(rec,residual,dec);
      if(raw!=dec)throw std::runtime_error("encoder roundtrip mismatch");
      jx::write_record(out,rec);
      jx::count(stats,rec);
      if(read<static_cast<std::streamsize>(jx::BLOCK)){
        if(in.bad())throw std::runtime_error("incomplete read");
        break;
      }
    }
    if(!in.eof() && in.bad())throw std::runtime_error("input stream failed");
    if(stats.tiles==0)throw std::runtime_error("cannot encode empty input");
    if(stats.tiles==jx::NTILES){
      // Detect trailing input beyond the virtual 1 TiB addressing boundary.
      if(in.peek()!=std::char_traits<char>::eof())throw std::runtime_error("input exceeds virtual volume");
    }
    out.flush();if(!out)throw std::runtime_error("cannot flush encoded output");
    std::cout<<"operation=encode input="<<path<<" output="<<dest<<" start_tile="<<start<<"\n";
  } else if(command=="decode" || command=="verify"){
    if(command=="decode" && argc!=4)throw std::runtime_error("decode STREAM OUTPUT");
    if(command=="verify" && argc!=3 && argc!=4)throw std::runtime_error("verify STREAM [ORIGINAL]");
    std::string src=argv[2];
    std::ifstream in(src,std::ios::binary);
    if(!in)throw std::runtime_error("cannot read encoded stream");
    std::ofstream out;
    std::ifstream orig;
    if(command=="decode"){
      check_same_file(src,argv[3]);require_new(argv[3]);
      out.open(argv[3],std::ios::binary|std::ios::trunc);
      if(!out)throw std::runtime_error("cannot create decoded output");
    }else if(argc==4){
      orig.open(argv[3],std::ios::binary);
      if(!orig)throw std::runtime_error("cannot read reference input");
    }
    jx::Record r;
    uint64_t expected=jx::NTILES;
    bool previously_partial=false;
    while(jx::read_record(in,r)){
      auto id=jx::id_of(r);
      if(expected==jx::NTILES)expected=id;
      if(id!=expected || previously_partial || id>=jx::NTILES)
        throw std::runtime_error("out-of-order, duplicate, or trailing tile after partial block");
      jx::decode(r,residual,dec);
      auto n=jx::g32(r.h,16);
      if(n<jx::BLOCK)previously_partial=true;
      if(out.is_open())out.write(reinterpret_cast<const char*>(dec.data()),n);
      if(orig.is_open()){
        orig.read(reinterpret_cast<char*>(raw.data()),n);
        if(orig.gcount()!=static_cast<std::streamsize>(n) || !std::equal(raw.begin(),raw.begin()+n,dec.begin()))
          throw std::runtime_error("reference input mismatch");
      }
      jx::count(stats,r);
      expected++;
    }
    if(stats.tiles==0)throw std::runtime_error("empty input stream");
    if(orig.is_open() && orig.peek()!=std::char_traits<char>::eof())throw std::runtime_error("reference contains trailing data");
    if(out.is_open()){out.flush();if(!out)throw std::runtime_error("decoded file write failed");}
    std::cout<<"operation="<<command<<" reference_verified="<<(orig.is_open()?"yes":"no")<<"\n";
  } else throw std::runtime_error("unknown command: "+command);
  auto t=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
  jx::report(stats,t);
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"ERROR: "<<e.what()<<"\n";
  return 1;
 }
}