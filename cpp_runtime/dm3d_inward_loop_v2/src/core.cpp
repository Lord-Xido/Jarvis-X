#include "dm3d/core.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace dm3d {
namespace {
constexpr char MAGIC[8] = {'D','M','3','D','A','R','C','2'};
constexpr uint32_t VERSION = 2;
constexpr uint64_t FNV_INIT = 14695981039346656037ULL;
constexpr uint64_t FNV_PRIME = 1099511628211ULL;
constexpr uint64_t TILE_COUNT_AXIS = AXIS / TILE_EDGE;

void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
void write_u8(std::ostream& s, uint8_t n) { s.put(static_cast<char>(n)); require(bool(s),"write failure"); }
void write_u32(std::ostream& s, uint32_t n) { for(int j=0;j<4;j++)write_u8(s, uint8_t(n>>(8*j))); }
void write_u64(std::ostream& s, uint64_t n) { for(int j=0;j<8;j++)write_u8(s, uint8_t(n>>(8*j))); }
uint8_t read_u8(std::istream& s) { int c=s.get();require(c!=EOF,"truncated or malformed archive");return uint8_t(c); }
uint32_t read_u32(std::istream& s) { uint32_t n=0;for(int j=0;j<4;j++) n|=uint32_t(read_u8(s))<<(8*j);return n; }
uint64_t read_u64(std::istream& s) { uint64_t n=0;for(int j=0;j<8;j++) n|=uint64_t(read_u8(s))<<(8*j);return n; }
uint64_t sample_hash(const uint8_t* p, size_t n, uint64_t seed) {
    for(size_t i=0;i<n;i++){seed^=p[i];seed*=FNV_PRIME;}return seed;
}
uint32_t be32(const std::vector<uint8_t>& a, size_t o) {
    if(a.size()<o+4) return 0;
    return (uint32_t(a[o])<<24)|(uint32_t(a[o+1])<<16)|(uint32_t(a[o+2])<<8)|a[o+3];
}
uint32_t le32(const std::vector<uint8_t>& a, size_t o) {
    if(a.size()<o+4) return 0;
    return (uint32_t(a[o+3])<<24)|(uint32_t(a[o+2])<<16)|(uint32_t(a[o+1])<<8)|a[o];
}
uint16_t le16(const std::vector<uint8_t>& a,size_t o) {
    if(a.size()<o+2) return 0;
    return uint16_t(a[o])|(uint16_t(a[o+1])<<8);
}
std::string lower(std::string s) { for(char& c:s) if(c>='A'&&c<='Z')c=char(c-'A'+'a'); return s; }
// Five reversible 3D causal predictors. All predictor operands refer only to
// already-decoded positions, making each spatial transform provably invertible.
uint8_t predictor(const std::array<uint8_t,TILE_BYTES>& a, size_t i, uint8_t mode) {
    const size_t x=i%TILE_EDGE, y=(i/TILE_EDGE)%TILE_EDGE, z=i/(TILE_EDGE*TILE_EDGE);
    const uint8_t px=x ? a[i-1] : 0;
    const uint8_t py=y ? a[i-TILE_EDGE] : 0;
    const uint8_t pz=z ? a[i-TILE_EDGE*TILE_EDGE] : 0;
    switch(mode){
      case 1: return x ? px : (y ? py : pz); // priority X, Y, Z
      case 2: return px ^ py ^ pz;            // 3-axis parity predictor
      case 3: return px;                      // X-only
      case 4: return py;                      // Y-only
      case 5: return pz;                      // Z-only
      default: throw std::runtime_error("unknown 3D predictor");
    }
}
std::vector<uint8_t> delta_rle(const std::array<uint8_t,TILE_BYTES>& a, uint8_t mode) {
    std::array<uint8_t,TILE_BYTES> diff{};
    for(size_t i=0;i<TILE_BYTES;i++)diff[i]=a[i]^predictor(a,i,mode);
    std::vector<uint8_t> out;
    out.reserve(TILE_BYTES/2);
    size_t i=0;
    while(i<TILE_BYTES){
        const bool zero=diff[i]==0;
        const size_t start=i;
        while(i<TILE_BYTES && (diff[i]==0)==zero && i-start<65535)++i;
        const uint16_t n=static_cast<uint16_t>(i-start);
        out.push_back(zero ? 0 : 1);
        out.push_back(uint8_t(n));out.push_back(uint8_t(n>>8));
        if(!zero)out.insert(out.end(),diff.begin()+start,diff.begin()+i);
    }
    return out;
}
std::array<uint8_t,TILE_BYTES> inflate_delta(const std::vector<uint8_t>& payload,uint8_t mode){
    std::array<uint8_t,TILE_BYTES> output{};
    size_t pos=0, i=0;
    while(i<TILE_BYTES){
        require(pos+3<=payload.size(),"RLE token truncated");
        auto kind=payload[pos++];
        uint16_t n=uint16_t(payload[pos])|(uint16_t(payload[pos+1])<<8);pos+=2;
        require(n>0 && size_t(n)<=TILE_BYTES-i && kind<=1,"RLE invalid count/mode");
        require(kind==0 || pos+n<=payload.size(),"RLE literal overrun");
        for(uint16_t j=0;j<n;j++){
            uint8_t delta=kind==0?0:payload[pos++];
            output[i]=delta^predictor(output,i,mode);++i;
        }
    }
    require(pos==payload.size(),"unexpected trailing RLE data");
    return output;
}
void write_header(std::ostream& os,const ArchiveInfo& h,double omega){
    os.write(MAGIC,8);require(bool(os),"cannot write archive magic");
    write_u32(os,VERSION);write_u32(os,static_cast<uint32_t>(h.modality));
    write_u64(os,h.original_size);write_u64(os,h.tiles);
    write_u64(os,h.file_hash);write_u64(os,std::bit_cast<uint64_t>(omega));
}
ArchiveInfo read_header(std::istream& is,double* omega=nullptr){
    char magic[8]{};is.read(magic,8);
    require(bool(is) && std::memcmp(magic,MAGIC,8)==0,"not a DM3DARC2 archive");
    require(read_u32(is)==VERSION,"unsupported archive version");
    uint32_t mod=read_u32(is);require(mod<=5,"unknown archive modality");
    ArchiveInfo h;
    h.modality=static_cast<Modality>(mod);
    h.original_size=read_u64(is);h.tiles=read_u64(is);h.file_hash=read_u64(is);
    uint64_t raw=read_u64(is);if(omega)*omega=std::bit_cast<double>(raw);
    require(h.tiles == h.original_size/TILE_BYTES+(h.original_size%TILE_BYTES!=0),"archive tile count is inconsistent");
    return h;
}
void write_tile(std::ostream& os,const EncodedTile& tile){
    write_u64(os,tile.origin.x);write_u64(os,tile.origin.y);write_u64(os,tile.origin.z);
    write_u32(os,tile.valid);write_u8(os,tile.mode);
    require(tile.payload.size()<=std::numeric_limits<uint32_t>::max(),"tile too large");
    write_u32(os,uint32_t(tile.payload.size()));write_u64(os,tile.checksum);
    os.write(reinterpret_cast<const char*>(tile.payload.data()),std::streamsize(tile.payload.size()));
    require(bool(os),"failed writing tile bytes");
}
EncodedTile read_tile(std::istream& is,uint64_t id){
    EncodedTile t;t.tile_id=id;
    t.origin={read_u64(is),read_u64(is),read_u64(is)};
    t.valid=read_u32(is);t.mode=read_u8(is);
    uint32_t sz=read_u32(is);t.checksum=read_u64(is);
    require(t.valid>0&&t.valid<=TILE_BYTES&&t.mode<=5&&sz<=TILE_BYTES*2,"invalid tile header");
    require(t.origin.x<AXIS&&t.origin.y<AXIS&&t.origin.z<AXIS,"tile coordinate out of range");
    require(t.origin.x%TILE_EDGE==0 && t.origin.y%TILE_EDGE==0 && t.origin.z%TILE_EDGE==0,"unaligned tile origin");
    require(t.origin.x==tile_origin(id).x && t.origin.y==tile_origin(id).y && t.origin.z==tile_origin(id).z,"out-of-order tile coordinate");
    t.payload.resize(sz);
    is.read(reinterpret_cast<char*>(t.payload.data()),sz);
    require(bool(is),"truncated tile payload");
    return t;
}
struct Pending {uint64_t id;uint32_t valid;std::array<uint8_t,TILE_BYTES> tile;};
}

U128 flatten(Coord c){
    require(c.x<AXIS&&c.y<AXIS&&c.z<AXIS,"3D coordinate outside virtual cube");
    return U128(c.x)+U128(AXIS)*(U128(c.y)+U128(AXIS)*c.z);
}
Coord unflatten(U128 n){
    require(n < U128(AXIS)*AXIS*AXIS,"index outside virtual cube");
    Coord c; c.x=uint64_t(n%AXIS);n/=AXIS;c.y=uint64_t(n%AXIS);n/=AXIS;c.z=uint64_t(n);return c;
}
std::string decimal(U128 n){
    if(!n)return "0";
    std::string s;while(n){s.push_back(char('0'+n%10));n/=10;}
    std::reverse(s.begin(),s.end());return s;
}
U128 parse_u128(const std::string& s){
    require(!s.empty(),"empty integer");
    U128 n=0,max=~U128(0);
    for(char c:s){require(c>='0'&&c<='9',"nondecimal integer");
        unsigned digit=unsigned(c-'0');require(n<=(max-digit)/10,"128-bit integer overflow");n=n*10+digit;
    }return n;
}
Coord tile_origin(uint64_t tile_id){
    uint64_t tx=tile_id%TILE_COUNT_AXIS;
    uint64_t rest=tile_id/TILE_COUNT_AXIS;
    uint64_t ty=rest%TILE_COUNT_AXIS;
    uint64_t tz=rest/TILE_COUNT_AXIS;
    require(tz<TILE_COUNT_AXIS,"tile index outside virtual cube");
    return {tx*TILE_EDGE,ty*TILE_EDGE,tz*TILE_EDGE};
}
// Persistent log-structured sparse 3D voxel store. Each record costs 33 bytes.
// Appending is O(1) amortized, lookup is O(number_of_written_events).
namespace {
constexpr char VOX_MAGIC[8]={'D','M','3','D','V','X','L','1'};
uint64_t voxel_record_hash(Coord c,uint8_t value){
    uint64_t h=FNV_INIT;
    for(uint64_t v:{c.x,c.y,c.z}){
        for(unsigned b=0;b<8;b++){
            const uint8_t byt=uint8_t(v>>(8*b));
            h=sample_hash(&byt,1,h);
        }
    }
    return sample_hash(&value,1,h);
}
void validate_voxel_file(std::istream& stream){
    char magic[8];stream.read(magic,8);
    require(bool(stream)&&std::memcmp(magic,VOX_MAGIC,8)==0,"invalid sparse voxel volume");
    require(read_u32(stream)==1,"unsupported voxel log version");
}
}
void voxel_set(const fs::path& volume,Coord c,uint8_t value){
    (void)flatten(c); // exact bounds validation
    if(!fs::exists(volume)){
        std::ofstream init(volume,std::ios::binary|std::ios::trunc);
        require(bool(init),"cannot initialize sparse volume");
        init.write(VOX_MAGIC,8);write_u32(init,1);
        init.close();require(bool(init),"cannot finalize sparse volume header");
    }else{
        std::ifstream old(volume,std::ios::binary);
        require(bool(old),"cannot open sparse volume for validation");
        validate_voxel_file(old);
    }
    std::ofstream out(volume,std::ios::binary|std::ios::app);
    require(bool(out),"cannot open sparse volume for append");
    write_u64(out,c.x);write_u64(out,c.y);write_u64(out,c.z);write_u8(out,value);
    write_u64(out,voxel_record_hash(c,value));
    out.flush();require(bool(out),"sparse voxel append failed");
}
std::optional<uint8_t> voxel_get(const fs::path& volume,Coord c){
    (void)flatten(c);
    if(!fs::exists(volume))return std::nullopt;
    std::ifstream in(volume,std::ios::binary);
    require(bool(in),"cannot open sparse volume");
    validate_voxel_file(in);
    std::optional<uint8_t> result;
    while(in.peek()!=EOF){
        Coord r{read_u64(in),read_u64(in),read_u64(in)};
        uint8_t value=read_u8(in);uint64_t expected=read_u64(in);
        require(r.x<AXIS && r.y<AXIS && r.z<AXIS,"sparse voxel record outside virtual cube");
        require(voxel_record_hash(r,value)==expected,"sparse voxel record corrupted");
        if(r.x==c.x&&r.y==c.y&&r.z==c.z)result=value;
    }
    return result;
}
std::string modality_name(Modality m){
    switch(m){case Modality::Text:return "text";case Modality::Image:return "image";
    case Modality::Audio:return "audio";case Modality::Video:return "video";
    case Modality::Document:return "document";default:return "binary";}
}
Modality detect_modality(const fs::path& p,const std::vector<uint8_t>& b){
    auto ext=lower(p.extension().string());
    if(b.size()>=8&&std::memcmp(b.data(),"\x89PNG\r\n\x1a\n",8)==0)return Modality::Image;
    if(b.size()>=3&&b[0]==0xff&&b[1]==0xd8&&b[2]==0xff)return Modality::Image;
    if(b.size()>=4&&std::memcmp(b.data(),"GIF8",4)==0)return Modality::Image;
    if(b.size()>=12&&std::memcmp(b.data(),"RIFF",4)==0 && std::memcmp(b.data()+8,"WAVE",4)==0)return Modality::Audio;
    if(b.size()>=12&&std::memcmp(b.data()+4,"ftyp",4)==0)return Modality::Video;
    if(b.size()>=4&&std::memcmp(b.data(),"%PDF",4)==0)return Modality::Document;
    if(ext==".png"||ext==".jpg"||ext==".jpeg"||ext==".bmp"||ext==".ppm")return Modality::Image;
    if(ext==".wav"||ext==".mp3"||ext==".flac"||ext==".ogg")return Modality::Audio;
    if(ext==".mp4"||ext==".mov"||ext==".mkv"||ext==".webm")return Modality::Video;
    if(ext==".pdf"||ext==".docx"||ext==".pptx")return Modality::Document;
    if(ext==".txt"||ext==".md"||ext==".csv"||ext==".json"||ext==".xml"||ext==".html")return Modality::Text;
    return Modality::Binary;
}
Features analyze_file(const fs::path& p){
    std::ifstream is(p,std::ios::binary);require(bool(is),"cannot open input: "+p.string());
    Features f;f.bytes=fs::file_size(p);
    std::vector<uint8_t> sample(1<<20);
    is.read(reinterpret_cast<char*>(sample.data()),std::streamsize(sample.size()));
    sample.resize(size_t(is.gcount()));
    f.modality=detect_modality(p,sample);
    std::array<uint64_t,256> hist{};
    bool in_word=false;
    for(uint8_t c:sample){hist[c]++;if(c=='\n')f.lines++;
        bool word=!(c==' '||c=='\n'||c=='\r'||c=='\t'||c==0);
        if(word&&!in_word) f.words++;
        in_word=word;
    }
    if(!sample.empty())for(auto count:hist)if(count){double prob=double(count)/sample.size();f.entropy_bits_per_byte-=prob*std::log2(prob);}
    std::ostringstream ss;
    if(f.modality==Modality::Image && sample.size()>=24 && std::memcmp(sample.data(),"\x89PNG",4)==0){
        ss<<"PNG "<<be32(sample,16)<<"x"<<be32(sample,20);
    }else if(f.modality==Modality::Audio && sample.size()>=36 && std::memcmp(sample.data(),"RIFF",4)==0){
        uint16_t channels=0,bits=0,format=0;
        uint32_t rate=0,bytes_per_sec=0,data_bytes=0;
        size_t data_offset=0;
        for(size_t pos=12;pos+8<=sample.size();){
            uint32_t chunk_size=le32(sample,pos+4);
            if(std::memcmp(sample.data()+pos,"fmt ",4)==0 && chunk_size>=16 && pos+24<=sample.size()){
                format=le16(sample,pos+8);channels=le16(sample,pos+10);
                rate=le32(sample,pos+12);bytes_per_sec=le32(sample,pos+16);
                bits=le16(sample,pos+22);
            }
            if(std::memcmp(sample.data()+pos,"data",4)==0){
                data_bytes=chunk_size;data_offset=pos+8;break;
            }
            size_t step=size_t(chunk_size)+8+(chunk_size&1U);
            if(step>sample.size()-pos)break;
            pos+=step;
        }
        ss<<"WAV "<<channels<<"ch "<<rate<<"Hz "<<bits<<"bit";
        if(bytes_per_sec&&data_bytes)ss<<" duration="<<std::fixed<<std::setprecision(3)
             <<(double(data_bytes)/bytes_per_sec)<<"s";
        if(format==1 && bits==16 && data_offset<sample.size()){
            size_t available=std::min<size_t>(data_bytes,sample.size()-data_offset);
            size_t count=available/2;
            double power=0.0,peak=0.0;
            for(size_t i=0;i<count;i++){
                int16_t pcm=int16_t(le16(sample,data_offset+i*2));
                double val=double(pcm)/32768.0;
                power+=val*val;peak=std::max(peak,std::abs(val));
            }
            if(count)ss<<" PCM16 sample_rms="<<std::fixed<<std::setprecision(5)
                <<std::sqrt(power/count)<<" peak="<<peak;
        }
    }else if(f.modality==Modality::Video && sample.size()>=12 && std::memcmp(sample.data()+4,"ftyp",4)==0){
        ss<<"ISO-BMFF/MP4 brand=";
        for(size_t i=8;i<12;i++)ss<<char(sample[i]>=32&&sample[i]<127?sample[i]:'?');
    }else if(f.modality==Modality::Text){ss<<"sample lines="<<f.lines<<" words="<<f.words;}
    else ss<<"byte-preserving input; no semantic media decoding";
    f.detail=ss.str();return f;
}
uint64_t hash_bytes(const uint8_t* p,size_t n,uint64_t seed){return sample_hash(p,n,seed);}
uint64_t hash_file(const fs::path& p){
    std::ifstream is(p,std::ios::binary);require(bool(is),"cannot open hash file");
    std::array<uint8_t,65536> b{};uint64_t h=FNV_INIT;
    while(is){is.read(reinterpret_cast<char*>(b.data()),b.size());auto n=is.gcount();if(n>0)h=hash_bytes(b.data(),size_t(n),h);}
    require(is.eof(),"hash read failed");return h;
}
LatentResult inward_fold(const std::array<uint8_t,TILE_BYTES>& tile,unsigned iterations){
    // Non-destructive *diagnostic* autoencoder: 16^3 -> 4^3 mean-pool; 6-neighbor periodic diffusion.
    constexpr unsigned E=4;
    std::array<double,64> seed{},state{},next{};
    for(unsigned z=0;z<16;z++)for(unsigned y=0;y<16;y++)for(unsigned x=0;x<16;x++){
        auto small=(z/4)*16+(y/4)*4+(x/4);
        seed[small]+=double(tile[z*256+y*16+x])/255.0/64.0;
    }
    double change=0;
    for(unsigned k=0;k<iterations;k++){
        change=0;
        for(unsigned z=0;z<E;z++)for(unsigned y=0;y<E;y++)for(unsigned x=0;x<E;x++){
            const unsigned i=z*16+y*4+x;
            auto get=[&](unsigned a,unsigned b,unsigned c){return state[(c%E)*16+(b%E)*4+(a%E)];};
            double near=(get(x+1,y,z)+get((x+E-1)%E,y,z)+get(x,y+1,z)+get(x,(y+E-1)%E,z)+get(x,y,z+1)+get(x,y,(z+E-1)%E))/6.0;
            next[i]=0.6*state[i]+0.4*(0.85*seed[i]+0.15*near);
            change=std::max(change,std::abs(next[i]-state[i]));
        }
        state=next;
    }
    const auto sum=std::accumulate(state.begin(),state.end(),0.0);
    return {change,sum/64.0,iterations};
}
EncodedTile encode_tile(const std::array<uint8_t,TILE_BYTES>& tile,uint32_t valid,
                        uint64_t id,unsigned iter,unsigned refine_passes,uint8_t preferred_mode){
    require(valid>0&&valid<=TILE_BYTES,"invalid valid length");
    require(refine_passes>=1&&refine_passes<=5,"refine passes must be 1..5");
    require(preferred_mode>=1&&preferred_mode<=5,"invalid preferred codec");
    EncodedTile best;
    best.tile_id=id;best.origin=tile_origin(id);best.valid=valid;
    best.checksum=hash_bytes(tile.data(),valid);
    best.latent=inward_fold(tile,iter);
    best.mode=0;best.payload.assign(tile.begin(),tile.end());
    // The memory of the previously successful spatial direction is used first.
    // Next, propose untested directions, reconstruct, measure bitwise discrepancy,
    // and commit *only* verified improvements. Raw fallback is always lossless.
    const std::array<uint8_t,5> modes{preferred_mode,1,2,3,4};
    std::array<bool,6> visited{};
    unsigned attempts=0;
    auto proposal=[&](uint8_t mode){
        auto payload=delta_rle(tile,mode);
        // Verification is independent of hash collisions: check all 4096 bytes.
        auto reconstructed=inflate_delta(payload,mode);
        require(reconstructed==tile,"CTR FAILED: trial predictor is not reversible");
        ++best.tested;
        if(payload.size()<best.payload.size()){
            best.mode=mode;best.payload=std::move(payload);
        }
    };
    for(auto mode:modes){
        if(attempts>=refine_passes)break;
        if(visited[mode])continue;
        visited[mode]=true;proposal(mode);++attempts;
    }
    // Search remaining directions so all five are reachable (including mode 5).
    for(uint8_t mode=1;mode<=5 && attempts<refine_passes;mode++){
        if(visited[mode])continue;
        visited[mode]=true;proposal(mode);++attempts;
    }
    return best;
}
std::array<uint8_t,TILE_BYTES> decode_tile(const EncodedTile& t){
    require(t.valid>0&&t.valid<=TILE_BYTES,"invalid valid count");
    std::array<uint8_t,TILE_BYTES> raw{};
    if(t.mode==0){
        require(t.payload.size()==TILE_BYTES,"bad raw tile length");
        std::copy(t.payload.begin(),t.payload.end(),raw.begin());
    }else if(t.mode>=1&&t.mode<=5){
        raw=inflate_delta(t.payload,t.mode);
    }else throw std::runtime_error("unknown tile mode");
    require(hash_bytes(raw.data(),t.valid)==t.checksum,"tile integrity check failed");
    return raw;
}
ArchiveInfo inspect_archive(const fs::path& p){
    std::ifstream is(p,std::ios::binary);require(bool(is),"cannot open archive");
    ArchiveInfo h=read_header(is);h.stored_size=fs::file_size(p);return h;
}
Stats encode_file(const fs::path& source,const fs::path& dest,unsigned threads,unsigned fold_iterations,unsigned refine_passes){
    require(threads>=1&&threads<=128,"worker count must be 1..128");
    require(fold_iterations>=1&&fold_iterations<=1000,"fold iterations must be 1..1000");
    require(refine_passes>=1&&refine_passes<=5,"refine passes must be 1..5");
    require(fs::absolute(source)!=fs::absolute(dest),"source and archive paths must differ");
    Features f=analyze_file(source);
    std::ifstream in(source,std::ios::binary);require(bool(in),"cannot open source");
    const fs::path tmp=fs::path(dest.string()+".partial");
    require(fs::absolute(source)!=fs::absolute(tmp),"temporary path conflicts with source");
    std::ofstream os(tmp,std::ios::binary|std::ios::trunc);require(bool(os),"cannot create destination");
    ArchiveInfo h;h.modality=f.modality;h.original_size=f.bytes;
    h.tiles=f.bytes/TILE_BYTES+uint64_t(f.bytes%TILE_BYTES!=0);
    h.file_hash=FNV_INIT;
    write_header(os,h,0.0);
    double sum_change=0,sum_mean=0,omega=0;
    uint8_t preferred_mode=1;
    uint64_t trials=0,raw_fallback=0,payload_saved=0;
    const uint64_t expected=h.tiles;
    uint64_t next=0;
    try{
        while(next<expected){
            std::vector<std::future<EncodedTile>> futures;
            const unsigned batch=unsigned(std::min<uint64_t>(threads,expected-next));
            futures.reserve(batch);
            for(unsigned j=0;j<batch;j++){
                Pending job{};job.id=next+j;
                in.read(reinterpret_cast<char*>(job.tile.data()),TILE_BYTES);
                const auto got=in.gcount();
                require(got>0 && got<=std::streamsize(TILE_BYTES),"source unexpectedly truncated");
                job.valid=uint32_t(got);
                h.file_hash=hash_bytes(job.tile.data(),job.valid,h.file_hash);
                futures.push_back(std::async(std::launch::async,[job,fold_iterations,refine_passes,preferred_mode]{
                    return encode_tile(job.tile,job.valid,job.id,fold_iterations,refine_passes,preferred_mode);
                }));
            }
            for(auto& fut:futures){
                EncodedTile t=fut.get();
                write_tile(os,t);
                trials+=t.tested;raw_fallback+=t.mode==0;
                payload_saved+=TILE_BYTES-t.payload.size();
                if(t.mode!=0)preferred_mode=t.mode;
                sum_change+=t.latent.change;sum_mean+=t.latent.mean;
                omega=0.9*omega+0.1*t.latent.mean;
            }
            next+=batch;
        }
        require(next==h.tiles,"tile mismatch");
        // Patch FNV checksum and persisted Omega register in the header.
        os.seekp(32,std::ios::beg);write_u64(os,h.file_hash);
        write_u64(os,std::bit_cast<uint64_t>(omega));
        os.flush();require(bool(os),"archive write failed");os.close();
        // Committing an archive is a distinct orchestration stage.
        fs::rename(tmp,dest);
    }catch(...){os.close();std::error_code ec;fs::remove(tmp,ec);throw;}
    h.stored_size=fs::file_size(dest);
    Stats s;s.archive=h;s.threads=threads;s.fold_iterations=fold_iterations;
    s.original_bytes=h.original_size;s.encoded_bytes=h.stored_size;
    s.refine_passes=refine_passes;s.candidate_trials=trials;
    s.raw_fallback_tiles=raw_fallback;s.saved_payload_bytes=payload_saved;
    if(h.tiles){s.mean_latent_change=sum_change/h.tiles;s.mean_latent_value=sum_mean/h.tiles;}
    return s;
}
ArchiveInfo decode_file(const fs::path& archive,const fs::path& dest){
    require(fs::absolute(archive)!=fs::absolute(dest),"archive and output paths must differ");
    std::ifstream in(archive,std::ios::binary);require(bool(in),"cannot open archive");
    ArchiveInfo h=read_header(in);
    const fs::path tmp=fs::path(dest.string()+".partial");
    require(fs::absolute(archive)!=fs::absolute(tmp),"temporary path conflicts with archive");
    std::ofstream os(tmp,std::ios::binary|std::ios::trunc);require(bool(os),"cannot create restored file");
    uint64_t hash=FNV_INIT,remaining=h.original_size;
    try{
        for(uint64_t id=0;id<h.tiles;id++){
            auto tile=read_tile(in,id);
            uint32_t valid=uint32_t(std::min<uint64_t>(remaining,TILE_BYTES));
            require(tile.valid==valid,"inconsistent tile valid bytes");
            auto raw=decode_tile(tile);
            os.write(reinterpret_cast<const char*>(raw.data()),valid);
            require(bool(os),"cannot write restored bytes");
            hash=hash_bytes(raw.data(),valid,hash);remaining-=valid;
        }
        require(remaining==0 && hash==h.file_hash,"file checksum differs after decode");
        require(in.peek()==EOF,"archive contains trailing bytes");
        os.flush();require(bool(os),"restored file flush failed");os.close();
        fs::rename(tmp,dest);
    }catch(...){os.close();std::error_code ec;fs::remove(tmp,ec);throw;}
    h.stored_size=fs::file_size(archive);return h;
}
void verify_file(const fs::path& archive,const fs::path& original){
    const auto h=inspect_archive(archive);
    require(h.original_size==fs::file_size(original),"file size verification failed");
    require(h.file_hash==hash_file(original),"original differs from archived input");
    std::ifstream in(archive,std::ios::binary);read_header(in);
    std::ifstream original_stream(original,std::ios::binary);
    require(bool(original_stream),"cannot open original for byte-exact CTR audit");
    uint64_t checksum=FNV_INIT;
    std::array<uint8_t,TILE_BYTES> comparison{};
    for(uint64_t id=0;id<h.tiles;id++){
        auto tile=read_tile(in,id);auto raw=decode_tile(tile);
        original_stream.read(reinterpret_cast<char*>(comparison.data()),tile.valid);
        require(original_stream.gcount()==std::streamsize(tile.valid),"original truncated during CTR audit");
        require(std::equal(raw.begin(),raw.begin()+tile.valid,comparison.begin()),
                "CTR FAILED: actual bytes differ from archived reconstruction");
        checksum=hash_bytes(raw.data(),tile.valid,checksum);
    }
    require(checksum==h.file_hash,"decoded archive differs from original hash");
    require(original_stream.peek()==EOF,"original changed during CTR audit");
    require(in.peek()==EOF,"extra bytes after last tile");
}
} // namespace dm3d
