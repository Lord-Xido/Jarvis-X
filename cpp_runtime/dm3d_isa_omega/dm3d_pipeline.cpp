// DM-3D-ISA-Omega v2 | bounded streaming, trained codec and self-hosted pipeline.
// Reference implementation only. NO physical device / actuator control.
#define DM3D_OMEGA_NO_MAIN
#include "dm3d_isa.cpp"
#include <cmath>
#include <future>
#include <list>
#include <numeric>
#include <thread>
#include <cstdio>
#ifdef DM3D_HAS_CUDA
extern "C" int dm3d_cuda_available();
extern "C" int dm3d_cuda_block_encode(const std::uint8_t*,std::uint8_t*);
#endif

namespace dm3d::v2 {
using Path=std::filesystem::path;
constexpr u64 MAX_TILES=(u64{1}<<35); // demonstration's 64-bit tile-index range
constexpr size_t MODEL_LEN=25*4;
constexpr size_t MAX_CAPSULE=40000;

u32 read32(std::istream& in){u8 b[4];in.read(reinterpret_cast<char*>(b),4);require(in.gcount()==4,"Truncated u32");return u32(b[0])|(u32(b[1])<<8)|(u32(b[2])<<16)|(u32(b[3])<<24);}
u64 read_u64(std::istream& in){u64 lo=read32(in),hi=read32(in);return lo|(hi<<32);}
void write_u32(std::ostream& out,u32 x){for(int i=0;i<4;++i)out.put(char((x>>(8*i))&255));require(bool(out),"u32 write failed");}
void write_u64(std::ostream& out,u64 x){write_u32(out,u32(x));write_u32(out,u32(x>>32));}
void write_bytes(std::ostream& out,const Bytes& b){out.write(reinterpret_cast<const char*>(b.data()),std::streamsize(b.size()));require(bool(out),"Binary write failed");}
void write_digest(std::ostream& out,const Digest& d){out.write(reinterpret_cast<const char*>(d.data()),32);require(bool(out),"Digest write failed");}
Digest read_digest(std::istream& in){Digest d{};in.read(reinterpret_cast<char*>(d.data()),32);require(in.gcount()==32,"Truncated digest");return d;}
Bytes read_exact(std::istream& in,size_t n){Bytes b(n);in.read(reinterpret_cast<char*>(b.data()),std::streamsize(n));require(size_t(in.gcount())==n,"Truncated payload");return b;}
void ensure_eof(std::istream& in){require(in.peek()==std::char_traits<char>::eof(),"Unexpected trailing bytes");}
Coord coord_for(u64 tile){require(tile<MAX_TILES,"Tile index exceeds configured range");return demorton3(u128(tile)<<15);}
Digest chain_step(Digest prior,Digest tile_sha,u64 i){Bytes b(prior.begin(),prior.end());b.insert(b.end(),tile_sha.begin(),tile_sha.end());append_u64(b,i);return sha256(b);}
Digest file_digest(const Path& path){std::ifstream in(path,std::ios::binary);require(bool(in),"Cannot open file for digest");SHA256 hash;std::array<u8,65536> b{};while(in){in.read(reinterpret_cast<char*>(b.data()),b.size());auto n=in.gcount();if(n>0)hash.update(b.data(),size_t(n));}require(in.eof(),"Digest file read failure");return hash.finish();}

struct Model {
    std::array<double,8> ew{},dw{},db{};
    double eb=0;
    Model(){ew.fill(0.125);dw.fill(1.0);db.fill(0);}
    // Actual one-hidden-unit linear autoencoder trained with backpropagation.
    // Its exact decoder weights are stored Q16.16 in the archive.
    void train(const std::vector<Bytes>& samples,unsigned epochs=5){
        require(!samples.empty() && epochs>=1 && epochs<=50,"Invalid training dataset or epochs");
        for(const auto& s:samples)require(s.size()==TILE_VOL,"Training tile shape invalid");
        for(unsigned ep=0;ep<epochs;++ep){
            for(const auto& s:samples){
                for(int z=0;z<TILE_EDGE;z+=2)for(int y=0;y<TILE_EDGE;y+=2)for(int x=0;x<TILE_EDGE;x+=2){
                    // Deterministic sparse sampling limits optimization work.
                    if(((x+3*y+7*z+int(ep))&7)!=0)continue;
                    double input[8],pred[8],err[8];
                    int n=0;
                    for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
                        input[n++]=s[linear_index(x+dx,y+dy,z+dz,TILE_EDGE)]/255.0;
                    }
                    double latent=eb;for(int j=0;j<8;++j)latent+=ew[j]*input[j];
                    double deriv=0;
                    for(int j=0;j<8;++j){
                        pred[j]=dw[j]*latent+db[j];
                        err[j]=std::clamp(pred[j],0.0,1.0)-input[j];
                        // No gradient through saturated clamp on out-of-range predictions.
                        if(pred[j]<0||pred[j]>1)err[j]=0;
                        deriv+=err[j]*dw[j];
                    }
                    constexpr double lr=0.006;
                    for(int j=0;j<8;++j){
                        dw[j]=std::clamp(dw[j]-lr*err[j]*latent,-3.0,3.0);
                        db[j]=std::clamp(db[j]-lr*err[j],-3.0,3.0);
                    }
                    for(int j=0;j<8;++j)ew[j]=std::clamp(ew[j]-lr*deriv*input[j],-3.0,3.0);
                    eb=std::clamp(eb-lr*deriv,-3.0,3.0);
                }
            }
        }
    }
    Bytes serialize() const{
        Bytes bytes;bytes.reserve(MODEL_LEN);
        auto emit=[&](double x){require(std::isfinite(x)&&std::abs(x)<32767,"Invalid model weight");
            std::int32_t q=std::int32_t(std::llround(x*65536.0));append_u32(bytes,std::bit_cast<u32>(q));};
        for(double x:ew)emit(x);
        emit(eb);
        for(double x:dw)emit(x);
        for(double x:db)emit(x);
        require(bytes.size()==MODEL_LEN,"Model serialization length error");return bytes;
    }
    static Model from_bytes(const Bytes& bytes){require(bytes.size()==MODEL_LEN,"Unexpected model length");Reader r(bytes);Model m;
        auto take=[&](){return double(std::bit_cast<std::int32_t>(r.get32()))/65536.0;};
        for(auto& x:m.ew)x=take();
        m.eb=take();
        for(auto& x:m.dw)x=take();
        for(auto& x:m.db)x=take();
        r.end();
        for(double x:m.ew)require(std::isfinite(x)&&std::abs(x)<=4,"Bad encoder coefficient");
        for(double x:m.dw)require(std::isfinite(x)&&std::abs(x)<=4,"Bad decoder coefficient");
        for(double x:m.db)require(std::isfinite(x)&&std::abs(x)<=4,"Bad decoder bias");
        require(std::abs(m.eb)<=4,"Bad latent bias");return m;
    }
    Model quantized() const{return from_bytes(serialize());}
    Bytes encode(const Bytes& tile) const{
        require(tile.size()==TILE_VOL,"Encoder requires one 32-cubed tile");Bytes z(LAT_VOL);
        for(int zz=0;zz<LAT_EDGE;++zz)for(int yy=0;yy<LAT_EDGE;++yy)for(int xx=0;xx<LAT_EDGE;++xx){
            int j=0;double latent=eb;
            for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
                latent+=ew[j++]*double(tile[linear_index(xx*2+dx,yy*2+dy,zz*2+dz,TILE_EDGE)])/255.0;
            z[linear_index(xx,yy,zz,LAT_EDGE)]=u8(std::lround(std::clamp(latent,0.0,1.0)*255.0));
        }
        return z;
    }
    Bytes encode_fast(const Bytes& tile) const{
        require(tile.size()==TILE_VOL,"Encoder requires one 32-cubed tile");
        Bytes z(LAT_VOL);
        for(int zz=0;zz<16;++zz)for(int yy=0;yy<16;++yy)for(int xx=0;xx<16;++xx){
            int pos=linear_index(xx*2,yy*2,zz*2,32);
            const std::array<int,8> offset{0,1,32,33,1024,1025,1056,1057};
            double latent=eb;
            for(int j=0;j<8;++j)latent+=ew[j]*double(tile[pos+offset[j]])/255.0;
            z[linear_index(xx,yy,zz,16)]=u8(std::lround(std::clamp(latent,0.0,1.0)*255.0));
        }
        return z;
    }
    Bytes decode(const Bytes& z) const{
        require(z.size()==LAT_VOL,"Decoder requires one 16-cubed latent tile");Bytes out(TILE_VOL);
        for(int zz=0;zz<LAT_EDGE;++zz)for(int yy=0;yy<LAT_EDGE;++yy)for(int xx=0;xx<LAT_EDGE;++xx){
            double latent=z[linear_index(xx,yy,zz,LAT_EDGE)]/255.0;int j=0;
            for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){
                double p=std::clamp(dw[j]*latent+db[j],0.0,1.0);
                out[linear_index(xx*2+dx,yy*2+dy,zz*2+dz,TILE_EDGE)]=u8(std::lround(p*255.0));++j;
            }
        }
        return out;
    }
};

// Per-tile packet, keyed by the same Morton tile index used by the original VM.
struct Frame{u64 index=0;u32 raw_len=0;Bytes bytes;Digest raw_sha{};};
Bytes encode_frame(const Bytes& actual,u64 index,const Model& m,bool fast=false){
    require(!actual.empty()&&actual.size()<=TILE_VOL,"Frame length must be in 1..32768");
    u8 mode=2;
    Bytes latent,payload=actual;
    if(std::all_of(actual.begin(),actual.end(),[](u8 x){return x==0;})){
        mode=3;payload.clear();
    }else if(std::all_of(actual.begin(),actual.end(),[first=actual.front()](u8 x){return x==first;})){
        mode=4;payload=Bytes{actual.front()};
    }else{
        Bytes raw_rle=zero_rle(actual);
        if(raw_rle.size()<payload.size()){mode=5;payload=std::move(raw_rle);}
        Bytes padded=actual;padded.resize(TILE_VOL,0);
        Bytes z=fast?m.encode_fast(padded):m.encode(padded);
        Bytes pred=m.decode(z),res=xor_equal(padded,pred),res_rle=zero_rle(res);
        if(res_rle.size()<res.size())res=std::move(res_rle);
        if(z.size()+res.size()<payload.size()){
            mode=res.size()<TILE_VOL?1:0;
            latent=std::move(z);payload=std::move(res);
        }
    }
    Bytes out;const std::string magic="DM3DTIL2";out.insert(out.end(),magic.begin(),magic.end());
    append_u64(out,index);append_u32(out,u32(actual.size()));out.push_back(mode);
    append_u32(out,u32(latent.size()));append_u32(out,u32(payload.size()));
    Digest sha=sha256(actual);out.insert(out.end(),sha.begin(),sha.end());
    out.insert(out.end(),latent.begin(),latent.end());out.insert(out.end(),payload.begin(),payload.end());
    require(out.size()<=MAX_CAPSULE,"Encoded frame exceeds limit");return out;
}
Frame decode_frame(const Bytes& packet,const Model& model){
    require(packet.size()<=MAX_CAPSULE,"Oversized frame");Reader r(packet);for(char ch:std::string("DM3DTIL2"))require(r.get()==u8(ch),"Bad tile magic");
    u64 index=r.get64();require(index<MAX_TILES,"Bad Morton tile index");
    u32 len=r.get32();require(len>=1&&len<=TILE_VOL,"Invalid decoded tile length");
    u8 mode=r.get();require(mode<=5,"Bad codec mode");
    u32 lat_len=r.get32(),payload_len=r.get32();
    require(payload_len<=TILE_VOL*2,"Oversized residual");
    require(((mode==2)&&lat_len==0&&payload_len==len)
        ||(mode==3&&lat_len==0&&payload_len==0)
        ||(mode==4&&lat_len==0&&payload_len==1)
        ||(mode==5&&lat_len==0&&payload_len>0)
        ||(mode<=1&&lat_len==LAT_VOL),"Invalid frame sections");
    auto expected_v=r.take(32);Digest expected{};std::copy(expected_v.begin(),expected_v.end(),expected.begin());
    Bytes latent=r.take(lat_len),payload=r.take(payload_len);r.end();
    Bytes raw;
    if(mode==2){raw=std::move(payload);}
    else if(mode==3){raw=Bytes(len,0);}
    else if(mode==4){raw=Bytes(len,payload.front());}
    else if(mode==5){raw=zero_unrle(payload,len);}
    else{
        auto res=(mode==1)?zero_unrle(payload,TILE_VOL):payload;
        require(res.size()==TILE_VOL,"Residual output shape mismatch");
        raw=xor_equal(model.decode(latent),res);raw.resize(len);
    }
    require(sha256(raw)==expected,"Tile SHA-256 mismatch");return Frame{index,len,std::move(raw),expected};
}

struct Statistics{u64 tiles=0,raw_bytes=0,stored_bytes=0,checkpoints=0;Digest chain{};std::vector<Digest> shard_chains;double seconds=0;};
struct WorkItem{u64 index;Bytes data;};
// Executes the actual v1 VM microprogram on every packed tile before storing it.
Bytes execute_tile_encode(WorkItem task,const Model& model,bool fast){
    Bytes padded=task.data;padded.resize(TILE_VOL,0);
    Coord c=coord_for(task.index);auto address=morton3(c);
    VM vm(1);vm.world.allocate(address)->bytes=padded;
    vm.run(build_demo_program(c));
    require(vm.ledger.size()==1 && vm.T[3]->bytes==padded,"VM microprogram failed tile invariants");
    return encode_frame(task.data,task.index,model,fast);
}
Statistics pack_stream(const Path& src,const Path& output,Model m,unsigned threads=2,unsigned shards=2,bool fast=false){
    require(threads>=1&&threads<=16&&shards>=1&&shards<=64,"Thread or logical shard limit exceeded");
    require(!std::filesystem::exists(output),"Archive already exists; refusing overwrite");
    const u64 length=std::filesystem::file_size(src);
    const u64 count=length/TILE_VOL+u64(length%TILE_VOL!=0);require(count<MAX_TILES,"Input exceeds addressable tiles");
    std::ifstream in(src,std::ios::binary);require(bool(in),"Cannot open stream source");
    auto tmp=Path(output.string()+".partial");require(!std::filesystem::exists(tmp),"Temporary archive exists");
    auto start=std::chrono::steady_clock::now();
    Statistics stats;stats.shard_chains.resize(shards);
    m=m.quantized();
    try{
        std::fstream out(tmp,std::ios::binary|std::ios::out|std::ios::in|std::ios::trunc);require(bool(out),"Cannot create archive");
        out.write("DM3DST02",8);write_u32(out,2);write_u64(out,length);write_u64(out,count);
        write_digest(out,Digest{});write_bytes(out,m.serialize());write_u32(out,shards);
        SHA256 overall;
        for(u64 i=0;i<count;){
            std::vector<std::future<Bytes>> jobs;std::vector<WorkItem> tasks;
            const unsigned batch=unsigned(std::min<u64>(threads,count-i));tasks.reserve(batch);jobs.reserve(batch);
            for(unsigned j=0;j<batch;++j){
                const u64 n=std::min<u64>(TILE_VOL,length-(i+j)*TILE_VOL);
                Bytes part=read_exact(in,size_t(n));overall.update(part);
                tasks.push_back(WorkItem{i+j,std::move(part)});
            }
            for(auto& t:tasks)jobs.emplace_back(std::async(std::launch::async,[&,task=std::move(t)]() mutable{return execute_tile_encode(std::move(task),m,fast);}));
            for(unsigned j=0;j<batch;++j){
                Bytes packet=jobs[j].get();write_u32(out,u32(packet.size()));write_bytes(out,packet);
                Frame frame=decode_frame(packet,m);require(frame.index==i+j,"Non-deterministic worker ordering");
                stats.chain=chain_step(stats.chain,frame.raw_sha,frame.index);
                auto& shard=stats.shard_chains[size_t(frame.index%shards)];shard=chain_step(shard,frame.raw_sha,frame.index);
                stats.raw_bytes+=frame.raw_len;stats.stored_bytes+=4+packet.size();++stats.tiles;++stats.checkpoints;
            }
            i+=batch;
        }
        ensure_eof(in);write_digest(out,stats.chain);
        for(const auto& hash:stats.shard_chains)write_digest(out,hash);
        out.seekp(28,std::ios::beg);write_digest(out,overall.finish());out.flush();require(bool(out),"Archive finalize failed");out.close();
        std::filesystem::rename(tmp,output);
    }catch(...){std::error_code ec;std::filesystem::remove(tmp,ec);throw;}
    stats.stored_bytes=std::filesystem::file_size(output);
    stats.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return stats;
}
Statistics unpack_stream(const Path& archive,const Path& output,unsigned threads=2){
    require(threads>=1&&threads<=16,"Invalid thread count");require(!std::filesystem::exists(output),"Decoded file already exists; refusing overwrite");
    auto tmp=Path(output.string()+".partial");require(!std::filesystem::exists(tmp),"Temporary decoded file exists");
    auto start=std::chrono::steady_clock::now();Statistics stats;
    std::ifstream in(archive,std::ios::binary);require(bool(in),"Cannot open archive");
    for(char c:std::string("DM3DST02"))require(in.get()==c,"Invalid archive magic");
    require(read32(in)==2,"Unsupported archive version");
    u64 expected_size=read_u64(in),count=read_u64(in);require(count<MAX_TILES,"Invalid archive tile count");
    require(count==expected_size/TILE_VOL+u64(expected_size%TILE_VOL!=0),"Inconsistent archive file size");
    Digest expected_sha=read_digest(in);Model model=Model::from_bytes(read_exact(in,MODEL_LEN));
    u32 shards=read32(in);require(shards>=1&&shards<=64,"Invalid shard count");stats.shard_chains.resize(shards);
    SHA256 overall;
    try{
        std::ofstream out(tmp,std::ios::binary|std::ios::trunc);require(bool(out),"Cannot create output file");
        for(u64 i=0;i<count;){
            const unsigned batch=unsigned(std::min<u64>(threads,count-i));
            std::vector<std::future<Frame>> jobs;jobs.reserve(batch);
            for(unsigned j=0;j<batch;++j){
                u32 n=read32(in);require(n>=61&&n<=MAX_CAPSULE,"Invalid tile frame size");
                Bytes capsule=read_exact(in,n);
                jobs.emplace_back(std::async(std::launch::async,[packet=std::move(capsule),model]() {return decode_frame(packet,model);}));
            }
            for(unsigned j=0;j<batch;++j){
                Frame frame=jobs[j].get();require(frame.index==i+j,"Out-of-order or duplicated tile index");
                u64 expected_len=std::min<u64>(TILE_VOL,expected_size-frame.index*TILE_VOL);
                require(frame.raw_len==expected_len,"Incorrect final tile length");
                write_bytes(out,frame.bytes);overall.update(frame.bytes);
                stats.chain=chain_step(stats.chain,frame.raw_sha,frame.index);
                auto& shard=stats.shard_chains[size_t(frame.index%shards)];shard=chain_step(shard,frame.raw_sha,frame.index);
                stats.raw_bytes+=frame.raw_len;stats.tiles++;stats.checkpoints++;
            }i+=batch;
        }
        require(read_digest(in)==stats.chain,"Stream checkpoint chain mismatch");
        for(const auto& h:stats.shard_chains)require(read_digest(in)==h,"Shard checkpoint chain mismatch");
        ensure_eof(in);require(overall.finish()==expected_sha,"Stream SHA-256 mismatch");
        out.flush();require(bool(out),"Decode output write failed");out.close();
        std::filesystem::rename(tmp,output);
    }catch(...){std::error_code ec;std::filesystem::remove(tmp,ec);throw;}
    stats.stored_bytes=std::filesystem::file_size(archive);
    stats.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return stats;
}

// Persistent tile store: durable per-Morton-key capsules and capacity-bounded RAM.
class DiskWorld {
    Path dir_;
    Model model_;
    size_t max_resident_;
    std::list<u64> lru_;
    struct Resident {Bytes bytes;std::list<u64>::iterator position;};
    std::unordered_map<u64,Resident> cache_;
    size_t evictions_=0,loads_=0;
    Path filename(u64 index)const{std::ostringstream s;s<<"tile_"<<std::hex<<std::setw(16)<<std::setfill('0')<<index<<".dm3dt";return dir_/s.str();}
    void evict(){while(cache_.size()>max_resident_){u64 k=lru_.back();lru_.pop_back();cache_.erase(k);++evictions_;}}
    void insert(u64 index,Bytes data){
        auto existing=cache_.find(index);if(existing!=cache_.end()){lru_.erase(existing->second.position);cache_.erase(existing);}
        lru_.push_front(index);cache_.emplace(index,Resident{std::move(data),lru_.begin()});evict();
    }
public:
    DiskWorld(Path path,Model model,size_t cap=2):dir_(std::move(path)),model_(model.quantized()),max_resident_(cap){
        require(cap>=1&&cap<=1024,"Resident cache cap invalid");std::filesystem::create_directories(dir_);
        auto manifest=dir_/"model.q16";Bytes serialized=model_.serialize();
        if(std::filesystem::exists(manifest))require(read_file(manifest.string())==serialized,"Persistent store model mismatch");
        else write_file(manifest.string(),serialized);
    }
    void put(u64 index,const Bytes& data){
        require(index<MAX_TILES,"Invalid tile index");Bytes capsule=encode_frame(data,index,model_);
        Path p=filename(index),tmp=Path(p.string()+".partial");
        require(!std::filesystem::exists(tmp),"Temporary store file exists");
        try{write_file(tmp.string(),capsule);std::filesystem::rename(tmp,p);}
        catch(...){std::error_code ec;std::filesystem::remove(tmp,ec);throw;}
        insert(index,data);
    }
    Bytes get(u64 index){
        auto it=cache_.find(index);if(it!=cache_.end()){Bytes b=it->second.bytes;insert(index,b);return b;}
        Bytes payload=read_file(filename(index).string());Frame f=decode_frame(payload,model_);
        require(f.index==index,"Storage key mismatch");++loads_;insert(index,f.bytes);return f.bytes;
    }
    size_t resident()const{return cache_.size();}
    size_t evictions()const{return evictions_;}
    size_t loads()const{return loads_;}
};

// Actual software volume raycaster: opacity integration through 3D scalar field -> PPM.
void render_tile(const Bytes& tile,const Path& path,int size=256){
    require(tile.size()==TILE_VOL,"Raycaster input tile wrong size");require(size>=32&&size<=1024,"Ray image dimension invalid");
    Bytes pix(size_t(size)*size*3);
    for(int py=0;py<size;++py)for(int px=0;px<size;++px){
        int x=std::min(31,px*32/size),y=std::min(31,py*32/size);
        double r=0,g=0,b=0,alpha=0;
        for(int z=0;z<32;++z){
            double density=tile[linear_index(x,y,z,32)]/255.0;
            if(density<=0)continue;
            auto sample=[&](int xx,int yy,int zz){return double(tile[linear_index(std::clamp(xx,0,31),std::clamp(yy,0,31),std::clamp(zz,0,31),32)]);};
            double gx=sample(x+1,y,z)-sample(x-1,y,z);
            double gy=sample(x,y+1,z)-sample(x,y-1,z);
            double gz=sample(x,y,z+1)-sample(x,y,z-1);
            double grad=std::sqrt(gx*gx+gy*gy+gz*gz);
            double shade=0.35+0.65*std::abs(gz)/(grad+1.0);
            double a=(1-std::exp(-density*0.30))*(1-alpha);
            r+=a*(0.2+0.8*shade);g+=a*(0.15+0.60*shade);b+=a*(0.25+0.55*shade);
            alpha+=a;if(alpha>0.995)break;
        }
        size_t o=(size_t(py)*size+px)*3;
        pix[o]=u8(std::lround(std::clamp(r+0.015,0.0,1.0)*255));
        pix[o+1]=u8(std::lround(std::clamp(g+0.015,0.0,1.0)*255));
        pix[o+2]=u8(std::lround(std::clamp(b+0.025,0.0,1.0)*255));
    }
    std::ofstream out(path,std::ios::binary|std::ios::trunc);require(bool(out),"Cannot open PPM output");
    out<<"P6\n"<<size<<' '<<size<<"\n255\n";write_bytes(out,pix);
}

// Bounded kernel autotuning: verify outputs before measuring candidate latency.
struct Tuning{std::string winner;double generic_ms=0,unrolled_ms=0;};
Bytes block_encode_alt(const Bytes& in){
    require(in.size()==TILE_VOL,"Invalid benchmark tile");Bytes out(LAT_VOL);
    for(int z=0;z<16;++z)for(int y=0;y<16;++y)for(int x=0;x<16;++x){
        int at=linear_index(2*x,2*y,2*z,32);
        unsigned sum=unsigned(in[at])+in[at+1]+in[at+32]+in[at+33]
               +in[at+1024]+in[at+1025]+in[at+1056]+in[at+1057];
        out[linear_index(x,y,z,16)]=u8((sum+4)/8);
    }
        return out;
}
Tuning tune_encoder(const Model& model){
    for(u32 seed:{0u,3u,9999u}){
        auto tile=make_tile(seed);
        require(model.encode(tile)==model.encode_fast(tile),"Learned kernel candidate failed equivalence test");
        require(block_encode(tile)==block_encode_alt(tile),"Block kernel mismatch");
    }
    const Bytes test=make_tile(42);volatile u64 sink=0;
    auto timeit=[&](auto fn){
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<48;++i){auto result=fn(test);sink=sink+result[i%LAT_VOL];}
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    };
    auto generic=[&](const Bytes& x){return model.encode(x);};
    auto unrolled=[&](const Bytes& x){return model.encode_fast(x);};
    (void)timeit(generic);(void)timeit(unrolled);
    double a=timeit(generic),b=timeit(unrolled);(void)sink;
    return Tuning{a<=b?"generic":"unrolled",a,b};
}

// No real hardware access; independent toy digital-twin command guard.
struct DeviceRequest{bool authenticated=false;double setpoint=0;bool emergency=false;};
bool permitted(const DeviceRequest& r){return r.authenticated && std::isfinite(r.setpoint)&&r.setpoint>=-30&&r.setpoint<=5&&!r.emergency;}

// ISA-level orchestration. These extension opcodes invoke bounded host services,
// while actual voxel processing runs through VM::run(build_demo_program()).
enum PipelineOp:u8 {TRAIN=0x80,TUNE=0x81,DISPATCH=0x82,STREAM_PACK=0x83,STREAM_UNPACK=0x84,
                    STORE_COMMIT=0x85,RENDER=0x86,POLICY_CHECK=0x87,VERIFY=0x88};
struct PipelineVM{
    Path in,archive,out,store,ppm;Model model;unsigned threads=2,shards=3;
    Statistics packed,unpacked;Tuning tuning;std::vector<std::string> logs;
    void run(const std::vector<u32>& code,size_t fuel=32){
        require(!code.empty()&&code.size()==10,"Invalid pipeline bytecode length");
        const std::array<u8,10> expected_ops{TRAIN,TUNE,DISPATCH,STREAM_PACK,STREAM_UNPACK,STORE_COMMIT,RENDER,POLICY_CHECK,VERIFY,HALT};
        for(size_t pc=0,steps=0;pc<code.size();++pc){
            require(steps++<fuel,"Pipeline instruction fuel exceeded");auto d=decode(code[pc]);
            require(d.op==expected_ops[pc],"Pipeline bytecode ordering violates verified schedule");
            require(d.ext==0&&d.dst==0&&d.a==0&&d.b==0&&d.flags==0,"Unsupported pipeline instruction arguments");
            switch(d.op){
                case HALT:require(pc==code.size()-1,"HALT must terminate pipeline");return;
                case TRAIN:{std::ifstream f(in,std::ios::binary);require(bool(f),"Training input missing");
                    std::vector<Bytes> samples;for(int i=0;i<12;++i){Bytes b(TILE_VOL);f.read(reinterpret_cast<char*>(b.data()),TILE_VOL);if(f.gcount()==0)break;
                        b.resize(size_t(f.gcount()));b.resize(TILE_VOL,0);samples.push_back(std::move(b));if(f.gcount()<TILE_VOL)break;}
                    if(!samples.empty())model.train(samples);
                    model=model.quantized();logs.push_back("TRAIN model="+hex_digest(sha256(model.serialize())));break;}
                case TUNE:{tuning=tune_encoder(model);logs.push_back("TUNE "+tuning.winner);break;}
                case DISPATCH:{require(threads>=1&&threads<=16&&shards>=1&&shards<=64,"Bad work scheduling");logs.push_back("DISPATCH CPU workers="+std::to_string(threads)+" logical shards="+std::to_string(shards));break;}
                case STREAM_PACK:packed=pack_stream(in,archive,model,threads,shards,tuning.winner=="unrolled");logs.push_back("STREAM.PACK "+std::to_string(packed.raw_bytes)+" bytes");break;
                case STREAM_UNPACK:unpacked=unpack_stream(archive,out,threads);logs.push_back("STREAM.UNPACK "+std::to_string(unpacked.raw_bytes)+" bytes");break;
                case STORE_COMMIT:{
                    std::vector<Bytes> expected;
                    {
                        DiskWorld world(store,model,2);
                        std::ifstream f(out,std::ios::binary);require(bool(f),"Cannot read decoded output for store");
                        for(u64 i=0;i<4;++i){
                            Bytes b(TILE_VOL);f.read(reinterpret_cast<char*>(b.data()),TILE_VOL);
                            if(f.gcount()==0)break;
                            b.resize(size_t(f.gcount()));expected.push_back(b);
                            world.put(i,b);require(world.get(i)==b,"Store write verification failed");
                            if(b.size()!=TILE_VOL)break;
                        }
                        require(world.evictions()>0||expected.size()<=2,"Cache eviction was not exercised");
                        logs.push_back("STORE.COMMIT resident="+std::to_string(world.resident())+" evicted="+std::to_string(world.evictions()));
                    }
                    DiskWorld reopened(store,model,1);
                    for(size_t i=0;i<expected.size();++i)require(reopened.get(u64(i))==expected[i],"Cold disk read mismatch");
                    require(reopened.loads()>0,"Store did not reload from persistent disk");
                    logs.push_back("STORE.REOPEN verified="+std::to_string(expected.size())+" tiles");
                    break;}
                case RENDER:{std::ifstream f(out,std::ios::binary);require(bool(f),"Cannot read volume");Bytes tile(TILE_VOL);
                    f.read(reinterpret_cast<char*>(tile.data()),TILE_VOL);require(f.gcount()>0,"Cannot render empty source");tile.resize(size_t(f.gcount()));tile.resize(TILE_VOL,0);
                    render_tile(tile,ppm);logs.push_back("RENDER "+ppm.string());break;}
                case POLICY_CHECK:require(permitted({true,-5,false})&&!permitted({false,-5,false})&&!permitted({true,50,false})&&!permitted({true,-5,true}),"Policy failure");logs.push_back("POLICY.CHECK sandbox only; no actuator access");break;
                case VERIFY:require(packed.tiles==unpacked.tiles&&packed.chain==unpacked.chain,"Mismatched pipeline checkpoint chains");
                    require(file_digest(in)==file_digest(out),"End-to-end stream digest mismatch");logs.push_back("VERIFY source SHA-256="+hex_digest(file_digest(in)));break;
                default:fail("Unknown pipeline ISA extension opcode "+std::to_string(d.op));
            }
        }
        fail("Pipeline program missing HALT");
    }
};
std::vector<u32> make_pipeline_program(){Builder b;for(u8 op:std::array<u8,10>{TRAIN,TUNE,DISPATCH,STREAM_PACK,STREAM_UNPACK,STORE_COMMIT,RENDER,POLICY_CHECK,VERIFY,HALT})b.add(op);return b.words;}
void generate_input(const Path& path){
    require(!std::filesystem::exists(path),"Sample input exists");
    std::ofstream out(path,std::ios::binary);require(bool(out),"Cannot create sample file");
    for(int i=0;i<7;++i){Bytes data=i==3?Bytes(TILE_VOL,0):make_tile(u32(i*113+19));write_bytes(out,data);}
    Bytes tail=make_tile(511);tail.resize(7117);write_bytes(out,tail);
}
void execute_demo(const Path& directory){
    std::filesystem::create_directories(directory);
    PipelineVM vm;vm.in=directory/"source.raw";vm.archive=directory/"volume.dm3ds2";
    vm.out=directory/"restored.raw";vm.store=directory/"sparse-store";vm.ppm=directory/"raycast.ppm";
    generate_input(vm.in);
    auto program=make_pipeline_program();auto code=words_to_bytes(program);
    write_file((directory/"pipeline.bytecode.bin").string(),code);
    require(bytes_to_words(read_file((directory/"pipeline.bytecode.bin").string()))==program,"Pipeline bytecode did not round-trip");
    vm.run(bytes_to_words(read_file((directory/"pipeline.bytecode.bin").string())));
    write_file((directory/"trained-model.q16").string(),vm.model.serialize());
    std::ostringstream report;
    report<<"DM-3D-ISA-OMEGA v2 end-to-end verified\n";
    for(const auto& line:vm.logs)report<<line<<'\n';
    report<<"Tiles="<<vm.packed.tiles<<" input_bytes="<<vm.packed.raw_bytes<<" archive_bytes="<<vm.packed.stored_bytes<<'\n';
    report<<"Pack_s="<<vm.packed.seconds<<" unpack_s="<<vm.unpacked.seconds<<'\n';
    report<<"Final checkpoint="<<hex_digest(vm.packed.chain)<<'\n';
    report<<"Tuning generic_ms="<<vm.tuning.generic_ms<<" unrolled_ms="<<vm.tuning.unrolled_ms<<'\n';
    report<<"Cloud/GPU/hardware devices: NOT deployed; pipeline uses CPU and logical shards.\n";
    write_text((directory/"report.txt").string(),report.str());
    std::cout<<report.str();
}
void self_test_v2(){
    Model model;auto raw=make_tile();model.train({raw},2);model=model.quantized();
    require(model.serialize()!=Model{}.serialize(),"Model training produced no weight update");
    require(model.encode(raw)==model.encode_fast(raw),"Learned model kernel parity failed");
    auto zero=Bytes(TILE_VOL,0);auto solid=Bytes(TILE_VOL,177);auto noisy=Bytes(TILE_VOL,0);
    require(encode_frame(zero,4,model).size()<120,"Zero tile did not compact");
    require(encode_frame(solid,4,model).size()<120,"Uniform tile did not compact");
    for(size_t i=0;i<noisy.size();++i)noisy[i]=u8((i*239+17)%256);
    for(const auto& tile:{raw,zero,solid,noisy,Bytes{1,2,3,4,5}}){
        Bytes padded=tile;padded.resize(TILE_VOL,0);
        require(model.encode(padded)==model.encode_fast(padded),"Optimized learned encoder mismatch");
        auto packet=encode_frame(tile,4,model);require(decode_frame(packet,model).bytes==tile,"Frame round-trip failed");
        auto tampered=packet;tampered.back()^=0x1;bool bad=false;
        try{(void)decode_frame(tampered,model);}catch(const std::exception&){bad=true;}require(bad,"Tampered frame was accepted");
    }
    require(demorton3(morton3(coord_for(123))).x==coord_for(123).x,"Spatial addressing mismatch");
    require(block_encode_alt(raw)==block_encode(raw),"Kernel equivalence failed");
    require(permitted({true,0,false})&&!permitted({false,0,false})&&!permitted({true,100,false}),"Safety policy test failed");
    auto src=quine_source();auto prog=assemble_quine_source(src);VM q;q.run(prog);require(src==q.output,"Original VM quine regression");
    std::cout<<"PASS: DM3D v2 model training, packet round-trip, corruption rejection, Morton address, kernel parity, policy simulation, original VM quine\n";
}
void print_help(){std::cout<<"DM-3D-ISA-OMEGA v2: end-to-end virtual pipeline\n"
    <<"  --self-test-v2\n  --e2e OUTPUT_DIRECTORY (must contain no generated files)\n"
    <<"  --stream-encode INPUT OUTPUT [THREADS]\n  --stream-decode INPUT OUTPUT [THREADS]\n"
    <<"  --render TILE_32768.raw OUTPUT.ppm\n  --tune\n  --cuda-check\n";}
unsigned get_threads(int argc,char** argv,int index){if(argc<=index)return 2;unsigned n=0;std::string s=argv[index];auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);require(e==std::errc{}&&p==s.data()+s.size()&&n>=1&&n<=16,"Threads must be 1..16");return n;}
} // namespace dm3d::v2

int main(int argc,char** argv){
    using namespace dm3d;using namespace dm3d::v2;
    try{
        require(argc>=2,"See --help");const std::string cmd=argv[1];
        if(cmd=="--e2e"){require(argc==3,"Usage --e2e DIRECTORY");execute_demo(argv[2]);}
        else if(cmd=="--self-test-v2"){require(argc==2,"Usage --self-test-v2");self_test_v2();}
        else if(cmd=="--stream-encode"){require(argc==4||argc==5,"Usage --stream-encode INPUT OUTPUT [THREADS]");
            Model model;std::ifstream in(argv[2],std::ios::binary);require(bool(in),"Training source missing");
            Bytes b(TILE_VOL);in.read(reinterpret_cast<char*>(b.data()),TILE_VOL);
            if(in.gcount()>0){b.resize(size_t(in.gcount()));b.resize(TILE_VOL,0);model.train({b});}
            auto tune=tune_encoder(model.quantized());
            auto stat=pack_stream(argv[2],argv[3],model,get_threads(argc,argv,4),2,tune.winner=="unrolled");
            std::cout<<"Packed "<<stat.raw_bytes<<" bytes into "<<stat.stored_bytes<<" bytes, "<<stat.tiles<<" tiles, SHA-chain "<<hex_digest(stat.chain)<<"\n";}
        else if(cmd=="--stream-decode"){require(argc==4||argc==5,"Usage --stream-decode INPUT OUTPUT [THREADS]");
            auto stat=unpack_stream(argv[2],argv[3],get_threads(argc,argv,4));
            std::cout<<"Verified and decoded "<<stat.raw_bytes<<" bytes in "<<stat.tiles<<" tiles\n";}
        else if(cmd=="--render"){require(argc==4,"Usage --render TILE_32768.raw IMAGE.ppm");auto t=read_file(argv[2]);render_tile(t,argv[3]);std::cout<<"Rendered PPM "<<argv[3]<<"\n";}
        else if(cmd=="--tune"){require(argc==2,"Usage --tune");auto t=tune_encoder(Model{});std::cout<<"Verified kernel winner="<<t.winner<<" generic="<<t.generic_ms<<" ms unrolled="<<t.unrolled_ms<<" ms\n";}
        else if(cmd=="--cuda-check"){
            require(argc==2,"Usage --cuda-check");
#ifdef DM3D_HAS_CUDA
            if(!dm3d_cuda_available()){std::cout<<"CUDA backend built; no available GPU for execution\n";}
            else{
                Bytes original=make_tile(44),result(LAT_VOL);
                int rc=dm3d_cuda_block_encode(original.data(),result.data());
                require(rc==0,"CUDA kernel execution failed");
                require(result==block_encode(original),"CUDA parity verification failed");
                std::cout<<"PASS: CUDA kernel produces byte-identical 2x2x2 latents\n";
            }
#else
            std::cout<<"CUDA unavailable: CUDA compiler/backend not built; CPU path operational\n";
#endif
        }
        else if(cmd=="--help")print_help();
        else fail("Unknown v2 command");
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
