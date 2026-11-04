#include "jx_core.h"
#include "jx_ann.hpp"
#include <bit>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <utility>

namespace {
using Clock = std::chrono::steady_clock;
struct Engine {
    uint32_t d, n;
    uint64_t rng=1;
    std::vector<float> psi, next, source, saved;
    std::vector<uint8_t> failed;
    std::vector<uint32_t> age;
    JXStats stats{};
    jx::TileANN ann;
    explicit Engine(uint32_t dimension, uint64_t seed): d(dimension), n(d*d*d), psi(n), next(n), source(n), saved(n), failed(n), age(n) {
        reset(seed);
    }
    uint64_t random() {
        rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
        return rng * UINT64_C(2685821657736338717);
    }
    void reset(uint64_t seed) {
        rng=seed?seed:UINT64_C(0x9e3779b97f4a7c15);
        stats={}; stats.dimension=d; stats.voxels=n;
        ann.initialize(seed);
        for(uint32_t z=0;z<d;z++) for(uint32_t y=0;y<d;y++) for(uint32_t x=0;x<d;x++) {
            const uint32_t i=x+d*y+d*d*z;
            const double X=(2.0*x/(d-1)-1.0), Y=(2.0*y/(d-1)-1.0), Z=(2.0*z/(d-1)-1.0);
            double r2=X*X+Y*Y+Z*Z;
            const float v=static_cast<float>(std::clamp(0.08+0.72*std::exp(-2.1*r2)
                     +0.09*std::sin(7*X+3*Z)*std::cos(5*Y)
                     +0.045*(double(random() & 1023)/1023.0),0.0,1.0));
            source[i]=v; psi[i]=v; next[i]=v; saved[i]=v;
            failed[i]=0; age[i]=0;
        }
        stats.ann_initial_mse=stats.ann_last_mse=ann.mse(tiled());
    }
    std::vector<uint8_t> bytes() const {
        std::vector<uint8_t> v(n);
        for(uint32_t i=0;i<n;i++) v[i]=static_cast<uint8_t>(std::lround(std::clamp(double(psi[i]),0.0,1.0)*255));
        return v;
    }
    std::vector<uint8_t> tiled() const {
        // True 2x2x2 spatial adjacency, not 8 consecutive raster positions.
        auto raster=bytes();
        std::vector<uint8_t> out;out.reserve(n);
        for(uint32_t z=0;z<d;z+=2)
        for(uint32_t y=0;y<d;y+=2)
        for(uint32_t x=0;x<d;x+=2)
        for(uint32_t dz=0;dz<2;dz++)
        for(uint32_t dy=0;dy<2;dy++)
        for(uint32_t dx=0;dx<2;dx++)
            out.push_back(raster[(z+dz)*d*d+(y+dy)*d+x+dx]);
        return out;
    }
    std::vector<uint8_t> untile(const std::vector<uint8_t>& packed) const {
        std::vector<uint8_t> raster(n);
        uint32_t at=0;
        for(uint32_t z=0;z<d;z+=2)
        for(uint32_t y=0;y<d;y+=2)
        for(uint32_t x=0;x<d;x+=2)
        for(uint32_t dz=0;dz<2;dz++)
        for(uint32_t dy=0;dy<2;dy++)
        for(uint32_t dx=0;dx<2;dx++)
            raster[(z+dz)*d*d+(y+dy)*d+x+dx]=packed[at++];
        return raster;
    }
    void tick() {
        auto previous=bytes();
        float max_delta=0;
        // Stable 6-neighbour convex-combination field evolution:
        // psi' = 0.78 psi + 0.18 average(healthy neighbours) + 0.04 source.
        for(uint32_t z=0;z<d;z++) for(uint32_t y=0;y<d;y++) for(uint32_t x=0;x<d;x++) {
            uint32_t i=x+d*y+d*d*z;
            if(failed[i]) {next[i]=0;continue;}
            float sum=0; int neighbors=0;
            auto add=[&](uint32_t j){if(!failed[j]) {sum+=psi[j]; neighbors++;}};
            if(x>0) add(i-1);
            if(x+1<d) add(i+1);
            if(y>0) add(i-d);
            if(y+1<d) add(i+d);
            if(z>0) add(i-d*d);
            if(z+1<d) add(i+d*d);
            const float avg=neighbors?sum/neighbors:psi[i];
            next[i]=std::clamp(0.78f*psi[i]+0.18f*avg+0.04f*source[i],0.0f,1.0f);
            max_delta=std::max(max_delta,std::abs(next[i]-psi[i]));
        }
        psi.swap(next);
        stats.last_max_delta=max_delta;
        for(uint32_t i=0;i<n;i++) if(failed[i]) {
            if(++age[i] >= 18) {
                failed[i]=0; age[i]=0;
                psi[i]=saved[i];
                stats.active_faults--; stats.repaired_faults++;
            }
        }
        auto current=bytes();
        uint64_t flips=0;
        for(uint32_t i=0;i<n;i++) flips+=std::popcount(unsigned(previous[i]^current[i]));
        stats.last_switching_bits=flips;
        stats.switching_bits_total+=flips;
        stats.byte_updates_total+=n;
        stats.ticks++;
    }
    uint32_t manhattan(uint32_t a,uint32_t b) const {
        const uint32_t dd=d*d;
        return static_cast<uint32_t>(std::abs(int(a%d)-int(b%d))
            +std::abs(int((a/d)%d)-int((b/d)%d))
            +std::abs(int(a/dd)-int(b/dd)));
    }
};
static uint32_t crc32(const uint8_t* data, uint32_t len) {
    uint32_t crc=0xffffffffu;
    for(uint32_t i=0;i<len;i++) {
        crc ^= data[i];
        for(int bit=0;bit<8;bit++) crc=(crc>>1)^((crc&1)?0xedb88320u:0u);
    }
    return ~crc;
}
static void put_u32(std::vector<uint8_t>& v, uint32_t n) {
    for(int j=0;j<4;j++) v.push_back(uint8_t((n>>(8*j))&255));
}
static uint32_t get_u32(const uint8_t* p) {
    return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);
}
}
extern "C" {
void* jx_create(uint32_t dimension, uint64_t seed) {
    if(dimension<2 || dimension>32 || dimension%2!=0) return nullptr;
    try {return new Engine(dimension,seed);} catch(...) {return nullptr;}
}
void jx_destroy(void* p) {delete static_cast<Engine*>(p);}
int jx_reset(void* p, uint64_t seed) {
    if(!p)return -1;
    static_cast<Engine*>(p)->reset(seed);return 0;
}
int jx_step(void* p, uint32_t iterations) {
    if(!p || iterations>10000) return -1;
    Engine& e=*static_cast<Engine*>(p);
    auto t=Clock::now();
    for(uint32_t i=0;i<iterations;i++)e.tick();
    const double ms=std::chrono::duration<double,std::milli>(Clock::now()-t).count();
    e.stats.last_step_ms=ms;
    e.stats.total_step_ms+=ms;
    return 0;
}
int jx_fail(void* p, uint32_t index) {
    if(!p)return -1;
    Engine& e=*static_cast<Engine*>(p);
    if(index>=e.n || e.failed[index])return -2;
    e.failed[index]=1;e.age[index]=0;e.saved[index]=e.psi[index];e.psi[index]=0;
    e.stats.active_faults++;e.stats.total_faults++;
    return 0;
}
int jx_route(void* p, uint32_t from,uint32_t to,uint32_t* out_path,uint32_t capacity) {
    if(!p || !out_path || capacity==0)return -2;
    Engine& e=*static_cast<Engine*>(p);
    if(from>=e.n || to>=e.n || e.failed[from] || e.failed[to]){e.stats.routes_failed++;return -2;}
    const uint32_t INF=UINT32_MAX;
    std::vector<uint32_t> g(e.n, INF), parent(e.n,INF);
    using Item=std::tuple<uint32_t,uint32_t,uint32_t>; // f, g, node
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> pq;
    g[from]=0;pq.emplace(e.manhattan(from,to),0,from);
    uint32_t expanded=0;
    while(!pq.empty()) {
        auto [f,score,u]=pq.top();pq.pop();
        if(score!=g[u])continue;
        expanded++;
        if(u==to)break;
        const uint32_t x=u%e.d,y=(u/e.d)%e.d,z=u/(e.d*e.d);
        auto relax=[&](uint32_t v) {
            if(!e.failed[v] && score+1<g[v]) {
                g[v]=score+1;parent[v]=u;
                pq.emplace(g[v]+e.manhattan(v,to),g[v],v);
            }
        };
        if(x>0)relax(u-1);
        if(x+1<e.d)relax(u+1);
        if(y>0)relax(u-e.d);
        if(y+1<e.d)relax(u+e.d);
        if(z>0)relax(u-e.d*e.d);
        if(z+1<e.d)relax(u+e.d*e.d);
    }
    e.stats.last_route_expansions=expanded;
    if(g[to]==INF) {e.stats.routes_failed++;return -1;}
    uint32_t length=g[to]+1;
    if(length>capacity)return -3;
    uint32_t u=to;
    for(uint32_t i=length;i>0;i--) {out_path[i-1]=u;if(i>1)u=parent[u];}
    e.stats.routes_found++;
    return static_cast<int>(length);
}
int jx_copy_field(void* p,uint8_t* out_values,uint8_t* out_faults,uint32_t capacity) {
    if(!p||!out_values||!out_faults)return -1;
    Engine& e=*static_cast<Engine*>(p);
    if(capacity<e.n)return -2;
    auto v=e.bytes();
    std::memcpy(out_values,v.data(),e.n);
    std::memcpy(out_faults,e.failed.data(),e.n);
    return int(e.n);
}
int jx_set_field(void* p,const uint8_t* values,uint32_t length) {
    if(!p||!values)return -1;
    Engine& e=*static_cast<Engine*>(p);
    if(length!=e.n)return -2;
    for(uint32_t i=0;i<e.n;i++) {
        e.psi[i]=e.saved[i]=e.next[i]=e.source[i]=float(values[i])/255.0f;
        e.failed[i]=0;e.age[i]=0;
    }
    e.stats.active_faults=0;
    e.stats.ann_last_mse=e.ann.mse(e.tiled());
    return 0;
}
int jx_stats(void* p,JXStats* out) {
    if(!p||!out)return -1;
    *out=static_cast<Engine*>(p)->stats;return 0;
}
uint32_t jx_codec_bound(uint32_t length) {
    if(length>100000000)return 0;
    return 12+length+(length+7)/8; // 12-byte header + one tag per tile
}
int jx_encode(const uint8_t* data,uint32_t length,uint8_t* out,uint32_t capacity) {
    if((!data && length)||!out||length>100000000)return -1;
    std::vector<uint8_t> v;
    v.reserve(jx_codec_bound(length));
    v.insert(v.end(),{'J','X','3','D'});
    put_u32(v,length);
    put_u32(v,crc32(data,length));
    for(uint32_t at=0;at<length;at+=8) {
        const uint32_t count=std::min(uint32_t(8),length-at);
        bool uniform=true;
        for(uint32_t k=1;k<count;k++) if(data[at+k]!=data[at]) {uniform=false;break;}
        if(uniform) {v.push_back(1);v.push_back(data[at]);continue;}
        uint32_t best_count=0;uint8_t base=0;
        for(uint32_t k=0;k<count;k++) {
            uint32_t occurrences=0;
            for(uint32_t j=0;j<count;j++) if(data[at+j]==data[at+k]) occurrences++;
            if(occurrences>best_count){best_count=occurrences;base=data[at+k];}
        }
        uint8_t mask=0;
        for(uint32_t k=0;k<count;k++)if(data[at+k]!=base)mask|=uint8_t(1u<<k);
        const uint32_t n_res=count-best_count;
        if(3+n_res<1+count) {
            v.push_back(2);v.push_back(base);v.push_back(mask);
            for(uint32_t k=0;k<count;k++)if(mask&(1u<<k))v.push_back(uint8_t(data[at+k]^base));
        } else {
            v.push_back(0);
            v.insert(v.end(),data+at,data+at+count);
        }
    }
    if(v.size()>capacity)return -2;
    std::memcpy(out,v.data(),v.size());
    return int(v.size());
}
int jx_decode(const uint8_t* encoded,uint32_t length,uint8_t* out,uint32_t capacity) {
    if(!encoded||length<12||std::memcmp(encoded,"JX3D",4)!=0)return -1;
    uint32_t raw=get_u32(encoded+4);
    if(raw>100000000||raw>capacity||(!out&&raw))return -2;
    std::vector<uint8_t> tmp;
    tmp.reserve(raw);
    uint32_t at=12;
    while(tmp.size()<raw) {
        if(at>=length)return -3;
        uint8_t tag=encoded[at++];
        uint32_t count=std::min(uint32_t(8),raw-uint32_t(tmp.size()));
        if(tag==0) {
            if(length-at<count)return -3;
            tmp.insert(tmp.end(),encoded+at,encoded+at+count);at+=count;
        } else if(tag==1) {
            if(at>=length)return -3;
            tmp.insert(tmp.end(),count,encoded[at++]);
        } else if(tag==2) {
            if(length-at<2)return -3;
            const uint8_t base=encoded[at++],mask=encoded[at++];
            if(count<8 && (mask>>count))return -3;
            for(uint32_t k=0;k<count;k++) {
                if(mask&(1u<<k)) {
                    if(at>=length)return -3;
                    const uint8_t residual=encoded[at++];
                    if(!residual)return -3;
                    tmp.push_back(base^residual);
                } else tmp.push_back(base);
            }
        } else return -3;
    }
    if(at!=length || crc32(tmp.data(),raw)!=get_u32(encoded+8)) return -4;
    std::memcpy(out,tmp.data(),raw);
    return int(raw);
}
int jx_encode_field(void* p,uint8_t* out,uint32_t capacity) {
    if(!p)return -1;
    Engine& e=*static_cast<Engine*>(p);
    auto bytes=e.tiled();
    int n=jx_encode(bytes.data(),e.n,out,capacity);
    if(n>0) {e.stats.last_encoded_bytes=uint32_t(n);e.stats.last_codec_ratio=double(n)/e.n;}
    return n;
}
int jx_import_field(void* p,const uint8_t* encoded,uint32_t length) {
    if(!p||!encoded)return -1;
    Engine& e=*static_cast<Engine*>(p);
    std::vector<uint8_t> packed(e.n);
    int bytes=jx_decode(encoded,length,packed.data(),e.n);
    if(bytes!=int(e.n))return -2;
    auto raster=e.untile(packed);
    return jx_set_field(p,raster.data(),e.n);
}
int jx_verify_field(void* p) {
    if(!p)return -1;
    Engine& e=*static_cast<Engine*>(p);
    auto original=e.tiled();
    std::vector<uint8_t> encoded(jx_codec_bound(e.n)),reconstructed(e.n);
    int n=jx_encode(original.data(),e.n,encoded.data(),encoded.size());
    if(n<0)return -2;
    int m=jx_decode(encoded.data(),uint32_t(n),reconstructed.data(),e.n);
    bool ok=m==int(e.n) && original==reconstructed;
    if(ok)ok=(e.untile(reconstructed)==e.bytes());
    e.stats.last_codec_ratio=double(n)/e.n;
    e.stats.last_encoded_bytes=uint32_t(n);
    e.stats.codec_verified=ok?1:0;
    return ok?0:-3;
}

uint32_t jx_bitplane_bound(uint32_t voxels) {
    if(voxels>100000000u)return 0;
    return 8*((voxels+7)/8);
}
int jx_bitplanes(void* p, uint8_t* out, uint32_t capacity) {
    if(!p||!out)return -1;
    Engine& e=*static_cast<Engine*>(p);
    const uint32_t stride=(e.n+7)/8, size=8*stride;
    if(capacity<size)return -2;
    std::memset(out,0,size);
    const auto q=e.bytes();
    for(uint32_t i=0;i<e.n;i++)for(uint32_t bit=0;bit<8;bit++)
        if((q[i]>>bit)&1u)out[bit*stride+i/8]|=uint8_t(1u<<(i%8));
    return int(size);
}
int jx_ann_train(void* p,uint32_t epochs,double rate) {
    if(!p||epochs<1||epochs>200||!std::isfinite(rate)||rate<=0||rate>10)return -1;
    Engine& e=*static_cast<Engine*>(p);
    const auto spatial=e.tiled();
    int improvements=0;
    for(uint32_t k=0;k<epochs;k++) {
        if(e.ann.train_epoch(spatial,rate)) improvements++;
        e.stats.ann_epochs++;
    }
    e.stats.ann_last_mse=e.ann.mse(spatial);
    return improvements;
}
int jx_ann_preview(void* p,uint8_t* out,uint32_t capacity) {
    if(!p||!out)return -1;
    Engine& e=*static_cast<Engine*>(p);
    if(capacity<e.n)return -2;
    auto tiles=e.tiled();
    std::vector<uint8_t> pred(e.n);
    uint8_t latent[4];
    for(uint32_t i=0;i<e.n;i+=8){
        e.ann.encode_latent(tiles.data()+i,latent);
        e.ann.decode_latent(latent,pred.data()+i);
    }
    const auto raster=e.untile(pred);
    std::memcpy(out,raster.data(),e.n);
    return int(e.n);
}
int jx_ann_latents(void* p,float* out,uint32_t capacity) {
    if(!p||!out)return -1;
    Engine& e=*static_cast<Engine*>(p);
    const uint32_t count=e.n/2;
    if(capacity<count)return -2;
    const auto tiled=e.tiled();
    uint8_t predicted[8];
    for(uint32_t i=0;i<e.n;i+=8)
        e.ann.predict(tiled.data()+i,predicted,out+(i/8)*4);
    return int(count);
}
uint32_t jx_ann_bound(uint32_t voxels) {
    const uint32_t b=jx_codec_bound(voxels);
    if(b==0||voxels>100000000u-400u)return 0;
    if(b>UINT32_MAX/2-400)return 0;
    return 24+4*jx::TileANN::param_count+b+jx_codec_bound(voxels/2);
}
int jx_ann_export(void* p,uint8_t* out,uint32_t capacity) {
    if(!p||!out)return -1;
    Engine& e=*static_cast<Engine*>(p);
    const auto original=e.tiled();
    std::vector<uint8_t> residual(e.n),prediction(e.n), qlatent(e.n/2);
    for(uint32_t i=0;i<e.n;i+=8) {
        e.ann.encode_latent(original.data()+i,qlatent.data()+(i/8)*4);
        e.ann.decode_latent(qlatent.data()+(i/8)*4,prediction.data()+i);
    }
    for(uint32_t i=0;i<e.n;i++)residual[i]=original[i]^prediction[i];
    std::vector<uint8_t> packed(jx_codec_bound(e.n)), packed_pred(jx_codec_bound(e.n/2));
    int size=jx_encode(residual.data(),e.n,packed.data(),packed.size());
    int pred_size=jx_encode(qlatent.data(),e.n/2,packed_pred.data(),packed_pred.size());
    if(size<0||pred_size<0)return -2;
    std::vector<uint8_t> encoded;encoded.reserve(24+4*jx::TileANN::param_count+size+pred_size);
    encoded.insert(encoded.end(),{'J','X','A','1'});
    put_u32(encoded,e.n);put_u32(encoded,e.d);put_u32(encoded,crc32(original.data(),e.n));
    put_u32(encoded,jx::TileANN::param_count);
    for(float w:e.ann.w)put_u32(encoded,std::bit_cast<uint32_t>(w));
    put_u32(encoded,uint32_t(pred_size));
    encoded.insert(encoded.end(),packed_pred.begin(),packed_pred.begin()+pred_size);
    encoded.insert(encoded.end(),packed.begin(),packed.begin()+size);
    if(encoded.size()>capacity)return -3;
    std::memcpy(out,encoded.data(),encoded.size());
    e.stats.last_ann_encoded_bytes=uint32_t(encoded.size());
    e.stats.last_ann_ratio=double(encoded.size())/e.n;
    return int(encoded.size());
}
int jx_ann_import(void* p,const uint8_t* encoded,uint32_t length) {
    if(!p||!encoded||length<24+4*jx::TileANN::param_count+24)return -1;
    Engine& e=*static_cast<Engine*>(p);
    if(std::memcmp(encoded,"JXA1",4)!=0 || get_u32(encoded+4)!=e.n
        || get_u32(encoded+8)!=e.d || get_u32(encoded+16)!=jx::TileANN::param_count)return -2;
    jx::TileANN model;
    for(uint32_t k=0;k<jx::TileANN::param_count;k++) {
        float w=std::bit_cast<float>(get_u32(encoded+20+4*k));
        if(!std::isfinite(w)||std::abs(w)>20.0f)return -3;
        model.w[k]=w;
    }
    const uint32_t offset=24+4*jx::TileANN::param_count;
    const uint32_t pred_size=get_u32(encoded+20+4*jx::TileANN::param_count);
    if(pred_size<12||pred_size>jx_codec_bound(e.n/2)||offset>length||pred_size>length-offset)return -4;
    std::vector<uint8_t> residual(e.n),prediction(e.n),original(e.n),qlatent(e.n/2);
    if(jx_decode(encoded+offset,pred_size,qlatent.data(),e.n/2)!=int(e.n/2))return -4;
    if(jx_decode(encoded+offset+pred_size,length-offset-pred_size,residual.data(),e.n)!=int(e.n))return -4;
    for(uint32_t i=0;i<e.n;i+=8)
        model.decode_latent(qlatent.data()+(i/8)*4,prediction.data()+i);
    for(uint32_t i=0;i<e.n;i++)original[i]=residual[i]^prediction[i];
    if(crc32(original.data(),e.n)!=get_u32(encoded+12))return -5;
    const auto raster=e.untile(original);
    int ret=jx_set_field(p,raster.data(),e.n);
    if(ret)return -6;
    e.ann=model;
    e.stats.ann_initial_mse=e.stats.ann_last_mse=model.mse(original);
    e.stats.ann_epochs=0;
    e.stats.last_ann_ratio=double(length)/e.n;
    e.stats.last_ann_encoded_bytes=length;
    return 0;
}
}
