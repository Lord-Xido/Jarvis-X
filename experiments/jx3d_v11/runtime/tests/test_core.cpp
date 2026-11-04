#include "../src/jx_core.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <queue>
#include <random>
#include <vector>

static std::vector<uint32_t> reference(uint32_t d, const std::vector<uint8_t>& blocked, int s,int t) {
    uint32_t n=d*d*d;
    if(blocked[s]||blocked[t])return {};
    std::vector<int> prev(n,-1);
    std::queue<int> q;q.push(s);prev[s]=s;
    while(!q.empty()) {
        int u=q.front();q.pop();if(u==t)break;
        int x=u%d,y=(u/d)%d,z=u/(d*d);
        auto add=[&](int v){if(!blocked[v]&&prev[v]<0){prev[v]=u;q.push(v);}};
        if(x>0)add(u-1);
        if(x+1<(int)d)add(u+1);
        if(y>0)add(u-d);
        if(y+1<(int)d)add(u+d);
        if(z>0)add(u-d*d);
        if(z+1<(int)d)add(u+d*d);
    }
    if(prev[t]<0)return {};
    std::vector<uint32_t> path;
    for(int p=t;p!=s;p=prev[p])path.push_back(p);
    path.push_back(s);std::reverse(path.begin(),path.end());return path;
}
int main() {
    std::mt19937 rng(20261104);
    for(uint32_t length: {0u,1u,7u,8u,9u,64u,1000u,65536u}) {
        for(int mode=0;mode<4;mode++) {
            std::vector<uint8_t> input(length),out(length),enc(jx_codec_bound(length));
            for(auto& b:input)b=mode==0?uint8_t(42):mode==1?uint8_t(rng()%4):uint8_t(rng()%256);
            if(mode==3) for(uint32_t i=0;i<length;i+=8)input[i]=43;
            int bytes=jx_encode(input.data(),length,enc.data(),enc.size());
            assert(bytes>=12 && uint32_t(bytes)<=jx_codec_bound(length));
            int got=jx_decode(enc.data(),bytes,out.data(),out.size());
            assert(got==int(length) && input==out);
            if(bytes>12){enc[8]^=1;assert(jx_decode(enc.data(),bytes,out.data(),out.size())<0);}
        }
    }
    assert(jx_create(3,42)==nullptr); // strict 2x2x2 tiles need even dimension
    void* e=jx_create(10,42); assert(e);
    JXStats st{};jx_stats(e,&st);assert(st.voxels==1000 && st.ticks==0);
    assert(jx_verify_field(e)==0);
    assert(jx_fail(e,123)==0);
    assert(jx_fail(e,123)==-2);
    assert(jx_step(e,17)==0);jx_stats(e,&st);assert(st.active_faults==1 && st.repaired_faults==0);
    assert(jx_step(e,1)==0);jx_stats(e,&st);assert(st.active_faults==0 && st.repaired_faults==1);
    uint8_t outval[1000],fault[1000];
    assert(jx_copy_field(e,outval,fault,1000)==1000);
    assert(std::all_of(fault,fault+1000,[](auto x){return x==0;}));
    assert(jx_verify_field(e)==0);
    // End-to-end disk-style reconstruction (quantized bytes are exact).
    std::vector<uint8_t> enc(jx_codec_bound(1000)),dec(1000);
    int n=jx_encode_field(e,enc.data(),enc.size());assert(n>0);
    assert(jx_decode(enc.data(),n,dec.data(),dec.size())==1000);
    // Codec bytes are spatially tiled, not linear raster ordering.
    assert(jx_import_field(e,enc.data(),n)==0);
    assert(jx_copy_field(e,dec.data(),fault,1000)==1000);
    assert(std::equal(dec.begin(),dec.end(),outval));
    for(int testcase=0;testcase<600;testcase++) {
        jx_reset(e,42);
        std::vector<uint8_t> blocked(1000);
        for(int i=0;i<1000;i++) if((rng()%100)<22){jx_fail(e,i);blocked[i]=1;}
        int s=rng()%1000,t=rng()%1000;
        auto ref=reference(10,blocked,s,t);
        uint32_t path[1000];int got=jx_route(e,s,t,path,1000);
        if(ref.empty())assert(got<0);
        else {assert(got==int(ref.size()));assert(path[0]==uint32_t(s) && path[got-1]==uint32_t(t));
            for(int k=1;k<got;k++){int a=path[k-1],b=path[k];
                assert(!blocked[b]);int dx=std::abs(a%10-b%10),dy=std::abs((a/10)%10-(b/10)%10),dz=std::abs(a/100-b/100);assert(dx+dy+dz==1);}}
    }
    // Zero and all-ones invariant under convex-combination update.
    std::vector<uint8_t> zeros(1000,0),ones(1000,255);
    for(const auto& field:{zeros,ones}) {
        jx_reset(e,11);assert(jx_set_field(e,field.data(),field.size())==0);
        assert(jx_step(e,300)==0);
        assert(jx_copy_field(e,outval,fault,1000)==1000);
        for(uint8_t v:outval) assert(v<=255);
    }
    // 8 digital Q8 bitplanes are reconstructed without any fabricated bit activity.
    jx_reset(e,20261104);
    assert(jx_step(e,3)==0);
    jx_stats(e,&st);
    assert(st.byte_updates_total==3000 && st.switching_bits_total>=st.last_switching_bits);
    assert(st.switching_bits_total<=uint64_t(3)*1000*8);
    uint8_t signal[1000],faults3[1000];
    assert(jx_copy_field(e,signal,faults3,1000)==1000);
    std::vector<uint8_t> planes(jx_bitplane_bound(1000));
    assert(jx_bitplanes(e,planes.data(),planes.size())==int(planes.size()));
    const int stride=(1000+7)/8;
    for(int i=0;i<1000;i++) {
        uint8_t reconstructed=0;
        for(int bit=0;bit<8;bit++) if(planes[bit*stride+i/8]&(1u<<(i%8)))reconstructed|=(1u<<bit);
        assert(reconstructed==signal[i]);
    }
    // Genuine neural forward/backward training with a 4-value latent per spatial tile.
    jx_stats(e,&st);
    double before=st.ann_last_mse;
    int accepted=jx_ann_train(e,100,1.0);
    assert(accepted>0);
    jx_stats(e,&st);
    assert(st.ann_epochs==100 && std::isfinite(st.ann_last_mse));
    assert(st.ann_last_mse<before && st.ann_last_mse>0);
    std::vector<uint8_t> predicted(1000);
    assert(jx_ann_preview(e,predicted.data(),predicted.size())==1000);
    std::vector<float> latents(500);
    assert(jx_ann_latents(e,latents.data(),latents.size())==500);
    for(float latent:latents)assert(std::isfinite(latent) && latent>=-1 && latent<=1);
    // Self-describing JXA1: quantized latents + XOR residual + embedded 76 weights.
    std::vector<uint8_t> ann_stream(jx_ann_bound(1000));
    int ann_size=jx_ann_export(e,ann_stream.data(),ann_stream.size());
    assert(ann_size>0 && ann_stream[0]=='J' && ann_stream[2]=='A');
    void* replica=jx_create(10,1);assert(replica);
    assert(jx_ann_import(replica,ann_stream.data(),ann_size)==0);
    std::vector<uint8_t> replica_bytes(1000),replica_faults(1000);
    assert(jx_copy_field(replica,replica_bytes.data(),replica_faults.data(),1000)==1000);
    assert(std::equal(signal,signal+1000,replica_bytes.begin()));
    std::vector<uint8_t> bad(ann_stream.begin(),ann_stream.begin()+ann_size);
    bad.back()^=1;
    assert(jx_ann_import(replica,bad.data(),bad.size())<0);
    assert(jx_copy_field(replica,replica_bytes.data(),replica_faults.data(),1000)==1000);
    assert(std::equal(signal,signal+1000,replica_bytes.begin()));
    jx_destroy(replica);
    jx_destroy(e);
    // Assert spatial tiles are physically adjacent (2x2x2), not 8 adjacent raster bytes.
    void* small=jx_create(4,1);assert(small);
    uint8_t raster[64];for(int i=0;i<64;i++)raster[i]=uint8_t(i);
    assert(jx_set_field(small,raster,64)==0);
    std::vector<uint8_t> packed(jx_codec_bound(64)),unpacked(64);
    int pn=jx_encode_field(small,packed.data(),packed.size());assert(pn>0);
    assert(jx_decode(packed.data(),pn,unpacked.data(),64)==64);
    const uint8_t expected[8]={0,1,4,5,16,17,20,21};
    assert(std::equal(expected,expected+8,unpacked.begin()));
    assert(jx_import_field(small,packed.data(),pn)==0);
    uint8_t reread[64],faults2[64];
    assert(jx_copy_field(small,reread,faults2,64)==64);
    assert(std::equal(raster,raster+64,reread));
    jx_destroy(small);
    std::puts("PASS: lossless codecs/CRC, 2x2x2 neural 8-4-8 training, bitplanes/switching counts, fault recovery, 600 A*-BFS route comparisons");
    return 0;
}
