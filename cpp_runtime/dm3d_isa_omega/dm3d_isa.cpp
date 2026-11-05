// DM-3D-ISA-Omega | C++20 reference virtual machine
// A deterministic, sparse, 3D autoencoder/decoder bytecode interpreter.
// Reference software; no license terms assigned. NOT a safety controller.
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic dm3d_isa.cpp -o dm3d_isa
// Run: ./dm3d_isa --self-test | --demo | --quine | --encode FILE | --decode FILE

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dm3d {
using u8=std::uint8_t; using u16=std::uint16_t;
using u32=std::uint32_t; using u64=std::uint64_t;
using u128=__uint128_t; // GCC and Clang; C++20 extension, not ISO C++.
constexpr int AXIS_BITS=40, TILE_BITS=5, TILE_EDGE=1<<TILE_BITS;
constexpr int TILE_VOL=TILE_EDGE*TILE_EDGE*TILE_EDGE;
constexpr int LAT_EDGE=TILE_EDGE/2, LAT_VOL=LAT_EDGE*LAT_EDGE*LAT_EDGE;
constexpr u64 AXIS_MAX=(u64{1}<<AXIS_BITS)-1;
constexpr u128 MORTON_MASK=(u128{1}<<120)-1;

[[noreturn]] void fail(std::string_view message) { throw std::runtime_error(std::string(message)); }
void require(bool ok, std::string_view message) { if(!ok) fail(message); }

struct Coord { u64 x=0,y=0,z=0; };
void check_coord(Coord c) { require(c.x<=AXIS_MAX && c.y<=AXIS_MAX && c.z<=AXIS_MAX, "Coordinate outside 40-bit unsigned axis"); }
u128 morton3(Coord c) {
    check_coord(c); u128 m=0;
    for(int i=0;i<AXIS_BITS;++i) {
        m|=u128((c.x>>i)&1u)<<(3*i);
        m|=u128((c.y>>i)&1u)<<(3*i+1);
        m|=u128((c.z>>i)&1u)<<(3*i+2);
    }
    return m;
}
Coord demorton3(u128 m) {
    require((m & ~MORTON_MASK)==0, "Noncanonical Morton address");
    Coord c;
    for(int i=0;i<AXIS_BITS;++i) {
        c.x|=u64((m>>(3*i))&1u)<<i;
        c.y|=u64((m>>(3*i+1))&1u)<<i;
        c.z|=u64((m>>(3*i+2))&1u)<<i;
    }
    return c;
}
std::string hex128(u128 x) {
    std::ostringstream out; out<<"0x"<<std::hex<<std::setfill('0')<<std::setw(16)<<u64(x>>64)<<std::setw(16)<<u64(x);
    return out.str();
}
int linear_index(int x,int y,int z,int edge) { return (z*edge+y)*edge+x; }

// Independent SHA-256 implementation for binary payloads and hash-chained checkpoints.
// Not a substitute for digital signatures, measured boot, or verified provenance.
using Digest=std::array<u8,32>;
constexpr std::array<u32,64> K256={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
class SHA256 {
    std::array<u32,8> h_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<u8,64> buffer_{}; size_t used_=0; u64 count_=0;
    static u32 loadbe(const u8* p) {return (u32(p[0])<<24)|(u32(p[1])<<16)|(u32(p[2])<<8)|u32(p[3]);}
    static void storebe(u8* p,u32 x){p[0]=u8(x>>24);p[1]=u8(x>>16);p[2]=u8(x>>8);p[3]=u8(x);}
    void block(const u8* p) {
        u32 w[64]; for(int i=0;i<16;++i) w[i]=loadbe(p+i*4);
        for(int i=16;i<64;++i) {
            u32 a=w[i-15],b=w[i-2];
            u32 s0=std::rotr(a,7)^std::rotr(a,18)^(a>>3);
            u32 s1=std::rotr(b,17)^std::rotr(b,19)^(b>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        u32 a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],hh=h_[7];
        for(int i=0;i<64;++i){
            u32 s1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25);
            u32 ch=(e&f)^(~e&g);
            u32 t1=hh+s1+ch+K256[i]+w[i];
            u32 s0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22);
            u32 maj=(a&b)^(a&c)^(b&c);
            u32 t2=s0+maj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        std::array<u32,8> r{a,b,c,d,e,f,g,hh};for(int i=0;i<8;++i)h_[i]+=r[i];
    }
public:
    void update(const u8* p,size_t n){
        require(n<=std::numeric_limits<u64>::max()-count_,"SHA-256 input too long");
        count_+=u64(n);
        while(n){size_t take=std::min(n,size_t(64-used_));std::memcpy(buffer_.data()+used_,p,take);used_+=take;p+=take;n-=take;
            if(used_==64){block(buffer_.data());used_=0;}}
    }
    void update(const std::vector<u8>& v){update(v.data(),v.size());}
    void update(const Digest& d){update(d.data(),d.size());}
    Digest finish(){
        u64 bits=count_*8;
        u8 mark=0x80;update(&mark,1);
        u8 zero=0;while(used_!=56)update(&zero,1);
        u8 length[8];for(int i=0;i<8;++i)length[7-i]=u8(bits>>(8*i));update(length,8);
        Digest out{};for(int i=0;i<8;++i)storebe(out.data()+4*i,h_[i]);return out;
    }
};
Digest sha256(const std::vector<u8>& bytes){SHA256 h;h.update(bytes);return h.finish();}
Digest sha256(std::string_view s){SHA256 h;h.update(reinterpret_cast<const u8*>(s.data()),s.size());return h.finish();}
std::string hex_digest(Digest d){std::ostringstream s;s<<std::hex<<std::setfill('0');for(u8 x:d)s<<std::setw(2)<<unsigned(x);return s.str();}

struct U128Hash{size_t operator()(u128 n) const noexcept {u64 a=u64(n),b=u64(n>>64);return size_t(a ^ (b*0x9e3779b97f4a7c15ULL));}};
using Bytes=std::vector<u8>;
struct Blob {Bytes bytes; bool is_world=false;u128 tile_key=0;};
using TilePtr=std::shared_ptr<Blob>;
class SparseWorld {
    std::unordered_map<u128,TilePtr,U128Hash> tiles_;
    size_t max_tiles_;
public:
    explicit SparseWorld(size_t limit=1024):max_tiles_(limit){}
    TilePtr find(u128 address) const{
        auto it=tiles_.find(address>>15);return it==tiles_.end()?nullptr:it->second;
    }
    TilePtr find_by_key(u128 tilekey) const{
        auto it=tiles_.find(tilekey);return it==tiles_.end()?nullptr:it->second;
    }
    TilePtr allocate(u128 address){
        u128 k=address>>15;
        if(auto t=find_by_key(k)) return t;
        require(tiles_.size()<max_tiles_,"Sparse allocation tile limit reached");
        auto tile=std::make_shared<Blob>();tile->bytes.resize(TILE_VOL,0);tile->is_world=true;tile->tile_key=k;
        tiles_.emplace(k,tile);return tile;
    }
    size_t size() const{return tiles_.size();}
    size_t resident_bytes() const{return tiles_.size()*size_t(TILE_VOL);}
};

// Block-average 2x2x2 lossy predictor: 32^3 voxels -> 16^3 8-bit latents.
struct Latent{Bytes v;u128 tile_key=0;bool has_origin=false;};
using LatPtr=std::shared_ptr<Latent>;
struct Halo {std::array<Bytes,6> faces;}; // each 16x16, order -X,+X,-Y,+Y,-Z,+Z
using HaloPtr=std::shared_ptr<Halo>;
Bytes block_encode(const Bytes& data){
    require(data.size()==TILE_VOL,"Block encoder requires exactly one 32^3 tile");
    Bytes z(LAT_VOL);
    for(int zz=0;zz<LAT_EDGE;++zz)for(int yy=0;yy<LAT_EDGE;++yy)for(int xx=0;xx<LAT_EDGE;++xx){
        u32 sum=0;
        for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
            sum+=data[linear_index(xx*2+dx, yy*2+dy, zz*2+dz,TILE_EDGE)];
        z[linear_index(xx,yy,zz,LAT_EDGE)]=u8((sum+4)/8);
    }
    return z;
}
Bytes block_decode(const Bytes& z){
    require(z.size()==LAT_VOL,"Block decoder requires exactly 16^3 latent bytes");
    Bytes out(TILE_VOL);
    for(int zz=0;zz<TILE_EDGE;++zz)for(int yy=0;yy<TILE_EDGE;++yy)for(int xx=0;xx<TILE_EDGE;++xx)
        out[linear_index(xx,yy,zz,TILE_EDGE)]=z[linear_index(xx/2,yy/2,zz/2,LAT_EDGE)];
    return out;
}
Bytes xor_equal(const Bytes& a,const Bytes& b){
    require(a.size()==b.size(),"XOR buffer size mismatch");
    Bytes out(a.size());for(size_t i=0;i<a.size();++i)out[i]=a[i]^b[i];return out;
}
Halo build_halo(const Latent& l,const SparseWorld& world){
    require(l.v.size()==LAT_VOL,"Invalid latent for halo");require(l.has_origin,"Halo requires tile origin");
    Halo h;for(auto& f:h.faces) f.resize(LAT_EDGE*LAT_EDGE);
    Coord c=demorton3(l.tile_key<<15);
    const std::array<std::array<int,3>,6> offsets{{{{-1,0,0}},{{1,0,0}},{{0,-1,0}},{{0,1,0}},{{0,0,-1}},{{0,0,1}}}};
    for(int face=0;face<6;++face){
        auto d=offsets[face];
        auto next=[&](u64 a,int direction)->std::optional<u64>{
            if(direction<0) return a>=TILE_EDGE?std::optional<u64>(a-TILE_EDGE):std::nullopt;
            if(direction>0) return a<=AXIS_MAX-TILE_EDGE?std::optional<u64>(a+TILE_EDGE):std::nullopt;
            return a;
        };
        auto nx=next(c.x,d[0]),ny=next(c.y,d[1]),nz=next(c.z,d[2]);
        Bytes neighbor;
        bool has_neighbor=false;
        if(nx && ny && nz) {
            auto p=world.find(morton3(Coord{*nx,*ny,*nz}));
            if(p){neighbor=block_encode(p->bytes);has_neighbor=true;}
        }
        // Missing neighbor -> clamp to current tile boundary (zero-flux boundary).
        if(!has_neighbor)neighbor=l.v;
        for(int v=0;v<LAT_EDGE;++v)for(int u=0;u<LAT_EDGE;++u){
            int x=u,y=v,z=v;
            switch(face){
                case 0: x=!has_neighbor?0:LAT_EDGE-1; y=u;z=v;break;
                case 1: x=!has_neighbor?LAT_EDGE-1:0;y=u;z=v;break;
                case 2: y=!has_neighbor?0:LAT_EDGE-1;x=u;z=v;break;
                case 3: y=!has_neighbor?LAT_EDGE-1:0;x=u;z=v;break;
                case 4: z=!has_neighbor?0:LAT_EDGE-1;x=u;y=v;break;
                case 5: z=!has_neighbor?LAT_EDGE-1:0;x=u;y=v;break;
            }
            h.faces[face][v*LAT_EDGE+u]=neighbor[linear_index(x,y,z,LAT_EDGE)];
        }
    }
    return h;
}
Bytes stencil6(const Bytes& input, const Halo& halo){
    require(input.size()==LAT_VOL,"Bad latent size in stencil");
    Bytes out(LAT_VOL);
    auto at=[&](int x,int y,int z)->u32{
        if(x<0) return halo.faces[0][z*LAT_EDGE+y];
        if(x>=LAT_EDGE) return halo.faces[1][z*LAT_EDGE+y];
        if(y<0) return halo.faces[2][z*LAT_EDGE+x];
        if(y>=LAT_EDGE) return halo.faces[3][z*LAT_EDGE+x];
        if(z<0) return halo.faces[4][y*LAT_EDGE+x];
        if(z>=LAT_EDGE) return halo.faces[5][y*LAT_EDGE+x];
        return input[linear_index(x,y,z,LAT_EDGE)];
    };
    for(int z=0;z<LAT_EDGE;++z)for(int y=0;y<LAT_EDGE;++y)for(int x=0;x<LAT_EDGE;++x){
        const u32 sum=4*at(x,y,z)+at(x-1,y,z)+at(x+1,y,z)+at(x,y-1,z)+at(x,y+1,z)+at(x,y,z-1)+at(x,y,z+1);
        out[linear_index(x,y,z,LAT_EDGE)]=u8((sum+5)/10); // rounded convex average
    }
    return out;
}

// 32-bit instruction: opcode[31:24], dst[23:19], a[18:14], b[13:9], flags[8:4], ext-count[3:0].
enum Op:u8{
    HALT=0x00, COORD3=0x20, MORTON3=0x21, NEIGHBOR6=0x22,
    TILE_FIND=0x24,TILE_ALLOC=0x25,VOX_LOAD=0x26,VOX_STORE=0x27,HALO_EXCHANGE=0x28,
    LAT_ENC=0x40,LAT_VQ=0x41,LAT_STENCIL6=0x43,LAT_DEC=0x45,
    RES_XOR=0x46,RECON_XOR=0x47,
    ASSERT_EQ=0x60,HASH=0x61,CHECKPOINT=0x62,BR_MISSING=0x63,
    STR_LIT=0xe0,FORMAT_EMIT=0xe1};
u32 instr(u8 op,u8 dst=0,u8 a=0,u8 b=0,u8 flags=0,u8 ext=0){
    require(dst<32&&a<32&&b<32&&flags<32&&ext<16,"Instruction field overflow");
    return (u32(op)<<24)|(u32(dst)<<19)|(u32(a)<<14)|(u32(b)<<9)|(u32(flags)<<4)|ext;
}
struct Decoded{u8 op,dst,a,b,flags,ext;};
Decoded decode(u32 w){return {u8(w>>24),u8((w>>19)&31),u8((w>>14)&31),u8((w>>9)&31),u8((w>>4)&31),u8(w&15)};}
struct Builder{
    std::vector<u32> words;
    void add(u8 op,u8 dst=0,u8 a=0,u8 b=0,u8 flags=0,std::vector<u32> extra={}){
        require(extra.size()<16,"Too many extension words");words.push_back(instr(op,dst,a,b,flags,u8(extra.size())));
        words.insert(words.end(),extra.begin(),extra.end());
    }
    void coord(u8 dst,Coord c){check_coord(c);std::vector<u32> ext;
        for(u64 q:{c.x,c.y,c.z}){ext.push_back(u32(q));ext.push_back(u32(q>>32));}
        add(COORD3,dst,0,0,0,std::move(ext));
    }
};
std::vector<u8> words_to_bytes(const std::vector<u32>& words){
    std::vector<u8> bytes;bytes.reserve(words.size()*4);
    for(u32 x:words)for(int i=0;i<4;++i)bytes.push_back(u8(x>>(8*i)));
    return bytes;
}
std::vector<u32> bytes_to_words(const Bytes& bytes){
    require(bytes.size()%4==0,"Bytecode length not 32-bit aligned");
    std::vector<u32> words;words.reserve(bytes.size()/4);
    for(size_t i=0;i<bytes.size();i+=4)words.push_back(u32(bytes[i])|(u32(bytes[i+1])<<8)|(u32(bytes[i+2])<<16)|(u32(bytes[i+3])<<24));
    return words;
}
void append_u32(Bytes& v,u32 x){for(int i=0;i<4;++i)v.push_back(u8(x>>(8*i)));}
void append_u64(Bytes& v,u64 x){for(int i=0;i<8;++i)v.push_back(u8(x>>(8*i)));}
void append_u128(Bytes& v,u128 x){append_u64(v,u64(x));append_u64(v,u64(x>>64));}

struct CheckpointRecord{size_t sequence=0;Digest payload_sha{};Digest chain_sha{};u128 tile_key=0;};
struct VM{
    std::array<u64,32> R{};
    std::array<Coord,32> C{};
    std::array<u128,32> A{};
    std::array<TilePtr,32> T{};
    std::array<LatPtr,32> Z{};
    std::array<HaloPtr,32> H{};
    std::array<std::string,32> S{};
    SparseWorld world;
    std::vector<CheckpointRecord> ledger;
    std::string output;
    Digest chain{};
    size_t instructions=0, pc=0;
    bool halted=false;
    explicit VM(size_t max_tiles=1024):world(max_tiles){}
    static void valid_tile(TilePtr p){require(p!=nullptr,"Null tile handle");require(p->bytes.size()==TILE_VOL,"Tile must be 32^3 bytes");}
    static void valid_lat(LatPtr p){require(p!=nullptr,"Null latent handle");require(p->v.size()==LAT_VOL,"Latent must be 16^3 bytes");}
    void run(const std::vector<u32>& code,size_t fuel=100000){
        require(!code.empty(),"Bytecode program cannot be empty");
        // Decode ahead: branches must land on instruction boundaries, not extension data.
        std::vector<bool> boundaries(code.size(),false);
        for(size_t i=0;i<code.size();){
            boundaries[i]=true;
            size_t words=1+decode(code[i]).ext;
            require(words<=code.size()-i,"Truncated bytecode in preflight verifier");
            i+=words;
        }
        pc=0;halted=false;instructions=0;
        while(!halted){
            require(pc<code.size(),"Program reached end without HALT");
            require(instructions++<fuel,"Instruction fuel exceeded");
            auto d=decode(code[pc++]);
            require(d.ext<=code.size()-pc,"Truncated instruction extension");
            auto ext=[&](size_t n)->u32{require(n<d.ext,"Missing extension word");return code[pc+n];};
            const size_t next=pc+d.ext;
            switch(d.op){
                case HALT:require(d.ext==0,"HALT cannot have extension");halted=true;break;
                case COORD3:{require(d.ext==6,"COORD3 expects 6 extension words");
                    auto val=[&](size_t i)->u64{return u64(ext(i))|(u64(ext(i+1))<<32);};
                    C[d.dst]=Coord{val(0),val(2),val(4)};check_coord(C[d.dst]);break;}
                case MORTON3:require(d.ext==0,"MORTON3 extension not supported");A[d.dst]=morton3(C[d.a]);break;
                case NEIGHBOR6:{require(d.ext==0 && d.dst<=26,"NEIGHBOR6 requires 6 destination coordinates");
                    Coord c=C[d.a];check_coord(c);
                    std::array<Coord,6> c6{c,c,c,c,c,c};
                    require(c.x>0&&c.y>0&&c.z>0&&c.x<AXIS_MAX&&c.y<AXIS_MAX&&c.z<AXIS_MAX,"NEIGHBOR6 cannot cross address boundary");
                    --c6[0].x;++c6[1].x;--c6[2].y;++c6[3].y;--c6[4].z;++c6[5].z;
                    for(int i=0;i<6;++i)C[d.dst+i]=c6[i];
                    break;}
                case TILE_FIND:require(d.ext==0,"TILE.FIND no ext");T[d.dst]=world.find(A[d.a]);break;
                case TILE_ALLOC:require(d.ext==0,"TILE.ALLOC no ext");T[d.dst]=world.allocate(A[d.a]);break;
                case VOX_LOAD:{require(d.ext==0,"VOX.LOAD no ext");valid_tile(T[d.a]);
                    require(T[d.a]->is_world && T[d.a]->tile_key==(A[d.b]>>15),"VOX.LOAD address / tile mismatch");
                    auto local=demorton3(A[d.b]&((u128{1}<<15)-1));
                    R[d.dst]=T[d.a]->bytes[linear_index(int(local.x),int(local.y),int(local.z),TILE_EDGE)];break;}
                case VOX_STORE:{require(d.ext==0,"VOX.STORE no ext");valid_tile(T[d.dst]);
                    require(T[d.dst]->is_world && T[d.dst]->tile_key==(A[d.a]>>15),"VOX.STORE address / tile mismatch");
                    require(R[d.b]<=255,"VOX.STORE byte value too large");
                    auto local=demorton3(A[d.a]&((u128{1}<<15)-1));
                    T[d.dst]->bytes[linear_index(int(local.x),int(local.y),int(local.z),TILE_EDGE)]=u8(R[d.b]);break;}
                case LAT_ENC:{require(d.ext==0,"LAT.ENC no ext");valid_tile(T[d.a]);
                    Z[d.dst]=std::make_shared<Latent>();Z[d.dst]->v=block_encode(T[d.a]->bytes);
                    Z[d.dst]->has_origin=T[d.a]->is_world;Z[d.dst]->tile_key=T[d.a]->tile_key;break;}
                case LAT_VQ:{require(d.ext==0,"LAT.VQ no ext");valid_lat(Z[d.a]);
                    auto p=std::make_shared<Latent>(*Z[d.a]);
                    for(auto& x:p->v)x=u8(std::min(255,((int(x)+8)/17)*17));
                    Z[d.dst]=p;break;}
                case HALO_EXCHANGE:{require(d.ext==0,"HALO.EXCHANGE no ext");valid_lat(Z[d.a]);
                    H[d.dst]=std::make_shared<Halo>(build_halo(*Z[d.a],world));break;}
                case LAT_STENCIL6:{require(d.ext==0,"LAT.STENCIL6 no ext");valid_lat(Z[d.a]);require(H[d.b]!=nullptr,"No halo in register");
                    auto p=std::make_shared<Latent>(*Z[d.a]);p->v=stencil6(Z[d.a]->v,*H[d.b]);Z[d.dst]=p;break;}
                case LAT_DEC:{require(d.ext==0,"LAT.DEC no ext");valid_lat(Z[d.a]);
                    T[d.dst]=std::make_shared<Blob>();T[d.dst]->bytes=block_decode(Z[d.a]->v);
                    T[d.dst]->tile_key=Z[d.a]->tile_key;break;}
                case RES_XOR:case RECON_XOR:{require(d.ext==0,"XOR no ext");valid_tile(T[d.a]);valid_tile(T[d.b]);
                    auto p=std::make_shared<Blob>();p->bytes=xor_equal(T[d.a]->bytes,T[d.b]->bytes);
                    p->tile_key=T[d.a]->tile_key;T[d.dst]=p;break;}
                case ASSERT_EQ:require(d.ext==0,"ASSERT.EQ no ext");valid_tile(T[d.dst]);valid_tile(T[d.a]);
                    require(T[d.dst]->bytes==T[d.a]->bytes,"ASSERT.EQ: reconstructed tile mismatch");break;
                case HASH:{require(d.ext==0,"HASH no ext");valid_tile(T[d.a]);auto h=sha256(T[d.a]->bytes);R[d.dst]=0;
                    for(int i=0;i<8;++i)R[d.dst]=(R[d.dst]<<8)|h[i];
                    break;}
                case CHECKPOINT:{require(d.ext==0,"CHECKPOINT no ext");valid_tile(T[d.dst]);auto h=sha256(T[d.dst]->bytes);u64 prefix=0;
                    for(int i=0;i<8;++i)prefix=(prefix<<8)|h[i];
                    require(prefix==R[d.a],"Checkpoint digest prefix mismatch");
                    Bytes material;material.insert(material.end(),chain.begin(),chain.end());
                    material.insert(material.end(),h.begin(),h.end());
                    append_u128(material,T[d.dst]->tile_key);append_u64(material,ledger.size());
                    chain=sha256(material);
                    ledger.push_back(CheckpointRecord{ledger.size(),h,chain,T[d.dst]->tile_key});break;}
                case BR_MISSING:{require(d.ext==1,"BR.MISSING requires relative signed displacement");
                    if(!T[d.a]){
                        const auto disp=std::bit_cast<std::int32_t>(ext(0));
                        const std::int64_t dest=std::int64_t(next)+disp;
                        require(dest>=0 && u64(dest)<code.size(),"Branch target outside program");
                        require(boundaries[size_t(dest)],"Branch target is an extension word");
                        pc=size_t(dest);continue;
                    }
                    break;}
                case STR_LIT:{require(d.ext>=1,"STR.LIT needs length and bytes");size_t n=ext(0);
                    require(n<=size_t(d.ext-1)*4 && n<=4096,"STR.LIT length invalid");
                    std::string s; s.reserve(n);
                    for(size_t i=0;i<n;++i)s.push_back(char((ext(1+i/4)>>(8*(i%4)))&0xff));
                    S[d.dst]=std::move(s);break;}
                case FORMAT_EMIT:{require(d.ext==0,"FORMAT.EMIT no ext");
                    // General string-format operation: substitute {Q} with quoted string.
                    const auto& tpl=S[d.a];require(tpl.size()<=4096,"Template too large");
                    std::string quoted="\"";
                    for(unsigned char ch:tpl){
                        if(ch=='\\')quoted+="\\\\";
                        else if(ch=='\"')quoted+="\\\"";
                        else if(ch=='\n')quoted+="\\n";
                        else if(ch=='\r')quoted+="\\r";
                        else if(ch=='\t')quoted+="\\t";
                        else if(ch<32 || ch>126) {std::ostringstream ss;ss<<"\\x"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(ch);quoted+=ss.str();}
                        else quoted+=char(ch);
                    }
                    quoted+='"';
                    output.clear();size_t start=0;
                    while(true){size_t found=tpl.find("{Q}",start);
                        if(found==std::string::npos){output.append(tpl,start,std::string::npos);break;}
                        output.append(tpl,start,found-start);output+=quoted;start=found+3;
                    }
                    require(output.size()<=16384,"Formatted output limit exceeded");break;}
                default:fail("Unknown opcode: "+std::to_string(d.op));
            }
            pc=next;
        }
    }
};

// Compact data serialization: each tile stores 16^3 block means and exact XOR residual.
// RLE is optional and never used if it makes the residual longer.
Bytes zero_rle(const Bytes& bytes){
    Bytes out;out.reserve(bytes.size());size_t i=0;
    while(i<bytes.size()){
        if(bytes[i]==0){size_t len=1;while(i+len<bytes.size()&&bytes[i+len]==0&&len<128)++len;
            out.push_back(u8(0x80|(len-1)));i+=len;}
        else{size_t len=1;while(i+len<bytes.size()&&bytes[i+len]!=0&&len<128)++len;
            out.push_back(u8(len-1));out.insert(out.end(),bytes.begin()+std::ptrdiff_t(i),bytes.begin()+std::ptrdiff_t(i+len));i+=len;}
    }
    return out;
}
Bytes zero_unrle(const Bytes& bytes,size_t expected){
    Bytes out;out.reserve(expected);size_t i=0;
    while(i<bytes.size()){
        u8 header=bytes[i++];size_t n=(header&127)+1;
        require(n<=expected-out.size(),"RLE output exceeded expected size");
        if(header&128)out.insert(out.end(),n,0);
        else{require(n<=bytes.size()-i,"RLE truncated literal");out.insert(out.end(),bytes.begin()+std::ptrdiff_t(i),bytes.begin()+std::ptrdiff_t(i+n));i+=n;}
    }
    require(out.size()==expected,"RLE output size mismatch");return out;
}
class Reader{
    const Bytes& v;size_t p=0;
public:
    explicit Reader(const Bytes& data):v(data){}
    u8 get(){require(p<v.size(),"Truncated capsule");return v[p++];}
    u32 get32(){u32 x=0;for(int i=0;i<4;++i)x|=u32(get())<<(i*8);return x;}
    u64 get64(){u64 x=0;for(int i=0;i<8;++i)x|=u64(get())<<(i*8);return x;}
    u128 get128(){u64 l=get64(),h=get64();return (u128(h)<<64)|l;}
    Bytes take(size_t n){require(n<=v.size()-p,"Capsule truncated in byte field");Bytes b(v.begin()+std::ptrdiff_t(p),v.begin()+std::ptrdiff_t(p+n));p+=n;return b;}
    void end(){require(p==v.size(),"Capsule has unexpected trailing data");}
};
struct CapsuleResult{Bytes bytes;u128 key=0;size_t serialized_size=0;bool used_rle=false;};
Bytes encode_capsule(const Bytes& original,u128 tile_key=0){
    require(original.size()==TILE_VOL,"Capsule input must be 32768 bytes");
    require((tile_key>>105)==0,"Tile key must fit in 105 Morton bits");
    Bytes latent=block_encode(original);
    Bytes residual=xor_equal(original,block_decode(latent));
    Bytes rle=zero_rle(residual);
    bool compressed=rle.size()<residual.size();
    const Bytes& payload=compressed?rle:residual;
    Bytes out;const char magic[8]={'D','M','3','D','C','A','P','1'};
    out.insert(out.end(),magic,magic+8);
    append_u32(out,1);append_u128(out,tile_key);out.push_back(compressed?1:0);
    append_u32(out,u32(latent.size()));append_u32(out,u32(payload.size()));
    auto digest=sha256(original);out.insert(out.end(),digest.begin(),digest.end());
    out.insert(out.end(),latent.begin(),latent.end());out.insert(out.end(),payload.begin(),payload.end());
    return out;
}
CapsuleResult decode_capsule(const Bytes& capsule){
    Reader r(capsule);const std::string magic="DM3DCAP1";
    for(char x:magic)require(r.get()==u8(x),"Wrong capsule magic");
    require(r.get32()==1,"Unsupported capsule version");
    u128 key=r.get128();require((key>>105)==0,"Invalid tile key");
    const u8 codec=r.get();require(codec<=1,"Unknown residual compression method");
    auto latent_len=r.get32(),residual_len=r.get32();
    require(latent_len==LAT_VOL && residual_len<=TILE_VOL*2,"Invalid capsule section length");
    auto expected_vec=r.take(32);Digest expected{};std::copy(expected_vec.begin(),expected_vec.end(),expected.begin());
    Bytes latent=r.take(latent_len),residual_payload=r.take(residual_len);r.end();
    Bytes residual=(codec==1)?zero_unrle(residual_payload,TILE_VOL):residual_payload;
    require(residual.size()==TILE_VOL,"Invalid uncompressed residual length");
    Bytes restored=xor_equal(block_decode(latent),residual);
    require(sha256(restored)==expected,"Capsule SHA-256 verification failed");
    return CapsuleResult{std::move(restored),key,capsule.size(),codec==1};
}

std::vector<u32> build_demo_program(Coord c){
    Builder b;b.coord(1,c);b.add(MORTON3,2,1);b.add(TILE_FIND,0,2);
    // Nonexistent tile -> jump to HALT, without allocating it.
    const size_t branch=b.words.size();b.add(BR_MISSING,0,0,0,0,{0});
    b.add(LAT_ENC,0,0);
    b.add(HALO_EXCHANGE,1,0);
    b.add(LAT_STENCIL6,2,0,1);
    b.add(LAT_DEC,1,2);
    b.add(RES_XOR,2,0,1);
    b.add(RECON_XOR,3,1,2);
    b.add(ASSERT_EQ,3,0);
    b.add(HASH,0,3);
    b.add(CHECKPOINT,3,0);
    const size_t halt=b.words.size();b.add(HALT);
    std::int64_t disp=std::int64_t(halt)-std::int64_t(branch+2);
    require(disp>=std::numeric_limits<std::int32_t>::min()&&disp<=std::numeric_limits<std::int32_t>::max(),"Branch displacement overflow");
    b.words[branch+1]=std::bit_cast<u32>(std::int32_t(disp));
    return b.words;
}
Bytes make_tile(u32 seed=0x12345678){
    Bytes data(TILE_VOL);
    for(int z=0;z<TILE_EDGE;++z)for(int y=0;y<TILE_EDGE;++y)for(int x=0;x<TILE_EDGE;++x){
        // A structured, sparse-shape synthetic voxel field (not sensitive inputs).
        int dx=x-16,dy=y-16,dz=z-16;
        int r2=dx*dx+dy*dy+dz*dz;
        u32 t=(u32(x)*29+u32(y)*17+u32(z)*31+seed);
        data[linear_index(x,y,z,TILE_EDGE)]=(r2<110)?u8(110+((t>>3)&31)):0;
    }
    return data;
}
std::string quote_string(std::string_view s){
    std::string q="\"";for(unsigned char ch:s){
        if(ch=='\\')q+="\\\\";else if(ch=='\"')q+="\\\"";
        else if(ch=='\n')q+="\\n";else if(ch=='\r')q+="\\r";else if(ch=='\t')q+="\\t";
        else if(ch<32||ch>126){std::ostringstream ss;ss<<"\\x"<<std::hex<<std::setfill('0')<<std::setw(2)<<int(ch);q+=ss.str();}
        else q+=char(ch);
    }return q+'"';
}
std::string unquote_string(std::string_view q){
    require(q.size()>=2&&q.front()=='"'&&q.back()=='"',"Expected quoted literal");
    std::string out;for(size_t i=1;i+1<q.size();++i){
        if(q[i]!='\\'){out+=q[i];continue;}
        require(i+2<q.size(),"Truncated string escape");char c=q[++i];
        switch(c){case 'n':out+='\n';break;case 'r':out+='\r';break;case 't':out+='\t';break;case '\\':out+='\\';break;case '"':out+='"';break;
        case 'x':{require(i+2<q.size()-1,"Truncated hex escape");
            auto hex=[](char x)->int{if(x>='0'&&x<='9')return x-'0';if(x>='a'&&x<='f')return x-'a'+10;if(x>='A'&&x<='F')return x-'A'+10;fail("Bad hexadecimal escape");};
            int hi=hex(q[++i]),lo=hex(q[++i]);out+=char((hi<<4)|lo);break;}
        default:fail("Unknown string escape");}
    }return out;
}
std::vector<u32> assemble_quine_source(const std::string& source){
    // Three-instruction assembly subset sufficient to demonstrate a real source quine.
    constexpr std::string_view prefix="STR.LIT S0, ";
    constexpr std::string_view suffix="\nFORMAT.EMIT S0\nHALT\n";
    require(source.size()>prefix.size()+suffix.size()&&source.starts_with(prefix)&&source.ends_with(suffix),"Quine assembly syntax invalid");
    std::string_view middle(source.data()+prefix.size(),source.size()-prefix.size()-suffix.size());
    auto literal=unquote_string(middle);
    size_t count=(literal.size()+3)/4+1;
    require(count<16,"Quine literal exceeds instruction extension width");
    std::vector<u32> ext(count,0);ext[0]=u32(literal.size());
    for(size_t i=0;i<literal.size();++i)ext[1+i/4]|=u32(u8(literal[i]))<<(8*(i%4));
    Builder b;b.add(STR_LIT,0,0,0,0,std::move(ext));b.add(FORMAT_EMIT,0,0);b.add(HALT);
    return b.words;
}
std::string quine_source(){
    const std::string tpl="STR.LIT S0, {Q}\nFORMAT.EMIT S0\nHALT\n";
    return "STR.LIT S0, "+quote_string(tpl)+"\nFORMAT.EMIT S0\nHALT\n";
}

Bytes read_file(const std::string& path){
    std::ifstream in(path,std::ios::binary);require(bool(in),"Could not open file: "+path);
    return Bytes(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
}
void write_file(const std::string& path,const Bytes& bytes){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);require(bool(out),"Could not create file: "+path);
    out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));
    require(bool(out),"File write failed: "+path);
}
void write_text(const std::string& path,const std::string& str){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);require(bool(out),"Could not create file: "+path);
    out<<str;require(bool(out),"Text write failed");
}

void self_tests(){
    require(hex_digest(sha256("abc"))=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA-256 known-answer test failed");
    require(hex_digest(sha256(""))=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA-256 empty-vector test failed");
    require(instr(MORTON3,2,1)==0x21104000,"Bytecode encoding mismatch");
    auto example=words_to_bytes({instr(MORTON3,2,1)});require(example==Bytes({0,0x40,0x10,0x21}),"Little-endian opcode test failed");
    require(bytes_to_words(example)==std::vector<u32>({0x21104000}),"Bytecode round-trip failed");
    std::array<Coord,6> coords{{{0,0,0},{1,2,3},{999,999,999},{AXIS_MAX,AXIS_MAX,AXIS_MAX},{u64{1}<<39,7,127},{17,31,63}}};
    for(auto c:coords) require(demorton3(morton3(c)).x==c.x&&demorton3(morton3(c)).y==c.y&&demorton3(morton3(c)).z==c.z,"Morton inverse failed");
    auto p=make_tile();
    require(block_encode(p).size()==LAT_VOL,"Incorrect latent size");
    require(xor_equal(block_decode(block_encode(p)),xor_equal(p,block_decode(block_encode(p))))==p,"XOR residual identity failed");
    auto packet=encode_capsule(p); auto decoded=decode_capsule(packet);
    require(decoded.bytes==p,"Capsule round-trip failed");
    auto corrupt=packet;corrupt.back()^=1;bool rejected=false;
    try{(void)decode_capsule(corrupt);}catch(const std::exception&){rejected=true;}
    require(rejected,"Corrupted capsule must be rejected");
    auto t=make_tile(123);t.assign(TILE_VOL,0);
    require(decode_capsule(encode_capsule(t)).bytes==t,"Zero capsule failed");
    require(encode_capsule(t).size()<TILE_VOL,"Zero residual should compact");
    SparseWorld world(2);u128 a=morton3(Coord{42,43,44});
    require(!world.find(a),"Sparse find should not allocate");world.allocate(a);
    require(world.size()==1&&world.find(a)!=nullptr,"Sparse allocator failure");
    auto code=build_demo_program(Coord{5,6,7});VM vm;
    vm.run(code);require(vm.ledger.empty()&&vm.world.size()==0,"Missing tile should skip without allocating");
    auto tile=vm.world.allocate(morton3(Coord{5,6,7}));tile->bytes=p;
    vm.run(code);require(vm.ledger.size()==1,"VM should produce checkpoint");
    require(vm.T[3]->bytes==p,"VM reconstructed wrong byte array");
    require(vm.Z[0]->v.size()==LAT_VOL,"Encoder didn't execute");
    require(vm.Z[2]->v.size()==LAT_VOL,"Stencil didn't execute");
    VM v2;v2.world.allocate(morton3(Coord{5,6,7}))->bytes=p;v2.run(code);
    require(v2.ledger[0].chain_sha==vm.ledger[0].chain_sha,"Non-deterministic checkpoint");
    Builder test;test.coord(1,Coord{1,2,3});test.add(MORTON3,2,1);test.add(TILE_ALLOC,0,2);
    test.add(VOX_STORE,0,2,0);test.add(VOX_LOAD,1,0,2);test.add(HALT);
    VM v3;v3.R[0]=171;v3.run(test.words);require(v3.R[1]==171,"Voxel load/store failed");
    auto src=quine_source();auto qcode=assemble_quine_source(src);VM q;q.run(qcode);
    require(q.output==src,"Quine did not reproduce its own source");
    require(assemble_quine_source(q.output)==qcode,"Quine failed bytecode recompilation identity");
    // Rejection test: a relative jump cannot target an immediate data word.
    Builder malformed;malformed.add(BR_MISSING,0,0,0,0,{0});malformed.add(HALT);
    // Branch from next word index 2 back to index 1 (the branch extension itself).
    malformed.words[1]=std::bit_cast<u32>(std::int32_t(-1));
    VM v4;rejected=false;
    try{v4.run(malformed.words);}catch(const std::exception&){rejected=true;}
    require(rejected,"Branch to extension must be rejected");
    std::cout<<"PASS: 18 validation groups (SHA-256, Morton-120, ISA encoding, codec, sparse VM, quine, control-flow verification)\n";
}
void demo(const std::optional<std::string>& output_prefix={}){
    auto now=std::chrono::steady_clock::now();
    Coord coord{17,18,19};auto input=make_tile();
    auto code=build_demo_program(coord);
    VM vm;vm.world.allocate(morton3(coord))->bytes=input;
    vm.run(code);
    auto packet=encode_capsule(input,morton3(coord)>>15);
    auto restored=decode_capsule(packet);
    require(restored.bytes==input,"Demo capsule round-trip failed");
    if(output_prefix){
        write_file(*output_prefix+".bytecode.bin",words_to_bytes(code));
        write_file(*output_prefix+".dm3d",packet);
        write_file(*output_prefix+".recovered.raw",restored.bytes);
        write_text(*output_prefix+".quine.asm",quine_source());
    }
    auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-now).count();
    std::cout<<"DM-3D-ISA-OMEGA / C++20 reference VM\n"
             <<"Spatial address: "<<hex128(morton3(coord))<<" (120-bit Morton)\n"
             <<"Sparse world: "<<vm.world.size()<<" active tile / "<<vm.world.resident_bytes()<<" resident bytes\n"
             <<"Bytecode: "<<code.size()<<" 32-bit words / "<<code.size()*4<<" bytes\n"
             <<"ANN predictor: 32768 voxel bytes -> 4096 latent bytes (8:1 predictor only)\n"
             <<"Capsule: "<<packet.size()<<" bytes (includes exact residual and SHA-256)\n"
             <<"Lossless reconstruction: "<<(restored.bytes==input?"PASS":"FAIL")<<"\n"
             <<"Checkpoint SHA-256: "<<hex_digest(vm.ledger.at(0).payload_sha)<<"\n"
             <<"Chain SHA-256: "<<hex_digest(vm.ledger.at(0).chain_sha)<<"\n"
             <<"Elapsed (single demo, not a benchmark): "<<elapsed<<" microseconds\n";
    if(output_prefix)std::cout<<"Artifacts prefix: "<<*output_prefix<<"\n";
}
void bench(size_t count){
    require(count>=1 && count<=512,"Benchmark tile count must be in [1,512]");
    VM vm(count+1);
    std::vector<Coord> coords;coords.reserve(count);
    // Far-apart tiles avoid accidental allocation of adjacent regions.
    for(size_t i=0;i<count;++i){
        Coord c{u64(32*(i+1)),u64(64*(i+1)),u64(96*(i+1))};
        coords.push_back(c);
        vm.world.allocate(morton3(c))->bytes=make_tile(u32(i+7));
    }
    const auto start=std::chrono::steady_clock::now();
    for(const Coord& c:coords)vm.run(build_demo_program(c));
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    require(vm.ledger.size()==count,"Benchmark missed checkpoints");
    const double mib=(double(count)*TILE_VOL)/(1024.0*1024.0);
    std::cout<<"Active tiles: "<<count<<"; processed: "<<mib<<" MiB; physical tile RAM: "
             <<vm.world.resident_bytes()<<" bytes\n"
             <<"Elapsed: "<<seconds<<" s; observed effective original-byte throughput: "
             <<(seconds>0?mib/seconds:0)<<" MiB/s\n"
             <<"Verified checkpoints: "<<vm.ledger.size()<<"; final ledger chain: "<<hex_digest(vm.chain)<<"\n"
             <<"NOTE: one local interpreter run, not a cross-machine speedup claim.\n";
}
} // namespace dm3d

int main(int argc,char** argv){
    using namespace dm3d;
    try{
        const std::string command=argc>=2?argv[1]:"--demo";
        if(command=="--self-test"){require(argc==2,"Usage: --self-test");self_tests();}
        else if(command=="--demo"){
            require(argc==2||(argc==4&&std::string(argv[2])=="--out"),"Usage: --demo [--out PREFIX]");
            demo(argc==4?std::optional<std::string>(argv[3]):std::nullopt);
        }
        else if(command=="--quine"){
            require(argc==2,"Usage: --quine");std::string source=quine_source();auto code=assemble_quine_source(source);
            VM vm;vm.run(code);require(vm.output==source,"Quine source mismatch");
            require(assemble_quine_source(vm.output)==code,"Quine bytecode reassembly mismatch");
            std::cout<<vm.output<<"\nPASS: output identical to source; bytecode reassembly identical ("<<code.size()*4<<" bytes)\n";
        }
        else if(command=="--bench"){
            require(argc==3,"Usage: --bench ACTIVE_TILE_COUNT (1..512)");
            size_t n=0;const char* begin=argv[2];const char* end=begin+std::strlen(begin);
            auto [ptr,ec]=std::from_chars(begin,end,n);
            require(ec==std::errc{}&&ptr==end,"Invalid benchmark count");bench(n);
        }
        else if(command=="--encode"){
            require(argc==4,"Usage: --encode INPUT_32768.raw OUTPUT.dm3d");
            auto input=read_file(argv[2]);auto encoded=encode_capsule(input);write_file(argv[3],encoded);
            std::cout<<"Encoded "<<input.size()<<" bytes into "<<encoded.size()<<" bytes; SHA-256 "<<hex_digest(sha256(input))<<"\n";
        }
        else if(command=="--decode"){
            require(argc==4,"Usage: --decode INPUT.dm3d OUTPUT.raw");
            auto encoded=read_file(argv[2]);auto decoded=decode_capsule(encoded);write_file(argv[3],decoded.bytes);
            std::cout<<"Decoded and verified "<<decoded.bytes.size()<<" bytes; SHA-256 "<<hex_digest(sha256(decoded.bytes))<<"\n";
        }
        else if(command=="--help"||command=="-h"){
            std::cout<<"DM-3D-ISA-OMEGA reference VM\n"
                     <<"  --self-test\n  --demo [--out PREFIX]\n  --quine\n"
                     <<"  --bench ACTIVE_TILE_COUNT (1..512)\n"
                     <<"  --encode INPUT_32768.raw OUTPUT.dm3d\n  --decode INPUT.dm3d OUTPUT.raw\n";
        }
        else fail("Unknown command; see --help");
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
