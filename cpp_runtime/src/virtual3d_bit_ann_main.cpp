// Jarvis-X virtual 4GiB x 4GiB x 4GiB bit-addressed 3D ANN codec VM.
// Research reference: sparse virtual addressing, integer ANN predictor, XOR residuals.
// Never allocates the declared 2^96-bit virtual volume.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jarvisx::virtual3d {
constexpr std::uint64_t axis_extent = (std::uint64_t{1} << 32);
constexpr std::uint32_t tile_side = 32;
constexpr std::uint32_t voxels_per_tile = tile_side * tile_side * tile_side;
constexpr std::uint32_t bytes_per_tile = voxels_per_tile / 8;
constexpr std::uint32_t tiles_per_axis = (std::uint32_t{1} << 27);
using Tile = std::array<std::uint8_t, bytes_per_tile>;

bool bit(const Tile& tile, std::uint32_t i) {
    return ((tile[i >> 3] >> (i & 7U)) & 1U) != 0U;
}
void set_bit(Tile& tile, std::uint32_t i, bool value) {
    const std::uint8_t mask = static_cast<std::uint8_t>(1U << (i & 7U));
    if (value) tile[i >> 3] |= mask;
    else tile[i >> 3] &= static_cast<std::uint8_t>(~mask);
}

// Portable 96-bit Morton code: no compiler-specific __int128 required.
struct Morton96 {
    std::array<std::uint32_t, 3> word{}; // little-endian 32-bit limbs
    bool operator==(const Morton96& other) const { return word == other.word; }
    bool operator<(const Morton96& other) const {
        for (int limb = 2; limb >= 0; --limb) {
            if (word[static_cast<std::size_t>(limb)] != other.word[static_cast<std::size_t>(limb)])
                return word[static_cast<std::size_t>(limb)] < other.word[static_cast<std::size_t>(limb)];
        }
        return false;
    }
};
Morton96 morton(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    Morton96 key;
    const std::array<std::uint32_t, 3> axes{{x,y,z}};
    for (std::uint32_t i = 0; i < 32; ++i) {
        for (std::uint32_t d = 0; d < 3; ++d) {
            const std::uint32_t pos = 3U * i + d;
            key.word[pos / 32] |= ((axes[d] >> i) & 1U) << (pos % 32);
        }
    }
    return key;
}
std::array<std::uint32_t,3> unmorton(const Morton96& key) {
    std::array<std::uint32_t,3> axes{};
    for (std::uint32_t i = 0; i < 32; ++i) {
        for (std::uint32_t d = 0; d < 3; ++d) {
            const std::uint32_t pos = 3U * i + d;
            axes[d] |= ((key.word[pos / 32] >> (pos % 32)) & 1U) << i;
        }
    }
    return axes;
}
struct MortonHash {
    std::size_t operator()(const Morton96& key) const {
        std::uint64_t v = 0xcbf29ce484222325ULL;
        for (std::uint32_t w : key.word) {
            v ^= w; v *= 0x100000001b3ULL;
        }
        return static_cast<std::size_t>(v);
    }
};
Morton96 tile_key(std::uint32_t tx, std::uint32_t ty, std::uint32_t tz) {
    if (tx >= tiles_per_axis || ty >= tiles_per_axis || tz >= tiles_per_axis)
        throw std::out_of_range("tile coordinate exceeds 4Gi-position axis");
    return morton(tx,ty,tz);
}

std::uint64_t fnv64(const Tile& t) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (std::uint8_t x : t) { h ^= x; h *= 0x100000001b3ULL; }
    return h; // non-cryptographic corruption detector only
}
void u8(std::ostream& s, std::uint8_t x) { s.put(static_cast<char>(x)); }
void u16(std::ostream& s, std::uint16_t x) { for (int k=0;k<2;++k) u8(s,static_cast<std::uint8_t>(x>>(8*k))); }
void u32(std::ostream& s, std::uint32_t x) { for (int k=0;k<4;++k) u8(s,static_cast<std::uint8_t>(x>>(8*k))); }
void u64(std::ostream& s, std::uint64_t x) { for (int k=0;k<8;++k) u8(s,static_cast<std::uint8_t>(x>>(8*k))); }
std::uint8_t read8(std::istream& s) {
    int x=s.get(); if (x==std::char_traits<char>::eof()) throw std::runtime_error("truncated snapshot");
    return static_cast<std::uint8_t>(x);
}
std::uint16_t read16(std::istream& s) { std::uint32_t x=0; for(int i=0;i<2;++i) x|=static_cast<std::uint32_t>(read8(s)) << (8*i); return static_cast<std::uint16_t>(x); }
std::uint32_t read32(std::istream& s) { std::uint32_t x=0; for(int i=0;i<4;++i) x|=static_cast<std::uint32_t>(read8(s)) << (8*i); return x; }
std::uint64_t read64(std::istream& s) { std::uint64_t x=0; for(int i=0;i<8;++i) x|=static_cast<std::uint64_t>(read8(s)) << (8*i); return x; }

constexpr int features=6, hidden=8, q_scale=256;
using Feature = std::array<int, features>;
Feature causal(const Tile& t, std::uint32_t i) {
    const std::uint32_t x=i%tile_side, y=(i/tile_side)%tile_side, z=i/(tile_side*tile_side);
    return {{x>0?static_cast<int>(bit(t,i-1)):0,
             y>0?static_cast<int>(bit(t,i-tile_side)):0,
             z>0?static_cast<int>(bit(t,i-tile_side*tile_side)):0,
             x>0&&y>0?static_cast<int>(bit(t,i-tile_side-1)):0,
             x>0&&z>0?static_cast<int>(bit(t,i-tile_side*tile_side-1)):0,
             y>0&&z>0?static_cast<int>(bit(t,i-tile_side*tile_side-tile_side)):0}};
}

struct QuantNet {
    std::array<std::int16_t,hidden*features> w1{};
    std::array<std::int16_t,hidden> b1{};
    std::array<std::int16_t,hidden> w2{};
    std::int16_t b2=0;
    bool predict(const Feature& f) const {
        std::int64_t score=static_cast<std::int64_t>(b2)*q_scale;
        for(int h=0;h<hidden;++h) {
            std::int32_t v=b1[static_cast<std::size_t>(h)];
            for(int j=0;j<features;++j) v+=w1[static_cast<std::size_t>(h*features+j)]*f[static_cast<std::size_t>(j)];
            score+=static_cast<std::int64_t>(w2[static_cast<std::size_t>(h)])*std::max(0,v);
        }
        return score>=0;
    }
    void write(std::ostream& s) const {
        for(auto a:w1) u16(s,static_cast<std::uint16_t>(a));
        for(auto a:b1) u16(s,static_cast<std::uint16_t>(a));
        for(auto a:w2) u16(s,static_cast<std::uint16_t>(a));
        u16(s,static_cast<std::uint16_t>(b2));
    }
    static std::int16_t signed_wire(std::istream& s) {
        const std::uint16_t raw=read16(s);
        const std::int32_t value=raw<0x8000U ? static_cast<std::int32_t>(raw)
                                              : static_cast<std::int32_t>(raw)-65536;
        return static_cast<std::int16_t>(value);
    }
    static QuantNet read(std::istream& s) {
        QuantNet n;
        for(auto& a:n.w1) a=signed_wire(s);
        for(auto& a:n.b1) a=signed_wire(s);
        for(auto& a:n.w2) a=signed_wire(s);
        n.b2=signed_wire(s);
        return n;
    }
    std::uint64_t id() const {
        std::ostringstream ss(std::ios::binary);
        write(ss);
        std::uint64_t h=0xcbf29ce484222325ULL;
        for(char c : ss.str()) { h^=static_cast<std::uint8_t>(c); h*=0x100000001b3ULL; }
        return h;
    }
};

std::uint32_t next_random(std::uint32_t& state) {
    state^=state<<13; state^=state>>17; state^=state<<5;
    return state;
}
float sigmoid(float s) {
    if(s>=0.F) {const float t=std::exp(-s);return 1.F/(1.F+t);}
    const float t=std::exp(s);return t/(1.F+t);
}
class Trainer {
    std::array<float,hidden*features> w1_{};
    std::array<float,hidden> b1_{}, w2_{};
    float b2_=0;
    static float clip(float v) { return std::clamp(v,-4.F,4.F); }
    static std::int16_t quantize(float v) {
        const long n=std::lround(clip(v)*static_cast<float>(q_scale));
        return static_cast<std::int16_t>(n);
    }
public:
    explicit Trainer(std::uint32_t seed) {
        if(seed==0) seed=1;
        for(float& v:w1_) v=(static_cast<float>(next_random(seed)%2049U)/2048.F-0.5F);
        for(float& v:b1_) v=0.1F;
        for(float& v:w2_) v=(static_cast<float>(next_random(seed)%2049U)/2048.F-0.5F);
    }
    QuantNet train(const std::vector<Tile>& fixtures, unsigned epochs, float learning_rate=0.02F) {
        if(fixtures.empty() || fixtures.size()>16 || epochs>20 || !std::isfinite(learning_rate) || learning_rate<=0.F || learning_rate>0.1F)
            throw std::invalid_argument("invalid ANN training contract");
        for(unsigned epoch=0;epoch<epochs;++epoch) {
            for(const Tile& tile:fixtures) {
                for(std::uint32_t i=0;i<voxels_per_tile;++i) {
                    const Feature f=causal(tile,i);
                    std::array<float,hidden> pre{}, act{}, old_w2=w2_;
                    float output=b2_;
                    for(int h=0;h<hidden;++h) {
                        float s=b1_[static_cast<std::size_t>(h)];
                        for(int j=0;j<features;++j) s+=w1_[static_cast<std::size_t>(h*features+j)]*static_cast<float>(f[static_cast<std::size_t>(j)]);
                        pre[static_cast<std::size_t>(h)]=s;
                        act[static_cast<std::size_t>(h)]=std::max(0.F,s);
                        output+=w2_[static_cast<std::size_t>(h)]*act[static_cast<std::size_t>(h)];
                    }
                    const float gradient=sigmoid(output)-(bit(tile,i)?1.F:0.F);
                    for(int h=0;h<hidden;++h) {
                        const std::size_t hh=static_cast<std::size_t>(h);
                        const float grad_h=(pre[hh]>0.F)? gradient*old_w2[hh] : 0.F;
                        for(int j=0;j<features;++j) {
                            const std::size_t k=static_cast<std::size_t>(h*features+j);
                            w1_[k]=clip(w1_[k]-learning_rate*grad_h*static_cast<float>(f[static_cast<std::size_t>(j)]));
                        }
                        b1_[hh]=clip(b1_[hh]-learning_rate*grad_h);
                        w2_[hh]=clip(w2_[hh]-learning_rate*gradient*act[hh]);
                    }
                    b2_=clip(b2_-learning_rate*gradient);
                }
            }
        }
        QuantNet net;
        for(std::size_t i=0;i<w1_.size();++i) net.w1[i]=quantize(w1_[i]);
        for(std::size_t i=0;i<b1_.size();++i) net.b1[i]=quantize(b1_[i]);
        for(std::size_t i=0;i<w2_.size();++i) net.w2[i]=quantize(w2_[i]);
        net.b2=quantize(b2_);
        return net;
    }
};

std::vector<std::uint8_t> zero_rle(const Tile& tile) {
    std::vector<std::uint8_t> out;
    for(std::size_t i=0;i<tile.size();) {
        if(tile[i]==0U) {
            std::size_t count=1;
            while(count<128 && i+count<tile.size() && tile[i+count]==0U) ++count;
            out.push_back(static_cast<std::uint8_t>(127U+count));
            i+=count;
        } else {
            std::size_t count=1;
            while(count<128 && i+count<tile.size() && tile[i+count]!=0U) ++count;
            out.push_back(static_cast<std::uint8_t>(count-1));
            out.insert(out.end(),tile.begin()+static_cast<std::ptrdiff_t>(i),tile.begin()+static_cast<std::ptrdiff_t>(i+count));
            i+=count;
        }
    }
    return out;
}
Tile decode_rle(const std::vector<std::uint8_t>& bytes) {
    Tile out{};std::size_t i=0,j=0;
    while(i<bytes.size()) {
        const std::uint8_t token=bytes[i++];
        if(token>=128) {
            const std::size_t count=static_cast<std::size_t>(token)-127U;
            if(j+count>out.size()) throw std::runtime_error("RLE zero-run overflow");
            j+=count;
        } else {
            const std::size_t count=static_cast<std::size_t>(token)+1U;
            if(i+count>bytes.size() || j+count>out.size()) throw std::runtime_error("RLE literal overflow");
            std::copy(bytes.begin()+static_cast<std::ptrdiff_t>(i),bytes.begin()+static_cast<std::ptrdiff_t>(i+count),out.begin()+static_cast<std::ptrdiff_t>(j));
            i+=count;j+=count;
        }
    }
    if(j!=out.size()) throw std::runtime_error("RLE tile underflow");
    return out;
}
Tile residual(const Tile& src,const QuantNet& net) {
    Tile errors{};
    for(std::uint32_t i=0;i<voxels_per_tile;++i)
        set_bit(errors,i,bit(src,i)!=net.predict(causal(src,i)));
    return errors;
}
Tile reconstruct(const Tile& errors,const QuantNet& net) {
    Tile out{};
    for(std::uint32_t i=0;i<voxels_per_tile;++i)
        set_bit(out,i,bit(errors,i)!=net.predict(causal(out,i)));
    return out;
}

// Five exact inward dyadic 3D folds: 32^3 -> 16^3 -> 8^3 -> 4^3 -> 2^3 -> 1^3.
// Each parent keeps one child and seven XOR details; this transform is reversible,
// not by itself a compression algorithm. All 32768 bits are represented.
Tile inward_fold(const Tile& source) {
    std::vector<std::uint8_t> current(voxels_per_tile);
    for(std::uint32_t i=0;i<voxels_per_tile;++i) current[i]=static_cast<std::uint8_t>(bit(source,i));
    std::array<std::vector<std::uint8_t>,5> detail;
    std::uint32_t side=tile_side;
    for(std::size_t level=0;level<detail.size();++level) {
        const std::uint32_t half=side/2U, groups=half*half*half;
        std::vector<std::uint8_t> parent(groups);
        detail[level].resize(static_cast<std::size_t>(groups)*7);
        for(std::uint32_t z=0;z<half;++z) for(std::uint32_t y=0;y<half;++y)
            for(std::uint32_t x=0;x<half;++x) {
                const std::uint32_t g=(z*half+y)*half+x;
                const std::uint32_t anchor=((2U*z)*side+2U*y)*side+2U*x;
                const std::uint8_t root=current[anchor];
                parent[g]=root;
                for(std::uint32_t child=1;child<8;++child) {
                    const std::uint32_t xx=2U*x+(child&1U);
                    const std::uint32_t yy=2U*y+((child>>1U)&1U);
                    const std::uint32_t zz=2U*z+((child>>2U)&1U);
                    const std::uint32_t offset=(zz*side+yy)*side+xx;
                    detail[level][static_cast<std::size_t>(g)*7+child-1]=
                        static_cast<std::uint8_t>(current[offset]^root);
                }
            }
        current=std::move(parent);
        side=half;
    }
    if(current.size()!=1U) throw std::logic_error("inward core size");
    Tile packed{};
    std::uint32_t position=0;
    set_bit(packed,position++,current[0]!=0);
    for(std::size_t level=detail.size();level>0;--level) {
        for(std::uint8_t v:detail[level-1]) set_bit(packed,position++,v!=0U);
    }
    if(position!=voxels_per_tile) throw std::logic_error("inward transform bit conservation");
    return packed;
}
Tile outward_unfold(const Tile& packed) {
    std::vector<std::uint8_t> current{static_cast<std::uint8_t>(bit(packed,0))};
    std::uint32_t position=1, half=1;
    for(int level=4;level>=0;--level) {
        const std::uint32_t side=2U*half;
        std::vector<std::uint8_t> expanded(static_cast<std::size_t>(side)*side*side);
        for(std::uint32_t z=0;z<half;++z) for(std::uint32_t y=0;y<half;++y)
            for(std::uint32_t x=0;x<half;++x) {
                const std::uint32_t g=(z*half+y)*half+x;
                for(std::uint32_t child=0;child<8;++child) {
                    const std::uint32_t xx=2U*x+(child&1U);
                    const std::uint32_t yy=2U*y+((child>>1U)&1U);
                    const std::uint32_t zz=2U*z+((child>>2U)&1U);
                    const std::uint32_t offset=(zz*side+yy)*side+xx;
                    const std::uint8_t difference=child==0?0U:static_cast<std::uint8_t>(bit(packed,position++));
                    expanded[offset]=static_cast<std::uint8_t>(current[g]^difference);
                }
            }
        current=std::move(expanded);
        half=side;
    }
    if(position!=voxels_per_tile || current.size()!=voxels_per_tile)
        throw std::logic_error("outward transform bit conservation");
    Tile restored{};
    for(std::uint32_t i=0;i<voxels_per_tile;++i) set_bit(restored,i,current[i]!=0U);
    return restored;
}

enum class Mode : std::uint8_t { Raw=0, RawRle=1, AnnXorRle=2, InwardRle=3 };
struct Page {
    Mode mode=Mode::Raw;
    std::uint64_t checksum=0, network_id=0;
    std::vector<std::uint8_t> payload;
};
Page encode(const Tile& tile,const QuantNet& net) {
    Page page;
    page.checksum=fnv64(tile);
    page.payload.assign(tile.begin(),tile.end());
    const std::vector<std::uint8_t> raw_rle=zero_rle(tile);
    if(raw_rle.size()<page.payload.size()) {page.mode=Mode::RawRle;page.payload=raw_rle;}
    const std::vector<std::uint8_t> ann_rle=zero_rle(residual(tile,net));
    if(ann_rle.size()<page.payload.size()) {
        page.mode=Mode::AnnXorRle;page.payload=ann_rle;page.network_id=net.id();
    }
    const std::vector<std::uint8_t> inward_rle=zero_rle(inward_fold(tile));
    if(inward_rle.size()<page.payload.size()) {
        page.mode=Mode::InwardRle;page.payload=inward_rle;page.network_id=0;
    }
    return page;
}
Tile decode(const Page& p,const QuantNet& net) {
    Tile out{};
    switch(p.mode) {
    case Mode::Raw:
        if(p.payload.size()!=bytes_per_tile) throw std::runtime_error("raw page length");
        std::copy(p.payload.begin(),p.payload.end(),out.begin());break;
    case Mode::RawRle: out=decode_rle(p.payload);break;
    case Mode::AnnXorRle:
        if(p.network_id!=net.id()) throw std::runtime_error("ANN checkpoint mismatch");
        out=reconstruct(decode_rle(p.payload),net);break;
    case Mode::InwardRle:
        out=outward_unfold(decode_rle(p.payload));break;
    default: throw std::runtime_error("unrecognized codec mode");
    }
    if(fnv64(out)!=p.checksum) throw std::runtime_error("page checksum mismatch");
    return out;
}

class SparseEngine {
    QuantNet network_;
    std::unordered_map<Morton96,Page,MortonHash> pages_;
    static constexpr std::size_t max_pages=256;
public:
    explicit SparseEngine(QuantNet n):network_(std::move(n)) {}
    const QuantNet& network() const {return network_;}
    void put(Morton96 key, const Tile& tile) {
        const auto coords=unmorton(key);
        (void)tile_key(coords[0],coords[1],coords[2]);
        const bool empty=std::all_of(tile.begin(),tile.end(),[](std::uint8_t v){return v==0;});
        if(empty) {pages_.erase(key);return;}
        if(pages_.count(key)==0 && pages_.size()>=max_pages) throw std::length_error("resident page cap exceeded");
        pages_.insert_or_assign(key,encode(tile,network_));
    }
    Tile get(Morton96 key) const {
        const auto coords=unmorton(key);
        (void)tile_key(coords[0],coords[1],coords[2]);
        const auto it=pages_.find(key);
        return it==pages_.end() ? Tile{} : decode(it->second,network_);
    }
    std::size_t pages() const {return pages_.size();}
    std::size_t payload_bytes() const {
        std::size_t size=0;for(const auto& item:pages_) size+=item.second.payload.size();return size;
    }
    std::array<std::size_t,4> mode_counts() const {
        std::array<std::size_t,4> out{};
        for(const auto& item:pages_) ++out[static_cast<std::size_t>(item.second.mode)];
        return out;
    }
    void save(std::ostream& stream) const {
        stream.write("JX4G3D01",8);u32(stream,1);u32(stream,tile_side);
        network_.write(stream);
        u64(stream,network_.id());
        u32(stream,static_cast<std::uint32_t>(pages_.size()));
        std::vector<Morton96> keys;keys.reserve(pages_.size());
        for(const auto& kv:pages_) keys.push_back(kv.first);
        std::sort(keys.begin(),keys.end());
        for(const auto& key:keys) {
            const Page& p=pages_.at(key);
            for(std::uint32_t limb:key.word) u32(stream,limb);
            u8(stream,static_cast<std::uint8_t>(p.mode));
            u64(stream,p.checksum);u64(stream,p.network_id);
            u32(stream,static_cast<std::uint32_t>(p.payload.size()));
            for(std::uint8_t b:p.payload) u8(stream,b);
        }
        if(!stream) throw std::runtime_error("snapshot write failure");
    }
    static SparseEngine load(std::istream& stream) {
        for(char c:std::string("JX4G3D01")) if(read8(stream)!=static_cast<std::uint8_t>(c)) throw std::runtime_error("bad snapshot magic");
        if(read32(stream)!=1 || read32(stream)!=tile_side) throw std::runtime_error("unsupported snapshot layout");
        QuantNet model=QuantNet::read(stream);
        if(read64(stream)!=model.id()) throw std::runtime_error("corrupt checkpoint model");
        SparseEngine engine(model);
        const std::uint32_t n=read32(stream);
        if(n>max_pages) throw std::runtime_error("snapshot page limit exceeded");
        for(std::uint32_t j=0;j<n;++j) {
            Morton96 key;
            for(auto& limb:key.word) limb=read32(stream);
            const auto coords=unmorton(key);
            (void)tile_key(coords[0],coords[1],coords[2]);
            Page p;
            p.mode=static_cast<Mode>(read8(stream));
            p.checksum=read64(stream);p.network_id=read64(stream);
            const std::uint32_t sz=read32(stream);
            if(sz>bytes_per_tile+128U) throw std::runtime_error("page payload cap exceeded");
            p.payload.resize(sz);
            for(auto& b:p.payload) b=read8(stream);
            (void)decode(p,engine.network_); // reject invalid checksums and model refs on load
            if(!engine.pages_.emplace(key,std::move(p)).second) throw std::runtime_error("duplicate snapshot page");
        }
        if(stream.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("trailing snapshot data");
        return engine;
    }
};

Tile make_fixture(std::uint32_t pattern,std::uint32_t seed=137) {
    Tile t{};
    for(std::uint32_t z=0;z<tile_side;++z) for(std::uint32_t y=0;y<tile_side;++y)
        for(std::uint32_t x=0;x<tile_side;++x) {
            const std::uint32_t index=(z*tile_side+y)*tile_side+x;
            bool v=false;
            if(pattern==0) v=(x>7&&x<24&&y>7&&y<24&&z>7&&z<24);
            else if(pattern==1) v=((x+y+z)%7==0);
            else if(pattern==2) v=(x+y+z>46);
            else if(pattern==3) v=((next_random(seed)&1U)!=0U);
            else v=(index%521U==0);
            set_bit(t,index,v);
        }
    return t;
}
std::vector<Tile> training_set() {
    return {make_fixture(0), make_fixture(1), make_fixture(2)};
}

void self_test() {
    if(axis_extent!=4294967296ULL || static_cast<std::uint64_t>(tile_side)*tiles_per_axis!=axis_extent)
        throw std::runtime_error("virtual extent error");
    for(const auto& coords:std::array<std::array<std::uint32_t,3>,5>{{
        {{0,0,0}}, {{1,2,3}}, {{0xffffffffU,0,37}},
        {{0,0xffffffffU,0xffffffffU}}, {{0xffffffffU,0xffffffffU,0xffffffffU}}
    }}) if(unmorton(morton(coords[0],coords[1],coords[2]))!=coords)
        throw std::runtime_error("96-bit Morton roundtrip failure");
    bool rejected=false;
    try {(void)tile_key(tiles_per_axis,0,0);} catch(const std::out_of_range&) {rejected=true;}
    if(!rejected) throw std::runtime_error("out of bounds tile accepted");
    const QuantNet model=Trainer(42).train(training_set(),2);
    SparseEngine engine(model);
    const Morton96 low=tile_key(0,0,0), high=tile_key(tiles_per_axis-1,tiles_per_axis-1,tiles_per_axis-1);
    Tile test0=make_fixture(0), test1=make_fixture(3);
    engine.put(low,test0); engine.put(high,test1);
    if(engine.get(low)!=test0 || engine.get(high)!=test1 || engine.pages()!=2)
        throw std::runtime_error("sparse roundtrip failure");
    if(engine.get(tile_key(1,2,3))!=Tile{}) throw std::runtime_error("unallocated tile not zero");
    for(std::uint32_t i=0;i<64;++i) {
        auto t=engine.get(high);
        const std::uint32_t index=(i*991U)%voxels_per_tile;
        set_bit(t,index,!bit(t,index));
        engine.put(high,t);
        if(engine.get(high)!=t) throw std::runtime_error("iterative decode failure");
    }
    std::stringstream buffer(std::ios::in|std::ios::out|std::ios::binary);
    engine.save(buffer);buffer.seekg(0);
    auto restored=SparseEngine::load(buffer);
    if(restored.get(low)!=test0 || restored.get(high)!=engine.get(high) || restored.network().id()!=model.id())
        throw std::runtime_error("checkpoint roundtrip mismatch");
    SparseEngine clean(model);
    clean.put(low,test0);clean.put(low,Tile{});
    if(clean.pages()!=0 || clean.get(low)!=Tile{}) throw std::runtime_error("zero tile removal failure");
    std::stringstream truncated(std::ios::in|std::ios::out|std::ios::binary);
    truncated.write("JX4G3D01",8);truncated.seekg(0);rejected=false;
    try {(void)SparseEngine::load(truncated);} catch(const std::runtime_error&) {rejected=true;}
    if(!rejected) throw std::runtime_error("truncated snapshot accepted");
    // Test each exact tile codec mode independently, even if the chooser prefers another.
    for(std::uint32_t mode=0;mode<4;++mode) {
        Page p=encode(test0,model);
        if(mode==0) {p.mode=Mode::Raw;p.network_id=0;p.payload.assign(test0.begin(),test0.end());}
        if(mode==1) {p.mode=Mode::RawRle;p.network_id=0;p.payload=zero_rle(test0);}
        if(mode==2) {p.mode=Mode::AnnXorRle;p.network_id=model.id();p.payload=zero_rle(residual(test0,model));}
        if(mode==3) {p.mode=Mode::InwardRle;p.network_id=0;p.payload=zero_rle(inward_fold(test0));}
        if(decode(p,model)!=test0) throw std::runtime_error("codec mode failed");
    }
    for(std::uint32_t pattern=0;pattern<5;++pattern) {
        const Tile src=make_fixture(pattern,137);
        if(outward_unfold(inward_fold(src))!=src)
            throw std::runtime_error("five-layer inward/outward transform failed");
    }
    std::cout<<"virtual3d-bit-ann-self-test=PASS pages="<<restored.pages()
             <<" encoded_payload_bytes="<<restored.payload_bytes()
             <<" checksum=FNV64_noncryptographic\n";
}

std::uint32_t bounded_arg(const std::string& str, std::uint32_t limit) {
    if(str.empty() || str[0]=='-' || str[0]=='+') throw std::invalid_argument("invalid nonnegative integer");
    std::size_t used=0;
    const unsigned long long n=std::stoull(str,&used,10);
    if(used!=str.size()||n>limit) throw std::invalid_argument("numeric argument exceeds configured cap");
    return static_cast<std::uint32_t>(n);
}
} // namespace jarvisx::virtual3d

int main(int argc,char** argv) {
    using namespace jarvisx::virtual3d;
    try {
        std::uint32_t iterations=24, resident=3, epochs=2, seed=42;
        bool run_tests=false;
        std::string save_path, load_path;
        for(int i=1;i<argc;++i) {
            const std::string option=argv[i];
            auto val=[&]() -> std::string { if(i+1==argc) throw std::invalid_argument("missing value: "+option);return argv[++i]; };
            if(option=="--self-test") run_tests=true;
            else if(option=="--iterations") iterations=bounded_arg(val(),10000);
            else if(option=="--active-tiles") resident=bounded_arg(val(),128);
            else if(option=="--training-epochs") epochs=bounded_arg(val(),20);
            else if(option=="--seed") seed=bounded_arg(val(),0xffffffffU);
            else if(option=="--save") save_path=val();
            else if(option=="--load") load_path=val();
            else if(option=="--help"||option=="-h") {
                std::cout<<"Usage: jarvisx-virtual3d-bit-ann [--self-test] [--iterations 0..10000]"
                         <<" [--active-tiles 1..128] [--training-epochs 0..20] [--seed N]"
                         <<" [--save FILE] [--load FILE]\n";
                return 0;
            } else throw std::invalid_argument("unknown option: "+option);
        }
        if(run_tests) {self_test();return 0;}
        if(!load_path.empty()) {
            std::ifstream in(load_path,std::ios::binary);
            if(!in) throw std::runtime_error("cannot read snapshot: "+load_path);
            auto vm=SparseEngine::load(in);
            std::cout<<"snapshot=verified resident_tiles="<<vm.pages()
                     <<" payload_bytes="<<vm.payload_bytes() <<" nn_id="<<vm.network().id()<<'\n';
            return 0;
        }
        if(resident==0) throw std::invalid_argument("active-tiles must be >= 1");
        const auto begin=std::chrono::steady_clock::now();
        const QuantNet model=Trainer(seed).train(training_set(),epochs);
        SparseEngine engine(model);
        std::vector<Morton96> keys;
        keys.reserve(resident);
        for(std::uint32_t i=0;i<resident;++i) {
            Morton96 k=(i==0)?tile_key(0,0,0):
                       (i==1)?tile_key(tiles_per_axis-1,tiles_per_axis-1,tiles_per_axis-1):
                              tile_key(i,i*3U,i*7U);
            keys.push_back(k);
            engine.put(k,make_fixture(i%5U,seed^i));
        }
        std::uint64_t validated_bits=0;
        for(std::uint32_t iter=0;iter<iterations;++iter) {
            const Morton96& key=keys[iter%keys.size()];
            Tile tile=engine.get(key);
            const std::uint32_t index=(iter*2654435761U)>>17;
            const std::uint32_t local=index%voxels_per_tile;
            set_bit(tile,local,!bit(tile,local));
            engine.put(key,tile);
            if(engine.get(key)!=tile) throw std::runtime_error("exact iterative reconstruction failed");
            validated_bits+=voxels_per_tile;
        }
        if(!save_path.empty()) {
            std::ofstream out(save_path,std::ios::binary|std::ios::trunc);
            if(!out) throw std::runtime_error("cannot create snapshot: "+save_path);
            engine.save(out);
        }
        const auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        const auto modes=engine.mode_counts();
        std::cout<<std::setprecision(8)
                 <<"model=virtual-3D-bit-ANN axis_positions=2^32 voxels=2^96 raw_packed_bytes=2^93\n"
                 <<"tile_edge=32 voxels_per_tile="<<voxels_per_tile<<" raw_bytes_per_tile="<<bytes_per_tile<<"\n"
                 <<"trained_epochs="<<epochs<<" nn_id="<<model.id()<<" resident_tiles="<<engine.pages()
                 <<" encoded_payload_bytes="<<engine.payload_bytes()<<"\n"
                 <<"mode_counts_raw="<<modes[0]<<" raw_rle="<<modes[1]<<" ann_xor_rle="<<modes[2]<<" inward_rle="<<modes[3]<<'\n'
                 <<"verified_iterations="<<iterations<<" verified_voxel_bits="<<validated_bits
                 <<" elapsed_seconds="<<elapsed<<" allocation=active_tiles_only\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"Jarvis-X virtual3d bit ANN error: "<<e.what()<<'\n';
        return 1;
    }
}