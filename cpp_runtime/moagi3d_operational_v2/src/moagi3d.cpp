// Dr Moagi 3D Multimodal Engine v2.0 — independently operational codec.
// Provenance: extends the ideas in the locked 2026-11-04 v1.0 baseline.
// Original v1.0 SHA-256:
// 4aa29bef45e2f0a72a5d41ed386fe3b22faef0aaa23e28ab988f2af08d545bf5
// This is a lossless 3D spatial predictor + residual codec + bounded 3D field.
// It is NOT an artificial neural network, cryptographic encryption, or an IP lock.
// C++17; no third-party dependencies. Format: MOAGI3D2 (little endian).

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#ifdef MOAGI_ENABLE_OPENSSL
#include <openssl/evp.h>
#include <openssl/rand.h>
#endif
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moagi3d {
using Bytes = std::vector<uint8_t>;
constexpr size_t MAX_RAW_BYTES = 64u * 1024u * 1024u;
constexpr uint32_t MAX_DIM = 4096;
constexpr uint32_t HEADER_BYTES = 128;
constexpr char MAGIC[8] = {'M','O','A','G','I','3','D','2'};
constexpr char BASELINE_SHA256[] = "4aa29bef45e2f0a72a5d41ed386fe3b22faef0aaa23e28ab988f2af08d545bf5";

[[noreturn]] void fail(const std::string &s) { throw std::runtime_error(s); }
void require(bool cond, const std::string &s) { if (!cond) fail(s); }

struct Shape {
    uint32_t w=0, h=0, d=0, c=0, modality=0, steps=8;
    size_t voxels() const {
        require(w && h && d && c, "dimensions and channels must be positive");
        require(w<=MAX_DIM && h<=MAX_DIM && d<=MAX_DIM, "dimension exceeds 4096");
        require(c<=16, "channel count exceeds 16");
        require(modality<=4, "modality must be 0:text, 1:image, 2:audio, 3:video, 4:3D");
        require(steps<=64, "diffusion iterations must be between 0 and 64");
        const uint64_t vox = static_cast<uint64_t>(w)*h*d;
        require(vox <= MAX_RAW_BYTES/c, "tensor exceeds 64 MiB cap");
        return static_cast<size_t>(vox);
    }
    size_t bytes() const { return voxels()*c; }
};

void put32(Bytes &out, uint32_t x) {
    for (int k=0;k<4;++k) out.push_back(static_cast<uint8_t>((x>>(8*k))&255u));
}
void put64(Bytes &out, uint64_t x) {
    for (int k=0;k<8;++k) out.push_back(static_cast<uint8_t>((x>>(8*k))&255u));
}
uint32_t get32(const Bytes &b, size_t &p) {
    require(p<=b.size() && b.size()-p>=4,"truncated 32-bit field");
    uint32_t v=0;
    for (int k=0;k<4;++k) v |= uint32_t(b[p++]) << (8*k);
    return v;
}
uint64_t get64(const Bytes &b, size_t &p) {
    require(p<=b.size() && b.size()-p>=8,"truncated 64-bit field");
    uint64_t v=0;
    for (int k=0;k<8;++k) v |= uint64_t(b[p++]) << (8*k);
    return v;
}

// IEEE CRC-32: corruption-detection checksum, NOT an authentication tag.
uint32_t crc32(const uint8_t *p, size_t n) {
    uint32_t crc=0xffffffffu;
    for (size_t i=0;i<n;++i) {
        crc ^= p[i];
        for (int j=0;j<8;++j) crc=(crc>>1) ^ (0xedb88320u & (0u-(crc&1u)));
    }
    return ~crc;
}
uint32_t crc32(const Bytes &b) { return crc32(b.data(),b.size()); }

// The spatial predictor is causal: only already-encoded/reconstructed voxels
// (x-1,y,z), (x,y-1,z), (x,y,z-1), same channel, can influence a byte.
uint8_t prediction(const Bytes &recon, const Shape &s, size_t voxel, uint32_t ch) {
    size_t x=voxel%s.w;
    size_t y=(voxel/s.w)%s.h;
    size_t z=voxel/(static_cast<size_t>(s.w)*s.h);
    unsigned total=0, count=0;
    if(x) { total+=recon[((voxel-1)*s.c)+ch]; ++count; }
    if(y) { total+=recon[((voxel-s.w)*s.c)+ch]; ++count; }
    if(z) { total+=recon[((voxel-static_cast<size_t>(s.w)*s.h)*s.c)+ch]; ++count; }
    return count ? static_cast<uint8_t>(total/count) : 0;
}

Bytes residual_encode(const Bytes &raw, const Shape &s) {
    Bytes e(raw.size());
    for (size_t v=0;v<s.voxels();++v)
        for(uint32_t ch=0;ch<s.c;++ch) {
            const size_t i=v*s.c+ch;
            e[i]=static_cast<uint8_t>(raw[i]^prediction(raw,s,v,ch));
        }
    return e;
}
Bytes residual_decode(const Bytes &e, const Shape &s) {
    require(e.size()==s.bytes(),"residual size mismatch");
    Bytes raw(e.size());
    for (size_t v=0;v<s.voxels();++v)
        for(uint32_t ch=0;ch<s.c;++ch) {
            const size_t i=v*s.c+ch;
            raw[i]=static_cast<uint8_t>(e[i]^prediction(raw,s,v,ch));
        }
    return raw;
}

// RLE grammar: control<128 => (control+1) literal bytes follow;
// control>=128 => (control&127)+1 zero bytes, no following bytes.
Bytes rle_encode(const Bytes &src) {
    Bytes out;
    out.reserve(src.size());
    size_t i=0;
    while(i<src.size()) {
        size_t zr=0;
        while(zr<128 && i+zr<src.size() && src[i+zr]==0) ++zr;
        if (zr>=3) {
            out.push_back(static_cast<uint8_t>(0x80u | uint8_t(zr-1)));
            i+=zr;
            continue;
        }
        const size_t begin=i;
        while(i<src.size() && i-begin<128) {
            size_t look=0;
            while(look<3 && i+look<src.size() && src[i+look]==0) ++look;
            if(look==3) break;
            ++i;
        }
        require(i>begin,"RLE encoder did not advance");
        out.push_back(static_cast<uint8_t>(i-begin-1));
        out.insert(out.end(),src.begin()+static_cast<std::ptrdiff_t>(begin),
                   src.begin()+static_cast<std::ptrdiff_t>(i));
    }
    return out;
}
Bytes rle_decode(const Bytes &packed, size_t expected) {
    require(expected<=MAX_RAW_BYTES,"decoded data exceeds cap");
    Bytes out;
    out.reserve(expected);
    for(size_t i=0;i<packed.size();) {
        uint8_t cmd=packed[i++];
        const size_t n=static_cast<size_t>(cmd&127u)+1;
        require(n<=expected-out.size(),"RLE output overflow");
        if(cmd&0x80u) out.insert(out.end(),n,0);
        else {
            require(n<=packed.size()-i,"truncated literal RLE sequence");
            out.insert(out.end(),packed.begin()+static_cast<std::ptrdiff_t>(i),
                       packed.begin()+static_cast<std::ptrdiff_t>(i+n));
            i+=n;
        }
    }
    require(out.size()==expected,"RLE output too short");
    return out;
}

// Integer Q0.16 spatial field. No floating-point reproducibility assumptions.
// U is the mean of voxel channels after mapping 0..255 to 0..65535.
// F_{t+1} = floor((35 F_t + 45 mean_6(F_t) + 20 U)/100).
// Out-of-bounds face neighbors have value 0; all 6 faces always counted.
struct Field {
    std::vector<uint32_t> input;
    std::vector<uint32_t> state;
    std::array<uint32_t,16> latent{};
    uint32_t checksum=0;
    uint32_t minimum=0, maximum=0;
    double mean=0;
};
Field compute_field(const Bytes &raw, const Shape &s) {
    const size_t nv=s.voxels();
    require(raw.size()==s.bytes(),"field input size mismatch");
    Field result;
    result.input.resize(nv);
    for(size_t v=0;v<nv;++v) {
        uint64_t sum=0;
        for(uint32_t ch=0;ch<s.c;++ch) sum+=uint32_t(raw[v*s.c+ch])*257u;
        result.input[v]=static_cast<uint32_t>(sum/s.c);
    }
    result.state=result.input;
    std::vector<uint32_t> next(nv);
    const size_t row=s.w;
    const size_t plane=static_cast<size_t>(s.w)*s.h;
    for(uint32_t t=0;t<s.steps;++t) {
        for(size_t v=0;v<nv;++v) {
            const size_t x=v%s.w, y=(v/s.w)%s.h, z=v/plane;
            uint64_t neighbor=0;
            if(x) neighbor+=result.state[v-1];
            if(x+1<s.w) neighbor+=result.state[v+1];
            if(y) neighbor+=result.state[v-row];
            if(y+1<s.h) neighbor+=result.state[v+row];
            if(z) neighbor+=result.state[v-plane];
            if(z+1<s.d) neighbor+=result.state[v+plane];
            uint64_t q=(35ull*result.state[v] + 45ull*(neighbor/6ull) + 20ull*result.input[v])/100ull;
            next[v]=static_cast<uint32_t>(q);
            require(next[v]<=65535u,"fixed-point field out of range");
        }
        result.state.swap(next);
    }
    uint64_t total=0;
    std::array<uint64_t,16> sums{};
    std::array<uint64_t,16> counts{};
    result.minimum=std::numeric_limits<uint32_t>::max();
    for(size_t v=0;v<nv;++v) {
        uint32_t q=result.state[v];
        total+=q;
        sums[v%16]+=q;
        ++counts[v%16];
        result.minimum=std::min(result.minimum,q);
        result.maximum=std::max(result.maximum,q);
    }
    result.mean=static_cast<double>(total)/nv/65535.0;
    for(size_t k=0;k<16;++k) result.latent[k]=counts[k] ? static_cast<uint32_t>(sums[k]/counts[k]) : 0;
    Bytes packed;
    packed.reserve(nv*4);
    for(auto q:result.state) put32(packed,q);
    result.checksum=crc32(packed);
    return result;
}

struct ArchiveInfo {
    Shape shape;
    uint32_t codec=0;
    uint64_t bytes=0, payload_bytes=0;
    uint32_t original_crc=0, field_crc=0;
    std::array<uint32_t,16> latent{};
};

Bytes make_archive(const Bytes &raw, const Shape &s) {
    require(raw.size()==s.bytes(),"file length does not equal width*height*depth*channels");
    Field f=compute_field(raw,s);
    Bytes residual=residual_encode(raw,s);
    Bytes rle=rle_encode(residual);
    uint32_t codec= rle.size()<residual.size() ? 1u:0u;
    Bytes &payload=codec ? rle:residual;
    Bytes hdr;
    hdr.insert(hdr.end(),MAGIC,MAGIC+8);
    put32(hdr,s.w);put32(hdr,s.h);put32(hdr,s.d);put32(hdr,s.c);
    put32(hdr,s.modality);put32(hdr,s.steps);put32(hdr,codec);
    put64(hdr,raw.size());put64(hdr,payload.size());
    put32(hdr,crc32(raw));put32(hdr,f.checksum);
    for(uint32_t q:f.latent) put32(hdr,q);
    require(hdr.size()==124,"internal format header size error");
    put32(hdr,crc32(hdr));
    require(hdr.size()==HEADER_BYTES,"internal header size error");
    hdr.insert(hdr.end(),payload.begin(),payload.end());
    return hdr;
}

ArchiveInfo parse_header(const Bytes &archive) {
    require(archive.size()>=HEADER_BYTES,"archive shorter than header");
    require(std::equal(MAGIC,MAGIC+8,archive.begin()),"invalid MOAGI3D2 magic");
    size_t pos=8;
    ArchiveInfo a;
    a.shape.w=get32(archive,pos);a.shape.h=get32(archive,pos);
    a.shape.d=get32(archive,pos);a.shape.c=get32(archive,pos);
    a.shape.modality=get32(archive,pos);a.shape.steps=get32(archive,pos);
    a.codec=get32(archive,pos);
    a.bytes=get64(archive,pos);a.payload_bytes=get64(archive,pos);
    a.original_crc=get32(archive,pos);a.field_crc=get32(archive,pos);
    for(uint32_t &q:a.latent) q=get32(archive,pos);
    const uint32_t header_crc=get32(archive,pos);
    require(pos==HEADER_BYTES,"header length invalid");
    require(header_crc==crc32(archive.data(),HEADER_BYTES-4),"header checksum mismatch");
    require(a.codec<=1,"unsupported codec");
    require(a.bytes==a.shape.bytes(),"header tensor size mismatch");
    require(a.payload_bytes<=MAX_RAW_BYTES &&
            a.payload_bytes==archive.size()-HEADER_BYTES,"payload length mismatch");
    return a;
}
Bytes recover(const Bytes &archive, ArchiveInfo *parsed=nullptr) {
    ArchiveInfo a=parse_header(archive);
    Bytes payload(archive.begin()+HEADER_BYTES,archive.end());
    Bytes residual=a.codec ? rle_decode(payload,static_cast<size_t>(a.bytes)) : payload;
    require(residual.size()==a.bytes,"uncompressed residual length invalid");
    Bytes raw=residual_decode(residual,a.shape);
    require(crc32(raw)==a.original_crc,"original byte CRC mismatch (archive corrupt)");
    Field field=compute_field(raw,a.shape);
    require(field.checksum==a.field_crc,"spatial field checksum mismatch");
    require(field.latent==a.latent,"spatial latent metadata mismatch");
    if(parsed) *parsed=a;
    return raw;
}


#ifdef MOAGI_ENABLE_OPENSSL
// Optional AES-256-GCM authenticated encryption, via OpenSSL. This prevents
// unauthorized plaintext reading without the secret key and authenticates the
// encrypted envelope. It is NOT an author's public-key signature.
// Envelope: ASCII MOAGISE2 (8), random 96-bit nonce (12), GCM tag (16), cipher.
constexpr char SEAL_MAGIC[8]={'M','O','A','G','I','S','E','2'};
constexpr size_t SEAL_HEADER_BYTES=36;
struct GcmCtxDel { void operator()(EVP_CIPHER_CTX *p) const { EVP_CIPHER_CTX_free(p); } };
using GcmCtx=std::unique_ptr<EVP_CIPHER_CTX,GcmCtxDel>;
Bytes seal_archive(const Bytes &plaintext,const Bytes &key) {
    require(key.size()==32,"AES-256-GCM key file must contain exactly 32 bytes");
    require(plaintext.size()<=MAX_RAW_BYTES+HEADER_BYTES,"plaintext too large");
    Bytes out;
    out.insert(out.end(),SEAL_MAGIC,SEAL_MAGIC+8);
    uint8_t nonce[12];
    require(RAND_bytes(nonce,12)==1,"CSPRNG nonce generation failed");
    out.insert(out.end(),nonce,nonce+12);
    const size_t tag_offset=out.size();
    out.insert(out.end(),16,0);
    out.resize(SEAL_HEADER_BYTES+plaintext.size());
    GcmCtx ctx(EVP_CIPHER_CTX_new());
    require(bool(ctx),"EVP context allocation failure");
    require(EVP_EncryptInit_ex(ctx.get(),EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1,"GCM init failure");
    require(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_IVLEN,12,nullptr)==1,"GCM nonce length failure");
    require(EVP_EncryptInit_ex(ctx.get(),nullptr,nullptr,key.data(),nonce)==1,"GCM key init failure");
    int n=0;
    require(EVP_EncryptUpdate(ctx.get(),nullptr,&n,out.data(),20)==1,"GCM AAD failure");
    int cipher_n=0;
    require(EVP_EncryptUpdate(ctx.get(),out.data()+SEAL_HEADER_BYTES,&cipher_n,
                              plaintext.data(),static_cast<int>(plaintext.size()))==1,"GCM encryption failure");
    int final_n=0;
    require(EVP_EncryptFinal_ex(ctx.get(),out.data()+SEAL_HEADER_BYTES+cipher_n,&final_n)==1,"GCM final failure");
    require(static_cast<size_t>(cipher_n+final_n)==plaintext.size(),"GCM ciphertext size mismatch");
    require(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_GET_TAG,16,out.data()+tag_offset)==1,"GCM tag error");
    return out;
}
Bytes unseal_archive(const Bytes &sealed,const Bytes &key) {
    require(key.size()==32,"AES-256-GCM key file must contain exactly 32 bytes");
    require(sealed.size()>=SEAL_HEADER_BYTES && sealed.size()<=MAX_RAW_BYTES+HEADER_BYTES+SEAL_HEADER_BYTES,
            "sealed file size invalid");
    require(std::equal(SEAL_MAGIC,SEAL_MAGIC+8,sealed.begin()),"invalid MOAGISE2 sealed magic");
    Bytes out(sealed.size()-SEAL_HEADER_BYTES);
    GcmCtx ctx(EVP_CIPHER_CTX_new());
    require(bool(ctx),"EVP context allocation failure");
    require(EVP_DecryptInit_ex(ctx.get(),EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1,"GCM init failure");
    require(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_IVLEN,12,nullptr)==1,"GCM nonce length failure");
    require(EVP_DecryptInit_ex(ctx.get(),nullptr,nullptr,key.data(),sealed.data()+8)==1,"GCM key init failure");
    int n=0;
    require(EVP_DecryptUpdate(ctx.get(),nullptr,&n,sealed.data(),20)==1,"GCM AAD failure");
    int bytes=0;
    require(EVP_DecryptUpdate(ctx.get(),out.data(),&bytes,sealed.data()+SEAL_HEADER_BYTES,
                              static_cast<int>(out.size()))==1,"GCM decryption failure");
    require(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_TAG,16,
                                 const_cast<uint8_t*>(sealed.data()+20))==1,"GCM tag setup failure");
    int final_bytes=0;
    require(EVP_DecryptFinal_ex(ctx.get(),out.data()+bytes,&final_bytes)==1,
            "AES-GCM authentication FAILED: incorrect key or tampered file");
    require(static_cast<size_t>(bytes+final_bytes)==out.size(),"GCM plaintext size mismatch");
    return out;
}
#endif

Bytes read_file(const std::string &path, size_t limit=MAX_RAW_BYTES+HEADER_BYTES) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f),"cannot read file: "+path);
    const std::streamoff size=f.tellg();
    require(size>=0 && static_cast<uint64_t>(size)<=limit,"file exceeds size cap: "+path);
    Bytes b(static_cast<size_t>(size));
    f.seekg(0,std::ios::beg);
    if(!b.empty()) f.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(b.size()));
    require(bool(f) || b.empty(),"file read failed: "+path);
    return b;
}
void write_file(const std::string &path, const Bytes &b) {
    std::ofstream f(path,std::ios::binary|std::ios::trunc);
    require(bool(f),"cannot create output: "+path);
    if(!b.empty()) f.write(reinterpret_cast<const char*>(b.data()),static_cast<std::streamsize>(b.size()));
    f.flush();
    require(bool(f),"file write failed: "+path);
}
uint32_t as_uint(const std::string &s,const std::string &name) {
    require(!s.empty() && s[0]!='-',"invalid "+name);
    size_t n=0;
    unsigned long long v=0;
    try { v=std::stoull(s,&n,10); } catch(...) { fail("invalid "+name+": "+s); }
    require(n==s.size() && v<=std::numeric_limits<uint32_t>::max(),"invalid "+name+": "+s);
    return static_cast<uint32_t>(v);
}
Shape parse_dims(std::string spec) {
    std::replace(spec.begin(),spec.end(),'X','x');
    std::replace(spec.begin(),spec.end(),'x',' ');
    std::istringstream in(spec);
    std::string a,b,c,extra;
    require(bool(in>>a>>b>>c) && !(in>>extra),"dimensions must be W x H x D (e.g. 16x16x16)");
    Shape s;s.w=as_uint(a,"width");s.h=as_uint(b,"height");s.d=as_uint(c,"depth");
    return s;
}
std::string codec_name(uint32_t id) { return id==1 ? "zero-RLE" : "raw-residual"; }
void print_info(const ArchiveInfo &a, const Field *f=nullptr) {
    const Shape &s=a.shape;
    std::cout << "format=MOAGI3D2\n"
              << "dims=" << s.w << 'x' << s.h << 'x' << s.d << " channels=" << s.c
              << " modality=" << s.modality << " steps=" << s.steps << '\n'
              << "raw_bytes=" << a.bytes << " payload_bytes=" << a.payload_bytes
              << " total_archive_bytes=" << (a.payload_bytes+HEADER_BYTES) << '\n'
              << "codec=" << codec_name(a.codec) << " original_crc32=0x" << std::hex
              << std::setfill('0') << std::setw(8) << a.original_crc
              << " field_crc32=0x" << std::setw(8) << a.field_crc << std::dec << '\n';
    if(f) std::cout << "field_q16_min=" << f->minimum << " max=" << f->maximum
                    << " normalized_mean=" << std::fixed << std::setprecision(6)
                    << f->mean << '\n';
    std::cout << "latent16_q16=[";
    for(size_t i=0;i<a.latent.size();++i) std::cout << (i?",":"") << a.latent[i];
    std::cout << "]\n";
}

Bytes generate_demo(Shape s) {
    Bytes b(s.bytes());
    for(uint32_t z=0;z<s.d;++z)
      for(uint32_t y=0;y<s.h;++y)
        for(uint32_t x=0;x<s.w;++x)
          for(uint32_t c=0;c<s.c;++c) {
            // Volumetric radial shell + gradient; all dimensions contribute.
            const int dx=int(x)-int(s.w/2),dy=int(y)-int(s.h/2),dz=int(z)-int(s.d/2);
            const int r2=dx*dx+dy*dy+dz*dz;
            uint8_t val=static_cast<uint8_t>(r2<48 ? ((17*x+13*y+7*z+23*c)&255) : 0);
            b[(((static_cast<size_t>(z)*s.h+y)*s.w+x)*s.c)+c]=val;
          }
    return b;
}
void write_field_csv(const std::string &path,const Bytes &raw,const Shape &s) {
    const Field f=compute_field(raw,s);
    std::ofstream out(path,std::ios::trunc);
    require(bool(out),"cannot open CSV output: "+path);
    out << "x,y,z,intensity_q16,field_q16\n";
    for(size_t v=0;v<s.voxels();++v) {
        size_t x=v%s.w,y=(v/s.w)%s.h,z=v/(static_cast<size_t>(s.w)*s.h);
        out << x << ',' << y << ',' << z << ',' << f.input[v] << ',' << f.state[v] << '\n';
    }
    out.flush();
    require(bool(out),"CSV output failed");
}
void selftest() {
    size_t cases=0;
    const std::vector<Shape> shapes = {
        {1,1,1,1,0,0},{1,1,1,1,1,8},{1,3,1,1,2,4},{3,1,2,3,3,6},
        {2,2,2,1,4,8},{3,5,4,4,1,12},{16,16,16,1,4,8},
        {23,19,7,2,3,10},{5,3,2,16,0,3}
    };
    std::mt19937 rng(20261104);
    for (const Shape &s:shapes) {
        for(int pattern=0;pattern<5;++pattern) {
            Bytes raw(s.bytes());
            for(size_t j=0;j<raw.size();++j) {
                if(pattern==0) raw[j]=0;
                if(pattern==1) raw[j]=255;
                if(pattern==2) raw[j]=static_cast<uint8_t>(j%256);
                if(pattern==3) raw[j]=static_cast<uint8_t>(rng()&255);
                if(pattern==4) raw[j]=static_cast<uint8_t>((j/7)%2 ? 0:123);
            }
            Bytes archive=make_archive(raw,s);
            ArchiveInfo info;
            Bytes recovered=recover(archive,&info);
            require(raw==recovered,"selftest roundtrip mismatch");
            require(info.bytes==raw.size(),"selftest header mismatch");
            // Corrupt uncompressed RLE/raw bytes and require a rejection.
            if (archive.size()>HEADER_BYTES) {
                Bytes tampered=archive;
                tampered.back()^=0x40u;
                bool rejected=false;
                try { (void)recover(tampered); } catch(const std::exception&) { rejected=true; }
                require(rejected,"selftest failed to reject corruption");
            }
            ++cases;
        }
    }
    // Header metadata tamper must be rejected by the header checksum.
    Shape s{5,4,3,1,4,8};
    Bytes archive=make_archive(generate_demo(s),s);
    archive[10]^=1;
    bool rejected=false;
    try { (void)recover(archive); } catch(const std::exception&) { rejected=true; }
    require(rejected,"selftest accepted header modification");
    std::cout << "SELFTEST PASS: " << cases
              << " lossless 3D roundtrips, " << cases
              << " payload corruption rejections, 1 header corruption rejection\n";
}
void usage() {
    std::cout << "Moagi 3D Operational Engine v2.0 (C++17)\n"
              << "  moagi3d encode INPUT OUTPUT --dims WxHxD [--channels 1..16] [--modality 0..4] [--steps 0..64]\n"
              << "  moagi3d decode ARCHIVE OUTPUT\n"
              << "  moagi3d inspect ARCHIVE\n"
              << "  moagi3d field ARCHIVE OUTPUT.csv\n"
              << "  moagi3d demo PREFIX\n"
#ifdef MOAGI_ENABLE_OPENSSL
              << "  moagi3d seal ARCHIVE OUTPUT.m3ds KEYFILE (AES-256-GCM)\n"
              << "  moagi3d unseal INPUT.m3ds ARCHIVE KEYFILE\n"
#endif
              << "  moagi3d selftest\n"
              << "Archive: 128B header + lossless spatial XOR residual (auto raw / zero-RLE).\n"
              << "3D Q0.16 bounded diffusion is deterministic telemetry, not a trained ANN.\n"
              << "Checksums detect accidental corruption, NOT malicious forgery.\n"
              << "Locked v1.0 SHA256: " << BASELINE_SHA256 << '\n';
}
} // namespace moagi3d

int main(int argc,char **argv) {
    using namespace moagi3d;
    try {
        if (argc<2) { usage();return 0; }
        const std::string command=argv[1];
        if(command=="--help" || command=="help") {usage();return 0;}
        if(command=="selftest") {require(argc==2,"selftest takes no arguments"); selftest();return 0;}
        if(command=="encode") {
            require(argc>=6,"encode: INPUT OUTPUT --dims WxHxD [options]");
            Shape s;
            for(int i=4;i<argc;++i) {
                std::string option=argv[i];
                require(i+1<argc,"missing value for "+option);
                const std::string value=argv[++i];
                if(option=="--dims") {Shape d=parse_dims(value);s.w=d.w;s.h=d.h;s.d=d.d;}
                else if(option=="--channels") s.c=as_uint(value,"channels");
                else if(option=="--modality") s.modality=as_uint(value,"modality");
                else if(option=="--steps") s.steps=as_uint(value,"steps");
                else fail("unknown option "+option);
            }
            if(!s.c) {
                bool channels_supplied=false;
                for(int k=4;k<argc;++k) if(std::string(argv[k])=="--channels") channels_supplied=true;
                require(!channels_supplied,"channels must be in the range 1..16");
                s.c=1;
            }
            Bytes src=read_file(argv[2]);
            Bytes blob=make_archive(src,s);
            write_file(argv[3],blob);
            ArchiveInfo info=parse_header(blob);
            Field f=compute_field(src,s);
            print_info(info,&f);
            std::cout << "encoded " << argv[2] << " -> " << argv[3] << '\n';
            return 0;
        }
        if(command=="decode") {
            require(argc==4,"decode: ARCHIVE OUTPUT");
            Bytes blob=read_file(argv[2]);
            ArchiveInfo info;
            Bytes raw=recover(blob,&info);
            write_file(argv[3],raw);
            print_info(info);
            std::cout << "decoded and CRC-verified " << argv[2] << " -> " << argv[3] << '\n';
            return 0;
        }
        if(command=="inspect") {
            require(argc==3,"inspect: ARCHIVE");
            Bytes blob=read_file(argv[2]);
            print_info(parse_header(blob));
            std::cout << "Header checksums verified; payload NOT decoded or verified.\n";
            return 0;
        }
        if(command=="field") {
            require(argc==4,"field: ARCHIVE OUTPUT.csv");
            Bytes blob=read_file(argv[2]);
            ArchiveInfo a;
            Bytes raw=recover(blob,&a);
            write_field_csv(argv[3],raw,a.shape);
            std::cout << "3D field exported: " << argv[3] << " (" << a.shape.voxels() << " voxels)\n";
            return 0;
        }

#ifdef MOAGI_ENABLE_OPENSSL
        if(command=="seal") {
            require(argc==5,"seal: ARCHIVE OUTPUT.m3ds KEYFILE");
            Bytes archive=read_file(argv[2]);
            (void)recover(archive); // Reject invalid archives before sealing.
            Bytes key=read_file(argv[4],32);
            Bytes sealed=seal_archive(archive,key);
            write_file(argv[3],sealed);
            std::cout << "SEALED: AES-256-GCM authenticated encrypted archive ("
                      << sealed.size() << " bytes); secret key remains external.\n";
            return 0;
        }
        if(command=="unseal") {
            require(argc==5,"unseal: INPUT.m3ds ARCHIVE KEYFILE");
            Bytes sealed=read_file(argv[2],MAX_RAW_BYTES+HEADER_BYTES+SEAL_HEADER_BYTES);
            Bytes key=read_file(argv[4],32);
            Bytes archive=unseal_archive(sealed,key);
            (void)recover(archive); // Require the codec's own checks before writing.
            write_file(argv[3],archive);
            std::cout << "UNSEALED: GCM tag and 3D codec checks verified.\n";
            return 0;
        }
#endif
        if(command=="demo") {
            require(argc==3,"demo: PREFIX");
            const std::string prefix=argv[2];
            const Shape s{16,16,16,1,4,8};
            Bytes raw=generate_demo(s);
            Bytes blob=make_archive(raw,s);
            ArchiveInfo info;
            Bytes out=recover(blob,&info);
            require(out==raw,"demo roundtrip failed");
            write_file(prefix+".raw",raw);
            write_file(prefix+".m3d",blob);
            write_file(prefix+".decoded.raw",out);
            write_field_csv(prefix+".csv",raw,s);
            Field f=compute_field(raw,s);
            print_info(info,&f);
            std::cout << "DEMO PASS: byte-exact 16x16x16 3D roundtrip; outputs " << prefix
                      << "{.raw,.m3d,.decoded.raw,.csv}\n";
            return 0;
        }
        fail("unknown command: "+command);
    } catch(const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
