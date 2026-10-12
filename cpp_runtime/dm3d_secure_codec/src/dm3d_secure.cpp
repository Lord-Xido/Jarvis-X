// Dr Moagi DM3D-SE v1: reversible 3D multiscale byte lifting and authenticated
// encryption of opaque multimedia byte streams. C++17 + OpenSSL + zlib.
// A reference *codec and secure container*, not a trained multimedia ANN.
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/rand.h>
#include <openssl/crypto.h>
#include <zlib.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace fs = std::filesystem;
namespace dm3d {
using Byte = unsigned char;
using Bytes = std::vector<Byte>;
using Key = std::array<Byte, 32>;
constexpr size_t EDGE = 8;
constexpr size_t TILE = EDGE*EDGE*EDGE;
constexpr size_t SALT = 16;
constexpr size_t IVLEN = 12;
constexpr size_t TAGLEN = 16;
constexpr uint64_t MAX_TILES = UINT64_C(0xffffffff);
constexpr char MAGIC[8] = {'D','M','3','D','S','E','0','1'};

[[noreturn]] void fail(const std::string& why) { throw std::runtime_error(why); }
void check(bool yes, const std::string& why) { if(!yes) fail(why); }

void secure_random(Byte* data, size_t length) {
    check(length <= static_cast<size_t>(std::numeric_limits<int>::max()), "random length overflow");
    check(RAND_bytes(data, static_cast<int>(length)) == 1, "system cryptographic RNG failed");
}
void append_u16(Bytes& b, uint16_t v) { b.push_back(static_cast<Byte>(v>>8)); b.push_back(static_cast<Byte>(v)); }
void append_u64(Bytes& b, uint64_t v) { for(int k=7;k>=0;--k) b.push_back(static_cast<Byte>(v>>(8*k))); }
uint16_t u16(const Byte* p) { return (uint16_t(p[0])<<8) | uint16_t(p[1]); }
uint64_t u64(const Byte* p) { uint64_t x=0; for(int i=0;i<8;i++) x=(x<<8)|p[i]; return x; }

void read_exact(std::istream& f, Byte* dst, size_t count) {
    check(count <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()), "read size overflow");
    if(count==0) return;
    f.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(count));
    check(static_cast<size_t>(f.gcount())==count, "truncated or unreadable encrypted container");
}
void write_exact(std::ostream& f, const Byte* src, size_t count) {
    if(count==0) return;
    f.write(reinterpret_cast<const char*>(src), static_cast<std::streamsize>(count));
    check(static_cast<bool>(f), "output write failed");
}

size_t idx(size_t x,size_t y,size_t z) { return z*EDGE*EDGE+y*EDGE+x; }
// Each mapping updates only the 'high' member of disjoint pairs. All operations
// are XOR-involutions. Reversing axis/level ordering yields the exact inverse.
void lift_axis(std::array<Byte,TILE>& x, int axis, size_t stride) {
    for(size_t z=0;z<EDGE;z++) for(size_t y=0;y<EDGE;y++) for(size_t xx=0;xx<EDGE;xx++) {
        const size_t pos[3]={xx,y,z};
        if((pos[axis]&stride)==0) continue;
        size_t prev[3]={xx,y,z}; prev[axis]-=stride;
        x[idx(xx,y,z)] ^= x[idx(prev[0],prev[1],prev[2])];
    }
}
void encode3d(std::array<Byte,TILE>& x) {
    for(size_t level=1;level<EDGE;level*=2)
        for(int axis=0;axis<3;axis++) lift_axis(x,axis,level);
}
void decode3d(std::array<Byte,TILE>& x) {
    for(size_t level=EDGE/2;level>0;level/=2)
        for(int axis=2;axis>=0;axis--) lift_axis(x,axis,level);
}

struct Header {
    Bytes bytes;
    std::array<Byte,SALT> salt{};
    uint64_t length=0;
    uint64_t count=0;
    std::string mime;
};
uint64_t expected_count(uint64_t length) {
    uint64_t t=length/TILE + ((length%TILE)!=0);
    t=std::max<uint64_t>(1,t); // authenticated sentinel even for empty inputs
    check(t<=MAX_TILES, "too many tiles for a single sealed file");
    return t;
}
Header make_header(uint64_t length, const std::string& mime) {
    check(!mime.empty() && mime.size()<=80, "MIME metadata must have 1..80 printable characters");
    for(unsigned char c : mime) check(c>=0x21 && c<=0x7e, "invalid MIME metadata");
    Header h;
    h.length=length; h.count=expected_count(length); h.mime=mime;
    secure_random(h.salt.data(),h.salt.size());
    h.bytes.insert(h.bytes.end(),MAGIC,MAGIC+8);
    h.bytes.push_back(1); // version
    h.bytes.push_back(static_cast<Byte>(EDGE));
    h.bytes.insert(h.bytes.end(),h.salt.begin(),h.salt.end());
    append_u64(h.bytes,h.length);
    append_u64(h.bytes,h.count);
    h.bytes.push_back(static_cast<Byte>(mime.size()));
    h.bytes.insert(h.bytes.end(),mime.begin(),mime.end());
    return h;
}
Header parse_header(std::istream& f) {
    // 8 magic + 1 version + 1 edge + 16 salt + 8 length + 8 count + 1 MIME length
    std::array<Byte,43> head{};
    read_exact(f,head.data(),head.size());
    check(std::equal(MAGIC,MAGIC+8,head.data()), "invalid DM3D magic");
    check(head[8]==1 && head[9]==EDGE,"unsupported format version/tile size");
    const auto mime_len=head[42];
    check(mime_len>=1 && mime_len<=80, "invalid MIME metadata length");
    Header h;
    std::copy_n(head.data()+10,SALT,h.salt.begin());
    h.length=u64(head.data()+26);
    h.count=u64(head.data()+34);
    check(h.count==expected_count(h.length),"invalid declared tile count");
    h.bytes.assign(head.begin(),head.end());
    Bytes mb(mime_len);
    read_exact(f,mb.data(),mb.size());
    for(Byte c:mb) check(c>=0x21 && c<=0x7e, "invalid MIME metadata");
    h.mime.assign(mb.begin(),mb.end());
    h.bytes.insert(h.bytes.end(),mb.begin(),mb.end());
    return h;
}

struct EVPKeyCtxDel { void operator()(EVP_PKEY_CTX* p) const { EVP_PKEY_CTX_free(p); } };
struct EVPCipherCtxDel { void operator()(EVP_CIPHER_CTX* p) const { EVP_CIPHER_CTX_free(p); } };
Key derive_file_key(const Key& master, const std::array<Byte,SALT>& salt) {
    std::unique_ptr<EVP_PKEY_CTX,EVPKeyCtxDel> ctx(EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF,nullptr));
    check(bool(ctx), "HKDF allocation failed");
    static constexpr char INFO[]="DM3D-SE-v1/AES-256-GCM/tile-key";
    check(EVP_PKEY_derive_init(ctx.get())>0 &&
          EVP_PKEY_CTX_set_hkdf_md(ctx.get(),EVP_sha256())>0 &&
          EVP_PKEY_CTX_set1_hkdf_salt(ctx.get(),salt.data(),int(salt.size()))>0 &&
          EVP_PKEY_CTX_set1_hkdf_key(ctx.get(),master.data(),int(master.size()))>0 &&
          EVP_PKEY_CTX_add1_hkdf_info(ctx.get(),reinterpret_cast<const Byte*>(INFO),int(sizeof(INFO)-1))>0,
          "HKDF initialization failed");
    Key derived{};
    size_t length=derived.size();
    check(EVP_PKEY_derive(ctx.get(),derived.data(),&length)>0 && length==derived.size(),"HKDF failed");
    return derived;
}
std::array<Byte,IVLEN> nonce(uint64_t tile) {
    std::array<Byte,IVLEN> n{};
    for(int j=0;j<8;j++) n[4+j]=static_cast<Byte>(tile>>(56-8*j));
    return n;
}
Bytes record_aad(const Header& h, uint64_t index, const std::array<Byte,5>& rec) {
    Bytes a=h.bytes;
    append_u64(a,index);
    a.insert(a.end(),rec.begin(),rec.end());
    return a;
}
// records: orig length (u16) | compressed (u8) | payload length (u16)
std::array<Byte,5> record_header(uint16_t n,uint8_t compressed,uint16_t stored) {
    return {Byte(n>>8),Byte(n),compressed,Byte(stored>>8),Byte(stored)};
}
struct Sealed { Bytes ciphertext; std::array<Byte,TAGLEN> tag{}; };
Sealed seal(const Key& key,uint64_t index,const Bytes& aad,const Bytes& plaintext) {
    check(plaintext.size()<=TILE, "record plaintext exceeds tile budget");
    std::unique_ptr<EVP_CIPHER_CTX,EVPCipherCtxDel> ctx(EVP_CIPHER_CTX_new());
    check(bool(ctx),"GCM context allocation failed");
    auto iv=nonce(index);
    check(EVP_EncryptInit_ex(ctx.get(),EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1 &&
          EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_IVLEN,IVLEN,nullptr)==1 &&
          EVP_EncryptInit_ex(ctx.get(),nullptr,nullptr,key.data(),iv.data())==1,
          "GCM initialization failed");
    int processed=0;
    check(EVP_EncryptUpdate(ctx.get(),nullptr,&processed,aad.data(),int(aad.size()))==1,"GCM AAD failed");
    Sealed result; result.ciphertext.resize(plaintext.size()+16);
    int got=0;
    check(EVP_EncryptUpdate(ctx.get(),result.ciphertext.data(),&got,plaintext.data(),int(plaintext.size()))==1,
          "GCM encryption failed");
    int final_n=0;
    check(EVP_EncryptFinal_ex(ctx.get(),result.ciphertext.data()+got,&final_n)==1,"GCM final failed");
    result.ciphertext.resize(size_t(got+final_n));
    check(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_GET_TAG,TAGLEN,result.tag.data())==1,
          "GCM tag generation failed");
    return result;
}
Bytes unseal(const Key& key,uint64_t index,const Bytes& aad,const Bytes& ciphertext,
             const std::array<Byte,TAGLEN>& tag) {
    std::unique_ptr<EVP_CIPHER_CTX,EVPCipherCtxDel> ctx(EVP_CIPHER_CTX_new());
    check(bool(ctx),"GCM context allocation failed");
    auto iv=nonce(index);
    check(EVP_DecryptInit_ex(ctx.get(),EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1 &&
          EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_IVLEN,IVLEN,nullptr)==1 &&
          EVP_DecryptInit_ex(ctx.get(),nullptr,nullptr,key.data(),iv.data())==1,
          "GCM initialization failed");
    int processed=0;
    check(EVP_DecryptUpdate(ctx.get(),nullptr,&processed,aad.data(),int(aad.size()))==1,"GCM AAD failed");
    Bytes out(ciphertext.size()+16);
    int got=0;
    check(EVP_DecryptUpdate(ctx.get(),out.data(),&got,ciphertext.data(),int(ciphertext.size()))==1,
          "GCM decryption failed");
    check(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_TAG,TAGLEN,
          const_cast<Byte*>(tag.data()))==1,"GCM tag configuration failed");
    int final_n=0;
    // NEVER expose or decode unauthenticated output before this succeeds.
    check(EVP_DecryptFinal_ex(ctx.get(),out.data()+got,&final_n)==1,
          "AUTHENTICATION FAILURE: wrong key, corrupted ciphertext or modified metadata");
    out.resize(size_t(got+final_n));
    return out;
}

Bytes pack(const std::array<Byte,TILE>& v, uint8_t& compressed) {
    uLongf bound=compressBound(TILE);
    Bytes candidate(static_cast<size_t>(bound));
    check(compress2(candidate.data(),&bound,v.data(),TILE,Z_BEST_SPEED)==Z_OK,
          "zlib compression failed");
    if(bound<TILE) { compressed=1; candidate.resize(static_cast<size_t>(bound)); return candidate; }
    compressed=0;
    return Bytes(v.begin(),v.end());
}
std::array<Byte,TILE> unpack(const Bytes& data, uint8_t compressed) {
    std::array<Byte,TILE> out{};
    if(compressed) {
        uLongf expected=TILE;
        check(uncompress(out.data(),&expected,data.data(),static_cast<uLong>(data.size()))==Z_OK &&
              expected==TILE,"invalid decompressed 3D tile");
    } else {
        check(data.size()==TILE,"uncompressed tile has wrong size");
        std::copy(data.begin(),data.end(),out.begin());
    }
    return out;
}

// POSIX mode 0600 temp output; atomic no-replace publication on same filesystem.
class TempOutput {
    fs::path dest_, temp_;
    bool published_=false;
public:
    explicit TempOutput(const fs::path& dest):dest_(dest) {
        check(!dest_.empty() && !fs::exists(dest_), "destination exists or is invalid");
        std::array<Byte,12> rnd{};
        for(int attempt=0;attempt<10;attempt++) {
            secure_random(rnd.data(),rnd.size());
            static constexpr char hex[]="0123456789abcdef";
            std::string postfix;
            for(Byte v:rnd) {postfix+=hex[v>>4];postfix+=hex[v&15];}
            temp_=fs::path(dest_.string()+".tmp."+postfix);
            int fd=::open(temp_.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
            if(fd>=0) {::close(fd); return;}
        }
        fail("could not create secure temporary output");
    }
    TempOutput(const TempOutput&)=delete;
    TempOutput& operator=(const TempOutput&)=delete;
    const fs::path& path()const{return temp_;}
    void publish() {
        check(!published_,"output already published");
        fs::create_hard_link(temp_,dest_); // fails if destination already exists
        published_=true;
        fs::remove(temp_);
    }
    ~TempOutput() {if(!published_){std::error_code ec;fs::remove(temp_,ec);}}
};

Key load_key(const fs::path& p) {
    check(fs::is_regular_file(p) && fs::file_size(p)==32, "key file must be 32 raw bytes");
    auto perms=fs::status(p).permissions();
    auto exposed=fs::perms::group_all|fs::perms::others_all;
    check((perms&exposed)==fs::perms::none,"key file has unsafe group/other permissions");
    Key key{};
    std::ifstream f(p,std::ios::binary);
    check(bool(f),"cannot open key");
    read_exact(f,key.data(),key.size());
    return key;
}
void new_key(const fs::path& p) {
    check(!fs::exists(p),"key file exists; refusing overwrite");
    Key key{};secure_random(key.data(),key.size());
    int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    check(fd>=0,"cannot securely create key file");
    size_t written=0;
    while(written<key.size()) {
        ssize_t w=::write(fd,key.data()+written,key.size()-written);
        if(w<=0) {::close(fd);OPENSSL_cleanse(key.data(),key.size());fail("key write failed");}
        written+=size_t(w);
    }
    check(::close(fd)==0,"key close failed");
    OPENSSL_cleanse(key.data(),key.size());
}

uint16_t expected_orig(uint64_t i, const Header& h) {
    if(h.length==0) return 0;
    if(i+1<h.count) return static_cast<uint16_t>(TILE);
    uint64_t rem=h.length%TILE;
    return static_cast<uint16_t>(rem==0?TILE:rem);
}

// The 'consider' path verifies each tile by encrypting/decrypting in memory,
// without producing an output file or changing authoritative data.
void process_encrypt(const Key& master, const fs::path& input, const fs::path& output,
                     const std::string& mime, bool consider_only) {
    check(fs::is_regular_file(input),"input must be a regular file");
    const uintmax_t size=fs::file_size(input);
    check(size<=UINT64_MAX,"input file size not representable");
    Header h=make_header(static_cast<uint64_t>(size),mime);
    Key subkey=derive_file_key(master,h.salt);
    std::ifstream in(input,std::ios::binary);
    check(bool(in),"cannot read input file");
    std::unique_ptr<TempOutput> tmp;
    std::ofstream out;
    if(!consider_only) {
        tmp=std::make_unique<TempOutput>(output);
        out.open(tmp->path(),std::ios::binary|std::ios::trunc);
        check(bool(out),"cannot open encrypted output");
        write_exact(out,h.bytes.data(),h.bytes.size());
    }
    uint64_t processed=0,compressed_count=0;
    for(uint64_t i=0;i<h.count;++i) {
        const uint16_t n=expected_orig(i,h);
        std::array<Byte,TILE> tile{};
        read_exact(in,tile.data(),n);
        const auto original=tile;
        encode3d(tile);
        uint8_t was_compressed=0;
        Bytes payload=pack(tile,was_compressed);
        if(was_compressed) ++compressed_count;
        const auto rec=record_header(n,was_compressed,static_cast<uint16_t>(payload.size()));
        Bytes aad=record_aad(h,i,rec);
        Sealed encrypted=seal(subkey,i,aad,payload);
        if(consider_only) {
            auto verified=unseal(subkey,i,aad,encrypted.ciphertext,encrypted.tag);
            auto decoded=unpack(verified,was_compressed);
            decode3d(decoded);
            check(decoded==original,"consideration round-trip invariant failed");
        } else {
            write_exact(out,rec.data(),rec.size());
            write_exact(out,encrypted.ciphertext.data(),encrypted.ciphertext.size());
            write_exact(out,encrypted.tag.data(),encrypted.tag.size());
        }
        processed+=n;
    }
    check(processed==size,"input size changed during processing");
    Byte additional=0;
    in.read(reinterpret_cast<char*>(&additional),1);
    check(in.eof(),"input changed or read error after expected end");
    if(!consider_only) {
        out.flush();check(bool(out),"flush failed");
        out.close();check(!out.fail(),"close failed");
        tmp->publish();
    }
    OPENSSL_cleanse(subkey.data(),subkey.size());
    std::cout<<(consider_only?"CONSIDERED (uncommitted)":"SEALED")
             <<" bytes="<<processed<<" tiles="<<h.count<<" compressed_tiles="<<compressed_count
             <<" mime="<<mime<<"\n";
}

void process_decrypt(const Key& master, const fs::path& input, const fs::path& output) {
    std::ifstream in(input,std::ios::binary);
    check(bool(in),"cannot open encrypted input");
    Header h=parse_header(in);
    Key subkey=derive_file_key(master,h.salt);
    TempOutput tmp(output);
    std::ofstream out(tmp.path(),std::ios::binary|std::ios::trunc);
    check(bool(out),"cannot create decryption temporary file");
    uint64_t recovered=0;
    for(uint64_t i=0;i<h.count;i++) {
        std::array<Byte,5> rec{};
        read_exact(in,rec.data(),rec.size());
        const uint16_t n=u16(rec.data());
        const Byte comp=rec[2];
        const uint16_t length=u16(rec.data()+3);
        check(n==expected_orig(i,h),"record original length mismatch");
        check(comp==0||comp==1,"invalid compression indicator");
        check(length>0 && length<=TILE,"record payload length exceeds tile budget");
        check(comp!=0 || length==TILE,"uncompressed tile must be 512 bytes");
        check(comp!=1 || length<TILE,"compressed tile must be smaller than raw tile");
        Bytes ct(length);
        read_exact(in,ct.data(),ct.size());
        std::array<Byte,TAGLEN> tag{};
        read_exact(in,tag.data(),tag.size());
        Bytes aad=record_aad(h,i,rec);
        Bytes recovered_payload=unseal(subkey,i,aad,ct,tag);
        std::array<Byte,TILE> tile=unpack(recovered_payload,comp);
        decode3d(tile);
        for(size_t j=n;j<TILE;j++) check(tile[j]==0,"padding invariant broken");
        write_exact(out,tile.data(),n);
        recovered+=n;
    }
    Byte extra=0;
    in.read(reinterpret_cast<char*>(&extra),1);
    check(in.eof(),"trailing bytes after authenticated tile sequence");
    check(recovered==h.length,"total recovered size mismatch");
    out.flush();check(bool(out),"flush failed");
    out.close();check(!out.fail(),"close failed");
    tmp.publish();
    OPENSSL_cleanse(subkey.data(),subkey.size());
    std::cout<<"OPENED bytes="<<recovered<<" tiles="<<h.count<<" mime="<<h.mime<<"\n";
}

int main(int argc, char** argv) {
    try {
        if(argc<2) {
            std::cerr<<"Usage:\n"
                "  dm3d-secure keygen <keyfile>\n"
                "  dm3d-secure seal <keyfile> <input> <output.dm3d> <mime>\n"
                "  dm3d-secure open <keyfile> <input.dm3d> <output>\n"
                "  dm3d-secure consider <keyfile> <input> <mime>\n"
                "  dm3d-secure selftest\n";
            return 2;
        }
        const std::string command=argv[1];
        if(command=="keygen" && argc==3) {
            new_key(argv[2]);std::cout<<"KEY_GENERATED (32 raw bytes; protect and back up separately)\n";
        } else if(command=="seal" && argc==6) {
            Key key=load_key(argv[2]);
            process_encrypt(key,argv[3],argv[4],argv[5],false);
            OPENSSL_cleanse(key.data(),key.size());
        } else if(command=="open" && argc==5) {
            Key key=load_key(argv[2]);
            process_decrypt(key,argv[3],argv[4]);
            OPENSSL_cleanse(key.data(),key.size());
        } else if(command=="consider" && argc==5) {
            Key key=load_key(argv[2]);
            process_encrypt(key,argv[3],fs::path(),argv[4],true);
            OPENSSL_cleanse(key.data(),key.size());
        } else if(command=="selftest" && argc==2) {
            for(unsigned pattern=0;pattern<256;pattern+=17) {
                std::array<Byte,TILE> in{};
                for(size_t i=0;i<TILE;i++) in[i]=static_cast<Byte>((i*37 + pattern) % 256);
                auto x=in;
                encode3d(x); decode3d(x);
                check(x==in,"3D lifting self-test failed");
            }
            std::array<Byte,TILE> delta{};delta[7]=255;auto x=delta;
            encode3d(x);decode3d(x);check(x==delta,"basis self-test failed");
            std::cout<<"SELFTEST PASS: 3D multiscale lifting inverse (all tested patterns)\n";
        } else { fail("invalid command or argument count (run without arguments for usage)"); }
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"ERROR: "<<e.what()<<"\n";
        return 1;
    }
}
} // namespace dm3d

int main(int argc,char** argv) { return dm3d::main(argc,argv); }
