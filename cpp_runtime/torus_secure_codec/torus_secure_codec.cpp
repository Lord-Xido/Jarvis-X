// Jarvis-X Secure Torus Codec: reversible 8^3 byte transform + AES-256-GCM.
// v1 file format: magic[8], original_size_le64, nonce[12], ciphertext[ceil(n/512)*512], tag[16].
// Not a learned ANN, not compression, and not a proprietary cryptographic primitive.
#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace {
constexpr std::size_t TILE = 512;
constexpr std::size_t HEADER = 28;
constexpr std::size_t TAG = 16;
using Block = std::array<unsigned char, TILE>;
using Key = std::array<unsigned char, 32>;
using Head = std::array<unsigned char, HEADER>;
constexpr std::array<unsigned char,8> MAGIC{{'J','X','T','O','R','3','D','1'}};

void require(bool b, const std::string& message) { if (!b) throw std::runtime_error(message); }
void random_bytes(unsigned char* p, std::size_t n) {
    require(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()), "random request too large");
    require(RAND_bytes(p, static_cast<int>(n)) == 1, "OpenSSL CSPRNG unavailable");
}
void exact_read(std::ifstream& f, unsigned char* p, std::size_t n) {
    f.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n));
    require(f.gcount() == static_cast<std::streamsize>(n), "truncated input");
}
std::size_t idx(int x, int y, int z) { return static_cast<std::size_t>((z*8+y)*8+x); }
void lift_axis(Block& b, int axis, int stride) {
    for (int z=0;z<8;++z) for(int y=0;y<8;++y) for(int x=0;x<8;++x) {
        int coord = axis==0 ? x : axis==1 ? y : z;
        if ((coord % (2*stride)) < stride) continue;
        int xx=x, yy=y, zz=z;
        if (axis==0) xx-=stride;
        if (axis==1) yy-=stride;
        if (axis==2) zz-=stride;
        b[idx(x,y,z)] ^= b[idx(xx,yy,zz)];
    }
}
void encode3d(Block& b) {
    for (int axis=0;axis<3;++axis) for(int stride : {1,2,4}) lift_axis(b,axis,stride);
}
void decode3d(Block& b) {
    for (int axis=2;axis>=0;--axis) for(int stride : {4,2,1}) lift_axis(b,axis,stride);
}
std::uint64_t get64(const unsigned char* p) {
    std::uint64_t v=0;
    for(int i=7;i>=0;--i) v=(v<<8)|p[i];
    return v;
}
void put64(unsigned char* p, std::uint64_t v) {
    for(int i=0;i<8;++i) {p[i]=static_cast<unsigned char>(v&255U);v>>=8;}
}
std::uint64_t padded(std::uint64_t n) {
    require(n <= std::numeric_limits<std::uint64_t>::max()-(TILE-1), "file too large");
    return ((n+TILE-1)/TILE)*TILE;
}
struct SecretKey {
    Key bytes{};
    ~SecretKey() { OPENSSL_cleanse(bytes.data(),bytes.size()); }
};
SecretKey load_key(const fs::path& path) {
    std::ifstream in(path,std::ios::binary | std::ios::ate);
    require(static_cast<bool>(in),"cannot read key file");
    require(in.tellg()==static_cast<std::streamoff>(32),"key file must contain exactly 32 raw bytes");
    in.seekg(0);
    SecretKey key;
    exact_read(in,key.bytes.data(),key.bytes.size());
    return key;
}
void keygen(const fs::path& path) {
    Key key{};
    random_bytes(key.data(),key.size());
    const int fd=::open(path.c_str(),O_CREAT|O_EXCL|O_WRONLY,0600);
    if(fd<0) {OPENSSL_cleanse(key.data(),key.size());throw std::runtime_error("cannot exclusively create key file");}
    const ssize_t count=::write(fd,key.data(),key.size());
    const int closed=::close(fd);
    OPENSSL_cleanse(key.data(),key.size());
    require(count==32 && closed==0,"failed to save key file");
}
struct TempOutput {
    fs::path path;
    std::FILE* stream=nullptr;
    bool committed=false;
    explicit TempOutput(const fs::path& target) {
        fs::path base=target.parent_path();
        if(base.empty()) base=".";
        std::string name=(base/(target.filename().string()+".tmp.XXXXXX")).string();
        std::vector<char> chars(name.begin(),name.end());chars.push_back('\0');
        int fd=::mkstemp(chars.data());
        require(fd>=0,"cannot create temporary output");
        path=chars.data();
        stream=::fdopen(fd,"wb");
        if(!stream){::close(fd);fs::remove(path);throw std::runtime_error("fdopen failed");}
    }
    void write(const unsigned char* p,std::size_t n) {
        require(std::fwrite(p,1,n,stream)==n,"output write failed");
    }
    void commit(const fs::path& dest) {
        require(std::fflush(stream)==0,"output flush failed");
        require(std::fclose(stream)==0,"output close failed");
        stream=nullptr;
        require(::link(path.c_str(), dest.c_str())==0,
                "exclusive output publish failed (destination may exist)");
        committed=true;
        std::error_code ec;fs::remove(path,ec);
    }
    ~TempOutput() {
        if(stream) std::fclose(stream);
        if(!committed && !path.empty()) {std::error_code ec;fs::remove(path,ec);}
    }
};
struct Cipher {
    EVP_CIPHER_CTX* p=EVP_CIPHER_CTX_new();
    Cipher(){require(p!=nullptr,"cipher allocation failed");}
    ~Cipher(){EVP_CIPHER_CTX_free(p);}
    Cipher(const Cipher&)=delete;
    Cipher& operator=(const Cipher&)=delete;
};
void encrypt_file(const fs::path& src,const fs::path& dest,const fs::path& key_path) {
    require(src!=dest,"input and output are identical");
    require(!fs::exists(dest),"destination exists; refusing overwrite");
    const auto n=fs::file_size(src);
    padded(n);
    std::ifstream in(src,std::ios::binary);
    require(static_cast<bool>(in),"cannot open input");
    SecretKey key=load_key(key_path);
    Head h{};
    std::copy(MAGIC.begin(),MAGIC.end(),h.begin());
    put64(h.data()+8,n);
    random_bytes(h.data()+16,12);
    Cipher c;
    require(EVP_EncryptInit_ex(c.p,EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1,"AES-GCM setup failed");
    require(EVP_CIPHER_CTX_ctrl(c.p,EVP_CTRL_GCM_SET_IVLEN,12,nullptr)==1,"nonce length rejected");
    require(EVP_EncryptInit_ex(c.p,nullptr,nullptr,key.bytes.data(),h.data()+16)==1,"AES-GCM key setup failed");
    int got=0;
    require(EVP_EncryptUpdate(c.p,nullptr,&got,h.data(),HEADER)==1,"AAD failed");
    TempOutput out(dest);out.write(h.data(),h.size());
    std::uint64_t remaining=n;
    for(std::uint64_t off=0;off<padded(n);off+=TILE) {
        Block block{};
        const std::size_t take=static_cast<std::size_t>(std::min<std::uint64_t>(remaining,TILE));
        exact_read(in,block.data(),take);
        remaining-=take;
        encode3d(block);
        Block ciphertext{};
        require(EVP_EncryptUpdate(c.p,ciphertext.data(),&got,block.data(),TILE)==1 && got==TILE,"encryption failed");
        out.write(ciphertext.data(),got);
    }
    unsigned char tail[16]{};
    require(EVP_EncryptFinal_ex(c.p,tail,&got)==1 && got==0,"GCM finalize failed");
    std::array<unsigned char,TAG> tag{};
    require(EVP_CIPHER_CTX_ctrl(c.p,EVP_CTRL_GCM_GET_TAG,TAG,tag.data())==1,"tag retrieval failed");
    out.write(tag.data(),tag.size());
    out.commit(dest);
}
void decrypt_file(const fs::path& src,const fs::path& dest,const fs::path& key_path) {
    require(src!=dest,"input and output are identical");
    require(!fs::exists(dest),"destination exists; refusing overwrite");
    std::ifstream in(src,std::ios::binary);
    require(static_cast<bool>(in),"cannot open input");
    Head h{};exact_read(in,h.data(),h.size());
    require(std::equal(MAGIC.begin(),MAGIC.end(),h.begin()),"unsupported file format");
    const std::uint64_t n=get64(h.data()+8);
    require(padded(n)<=std::numeric_limits<std::uint64_t>::max()-HEADER-TAG,"file length overflow");
    require(fs::file_size(src)==HEADER+padded(n)+TAG,"encrypted length mismatch");
    SecretKey key=load_key(key_path);
    Cipher c;
    require(EVP_DecryptInit_ex(c.p,EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1,"AES-GCM setup failed");
    require(EVP_CIPHER_CTX_ctrl(c.p,EVP_CTRL_GCM_SET_IVLEN,12,nullptr)==1,"nonce length rejected");
    require(EVP_DecryptInit_ex(c.p,nullptr,nullptr,key.bytes.data(),h.data()+16)==1,"AES-GCM key setup failed");
    int got=0;
    require(EVP_DecryptUpdate(c.p,nullptr,&got,h.data(),HEADER)==1,"AAD failed");
    TempOutput out(dest);
    std::uint64_t remaining=n;
    for(std::uint64_t off=0;off<padded(n);off+=TILE) {
        Block ciphertext{};exact_read(in,ciphertext.data(),TILE);
        Block block{};
        require(EVP_DecryptUpdate(c.p,block.data(),&got,ciphertext.data(),TILE)==1 && got==TILE,"decryption failed");
        decode3d(block);
        const std::size_t take=static_cast<std::size_t>(std::min<std::uint64_t>(remaining,TILE));
        out.write(block.data(),take);
        remaining-=take;
    }
    std::array<unsigned char,TAG> tag{};
    exact_read(in,tag.data(),TAG);
    require(EVP_CIPHER_CTX_ctrl(c.p,EVP_CTRL_GCM_SET_TAG,TAG,tag.data())==1,"tag configuration failed");
    unsigned char tail[16]{};
    require(EVP_DecryptFinal_ex(c.p,tail,&got)==1 && got==0,"AUTHENTICATION FAILED: wrong key or tampered file");
    out.commit(dest);
}
void selftest() {
    for(int length : {0,1,7,511,512,513,1025}) {
        Block b{};
        for(std::size_t i=0;i<TILE;++i)b[i]=static_cast<unsigned char>((i*97+length)&255);
        Block saved=b;encode3d(b);decode3d(b);
        require(b==saved,"3D lifting inverse mismatch");
    }
    std::cout << "SELFTEST PASS: 3D lifting round-trip\n";
}
} // namespace
int main(int argc,char** argv) {
    try {
        require(argc>=2,"usage: torus_secure_codec keygen <keyfile> | seal/open <input> <output> <keyfile> | selftest");
        const std::string cmd=argv[1];
        if(cmd=="selftest") {require(argc==2,"selftest needs no arguments");selftest();}
        else if(cmd=="keygen") {require(argc==3,"usage: keygen <keyfile>");keygen(argv[2]);std::cout<<"Key generated (32 bytes; protect it and keep it out of Git).\n";}
        else if(cmd=="seal") {require(argc==5,"usage: seal <input> <output> <keyfile>");encrypt_file(argv[2],argv[3],argv[4]);std::cout<<"SEAL OK\n";}
        else if(cmd=="open") {require(argc==5,"usage: open <input> <output> <keyfile>");decrypt_file(argv[2],argv[3],argv[4]);std::cout<<"OPEN OK (authenticated)\n";}
        else throw std::runtime_error("unknown command");
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
