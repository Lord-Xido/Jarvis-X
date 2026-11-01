// JARVIS X v6: authenticated streaming envelope over any binary stream, including JX3DSTR3.
// Build: g++ -std=c++17 -O3 -Wall -Wextra -Wpedantic secure_aead.cpp -lcrypto -o jarvisx_secure
// Crypto: random 256-bit master key in a private file, per-file HKDF-SHA256, AES-256-GCM.
// Each 64-KiB chunk gets a unique 96-bit nonce and 128-bit authentication tag.
// The full fixed header, chunk index and plaintext chunk size are authenticated as AAD.
// No plaintext output is published unless all chunk tags pass authentication.
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/rand.h>
#include <openssl/crypto.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
namespace fs=std::filesystem;
namespace sx {
using B=uint8_t;
constexpr size_t CHUNK=65536, HDR=44, TAG=16;
constexpr B MAGIC[8]={'J','X','6','A','E','A','D','1'};
constexpr uint32_t VERSION=1;
constexpr uint64_t LIMIT=1ULL<<40; // explicit 1 TiB input cap for the reference runtime
using Key=std::array<B,32>;
using Salt=std::array<B,16>;
using Nonce=std::array<B,12>;
using Header=std::array<B,HDR>;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void random(B* p,size_t n){check(RAND_bytes(p,static_cast<int>(n))==1,"secure RNG failed");}
void wr32(B* p,uint32_t n){for(int i=0;i<4;i++)p[i]=B(n>>(8*i));}
void wr64(B* p,uint64_t n){for(int i=0;i<8;i++)p[i]=B(n>>(8*i));}
uint32_t rd32(const B* p){uint32_t v=0;for(int i=0;i<4;i++)v|=uint32_t(p[i])<<(8*i);return v;}
uint64_t rd64(const B* p){uint64_t v=0;for(int i=0;i<8;i++)v|=uint64_t(p[i])<<(8*i);return v;}
void randfile(const std::string& p){
 check(!fs::exists(p),"key file already exists");Key k{};random(k.data(),k.size());
 int fd=::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
 check(fd>=0,"cannot create private key file");
 size_t done=0;while(done<k.size()){auto n=::write(fd,k.data()+done,k.size()-done);
  if(n<=0){::close(fd);::unlink(p.c_str());OPENSSL_cleanse(k.data(),k.size());throw std::runtime_error("key file write failed");}
  done+=size_t(n);
 }
 bool ok=(::fsync(fd)==0)&&(::close(fd)==0);OPENSSL_cleanse(k.data(),k.size());
 check(ok,"private key file flush failed");
}
struct Secret{Key k{};~Secret(){OPENSSL_cleanse(k.data(),k.size());}Secret()=default;Secret(const Secret&)=delete;Secret& operator=(const Secret&)=delete;Secret(Secret&& o) noexcept{k=o.k;OPENSSL_cleanse(o.k.data(),o.k.size());}};
Secret read_key(const std::string& p){
 struct stat st{};check(::lstat(p.c_str(),&st)==0 && S_ISREG(st.st_mode),"invalid key file");
 check((st.st_mode&0077)==0,"key file must be owner-only (chmod 600)");
 check(st.st_size==32,"key file must contain exactly 32 random bytes");
 Secret s;std::ifstream f(p,std::ios::binary);check(bool(f),"cannot open key file");
 f.read(reinterpret_cast<char*>(s.k.data()),s.k.size());check(bool(f),"key file read failed");
 return s;
}
struct KeyHolder{Key k{};~KeyHolder(){OPENSSL_cleanse(k.data(),k.size());}};
void derive(const Key& master,const B* salt,Key& derived){
 using Ctx=std::unique_ptr<EVP_PKEY_CTX, decltype(&EVP_PKEY_CTX_free)>;
 Ctx ctx(EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF,nullptr),EVP_PKEY_CTX_free);
 check(ctx!=nullptr,"HKDF context allocation failed");
 const char info[]="JARVIS-X-v6-AES-256-GCM-stream-v1";
 check(EVP_PKEY_derive_init(ctx.get())>0,"HKDF init failed");
 check(EVP_PKEY_CTX_set_hkdf_md(ctx.get(),EVP_sha256())>0,"HKDF digest failed");
 check(EVP_PKEY_CTX_set1_hkdf_salt(ctx.get(),salt,16)>0,"HKDF salt failed");
 check(EVP_PKEY_CTX_set1_hkdf_key(ctx.get(),master.data(),int(master.size()))>0,"HKDF key failed");
 check(EVP_PKEY_CTX_add1_hkdf_info(ctx.get(),reinterpret_cast<const B*>(info),sizeof(info)-1)>0,"HKDF info failed");
 size_t n=derived.size();check(EVP_PKEY_derive(ctx.get(),derived.data(),&n)>0&&n==32,"HKDF derive failed");
}
Nonce nonce(const Header& h,uint64_t index){Nonce n{};std::copy(h.begin()+40,h.begin()+44,n.begin());
 for(int i=0;i<8;i++)n[4+i]=B(index>>(56-8*i));
 return n;}
Header build_header(uint64_t bytes){Header h{};std::copy(std::begin(MAGIC),std::end(MAGIC),h.begin());
 wr32(h.data()+8,VERSION);wr32(h.data()+12,CHUNK);wr64(h.data()+16,bytes);
 random(h.data()+24,16);random(h.data()+40,4);return h;}
uint64_t n_chunks(uint64_t n){return n?((n-1)/CHUNK+1):1;}
void parse_header(const Header& h){check(std::memcmp(h.data(),MAGIC,8)==0,"invalid encrypted magic");
 check(rd32(h.data()+8)==VERSION,"unsupported envelope version");
 check(rd32(h.data()+12)==CHUNK,"unsupported chunk size");
 check(rd64(h.data()+16)<=LIMIT,"input exceeds reference envelope capacity");
 check(n_chunks(rd64(h.data()+16))<=(1ULL<<32),"too many chunks");}
void crypt_chunk(bool enc,const Key& key,const Header& header,uint64_t i,const B* in,int n,B* out,B* tag){
 using Ctx=std::unique_ptr<EVP_CIPHER_CTX, decltype(&EVP_CIPHER_CTX_free)>;
 Ctx ctx(EVP_CIPHER_CTX_new(),EVP_CIPHER_CTX_free);check(ctx!=nullptr,"cipher allocation failed");
 auto iv=nonce(header,i);B aad_extra[12]{};wr64(aad_extra,i);wr32(aad_extra+8,uint32_t(n));int outlen=0,tmp=0;
 if(enc){
  check(EVP_EncryptInit_ex(ctx.get(),EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1,"encrypt init failed");
  check(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_IVLEN,12,nullptr)==1,"IV len failed");
  check(EVP_EncryptInit_ex(ctx.get(),nullptr,nullptr,key.data(),iv.data())==1,"encrypt key/IV failed");
  check(EVP_EncryptUpdate(ctx.get(),nullptr,&tmp,header.data(),header.size())==1,"AAD header failed");
  check(EVP_EncryptUpdate(ctx.get(),nullptr,&tmp,aad_extra,sizeof(aad_extra))==1,"AAD index failed");
  if(n)check(EVP_EncryptUpdate(ctx.get(),out,&outlen,in,n)==1&&outlen==n,"encrypt bytes failed");
  check(EVP_EncryptFinal_ex(ctx.get(),out+outlen,&tmp)==1&&tmp==0,"encrypt final failed");
  check(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_GET_TAG,TAG,tag)==1,"tag failed");
 }else{
  check(EVP_DecryptInit_ex(ctx.get(),EVP_aes_256_gcm(),nullptr,nullptr,nullptr)==1,"decrypt init failed");
  check(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_IVLEN,12,nullptr)==1,"IV len failed");
  check(EVP_DecryptInit_ex(ctx.get(),nullptr,nullptr,key.data(),iv.data())==1,"decrypt key/IV failed");
  check(EVP_DecryptUpdate(ctx.get(),nullptr,&tmp,header.data(),header.size())==1,"AAD header failed");
  check(EVP_DecryptUpdate(ctx.get(),nullptr,&tmp,aad_extra,sizeof(aad_extra))==1,"AAD index failed");
  if(n)check(EVP_DecryptUpdate(ctx.get(),out,&outlen,in,n)==1&&outlen==n,"decrypt bytes failed");
  check(EVP_CIPHER_CTX_ctrl(ctx.get(),EVP_CTRL_GCM_SET_TAG,TAG,tag)==1,"set tag failed");
  check(EVP_DecryptFinal_ex(ctx.get(),out+outlen,&tmp)==1&&tmp==0,"AUTHENTICATION FAILED: wrong key or modified stream");
 }
}
struct TmpFile{std::string path;bool published=false;explicit TmpFile(const std::string& dest){
  auto a=dest+".tmp.XXXXXX";std::vector<char> s(a.begin(),a.end());s.push_back('\0');
  int fd=mkstemp(s.data());check(fd>=0,"cannot create secure temporary file");path=s.data();
  check(::close(fd)==0,"close temp file failed");
 }~TmpFile(){if(!published)::unlink(path.c_str());}};
void publish(TmpFile& t,const std::string& dst){check(!fs::exists(dst),"destination already exists");
 check(::link(t.path.c_str(),dst.c_str())==0,"cannot publish destination (already exists or cross-device)");
 check(::unlink(t.path.c_str())==0,"cannot remove temp file");t.published=true;}
void samepath(const std::string&a,const std::string&b){check(fs::absolute(a).lexically_normal()!=fs::absolute(b).lexically_normal(),"source and destination must differ");}
struct Report{uint64_t bytes=0,encrypted=0,blocks=0;double seconds=0;};
Report run(bool enc,const std::string& inpath,const std::string& outpath,const std::string& keypath){
 samepath(inpath,outpath);check(!fs::exists(outpath),"destination already exists");
 check(fs::is_regular_file(inpath),"input is not a regular file");
 const auto total=fs::file_size(inpath);check(total<=(LIMIT+HDR+(n_chunks(LIMIT)*TAG)),"input too large");
 Secret master=read_key(keypath);KeyHolder derived;Header header{};
 auto start=std::chrono::steady_clock::now();std::ifstream in(inpath,std::ios::binary);check(bool(in),"cannot open input");
 if(enc){check(total<=LIMIT,"plaintext input exceeds limit");header=build_header(total);}
 else{in.read(reinterpret_cast<char*>(header.data()),header.size());check(bool(in),"truncated encrypted header");parse_header(header);
  uint64_t orig=rd64(header.data()+16),count=n_chunks(orig);
  check(count <= (std::numeric_limits<uint64_t>::max()-HDR-orig)/TAG,"envelope length overflow");
  check(total==HDR+orig+count*TAG,"encrypted payload has wrong length");}
 derive(master.k,header.data()+24,derived.k);
 TmpFile t(outpath);std::ofstream out(t.path,std::ios::binary|std::ios::trunc);check(bool(out),"cannot open temp output");
 if(enc){out.write(reinterpret_cast<const char*>(header.data()),header.size());check(bool(out),"header write failed");}
 uint64_t plain=enc?total:rd64(header.data()+16),chunks=n_chunks(plain);
 std::vector<B> a(CHUNK),b(CHUNK);std::array<B,TAG> tag{};
 for(uint64_t i=0;i<chunks;i++){
  size_t n=static_cast<size_t>(std::min<uint64_t>(CHUNK,plain-i*CHUNK));
  if(n){in.read(reinterpret_cast<char*>(a.data()),n);check(bool(in),"input truncated");}
  if(!enc){in.read(reinterpret_cast<char*>(tag.data()),TAG);check(bool(in),"missing authentication tag");}
  crypt_chunk(enc,derived.k,header,i,a.data(),int(n),b.data(),tag.data());
  if(n)out.write(reinterpret_cast<const char*>(b.data()),n);
  if(enc)out.write(reinterpret_cast<const char*>(tag.data()),TAG);
  check(bool(out),"payload write failed");
 }
 check(in.peek()==std::char_traits<char>::eof(),"unconsumed input bytes");
 out.flush();check(bool(out),"output flush failed");out.close();check(bool(out),"output close failed");
 publish(t,outpath);Report r;r.bytes=plain;r.blocks=chunks;r.encrypted=HDR+plain+TAG*chunks;
 r.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return r;
}
void show(const char* mode,const Report&r){std::cout<<std::fixed<<std::setprecision(5)
 <<"{\"action\":\""<<mode<<"\",\"plaintext_bytes\":"<<r.bytes
 <<",\"bits\":"<<(r.bytes*8)<<",\"chunks\":"<<r.blocks
 <<",\"envelope_bytes\":"<<r.encrypted<<",\"overhead_bytes\":"<<(r.encrypted-r.bytes)
 <<",\"seconds\":"<<r.seconds<<",\"MiB_per_second\":"<<(r.seconds>0?(r.bytes/1048576.)/r.seconds:0.0)<<"}\n";}
} // namespace sx
int main(int argc,char**argv){try{
 if(argc==3&&std::string(argv[1])=="init-key"){sx::randfile(argv[2]);std::cout<<"Key created with owner-only permissions; back it up securely.\n";return 0;}
 if(argc==3&&std::string(argv[1])=="inspect"){
  std::ifstream in(argv[2],std::ios::binary);sx::check(bool(in),"cannot inspect file");sx::Header h{};
  in.read(reinterpret_cast<char*>(h.data()),h.size());sx::check(bool(in),"truncated header");sx::parse_header(h);
  const auto sz=sx::rd64(h.data()+16),count=sx::n_chunks(sz);
  sx::check(std::filesystem::file_size(argv[2])==sx::HDR+sz+count*sx::TAG,"invalid envelope size");
  std::cout<<"JX6AEAD1 version=1 plaintext_bytes="<<sz<<" chunks="<<count<<" chunk_size="<<sx::CHUNK
   <<" authentication=UNVERIFIED_without_key\n";return 0;}
 if(argc==6&&std::string(argv[4])=="--key-file"){
  std::string action(argv[1]);sx::check(action=="encrypt"||action=="decrypt","unknown operation");
  sx::show(action.c_str(),sx::run(action=="encrypt",argv[2],argv[3],argv[5]));return 0;}
 std::cerr<<"Usage: jarvisx_secure init-key FILE | encrypt IN OUT --key-file SECRET | decrypt IN OUT --key-file SECRET | inspect FILE\n";return 2;
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}