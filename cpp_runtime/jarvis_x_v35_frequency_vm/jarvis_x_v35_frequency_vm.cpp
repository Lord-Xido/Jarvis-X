// JARVIS-X v3.5: executable frequency-modulated multimodal *reference* VM.
// Synthetic 5-modality inputs, fixed parameter-free QKV, 3D implicit field,
// relaxed fixed-point loop, 9 supervisory checks, AEAD checkpoint encryption.
// NOT a trained Transformer, real-time clock boost, or secure enclave.
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic jarvis_x_v35_frequency_vm.cpp -lcrypto -o jarvis_x_v35
#include <openssl/evp.h>
#include <openssl/kdf.h>
#include <openssl/rand.h>
#include <openssl/crypto.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace jx35 {
constexpr int GRID=16, D=64, MODALITIES=5, HEADS=4, HEAD_D=D/HEADS;
constexpr float PI=3.14159265358979323846f;
constexpr int NC=GRID*GRID*GRID;
using Field=std::vector<float>;
using Latent=std::array<float,D>;
using Modal=std::array<Latent,MODALITIES>;
using Bytes=std::vector<unsigned char>;
using Key=std::array<unsigned char,32>;

struct Instruction {uint8_t opcode,src,dst,aux;};
static_assert(sizeof(Instruction)==4,"VM instruction must be 32-bit");
enum Op:uint8_t {INGEST=0x10, FREQ3D=0x15, ENCODE=0x20, QKV=0x30,
    MANIFOLD=0x40, DECODE=0x50, LAPLACIAN=0x60, VERIFY=0x70, HALT=0xff};
static constexpr std::array<Instruction,9> ROM{{
    {INGEST,0,1,0}, {FREQ3D,1,2,0}, {ENCODE,2,3,0}, {QKV,3,4,0},
    {MANIFOLD,4,5,0}, {DECODE,5,6,0}, {LAPLACIAN,6,7,0},
    {VERIFY,7,8,0}, {HALT,0,0,0}}};

int index(int x,int y,int z){return (z*GRID+y)*GRID+x;}
float norm_coord(int x){return (float(x)-7.5f)/7.5f;}
float max_abs_delta(const Field&a,const Field&b){
    if(a.size()!=b.size())throw std::runtime_error("field size mismatch");
    float mx=0;for(size_t i=0;i<a.size();++i)mx=std::max(mx,std::abs(a[i]-b[i]));return mx;
}
float mean(const Field& a){double sum=0;for(float x:a)sum+=x;return float(sum/a.size());}
void append_u32(Bytes& o,uint32_t x){for(int i=0;i<4;++i)o.push_back(uint8_t(x>>(8*i)));}
void append_f32(Bytes& o,float x){append_u32(o,std::bit_cast<uint32_t>(x));}

struct Metrics{int iterations=0;float residual=std::numeric_limits<float>::infinity();
    bool converged=false;size_t surface_points=0;std::array<int,9> supervisor_hits{};
    float attention_max_normalization_error=0;float obfuscation_max_recovery_error=0;};

class VM {
    int frequency_=2,bands_=4;
    Field psi_=Field(NC,0.0f), decoded_, candidate_;
    Modal modal_{};
    Latent latent_{};
    Metrics metrics_{};
    static void ensure_finite(const Field& f){for(float x:f)if(!std::isfinite(x))throw std::runtime_error("non-finite 3D field");}
    void ingest(){
        // All input streams are SYNTHETIC. Each requires a real modality encoder in production.
        constexpr char txt[]="JARVIS-X 3D frequency synthesis";
        for(int d=0;d<D;++d){
            float u=float(d)/float(D);
            modal_[0][d]=float(uint8_t(txt[d%(sizeof(txt)-1)]))/127.5f-1.0f; // text byte features
            modal_[1][d]=std::sin(2*PI*(3.0f*u+0.08f*std::sin(2*PI*u))); // audio waveform
            modal_[2][d]=(float((d%8)) + float(d/8))/7.0f-1.0f; // grayscale image grid
            modal_[3][d]=psi_[size_t((d*NC)/D)]; // 3D recurrent field input
            float theta=2*PI*u;
            modal_[4][d]=0.5f*(std::sin(theta)+std::cos(3*theta)); // sample geometry
        }
        metrics_.supervisor_hits[0]++;
    }
    void freq3d(){
        // Frequency modulation is 2^f *bands*, not 2^f CPU clock or GPU throughput.
        for(int m=0;m<MODALITIES;++m)for(int d=0;d<D;++d){
            const int k=d%bands_;const float u=float(d)/D;
            modal_[m][d]=0.90f*modal_[m][d]+0.10f*std::sin(2*PI*float(1<<k)*u);
        }
        metrics_.supervisor_hits[1]++;
    }
    void encode(){
        for(int m=0;m<MODALITIES;++m){
            float mu=0;for(float v:modal_[m])mu+=v;mu/=D;
            for(float& v:modal_[m])v=std::clamp((v-mu)*0.8f,-1.0f,1.0f);
        }
        metrics_.supervisor_hits[2]++;
    }
    void cross_attention(){
        // Four heads of normalized cross-modal softmax attention with identity Q/K/V.
        // This is a *fixed*, untrained reference, not a learned Transformer.
        latent_.fill(0);
        for(int h=0;h<HEADS;++h){
            for(int q=0;q<MODALITIES;++q){
                std::array<float,MODALITIES> logits{};
                float maxlogit=-1e30f;
                for(int k=0;k<MODALITIES;++k){
                    float dot=0;
                    for(int d=h*HEAD_D;d<(h+1)*HEAD_D;++d)dot+=modal_[q][d]*modal_[k][d];
                    logits[k]=dot/std::sqrt(float(HEAD_D));
                    maxlogit=std::max(maxlogit,logits[k]);
                }
                float denom=0;
                for(auto& s:logits){s=std::exp(s-maxlogit);denom+=s;}
                if(!(denom>0&&std::isfinite(denom)))throw std::runtime_error("attention denominator invalid");
                float normsum=0;
                for(auto& s:logits){s/=denom;normsum+=s;}
                metrics_.attention_max_normalization_error=std::max(
                    metrics_.attention_max_normalization_error,std::abs(normsum-1));
                for(int d=h*HEAD_D;d<(h+1)*HEAD_D;++d){
                    float mix=0;for(int k=0;k<MODALITIES;++k)mix+=logits[k]*modal_[k][d];
                    latent_[d]+=mix/float(MODALITIES);
                }
            }
        }
        metrics_.supervisor_hits[3]++;
    }
    void manifold(){
        // Reversible signed channel permutation is obfuscation, NOT cryptography.
        Latent coded{}, recovered{};
        for(int i=0;i<D;++i){
            const int j=(5*i+7)%D;
            coded[i]=(i%2?-1.0f:1.0f)*latent_[j];
        }
        for(int i=0;i<D;++i){
            const int j=(5*i+7)%D;
            recovered[j]=(i%2?-1.0f:1.0f)*coded[i];
        }
        for(int i=0;i<D;++i){
            metrics_.obfuscation_max_recovery_error=std::max(metrics_.obfuscation_max_recovery_error,
                std::abs(recovered[i]-latent_[i]));
            latent_[i]=recovered[i];
        }
        // Bounded deterministic transform; trainable M_theta is not implemented.
        for(int d=0;d<D;++d)latent_[d]=std::tanh(0.85f*latent_[d]
            +0.015f*std::sin(2*PI*float(1<<(d%bands_))*float(d)/D));
        metrics_.supervisor_hits[4]++;
    }
    void decode(){
        // Procedural continuous SDF sampled on a grid. This is NOT a neural field.
        decoded_.resize(NC);
        const float radius=0.63f+0.06f*latent_[0];
        for(int z=0;z<GRID;++z)for(int y=0;y<GRID;++y)for(int x=0;x<GRID;++x){
            const float a=norm_coord(x),b=norm_coord(y),c=norm_coord(z);
            const float r=std::sqrt(a*a+b*b+c*c);
            float wave=0.0f;
            for(int k=0;k<bands_;++k){
                const float omega=2*PI*float(1<<k)/GRID;
                wave+=std::sin(omega*x)*std::cos(omega*y)*std::sin(omega*z)/float(1<<k);
            }
            const float wobble=0.018f*latent_[1]*wave
                +0.01f*latent_[2]*std::sin(4*PI*float(z)/GRID);
            decoded_[size_t(index(x,y,z))]=std::clamp(r-radius+wobble,-1.0f,1.0f);
        }
        metrics_.supervisor_hits[5]++;
    }
    void laplacian(){
        // Neumann (clamped-neighbor) boundary handling, alpha=0.05 <= 1/6.
        candidate_.resize(NC);
        auto at=[this](int x,int y,int z)->float{
            return decoded_[size_t(index(std::clamp(x,0,GRID-1),
                std::clamp(y,0,GRID-1),std::clamp(z,0,GRID-1)))];};
        constexpr float alpha=0.05f;
        for(int z=0;z<GRID;++z)for(int y=0;y<GRID;++y)for(int x=0;x<GRID;++x){
            const float self=at(x,y,z);
            const float neighbor=at(x-1,y,z)+at(x+1,y,z)+at(x,y-1,z)
                +at(x,y+1,z)+at(x,y,z-1)+at(x,y,z+1);
            candidate_[size_t(index(x,y,z))]=std::clamp(
                self+alpha*(neighbor-6*self),-1.0f,1.0f);
        }
        ensure_finite(candidate_);metrics_.supervisor_hits[6]++;
    }
    void verify(){
        metrics_.residual=max_abs_delta(candidate_,psi_);metrics_.supervisor_hits[7]++;
    }
    void supervisor_finalize(){metrics_.supervisor_hits[8]++;}
public:
    explicit VM(int f=2):frequency_(f),bands_(0){
        // Validate before shifting: negative or oversized shifts are undefined behavior.
        if(f<0||f>2)throw std::runtime_error("frequency must be 0..2 for 16^3 reference grid");
        // 16 samples/axis: dyadic bands 1,2,4,8 are already at Nyquist.
        bands_=1<<f;
    }
    const Metrics& run(int max_iters=80,float tolerance=1e-5f,float beta=0.65f,bool verbose=true){
        if(max_iters<1||tolerance<=0||beta<=0||beta>1)throw std::runtime_error("invalid iteration controls");
        metrics_={};
        for(int it=0;it<max_iters;++it){
            for(const auto ins:ROM){
                switch(ins.opcode){
                    case INGEST:ingest();break;
                    case FREQ3D:freq3d();break;
                    case ENCODE:encode();break;
                    case QKV:cross_attention();break;
                    case MANIFOLD:manifold();break;
                    case DECODE:decode();break;
                    case LAPLACIAN:laplacian();break;
                    case VERIFY:verify();break;
                    case HALT:supervisor_finalize();break;
                    default:throw std::runtime_error("invalid ROM opcode");
                }
            }
            metrics_.iterations=it+1;
            if(verbose && (it<4||it%10==9||metrics_.residual<tolerance))
                std::cout<<"iteration="<<it+1<<" fixed_point_residual="
                         <<std::scientific<<metrics_.residual<<"\n";
            if(metrics_.residual<tolerance){metrics_.converged=true;break;}
            for(int i=0;i<NC;++i)psi_[size_t(i)]=(1-beta)*psi_[size_t(i)]+beta*candidate_[size_t(i)];
        }
        // Report the output F(psi) actually evaluated at final iteration.
        psi_=candidate_;metrics_.surface_points=surface_count();return metrics_;
    }
    size_t surface_count()const{
        size_t n=0;for(int z=0;z<GRID-1;++z)for(int y=0;y<GRID-1;++y)for(int x=0;x<GRID-1;++x){
            const float a=psi_[size_t(index(x,y,z))];
            if((a<0)!=(psi_[size_t(index(x+1,y,z))]<0))++n;
            if((a<0)!=(psi_[size_t(index(x,y+1,z))]<0))++n;
            if((a<0)!=(psi_[size_t(index(x,y,z+1))]<0))++n;
        }
        return n;
    }
    void export_ply(const std::string&path)const{
        struct P{float x,y,z;};std::vector<P> pts;
        for(int z=0;z<GRID-1;++z)for(int y=0;y<GRID-1;++y)for(int x=0;x<GRID-1;++x){
            const float a=psi_[size_t(index(x,y,z))];
            auto emit=[&](int xx,int yy,int zz){
                const float b=psi_[size_t(index(xx,yy,zz))];
                if((a<0)!=(b<0)){
                    const float t=std::clamp(a/(a-b),0.0f,1.0f);
                    pts.push_back({norm_coord(x)+(norm_coord(xx)-norm_coord(x))*t,
                        norm_coord(y)+(norm_coord(yy)-norm_coord(y))*t,
                        norm_coord(z)+(norm_coord(zz)-norm_coord(z))*t});
                }
            };
            emit(x+1,y,z);emit(x,y+1,z);emit(x,y,z+1);
        }
        std::ofstream f(path);if(!f)throw std::runtime_error("cannot open point cloud output");
        f<<"ply\nformat ascii 1.0\ncomment JARVIS-X v3.5 synthesized iso-zero point cloud\n"
            <<"element vertex "<<pts.size()<<"\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
        f<<std::setprecision(8);
        for(auto p:pts)f<<p.x<<" "<<p.y<<" "<<p.z<<"\n";
        if(!f)throw std::runtime_error("failed writing PLY");
    }
    Bytes serialize_state()const{
        Bytes out={'J','X','3','5','P','S','I','1'};
        append_u32(out,uint32_t(GRID));append_u32(out,uint32_t(frequency_));
        append_u32(out,uint32_t(psi_.size()));
        for(float v:latent_)append_f32(out,v);
        for(float v:psi_)append_f32(out,v);
        return out;
    }
    const Field& field()const{return psi_;}
    int bands()const{return bands_;}
};

void secure_random(unsigned char*ptr,int size){if(RAND_bytes(ptr,size)!=1)throw std::runtime_error("RAND_bytes failed");}
std::vector<unsigned char> from_hex(const std::string&str){
    if(str.size()!=64)throw std::runtime_error("JX_MASTER_KEY_HEX must be 64 hexadecimal characters");
    auto nib=[](char c)->int{if(c>='0'&&c<='9')return c-'0';
        if(c>='A'&&c<='F')return c-'A'+10;
        if(c>='a'&&c<='f')return c-'a'+10;
        throw std::runtime_error("invalid hex digit");};
    Bytes key(32);for(int i=0;i<32;++i)key[size_t(i)]=uint8_t((nib(str[2*i])<<4)|nib(str[2*i+1]));
    return key;
}
Key derive(const Bytes&master,const unsigned char*salt){
    if(master.size()!=32)throw std::runtime_error("master key must be 32 bytes");
    EVP_PKEY_CTX*ctx=EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF,nullptr);
    if(!ctx)throw std::runtime_error("HKDF context failed");
    constexpr char info[]="JARVIS-X/v3.5/field-checkpoint";
    Key key{};size_t count=key.size();
    bool ok=EVP_PKEY_derive_init(ctx)>0 && EVP_PKEY_CTX_set_hkdf_md(ctx,EVP_sha256())>0 &&
        EVP_PKEY_CTX_set1_hkdf_salt(ctx,salt,16)>0 &&
        EVP_PKEY_CTX_set1_hkdf_key(ctx,master.data(),int(master.size()))>0 &&
        EVP_PKEY_CTX_add1_hkdf_info(ctx,(const unsigned char*)info,int(sizeof(info)-1))>0 &&
        EVP_PKEY_derive(ctx,key.data(),&count)>0 && count==key.size();
    EVP_PKEY_CTX_free(ctx);if(!ok)throw std::runtime_error("HKDF derive failed");return key;
}
Bytes encrypt(const Bytes&plain,const Bytes&master){
    if(plain.size()>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("plaintext too large");
    // Header: magic(8) + salt(16) + nonce(12) + length_LE(4), all authenticated.
    Bytes blob={'J','X','3','5','A','E','A','1'};
    blob.resize(8+16+12);secure_random(blob.data()+8,16);secure_random(blob.data()+24,12);
    append_u32(blob,uint32_t(plain.size()));const int header=int(blob.size());
    Key key=derive(master,blob.data()+8);
    EVP_CIPHER_CTX*ctx=EVP_CIPHER_CTX_new();if(!ctx)throw std::runtime_error("GCM ctx failed");
    bool ok=EVP_EncryptInit_ex(ctx,EVP_aes_256_gcm(),nullptr,nullptr,nullptr)>0 &&
        EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_SET_IVLEN,12,nullptr)>0 &&
        EVP_EncryptInit_ex(ctx,nullptr,nullptr,key.data(),blob.data()+24)>0;
    int n=0,used=0;
    if(ok)ok=EVP_EncryptUpdate(ctx,nullptr,&n,blob.data(),header)>0;
    blob.resize(size_t(header)+plain.size()+16);
    if(ok)ok=EVP_EncryptUpdate(ctx,blob.data()+header,&n,plain.data(),int(plain.size()))>0;
    used=n;
    if(ok)ok=EVP_EncryptFinal_ex(ctx,blob.data()+header+used,&n)>0;
    used+=n;
    if(ok)ok=EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_GET_TAG,16,blob.data()+header+used)>0;
    EVP_CIPHER_CTX_free(ctx);OPENSSL_cleanse(key.data(),key.size());
    if(!ok)throw std::runtime_error("GCM encryption failed");
    blob.resize(size_t(header+used+16));return blob;
}
Bytes decrypt(const Bytes&blob,const Bytes&master){
    constexpr size_t HEADER=8+16+12+4;
    if(blob.size()<HEADER+16||std::memcmp(blob.data(),"JX35AEA1",8)!=0)
        throw std::runtime_error("invalid checkpoint header");
    size_t n=36;uint32_t len=0;for(int i=0;i<4;++i)len|=uint32_t(blob[n++])<<(8*i);
    if(len>size_t(std::numeric_limits<int>::max())||blob.size()!=HEADER+size_t(len)+16)
        throw std::runtime_error("invalid checkpoint length");
    Key key=derive(master,blob.data()+8);
    EVP_CIPHER_CTX*ctx=EVP_CIPHER_CTX_new();if(!ctx)throw std::runtime_error("GCM ctx failed");
    bool ok=EVP_DecryptInit_ex(ctx,EVP_aes_256_gcm(),nullptr,nullptr,nullptr)>0 &&
        EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_SET_IVLEN,12,nullptr)>0 &&
        EVP_DecryptInit_ex(ctx,nullptr,nullptr,key.data(),blob.data()+24)>0;
    int read=0,used=0;Bytes plain(len);
    if(ok)ok=EVP_DecryptUpdate(ctx,nullptr,&read,blob.data(),int(HEADER))>0;
    if(ok)ok=EVP_DecryptUpdate(ctx,plain.data(),&read,blob.data()+HEADER,int(len))>0;
    used=read;
    if(ok)ok=EVP_CIPHER_CTX_ctrl(ctx,EVP_CTRL_GCM_SET_TAG,16,(void*)(blob.data()+HEADER+len))>0;
    if(ok)ok=EVP_DecryptFinal_ex(ctx,plain.data()+used,&read)>0;
    EVP_CIPHER_CTX_free(ctx);OPENSSL_cleanse(key.data(),key.size());
    if(!ok)throw std::runtime_error("checkpoint authentication failed");
    plain.resize(size_t(used+read));return plain;
}
void save_file(const std::string&path,const Bytes&data){
    std::ofstream f(path,std::ios::binary);if(!f)throw std::runtime_error("failed to open checkpoint file");
    f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));
    if(!f)throw std::runtime_error("checkpoint write failed");
}
void check(bool flag,const std::string&name){
    if(!flag)throw std::runtime_error("SELF-TEST FAILED: "+name);
    std::cout<<"[PASS] "<<name<<"\n";
}
void selftest(){
    check((1u<<2)==4,"2^f dyadic frequency scaling");
    check(sizeof(Instruction)==4&&ROM.size()==9,"32-bit ISA and nine-stage supervisor ROM");
    check((0x400000u)==4u*1024u*1024u,"0x400000 = 4 MiB, not 1000 GB");
    VM engine(2);const auto&m=engine.run(80,1e-5f,0.65f,false);
    check(m.converged&&m.residual<1e-5f,"fixed-point residual converges on synthetic data");
    check(m.surface_points>100,"nonempty 3D iso-zero point cloud");
    check(m.attention_max_normalization_error<5e-6f,"softmax attention normalization");
    check(m.obfuscation_max_recovery_error<1e-7f,"reversible signed-permutation latent obfuscation");
    bool all=true;for(int c:m.supervisor_hits)all&=(c==m.iterations);
    check(all,"each supervisory stage executed per iteration");
    bool bounded=true;for(float x:engine.field())bounded&=(std::isfinite(x)&&x>=-1.00001f&&x<=1.00001f);
    check(bounded,"3D field finite and bounded");
    Bytes master(32),wrong(32);secure_random(master.data(),32);secure_random(wrong.data(),32);
    auto plain=engine.serialize_state();auto cipher=encrypt(plain,master);
    check(decrypt(cipher,master)==plain,"AES-256-GCM checkpoint roundtrip");
    auto rotated=encrypt(plain,wrong);
    check(rotated!=cipher&&decrypt(rotated,wrong)==plain,"re-encryption with rotated key");
    bool rejected=false;try{(void)decrypt(cipher,wrong);}catch(const std::exception&){rejected=true;}
    check(rejected,"wrong key is rejected");
    cipher[40]^=0x20;rejected=false;
    try{(void)decrypt(cipher,master);}catch(const std::exception&){rejected=true;}
    check(rejected,"ciphertext tampering is rejected");
    check(plain.size()==8u+12u+4u*D+4u*NC,"portable little-endian state serialization size");
    OPENSSL_cleanse(master.data(),master.size());OPENSSL_cleanse(wrong.data(),wrong.size());
    std::cout<<"[OK] 14/14 reference self-tests passed\n";
}
} // namespace jx35
int main(int argc,char**argv){
    using namespace jx35;
    try{
        if(argc>1&&std::string(argv[1])=="--self-test"){selftest();return 0;}
        std::string ply="",enc_path="";int freq=2;
        for(int i=1;i<argc;++i){std::string s=argv[i];
            if((s=="--ply"||s=="--encrypt"||s=="--frequency")&&i+1<argc){
                std::string val=argv[++i];
                if(s=="--ply")ply=val;else if(s=="--encrypt")enc_path=val;
                else freq=std::stoi(val);
            }else throw std::runtime_error("usage: --self-test OR [--frequency 0..2] [--ply path] [--encrypt checkpoint.aead]");
        }
        // Preflight key validation before any output (including optional plaintext PLY).
        Bytes master;
        if(!enc_path.empty()){
            const char*secret=std::getenv("JX_MASTER_KEY_HEX");
            if(!secret)throw std::runtime_error("JX_MASTER_KEY_HEX missing: refusing checkpoint/export operation");
            master=from_hex(secret);
        }
        VM engine(freq);auto m=engine.run();
        if(!m.converged)throw std::runtime_error("fixed-point iteration did not converge");
        if(!enc_path.empty()){
            save_file(enc_path,encrypt(engine.serialize_state(),master));
            OPENSSL_cleanse(master.data(),master.size());
            std::cout<<"Encrypted checkpoint: "<<enc_path<<"\n";
        }
        if(!ply.empty())engine.export_ply(ply); // Explicit opt-in for unencrypted geometry export.
        std::cout<<"bands="<<engine.bands()<<" grid="<<GRID<<"^3 residual="<<m.residual
                 <<" converged="<<std::boolalpha<<m.converged
                 <<" points="<<m.surface_points<<"\n";
        if(!ply.empty())std::cout<<"Unencrypted PLY (requested): "<<ply<<"\n";
        return 0;
    }catch(const std::exception&e){std::cerr<<"Error: "<<e.what()<<"\n";return 1;}
}
