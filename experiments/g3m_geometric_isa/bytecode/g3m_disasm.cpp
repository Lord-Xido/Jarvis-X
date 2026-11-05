// G3MBC001: shape-checking bytecode plan disassembler (C++17, dependency-free).
// It verifies an instruction plan; it does NOT perform ANN computation.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
uint16_t u16(const std::vector<uint8_t>& b,size_t x){return uint16_t(b.at(x))|(uint16_t(b.at(x+1))<<8);}
uint32_t u32(const std::vector<uint8_t>& b,size_t x){return uint32_t(b.at(x))|(uint32_t(b.at(x+1))<<8)|(uint32_t(b.at(x+2))<<16)|(uint32_t(b.at(x+3))<<24);}
uint32_t crc32(const uint8_t* p,size_t n){
  uint32_t c=0xFFFFFFFFU;
  for(size_t i=0;i<n;++i){c^=p[i];for(unsigned j=0;j<8;++j)c=(c>>1) ^ (0xEDB88320U & (0U-(c&1U)));}
  return ~c;
}
struct Reg { std::string desc; bool ready=false; };
const char* op_name(uint8_t op){
 switch(op){
  case 0x01:return "LOAD_XYZ";case 0x02:return "NORMALIZE";case 0x03:return "SUPPORT";
  case 0x04:return "KNN";case 0x05:return "EDGECONV";case 0x06:return "GEOATTN";
  case 0x07:return "VAE512";case 0x08:return "REFINE";case 0x09:return "FOLD";
  case 0x0A:return "DENORM";case 0x0B:return "VERIFY";case 0x0C:return "WRITE_XYZ";
  case 0x0D:return "EVIDENCE";case 0x20:return "TRAIN_ADAM";case 0xFF:return "HALT";
  default:return nullptr;
 }
}
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
}
int main(int argc,char** argv){
 try{
  require(argc==2,"usage: g3m_disasm <file.g3mb>");
  std::ifstream f(argv[1],std::ios::binary); require(bool(f),"cannot open input");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),{});
  require(bytes.size()>=32,"truncated header");
  require(std::string(reinterpret_cast<const char*>(bytes.data()),8)=="G3MBC001","bad magic");
  const uint32_t cnt=u32(bytes,12), max_points=u32(bytes,20), stored_crc=u32(bytes,24);
  require(u16(bytes,8)==1 && u16(bytes,10)==16 && u32(bytes,16)==1 && u32(bytes,28)==0,"unsupported version or flags");
  require(cnt>0 && cnt<=4096 && bytes.size()==32ULL+16ULL*cnt,"invalid instruction count/length");
  require(max_points>=2 && max_points<=1000000,"unsafe max_points");
  require(crc32(bytes.data()+32,bytes.size()-32)==stored_crc,"program CRC32 mismatch");
  const std::vector<uint8_t> expected={0x01,0x02,0x03,0x20,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B,0x0C,0x0D,0xFF};
  require(cnt==expected.size(),"this reference accepts exactly its canonical pipeline");
  Reg reg[16];
  auto check=[&](uint16_t r,const std::string& shape){require(r<16,"register number out of range");require(reg[r].ready && reg[r].desc==shape,"incorrect register type/shape dependency");};
  auto bind=[&](uint16_t r,std::string shape){require(r<16,"destination register out of range");require(!reg[r].ready,"register overwritten without explicit release");reg[r]={std::move(shape),true};};
  uint32_t N=0,S=0,K=0,M=0;
  std::cout<<"G3MBC001 plan | v1 | 16B instructions | max_points="<<max_points<<" | CRC32 OK\n";
  for(uint32_t pc=0;pc<cnt;++pc){
   const size_t off=32+pc*16; const uint8_t op=bytes[off], flags=bytes[off+1];
   const uint16_t dst=u16(bytes,off+2), a=u16(bytes,off+4), b=u16(bytes,off+6);
   const uint32_t imm=u32(bytes,off+8), aux=u32(bytes,off+12);
   require(op==expected.at(pc) && flags==0 && op_name(op)!=nullptr,"unrecognized/out-of-order opcode");
   std::string shape="";
   switch(op){
    case 0x01:require(dst==0 && a==65535 && b==65535 && imm>=2 && imm<=max_points && aux==0,"LOAD invalid");N=imm;shape="XYZ["+std::to_string(N)+",3]";bind(dst,shape);break;
    case 0x02:check(a,"XYZ["+std::to_string(N)+",3]");require(dst==1 && b==65535 && imm==0 && aux==0,"NORMALIZE invalid");shape="normalized["+std::to_string(N)+",3]+transform";bind(dst,shape);break;
    case 0x03:check(a,"normalized["+std::to_string(N)+",3]+transform");require(dst==2 && b==65535 && imm>=16 && imm<=1024 && imm<=N && aux==0,"SUPPORT invalid");S=imm;shape="support["+std::to_string(S)+",3]";bind(dst,shape);break;
    case 0x20:check(a,"support["+std::to_string(S)+",3]");require(dst==10 && b==65535 && imm>0 && imm<=100000 && aux==2000,"TRAIN invalid");shape="trained_model[207171 params]";bind(dst,shape);break;
    case 0x04:check(a,"support["+std::to_string(S)+",3]");require(dst==3 && b==65535 && imm>0 && imm<S && aux==0,"KNN invalid");K=imm;shape="neighbor_indices["+std::to_string(S)+","+std::to_string(K)+"]";bind(dst,shape);break;
    case 0x05:check(a,"support["+std::to_string(S)+",3]");check(b,"neighbor_indices["+std::to_string(S)+","+std::to_string(K)+"]");require(dst==4 && imm==64 && aux==0,"EDGECONV invalid");shape="features["+std::to_string(S)+",64]";bind(dst,shape);break;
    case 0x06:check(a,"features["+std::to_string(S)+",64]");check(b,"support["+std::to_string(S)+",3]");require(dst==5 && imm==64 && aux==500,"GEOATTN invalid");shape="attended_features["+std::to_string(S)+",64]";bind(dst,shape);break;
    case 0x07:check(a,"attended_features["+std::to_string(S)+",64]");check(b,"trained_model[207171 params]");require(dst==6 && imm==512 && aux==0,"VAE invalid");shape="latent[512]";bind(dst,shape);break;
    case 0x08:check(a,"latent[512]");check(b,"support["+std::to_string(S)+",3]");require(dst==7 && imm<=32 && aux==10,"REFINE invalid");shape="refined_latent[512]";bind(dst,shape);break;
    case 0x09:check(a,"refined_latent[512]");require(dst==8 && b==65535 && imm>=2 && imm<=max_points && aux>=1 && aux<=65536,"FOLD invalid");M=imm;shape="generated["+std::to_string(M)+",3]";bind(dst,shape);break;
    case 0x0A:check(a,"generated["+std::to_string(M)+",3]");check(b,"normalized["+std::to_string(N)+",3]+transform");require(dst==9 && imm==0 && aux==0,"DENORM invalid");shape="world_xyz["+std::to_string(M)+",3]";bind(dst,shape);break;
    case 0x0B:check(a,"world_xyz["+std::to_string(M)+",3]");require(dst==11 && b==65535 && imm==M && aux==0,"VERIFY invalid");shape="validated_shape_and_finite";bind(dst,shape);break;
    case 0x0C:check(a,"world_xyz["+std::to_string(M)+",3]");check(b,"validated_shape_and_finite");require(dst==12 && imm==0 && aux==0,"WRITE invalid");shape="file_receipt";bind(dst,shape);break;
    case 0x0D:check(a,"file_receipt");check(b,"validated_shape_and_finite");require(dst==13 && imm==0 && aux==0,"EVIDENCE invalid");shape="evidence_descriptor";bind(dst,shape);break;
    case 0xFF:require(dst==65535 && a==65535 && b==65535 && imm==0 && aux==0,"HALT invalid");shape="halt";break;
    default:throw std::runtime_error("unreachable opcode");
   }
   std::cout<<std::right<<std::setw(2)<<std::setfill('0')<<pc<<" +0x"<<std::hex<<std::setw(4)<<off<<std::dec<<std::setfill(' ')
            <<" "<<std::left<<std::setw(12)<<op_name(op)<<" r"<<dst<<" <- r"<<a<<",r"<<b
            <<" imm="<<imm<<" aux="<<aux<<" | "<<shape<<"\n";
  }
  std::cout<<"PLAN VALID: "<<cnt<<" instructions, "<<bytes.size()<<" bytes. ANN numerical execution NOT performed.\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"INVALID PLAN: "<<e.what()<<"\n";return 1;}
}
