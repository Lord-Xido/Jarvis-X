#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dm3d {
enum class Op:uint8_t{FABRIC=1,STREAM,ZEROS,ENCODE,FUSE,MAP3D,ATTEND,FOLD3D,DECODE,RESIDUAL,OMEGA,LOOP_BEG,LOOP_END,EMIT,HALT=255};
static uint64_t pack(Op o,uint8_t d=0,uint8_t a=0,uint8_t b=0,uint32_t imm=0){return(uint64_t(uint8_t(o))<<56)|(uint64_t(d)<<48)|(uint64_t(a)<<40)|(uint64_t(b)<<32)|imm;}
struct I{Op o;uint8_t d,a,b;uint32_t imm;};
static I dec(uint64_t w){return{Op((w>>56)&255),uint8_t(w>>48),uint8_t(w>>40),uint8_t(w>>32),uint32_t(w)};}
static uint32_t fb(float x){return std::bit_cast<uint32_t>(x);} static float ff(uint32_t x){return std::bit_cast<float>(x);}
static std::string trim(std::string s){while(!s.empty()&&std::isspace((unsigned char)s.front()))s.erase(s.begin());while(!s.empty()&&std::isspace((unsigned char)s.back()))s.pop_back();return s;}
static std::vector<std::string> tok(const std::string&s){std::istringstream q(s);std::vector<std::string>v;for(std::string x;q>>x;)v.push_back(x);return v;}
static uint8_t mod(const std::string&s){static std::unordered_map<std::string,uint8_t>m{{"video",1},{"audio",2},{"image",3},{"text",4},{"depth",5},{"metadata",6},{"tokens",7}};auto i=m.find(s);if(i==m.end())throw std::runtime_error("bad modality "+s);return i->second;}
struct Program{std::vector<uint64_t> c;};

class Compiler{
 std::unordered_map<std::string,uint8_t> r_; uint16_t nr_=1; struct LM{uint8_t r;float e;}; std::vector<LM> lm_;
 uint8_t r(const std::string&s){auto i=r_.find(s);if(i!=r_.end())return i->second;if(nr_>=255)throw std::runtime_error("register limit");return r_[s]=uint8_t(nr_++);}
 static uint32_t u(const std::string&s){return uint32_t(std::stoul(s));} static float f(const std::string&s){return std::stof(s);} static void n(const std::vector<std::string>&v,size_t z){if(v.size()!=z)throw std::runtime_error("bad argument count");}
public:
 Program file(const std::string&p){std::ifstream in(p);if(!in)throw std::runtime_error("open source");std::ostringstream ss;ss<<in.rdbuf();return text(ss.str());}
 Program text(const std::string&s){Program p;std::istringstream in(s);std::string line;int ln=0,depth=0;while(std::getline(in,line)){++ln;auto h=line.find('#');if(h!=std::string::npos)line.resize(h);line=trim(line);if(line.empty())continue;try{
  if(line=="}"){if(depth<=0||lm_.empty())throw std::runtime_error("unmatched }");auto m=lm_.back();lm_.pop_back();p.c.push_back(pack(Op::LOOP_END,0,m.r,0,fb(m.e)));--depth;continue;}
  auto t=tok(line);auto&o=t[0];
  if(o=="fabric"){n(t,2);p.c.push_back(pack(Op::FABRIC,0,0,0,u(t[1])));}
  else if(o=="stream"){n(t,4);p.c.push_back(pack(Op::STREAM,r(t[1]),mod(t[2]),0,u(t[3])));}
  else if(o=="zeros"){n(t,3);p.c.push_back(pack(Op::ZEROS,r(t[1]),0,0,u(t[2])));}
  else if(o=="encode"||o=="embed"){n(t,5);if(t[2]!="->")throw std::runtime_error("expected ->");p.c.push_back(pack(Op::ENCODE,r(t[3]),r(t[1]),0,u(t[4])));}
  else if(o=="fuse"){n(t,5);if(t[3]!="->")throw std::runtime_error("expected ->");p.c.push_back(pack(Op::FUSE,r(t[4]),r(t[1]),r(t[2])));}
  else if(o=="map3d"){n(t,6);if(t[2]!="->"||t[4]!="scale")throw std::runtime_error("map3d syntax");p.c.push_back(pack(Op::MAP3D,r(t[3]),r(t[1]),0,fb(f(t[5]))));}
  else if(o=="llm_attend"){n(t,6);if(t[2]!="->"||t[4]!="heads")throw std::runtime_error("llm_attend syntax");p.c.push_back(pack(Op::ATTEND,r(t[3]),r(t[1]),0,u(t[5])));}
  else if(o=="fold3d"){n(t,7);if(t[3]!="->"||t[5]!="lambda")throw std::runtime_error("fold3d syntax");p.c.push_back(pack(Op::FOLD3D,r(t[4]),r(t[1]),r(t[2]),fb(f(t[6]))));}
  else if(o=="decode"){n(t,5);if(t[2]!="->")throw std::runtime_error("expected ->");p.c.push_back(pack(Op::DECODE,r(t[3]),r(t[1]),0,u(t[4])));}
  else if(o=="residual"){n(t,5);if(t[3]!="->")throw std::runtime_error("expected ->");p.c.push_back(pack(Op::RESIDUAL,r(t[4]),r(t[1]),r(t[2])));}
  else if(o=="omega"){n(t,5);if(t[3]!="beta")throw std::runtime_error("omega syntax");p.c.push_back(pack(Op::OMEGA,r(t[1]),r(t[2]),0,fb(f(t[4]))));}
  else if(o=="repeat"){n(t,7);if(t[2]!="until"||t[4]!="rms"||t[6]!="{")throw std::runtime_error("repeat syntax");p.c.push_back(pack(Op::LOOP_BEG,0,0,0,u(t[1])));lm_.push_back({r(t[3]),f(t[5])});++depth;}
  else if(o=="emit"){n(t,2);p.c.push_back(pack(Op::EMIT,0,r(t[1])));} else if(o=="halt")p.c.push_back(pack(Op::HALT)); else throw std::runtime_error("unknown op "+o);
 }catch(const std::exception&e){throw std::runtime_error("line "+std::to_string(ln)+": "+e.what());}}
 if(depth)throw std::runtime_error("unterminated repeat");
 if(p.c.empty()||dec(p.c.back()).o!=Op::HALT)p.c.push_back(pack(Op::HALT));
 return p;}
};

static void save(const Program&p,const std::string&f){std::ofstream o(f,std::ios::binary);const char m[8]={'D','M','3','D','R','P','L','1'};o.write(m,8);uint32_t n=p.c.size();o.write((char*)&n,4);o.write((char*)p.c.data(),n*8);if(!o)throw std::runtime_error("write bytecode");}
static Program load(const std::string&f){std::ifstream i(f,std::ios::binary);char m[8];i.read(m,8);if(std::string(m,8)!="DM3DRPL1")throw std::runtime_error("bad bytecode");uint32_t n=0;i.read((char*)&n,4);Program p;p.c.resize(n);i.read((char*)p.c.data(),n*8);if(!i)throw std::runtime_error("truncated bytecode");return p;}
struct T{std::vector<float>x;void z(size_t n,float v=0){x.assign(n,v);}size_t n()const{return x.size();}};static float rms(const T&t){double s=0;for(float v:t.x)s+=double(v)*v;return t.n()?float(std::sqrt(s/t.n())):0;}

class VM{
 Program p_;std::array<T,256>t_{};size_t pc_=0;bool run_=true;struct L{size_t pc;uint32_t max,it;};std::vector<L>ls_;
 static float q(float x){return std::tanh(x);}
 static void stream(T&o,uint8_t m,uint32_t n){o.z(n);for(uint32_t i=0;i<n;++i){float x=float(i%1021)/1021.f;o.x[i]=(m==2)?(.7f*std::sin(31*x)+.2f*std::sin(113*x)):(m==4?float((i*31+m*17)%257)/128.f-1.f:.55f*std::sin(6.2831853f*(x+.071f*m))+.25f*std::cos(17*x));}}
 static void enc(const T&i,T&o,uint32_t n){if(!i.n())throw std::runtime_error("encode empty");o.z(n);size_t w=std::max<size_t>(1,i.n()/n);for(uint32_t j=0;j<n;++j){float s=0;size_t b=(size_t(j)*w)%i.n();for(size_t k=0;k<w;++k)s+=i.x[(b+k)%i.n()];o.x[j]=q(s/w);}}
 static void fuse(const T&a,const T&b,T&o){size_t n=std::max(a.n(),b.n());o.z(n);for(size_t i=0;i<n;++i)o.x[i]=q(.5f*(a.n()?a.x[i%a.n()]:0)+.5f*(b.n()?b.x[i%b.n()]:0));}
 static void map3(const T&i,T&o,float s){o=i;size_t n=i.n(),d=std::max<size_t>(1,size_t(std::cbrt(double(n))));for(size_t k=0;k<n;++k){size_t x=k%d,y=(k/d)%d,z=(k/(d*d))%d;o.x[k]=q(s*i.x[k]+.03f*std::sin(.013f*x+.017f*y+.019f*z));}}
 static void att(const T&i,T&o,uint32_t h){if(!i.n())throw std::runtime_error("attention empty");h=std::max(1u,h);o.z(i.n());for(size_t x=0;x<i.n();++x){float a=0,w=0;for(uint32_t j=0;j<h;++j){size_t k=(x+1+13*j)%i.n();float z=std::exp(std::clamp(i.x[x]*i.x[k],-8.f,8.f));a+=z*i.x[(k+j+1)%i.n()];w+=z;}o.x[x]=q(.65f*i.x[x]+.35f*a/std::max(w,1e-6f));}}
 static void fold(const T&i,const T&m,T&o,float l){o.z(i.n());for(size_t x=0;x<i.n();++x){size_t a=(x+i.n()-1)%i.n(),b=(x+1)%i.n();float lap=i.x[a]-2*i.x[x]+i.x[b],om=m.n()?m.x[x%m.n()]:0;o.x[x]=q(l*i.x[x]+.08f*lap+.12f*om);}}
 static void decod(const T&i,T&o,uint32_t n){o.z(n);for(uint32_t x=0;x<n;++x){double p=n<=1?0:double(x)*(i.n()-1)/(n-1);size_t a=p,b=std::min(a+1,i.n()-1);float f=p-a;o.x[x]=(1-f)*i.x[a]+f*i.x[b];}}
 static void res(const T&a,const T&b,T&o){size_t n=std::min(a.n(),b.n());o.z(n);for(size_t i=0;i<n;++i)o.x[i]=a.x[i]-b.x[i];}
 static void om(T&o,const T&r,float b){if(o.n()!=r.n())o.z(r.n());for(size_t i=0;i<r.n();++i)o.x[i]=b*o.x[i]+(1-b)*r.x[i];}
public:explicit VM(Program p):p_(std::move(p)){}void go(){while(run_&&pc_<p_.c.size()){auto i=dec(p_.c[pc_++]);switch(i.o){
 case Op::FABRIC:{uint64_t s=i.imm;std::cout<<"[FABRIC] "<<s<<" x "<<s<<" = "<<s*s<<" logical positions\n";break;}
 case Op::STREAM:stream(t_[i.d],i.a,i.imm);std::cout<<"[STREAM] r"<<unsigned(i.d)<<" elements="<<i.imm<<"\n";break;
 case Op::ZEROS:t_[i.d].z(i.imm);break;case Op::ENCODE:enc(t_[i.a],t_[i.d],i.imm);break;case Op::FUSE:fuse(t_[i.a],t_[i.b],t_[i.d]);break;case Op::MAP3D:map3(t_[i.a],t_[i.d],ff(i.imm));break;case Op::ATTEND:att(t_[i.a],t_[i.d],i.imm);break;case Op::FOLD3D:fold(t_[i.a],t_[i.b],t_[i.d],ff(i.imm));break;case Op::DECODE:decod(t_[i.a],t_[i.d],i.imm);break;case Op::RESIDUAL:res(t_[i.a],t_[i.b],t_[i.d]);break;case Op::OMEGA:om(t_[i.d],t_[i.a],ff(i.imm));break;
 case Op::LOOP_BEG:ls_.push_back({pc_,i.imm,0});break;case Op::LOOP_END:{auto&l=ls_.back();++l.it;float e=rms(t_[i.a]),ep=ff(i.imm);std::cout<<"[REFINE] k="<<l.it<<" residual="<<std::fixed<<std::setprecision(6)<<e<<"\n";if(e>=ep&&l.it<l.max)pc_=l.pc;else{std::cout<<"[REFINE] stop="<<(e<ep?"converged":"iteration-cap")<<"\n";ls_.pop_back();}break;}
 case Op::EMIT:std::cout<<"[EMIT] r"<<unsigned(i.a)<<" elements="<<t_[i.a].n()<<" rms="<<std::fixed<<std::setprecision(6)<<rms(t_[i.a])<<"\n";break;case Op::HALT:run_=false;std::cout<<"[HALT]\n";break;default:throw std::runtime_error("bad opcode");}}}
};
}
int main(int ac,char**av){using namespace dm3d;try{if(ac<3){std::cerr<<"dm3d_rpl run source | compile source out.bc | exec file.bc\n";return 2;}std::string c=av[1];if(c=="run"){Compiler x;VM(x.file(av[2])).go();}else if(c=="compile"){if(ac<4)throw std::runtime_error("output path required");Compiler x;auto p=x.file(av[2]);save(p,av[3]);std::cout<<"[COMPILE] instructions="<<p.c.size()<<"\n";}else if(c=="exec"){VM(load(av[2])).go();}else throw std::runtime_error("bad command");return 0;}catch(const std::exception&e){std::cerr<<"fatal: "<<e.what()<<"\n";return 1;}}
