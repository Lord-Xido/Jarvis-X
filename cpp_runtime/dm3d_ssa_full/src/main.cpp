#include "dm3d_ssa.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <tuple>
#ifdef _WIN32
#include <process.h>
#endif
namespace fs=std::filesystem;
using namespace dm3d;
#ifndef DM3D_SSA_HEADER
#define DM3D_SSA_HEADER "include/dm3d_ssa.hpp"
#endif
#ifndef DM3D_CXX
#define DM3D_CXX "g++"
#endif
struct Candidate {Genome g;double error{},mseValue{},timeUs{},memoryMB{},fitness{-1e100};bool valid{};};
static constexpr int GX=8,GY=8,GZ=12;
static int wrap(int i,int n){i%=n;return i<0?i+n:i;}
static int delta(int x,int y,int n){int d=wrap(y-x,n);return d>n/2?d-n:d;}
static Candidate assess(Genome g,const Value& x,const Value& target){
    Candidate c;c.g=g;
    try{
        auto graph=make_graph(g,x.shape.dim[0]);auto shapes=validate(graph,x.shape);
        auto v=execute(graph,x);
        c.error=max_error(v,target);c.mseValue=mse(v,target);
        c.valid=std::isfinite(c.error)&&c.error<=.055;
        if(!c.valid)return c;
        auto t=std::chrono::steady_clock::now();
        for(int i=0;i<3;++i){volatile double observed=checksum(execute(graph,x));(void)observed;}
        c.timeUs=1e6*std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count()/3.0;
        size_t total=0;for(const auto& s:shapes)total+=s.count()*sizeof(float);
        c.memoryMB=double(total)/1e6;
        c.fitness=10.0/(1.0+150.0*c.mseValue)-.1*graph.nodes.size()-.02*c.memoryMB-.002*std::log1p(c.timeUs);
    }catch(const std::exception&){c.valid=false;}
    return c;
}
static Genome mutate(Genome g,Genome elite,double contract,double mutation,std::mt19937& rng){
    std::uniform_real_distribution<double> u(0,1);std::uniform_int_distribution<int> step(-2,2);
    auto move=[&](int x,int e,int N){int nx=wrap(x+int(std::round(contract*delta(x,e,N))),N);if(u(rng)<mutation)nx=wrap(nx+step(rng),N);return nx;};
    Genome out{move(g.x,elite.x,GX),move(g.y,elite.y,GY),move(g.z,elite.z,GZ),g.extra};
    if(u(rng)<mutation*.8)out.extra=!out.extra;
    return out;
}
static std::string f(float x){std::ostringstream os;os<<std::setprecision(9)<<std::fixed<<x<<'f';return os.str();}
static std::string shape_cpp(const Shape& s){std::ostringstream o;o<<"{Kind::"<<(s.kind==Kind::Scalar?"Scalar":s.kind==Kind::Vector?"Vector":s.kind==Kind::Matrix?"Matrix":"Tensor3D")<<",{"<<s.dim[0]<<","<<s.dim[1]<<","<<s.dim[2]<<"}}";return o.str();}
static void emit(const fs::path& path,const Graph& g,int N){
    std::ifstream header(DM3D_SSA_HEADER,std::ios::binary);
    if(!header)throw std::runtime_error("unable to read runtime header for standalone emission");
    fs::create_directories(path.parent_path());
    std::ofstream os(path,std::ios::binary);if(!os)throw std::runtime_error("unable to open output cpp");
    os<<"// DM3D SSA generated C++17 executable. Self-contained runtime follows.\n";
    std::string headerText((std::istreambuf_iterator<char>(header)),{});
    const std::string guard="#pragma once";
    if(headerText.compare(0,guard.size(),guard)==0)headerText.erase(0,guard.size());
    os<<headerText;
    os<<"\n#include <fstream>\n#include <iostream>\nint main(int argc,char**argv){try{\n";
    os<<"using namespace dm3d; const int N="<<N<<"; Graph g;g.nodes={\n";
    for(const Node& n:g.nodes)os<<"{Op::"<<name(n.op)<<","<<n.a<<","<<n.b<<","<<f(n.p)<<","<<f(n.q)<<","<<shape_cpp(n.requested)<<"},\n";
    os<<"}; g.output="<<g.output<<";auto input=make_input(N);auto result=execute(g,input);auto truth=execute(target_graph(N),input);\n";
    os<<"if(argc<2)return 2;std::ofstream output(argv[1],std::ios::binary);if(!output)return 3;";
    os<<"output.write(reinterpret_cast<const char*>(result.data.data()),std::streamsize(result.data.size()*sizeof(float)));";
    os<<"if(!output)return 4;std::cout<<std::setprecision(15)<<\"checksum=\"<<checksum(result)<<\" max_error=\"<<max_error(result,truth)<<\" count=\"<<result.data.size()<<'\\n';return 0;";
    os<<"}catch(const std::exception&e){std::cerr<<e.what()<<'\\n';return 5;}}\n";
    if(!os)throw std::runtime_error("C++ emission failed");
}
static Value load_result(const fs::path& p, Shape s){
    Value out=filled(s,0);std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("generated binary did not emit output");
    f.read(reinterpret_cast<char*>(out.data.data()),std::streamsize(out.data.size()*sizeof(float)));
    if(f.gcount()!=std::streamsize(out.data.size()*sizeof(float))||f.peek()!=EOF)throw std::runtime_error("generated binary output size mismatch");
    check_value(out);return out;
}
int main(int argc,char** argv){try{
    int generations=argc>1?std::clamp(std::atoi(argv[1]),1,30):8;
    int population=argc>2?std::clamp(std::atoi(argv[2]),4,96):16;
    int N=argc>3?std::clamp(std::atoi(argv[3]),8,24):8;
    if(N%2)throw std::invalid_argument("even N required for Pool2");
    fs::path dir=argc>4?fs::path(argv[4]):fs::path("generated");
    fs::create_directories(dir);
    auto x=make_input(N),target=execute(target_graph(N),x);
    std::mt19937 rng(17);std::uniform_int_distribution<int> dx(0,7),dy(0,7),dz(0,11);
    std::vector<Genome> pop;pop.push_back({4,3,3,true});pop.push_back({4,3,3,false});
    while(int(pop.size())<population)pop.push_back({dx(rng),dy(rng),dz(rng),bool(dx(rng)&1)});
    Candidate champion;double mutation=.42,previousBest=-1e100;int stale=0;
    std::cout<<"DM3D SSA FULL SYSTEM G=Z8xZ8xZ12 generations="<<generations<<" population="<<population<<" N="<<N<<"\n";
    std::ofstream trace(dir/"evolution.csv");trace<<"generation,x,y,z,extra,valid,error,mse,us,memory_mb,fitness,mutation\n";
    for(int gen=0;gen<generations;++gen){
        std::vector<Candidate> cand;for(const auto& g:pop)cand.push_back(assess(g,x,target));
        std::sort(cand.begin(),cand.end(),[](const Candidate& a,const Candidate& b){return a.fitness>b.fitness;});
        if(!cand.front().valid)throw std::runtime_error("no valid genomes");
        if(cand.front().fitness>champion.fitness)champion=cand.front();
        if(cand.front().fitness>previousBest+1e-4){previousBest=cand.front().fitness;stale=0;mutation=std::max(.12,mutation*.87);}else{stale++;if(stale>=2)mutation=std::min(.82,mutation*1.2);}
        auto e=cand.front();trace<<gen<<','<<e.g.x<<','<<e.g.y<<','<<e.g.z<<','<<e.g.extra<<','<<e.valid<<','<<e.error<<','<<e.mseValue<<','<<e.timeUs<<','<<e.memoryMB<<','<<e.fitness<<','<<mutation<<'\n';
        std::cout<<"gen="<<gen<<" elite=("<<e.g.x<<','<<e.g.y<<','<<e.g.z<<") extra="<<e.g.extra<<" maxerr="<<std::setprecision(6)<<e.error<<" fitness="<<e.fitness<<" mutation="<<mutation<<'\n';
        std::vector<Genome> next{e.g,champion.g};double contraction=.24+.58*double(gen+1)/generations;
        while(int(next.size())<population){const auto& parent=cand[size_t(next.size())%std::min<size_t>(5,cand.size())].g;next.push_back(mutate(parent,e.g,contraction,mutation,rng));}
        pop.swap(next);
    }
    Graph winner=make_graph(champion.g,N);auto expected=execute(winner,x);
    fs::path cpp=dir/"best_kernel.cpp",bin=dir/"best_kernel",dat=dir/"best_kernel.bin";
    emit(cpp,winner,N);
#ifdef _WIN32
    // A native CMake sub-build uses the already configured MSVC toolchain.
    // Direct std::system() calls to cl.exe fail when VS lives in Program Files.
    fs::path native=dir/"standalone", build=native/"build";
    fs::create_directories(native);
    fs::copy_file(cpp,native/"best_kernel.cpp",fs::copy_options::overwrite_existing);
    std::ofstream conf(native/"CMakeLists.txt");
    conf<<"cmake_minimum_required(VERSION 3.16)\nproject(dm3d_emitted LANGUAGES CXX)\n"
        <<"set(CMAKE_CXX_STANDARD 17)\nadd_executable(best_kernel best_kernel.cpp)\n";
    conf.close();
    std::string configure="cmake -S \""+native.string()+"\" -B \""+build.string()+"\"";
    std::string compile="cmake --build \""+build.string()+"\" --config Release --parallel 2";
    std::cout<<"[CONFIGURE] "<<configure<<'\n';
    if(std::system(configure.c_str())!=0)throw std::runtime_error("generated CMake configure failed");
    std::cout<<"[COMPILE] "<<compile<<'\n';
    if(std::system(compile.c_str())!=0)throw std::runtime_error("generated source compilation failed");
    bin=build/"Release"/"best_kernel.exe";
    // Avoid cmd.exe quoting ambiguities by invoking the binary directly.
    std::string binary=bin.string(),data=dat.string();
    const char* arguments[]={binary.c_str(),data.c_str(),nullptr};
    std::cout<<"[EXECUTE] "<<binary<<'\n';
    if(_spawnv(_P_WAIT,binary.c_str(),arguments)!=0)throw std::runtime_error("generated binary execution failed");
#else
    std::string compile=std::string("\"")+DM3D_CXX+"\" -O2 -std=c++17 \""+cpp.string()+"\" -o \""+bin.string()+"\"";
    std::cout<<"[COMPILE] "<<compile<<'\n';if(std::system(compile.c_str())!=0)throw std::runtime_error("generated source compilation failed");
    std::string run="\""+bin.string()+"\" \""+dat.string()+"\"";
    std::cout<<"[EXECUTE] "<<run<<'\n';if(std::system(run.c_str())!=0)throw std::runtime_error("generated binary execution failed");
#endif
    Value actual=load_result(dat,expected.shape);
    double err=max_error(expected,actual);
    std::cout<<"[VERIFY] lowering_max_error="<<std::setprecision(12)<<err<<" task_max_error="<<max_error(actual,target)<<"\n";
    if(err>1e-6||max_error(actual,target)>.055)throw std::runtime_error("lowering / task invariant failed");
    std::cout<<"PASS emitted="<<cpp.string()<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"DM3D FAIL: "<<e.what()<<'\n';return 1;}}
