#include "dm3d_ir_runtime.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
using namespace dm3d_ir; namespace fs=std::filesystem; using Clock=std::chrono::steady_clock;
#ifndef DM3D_IR_INCLUDE_DIR
#define DM3D_IR_INCLUDE_DIR "."
#endif
struct Eval{Graph g;double err{},mseErr{},ms{},cost{},fitness{-1e300};bool valid{false};};
static Graph seed(){Graph g=target();g.insert(g.begin(),{Op::DMA,0,0,1});g.push_back({Op::DMA,0,0,1});return g;}
static double cost(const Graph&g,int N){double V=double(N)*N*N,c=0;for(auto n:g){static const double w[9]={1,18,12,16,6,35,80,3,14};c+=w[(int)n.op]*V*(n.op==Op::FIX_POINT?n.iters:1);}return c/1e6;}
static Eval eval(const Graph&g,const Volume&x,const Volume&t,double hard){Eval e;e.g=g;auto a=Clock::now();Volume y=execute(x,g);auto b=Clock::now();e.ms=1e3*std::chrono::duration<double>(b-a).count();e.err=maxerr(y,t);e.mseErr=mse(y,t);e.cost=cost(g,x.N);bool finite=true;for(float q:y.v)if(!std::isfinite(q)){finite=false;break;}e.valid=finite&&!g.empty()&&g.size()<=8&&e.err<=hard;if(e.valid)e.fitness=6.0/(1+80*e.mseErr)+0.02*std::log1p(double(x.v.size())/1e6/std::max(e.ms/1000.0,1e-9))-0.035*g.size()-0.0008*e.cost;return e;}
static Graph mutate(Graph g,std::mt19937&r,double mu){std::uniform_real_distribution<double>U(0,1);std::normal_distribution<float>N(0,.07f);std::uniform_int_distribution<int>OD(0,8);if(U(r)<mu&&!g.empty()){size_t i=std::uniform_int_distribution<size_t>(0,g.size()-1)(r);if(U(r)<.32)g[i].op=(Op)OD(r);g[i].p0=std::clamp(g[i].p0+N(r),0.0f,1.5f);g[i].p1=std::clamp(g[i].p1+N(r),0.0f,1.0f);if(U(r)<.25)g[i].iters=std::clamp(g[i].iters+(U(r)<.5?-1:1),1,4);}if(U(r)<.3*mu&&g.size()<8){Node n{(Op)OD(r),.5f,.2f,1};g.insert(g.begin()+std::uniform_int_distribution<size_t>(0,g.size())(r),n);}if(U(r)<.32*mu&&g.size()>1)g.erase(g.begin()+std::uniform_int_distribution<size_t>(0,g.size()-1)(r));return g;}
static std::string gs(const Graph&g){std::ostringstream o;for(size_t i=0;i<g.size();++i){if(i)o<<" -> ";o<<name(g[i].op);}return o.str();}
static std::string f(float v){std::ostringstream o;o<<std::fixed<<std::setprecision(9)<<v;return o.str()+"f";}
static std::string emit(const Graph&g){
    std::ostringstream o;
    o << R"CPP(#include "dm3d_ir_runtime.hpp"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
using namespace dm3d_ir;
int main(int c,char**v){
    int N=c>1?std::clamp(std::atoi(v[1]),8,64):10;
    Graph g{
)CPP";
    for(size_t i=0;i<g.size();++i){
        auto n=g[i];
        o << "{Op::" << name(n.op) << "," << f(n.p0) << "," << f(n.p1) << "," << n.iters << "}";
        if(i+1<g.size()) o << ",";
        o << '\n';
    }
    o << R"CPP(    };
    auto x=input(N),y=execute(x,g),t=execute(x,target());
    std::cout<<std::setprecision(12)<<"checksum="<<checksum(y)
             <<" max_error="<<maxerr(y,t)<<" nodes="<<g.size()<<"\n";
}
)CPP";
    return o.str();
}
static bool parse(const fs::path&p,double&c,double&e){std::ifstream f(p);std::string s((std::istreambuf_iterator<char>(f)),{});auto g=[&](const char*k,double&o){auto q=s.find(k);if(q==std::string::npos)return false;q+=std::strlen(k);char*end=nullptr;o=std::strtod(s.c_str()+q,&end);return end!=s.c_str()+q;};return g("checksum=",c)&&g("max_error=",e);}
int main(int ac,char**av){int G=ac>1?std::clamp(std::atoi(av[1]),1,30):8,P=ac>2?std::clamp(std::atoi(av[2]),4,64):12,N=ac>3?std::clamp(std::atoi(av[3]),8,24):10;fs::path out=ac>4?av[4]:"generated/typed_ir_kernel.cpp";const double HARD=.065;std::mt19937 r(11);auto x=input(N),t=execute(x,target());std::vector<Graph>pop;pop.push_back(seed());for(int i=1;i<P;++i)pop.push_back(mutate(seed(),r,.85));Eval best;double mu=.60,last=-1e300;int stale=0;std::cout<<"DM3D TYPED 3D IR SYNTHESIZER\n";for(int gen=0;gen<G;++gen){std::vector<Eval>es;for(auto&g:pop)es.push_back(eval(g,x,t,HARD));std::sort(es.begin(),es.end(),[](auto&a,auto&b){return a.fitness>b.fitness;});auto e=es.front();if(e.fitness>best.fitness)best=e;if(e.fitness>last+1e-6){last=e.fitness;stale=0;mu=std::max(.18,mu*.88);}else{++stale;mu=std::min(.95,mu*(stale>=2?1.18:1.05));}std::cout<<"gen "<<gen<<" fit="<<std::fixed<<std::setprecision(4)<<e.fitness<<" err="<<std::scientific<<e.err<<std::fixed<<" mu="<<std::setprecision(2)<<mu<<" :: "<<gs(e.g)<<"\n";std::vector<Graph>nx{e.g};while((int)nx.size()<P)nx.push_back(mutate(e.g,r,mu));pop=std::move(nx);}fs::create_directories(out.parent_path().empty()?fs::path("."):out.parent_path());std::ofstream(out)<<emit(best.g);fs::path bin=out;bin.replace_extension("");fs::path log=out.parent_path()/"typed_ir_run.log";std::ostringstream cc;cc<<"g++ -O3 -std=c++17 -I\""<<DM3D_IR_INCLUDE_DIR<<"\" \""<<out.string()<<"\" -o \""<<bin.string()<<"\"";std::cout<<"[LOWER] "<<cc.str()<<"\n";if(std::system(cc.str().c_str()))return 2;std::ostringstream run;run<<"\""<<bin.string()<<"\" "<<N<<" > \""<<log.string()<<"\"";if(std::system(run.str().c_str()))return 3;double ec=0,ee=0;if(!parse(log,ec,ee))return 4;double ic=checksum(execute(x,best.g)),le=std::abs(ic-ec);std::cout<<"WINNER "<<gs(best.g)<<"\nmax_error="<<std::scientific<<best.err<<" lowering_checksum_error="<<le<<" emitted_error="<<ee<<"\n";if(le>1e-5||ee>HARD)return 5;std::cout<<"[VERIFY] PASS\n";}
