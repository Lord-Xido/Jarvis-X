#include <array>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <unordered_map>

namespace jx {
constexpr int N = 1000;
constexpr std::uint64_t nodes = std::uint64_t(N)*N*N;
constexpr std::uint64_t parameters = nodes*1000;
static_assert(parameters == 1000000000000ULL);
using Vec = std::array<double,10>;
using Features = std::array<double,6>;
using Id = std::uint64_t;
struct P { int x,y,z; };
constexpr int wrap(int x) { return (x%N+N)%N; }
constexpr Id id(P p) { return Id(wrap(p.x))+N*Id(wrap(p.y))+Id(N)*N*wrap(p.z); }
constexpr P point(Id i) { return {int(i%N),int(i/N%N),int(i/N/N)}; }
constexpr Id parameter_address(P p, P u) {
    return Id(10*p.x+u.x)+10000ULL*(10*p.y+u.y)+100000000ULL*(10*p.z+u.z);
}
struct Node {
    std::array<double,1000> theta{}; // 340 + 420 + 210 + 30
    Vec h{};
    Features f{};
};
using Field = std::unordered_map<Id,Node>; // active/loaded nodes only

template<std::size_t In,std::size_t Out>
std::array<double,Out> layer(const std::array<double,In>& x,
                             const Node& n, std::size_t off) {
    std::array<double,Out> y{};
    for(std::size_t j=0;j<Out;++j) {
        double s=n.theta[off+In*Out+j];
        for(std::size_t i=0;i<In;++i) s+=n.theta[off+j*In+i]*x[i];
        y[j]=std::tanh(s);
    }
    return y;
}

// Read only the previous sweep; missing neighbours contribute zero.
// link(from,to) is a trusted communication policy supplied by the runtime.
template<class Link>
Vec transition(Id key,const Field& old,double lambda,Link link) {
    const Node& n=old.at(key); const P p=point(key);
    std::array<Vec,3> mean{};
    for(int a=0;a<3;++a) {
        int count=0;
        for(int s:{-1,1}) {
            P q=p; if(a==0) q.x+=s; else if(a==1) q.y+=s; else q.z+=s;
            const Id qid=id(q); auto it=old.find(qid);
            if(it==old.end() || !link(qid,key)) continue;
            ++count; for(int j=0;j<10;++j) mean[a][j]+=it->second.h[j];
        }
        if(count) for(double& v:mean[a]) v/=count;
    }
    std::array<double,16> u{};
    std::copy(n.f.begin(),n.f.end(),u.begin());
    for(int j=0;j<10;++j) {
        double num=0,den=1e-12;
        for(int a=0;a<3;++a) {
            const double g=1.0/(1.0+std::exp(-n.theta[970+a*10+j]));
            num+=g*mean[a][j]; den+=g;
        }
        u[6+j]=num/den;
    }
    const auto h1=layer<16,20>(u,n,0);
    const auto h2=layer<20,20>(h1,n,340);
    Vec out=layer<20,10>(h2,n,760);
    for(int j=0;j<10;++j) out[j]=(1-lambda)*n.h[j]+lambda*out[j];
    return out;
}

template<class Link>
double sweep(Field& field,double lambda,Link link) {
    if (!(lambda>0 && lambda<=1) || field.empty())
        return std::numeric_limits<double>::infinity();
    std::unordered_map<Id,Vec> next;
    double residual=0;
    for(const auto& kv:field) {
        if (kv.first >= nodes) return std::numeric_limits<double>::infinity();
        auto h=transition(kv.first,field,lambda,link);
        for (double v : h) if (!std::isfinite(v))
            return std::numeric_limits<double>::infinity();
        for(int j=0;j<10;++j) residual=std::max(residual,std::abs(h[j]-kv.second.h[j]));
        next.emplace(kv.first,h);
    }
    for(const auto& kv:next) field.at(kv.first).h=kv.second;
    return residual;
}

// Operational equation: adapters define actual encoding, routing, scheduling,
// scoped authorisation, durable dispatch, evidence and CTR validation.
// execute_once must enforce the grant and idempotency at the execution boundary.
// reconcile must not blindly repeat an action with unknown outcome.
template<class Runtime,class State,class Input>
bool cycle(Runtime& r,State& omega,const Input& input) {
    auto q=r.encode(input);
    Field active=r.route(omega,q);
    double residual=0;
    bool converged=false;
    for(int k=0;k<128;++k) {
        residual=sweep(active,0.5,[&](Id a,Id b){return r.link(a,b);});
        if(residual<=1e-8) {converged=true;break;}
    }
    if(!converged) {r.escalate(omega);return false;}
    auto z=r.fold(active,q);
    auto plan=r.coordinate(r.decode(z));
    auto grant=r.authorise(plan);
    if(!grant) {r.escalate(omega);return false;}
    auto receipt=r.execute_once(plan,*grant);
    auto evidence=r.measure(receipt);
    if(!r.verify(plan,receipt,evidence)) {
        r.reconcile(omega,receipt);return false;
    }
    auto candidate=r.learn(omega,active,evidence);
    if(r.ctr(omega,candidate)) r.commit_parameters(omega,candidate);
    r.record(omega,receipt,evidence);
    return true;
}
} // namespace jx

// Deliberately synthetic adapters: test orchestration, not real authorisation.
struct MockState { int executions=0, commits=0, records=0, escalations=0, reconciles=0; };
struct MockRuntime {
    MockState& s;
    bool permit=true, valid=true, accept=true;
    int encode(int x) {return x;}
    jx::Field route(MockState&,int) {return {{0,jx::Node{}}};}
    bool link(jx::Id,jx::Id) {return true;}
    int fold(const jx::Field&,int q) {return q;}
    int decode(int z) {return z;}
    int coordinate(int p) {return p;}
    std::optional<int> authorise(int) {if(permit) return 1;return std::nullopt;}
    int execute_once(int p,int) {++s.executions;return p;}
    int measure(int receipt) {return receipt;}
    bool verify(int,int,int) {return valid;}
    void reconcile(MockState&,int) {++s.reconciles;}
    void escalate(MockState&) {++s.escalations;}
    int learn(MockState&,const jx::Field&,int e) {return e;}
    bool ctr(MockState&,int) {return accept;}
    void commit_parameters(MockState&,int) {++s.commits;}
    void record(MockState&,int,int) {++s.records;}
};

void check(bool ok) { if (!ok) throw std::runtime_error("regression failed"); }
int main() {
    using namespace jx;
    check(id({999,999,999})==nodes-1);
    check(id({-1,0,0})==999);
    check(parameter_address({999,999,999},{9,9,9})==parameters-1);
    Field f; f[id({0,0,0})].h.fill(1.0); f[id({999,0,0})].h.fill(-1.0);
    // Zero weights give zero ANN output, so h halves at each step.
    for(int k=0;k<30;++k) sweep(f,0.5,[](Id,Id){return true;});
    check(std::abs(f.at(0).h[0]-std::ldexp(1.0,-30))<1e-15);
    // Construct a nonzero path from x-neighbour h[0] to output[0].
    Field g; auto& n=g[id({0,0,0})]; n.theta[6]=1; n.theta[340]=1; n.theta[760]=1;
    g[id({999,0,0})].h[0]=1;
    auto allow=transition(0,g,1,[](Id,Id){return true;});
    auto deny=transition(0,g,1,[](Id,Id){return false;});
    check(std::abs(allow[0]-std::tanh(std::tanh(std::tanh(1.0/3))))<1e-12);
    check(deny[0]==0);
    { MockState s; MockRuntime r{s}; check(cycle(r,s,42));
      check(s.executions==1 && s.commits==1 && s.records==1); }
    { MockState s; MockRuntime r{s,false,true,true}; check(!cycle(r,s,42));
      check(s.executions==0 && s.escalations==1 && s.commits==0); }
    { MockState s; MockRuntime r{s,true,false,true}; check(!cycle(r,s,42));
      check(s.executions==1 && s.reconciles==1 && s.commits==0); }
    { MockState s; MockRuntime r{s,true,true,false}; check(cycle(r,s,42));
      check(s.executions==1 && s.commits==0 && s.records==1); }
    { Field empty; check(!std::isfinite(sweep(empty,0.5,[](Id,Id){return true;}))); }
    { Field bad; bad[0].h[0]=std::numeric_limits<double>::quiet_NaN();
      check(!std::isfinite(sweep(bad,0.5,[](Id,Id){return true;}))); }
    std::cout<<"PASS: geometry, toroidal routing, recurrent sweep, link gating\n";
    std::cout<<"PASS: nominal, denied, verification failure, CTR rejection (mock adapters)\n";
}
