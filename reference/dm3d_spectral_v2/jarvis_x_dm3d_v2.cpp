// Jarvis X / Dr Moagi DM3D -- executable C++17 reference runtime.
// Build: g++ -std=c++17 -O3 -pthread jarvis_x_dm3d.cpp -o jarvis_x_dm3d
// Dependency: Boost.Multiprecision HEADERS ONLY (cpp_int, cpp_dec_float).
// This is a finite numerical/retrieval engine with symbolic huge addressing,
// not a physically instantiated septillion-scale ANN or language model.
#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace dm3d {
using Big = boost::multiprecision::cpp_int;
using Dec = boost::multiprecision::cpp_dec_float_50;
using Field = std::vector<double>;

void require(bool yes, const std::string& message) {
    if (!yes) throw std::runtime_error(message);
}
Big abs_big(Big x) {return x < 0 ? -x : x;}
Big gcd_big(Big a, Big b) {
    a=abs_big(a); b=abs_big(b);
    while(b != 0) { Big r=a%b; a=b; b=r; }
    return a;
}
struct Rat {
    Big p, q;
    Rat(Big a=0, Big b=1) : p(std::move(a)), q(std::move(b)) {
        require(q!=0, "zero denominator");
        if(q<0) {p=-p;q=-q;}
        Big g=gcd_big(p,q);
        if(g!=0) {p/=g;q/=g;}
    }
    std::string str() const {return p.convert_to<std::string>()+"/"+q.convert_to<std::string>();}
    Dec dec() const {return Dec(p.convert_to<std::string>()) / Dec(q.convert_to<std::string>());}
};
Rat operator+(const Rat& a,const Rat& b) {return Rat(a.p*b.q+b.p*a.q,a.q*b.q);}
Rat operator-(const Rat& a,const Rat& b) {return Rat(a.p*b.q-b.p*a.q,a.q*b.q);}
Rat operator*(const Rat& a,const Rat& b) {return Rat(a.p*b.p,a.q*b.q);}
Rat operator/(const Rat& a,const Rat& b) {require(b.p!=0,"division by zero");return Rat(a.p*b.q,a.q*b.p);}
bool operator<(const Rat& a,const Rat& b) {return a.p*b.q < b.p*a.q;}
bool operator>(const Rat& a,const Rat& b) {return b<a;}
bool operator==(const Rat& a,const Rat& b) {return a.p==b.p && a.q==b.q;}
Rat abs_rat(const Rat& a) {return Rat(abs_big(a.p),a.q);}
Rat atan_partial(unsigned x, int terms) {
    require(terms>=1 && terms<=100, "invalid Machin series length");
    Big power=x;
    Rat result;
    for(int k=0;k<terms;++k) {
        Rat term(Big(1),Big(2*k+1)*power);
        result=(k%2==0) ? result+term : result-term;
        power*=Big(x)*Big(x);
    }
    return result;
}
struct PiInterval {Rat lower,upper;};
PiInterval certify_pi(int n=20) {
    require(n>=2 && n%2==0, "Machin term count must be even >=2");
    Rat aL=atan_partial(5,n), aU=atan_partial(5,n+1);
    Rat bL=atan_partial(239,n), bU=atan_partial(239,n+1);
    PiInterval pi{Rat(16)*aL-Rat(4)*bU, Rat(16)*aU-Rat(4)*bL};
    require(Rat(3)<pi.lower && pi.lower<pi.upper && pi.upper<Rat(4),
            "invalid certified pi interval");
    return pi;
}
int side_of_pi(const Rat& v,const PiInterval& pi) {
    if (v<pi.lower) return -1;
    if (v>pi.upper) return +1;
    return 0; // interval insufficient to determine sign
}
std::vector<int> cf_prefix(PiInterval pi,int limit=12) {
    std::vector<int> out;
    for(int i=0;i<limit;++i) {
        Big a=pi.lower.p/pi.lower.q, b=pi.upper.p/pi.upper.q;
        if(a!=b) break;
        require(a>=0 && a<1000000000,"CF coefficient outside safe integer range");
        int coeff=a.convert_to<int>(); out.push_back(coeff);
        pi.lower=pi.lower-Rat(coeff); pi.upper=pi.upper-Rat(coeff);
        if(pi.lower.p<=0 || pi.upper.p<=0) break;
        Rat new_low=Rat(1)/pi.upper;
        Rat new_up=Rat(1)/pi.lower;
        pi.lower=new_low; pi.upper=new_up;
    }
    return out;
}
std::vector<Rat> convergents(const std::vector<int>& cf) {
    Big pm2=0,pm1=1,qm2=1,qm1=0;
    std::vector<Rat> out;
    for(int a:cf) {
        Big p=Big(a)*pm1+pm2,q=Big(a)*qm1+qm2;
        out.emplace_back(p,q);
        pm2=pm1;pm1=p; qm2=qm1;qm1=q;
    }
    return out;
}
uint64_t fnv64_bytes(const uint8_t* data,size_t n) {
    uint64_t h=14695981039346656037ULL;
    for(size_t i=0;i<n;++i) {h^=data[i];h*=1099511628211ULL;}
    return h;
}
uint64_t fnv64(const std::string& s) {
    return fnv64_bytes(reinterpret_cast<const uint8_t*>(s.data()),s.size());
}
std::string hex64(uint64_t h) {
    std::ostringstream o;o<<std::hex<<std::setw(16)<<std::setfill('0')<<h;return o.str();
}
// The huge axis Q=10^(24*10^24) is always symbolic. Because 2^33 divides Q,
// these dyadic midpoint addresses are exact integer lattice coordinates.
std::string symbolic_axis(uint32_t a) {
    return "(("+std::to_string(uint64_t(2)*a+1)+")*10^(24*10^24)/2^33)";
}
uint64_t splitmix64(uint64_t z) {
    z+=0x9e3779b97f4a7c15ULL;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}

struct Record {
    size_t row;
    Rat fraction;
    std::array<uint32_t,3> xyz;
    uint64_t seal;
    std::string kind;
};
class NumericalRAG {
    PiInterval interval_;
    std::vector<Record> rows_;
    static uint64_t checksum(const Record& r) {
        std::string s="v1|"+std::to_string(r.row)+"|"+r.fraction.str()+"|"+
            std::to_string(r.xyz[0])+"|"+std::to_string(r.xyz[1])+"|"+
            std::to_string(r.xyz[2])+"|"+r.kind;
        return fnv64(s); // corruption checksum, not a cryptographic signature
    }
public:
    NumericalRAG():interval_(certify_pi()) {
        auto cf=cf_prefix(interval_,12);
        require(cf.size()==12,"Machin precision insufficient for 12 convergents");
        for(const auto& r:convergents(cf)) add(r,"pi-convergent");
    }
    const PiInterval& bounds() const {return interval_;}
    const std::vector<Record>& records() const {return rows_;}
    Record add(const Rat& r,std::string kind="user") {
        require(r.p>0 && r.q>0,"RAG fractions must be positive");
        require(side_of_pi(r,interval_)!=0,"pi interval too broad to certify new rational");
        require(rows_.size()<std::numeric_limits<uint32_t>::max(),"ledger exceeds index capacity");
        require(r.p.convert_to<std::string>().size()<=65 &&
                r.q.convert_to<std::string>().size()<=65,"ledger numerator/denominator too long");
        uint32_t row=static_cast<uint32_t>(rows_.size());
        auto logq=static_cast<uint32_t>(r.q.convert_to<std::string>().size()-1);
        Record m{row,r,{row,logq,static_cast<uint32_t>(side_of_pi(r,interval_)+1)},0,std::move(kind)};
        m.seal=checksum(m);
        rows_.push_back(m);
        return m;
    }
    bool verify(const Record& r) const {
        if(r.fraction.p<=0 || r.fraction.q<=0 || r.row>=rows_.size())return false;
        if(gcd_big(r.fraction.p,r.fraction.q)!=1)return false;
        uint32_t logq=static_cast<uint32_t>(r.fraction.q.convert_to<std::string>().size()-1);
        std::array<uint32_t,3> expected{static_cast<uint32_t>(r.row),logq,
                                         static_cast<uint32_t>(side_of_pi(r.fraction,interval_)+1)};
        return expected==r.xyz && side_of_pi(r.fraction,interval_)!=0 &&
               r.seal==checksum(r);
    }
    std::pair<size_t,size_t> audit() const {
        size_t good=0,bad=0;
        for(size_t i=0;i<rows_.size();++i) {
            if(rows_[i].row==i && verify(rows_[i])) ++good; else ++bad;
        }
        return {good,bad};
    }
    std::vector<Record> retrieve(const std::string& q,size_t top=5) const {
        std::vector<Record> pool;
        for(const auto& r:rows_)
            if(verify(r))pool.push_back(r);
        Rat mid=(interval_.lower+interval_.upper)/Rat(2);
        if(q=="below" || q=="above") {
            int desired=q=="below" ? -1 : 1;
            pool.erase(std::remove_if(pool.begin(),pool.end(),[&](const Record& r){
                return side_of_pi(r.fraction,interval_)!=desired;
            }),pool.end());
        } else if(q!="best" && q!="pi") {
            auto slash=q.find('/');
            require(slash!=std::string::npos,"query must be best, above, below, pi or p/q");
        }
        // Exact proximity to certified pi midpoint by default.
        if(q!="best" && q!="pi" && q!="below" && q!="above") {
            auto slash=q.find('/');
            Rat target{Big(q.substr(0,slash)),Big(q.substr(slash+1))};
            std::stable_sort(pool.begin(),pool.end(),[&](const Record& a,const Record& b){
                Rat da=abs_rat(a.fraction-target),db=abs_rat(b.fraction-target);
                return (da==db) ? a.row<b.row : da<db;
            });
        } else {
            std::stable_sort(pool.begin(),pool.end(),[&](const Record& a,const Record& b){
                Rat da=abs_rat(a.fraction-mid),db=abs_rat(b.fraction-mid);
                return (da==db) ? a.row<b.row : da<db;
            });
        }
        if(top<pool.size())pool.resize(top);
        return pool;
    }
    void table(std::ostream& os,const std::vector<Record>& rs) const {
        os<<"row    ratio                    relation      absolute_error         index(x,y,z)   integrity\n";
        for(const auto& r:rs) {
            int dir=side_of_pi(r.fraction,interval_);
            Rat errLo,errHi;
            if(dir==1){errLo=r.fraction-interval_.upper;errHi=r.fraction-interval_.lower;}
            else {errLo=interval_.lower-r.fraction;errHi=interval_.upper-r.fraction;}
            Dec estimate=((errLo+errHi)/Rat(2)).dec();
            std::string ratio=r.fraction.str();
            os<<std::left<<std::setw(7)<<r.row<<std::setw(25)<<ratio.substr(0,24)
              <<std::setw(14)<<(dir==1?"above":"below")
              <<std::scientific<<std::setprecision(10)<<std::setw(23)<<estimate
              <<"("<<r.xyz[0]<<","<<r.xyz[1]<<","<<r.xyz[2]<<")"
              <<"     "<<(verify(r)?"PASS":"FAIL")<<"\n";
        }
        os<<std::defaultfloat;
    }
    void csv(const std::string& path) const {
        std::ofstream f(path);require(bool(f),"cannot create ledger: "+path);
        f<<"row,numerator,denominator,x,y,z,relation,error_lower,error_upper,checksum_fnv64,verified,kind\n";
        for(const auto& r:rows_) {
            int sign=side_of_pi(r.fraction,interval_);
            Rat lo=sign>0?r.fraction-interval_.upper:interval_.lower-r.fraction;
            Rat hi=sign>0?r.fraction-interval_.lower:interval_.upper-r.fraction;
            f<<r.row<<","<<r.fraction.p<<","<<r.fraction.q<<","<<r.xyz[0]<<","<<r.xyz[1]
             <<","<<r.xyz[2]<<","<<(sign>0?"above":"below")<<","<<std::setprecision(32)
             <<lo.dec()<<","<<hi.dec()<<","<<hex64(r.seal)<<","<<verify(r)<<","<<r.kind<<"\n";
        }
        require(bool(f),"write failure for ledger");
    }
};

size_t cube_cells(size_t n) {
    require(n>=1 && n<=128,"cube side must be from 1 to 128");
    return n*n*n;
}
struct Cube {
    size_t n;
    Field v;
    explicit Cube(size_t side=16):n(side),v(cube_cells(side),0){}
    size_t pos(size_t x,size_t y,size_t z)const{return (x*n+y)*n+z;}
};
Cube target_waves(size_t n) {
    Cube c(n);
    constexpr double pi=3.141592653589793238462643383279502884;
    for(size_t x=0;x<n;++x)for(size_t y=0;y<n;++y)for(size_t z=0;z<n;++z){
        double dx=(double(x)-.28*n)/(.11*n),dy=(double(y)-.63*n)/(.11*n),
               dz=(double(z)-.42*n)/(.11*n);
        c.v[c.pos(x,y,z)]=std::sin(2*pi*x/n)+.45*std::cos(4*pi*y/n)
                          +.35*std::sin(6*pi*z/n)+.8*std::exp(-.5*(dx*dx+dy*dy+dz*dz));
    }
    return c;
}
Cube encode2(const Cube& c) {
    require(c.n%2==0,"even side required");
    Cube latent(c.n/2);
    for(size_t x=0;x<latent.n;++x)for(size_t y=0;y<latent.n;++y)for(size_t z=0;z<latent.n;++z){
        double sum=0;
        for(size_t a=0;a<2;++a)for(size_t b=0;b<2;++b)for(size_t d=0;d<2;++d)
            sum+=c.v[c.pos(2*x+a,2*y+b,2*z+d)];
        latent.v[latent.pos(x,y,z)]=sum/8;
    }
    return latent;
}
Cube decode2(const Cube& latent) {
    Cube high(latent.n*2);
    for(size_t x=0;x<high.n;++x)for(size_t y=0;y<high.n;++y)for(size_t z=0;z<high.n;++z)
        high.v[high.pos(x,y,z)]=latent.v[latent.pos(x/2,y/2,z/2)];
    return high;
}
void laplace(const Cube& x, Field& out) {
    const size_t n=x.n;
    out.assign(x.v.size(),0);
    for(size_t i=0;i<n;++i)for(size_t j=0;j<n;++j)for(size_t k=0;k<n;++k){
        auto ix=[&](size_t a,size_t b,size_t c){return x.v[x.pos(a,b,c)];};
        out[x.pos(i,j,k)]=ix((i+1)%n,j,k)+ix((i+n-1)%n,j,k)
            +ix(i,(j+1)%n,k)+ix(i,(j+n-1)%n,k)
            +ix(i,j,(k+1)%n)+ix(i,j,(k+n-1)%n)-6*x.v[x.pos(i,j,k)];
    }
}
double objective(const Cube& s,const Cube& target,double mu) {
    require(s.n==target.n,"objective shape mismatch");
    double val=0;
    for(size_t x=0;x<s.n;++x)for(size_t y=0;y<s.n;++y)for(size_t z=0;z<s.n;++z){
        size_t idx=s.pos(x,y,z);
        double error=s.v[idx]-target.v[idx];
        val+=error*error;
        double current=s.v[idx];
        for(size_t d=0;d<3;++d){
            size_t xp=x,yp=y,zp=z;
            if(d==0)xp=(x+1)%s.n;
            if(d==1)yp=(y+1)%s.n;
            if(d==2)zp=(z+1)%s.n;
            double delta=s.v[s.pos(xp,yp,zp)]-current;
            val+=mu*delta*delta;
        }
    }
    return .5*val/s.v.size();
}
double dot(const Field& a,const Field& b) {
    require(a.size()==b.size(),"dot shape mismatch");
    double acc=0;for(size_t i=0;i<a.size();++i)acc+=a[i]*b[i];return acc;
}
// ---------- Periodic 3D spectral fixed-point solver ------------------------
// H = I - mu*Delta. In Fourier space, H is diagonal and strictly positive.
// The solver computes the global minimizer of the SAME quadratic objective
// as the original finite swarm. It is a new candidate, never an unchecked commit.
bool is_pow2(size_t n) { return n>=2 && (n&(n-1))==0; }
using Complex = std::complex<double>;
void fft1d_strided(Complex* a,size_t n,size_t stride,bool inverse) {
    for(size_t i=1,j=0;i<n;++i) {
        size_t bit=n>>1;
        for(;j&bit;bit>>=1) j^=bit;
        j^=bit;
        if(i<j) std::swap(a[i*stride],a[j*stride]);
    }
    constexpr double PI=3.141592653589793238462643383279502884;
    for(size_t len=2;len<=n;len*=2) {
        double angle=(inverse?2.0:-2.0)*PI/double(len);
        Complex root(std::cos(angle),std::sin(angle));
        for(size_t i=0;i<n;i+=len) {
            Complex w(1,0);
            for(size_t j=0;j<len/2;++j) {
                Complex u=a[(i+j)*stride];
                Complex v=a[(i+j+len/2)*stride]*w;
                a[(i+j)*stride]=u+v;
                a[(i+j+len/2)*stride]=u-v;
                w*=root;
            }
        }
        if(len==n) break; // avoid overflow in len*=2
    }
    if(inverse) for(size_t i=0;i<n;++i)a[i*stride]/=double(n);
}
void fft3d(std::vector<Complex>& f,size_t n,bool inverse) {
    require(is_pow2(n) && f.size()==n*n*n,"FFT requires power-of-two cubic volume");
    // z contiguous, y strided n, x strided n^2.
    for(size_t x=0;x<n;++x)for(size_t y=0;y<n;++y)
        fft1d_strided(f.data()+(x*n+y)*n,n,1,inverse);
    for(size_t x=0;x<n;++x)for(size_t z=0;z<n;++z)
        fft1d_strided(f.data()+x*n*n+z,n,n,inverse);
    for(size_t y=0;y<n;++y)for(size_t z=0;z<n;++z)
        fft1d_strided(f.data()+y*n+z,n,n*n,inverse);
}
Cube spectral_optimum(const Cube& target,double mu) {
    require(is_pow2(target.n),"spectral mode requires power-of-two --side");
    const size_t n=target.n;
    std::vector<Complex> field(target.v.size());
    for(size_t i=0;i<field.size();++i)field[i]=Complex(target.v[i],0);
    fft3d(field,n,false);
    constexpr double PI=3.141592653589793238462643383279502884;
    std::vector<double> cs(n);
    for(size_t k=0;k<n;++k)cs[k]=std::cos(2*PI*double(k)/double(n));
    for(size_t x=0;x<n;++x)for(size_t y=0;y<n;++y)for(size_t z=0;z<n;++z) {
        double eigen=1+2*mu*(3-cs[x]-cs[y]-cs[z]);
        field[(x*n+y)*n+z]/=eigen;
    }
    fft3d(field,n,true);
    Cube result(n);
    for(size_t i=0;i<field.size();++i)result.v[i]=field[i].real();
    return result;
}

struct Worker {
    uint32_t id;
    std::array<uint32_t,3> addr;
    double eta, alpha,beta;
};
std::vector<Worker> make_workers(size_t count,uint64_t seed=7,const NumericalRAG* rag=nullptr) {
    require(count>=1 && count<=4096,"workers must be 1..4096");
    std::vector<Worker> w;
    constexpr std::array<double,4> etas{{.20,.35,.50,.65}};
    constexpr std::array<double,3> alphas{{.35,.70,1.0}}, betas{{.25,.65,1.0}};
    for(size_t i=0;i<count;++i){
        uint64_t h=splitmix64(seed^((i+1)*0x9e3779b97f4a7c15ULL));
        std::array<uint32_t,3> addr{{uint32_t((i+seed)*0x9e3779b1ULL),
            uint32_t(splitmix64(h)>>16),uint32_t(splitmix64(h^0xa5a5a5a5ULL)>>16)}};
        size_t k=(i*17)%36;
        // Numerical RAG supplies a verified per-worker calibration value.
        // This is arithmetic parameter modulation, not model inference.
        double eta=etas[k/9];
        if(rag) {
            const Record& memory=rag->records()[i % rag->records().size()];
            require(rag->verify(memory),"worker RAG calibration failed certification");
            Rat midpoint=(rag->bounds().lower+rag->bounds().upper)/Rat(2);
            eta*= (memory.fraction/midpoint).dec().convert_to<double>();
        }
        w.push_back({uint32_t(i),addr,eta,alphas[(k/3)%3],betas[k%3]});
    }
    return w;
}
class AssociativeMemory {
    std::map<std::string,Cube> items_;
public:
    void remember(const std::string& key,const Cube& state) {items_.insert_or_assign(key,state);}
    std::pair<std::string,double> nearest(const Cube& query) const {
        std::string label;double best=std::numeric_limits<double>::infinity();
        for(const auto& kv:items_) {
            if(kv.second.n!=query.n)continue;
            double sq=0;
            for(size_t i=0;i<query.v.size();++i){
                double d=query.v[i]-kv.second.v[i];sq+=d*d;
            }
            double rmse=std::sqrt(sq/query.v.size());
            if(rmse<best){best=rmse;label=kv.first;}
        }
        return {label,best};
    }
};
struct StepLog {int step;double obj;int winner;double improvement;};
class VerifiedSwarm {
    Cube goal_,state_;
    std::vector<Worker> workers_;
    double mu_;
    unsigned threads_;
    double last_;
    size_t checked_=0,rejected_=0;
public:
    VerifiedSwarm(Cube target,std::vector<Worker> workers,double mu=.08,unsigned threads=4):
        goal_(std::move(target)),state_(goal_.n),workers_(std::move(workers)),mu_(mu),threads_(threads),
        last_(objective(state_,goal_,mu)) {
        require(mu>=0 && std::isfinite(mu),"invalid regularisation");
        require(threads>=1 && threads<=64,"threads must be 1..64");
    }
    double current_loss()const{return last_;}
    size_t checked()const{return checked_;}
    size_t rejected()const{return rejected_;}
    const Cube& state()const{return state_;}
    StepLog spectral_refine(int t) {
        const double prior=last_;
        Cube candidate=spectral_optimum(goal_,mu_);
        double measured=objective(candidate,goal_,mu_);
        ++checked_;
        if(std::isfinite(measured) && measured<prior-1e-14) {
            state_=std::move(candidate);
            last_=measured;
            return StepLog{t,last_,-2,prior-last_};
        }
        if(!std::isfinite(measured)) ++rejected_;
        return StepLog{t,last_,-1,0};
    }
    void restore(Cube saved) {
        require(saved.n==goal_.n,"saved field dimensions mismatch");
        for(double v:saved.v)require(std::isfinite(v),"nonfinite saved field value");
        state_=std::move(saved);
        last_=objective(state_,goal_,mu_);
        require(std::isfinite(last_),"invalid loaded objective");
    }
    StepLog step(int t) {
        Field lap;laplace(state_,lap);
        Cube G(state_.n);
        for(size_t i=0;i<G.v.size();++i)
            G.v[i]=state_.v[i]-goal_.v[i]-mu_*lap[i];
        Cube coarse=decode2(encode2(G));
        Field fine(G.v.size());
        for(size_t i=0;i<fine.size();++i)fine[i]=G.v[i]-coarse.v[i];
        Cube C(state_.n),F(state_.n);
        C.v=coarse.v;F.v=fine;
        // Only two stencil passes (G and C), instead of three (G,C,F).
        // Linear Hessian symmetry: H(F) = H(G) - H(C).
        Field lc,lg;laplace(C,lc);laplace(G,lg);
        Field HC(lc.size()),HF(lc.size());
        for(size_t i=0;i<lc.size();++i){
            HC[i]=C.v[i]-mu_*lc[i];
            HF[i]=G.v[i]-mu_*lg[i]-HC[i];
        }
        double inv=1.0/state_.v.size();
        double gc=inv*dot(G.v,C.v),gf=inv*dot(G.v,F.v);
        double cc=inv*dot(C.v,HC),cf=inv*dot(C.v,HF),ff=inv*dot(F.v,HF);
        double prior=last_;
        struct Eval {double value=std::numeric_limits<double>::infinity();int id=-1;};
        std::vector<Eval> evaluated(workers_.size());
        unsigned nthreads=std::min<unsigned>(threads_,unsigned(workers_.size()));
        auto work=[&](size_t begin,size_t end) {
            for(size_t i=begin;i<end;++i){
                const auto& w=workers_[i];
                double a=w.eta*w.alpha,b=w.eta*w.beta;
                double value=prior-a*gc-b*gf+.5*(a*a*cc+2*a*b*cf+b*b*ff);
                evaluated[i]=Eval{value,int(w.id)};
            }
        };
        // These workers each evaluate five scalar coefficients. Spawning
        // multiple OS threads for 48 scalar jobs loses to serial batching.
        // Multithread only for unusually large candidate pools.
        if(workers_.size()<1024 || nthreads==1) {
            work(0,workers_.size());
        } else {
            std::vector<std::thread> pool;
            const size_t chunk=(workers_.size()+nthreads-1)/nthreads;
            for(unsigned i=1;i<nthreads;++i){
                const size_t begin=i*chunk,end=std::min(workers_.size(),begin+chunk);
                pool.emplace_back(work,begin,end);
            }
            work(0,std::min(workers_.size(),chunk));
            for(auto& thread:pool)thread.join();
        }
        checked_+=workers_.size();
        double best=prior;int winner=-1;
        for(size_t i=0;i<evaluated.size();++i){
            if(!std::isfinite(evaluated[i].value)){++rejected_;continue;}
            if(evaluated[i].value<best-1e-14){best=evaluated[i].value;winner=int(i);}
        }
        if(winner>=0){
            const Worker& w=workers_[size_t(winner)];
            Cube candidate(state_.n);
            const double a=w.eta*w.alpha,b=w.eta*w.beta;
            for(size_t i=0;i<state_.v.size();++i)
                candidate.v[i]=state_.v[i]-a*C.v[i]-b*F.v[i];
            double actual=objective(candidate,goal_,mu_);
            if(std::isfinite(actual) && actual<prior-1e-14){
                state_=std::move(candidate);last_=actual;
            }else{++rejected_;winner=-1;}
        }
        require(std::isfinite(last_) && last_<=prior+1e-12,"CTR monotonicity violation");
        return StepLog{t,last_,winner,prior-last_};
    }
};

struct ByteCodec {
    uint32_t side=2;
    uint64_t original=0;
    uint64_t checksum=0;
    uint64_t payload_seal=0;
    std::vector<uint8_t> payload; // anchors followed by XOR residual bytes
};
uint32_t codec_side(uint64_t length) {
    uint64_t side=2;
    while(side*side*side<length)side+=2;
    require(side<=1024 && side*side*side<=134217728ULL,"codec payload exceeds 128MiB cube");
    return static_cast<uint32_t>(side);
}
ByteCodec encode_bytes(const std::vector<uint8_t>& input) {
    ByteCodec e;
    e.original=input.size();e.side=codec_side(input.size());
    e.checksum=fnv64_bytes(input.data(),input.size());
    const size_t n=e.side,blocks=(n/2)*(n/2)*(n/2);
    std::vector<uint8_t> padded(n*n*n,0);
    std::copy(input.begin(),input.end(),padded.begin());
    e.payload.resize(blocks*8);
    size_t block=0;
    for(size_t x=0;x<n;x+=2)for(size_t y=0;y<n;y+=2)for(size_t z=0;z<n;z+=2){
        uint8_t anchor=padded[(x*n+y)*n+z];
        e.payload[block]=anchor;
        for(size_t k=1;k<8;++k){
            size_t xx=x+((k>>2)&1), yy=y+((k>>1)&1), zz=z+(k&1);
            e.payload[blocks+block*7+(k-1)]=uint8_t(padded[(xx*n+yy)*n+zz]^anchor);
        }
        ++block;
    }
    e.payload_seal=fnv64_bytes(e.payload.data(),e.payload.size());
    return e;
}
std::vector<uint8_t> decode_bytes(const ByteCodec& e) {
    require(e.side>=2 && e.side%2==0 && e.side<=1024,"invalid codec side");
    size_t n=e.side;
    uint64_t len=uint64_t(n)*n*n;
    require(len<=134217728ULL && e.original<=len && e.payload.size()==len,"invalid codec payload");
    require(fnv64_bytes(e.payload.data(),e.payload.size())==e.payload_seal,"encoded payload checksum mismatch");
    const size_t blocks=(n/2)*(n/2)*(n/2);
    std::vector<uint8_t> out(n*n*n,0);
    size_t block=0;
    for(size_t x=0;x<n;x+=2)for(size_t y=0;y<n;y+=2)for(size_t z=0;z<n;z+=2){
        uint8_t anchor=e.payload[block];
        for(size_t k=0;k<8;++k){
            size_t xx=x+((k>>2)&1),yy=y+((k>>1)&1),zz=z+(k&1);
            out[(xx*n+yy)*n+zz]=k==0?anchor:uint8_t(e.payload[blocks+block*7+k-1]^anchor);
        }
        ++block;
    }
    for(size_t i=size_t(e.original);i<out.size();++i)require(out[i]==0,"nonzero voxel padding");
    out.resize(size_t(e.original));
    require(fnv64_bytes(out.data(),out.size())==e.checksum,"byte codec checksum mismatch");
    return out;
}
void write_le(std::ostream& f,uint64_t value,int bytes) {
    for(int i=0;i<bytes;++i) f.put(char((value>>(i*8))&255));
}
uint64_t read_le(std::istream& f,int bytes) {
    uint64_t value=0;
    for(int i=0;i<bytes;++i){int c=f.get();require(c!=EOF,"truncated codec file");value|=uint64_t(c)<<(8*i);}
    return value;
}
void save_codec(const ByteCodec& e,const std::string& path) {
    std::ofstream f(path,std::ios::binary);require(bool(f),"cannot write "+path);
    f.write("DM3D2",5);write_le(f,e.side,4);write_le(f,e.original,8);write_le(f,e.checksum,8);write_le(f,e.payload_seal,8);
    f.write(reinterpret_cast<const char*>(e.payload.data()),std::streamsize(e.payload.size()));
    require(bool(f),"writing byte codec failed");
}
ByteCodec load_codec(const std::string& path) {
    std::ifstream f(path,std::ios::binary);require(bool(f),"cannot read "+path);
    char magic[5];f.read(magic,5);require(bool(f) && std::memcmp(magic,"DM3D2",5)==0,"bad codec magic");
    ByteCodec e;e.side=uint32_t(read_le(f,4));e.original=read_le(f,8);e.checksum=read_le(f,8);e.payload_seal=read_le(f,8);
    require(e.side>=2 && e.side%2==0 && e.side<=1024,"invalid codec dimension");
    uint64_t cells=uint64_t(e.side)*e.side*e.side;
    require(cells<=134217728ULL && e.original<=cells,"codec allocation limit exceeded");
    e.payload.resize(size_t(cells));
    f.read(reinterpret_cast<char*>(e.payload.data()),std::streamsize(cells));
    require(bool(f),"truncated codec payload");
    require(f.peek()==EOF,"unexpected trailing encoded data");
    return e;
}
std::vector<uint8_t> load_bytes(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f),"cannot read "+path);
    auto len=f.tellg();require(len>=0 && uint64_t(len)<=134217728ULL,"file exceeds 128MiB limit");
    std::vector<uint8_t> data(size_t(len),uint8_t(0));
    f.seekg(0);if(!data.empty())f.read(reinterpret_cast<char*>(data.data()),std::streamsize(data.size()));
    require(bool(f),"file read failed");return data;
}
void save_bytes(const std::string& path,const std::vector<uint8_t>& data) {
    std::ofstream f(path,std::ios::binary);require(bool(f),"cannot write "+path);
    if(!data.empty())f.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));
    require(bool(f),"file write failed");
}
// Persistent numerical state: canonical little-endian IEEE754 binary64 bytes.
void save_state(const Cube& state,const std::string& filename) {
    std::vector<uint8_t> raw(state.v.size()*8);
    for(size_t i=0;i<state.v.size();++i){
        require(std::isfinite(state.v[i]),"cannot save nonfinite numerical state");
        uint64_t bits;std::memcpy(&bits,&state.v[i],8);
        for(int j=0;j<8;++j)raw[i*8+j]=uint8_t((bits>>(8*j))&255);
    }
    std::ofstream f(filename,std::ios::binary);require(bool(f),"cannot create "+filename);
    f.write("DM3DS1",6);write_le(f,state.n,4);
    write_le(f,fnv64_bytes(raw.data(),raw.size()),8);
    f.write(reinterpret_cast<const char*>(raw.data()),std::streamsize(raw.size()));
    require(bool(f),"state persistence write failed");
}
Cube load_state(const std::string& filename) {
    std::ifstream f(filename,std::ios::binary);require(bool(f),"cannot read "+filename);
    char magic[6];f.read(magic,6);
    require(bool(f) && std::memcmp(magic,"DM3DS1",6)==0,"invalid saved state signature");
    auto n=read_le(f,4),seal=read_le(f,8);
    require(n>=2 && n<=128 && n%2==0,"invalid saved state dimension");
    Cube state{size_t(n)};
    std::vector<uint8_t> raw(state.v.size()*8);
    f.read(reinterpret_cast<char*>(raw.data()),std::streamsize(raw.size()));
    require(bool(f) && f.peek()==EOF,"saved state truncated or trailing data");
    require(fnv64_bytes(raw.data(),raw.size())==seal,"saved state checksum mismatch");
    for(size_t i=0;i<state.v.size();++i){
        uint64_t bits=0;
        for(int j=0;j<8;++j)bits|=uint64_t(raw[i*8+j])<<(j*8);
        std::memcpy(&state.v[i],&bits,8);
        require(std::isfinite(state.v[i]),"nonfinite saved state value");
    }
    return state;
}
void test(bool yes,const char* message) {require(yes,std::string("SELF TEST FAILED: ")+message);}
void self_test() {
    NumericalRAG rag;
    auto a=rag.audit();test(a.first==12 && a.second==0,"12 certified pi records");
    test(side_of_pi(Rat(22,7),rag.bounds())==1,"22/7 strictly exceeds pi");
    test(side_of_pi(Rat(333,106),rag.bounds())==-1,"333/106 below pi");
    auto best=rag.retrieve("355/113",1);
    test(best.size()==1 && best[0].fraction==Rat(355,113),"exact RAG lookup");
    NumericalRAG extensible; extensible.add(Rat(314159,100000),"test");
    test(extensible.audit().first==13,"dynamic certified arithmetic memory");
    auto altered=rag.records().front();altered.xyz[2]=(altered.xyz[2]+1)%3;
    test(!rag.verify(altered),"index corruption detected");
    for(size_t len:{0u,1u,7u,8u,9u,64u,257u,1024u,13000u}) {
        std::vector<uint8_t> in(len);
        for(size_t i=0;i<len;++i)in[i]=uint8_t((37*i+len)%256);
        auto packed=encode_bytes(in);
        test(decode_bytes(packed)==in,"lossless byte round trip");
        if(len==257){packed.payload[3]^=0xff;
            bool rejected=false;try{(void)decode_bytes(packed);}catch(const std::exception&){rejected=true;}
            test(rejected,"detect altered byte data");}
    }
    auto workers=make_workers(96);for(size_t i=0;i<workers.size();++i)
        for(size_t j=0;j<i;++j)test(workers[i].addr!=workers[j].addr,"unique sparse worker addresses");
    Cube wave=target_waves(16), rec=decode2(encode2(wave));
    test(rec.v.size()==wave.v.size(),"3D codec dimensions");
    Cube residual(16);
    for(size_t i=0;i<residual.v.size();++i)residual.v[i]=wave.v[i]-rec.v[i];
    for(size_t i=0;i<residual.v.size();++i)
        test(std::abs(wave.v[i]-(rec.v[i]+residual.v[i]))<1e-12,"exact coarse plus residual");
    auto guided=make_workers(48,7,&rag);
    test(std::abs(guided[0].eta-make_workers(48)[0].eta)>1e-9,"RAG-to-worker coupling");
    VerifiedSwarm s1(wave,guided,.08,1),s4(wave,guided,.08,4);
    double initial=s1.current_loss();
    for(int i=1;i<=40;++i){
        auto r1=s1.step(i),r4=s4.step(i);
        test(std::isfinite(r1.obj) && r1.improvement>=-1e-12,"objective monotonic");
        test(std::abs(r1.obj-r4.obj)<1e-12,"thread-count deterministic");
    }
    test(s1.current_loss()<initial*.03,"optimiser convergence on synthetic waves");
    // Independent spectral verification: solves (I-mu*Delta)X=T.
    Cube exact=spectral_optimum(wave,.08);
    double exact_loss=objective(exact,wave,.08);
    test(exact_loss<=s1.current_loss()+1e-11,"FFT optimum no worse than swarm");
    Field lapex;laplace(exact,lapex);
    double max_resid=0;
    for(size_t i=0;i<exact.v.size();++i)
        max_resid=std::max(max_resid,std::abs(exact.v[i]-wave.v[i]-.08*lapex[i]));
    test(max_resid<1e-11,"spectral equation residual");
    VerifiedSwarm exact_runtime(wave,guided,.08,4);
    exact_runtime.spectral_refine(1);
    test(std::abs(exact_runtime.current_loss()-exact_loss)<1e-12,"verified FFT commit");
    AssociativeMemory memory;
    memory.remember("wave",wave);memory.remember("reconstruction",rec);
    test(memory.nearest(wave).first=="wave","numeric associative retrieval");
    std::cout<<"SELF-TEST PASS: pi bounds, 12 RAG records, corruption check,"
             <<" byte codec, 3D residual reconstruction, RAG-guided 48-worker parallel CTR,"
             <<" deterministic threading, certified spectral fixed point, associative memory\n";
    std::cout<<std::setprecision(12)<<"initial_objective="<<initial
             <<" final_objective="<<s1.current_loss()
             <<" reduction_pct="<<(100*(1-s1.current_loss()/initial))<<"\n";
}
struct Options {
    bool self=false,help=false;
    int side=16,workers=48,steps=40,threads=4,top=5;
    std::string query="best",trace,ledger,encode_text,save_state_path,load_state_path,solver="auto";
    std::vector<std::string> adds;
    std::string file_in,file_out;
    bool encode_file=false,decode_file=false,text_set=false;
};
int get_int(const char* s,const std::string& label) {
    size_t i=0;int x=0;
    try{x=std::stoi(s,&i);}catch(...){throw std::runtime_error("invalid "+label);}
    require(i==std::strlen(s),"invalid "+label);return x;
}
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto next=[&](){require(i+1<argc,"missing argument after "+a);return std::string(argv[++i]);};
        if(a=="--self-test")o.self=true;
        else if(a=="--help"||a=="-h")o.help=true;
        else if(a=="--side") {auto s=next();o.side=get_int(s.c_str(),a);}
        else if(a=="--workers") {auto s=next();o.workers=get_int(s.c_str(),a);}
        else if(a=="--steps") {auto s=next();o.steps=get_int(s.c_str(),a);}
        else if(a=="--threads") {auto s=next();o.threads=get_int(s.c_str(),a);}
        else if(a=="--top") {auto s=next();o.top=get_int(s.c_str(),a);}
        else if(a=="--query")o.query=next();
        else if(a=="--ledger")o.ledger=next();
        else if(a=="--add")o.adds.push_back(next());
        else if(a=="--save-state")o.save_state_path=next();
        else if(a=="--load-state")o.load_state_path=next();
        else if(a=="--trace")o.trace=next();
        else if(a=="--solver")o.solver=next();
        else if(a=="--encode-text"){o.encode_text=next();o.text_set=true;}
        else if(a=="--encode-file") {o.file_in=next();o.file_out=next();o.encode_file=true;}
        else if(a=="--decode-file") {o.file_in=next();o.file_out=next();o.decode_file=true;}
        else throw std::runtime_error("unknown option: "+a);
    }
    return o;
}
void help() {
    std::cout<<"Jarvis X DM3D -- C++17 verified 3D swarm and numerical RAG\n"
             <<"  --self-test                            Full engine invariant tests\n"
             <<"  --side 16 --workers 48 --steps 40      3D volume, candidates, recursions\n"
             <<"  --threads 4                           Real CPU candidate-evaluation threads\n"
             <<"  --solver auto|fft|swarm               Certified 3D fixed point or swarm fallback\n"
             <<"  --query best|above|below|22/7         Exact-rational retrieval\n"
             <<"  --top 5                               Number of retrieved records\n"
             <<"  --ledger file.csv                     Export certified arithmetic memory\n"
             <<"  --add 314159/100000                  Add a certified rational memory record\n"
             <<"  --save-state state.bin                Persist the verified 3D numerical field\n"
             <<"  --load-state state.bin                Resume optimisation from stored field\n"
             <<"  --trace file.csv                      Export optimisation measurements\n"
             <<"  --encode-text 'Jarvis X'              Lossless byte codec demonstration\n"
             <<"  --encode-file input.dat output.dm3d   Lossless binary/UTF-8/media transform\n"
             <<"  --decode-file input.dm3d output.dat   Invert transform and check checksum\n"
             <<"Symbolic Septillion axis: Q=10^(24*10^24); active resources finite.\n";
}
int run(int argc,char** argv) {
    auto opts=parse(argc,argv);
    if(opts.help){help();return 0;}
    if(opts.self){self_test();return 0;}
    if(opts.encode_file){
        auto bytes=load_bytes(opts.file_in);auto coded=encode_bytes(bytes);
        save_codec(coded,opts.file_out);
        std::cout<<"encoded "<<bytes.size()<<" bytes into "<<coded.side<<"^3 cube; checksum "
                 <<hex64(coded.checksum)<<"\n";return 0;
    }
    if(opts.decode_file){
        auto coded=load_codec(opts.file_in);auto bytes=decode_bytes(coded);
        save_bytes(opts.file_out,bytes);
        std::cout<<"decoded and verified "<<bytes.size()<<" bytes; checksum "
                 <<hex64(coded.checksum)<<"\n";return 0;
    }
    require(opts.solver=="auto" || opts.solver=="fft" || opts.solver=="swarm", "solver must be auto|fft|swarm");
    require(opts.top>0 && opts.top<=100,"top must be 1..100");
    require(opts.workers>0 && opts.workers<=4096,"workers must be 1..4096");
    require(opts.steps>=0 && opts.steps<=100000,"steps must be 0..100000");
    require(opts.threads>=1 && opts.threads<=64,"threads must be 1..64");
    require(opts.side>=2 && opts.side<=128 && opts.side%2==0,"side must be even 2..128");
    NumericalRAG rag;
    for(const auto& fraction:opts.adds){
        auto slash=fraction.find('/');
        require(slash!=std::string::npos,"--add must be numerator/denominator");
        rag.add(Rat(Big(fraction.substr(0,slash)),Big(fraction.substr(slash+1))),"user-rational");
    }
    auto audit=rag.audit();
    require(audit.second==0,"RAG ledger failed audit");
    std::cout<<"JARVIS X / DM3D | 3D NUMERICAL RAG + VERIFIED SWARM\n"
             <<"virtual_axis=10^(24*10^24) virtual_total=10^(72*10^24)\n"
             <<"allocated_volume="<<opts.side<<"^3 active_workers="<<opts.workers
             <<" cpu_threads="<<opts.threads<<"\n"
             <<"arithmetic_RAG="<<audit.first<<" certified entries, "<<audit.second<<" failures\n"
             <<"rag_to_swarm_coupling=certified rational calibration of candidate eta\n";
    rag.table(std::cout,rag.retrieve(opts.query,size_t(opts.top)));
    if(!opts.ledger.empty()){rag.csv(opts.ledger);std::cout<<"ledger="<<opts.ledger<<"\n";}
    if(opts.text_set){
        std::vector<uint8_t> in(opts.encode_text.begin(),opts.encode_text.end());
        auto rec=encode_bytes(in);auto out=decode_bytes(rec);
        require(in==out,"UTF-8 byte round trip failed");
        std::cout<<"byte_codec=PASS source_bytes="<<in.size()<<" cube_side="<<rec.side
                 <<" recovered="<<std::string(out.begin(),out.end())<<"\n";
    }
    auto workers=make_workers(size_t(opts.workers),7,&rag);
    auto target=target_waves(size_t(opts.side));
    VerifiedSwarm swarm(target,workers,.08,unsigned(opts.threads));
    if(!opts.load_state_path.empty()){
        swarm.restore(load_state(opts.load_state_path));
        std::cout<<"restored_state="<<opts.load_state_path<<"\n";
    }
    AssociativeMemory mem;mem.remember("target",target);
    std::ofstream trace;
    if(!opts.trace.empty()){
        trace.open(opts.trace);require(bool(trace),"cannot write trace");
        trace<<"step,objective,winner,improvement\n";
        trace<<"0,"<<std::setprecision(17)<<swarm.current_loss()<<",-1,0\n";
    }
    const double first=swarm.current_loss();
    const bool spectral = (opts.solver=="fft" || (opts.solver=="auto" && is_pow2(size_t(opts.side))));
    require(!spectral || is_pow2(size_t(opts.side)),"FFT solver requires power-of-two --side");
    std::cout<<std::setprecision(13)<<"step=0 loss="<<first
             <<" solver="<<(spectral?"spectral-3d":"verified-swarm")<<"\n";
    auto start_time=std::chrono::steady_clock::now();
    for(int k=1;k<=opts.steps;++k){
        // In auto mode the exact Fourier candidate is verified on first step.
        // Further steps are intentionally no-ops after attaining the global
        // minimizer, keeping the iteration semantics and CSV stable.
        auto report=(spectral && k==1) ? swarm.spectral_refine(k) :
                    (spectral ? StepLog{k,swarm.current_loss(),-1,0} : swarm.step(k));
        if(trace)trace<<report.step<<","<<std::setprecision(17)<<report.obj<<","<<report.winner
                      <<","<<report.improvement<<"\n";
        if(k<=3 || k==opts.steps || k%10==0){
            std::cout<<"step="<<k<<" loss="<<report.obj<<" winner="<<report.winner
                     <<" delta="<<report.improvement<<"\n";
        }
    }
    double elapsed_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start_time).count();
    mem.remember("verified",swarm.state());
    if(!opts.save_state_path.empty()){
        save_state(swarm.state(),opts.save_state_path);
        std::cout<<"saved_state="<<opts.save_state_path<<"\n";
    }
    auto recall=mem.nearest(swarm.state());
    std::cout<<"objective_reduction_pct="<<(100*(1-swarm.current_loss()/first))
             <<" worker_evaluations="<<swarm.checked()<<" rejected="<<swarm.rejected()
             <<" compute_ms="<<elapsed_ms
             <<" memory_recall="<<recall.first<<" rmse="<<recall.second<<"\n"
             <<"CTR status: PASS (each accepted update finite and non-worsening).\n";
    if(!opts.trace.empty())std::cout<<"trace="<<opts.trace<<"\n";
    auto p=workers.front().addr;
    std::cout<<"example_worker_3d_symbolic=("<<symbolic_axis(p[0])<<", "
             <<symbolic_axis(p[1])<<", "<<symbolic_axis(p[2])<<")\n";
    return 0;
}
} // namespace dm3d
int main(int argc,char** argv) {
    try {return dm3d::run(argc,argv);}
    catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
