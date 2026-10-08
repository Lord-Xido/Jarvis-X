#include "dm3d/fiber264.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
using namespace dm3d::fiber;

static int count=0;
static void check(bool value,const char* label) {
    if(!value)throw std::runtime_error(label);
    ++count;
}
static void must_throw(const std::function<void()>& f,const char* label) {
    bool threw=false;try{f();}catch(const std::exception&){threw=true;}
    check(threw,label);
}
int main() {
    try {
        check(BASE_SIZE==264,"base group count");
        check(STATE_COUNT==18399744,"full state cardinality");
        check(LATENT_DIM==512,"latent adapter dimension");
        check(collapse(State{})==V_STAR && collapse(V_STAR)==V_STAR,"collapse fixed point");
        check(from_index(0)==V_STAR,"zero coordinate origin");
        const auto last=from_index(STATE_COUNT-1);
        check(to_index(last)==STATE_COUNT-1,"last element round trip");
        check(to_index(from_xyz({263,263,263}))==STATE_COUNT-1,"XYZ cardinal endpoint");
        const auto point=from_xyz({132,55,99});
        const auto xyz=to_xyz(point);
        check(xyz.x==132 && xyz.y==55 && xyz.z==99,"XYZ bijection");
        check(from_index(to_index(point))==point,"index bijection");
        must_throw([]{(void)from_index(STATE_COUNT);},"index bounds");
        must_throw([]{(void)from_xyz({264,0,0});},"XYZ bounds");
        State bad;bad.coord[0]=11;
        must_throw([&]{(void)add(bad,V_STAR);},"invalid digit rejected");

        const auto g=control_shift(42,17);
        const auto x=from_index(752001);
        check(add(add(x,g),inverse(g))==x,"group inverse");
        check(add(x,g)==add(g,x),"abelian law");
        State z;z.coord={1,1,1,0,0,0,0,0,0};
        check(translation_order(z)==132,"maximal translation order");
        check(translation_order(V_STAR)==1,"identity order");
        std::unordered_set<std::uint64_t> orbit;
        State y=V_STAR;
        for(int i=0;i<132;++i){
            orbit.insert(to_index(y));
            y=add(y,z);
        }
        check(y==V_STAR && orbit.size()==132,"132-cycle exactly");
        check(one_cycle_permutation(STATE_COUNT-1)==0
              && one_cycle_permutation(0)==1,"independent giant permutation");
        must_throw([]{(void)one_cycle_permutation(STATE_COUNT);},"permutation bounds");

        bool dynamic=true;
        auto state=x;
        for(int n=0;n<1000;++n){
            const auto shift=control_shift(std::uint64_t(n),std::uint64_t(n*37));
            dynamic=dynamic && ctr_invariant(state,shift);
            state=add(state,shift);
        }
        check(dynamic,"1000 dynamic translations preserve the exact collapsed output");
        check(psi(0,0,0,0)==psi(100,12,-30,5.7),"spatial and temporal constancy");
        check(psi(0,0,0,0)==V_STAR,"constant field value");
        check(collapse(x)==collapse(last) && x!=last,"unobservable hidden distinction");

        const auto obs=readout(last);
        check(decode_readout(obs)==last,"injective observable channel");
        check(readout(x)!=readout(last),"nonconstant readout distinguishes states");
        const auto target=embed512(last);
        check(decode512(target)==last,"512D adapter roundtrip");
        check(std::abs(target[0]-obs[0])<1e-15,"observable preserved in embedding");
        Latent512 zero{};
        const double initial_mse=mse(zero,target);
        const auto result=refine(zero,target,0.73,100,1e-8);
        check(result.converged && result.iterations>1,"anchored contraction converges");
        check(result.final_mse<initial_mse && result.final_mse<1e-12,"measured target MSE decreases");
        check(result.final_delta<=1e-8,"stopping criterion honored");
        check(decode512(result.latent)==last,"decoded hidden state recovered via observable channel");
        const auto immediate=refine(zero,target,0.0,1,1e-10);
        check(immediate.converged && max_residual(immediate.latent,target)==0.0,"lambda zero exact update");
        must_throw([&]{(void)refine(zero,target,1.0);},"noncontractive lambda rejected");
        must_throw([&]{(void)refine(zero,target,-0.5);},"negative lambda rejected");
        must_throw([&]{(void)refine(zero,target,0.7,0);},"invalid iteration count rejected");
        std::cout<<"PASS "<<count<<" exact/group/readout/refinement checks\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"FAIL "<<e.what()<<"\n";
        return 1;
    }
}
