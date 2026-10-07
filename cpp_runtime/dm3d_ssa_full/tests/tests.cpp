#include "dm3d_ssa.hpp"
#include <iostream>
#include <functional>
using namespace dm3d;
static int passed=0;
void check(bool yes,const char* what){if(!yes)throw std::runtime_error(what);++passed;}
void rejected(const std::function<void()>& op){bool fail=false;try{op();}catch(const std::exception&){fail=true;}check(fail,"invalid graph incorrectly accepted");}
int main(){try{
    auto X=make_input(8);check(X.data.size()==512,"3D count");
    auto g=target_graph(8);auto val=execute(g,X);check(val.shape==X.shape,"shape closure");
    auto shapes=validate(g,X.shape);check(shapes[3]==Shape::tensor(4,4,4),"pool shape");check(shapes[4]==Shape::tensor(8,8,8),"up shape");check(shapes[6]==Shape::scalar(),"reduce shape");
    check(max_error(val,execute(g,X))==0,"determinism");
    Value v{Shape::vector(3),{1,2,3}};Graph dg{{{Op::Input},{Op::Dot,0,0}},1};auto dot=execute(dg,v);check(dot.shape==Shape::scalar()&&dot.data[0]==14,"vector dot");
    Value m{Shape::matrix(2,2),{1,2,3,4}};Graph mm{{{Op::Input},{Op::Matmul,0,0}},1};auto m2=execute(mm,m);check(m2.data==std::vector<float>({7,10,15,22}),"matrix matmul");
    Graph reshape{{{Op::Input},{Op::Reshape,0,-1,0,0,Shape::matrix(1,3)}},1};auto rv=execute(reshape,v);check(rv.shape==Shape::matrix(1,3),"vector to matrix");
    Graph broadcast{{{Op::Input},{Op::Mean,0},{Op::Broadcast,1,-1,0,0,Shape::matrix(2,3)}},2};auto bv=execute(broadcast,v);check(bv.data==std::vector<float>(6,2.f),"broadcast scalar");
    Graph bad{{{Op::Input},{Op::Matmul,0,0}},1};rejected([&](){execute(bad,v);});
    Graph future{{{Op::Input},{Op::Add,0,2}},1};rejected([&](){execute(future,v);});
    Graph mismatch{{{Op::Input},{Op::Mean,0},{Op::Add,0,1}},2};rejected([&](){execute(mismatch,v);});
    Graph pool{{{Op::Input},{Op::Pool2,0}},1};rejected([&](){execute(pool,make_input(9));});
    Graph invalidReshape{{{Op::Input},{Op::Reshape,0,-1,0,0,Shape::vector(4)}},1};rejected([&](){execute(invalidReshape,v);});
    auto champion=make_graph({4,3,3,false},8);check(max_error(execute(champion,X),val)==0,"reference genotype");
    check(std::isfinite(checksum(val)),"finite checksum");
    std::cout<<"PASS "<<passed<<" unit assertions\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
