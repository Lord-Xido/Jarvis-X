#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dm3d {

enum class Kind { Scalar, Vector, Matrix, Tensor3D };
struct Shape {
    Kind kind{Kind::Scalar};
    std::array<int,3> dim{1,1,1};
    static Shape scalar() { return {}; }
    static Shape vector(int n) { return {Kind::Vector,{n,1,1}}; }
    static Shape matrix(int m,int n) { return {Kind::Matrix,{m,n,1}}; }
    static Shape tensor(int d,int h,int w) { return {Kind::Tensor3D,{d,h,w}}; }
    bool operator==(const Shape& b) const { return kind==b.kind && dim==b.dim; }
    bool operator!=(const Shape& b) const { return !(*this==b); }
    size_t count() const {
        size_t n=1;
        int rank=kind==Kind::Scalar?0:kind==Kind::Vector?1:kind==Kind::Matrix?2:3;
        for(int i=0;i<rank;++i){if(dim[i]<1||dim[i]>128)throw std::invalid_argument("invalid shape extent");n*=size_t(dim[i]);}
        if(n>2000000)throw std::invalid_argument("volume too large");
        return n;
    }
    std::string str() const {
        std::ostringstream s;
        s<<(kind==Kind::Scalar?"Scalar":kind==Kind::Vector?"Vector":kind==Kind::Matrix?"Matrix":"Tensor3D");
        if(kind!=Kind::Scalar){s<<"["<<dim[0];if(kind==Kind::Matrix||kind==Kind::Tensor3D)s<<","<<dim[1];if(kind==Kind::Tensor3D)s<<","<<dim[2];s<<"]";}
        return s.str();
    }
};
struct Value { Shape shape; std::vector<float> data; };
inline Value filled(Shape s,float v){return {s,std::vector<float>(s.count(),v)};}
inline void check_value(const Value& v){if(v.data.size()!=v.shape.count())throw std::invalid_argument("value shape mismatch");for(float x:v.data)if(!std::isfinite(x))throw std::runtime_error("non-finite tensor");}

enum class Op { Input, Constant, Copy, Add, Multiply, Relu, Conv3D, Pool2, Up2, Mean, Broadcast, Matmul, Dot, Reshape, Mix };
inline const char* name(Op op){
    switch(op){case Op::Input:return "Input";case Op::Constant:return "Constant";case Op::Copy:return "Copy";case Op::Add:return "Add";case Op::Multiply:return "Multiply";case Op::Relu:return "Relu";case Op::Conv3D:return "Conv3D";case Op::Pool2:return "Pool2";case Op::Up2:return "Up2";case Op::Mean:return "Mean";case Op::Broadcast:return "Broadcast";case Op::Matmul:return "Matmul";case Op::Dot:return "Dot";case Op::Reshape:return "Reshape";case Op::Mix:return "Mix";}return "?";
}
struct Node { Op op{Op::Input};int a{-1},b{-1};float p{0},q{0};Shape requested{}; };
struct Graph { std::vector<Node> nodes;int output{-1}; };
inline int pos(int i,int n){return (i+n)%n;}
inline Shape infer_node(const Node& n,const std::vector<Shape>& shapes,int id,const Shape& inputShape){
    auto need=[&](int ref)->const Shape& {if(ref<0||ref>=id)throw std::invalid_argument("invalid SSA input index");return shapes[size_t(ref)];};
    auto identical=[&](){auto a=need(n.a),b=need(n.b);if(a!=b)throw std::invalid_argument("binary shape mismatch");return a;};
    switch(n.op){
    case Op::Input: if(id!=0||n.a!=-1||n.b!=-1)throw std::invalid_argument("input must be node zero");return inputShape;
    case Op::Constant: if(n.a!=-1||n.b!=-1||!std::isfinite(n.p))throw std::invalid_argument("invalid constant");(void)n.requested.count();return n.requested;
    case Op::Copy:case Op::Relu:case Op::Mean: {
        auto a=need(n.a);if(n.b!=-1)throw std::invalid_argument("unexpected second operand");return n.op==Op::Mean?Shape::scalar():a;
    }
    case Op::Add:case Op::Multiply:case Op::Mix: return identical();
    case Op::Conv3D: {auto a=need(n.a);if(a.kind!=Kind::Tensor3D)throw std::invalid_argument("Conv3D requires Tensor3D");return a;}
    case Op::Pool2: {auto a=need(n.a);if(a.kind!=Kind::Tensor3D||a.dim[0]%2||a.dim[1]%2||a.dim[2]%2||a.dim[0]<2||a.dim[1]<2||a.dim[2]<2)throw std::invalid_argument("Pool2 requires even 3D extents");return Shape::tensor(a.dim[0]/2,a.dim[1]/2,a.dim[2]/2);}
    case Op::Up2: {auto a=need(n.a);if(a.kind!=Kind::Tensor3D)throw std::invalid_argument("Up2 requires Tensor3D");return Shape::tensor(a.dim[0]*2,a.dim[1]*2,a.dim[2]*2);}
    case Op::Broadcast: {if(need(n.a).kind!=Kind::Scalar)throw std::invalid_argument("Broadcast requires scalar");(void)n.requested.count();return n.requested;}
    case Op::Matmul: {auto a=need(n.a),b=need(n.b);if(a.kind!=Kind::Matrix||b.kind!=Kind::Matrix||a.dim[1]!=b.dim[0])throw std::invalid_argument("Matmul inner dimensions mismatch");return Shape::matrix(a.dim[0],b.dim[1]);}
    case Op::Dot: {auto a=need(n.a),b=need(n.b);if(a.kind!=Kind::Vector||b.kind!=Kind::Vector||a!=b)throw std::invalid_argument("Dot vector dimension mismatch");return Shape::scalar();}
    case Op::Reshape: {auto a=need(n.a);if(a.count()!=n.requested.count())throw std::invalid_argument("reshape element count mismatch");return n.requested;}
    }
    throw std::invalid_argument("unknown opcode");
}
inline std::vector<Shape> validate(const Graph& g,const Shape& input){
    if(g.nodes.empty()||g.nodes.size()>96||g.output<0||size_t(g.output)>=g.nodes.size())throw std::invalid_argument("invalid graph size/output");
    std::vector<Shape> shapes;shapes.reserve(g.nodes.size());
    for(size_t i=0;i<g.nodes.size();++i){
        const auto& n=g.nodes[i];if(!std::isfinite(n.p)||!std::isfinite(n.q))throw std::invalid_argument("non-finite parameter");
        shapes.push_back(infer_node(n,shapes,int(i),input));
    }
    return shapes;
}
inline Value run_node(const Node& n,const std::vector<Value>& vs,const Shape& shape,const Value& input){
    auto get=[&](int i)->const Value&{return vs.at(size_t(i));};
    if(n.op==Op::Input)return input;
    if(n.op==Op::Constant)return filled(shape,n.p);
    const Value& a=get(n.a);
    Value out=filled(shape,0);
    if(n.op==Op::Copy||n.op==Op::Reshape){out.data=a.data;return out;}
    if(n.op==Op::Mean){double s=0;for(float x:a.data)s+=x;out.data[0]=float(s/a.data.size());return out;}
    if(n.op==Op::Broadcast){std::fill(out.data.begin(),out.data.end(),a.data[0]);return out;}
    if(n.op==Op::Relu){for(size_t i=0;i<out.data.size();++i)out.data[i]=std::max(0.f,a.data[i]);return out;}
    if(n.op==Op::Add||n.op==Op::Multiply||n.op==Op::Mix){const Value& b=get(n.b);for(size_t i=0;i<out.data.size();++i){float x=a.data[i],y=b.data[i];out.data[i]=n.op==Op::Add?x+y:n.op==Op::Multiply?x*y:(1-n.p)*x+n.p*y;}return out;}
    if(n.op==Op::Dot){const auto& b=get(n.b);double s=0;for(size_t i=0;i<a.data.size();++i)s+=double(a.data[i])*b.data[i];out.data[0]=float(s);return out;}
    if(n.op==Op::Matmul){const auto& b=get(n.b);int m=a.shape.dim[0],k=a.shape.dim[1],p=b.shape.dim[1];for(int i=0;i<m;++i)for(int j=0;j<p;++j){double s=0;for(int h=0;h<k;++h)s+=double(a.data[size_t(i)*k+h])*b.data[size_t(h)*p+j];out.data[size_t(i)*p+j]=float(s);}return out;}
    const int D=a.shape.dim[0],H=a.shape.dim[1],W=a.shape.dim[2];
    auto idx=[&](int d,int h,int w)->size_t{return(size_t(d)*H+h)*W+w;};
    if(n.op==Op::Conv3D){
        for(int z=0;z<D;++z)for(int y=0;y<H;++y)for(int x=0;x<W;++x){
            float center=a.data[idx(z,y,x)];float sum=a.data[idx(pos(z-1,D),y,x)]+a.data[idx(pos(z+1,D),y,x)]+a.data[idx(z,pos(y-1,H),x)]+a.data[idx(z,pos(y+1,H),x)]+a.data[idx(z,y,pos(x-1,W))]+a.data[idx(z,y,pos(x+1,W))];
            out.data[idx(z,y,x)]=n.p*center+n.q*(sum/6.f);
        }return out;
    }
    if(n.op==Op::Pool2){int h2=H/2,w2=W/2;for(int z=0;z<D/2;++z)for(int y=0;y<h2;++y)for(int x=0;x<w2;++x){float sum=0;for(int dz=0;dz<2;++dz)for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)sum+=a.data[idx(2*z+dz,2*y+dy,2*x+dx)];out.data[(size_t(z)*h2+y)*w2+x]=sum/8.f;}return out;}
    if(n.op==Op::Up2){int h2=H*2,w2=W*2;for(int z=0;z<D*2;++z)for(int y=0;y<h2;++y)for(int x=0;x<w2;++x)out.data[(size_t(z)*h2+y)*w2+x]=a.data[idx(z/2,y/2,x/2)];return out;}
    throw std::invalid_argument("unsupported op");
}
inline Value execute(const Graph& g,const Value& input){check_value(input);auto shapes=validate(g,input.shape);std::vector<Value> values;values.reserve(g.nodes.size());for(size_t i=0;i<g.nodes.size();++i){values.push_back(run_node(g.nodes[i],values,shapes[i],input));check_value(values.back());}return values[size_t(g.output)];}
inline Value make_input(int N){Shape sh=Shape::tensor(N,N,N);Value v=filled(sh,0);for(int z=0;z<N;++z)for(int y=0;y<N;++y)for(int x=0;x<N;++x){double pi=3.14159265358979323846;v.data[(size_t(z)*N+y)*N+x]=float(.52*std::sin(2*pi*x/N)+.25*std::cos(4*pi*y/N)+.13*std::sin(6*pi*z/N+pi*x/N));}return v;}
inline double max_error(const Value& a,const Value& b){if(a.shape!=b.shape)return std::numeric_limits<double>::infinity();double e=0;for(size_t i=0;i<a.data.size();++i)e=std::max(e,std::abs(double(a.data[i])-double(b.data[i])));return e;}
inline double mse(const Value& a,const Value& b){if(a.shape!=b.shape)return std::numeric_limits<double>::infinity();double s=0;for(size_t i=0;i<a.data.size();++i){double e=double(a.data[i])-b.data[i];s+=e*e;}return s/a.data.size();}
inline double checksum(const Value& a){double s=0;for(float x:a.data)s+=x;return s;}
inline uint64_t fingerprint(const Value& a){uint64_t h=14695981039346656037ull;for(float x:a.data){uint32_t u;static_assert(sizeof(u)==sizeof(x));std::memcpy(&u,&x,sizeof(u));for(int k=0;k<4;++k){h^=(u>>(k*8))&255u;h*=1099511628211ull;}}return h;}
struct Genome{int x{4},y{3},z{3};bool extra{false};};
inline Graph make_graph(Genome g,int N=8){
    // The static genome grammar guarantees an acyclic, well-shaped SSA graph.
    // Multiple types are used: Tensor3D and Scalar; generic vector/matrix ops are available too.
    Graph r;auto add=[&](Op op,int a=-1,int b=-1,float p=0,float q=0,Shape sh={}){r.nodes.push_back({op,a,b,p,q,sh});return int(r.nodes.size()-1);};
    int x=add(Op::Input),c=add(Op::Conv3D,x,-1,.72f+.03f*g.x,.08f+.02f*g.y);
    int act=add(Op::Relu,c),down=add(Op::Pool2,act),up=add(Op::Up2,down);
    int mix=add(Op::Mix,x,up,.16f+.025f*(g.z%6));
    if(g.extra)mix=add(Op::Copy,mix);
    int mean=add(Op::Mean,mix),broad=add(Op::Broadcast,mean,-1,0,0,Shape::tensor(N,N,N));
    int final=add(Op::Mix,mix,broad,.02f);
    r.output=final;return r;
}
inline Graph target_graph(int N=8){return make_graph({4,3,3,false},N);}
} // namespace dm3d
