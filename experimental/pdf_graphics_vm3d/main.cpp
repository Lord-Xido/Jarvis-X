// JARVIS-X PDF Graphics VM3D: bounded PDF content-stream subset (C++20).
// This is NOT a general PDF parser, native machine-code JIT, or GPU backend.
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jx {
constexpr std::size_t max_source = 1024 * 1024;
constexpr std::size_t max_ops = 100000;
constexpr std::size_t max_polys = 300000;
constexpr int side = 128;

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }
void require(bool ok, const std::string& msg) { if (!ok) fail(msg); }
uint64_t fnv(const uint8_t* p, std::size_t n) {
    uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}
uint64_t fnv(const std::string& s) { return fnv(reinterpret_cast<const uint8_t*>(s.data()), s.size()); }
std::string hex64(uint64_t v) { std::ostringstream ss; ss << std::hex << std::setfill('0') << std::setw(16) << v; return ss.str(); }

enum class Kind { Number, Name, String, Operator };
struct Atom { Kind kind; double number = 0; std::string text; };
std::vector<Atom> lex(const std::string& source) {
    require(source.size() <= max_source, "stream byte limit exceeded");
    std::vector<Atom> out;
    std::size_t i = 0;
    while (i < source.size()) {
        unsigned char c = static_cast<unsigned char>(source[i]);
        if (c == '%') { while (i < source.size() && source[i] != '\n' && source[i] != '\r') ++i; continue; }
        if (c <= ' ') { ++i; continue; }
        if (c == '(') {
            ++i; int depth = 1; std::string value;
            while (i < source.size() && depth > 0) {
                char x = source[i++];
                if (x == '\\') {
                    require(i < source.size(), "truncated string escape");
                    const char e = source[i++];
                    if (e == 'n') value += '\n'; else if (e == 'r') value += '\r';
                    else if (e == 't') value += '\t'; else if (e == 'b') value += '\b';
                    else if (e == 'f') value += '\f';
                    else if (e == '\n') {} else if (e == '\r') { if (i < source.size() && source[i] == '\n') ++i; }
                    else if (e >= '0' && e <= '7') {
                        int n = e - '0';
                        for (int k = 0; k < 2 && i < source.size() && source[i] >= '0' && source[i] <= '7'; ++k)
                            n = 8*n + (source[i++] - '0');
                        value += static_cast<char>(n & 255);
                    } else value += e;
                } else if (x == '(') { require(++depth <= 16, "string nesting limit"); value += x; }
                else if (x == ')') { if (--depth > 0) value += x; }
                else value += x;
                require(value.size() <= max_source, "string too large");
            }
            require(depth == 0, "unclosed PDF literal string");
            out.push_back({Kind::String, 0, std::move(value)});
        } else {
            const bool name = c == '/';
            if (name) ++i;
            const std::size_t begin = i;
            while (i < source.size()) {
                const char x = source[i];
                if (static_cast<unsigned char>(x) <= 32 || x == '%' || x == '(' || x == ')' || x == '/' ||
                    x == '<' || x == '>' || x == '[' || x == ']' || x == '{' || x == '}') break;
                ++i;
            }
            require(i > begin, "unsupported PDF delimiter");
            const std::string word = source.substr(begin, i - begin);
            if (name) out.push_back({Kind::Name, 0, word});
            else {
                char* end = nullptr;
                const double v = std::strtod(word.c_str(), &end);
                if (end != word.c_str() && *end == '\0') {
                    require(std::isfinite(v), "nonfinite numeric operand");
                    out.push_back({Kind::Number, v, {}});
                } else out.push_back({Kind::Operator, 0, word});
            }
        }
        require(out.size() <= max_ops * 8, "token limit exceeded");
    }
    return out;
}

enum Op : uint16_t { QSave, QRestore, CM, BT, ET, TF, TM, TJ, RE, FILL, RG };
const char* op_name(Op op) {
    switch(op) {
    case QSave:return "q";case QRestore:return "Q";case CM:return "cm";
    case BT:return "BT";case ET:return "ET";case TF:return "Tf";case TM:return "Tm";
    case TJ:return "Tj";case RE:return "re";case FILL:return "f";case RG:return "rg";
    } return "?";
}
// Exactly 128 bits; arbitrary PDF operands are held in external pools.
struct Instruction { uint16_t op, argc; uint32_t number_index, text_index, reserved; };
static_assert(sizeof(Instruction) == 16);
struct Program {
    std::string original;
    std::vector<Instruction> code;
    std::vector<std::array<double,6>> numbers;
    std::vector<std::string> strings;
    uint64_t input_hash = 0;
    std::string decode_original() const { return original; } // byte-preserving content transport, not PDF decompression
};
Program parse(const std::string& src) {
    Program p; p.original = src; p.input_hash = fnv(src);
    std::vector<Atom> stack;
    for (auto& a: lex(src)) {
        if (a.kind != Kind::Operator) {
            require(stack.size() < 8, "PDF operand stack limit exceeded");
            stack.push_back(a); continue;
        }
        Op op{}; std::vector<Kind> kinds;
        if (a.text == "q") op=QSave;
        else if (a.text == "Q") op=QRestore;
        else if (a.text == "cm") { op=CM; kinds={Kind::Number,Kind::Number,Kind::Number,Kind::Number,Kind::Number,Kind::Number}; }
        else if (a.text == "BT") op=BT;
        else if (a.text == "ET") op=ET;
        else if (a.text == "Tf") { op=TF; kinds={Kind::Name,Kind::Number}; }
        else if (a.text == "Tm") { op=TM; kinds={Kind::Number,Kind::Number,Kind::Number,Kind::Number,Kind::Number,Kind::Number}; }
        else if (a.text == "Tj") { op=TJ; kinds={Kind::String}; }
        else if (a.text == "re") { op=RE; kinds={Kind::Number,Kind::Number,Kind::Number,Kind::Number}; }
        else if (a.text == "f") op=FILL;
        else if (a.text == "rg") { op=RG; kinds={Kind::Number,Kind::Number,Kind::Number}; }
        else fail("unsupported PDF operator: " + a.text);
        require(stack.size()==kinds.size(), "wrong operand count for " + a.text);
        std::array<double,6> nums{}; uint32_t text_idx=UINT32_MAX;
        for (std::size_t k=0; k<kinds.size(); ++k) {
            require(stack[k].kind==kinds[k], "wrong operand type for " + a.text);
            if (kinds[k]==Kind::Number) nums[k] = stack[k].number;
            else { text_idx = static_cast<uint32_t>(p.strings.size()); p.strings.push_back(stack[k].text); }
        }
        const uint32_t n_idx = static_cast<uint32_t>(p.numbers.size());
        p.numbers.push_back(nums);
        p.code.push_back({static_cast<uint16_t>(op),static_cast<uint16_t>(kinds.size()),n_idx,text_idx,0});
        require(p.code.size() <= max_ops, "instruction limit exceeded");
        stack.clear();
    }
    require(stack.empty(), "unconsumed operands");
    return p;
}

struct Point { double x=0,y=0; };
struct Matrix {
    double a=1,b=0,c=0,d=1,e=0,f=0;
    Point map(Point p) const {return {a*p.x+c*p.y+e,b*p.x+d*p.y+f};}
};
Matrix multiply(const Matrix& x, const Matrix& y) {
    return {x.a*y.a+x.c*y.b, x.b*y.a+x.d*y.b,
            x.a*y.c+x.c*y.d, x.b*y.c+x.d*y.d,
            x.a*y.e+x.c*y.f+x.e,x.b*y.e+x.d*y.f+x.f};
}
Matrix from(const std::array<double,6>& x) { return {x[0],x[1],x[2],x[3],x[4],x[5]}; }
struct Color { uint8_t r=0,g=0,b=0; };
struct Quad {
    std::array<Point,4> v{}; Color color{};
    uint32_t seq=0, tx=0,ty=0,tz=0; // virtual tile coordinate (not allocated 10^18 voxels)
};
struct State { Matrix ctm; Color fill; double font_size=12; std::string font="builtin5x7"; };
uint8_t channel(double x) {return static_cast<uint8_t>(std::llround(std::clamp(x,0.0,1.0)*255.0));}
// Deterministic built-in 5x7 demonstration font, NOT PDF font decoding or Unicode shaping.
std::array<uint8_t,7> glyph(char c) {
    c=static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    switch (c) {
        case 'A':return {14,17,17,31,17,17,17};case 'E':return {31,16,16,30,16,16,31};
        case 'H':return {17,17,17,31,17,17,17};case 'I':return {31,4,4,4,4,4,31};
        case 'J':return {7,2,2,2,18,18,12};case 'L':return {16,16,16,16,16,16,31};
        case 'O':return {14,17,17,17,17,17,14};case 'R':return {30,17,17,30,20,18,17};
        case 'S':return {15,16,16,14,1,1,30};case 'V':return {17,17,17,17,17,10,4};
        case ' ':return {0,0,0,0,0,0,0};default:return {31,17,17,17,17,17,31};
    }
}
struct TraceEntry { std::size_t pc; std::string op; uint32_t emitted; uint64_t accum; };
struct Geometry { std::vector<Quad> quads; std::vector<TraceEntry> trace; };
void add_polygon(Geometry& geom, const std::array<Point,4>& points, Color color) {
    require(geom.quads.size()<max_polys, "primitive limit exceeded");
    Quad q; q.v=points;
    for (Point p:q.v) require(std::isfinite(p.x)&&std::isfinite(p.y), "coordinate overflow");
    q.color=color; q.seq=static_cast<uint32_t>(geom.quads.size());
    const auto minx=std::min({q.v[0].x,q.v[1].x,q.v[2].x,q.v[3].x});
    const auto miny=std::min({q.v[0].y,q.v[1].y,q.v[2].y,q.v[3].y});
    q.tx=static_cast<uint32_t>(std::clamp(std::floor(minx/16),0.0,999999.0));
    q.ty=static_cast<uint32_t>(std::clamp(std::floor(miny/16),0.0,999999.0));
    q.tz=q.seq; require(q.tz<1000000,"virtual z coordinate overflow");
    geom.quads.push_back(q);
}
void add_quad(Geometry& geom, const Matrix& mat, double x,double y,double w,double h,Color color) {
    require(std::isfinite(x) && std::isfinite(y) && std::isfinite(w) && std::isfinite(h), "invalid primitive");
    add_polygon(geom,{mat.map({x,y}),mat.map({x+w,y}),mat.map({x+w,y+h}),mat.map({x,y+h})},color);
}
Geometry lower(const Program& p) {
    Geometry geom; State gs; std::vector<State> saved; Matrix text_matrix;
    bool in_text=false; std::vector<std::array<Point,4>> path;
    uint64_t trace_hash=14695981039346656037ULL;
    for (std::size_t pc=0;pc<p.code.size();++pc) {
        const auto& i=p.code[pc]; const auto& n=p.numbers.at(i.number_index);
        const auto op=static_cast<Op>(i.op);
        const std::size_t before=geom.quads.size();
        switch (op) {
            case QSave:require(saved.size()<64,"graphics state stack overflow");saved.push_back(gs);break;
            case QRestore:require(!saved.empty(),"graphics state underflow");gs=saved.back();saved.pop_back();break;
            case CM: gs.ctm=multiply(gs.ctm,from(n));break;
            case BT:require(!in_text,"nested BT");in_text=true;text_matrix={};break;
            case ET:require(in_text,"ET outside text object");in_text=false;break;
            case TF:require(in_text,"Tf outside text object");require(n[1]>0&&n[1]<1024,"bad font size");
                gs.font=p.strings.at(i.text_index);gs.font_size=n[1];break;
            case TM:require(in_text,"Tm outside text object");text_matrix=from(n);break;
            case TJ: {
                require(in_text,"Tj outside text object");
                const auto& s=p.strings.at(i.text_index);
                require(s.size()<=16384,"text length limit");
                const Matrix combined=multiply(gs.ctm,text_matrix);
                const double cell=gs.font_size/7.0;
                for (std::size_t ch=0;ch<s.size();++ch) {
                    const auto bitmap=glyph(s[ch]);
                    for (int row=0;row<7;++row) for(int col=0;col<5;++col)
                        if (bitmap[static_cast<std::size_t>(row)] & (1 << (4-col)))
                            add_quad(geom,combined,(static_cast<double>(ch)*6+col)*cell,
                                     static_cast<double>(6-row)*cell,cell,cell,gs.fill);
                }
                // Text advance retained as text matrix translation for subsequent Tj.
                text_matrix=multiply(text_matrix,{1,0,0,1,static_cast<double>(s.size())*6*cell,0});
                break;
            }
            case RE: {
                require(path.size()<max_polys,"path limit exceeded");
                const double x=n[0],y=n[1],w=n[2],h=n[3];
                path.push_back({gs.ctm.map({x,y}),gs.ctm.map({x+w,y}),gs.ctm.map({x+w,y+h}),gs.ctm.map({x,y+h})});
                break;
            }
            case FILL:
                for(const auto& rect:path)add_polygon(geom,rect,gs.fill);
                path.clear();break;
            case RG:gs.fill={channel(n[0]),channel(n[1]),channel(n[2])};break;
        }
        trace_hash ^= static_cast<uint64_t>(i.op);trace_hash *= 1099511628211ULL;
        trace_hash ^= static_cast<uint64_t>(geom.quads.size());trace_hash *= 1099511628211ULL;
        geom.trace.push_back({pc,op_name(op),static_cast<uint32_t>(geom.quads.size()-before),trace_hash});
    }
    require(saved.empty(),"unclosed graphics state");
    require(!in_text,"unclosed text object");
    return geom;
}
struct Frame {
    int w=side,h=side;
    std::vector<uint8_t> rgb;
    Frame() : rgb(static_cast<std::size_t>(side*side*3),255) {}
    bool operator==(const Frame& o) const {return w==o.w&&h==o.h&&rgb==o.rgb;}
    uint64_t hash() const {return fnv(rgb.data(),rgb.size());}
};
// Convex quad test, sample center, PDF coordinates are y-up and framebuffer y-down.
bool inside(const std::array<Point,4>& v,double x,double y) {
    int sign=0;
    for(int k=0;k<4;++k) {
        const auto& a=v[static_cast<std::size_t>(k)];const auto& b=v[static_cast<std::size_t>((k+1)%4)];
        const double cross=(b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x);
        if(std::abs(cross)<1e-10)continue;
        const int s=cross>0?1:-1;
        if(sign && sign!=s)return false;
        sign=s;
    }
    return sign!=0;
}
void paint(Frame& f,const Quad& q) {
    const auto minx=std::min({q.v[0].x,q.v[1].x,q.v[2].x,q.v[3].x});
    const auto maxx=std::max({q.v[0].x,q.v[1].x,q.v[2].x,q.v[3].x});
    const auto miny=std::min({q.v[0].y,q.v[1].y,q.v[2].y,q.v[3].y});
    const auto maxy=std::max({q.v[0].y,q.v[1].y,q.v[2].y,q.v[3].y});
    if (maxx<0 || maxy<0 || minx>=f.w || miny>=f.h) return;
    const int xa=static_cast<int>(std::max(0.0,std::floor(minx)));
    const int xb=static_cast<int>(std::min(static_cast<double>(f.w),std::ceil(maxx)));
    const int ya=static_cast<int>(std::max(0.0,std::floor(miny)));
    const int yb=static_cast<int>(std::min(static_cast<double>(f.h),std::ceil(maxy)));
    for (int y=ya;y<yb;++y)for(int x=xa;x<xb;++x)if(inside(q.v,x+0.5,y+0.5)){
        const auto off=static_cast<std::size_t>(((f.h-1-y)*f.w+x)*3);
        f.rgb[off]=q.color.r;f.rgb[off+1]=q.color.g;f.rgb[off+2]=q.color.b;
    }
}
Frame render_geometry(const Geometry& g){Frame f;for(const auto& q:g.quads)paint(f,q);return f;}
Frame reference(const Program& p){return render_geometry(lower(p));}
struct Plan { Geometry g; uint64_t source_hash=0; };
Plan specialize(const Program& p) {return {lower(p),p.input_hash};} // typed IR -> static draw-command plan, NOT native JIT
Frame execute(const Plan& p){return render_geometry(p.g);}
struct Receipt { bool pixel_equal=false; bool faster=false; bool commit=false; double interpreter_ns=0,plan_ns=0; };
double measure(const Program& p,const Plan& plan,bool use_plan,int repeats) {
    auto t0=std::chrono::steady_clock::now();uint64_t sink=0;
    for(int k=0;k<repeats;++k) {
        Frame f=use_plan?execute(plan):reference(p);
        sink ^= f.hash();
    }
    const double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-t0).count();
    if(sink==UINT64_MAX)std::cerr<<"unlikely checksum\n";
    return ns/repeats;
}
Receipt optimize(const Program& p,const Plan& plan,int repeats) {
    require(repeats>=1&&repeats<=10000,"invalid repetitions");
    Receipt r;
    r.pixel_equal=(p.input_hash==plan.source_hash && reference(p)==execute(plan));
    if(!r.pixel_equal)return r;
    std::array<double,3> a{},b{};
    for(int k=0;k<3;++k){a[static_cast<std::size_t>(k)]=measure(p,plan,false,repeats);b[static_cast<std::size_t>(k)]=measure(p,plan,true,repeats);}
    std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
    r.interpreter_ns=a[1];r.plan_ns=b[1];r.faster=r.plan_ns<r.interpreter_ns;
    r.commit=r.pixel_equal && r.faster;
    return r;
}
std::string sample() {
    return "q\n0.5 0 0 0.5 20 40 cm\nBT\n/F1 24 Tf\n1 0 0 1 100 100 Tm\n(Hello JARVIS) Tj\nET\n1 0 0 rg\n100 40 100 50 re\nf\nQ\n";
}
void save_ppm(const Frame& f,const std::string& path) {
    std::ofstream o(path,std::ios::binary);require(bool(o),"cannot open image output");
    o<<"P6\n"<<f.w<<" "<<f.h<<"\n255\n";
    o.write(reinterpret_cast<const char*>(f.rgb.data()),static_cast<std::streamsize>(f.rgb.size()));
    require(bool(o),"failed image write");
}
void save_trace(const Geometry& g,const std::string& path) {
    std::ofstream o(path);require(bool(o),"cannot open trace output");
    o<<"kind,pc,op,emitted_polygons,trace_hash,polygon,tile_x,tile_y,tile_z\n";
    for(const auto& e:g.trace)o<<"instruction,"<<e.pc<<","<<e.op<<","<<e.emitted<<","<<hex64(e.accum)<<",,,,\n";
    for(const auto& q:g.quads)o<<"primitive,,,,,"<<q.seq<<","<<q.tx<<","<<q.ty<<","<<q.tz<<"\n";
}
void tests() {
    const std::string raw=sample();const Program p=parse(raw);
    require(p.decode_original()==raw,"byte transport failed");
    require(p.code.size()==11,"wrong opcode count");
    const auto mat=multiply(Matrix{0.5,0,0,0.5,20,40},Matrix{1,0,0,1,100,100});
    const auto v=mat.map({0,0});require(v.x==70&&v.y==90,"matrix fusion wrong");
    auto g=lower(p);auto plan=specialize(p);
    require(!g.quads.empty(),"empty geometry");
    require(reference(p)==execute(plan),"interpreter/plan differential mismatch");
    const auto f=execute(plan);require(std::count_if(f.rgb.begin(),f.rgb.end(),[](uint8_t x){return x!=255;})>0,"empty output");
    const auto red=f.rgb[static_cast<std::size_t>(((side-1-70)*side+80)*3)];
    require(red==255,"unexpected rectangle red value");
    const auto green=f.rgb[static_cast<std::size_t>(((side-1-70)*side+80)*3+1)];
    require(green==0,"rectangle pixel missing");
    require(parse("(A\\(B\\)) Tj").strings.front()=="A(B)","escape decode failed");
    for(const std::string bad:{"Q", "1 2 cm", "BT (x) Tj", "1 2 3 4 5 6 zz", "(abc", "1e999 2 3 rg"}) {
        bool threw=false;try{auto x=parse(bad);(void)lower(x);}catch(const std::exception&){threw=true;}
        require(threw,"malformed input accepted: "+bad);
    }
    // PDF path coordinates are frozen under CTM at path construction, not at fill.
    const auto path_transform=parse("1 0 0 1 10 0 cm 0 0 10 10 re 1 0 0 1 30 0 cm f");
    const auto transformed=lower(path_transform);
    require(transformed.quads.size()==1 && transformed.quads[0].v[0].x==10,
            "path incorrectly re-transformed at fill time");
    // q/Q restores the CTM, and does not rewind the current path.
    const auto state_example=lower(parse("q 1 0 0 1 20 0 cm 0 0 2 2 re Q f"));
    require(state_example.quads.size()==1 && state_example.quads[0].v[0].x==20,
            "state restoration incorrectly moved existing path");
    require(parse("q Q").code.size()==2,"operator parser failed");
    require(parse("0.1 0.2 0.3 rg").code.size()==1,"color parser failed");
    std::cout<<"SELF_TEST_PASS instruction16, byte-roundtrip, operator-validation, matrix-fusion, "
             <<"graphics-state, 3D-tile-address, trace, pixel differential\n"
             <<"source_hash="<<hex64(p.input_hash)<<" frame_hash="<<hex64(f.hash())
             <<" quads="<<g.quads.size()<<" ops="<<p.code.size()<<"\n";
}
} // namespace jx
int main(int argc,char** argv) {
    try {
        if(argc>=2&&std::string_view(argv[1])=="--self-test") {jx::tests();return 0;}
        std::string src;
        int repeats=100;
        if(argc>=2&&std::string_view(argv[1])=="--content") {
            jx::require(argc>=3,"--content needs a content-stream file");
            std::ifstream file(argv[2],std::ios::binary);jx::require(bool(file),"cannot open content stream");
            file.seekg(0,std::ios::end);
            const auto length=file.tellg();
            jx::require(length>=0 && length<=static_cast<std::streamoff>(jx::max_source),"stream byte limit exceeded");
            file.seekg(0);
            src=std::string(std::istreambuf_iterator<char>(file),{});
        } else if(argc>=2&&std::string_view(argv[1])=="--demo")src=jx::sample();
        else {std::cerr<<"Usage: pdf_graphics_vm3d --self-test | --demo [out.ppm trace.csv] | --content stream.txt [out.ppm trace.csv]\n";return 2;}
        if(argc>=6) {
            const std::string v=argv[5];
            char* end=nullptr;long n=std::strtol(v.c_str(),&end,10);
            jx::require(end!=v.c_str()&&*end=='\0'&&n>0&&n<=10000,"bad repetition count");repeats=static_cast<int>(n);
        }
        const bool external=(argc>=2&&std::string_view(argv[1])=="--content");
        const int image_arg=external?3:2;
        const int trace_arg=external?4:3;
        const std::string img=(argc>image_arg?argv[image_arg]:"jx_pdf3d.ppm");
        const std::string trace=(argc>trace_arg?argv[trace_arg]:"jx_pdf3d_trace.csv");
        auto p=jx::parse(src);auto plan=jx::specialize(p);
        const auto receipt=jx::optimize(p,plan,repeats);
        jx::require(receipt.pixel_equal,"optimized output differs from reference");
        const auto frame=receipt.commit?jx::execute(plan):jx::reference(p);
        jx::save_ppm(frame,img);jx::save_trace(plan.g,trace);
        std::cout<<"input_bytes="<<src.size()<<" instruction_count="<<p.code.size()
                 <<" polygon_count="<<plan.g.quads.size()<<" source_fnv="<<jx::hex64(p.input_hash)
                 <<" frame_fnv="<<jx::hex64(frame.hash())<<"\n"
                 <<"reference_ns="<<std::fixed<<std::setprecision(1)<<receipt.interpreter_ns
                 <<" optimized_ns="<<receipt.plan_ns<<" differential=PASS accepted="<<(receipt.commit?"YES":"NO")
                 <<"\n"<<"image="<<img<<" trace="<<trace<<"\n";
        return 0;
    } catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
