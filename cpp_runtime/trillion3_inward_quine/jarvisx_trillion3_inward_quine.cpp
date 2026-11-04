// JARVIS-X Trillion^3 Inward-Loop Quine, C++20
// Geometry-preserving radial contraction with a deterministic lossy feature map.
// A 10^36-position virtual coordinate domain is not a 10^36-parameter ANN.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace std;

struct Vec3 { double x, y, z; };
struct Latent { double lx, ly, lz, energy; uint32_t expert_id; };

class Trillion3AutoEncoder {
public:
    static constexpr double T = 1.0e12;
    static constexpr Vec3 center() { return {0.5 * T, 0.5 * T, 0.5 * T}; }
    static double fract(double a) { return a - floor(a); }
    static double norm(Vec3 a) { return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }
    static Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
    static void check(Vec3 p) {
        if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) ||
            p.x < 0 || p.y < 0 || p.z < 0 || p.x > T || p.y > T || p.z > T)
            throw invalid_argument("Coordinate outside finite [0,10^12]^3 cube");
    }
    static uint64_t mix64(uint64_t z) {
        z += UINT64_C(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
        return z ^ (z >> 31);
    }
    Latent encode(Vec3 p) const {
        check(p);
        const double x=p.x/T, y=p.y/T, z=p.z/T;
        Latent L{};
        L.lx=fract(sin(x*12.9898+y*78.233)*43758.5453);
        L.ly=fract(cos(y*93.9898+z*67.345)*127.1);
        L.lz=fract(sin(z*53.739+x*19.237)*269.5);
        L.energy=sqrt(x*x+y*y+z*z);
        const uint64_t hx=static_cast<uint64_t>(L.lx*1e9);
        const uint64_t hy=static_cast<uint64_t>(L.ly*1e9);
        const uint64_t hz=static_cast<uint64_t>(L.lz*1e9);
        L.expert_id=static_cast<uint32_t>(mix64(hx) ^ mix64(hy+17) ^ mix64(hz+31)) % 1000000U;
        return L;
    }
    Vec3 decode(const Latent& L, double phase=0.0) const {
        // Heuristic reconstruction: not an inverse, not trained weights.
        Vec3 p{
            (L.lx + 0.3*sin(L.ly*6.283185307179586+phase))*0.5*T,
            (L.ly + 0.3*cos(L.lz*6.283185307179586+phase))*0.5*T,
            (L.lz + 0.3*sin(L.lx*6.283185307179586+phase))*0.5*T
        };
        p.x=clamp(p.x,0.0,T); p.y=clamp(p.y,0.0,T); p.z=clamp(p.z,0.0,T);
        return p;
    }
    double loss(Vec3 original, Vec3 recovered) const {
        return norm(sub(original,recovered))/T; // cube-normalized Euclidean distance
    }
    static Vec3 inward(Vec3 old, Vec3 decoded, double rho) {
        if (!(rho>0.0 && rho<1.0)) throw invalid_argument("Expected 0<rho<1");
        const Vec3 c=center();
        Vec3 u=sub(decoded,c);
        const double ur=norm(u), radius=norm(sub(old,c));
        if (ur<1e-12 || radius<1e-12) return c;
        const double s=(rho*radius)/ur;
        Vec3 next{c.x+s*u.x,c.y+s*u.y,c.z+s*u.z};
        // Projection into convex cube cannot increase distance to center.
        next.x=clamp(next.x,0.0,T);
        next.y=clamp(next.y,0.0,T);
        next.z=clamp(next.z,0.0,T);
        return next;
    }
};

// This is the source template; --quine reconstructs the exact compiled source.
// Unlike reading __FILE__, the program contains all information required.
static constexpr const char* kSourceTemplate = "// JARVIS-X Trillion^3 Inward-Loop Quine, C++20\n// Geometry-preserving radial contraction with a deterministic lossy feature map.\n// A 10^36-position virtual coordinate domain is not a 10^36-parameter ANN.\n#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstdlib>\n#include <iomanip>\n#include <iostream>\n#include <limits>\n#include <random>\n#include <stdexcept>\n#include <string>\n#include <string_view>\n\nusing namespace std;\n\nstruct Vec3 { double x, y, z; };\nstruct Latent { double lx, ly, lz, energy; uint32_t expert_id; };\n\nclass Trillion3AutoEncoder {\npublic:\n    static constexpr double T = 1.0e12;\n    static constexpr Vec3 center() { return {0.5 * T, 0.5 * T, 0.5 * T}; }\n    static double fract(double a) { return a - floor(a); }\n    static double norm(Vec3 a) { return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }\n    static Vec3 sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }\n    static void check(Vec3 p) {\n        if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z) ||\n            p.x < 0 || p.y < 0 || p.z < 0 || p.x > T || p.y > T || p.z > T)\n            throw invalid_argument(\"Coordinate outside finite [0,10^12]^3 cube\");\n    }\n    static uint64_t mix64(uint64_t z) {\n        z += UINT64_C(0x9e3779b97f4a7c15);\n        z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);\n        z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);\n        return z ^ (z >> 31);\n    }\n    Latent encode(Vec3 p) const {\n        check(p);\n        const double x=p.x/T, y=p.y/T, z=p.z/T;\n        Latent L{};\n        L.lx=fract(sin(x*12.9898+y*78.233)*43758.5453);\n        L.ly=fract(cos(y*93.9898+z*67.345)*127.1);\n        L.lz=fract(sin(z*53.739+x*19.237)*269.5);\n        L.energy=sqrt(x*x+y*y+z*z);\n        const uint64_t hx=static_cast<uint64_t>(L.lx*1e9);\n        const uint64_t hy=static_cast<uint64_t>(L.ly*1e9);\n        const uint64_t hz=static_cast<uint64_t>(L.lz*1e9);\n        L.expert_id=static_cast<uint32_t>(mix64(hx) ^ mix64(hy+17) ^ mix64(hz+31)) % 1000000U;\n        return L;\n    }\n    Vec3 decode(const Latent& L, double phase=0.0) const {\n        // Heuristic reconstruction: not an inverse, not trained weights.\n        Vec3 p{\n            (L.lx + 0.3*sin(L.ly*6.283185307179586+phase))*0.5*T,\n            (L.ly + 0.3*cos(L.lz*6.283185307179586+phase))*0.5*T,\n            (L.lz + 0.3*sin(L.lx*6.283185307179586+phase))*0.5*T\n        };\n        p.x=clamp(p.x,0.0,T); p.y=clamp(p.y,0.0,T); p.z=clamp(p.z,0.0,T);\n        return p;\n    }\n    double loss(Vec3 original, Vec3 recovered) const {\n        return norm(sub(original,recovered))/T; // cube-normalized Euclidean distance\n    }\n    static Vec3 inward(Vec3 old, Vec3 decoded, double rho) {\n        if (!(rho>0.0 && rho<1.0)) throw invalid_argument(\"Expected 0<rho<1\");\n        const Vec3 c=center();\n        Vec3 u=sub(decoded,c);\n        const double ur=norm(u), radius=norm(sub(old,c));\n        if (ur<1e-12 || radius<1e-12) return c;\n        const double s=(rho*radius)/ur;\n        Vec3 next{c.x+s*u.x,c.y+s*u.y,c.z+s*u.z};\n        // Projection into convex cube cannot increase distance to center.\n        next.x=clamp(next.x,0.0,T);\n        next.y=clamp(next.y,0.0,T);\n        next.z=clamp(next.z,0.0,T);\n        return next;\n    }\n};\n\n// This is the source template; --quine reconstructs the exact compiled source.\n// Unlike reading __FILE__, the program contains all information required.\nstatic constexpr const char* kSourceTemplate = @@QUINE_LITERAL@@;\n\nstring cpp_literal(string_view s) {\n    string out=\"\\\"\";\n    for(unsigned char c : s) {\n        if(c=='\\\\') out+=\"\\\\\\\\\";\n        else if(c=='\\\"') out+=\"\\\\\\\"\";\n        else if(c=='\\n') out+=\"\\\\n\";\n        else if(c=='\\r') out+=\"\\\\r\";\n        else if(c=='\\t') out+=\"\\\\t\";\n        else if(c>=32 && c<=126) out+=char(c);\n        else { // Exact octal escapes for any other byte.\n            out+='\\\\'; out+=char('0'+((c>>6)&7));\n            out+=char('0'+((c>>3)&7)); out+=char('0'+(c&7));\n        }\n    }\n    return out+'\\\"';\n}\nstring own_source() {\n    string s(kSourceTemplate);\n    const string marker=\"@@QUINE_LITERAL@@\";\n    // Template has marker in both code and string literal; replace ONLY declaration.\n    const string declaration=\"kSourceTemplate = \"+marker+\";\";\n    size_t p=s.find(declaration);\n    if(p==string::npos) throw logic_error(\"Quine marker missing\");\n    s.replace(p+string(\"kSourceTemplate = \").size(),marker.size(),cpp_literal(kSourceTemplate));\n    return s;\n}\nuint64_t fnv1a(string_view s) {\n    uint64_t h=UINT64_C(14695981039346656037);\n    for(unsigned char b:s) { h^=b; h*=UINT64_C(1099511628211); }\n    return h;\n}\nVec3 initial_seed() {\n    const uint64_t h=fnv1a(own_source());\n    const double u=static_cast<double>(h&0xffff)/65535.0;\n    const double v=static_cast<double>((h>>16)&0xffff)/65535.0;\n    const double w=static_cast<double>((h>>32)&0xffff)/65535.0;\n    return {(0.45+0.10*u)*Trillion3AutoEncoder::T,\n            (0.45+0.10*v)*Trillion3AutoEncoder::T,\n            (0.45+0.10*w)*Trillion3AutoEncoder::T};\n}\n\nstruct RunStats { double initial_radius, final_radius, avg_reconstruction_loss; };\nRunStats run(int steps, bool verbose) {\n    constexpr double rho=0.85;\n    Trillion3AutoEncoder E;\n    Vec3 p=initial_seed();\n    const Vec3 c=E.center();\n    const double initial=E.norm(E.sub(p,c))/E.T;\n    double sum_loss=0;\n    if(verbose) {\n        cout<<\"JARVIS-X Trillion^3 Inward Quine | virtual positions = 10^36, learned parameters = 0\\n\";\n        cout<<\"Source bytes=\"<<own_source().size()<<\" FNV-1a=0x\"<<hex<<fnv1a(own_source())<<dec<<\"\\n\";\n        cout<<fixed<<setprecision(9);\n        cout<<\"Initial normalized center radius=\"<<initial<<\"\\n\";\n    }\n    for(int t=0;t<steps;t++) {\n        const auto L=E.encode(p);\n        const auto rec=E.decode(L,0.314*t);\n        const double ell=E.loss(p,rec);\n        const auto next=E.inward(p,rec,rho);\n        const double r_old=E.norm(E.sub(p,c))/E.T;\n        const double r_new=E.norm(E.sub(next,c))/E.T;\n        if(!isfinite(ell) || !isfinite(r_new) || r_new>rho*r_old+1e-12)\n            throw runtime_error(\"Inward invariant violated\");\n        if(L.expert_id>=1000000U) throw runtime_error(\"Invalid expert\");\n        sum_loss+=ell;\n        if(verbose) cout<<\"step=\"<<setw(2)<<t+1<<\" expert=\"<<setw(6)<<L.expert_id\n                         <<\" loss=\"<<ell<<\" center_radius=\"<<r_new<<'\\n';\n        p=next;\n    }\n    return {initial,E.norm(E.sub(p,c))/E.T,steps?sum_loss/steps:0.0};\n}\nvoid selftest() {\n    Trillion3AutoEncoder E;\n    mt19937_64 rng(0xC0DEC0DE);\n    uniform_real_distribution<double> dist(0.0,E.T);\n    for(int i=0;i<10000;i++) {\n        const Vec3 p{dist(rng),dist(rng),dist(rng)};\n        const auto L=E.encode(p);\n        const auto q=E.decode(L,0.314*i);\n        if(!isfinite(E.loss(p,q))||L.expert_id>=1000000U) throw runtime_error(\"Bad codec result\");\n        if(E.loss(p,q)>sqrt(3.0)+1e-12) throw runtime_error(\"Decoded outside cube\");\n    }\n    auto stats=run(12,false);\n    if(stats.final_radius>stats.initial_radius*pow(0.85,12)+1e-12)\n        throw runtime_error(\"Radial bound failed\");\n    if(own_source().find(\"kSourceTemplate = \\\"// JARVIS-X\") == string::npos)\n        throw runtime_error(\"Quine representation invalid\");\n    cout<<\"PASS: 10,000 finite latent/decode tests, expert routing, radial contraction, source reconstruction\\n\";\n    cout<<setprecision(12)<<\"Initial radius=\"<<stats.initial_radius\n        <<\" Final radius=\"<<stats.final_radius<<\" mean loss=\"<<stats.avg_reconstruction_loss<<'\\n';\n}\nint main(int argc,char**argv) {\n    try {\n        if(argc==2 && string(argv[1])==\"--quine\") { cout<<own_source(); return 0; }\n        if(argc==2 && string(argv[1])==\"--selftest\") { selftest(); return 0; }\n        if(argc==1) { run(12,true); return 0; }\n        if(argc==3 && string(argv[1])==\"--run\") {\n            const int steps=stoi(argv[2]);\n            if(steps<0 || steps>100000) throw invalid_argument(\"steps out of range 0..100000\");\n            run(steps,true); return 0;\n        }\n        cerr<<\"Usage: jx_inward [--selftest|--quine|--run STEPS]\\n\";\n        return 2;\n    } catch(const exception& e) { cerr<<\"ERROR: \"<<e.what()<<'\\n'; return 1; }\n}\n";

string cpp_literal(string_view s) {
    string out="\"";
    for(unsigned char c : s) {
        if(c=='\\') out+="\\\\";
        else if(c=='\"') out+="\\\"";
        else if(c=='\n') out+="\\n";
        else if(c=='\r') out+="\\r";
        else if(c=='\t') out+="\\t";
        else if(c>=32 && c<=126) out+=char(c);
        else { // Exact octal escapes for any other byte.
            out+='\\'; out+=char('0'+((c>>6)&7));
            out+=char('0'+((c>>3)&7)); out+=char('0'+(c&7));
        }
    }
    return out+'\"';
}
string own_source() {
    string s(kSourceTemplate);
    const string marker="@@QUINE_LITERAL@@";
    // Template has marker in both code and string literal; replace ONLY declaration.
    const string declaration="kSourceTemplate = "+marker+";";
    size_t p=s.find(declaration);
    if(p==string::npos) throw logic_error("Quine marker missing");
    s.replace(p+string("kSourceTemplate = ").size(),marker.size(),cpp_literal(kSourceTemplate));
    return s;
}
uint64_t fnv1a(string_view s) {
    uint64_t h=UINT64_C(14695981039346656037);
    for(unsigned char b:s) { h^=b; h*=UINT64_C(1099511628211); }
    return h;
}
Vec3 initial_seed() {
    const uint64_t h=fnv1a(own_source());
    const double u=static_cast<double>(h&0xffff)/65535.0;
    const double v=static_cast<double>((h>>16)&0xffff)/65535.0;
    const double w=static_cast<double>((h>>32)&0xffff)/65535.0;
    return {(0.45+0.10*u)*Trillion3AutoEncoder::T,
            (0.45+0.10*v)*Trillion3AutoEncoder::T,
            (0.45+0.10*w)*Trillion3AutoEncoder::T};
}

struct RunStats { double initial_radius, final_radius, avg_reconstruction_loss; };
RunStats run(int steps, bool verbose) {
    constexpr double rho=0.85;
    Trillion3AutoEncoder E;
    Vec3 p=initial_seed();
    const Vec3 c=E.center();
    const double initial=E.norm(E.sub(p,c))/E.T;
    double sum_loss=0;
    if(verbose) {
        cout<<"JARVIS-X Trillion^3 Inward Quine | virtual positions = 10^36, learned parameters = 0\n";
        cout<<"Source bytes="<<own_source().size()<<" FNV-1a=0x"<<hex<<fnv1a(own_source())<<dec<<"\n";
        cout<<fixed<<setprecision(9);
        cout<<"Initial normalized center radius="<<initial<<"\n";
    }
    for(int t=0;t<steps;t++) {
        const auto L=E.encode(p);
        const auto rec=E.decode(L,0.314*t);
        const double ell=E.loss(p,rec);
        const auto next=E.inward(p,rec,rho);
        const double r_old=E.norm(E.sub(p,c))/E.T;
        const double r_new=E.norm(E.sub(next,c))/E.T;
        if(!isfinite(ell) || !isfinite(r_new) || r_new>rho*r_old+1e-12)
            throw runtime_error("Inward invariant violated");
        if(L.expert_id>=1000000U) throw runtime_error("Invalid expert");
        sum_loss+=ell;
        if(verbose) cout<<"step="<<setw(2)<<t+1<<" expert="<<setw(6)<<L.expert_id
                         <<" loss="<<ell<<" center_radius="<<r_new<<'\n';
        p=next;
    }
    return {initial,E.norm(E.sub(p,c))/E.T,steps?sum_loss/steps:0.0};
}
void selftest() {
    Trillion3AutoEncoder E;
    mt19937_64 rng(0xC0DEC0DE);
    uniform_real_distribution<double> dist(0.0,E.T);
    for(int i=0;i<10000;i++) {
        const Vec3 p{dist(rng),dist(rng),dist(rng)};
        const auto L=E.encode(p);
        const auto q=E.decode(L,0.314*i);
        if(!isfinite(E.loss(p,q))||L.expert_id>=1000000U) throw runtime_error("Bad codec result");
        if(E.loss(p,q)>sqrt(3.0)+1e-12) throw runtime_error("Decoded outside cube");
    }
    auto stats=run(12,false);
    if(stats.final_radius>stats.initial_radius*pow(0.85,12)+1e-12)
        throw runtime_error("Radial bound failed");
    if(own_source().find("kSourceTemplate = \"// JARVIS-X") == string::npos)
        throw runtime_error("Quine representation invalid");
    cout<<"PASS: 10,000 finite latent/decode tests, expert routing, radial contraction, source reconstruction\n";
    cout<<setprecision(12)<<"Initial radius="<<stats.initial_radius
        <<" Final radius="<<stats.final_radius<<" mean loss="<<stats.avg_reconstruction_loss<<'\n';
}
int main(int argc,char**argv) {
    try {
        if(argc==2 && string(argv[1])=="--quine") { cout<<own_source(); return 0; }
        if(argc==2 && string(argv[1])=="--selftest") { selftest(); return 0; }
        if(argc==1) { run(12,true); return 0; }
        if(argc==3 && string(argv[1])=="--run") {
            const int steps=stoi(argv[2]);
            if(steps<0 || steps>100000) throw invalid_argument("steps out of range 0..100000");
            run(steps,true); return 0;
        }
        cerr<<"Usage: jx_inward [--selftest|--quine|--run STEPS]\n";
        return 2;
    } catch(const exception& e) { cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }
}
