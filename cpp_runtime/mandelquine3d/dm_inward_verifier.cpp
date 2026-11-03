// Dr Moagi 3D Inward-Loop Verifier -- C++20 executable audit of the supplied engine.
// The original source is embedded as a separate implementation and its main renamed.
// Build: g++ -O2 -std=c++20 -Wall -Wextra -pedantic dm_inward_verifier.cpp -o dm_inward_verifier
// Run: ./dm_inward_verifier audit_outputs /path/to/original_source.cpp
// Tests: gradient correctness, spatial map, contraction, convergence, 4-step equivalence,
// nested archive recursion, bytewise multimedia roundtrip, deterministic replay, quine,
// and corruption detection / missing integrity.
#define main dm_original_cli_main
#include "dm_mandelquine.cpp"
#undef main

#include <chrono>
#include <functional>
#include <sstream>
#include <cstdlib>
#include <numeric>

namespace audit {
namespace fs = std::filesystem;
using namespace dm;
using Clock = std::chrono::steady_clock;

void check(bool test, const std::string &label) {
    if (!test) throw std::runtime_error("AUDIT FAIL: " + label);
}
std::uint64_t fnv1a64(const Bytes& b) {
    std::uint64_t h=14695981039346656037ull;
    for(Byte x:b){h^=x;h*=1099511628211ull;}return h;
}
std::string hex64(std::uint64_t v) {
    std::ostringstream s;s<<std::hex<<std::setw(16)<<std::setfill('0')<<v;return s.str();
}
double seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now()-begin).count();
}
float maxnorm(const std::vector<Lat>& a,const std::vector<Lat>& b) {
    check(a.size()==b.size(),"latent sizes equal");
    float m=0;
    for(std::size_t i=0;i<a.size();i++)for(int k=0;k<Z;k++)
        m=std::max(m,std::abs(a[i][k]-b[i][k]));
    return m;
}
std::vector<Lat> spatial_step(const std::vector<Lat>& state,
                              const std::vector<Lat>& frozen_base,
                              const std::vector<Cell>& geom, std::uint32_t s) {
    check(state.size()==frozen_base.size()&&state.size()==geom.size(),"recurrence shapes");
    std::vector<Lat> result(state.size());
    std::size_t ss=std::size_t(s)*s;
    for(std::size_t i=0;i<state.size();++i) {
        const auto x=i%s,y=(i/s)%s,z=i/ss;
        Lat neighbors{};int count=0;
        auto take=[&](std::size_t j){if(j<state.size()){
            for(int k=0;k<Z;k++) neighbors[k]+=state[j][k];
            ++count;
        }};
        if(x) take(i-1);
        if(x+1<s) take(i+1);
        if(y) take(i-s);
        if(y+1<s) take(i+s);
        if(z) take(i-ss);
        if(z+1<s) take(i+ss);
        for(int k=0;k<Z;k++){
            const float v=frozen_base[i][k]+(count?.18f*neighbors[k]/count:0.f)+.08f*geom[i].bulb;
            result[i][k]=.68f*state[i][k]+.32f*std::tanh(v);
        }
    }
    return result;
}
std::vector<float*> params(Network& n) {
    std::vector<float*> p;
    auto collect=[&](auto& a){for(auto& v:a)p.push_back(&v);};
    collect(n.ew);collect(n.eb);collect(n.zw);collect(n.zb);
    collect(n.dw);collect(n.db);collect(n.ow);p.push_back(&n.ob);
    return p;
}
struct GradientAudit {std::size_t count;double max_abs,max_rel;};
GradientAudit check_backprop() {
    Network base(true);
    std::array<float,FI> x{-.47f,.28f,-.37f,.51f,-.63f,.16f};
    auto loss=[&](const Network& n){const double e=double(n.forward(x).y)-x[0];return .5*e*e;};
    const auto count=params(base).size();
    check(count==269,"269 scalar ANN parameters");
    double largest_abs=0,largest_rel=0;
    constexpr float h=.002f,lr=.02f;
    for(std::size_t i=0;i<count;i++) {
        Network a=base,b=base,c=base;
        auto pa=params(a),pb=params(b),pc=params(c);
        *pa[i]+=h;*pb[i]-=h;
        const double numeric=(loss(a)-loss(b))/(2.*h);
        const double before=*pc[i];c.train_one(x,lr);
        const double analytic=(before-*pc[i])/lr;
        const double abs=std::abs(numeric-analytic);
        largest_abs=std::max(largest_abs,abs);
        largest_rel=std::max(largest_rel,abs/std::max(1e-4,std::abs(numeric)));
        check(abs<=.0007+.028*std::abs(numeric),"finite-difference ANN gradient index "+std::to_string(i));
    }
    return {count,largest_abs,largest_rel};
}
void check_geometry() {
    for(const auto n:std::array<std::size_t,12>{1,2,7,8,9,26,27,28,64,65,729,6912}){
        auto s=cube_side(n);
        check(std::uint64_t(s)*s*s>=n,"cube covers payload");
        if(s>1)check(std::uint64_t(s-1)*(s-1)*(s-1)<n,"cube side minimal");
        for(std::size_t i=0;i<n;i++) {
            Point p=position(i,s);
            check(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"finite 3D coordinates");
            check(p.x>=-1.001f&&p.x<=1.001f&&p.y>=-1.001f&&p.y<=1.001f&&p.z>=-1.001f&&p.z<=1.001f,"bounded 3D coordinates");
            auto bulb=mandelbulb(p);
            check(std::isfinite(bulb)&&bulb>=-1.001f&&bulb<=1.001f,"finite bounded Mandelbulb feature");
            const auto xx=i%s,yy=(i/s)%s,zz=i/(std::size_t(s)*s);
            check(xx+std::size_t(s)*yy+std::size_t(s)*s*zz==i,"3D index reversible");
        }
    }
}

struct RecurrenceResult {float max_step_4, max_step_24, fixed_residual, contraction_ratio, replay_max;};
RecurrenceResult check_recurrence(const Archive& a,const Bytes& payload,
                                  const fs::path& dest) {
    check(a.latent.size()==payload.size(),"recurrence payload size");
    std::vector<Lat> initial(payload.size());
    for(std::size_t i=0;i<payload.size();i++) initial[i]=a.net.forward(features(payload[i],a.geom[i])).z;
    auto current=initial;
    std::mt19937 rng(123458u);std::uniform_real_distribution<float> jitter(-.2f,.2f);
    auto perturbed=current;
    for(auto& lat:perturbed)for(auto& x:lat)x+=jitter(rng);
    float last_difference=maxnorm(current,perturbed),worst_ratio=0;
    std::ofstream metrics(dest/"recurrence.csv"),trace(dest/"spatial_trace.csv");
    check(bool(metrics)&&bool(trace),"recurrence CSV open");
    metrics<<"round,max_step,max_fixedpoint_residual,perturbation_distance,empirical_ratio\n";
    trace<<"round,voxel,x,y,z,bulb,byte,latent0,latent1,latent2,latent3\n";
    const int steps=24;
    float delta4=0,delta24=0;
    auto record=[&](int t,const std::vector<Lat>& state){
        // Fixed geometric subset: all occupied voxels of the 9^3 audit cube.
        for(std::size_t i=0;i<state.size();++i){
            const auto& cell=a.geom[i];
            trace<<t<<","<<i<<","<<cell.p.x<<","<<cell.p.y<<","<<cell.p.z
                 <<","<<cell.bulb<<","<<unsigned(payload[i]);
            for(float v:state[i])trace<<","<<v;
            trace<<"\n";
        }
    };
    record(0,current);
    for(int t=1;t<=steps;t++){
        auto next=spatial_step(current,initial,a.geom,a.side);
        auto pertnext=spatial_step(perturbed,initial,a.geom,a.side);
        auto step=maxnorm(next,current);
        if(t==4) delta4=step;
        if(t==steps)delta24=step;
        auto dist=maxnorm(next,pertnext);
        float ratio=last_difference>0?dist/last_difference:0;
        worst_ratio=std::max(worst_ratio,ratio);
        check(dist<=.7376f*last_difference+3e-6f,"contractive two-trajectory bound at round "+std::to_string(t));
        auto nextnext=spatial_step(next,initial,a.geom,a.side);
        const float residual=maxnorm(nextnext,next);
        metrics<<t<<","<<step<<","<<residual<<","<<dist<<","<<ratio<<"\n";
        current.swap(next);perturbed.swap(pertnext);last_difference=dist;
        record(t,current);
    }
    auto original4=initial;
    dm::recursive_refine(original4,a.geom,a.side,4);
    auto manual4=initial;
    for(int i=0;i<4;i++)manual4=spatial_step(manual4,initial,a.geom,a.side);
    const float replay=maxnorm(original4,manual4);
    check(replay<=1e-7f,"4-step original engine / independent auditor equivalence");
    const float archive_match=maxnorm(original4,a.latent);
    check(archive_match<=1e-7f,"archive stores four-step recurrence result");
    auto finalres=maxnorm(spatial_step(current,initial,a.geom,a.side),current);
    check(finalres<.00005f,"24-step approximate fixed point");
    return {delta4,delta24,finalres,worst_ratio,replay};
}

struct LoopResult {std::string name;std::size_t original, archive; double ratio, rmse, seconds;int predictor_mismatches;std::uint64_t hash;};
LoopResult roundtrip_cycles(const std::string& name,const Bytes& original,
                           const fs::path& dest,int epochs=2,int cycles=3) {
    auto start=Clock::now();
    Bytes input=original;
    Bytes firstArchive;
    std::size_t size=0;
    double rmse=0;
    int mismatches=0;
    std::uint64_t hash=0;
    for(int k=0;k<cycles;k++) {
        auto med=parse_media(name,input);
        auto archive=encode(med,epochs);
        auto path=dest/(name+"_cycle_"+std::to_string(k)+".dmq");
        save(archive,path);
        auto archived=read_file(path);
        if(k==0)firstArchive=archived;
        else check(archived==firstArchive,"nested "+name+" archive deterministic recurrence cycle "+std::to_string(k));
        auto loaded=load(path);
        auto payload=restore(loaded);
        Bytes restored=combine(loaded.media,payload);
        check(restored==original,"complete "+name+" round-trip cycle "+std::to_string(k));
        int nonmatch=0;double sq=0;
        for(std::size_t i=0;i<payload.size();++i) {
            nonmatch+=dm::quantize(loaded.net.decode(loaded.latent[i],loaded.geom[i]))!=loaded.prediction[i];
            auto diff=int(payload[i])-int(loaded.prediction[i]);sq+=double(diff)*diff;
        }
        check(nonmatch==0,"neural decoding deterministic replay for "+name);
        if(k==0) {
            size=archived.size();hash=fnv1a64(archived);
            rmse=std::sqrt(sq/payload.size());mismatches=nonmatch;
        }
        input=std::move(restored);
    }
    return {name,original.size(),size,double(size)/original.size(),rmse,seconds(start),mismatches,hash};
}

void verify_quine(const fs::path& originalSource) {
    std::ifstream f(originalSource,std::ios::binary);
    check(bool(f),"open original C++ file");
    std::ostringstream source;source<<f.rdbuf();
    std::ostringstream replication;
    auto* orig=std::cout.rdbuf(replication.rdbuf());
    try{emit_quine();}catch(...){std::cout.rdbuf(orig);throw;}
    std::cout.rdbuf(orig);
    check(replication.str()==source.str(),"complete quine source identical byte for byte");
}

struct TamperResult {bool magic_rejected=false,truncate_rejected=false,residual_tamper_undetected=false;};
TamperResult tamper_audit(const fs::path& dir) {
    const auto path=dir/"text_cycle_0.dmq";
    auto ref=read_file(path);
    auto attempt=[&](Bytes b, const std::string& basename){
        auto file=dir/basename;write_file(file,b);
        try{return std::pair<bool,Bytes>{true,restore(load(file))};}
        catch(const std::exception&){return std::pair<bool,Bytes>{false,{}};}
    };
    Bytes badMagic=ref;badMagic[0]^=0x80;
    auto mg=attempt(badMagic,"tampered_magic.dmq");
    Bytes truncated=ref;truncated.pop_back();
    auto tr=attempt(truncated,"tampered_truncated.dmq");
    Bytes residual=ref;residual.back()^=0x01;
    auto changed=attempt(residual,"tampered_residual.dmq");
    auto original=restore(load(path));
    check(!mg.first,"invalid archive magic rejected");
    check(!tr.first,"truncated archive rejected");
    check(changed.first&&changed.second!=original,"demonstrate missing residual integrity check");
    return {true,true,true};
}

int main_impl(int argc,char** argv) {
    fs::path dest=argc>=2?argv[1]:"audit_outputs";
    fs::path originalCpp=argc>=3?argv[2]:"dm_mandelquine.cpp";
    fs::create_directories(dest);
    std::ofstream report(dest/"audit_report.txt");
    check(bool(report),"report writable");
    auto log=[&](const std::string& line){std::cout<<line<<"\n";report<<line<<"\n";};
    log("Dr Moagi 3D Recursive Mandelbulb ANN Quine: independent inward verification");
    log("Scope: original supplied C++20 source included verbatim; frozen-base 3D ANN recurrence");
    check_geometry();log("PASS 3D index inverse, bounding box, minimal cube mapping and fractal finite-values (12 shapes)");
    const auto g=check_backprop();
    log("PASS finite difference checks for all "+std::to_string(g.count)+" trainable parameters; max absolute gradient error="+std::to_string(g.max_abs));
    verify_quine(originalCpp);log("PASS self-generated quine exactly matches original C++ source bytes");

    auto base=fs::path(originalCpp).parent_path()/"demo_outputs";
    // The actual original demo assets are read, not invented or approximated.
    Bytes text=read_file(base/"input.txt");
    Bytes image=read_file(base/"input.ppm");
    Bytes audio=read_file(base/"input.wav");
    Bytes binary(729);
    for(std::size_t i=0;i<binary.size();++i) {
        auto x=i%9,y=(i/9)%9,z=i/81;
        binary[i]=Byte((x*13+y*29+z*47+(x*y+3*z*x)%71)&255);
    }
    std::ofstream summary(dest/"loop_summary.csv");
    summary<<"kind,source_bytes,archive_bytes,archive_to_source_ratio,neural_rmse_byte_domain,nested_passes,predictor_mismatches,elapsed_seconds,fnv64\n";
    std::vector<LoopResult> results;
    for(auto& pair:std::array<std::pair<std::string,Bytes>,4>{{{"text",text},{"image",image},{"audio",audio},{"bytes",binary}}}) {
        auto r=roundtrip_cycles(pair.first,pair.second,dest,2,3);results.push_back(r);
        summary<<r.name<<","<<r.original<<","<<r.archive<<","<<r.ratio<<","<<r.rmse<<",3,"<<r.predictor_mismatches<<","<<r.seconds<<","<<hex64(r.hash)<<"\n";
        std::ostringstream line;line<<"PASS "<<r.name<<" 3 inward archive cycles, exact restored source; ANN RMSE="
          <<std::fixed<<std::setprecision(4)<<r.rmse<<" byte units; archive/source="<<r.ratio
          <<"; elapsed="<<r.seconds<<" s";log(line.str());
    }
    // Empty/near-boundary layouts, independent of modality wrappers.
    for(const auto n:std::array<std::size_t,5>{1,2,8,27,65}) {
        Bytes b(n);for(std::size_t i=0;i<n;i++)b[i]=Byte((i*91+7)&255);
        auto r=roundtrip_cycles("bytes",b,dest,1,2);
        log("PASS edge payload="+std::to_string(n)+" bytes, 2 archive cycles");
    }
    // Recreate audit cube archive after edge cases overwrite the same path.
    auto auditCube=encode(parse_media("bytes",binary),2);
    save(auditCube,dest/"cube_729.dmq");
    auto cubeLoaded=load(dest/"cube_729.dmq");
    auto spatial=check_recurrence(cubeLoaded,binary,dest);
    std::ostringstream rec;
    rec<<std::setprecision(9)<<"PASS spatial recurrence: original 4-step replay="<<spatial.replay_max
       <<", 4th step="<<spatial.max_step_4<<", 24th step="<<spatial.max_step_24
       <<", fixed-point residual="<<spatial.fixed_residual
       <<", worst empirical two-trajectory contraction="<<spatial.contraction_ratio<<" (bound 0.7376)";
    log(rec.str());
    // Fresh text re-encoding restores the correct test archive, after edge-case overwrite.
    auto textArc=encode(parse_media("text",text),2);
    save(textArc,dest/"text_cycle_0.dmq");
    auto tamper=tamper_audit(dest);
    if(tamper.magic_rejected&&tamper.truncate_rejected)log("PASS malformed magic/truncation rejected");
    if(tamper.residual_tamper_undetected)log("SECURITY GAP OBSERVED: last-byte residual bit tampering changes decoded content without detection");
    // Generative cascade (not a neural fixed point): reencode each generated image twice.
    Bytes generative=image;
    std::ofstream gen(dest/"generative_cascade.csv");
    gen<<"generation,output_bytes,mean_abs_pixel_change,fnv64\n";
    for(int i=0;i<3;i++){
        auto med=parse_media("image",generative);
        auto arc=encode(med,2);
        auto produced=combine(arc.media,generate(arc,777u,.4f));
        check(produced.size()==generative.size(),"valid output image size");
        double mad=0;
        auto p=arc.media.prefix.size();
        for(std::size_t j=p;j<produced.size();j++)mad+=std::abs(int(produced[j])-int(generative[j]));
        mad/=(produced.size()-p);
        gen<<i+1<<","<<produced.size()<<","<<mad<<","<<hex64(fnv1a64(produced))<<"\n";
        generative=std::move(produced);
        log("OBSERVE generated re-encoding pass "+std::to_string(i+1)+": mean absolute pixel change="+std::to_string(mad));
    }
    write_file(dest/"image_after_3_generations.ppm",generative);
    log("AUDIT COMPLETE. Proof: 0.68+0.32*0.18=0.7376 < 1 for fixed-base 3D recurrence.");
    log("NOTE: full archival exactness is XOR residual-driven; ANN alone is lossy; archive lacks integrity MAC/checksum.");
    return 0;
}
} // namespace audit

int main(int argc,char** argv) {
    try{return audit::main_impl(argc,argv);}
    catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
