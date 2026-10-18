// JARVIS-X EM-aware 3D autoencoding and CTR execution proxy.
// This simulates numerical computation and estimates hardware costs; it is not a Maxwell solver.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dm3d {
struct Config {
    std::size_t edge = 16;
    std::size_t iterations = 48;
    double gamma = 0.50;
    double tolerance = 1e-7;
    double vdd = 0.9;             // assumed supply voltage [V]
    double capacitance_ff = 1.0; // assumed effective capacitance / toggle [fF]
    double toggles_per_op = 100;  // illustrative switched node transitions / algorithmic FLOP
    double memory_pj_per_byte = 5.0; // assumed memory transfer energy [pJ/byte]
    double gflops = 10.0;        // assumed effective throughput [GFLOP/s]
    double gbps = 5.0;          // assumed effective bandwidth [GB/s]
    bool exact_residual = true;
};

struct Meter {
    std::uint64_t float_ops = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    void work(std::uint64_t ops, std::uint64_t r, std::uint64_t w) {
        float_ops += ops;
        bytes_read += r;
        bytes_written += w;
    }
};

struct Report {
    Config cfg;
    Meter meter;
    std::size_t steps = 0;
    bool fixed_point = false;
    bool ctr_pass = false;
    double initial_latent_error = 0;
    double latent_inf_error = 0;
    double omega_inf_error = 0;
    double base_mse = 0;
    double reconstructed_mse = 0;
    double model_logic_j = 0;
    double model_memory_j = 0;
    double model_total_j = 0;
    double model_seconds = 0;
    double measured_wall_seconds = 0;
    std::vector<double> latent_history;
    std::vector<float> middle_slice;
};

std::size_t parse_count(const std::string& raw, std::size_t low, std::size_t high, const char* name) {
    if (raw.empty() || raw.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(name) + " must be an unsigned decimal integer");
    std::size_t taken = 0;
    const unsigned long long number = std::stoull(raw, &taken, 10);
    if (taken != raw.size() || number < low || number > high)
        throw std::invalid_argument(std::string(name) + " outside allowed range");
    return static_cast<std::size_t>(number);
}

double parse_real(const std::string& raw, double low, double high, const char* name) {
    std::size_t taken = 0;
    const double number = std::stod(raw, &taken);
    if (taken != raw.size() || !std::isfinite(number) || !(number > low && number <= high))
        throw std::invalid_argument(std::string(name) + " must be finite and in its allowed interval");
    return number;
}

std::size_t at(std::size_t x, std::size_t y, std::size_t z, std::size_t n) {
    return x + n * (y + n * z);
}

float sample(std::size_t x, std::size_t y, std::size_t z, std::size_t n) {
    const double ax = 6.2831853071795864769 * static_cast<double>(x) / static_cast<double>(n);
    const double ay = 6.2831853071795864769 * static_cast<double>(y) / static_cast<double>(n);
    const double az = 6.2831853071795864769 * static_cast<double>(z) / static_cast<double>(n);
    return static_cast<float>(0.5 + 0.22 * std::sin(ax) + 0.17 * std::cos(ay + az)
                              + 0.08 * std::sin(2.0 * az + ax));
}

Report run(const Config& cfg) {
    if (cfg.edge < 4 || cfg.edge > 128 || cfg.edge % 2 != 0 ||
        cfg.iterations < 1 || cfg.iterations > 256 ||
        !std::isfinite(cfg.gamma) || cfg.gamma <= 0 || cfg.gamma >= 1 ||
        !std::isfinite(cfg.tolerance) || cfg.tolerance <= 0 || cfg.tolerance > 1 ||
        !std::isfinite(cfg.vdd) || cfg.vdd <= 0 || cfg.vdd > 5 ||
        !std::isfinite(cfg.capacitance_ff) || cfg.capacitance_ff <= 0 ||
        !std::isfinite(cfg.toggles_per_op) || cfg.toggles_per_op <= 0 ||
        !std::isfinite(cfg.memory_pj_per_byte) || cfg.memory_pj_per_byte <= 0 ||
        !std::isfinite(cfg.gflops) || cfg.gflops <= 0 ||
        !std::isfinite(cfg.gbps) || cfg.gbps <= 0)
        throw std::invalid_argument("invalid configuration");
    const auto clock0 = std::chrono::steady_clock::now();
    Report result;
    result.cfg = cfg;
    Meter& m = result.meter;
    const std::size_t n = cfg.edge, b = n / 2, count = n*n*n, latent_count = b*b*b;
    std::vector<float> source(count), target(latent_count, 0.f), latent(latent_count, 0.f);
    std::vector<float> next(latent_count, 0.f), omega(latent_count, 0.f);
    // Generate a reproducible volumetric source; real multimodal I/O is out of scope.
    for (std::size_t z=0; z<n; ++z)
      for (std::size_t y=0; y<n; ++y)
        for (std::size_t x=0; x<n; ++x)
          source[at(x,y,z,n)] = sample(x,y,z,n);
    m.work(0,0,static_cast<std::uint64_t>(4*count));
    // E: 2x2x2 block averaging. The latent has one eighth as many float elements.
    for (std::size_t z=0; z<b; ++z)
      for (std::size_t y=0; y<b; ++y)
        for (std::size_t x=0; x<b; ++x) {
          double value = 0;
          for (std::size_t dz=0; dz<2; ++dz)
            for (std::size_t dy=0; dy<2; ++dy)
              for (std::size_t dx=0; dx<2; ++dx)
                value += source[at(2*x+dx,2*y+dy,2*z+dz,n)];
          target[at(x,y,z,b)] = static_cast<float>(value/8.0);
        }
    m.work(8*latent_count, 8*4*latent_count, 4*latent_count);
    double max_err = 0;
    for (float v:target) max_err = std::max(max_err, static_cast<double>(std::abs(v)));
    result.initial_latent_error = max_err;
    // Coupled stable recurrence with Omega feedback, e=z-target and w=Omega-target:
    // e_next=gamma*(0.75*e+0.20*mean_6(e)+0.05*w).
    // w_next=0.35*w+0.65*e_next. Both are contractive in joint infinity norm.
    for (std::size_t pass=0; pass<cfg.iterations; ++pass) {
      double pass_inf = 0;
      for (std::size_t z=0; z<b; ++z)
        for (std::size_t y=0; y<b; ++y)
          for (std::size_t x=0; x<b; ++x) {
            const std::size_t i = at(x,y,z,b);
            const std::size_t xp=(x+1)%b, xm=(x+b-1)%b;
            const std::size_t yp=(y+1)%b, ym=(y+b-1)%b;
            const std::size_t zp=(z+1)%b, zm=(z+b-1)%b;
            const double e = static_cast<double>(latent[i])-target[i];
            const auto residual = [&](std::size_t j) { return static_cast<double>(latent[j])-target[j]; };
            const double neighbor = (residual(at(xp,y,z,b))+residual(at(xm,y,z,b))+
                         residual(at(x,yp,z,b))+residual(at(x,ym,z,b))+
                         residual(at(x,y,zp,b))+residual(at(x,y,zm,b)))/6.0;
            const double omega_error = static_cast<double>(omega[i])-target[i];
            next[i] = static_cast<float>(static_cast<double>(target[i]) +
                                        cfg.gamma * (0.75*e + 0.20*neighbor + 0.05*omega_error));
            // Omega is a coupled numerical feedback state, not a magnetic field.
            omega[i] = static_cast<float>(0.35*omega[i] + 0.65*next[i]);
            pass_inf = std::max(pass_inf, std::abs(static_cast<double>(next[i])-target[i]));
          }
      m.work(29*latent_count, 17*4*latent_count, 2*4*latent_count);
      latent.swap(next);
      result.latent_history.push_back(pass_inf);
      result.steps = pass+1;
      result.latent_inf_error = pass_inf;
      double omega_inf=0;
      for (std::size_t i=0; i<latent_count; ++i)
          omega_inf=std::max(omega_inf,std::abs(static_cast<double>(omega[i])-target[i]));
      result.omega_inf_error=omega_inf;
      m.work(latent_count, 2*4*latent_count, 0);
      if (pass_inf <= cfg.tolerance && omega_inf <= cfg.tolerance) break;
    }
    result.fixed_point = result.latent_inf_error <= cfg.tolerance &&
                         result.omega_inf_error <= cfg.tolerance;
    // D: nearest-block synthesis. Residual storage is an optional exact reconstruction
    // side-channel. It is NOT compression (latent + float residual exceeds source size).
    double base_sse=0, corrected_sse=0;
    result.middle_slice.resize(n*n);
    for (std::size_t z=0; z<n; ++z)
      for (std::size_t y=0; y<n; ++y)
        for (std::size_t x=0; x<n; ++x) {
          const std::size_t i=at(x,y,z,n);
          const float base=latent[at(x/2,y/2,z/2,b)];
          const double difference=static_cast<double>(source[i])-base;
          const float residual=static_cast<float>(difference);
          const float reconstructed = cfg.exact_residual ? base+residual : base;
          const double corrected = static_cast<double>(source[i])-reconstructed;
          base_sse += difference*difference;
          corrected_sse += corrected*corrected;
          if (z==n/2) result.middle_slice[at(x,y,0,n)]=reconstructed;
        }
    m.work(4*count, 3*4*count, static_cast<std::uint64_t>(4*count*(cfg.exact_residual ? 2 : 1)));
    result.base_mse=base_sse/static_cast<double>(count);
    result.reconstructed_mse=corrected_sse/static_cast<double>(count);
    result.ctr_pass=result.fixed_point && result.reconstructed_mse <=
                                          (cfg.exact_residual ? 1e-12 : result.base_mse + 1e-12);
    // Hardware assumptions, never measured: estimate only (not a SPICE/EM simulation).
    result.model_logic_j = static_cast<double>(m.float_ops)*cfg.toggles_per_op *
                        cfg.capacitance_ff*1e-15 * cfg.vdd*cfg.vdd;
    result.model_memory_j = static_cast<double>(m.bytes_read+m.bytes_written) *
                            cfg.memory_pj_per_byte*1e-12;
    result.model_total_j = result.model_logic_j+result.model_memory_j;
    result.model_seconds = static_cast<double>(m.float_ops)/(cfg.gflops*1e9) +
                           static_cast<double>(m.bytes_read+m.bytes_written)/(cfg.gbps*1e9);
    result.measured_wall_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-clock0).count();
    return result;
}

std::string json(const Report& r) {
    std::ostringstream s;
    s << std::setprecision(16);
    s << "{\n  \"schema\": \"jarvisx.em_aware_3d.v1\",\n"
      << "  \"provenance\": \"numerical-execution-with-assumption-based-energy-estimate\",\n"
      << "  \"physical_em_fields_simulated\": false,\n"
      << "  \"edge\": " << r.cfg.edge << ",\n"
      << "  \"voxels\": " << r.cfg.edge*r.cfg.edge*r.cfg.edge << ",\n"
      << "  \"latent_voxels\": " << r.cfg.edge*r.cfg.edge*r.cfg.edge/8 << ",\n"
      << "  \"exact_residual\": " << (r.cfg.exact_residual?"true":"false") << ",\n"
      << "  \"steps\": " << r.steps << ",\n"
      << "  \"ctr\": {\"fixed_point\": " << (r.fixed_point?"true":"false")
      << ", \"pass\": " << (r.ctr_pass?"true":"false")
      << ", \"initial_latent_inf_error\": " << r.initial_latent_error
      << ", \"final_latent_inf_error\": " << r.latent_inf_error
      << ", \"final_omega_inf_error\": " << r.omega_inf_error
      << ", \"base_mse\": " << r.base_mse
      << ", \"reconstructed_mse\": " << r.reconstructed_mse << "},\n"
      << "  \"work\": {\"modeled_float_ops\": " << r.meter.float_ops
      << ", \"modeled_bytes_read\": " << r.meter.bytes_read
      << ", \"modeled_bytes_written\": " << r.meter.bytes_written << "},\n"
      << "  \"assumptions\": {\"vdd_v\": " << r.cfg.vdd
      << ", \"effective_capacitance_ff\": " << r.cfg.capacitance_ff
      << ", \"node_toggles_per_float_op\": " << r.cfg.toggles_per_op
      << ", \"memory_pj_per_byte\": " << r.cfg.memory_pj_per_byte
      << ", \"effective_gflops\": " << r.cfg.gflops
      << ", \"effective_gbps\": " << r.cfg.gbps << "},\n"
      << "  \"energy_estimates_j\": {\"logic\": " << r.model_logic_j
      << ", \"memory\": " << r.model_memory_j
      << ", \"total\": " << r.model_total_j << "},\n"
      << "  \"modeled_seconds\": " << r.model_seconds << ",\n"
      << "  \"measured_wall_seconds\": " << r.measured_wall_seconds << ",\n"
      << "  \"latent_trace_inf\": [";
    for (std::size_t i=0; i<r.latent_history.size(); ++i)
        s << (i?", ":"") << r.latent_history[i];
    s << "]\n}\n";
    return s.str();
}

void slice_pgm(const std::string& path, const Report& r) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write PGM: "+path);
    const std::size_t n=r.cfg.edge;
    out << "P5\n" << n << ' ' << n << "\n255\n";
    for (float v:r.middle_slice) {
        const unsigned char pixel=static_cast<unsigned char>(std::lround(255.0*std::clamp(static_cast<double>(v),0.0,1.0)));
        out.write(reinterpret_cast<const char*>(&pixel),1);
    }
    if (!out) throw std::runtime_error("PGM write failure");
}

void require(bool cond, const std::string& description) {
    if (!cond) throw std::runtime_error("self-test: "+description);
}

void self_test() {
    Config c;
    c.edge=8;
    c.iterations=64;
    c.tolerance=1e-7;
    auto a=run(c);
    require(a.steps>0 && a.steps<=c.iterations,"step limit");
    require(a.fixed_point,"toroidal contraction converges");
    require(a.ctr_pass,"CTR passes on exact residual");
    require(a.base_mse>1e-6,"base reconstruction is lossy");
    require(a.reconstructed_mse<=1e-12,"exact residual reconstruction");
    require(a.latent_history.back()<a.initial_latent_error,"residual decreases");
    for (std::size_t i=1;i<a.latent_history.size();++i)
        require(a.latent_history[i]<=a.latent_history[i-1]+2e-7,"non-increasing coupled latent residual");
    require(a.omega_inf_error<=c.tolerance,"Omega feedback converges");
    Config nores=c;
    nores.exact_residual=false;
    auto b=run(nores);
    require(b.reconstructed_mse>1e-6 && b.ctr_pass,"lossy mode identified and validated");
    Config voltage=c;
    voltage.vdd=1.8;
    auto d=run(voltage);
    require(std::abs(d.model_logic_j/a.model_logic_j-4.0)<1e-8,"V squared dynamic switching energy");
    require(std::abs(d.model_memory_j-a.model_memory_j)<1e-18,"voltage-independent memory assumption");
    require(a.meter.float_ops>0 && a.meter.bytes_read>0 && a.meter.bytes_written>0,"meter populated");
    require(at(7,7,7,8)==511,"3D row-major addressing");
    try { (void)parse_count("-1",1,100,"iterations"); throw std::runtime_error("negative count accepted"); }
    catch (const std::invalid_argument&) {}
    try { (void)parse_count("18446744073709551616",1,100,"iterations"); throw std::runtime_error("overflow accepted"); }
    catch (const std::out_of_range&) {}
    try { (void)parse_real("nan",0,1,"gamma"); throw std::runtime_error("NaN accepted"); }
    catch (const std::invalid_argument&) {}
    std::cout << "PASS: toroidal convergence, CTR, modes, V^2 scaling, metering, indexing, bad input\n";
}
}

int main(int argc,char** argv) {
  try {
    dm3d::Config c;
    std::string receipt, pgm;
    for (int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if (key=="--self-test") { dm3d::self_test(); return 0; }
        if (key=="--help") {
            std::cout << "Usage: jarvisx-em-aware-3d [--edge EVEN_4_TO_128] [--iterations 1_TO_256] "
                         "[--gamma 0_TO_1] [--tol 0_TO_1] [--voltage V] [--cap-ff fF] "
                         "[--toggles-per-op N] [--mem-pj-per-byte N] [--gflops N] [--gbps N] "
                         "[--no-residual] [--receipt output.json] [--slice output.pgm] [--self-test]\n";
            return 0;
        }
        if (key=="--no-residual") {c.exact_residual=false; continue;}
        if (++i>=argc) throw std::invalid_argument("missing value for "+key);
        const std::string value=argv[i];
        if (key=="--edge") c.edge=dm3d::parse_count(value,4,128,"edge");
        else if (key=="--iterations") c.iterations=dm3d::parse_count(value,1,256,"iterations");
        else if (key=="--gamma") c.gamma=dm3d::parse_real(value,0,0.999999,"gamma");
        else if (key=="--tol") c.tolerance=dm3d::parse_real(value,0,1,"tolerance");
        else if (key=="--voltage") c.vdd=dm3d::parse_real(value,0,5,"voltage");
        else if (key=="--cap-ff") c.capacitance_ff=dm3d::parse_real(value,0,1e9,"capacitance");
        else if (key=="--toggles-per-op") c.toggles_per_op=dm3d::parse_real(value,0,1e9,"toggles");
        else if (key=="--mem-pj-per-byte") c.memory_pj_per_byte=dm3d::parse_real(value,0,1e9,"memory energy");
        else if (key=="--gflops") c.gflops=dm3d::parse_real(value,0,1e9,"gflops");
        else if (key=="--gbps") c.gbps=dm3d::parse_real(value,0,1e9,"gbps");
        else if (key=="--receipt") receipt=value;
        else if (key=="--slice") pgm=value;
        else throw std::invalid_argument("unknown option: "+key);
    }
    if (c.edge%2!=0) throw std::invalid_argument("edge must be even");
    dm3d::Report report=dm3d::run(c);
    if (!pgm.empty()) dm3d::slice_pgm(pgm,report);
    if (!receipt.empty()) {
        std::ofstream output(receipt);
        if (!output) throw std::runtime_error("cannot open receipt: "+receipt);
        output << dm3d::json(report);
        if (!output) throw std::runtime_error("receipt write failure");
    } else std::cout << dm3d::json(report);
    return report.ctr_pass ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
