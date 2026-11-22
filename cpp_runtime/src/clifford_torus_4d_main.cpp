// Jarvis-X: bounded 4D Clifford torus perspective projection and telemetry.
// C++17, no external dependencies. Not an encoder, AI model or GPU kernel.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace jarvisx {
namespace clifford {
constexpr double pi = 3.141592653589793238462643383279502884;
using Clock = std::chrono::steady_clock;
struct Vec4 { double x, y, z, w; };
struct Vec3 { double x, y, z; };
using Matrix4 = std::array<std::array<double, 4>, 4>;

struct Config {
    double radius1 = 2.0, radius2 = 1.0, projection_radius = 4.0;
    double dt = 0.015, time_scale = 1.0;
    std::size_t grid = 40, frames = 120;
    std::size_t queue_capacity = 128;
};

void validate(const Config& c) {
    if (!(std::isfinite(c.radius1) && c.radius1 > 0 &&
          std::isfinite(c.radius2) && c.radius2 > 0 &&
          std::isfinite(c.projection_radius) &&
          c.projection_radius > std::hypot(c.radius1, c.radius2)))
        throw std::invalid_argument("projection radius must exceed the 4D torus norm");
    if (!(std::isfinite(c.dt) && c.dt > 0 && c.dt <= 1 &&
          std::isfinite(c.time_scale) && c.time_scale > 0 && c.time_scale <= 1e9))
        throw std::invalid_argument("dt/time-scale outside supported finite bounds");
    if (c.grid < 2 || c.grid > 1024 || c.frames == 0 || c.frames > 1000000 ||
        c.queue_capacity == 0 || c.queue_capacity > 1000000)
        throw std::invalid_argument("grid, frame count or queue capacity outside bounds");
}

Vec4 parametric_point(double u, double v, const Config& c) {
    return {c.radius1 * std::cos(u), c.radius1 * std::sin(u),
            c.radius2 * std::cos(v), c.radius2 * std::sin(v)};
}

Vec4 rotate_planes(Vec4 p, const std::array<double, 3>& a) {
    // XW -> YW -> ZW, matching the supplied engine's rotation order.
    const double cx = std::cos(a[0]), sx = std::sin(a[0]);
    const double cy = std::cos(a[1]), sy = std::sin(a[1]);
    const double cz = std::cos(a[2]), sz = std::sin(a[2]);
    const double x = p.x * cx - p.w * sx;
    p.w = p.x * sx + p.w * cx; p.x = x;
    const double y = p.y * cy - p.w * sy;
    p.w = p.y * sy + p.w * cy; p.y = y;
    const double z = p.z * cz - p.w * sz;
    p.w = p.z * sz + p.w * cz; p.z = z;
    return p;
}

Matrix4 rotation_matrix(const std::array<double, 3>& a) {
    Matrix4 result{};
    const std::array<Vec4,4> basis{{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}};
    for (std::size_t j=0;j<4;++j) {
        const Vec4 b = rotate_planes(basis[j], a);
        result[0][j]=b.x; result[1][j]=b.y;
        result[2][j]=b.z; result[3][j]=b.w;
    }
    return result;
}

Vec4 multiply(const Matrix4& m, Vec4 p) {
    const std::array<double,4> x{{p.x,p.y,p.z,p.w}};
    std::array<double,4> y{{0,0,0,0}};
    for (std::size_t i=0;i<4;++i)
        for (std::size_t j=0;j<4;++j) y[i]+=m[i][j]*x[j];
    return {y[0],y[1],y[2],y[3]};
}

double norm2(Vec4 p) { return p.x*p.x+p.y*p.y+p.z*p.z+p.w*p.w; }

double orthogonality_error(const Matrix4& m) {
    double max_error=0;
    for (std::size_t i=0;i<4;++i)
        for (std::size_t j=0;j<4;++j) {
            double dot=0;
            for(std::size_t k=0;k<4;++k) dot+=m[k][i]*m[k][j];
            max_error=std::max(max_error,std::abs(dot-(i==j?1.0:0.0)));
        }
    return max_error;
}

Vec3 project(Vec4 p, const Config& c) {
    // Pole-free perspective mapping. Not a stereographic projection of S^3:
    // projection_radius is deliberately greater than the torus S^3 radius.
    const double scale=c.projection_radius/(c.projection_radius-p.w);
    return {p.x*scale,p.y*scale,p.z*scale};
}

struct FramePacket {
    std::uint64_t frame=0;
    Matrix4 matrix{};
    Vec3 centroid{};
    std::array<double,4> mean_abs_flux{};
    double max_norm2_error=0, max_orthogonality_error=0;
    double min_projection_denominator=0;
};

// A bounded asynchronous telemetry consumer, not an ML inference pipeline.
class TelemetryBridge {
    std::mutex mutex_;
    std::condition_variable ready_;
    std::queue<FramePacket> queue_;
    const std::size_t capacity_;
    bool closed_=false;
    std::thread worker_;
    std::atomic<std::uint64_t> processed_{0}, dropped_{0};
    std::atomic<double> flux_index_{0};
    void consume() {
        while (true) {
            FramePacket p;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock,[this]{return closed_ || !queue_.empty();});
                if (queue_.empty() && closed_) break;
                p=queue_.front(); queue_.pop();
            }
            double magnitude=0;
            for(double v:p.mean_abs_flux) magnitude+=v;
            flux_index_.store(std::tanh(magnitude/4.0));
            processed_.fetch_add(1);
        }
    }
public:
    explicit TelemetryBridge(std::size_t capacity):capacity_(capacity),worker_([this]{consume();}) {}
    TelemetryBridge(const TelemetryBridge&)=delete;
    TelemetryBridge& operator=(const TelemetryBridge&)=delete;
    ~TelemetryBridge() { finish(); }
    bool push(const FramePacket& p) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(closed_) return false;
            if(queue_.size()>=capacity_) {queue_.pop(); dropped_.fetch_add(1);}
            queue_.push(p);
        }
        ready_.notify_one(); return true;
    }
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex_); closed_=true;
        }
        ready_.notify_all();
        if(worker_.joinable()) worker_.join();
    }
    std::uint64_t processed() const {return processed_.load();}
    std::uint64_t dropped() const {return dropped_.load();}
    double flux_index() const {return flux_index_.load();}
};

struct Summary {
    std::uint64_t frames=0, vertices=0, packets_processed=0, packets_dropped=0;
    double wall_seconds=0, simulated_seconds=0, flux_index=0;
    FramePacket last{};
};

Summary simulate(const Config& c) {
    validate(c);
    TelemetryBridge bridge(c.queue_capacity);
    Summary summary;
    std::array<double,3> a{{0,0,0}};
    std::vector<Vec3> mesh;
    mesh.reserve(c.grid*c.grid);
    const auto start=Clock::now();
    for(std::size_t frame=0;frame<c.frames;++frame) {
        const double scaled_dt=c.dt*c.time_scale;
        a[0]=std::remainder(a[0]+0.012*scaled_dt,2*pi);
        a[1]=std::remainder(a[1]+0.008*scaled_dt,2*pi);
        // ZW rotation disabled in the supplied defaults.
        const Matrix4 m=rotation_matrix(a);
        FramePacket packet;
        packet.frame=static_cast<std::uint64_t>(frame);
        packet.matrix=m;
        packet.min_projection_denominator=std::numeric_limits<double>::infinity();
        packet.max_orthogonality_error=orthogonality_error(m);
        mesh.clear();
        const double r2=c.radius1*c.radius1+c.radius2*c.radius2;
        for(std::size_t i=0;i<c.grid;++i) {
            const double u=(2*pi*static_cast<double>(i))/static_cast<double>(c.grid);
            for(std::size_t j=0;j<c.grid;++j) {
                const double v=(2*pi*static_cast<double>(j))/static_cast<double>(c.grid);
                const Vec4 p=multiply(m,parametric_point(u,v,c));
                const Vec3 q=project(p,c);
                packet.centroid.x+=q.x; packet.centroid.y+=q.y; packet.centroid.z+=q.z;
                const double components[4]={p.x,p.y,p.z,p.w};
                for(std::size_t k=0;k<4;++k) packet.mean_abs_flux[k]+=std::abs(components[k]);
                packet.max_norm2_error=std::max(packet.max_norm2_error,std::abs(norm2(p)-r2));
                packet.min_projection_denominator=std::min(packet.min_projection_denominator,c.projection_radius-p.w);
                mesh.push_back(q);
            }
        }
        const double inv_count=1.0/static_cast<double>(mesh.size());
        packet.centroid.x*=inv_count; packet.centroid.y*=inv_count; packet.centroid.z*=inv_count;
        for(double& flux:packet.mean_abs_flux) flux*=inv_count;
        if (!(std::isfinite(packet.centroid.x)&&std::isfinite(packet.centroid.y)&&
              std::isfinite(packet.centroid.z)&&packet.min_projection_denominator>0))
            throw std::runtime_error("nonfinite projected geometry");
        if (!bridge.push(packet)) throw std::runtime_error("telemetry bridge unexpectedly closed");
        summary.last=packet;
        summary.vertices+=static_cast<std::uint64_t>(mesh.size());
    }
    bridge.finish();
    summary.wall_seconds=std::chrono::duration<double>(Clock::now()-start).count();
    summary.frames=static_cast<std::uint64_t>(c.frames);
    summary.simulated_seconds=static_cast<double>(c.frames)*c.dt*c.time_scale;
    summary.packets_processed=bridge.processed();
    summary.packets_dropped=bridge.dropped();
    summary.flux_index=bridge.flux_index();
    if(summary.packets_processed+summary.packets_dropped!=summary.frames)
        throw std::runtime_error("telemetry accounting invariant failed");
    return summary;
}


/*
 * NDJSON frame protocol for the optional local visualizer.
 * One complete finite projected mesh per line, plus actual measured compute
 * throughput. Simulated time_scale is explicitly NOT a compute speedup.
 */
void stream_json(const Config& c, unsigned pace_ms) {
    validate(c);
    if (c.grid > 64) throw std::invalid_argument("stream grid must be <=64");
    if (pace_ms > 1000) throw std::invalid_argument("pace-ms must be <=1000");
    std::array<double,3> angles{{0,0,0}};
    std::vector<Vec3> mesh;
    mesh.reserve(c.grid*c.grid);
    // Precompute fixed torus samples once; no trig calls in the frame loop.
    // Excluded from per-frame compute_seconds; accounted for as setup cost.
    std::vector<Vec4> source_grid;
    source_grid.reserve(c.grid*c.grid);
    for (std::size_t i=0;i<c.grid;++i) {
        const double u=2*pi*static_cast<double>(i)/static_cast<double>(c.grid);
        for(std::size_t j=0;j<c.grid;++j) {
            const double v=2*pi*static_cast<double>(j)/static_cast<double>(c.grid);
            source_grid.push_back(parametric_point(u,v,c));
        }
    }
    const double target_norm2 = c.radius1*c.radius1+c.radius2*c.radius2;
    const double scaled_dt = c.dt*c.time_scale;
    std::cout << std::setprecision(12);
    for (std::size_t frame=0;frame<c.frames;++frame) {
        const auto started=Clock::now();
        angles[0]=std::remainder(angles[0]+0.012*scaled_dt,2*pi);
        angles[1]=std::remainder(angles[1]+0.008*scaled_dt,2*pi);
        const Matrix4 matrix=rotation_matrix(angles);
        const double orth_error=orthogonality_error(matrix);
        double norm_error=0;
        double min_denom=std::numeric_limits<double>::infinity();
        mesh.clear();
        for (const Vec4& base : source_grid) {
                const Vec4 p=multiply(matrix,base);
                norm_error=std::max(norm_error,std::abs(norm2(p)-target_norm2));
                min_denom=std::min(min_denom,c.projection_radius-p.w);
                const Vec3 q=project(p,c);
                if (!(std::isfinite(q.x)&&std::isfinite(q.y)&&std::isfinite(q.z)))
                    throw std::runtime_error("nonfinite projection in streamed frame");
                mesh.push_back(q);
        }
        if(!(min_denom>0 && std::isfinite(norm_error) && std::isfinite(orth_error)))
            throw std::runtime_error("invalid streamed geometry invariant");
        const double compute_s=std::max(
            std::chrono::duration<double>(Clock::now()-started).count(),1e-12);
        const auto serialization_started=Clock::now();
        std::ostringstream line;
        line << std::setprecision(12)
             << "{\"protocol\":\"jx-clifford-frame-v1\",\"frame\":" << frame
             << ",\"simulated_s\":" << (static_cast<double>(frame)+1)*scaled_dt
             << ",\"time_scale\":" << c.time_scale << ",\"dt\":" << c.dt
             << ",\"grid\":" << c.grid
             << ",\"radii\":[" << c.radius1 << ',' << c.radius2 << ']'
             << ",\"projection_radius\":" << c.projection_radius
             << ",\"angles\":[" << angles[0] << ',' << angles[1] << ',' << angles[2] << ']'
             << ",\"compute_seconds\":" << compute_s
             << ",\"measured_vertices_per_s\":" << static_cast<double>(mesh.size())/compute_s
             << ",\"max_norm2_error\":" << norm_error
             << ",\"orthogonality_error\":" << orth_error
             << ",\"min_projection_denominator\":" << min_denom
             << ",\"vertices\":[";
        for(std::size_t i=0;i<mesh.size();++i) {
            if(i) line << ',';
            line << mesh[i].x << ',' << mesh[i].y << ',' << mesh[i].z;
        }
        const double json_encode_s=std::max(
            std::chrono::duration<double>(Clock::now()-serialization_started).count(),0.0);
        line << "],\"json_encode_seconds\":" << json_encode_s << '}';
        std::cout << line.str() << '\n' << std::flush;
        if(!std::cout) return; // transport disconnected
        if(pace_ms) std::this_thread::sleep_for(std::chrono::milliseconds(pace_ms));
    }
}

void require(bool test, const char* msg) {if(!test) throw std::runtime_error(msg);}
void self_test() {
    Config c; validate(c);
    const Vec4 p=parametric_point(.23,1.71,c);
    const std::array<double,3> a{{.17,-.64,.29}};
    const Vec4 r=rotate_planes(p,a);
    require(std::abs(norm2(p)-norm2(r))<1e-12,"norm preservation");
    const auto m=rotation_matrix(a);
    const Vec4 r2=multiply(m,p);
    require(std::abs(r.x-r2.x)+std::abs(r.y-r2.y)+std::abs(r.z-r2.z)+std::abs(r.w-r2.w)<1e-12,"matrix composition");
    require(orthogonality_error(m)<1e-12,"orthonormality");
    const auto projected=project(parametric_point(0,0,c),c);
    require(std::abs(projected.x-2)<1e-12 && std::abs(projected.z-1)<1e-12,"projection identity");
    bool rejects_pole=false;
    try { Config invalid=c; invalid.projection_radius=std::hypot(c.radius1,c.radius2); validate(invalid); }
    catch(const std::invalid_argument&) {rejects_pole=true;}
    require(rejects_pole,"unsafe projection radius must be rejected");
    c.frames=64; c.grid=8; c.queue_capacity=3; c.time_scale=1000000;
    const auto s=simulate(c);
    require(s.frames==64 && s.vertices==4096,"bounded grid accounting");
    require(s.packets_processed+s.packets_dropped==s.frames,"bounded queue accounting");
    require(std::isfinite(s.flux_index) && s.flux_index>=0 && s.flux_index<1,"flux index finite");
    require(s.last.max_norm2_error<1e-11 && s.last.max_orthogonality_error<1e-12,"rotated torus invariants");
    require(s.last.min_projection_denominator>0,"projection denominator");
    std::cout << "PASS: 4D geometry, bounded bridge, projection safety, and frame accounting\n";
}

} // namespace clifford
} // namespace jarvisx

int main(int argc,char** argv) {
    using namespace jarvisx::clifford;
    try {
        Config c; bool test=false, streaming=false; unsigned pace_ms=33;
        for(int i=1;i<argc;++i) {
            const std::string opt=argv[i];
            auto value=[&]() -> std::string { if(++i>=argc) throw std::invalid_argument("missing option value: "+opt); return argv[i]; };
            if(opt=="--self-test") test=true;
            else if(opt=="--stream-json") streaming=true;
            else if(opt=="--pace-ms") { const auto n=std::stoull(value()); if(n>1000) throw std::invalid_argument("pace-ms must be <=1000"); pace_ms=static_cast<unsigned>(n); }
            else if(opt=="--frames") c.frames=std::stoull(value());
            else if(opt=="--grid") c.grid=std::stoull(value());
            else if(opt=="--dt") c.dt=std::stod(value());
            else if(opt=="--time-scale") c.time_scale=std::stod(value());
            else if(opt=="--queue-capacity") c.queue_capacity=std::stoull(value());
            else throw std::invalid_argument("unknown argument: "+opt);
        }
        if(test) { self_test(); return 0; }
        if(streaming) { stream_json(c,pace_ms); return 0; }
        const auto s=simulate(c);
        std::cout<<std::setprecision(9)
            <<"Clifford torus 4D -> 3D perspective telemetry\n"
            <<"frames="<<s.frames<<" vertices="<<s.vertices<<" wall_s="<<s.wall_seconds<<"\n"
            <<"measured_frames_per_s="<<(static_cast<double>(s.frames)/s.wall_seconds)
            <<" measured_vertices_per_s="<<(static_cast<double>(s.vertices)/s.wall_seconds)<<"\n"
            <<"simulated_s="<<s.simulated_seconds<<" time_scale="<<c.time_scale<<"\n"
            <<"processed="<<s.packets_processed<<" dropped="<<s.packets_dropped
            <<" flux_index="<<s.flux_index<<"\n"
            <<"max_norm2_error="<<s.last.max_norm2_error
            <<" orthogonality_error="<<s.last.max_orthogonality_error
            <<" min_projection_denominator="<<s.last.min_projection_denominator<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"clifford-torus error: "<<e.what()<<'\n'; return 1;
    }
}
