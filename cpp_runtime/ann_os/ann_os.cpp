// ann_os.cpp -- verified user-space ANN runtime (C++17)
// Build: g++ -O3 -std=c++17 -Wall -Wextra -pedantic ann_os.cpp -o ann_os
// Run:   ./ann_os --self ann_os.cpp
// Test:  ./ann_os --self ann_os.cpp --batch
// A user-space tool, NOT a bootable operating-system kernel.
#include "ann.hpp"
#include <algorithm>
#include <array>
#include <chrono>
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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static std::string read_text(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read file: " + path.string());
    std::ostringstream out;
    out << f.rdbuf();
    if (!f.eof() && f.fail()) throw std::runtime_error("File read failed");
    return out.str();
}

static void write_text(const fs::path& path, const std::string& content) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot write file: " + path.string());
    f.write(content.data(), static_cast<std::streamsize>(content.size()));
    if (!f) throw std::runtime_error("File write failed");
}

struct CodeCube {
    std::size_t D = 0;
    std::size_t logical_bytes = 0;
    std::vector<float> voxels; // x + D*(y + D*z), values in [0,1]

    static CodeCube encode(const std::string& bytes) {
        constexpr std::size_t max_edge = 64; // 64^3 float voxels, max 1 MiB per cube
        if (bytes.size() > max_edge * max_edge * max_edge)
            throw std::length_error("Source exceeds 64^3-voxel cube capacity");
        CodeCube cube;
        cube.D = 1;
        while (cube.D * cube.D * cube.D < bytes.size()) ++cube.D;
        cube.logical_bytes = bytes.size();
        cube.voxels.assign(cube.D * cube.D * cube.D, 0.0f);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            cube.voxels[i] = static_cast<unsigned char>(bytes[i]) / 255.0f;
        return cube;
    }

    std::string decode() const {
        if (D == 0 || D > 64 || voxels.size() != D * D * D ||
            logical_bytes > voxels.size())
            throw std::runtime_error("Invalid cube dimensions");
        std::string result(logical_bytes, '\0');
        for (std::size_t i = 0; i < logical_bytes; ++i) {
            const float v = voxels[i];
            if (!std::isfinite(v) || v < 0.f || v > 1.f)
                throw std::runtime_error("Invalid cube voxel");
            result[i] = static_cast<char>(std::lround(v * 255.0f));
        }
        return result;
    }

    std::size_t index(std::size_t x, std::size_t y, std::size_t z) const {
        if (x >= D || y >= D || z >= D)
            throw std::out_of_range("3D cube index out of range");
        return x + D * (y + D * z);
    }
};

struct CubeFile {
    std::string name;
    CodeCube cube;
    double last_validation_mse = 0.0;
};

class CubeFS {
    std::map<std::string, CubeFile> files_;
public:
    void write(const std::string& path, const CodeCube& cube) {
        if (path.empty() || path.front() != '/')
            throw std::invalid_argument("CubeFS paths must be absolute");
        files_[path] = CubeFile{path, cube, 0.0};
    }
    const CodeCube& read(const std::string& path) const {
        auto it = files_.find(path);
        if (it == files_.end()) throw std::out_of_range("CubeFS missing file: " + path);
        return it->second.cube;
    }
    void set_loss(const std::string& path, double loss) {
        auto it = files_.find(path);
        if (it == files_.end()) throw std::out_of_range("CubeFS missing file: " + path);
        it->second.last_validation_mse = loss;
    }
    void ls(std::ostream& out) const {
        for (const auto& item : files_)
            out << item.first << " [" << item.second.cube.D << "x"
                << item.second.cube.D << "x" << item.second.cube.D
                << ", " << item.second.cube.logical_bytes << " bytes]" << '\n';
    }
};

struct LatentTask {
    std::size_t id = 0;
    std::vector<double> latent;
};

class ANNScheduler {
    std::vector<double> goal_;
    std::vector<LatentTask> queue_;
    std::size_t next_id_ = 0;
    static double distance_squared(const std::vector<double>& a,
                                   const std::vector<double>& b) {
        double accum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double delta = a[i] - b[i];
            accum += delta * delta;
        }
        return accum;
    }
public:
    explicit ANNScheduler(std::vector<double> goal) : goal_(std::move(goal)) {
        if (goal_.empty()) throw std::invalid_argument("Scheduler goal is empty");
        for (double v : goal_) if (!std::isfinite(v))
            throw std::invalid_argument("Non-finite scheduler goal");
    }
    std::size_t push(std::vector<double> latent) {
        if (latent.size() != goal_.size())
            throw std::invalid_argument("Latent dimension mismatch");
        for (double v : latent) if (!std::isfinite(v))
            throw std::invalid_argument("Non-finite latent");
        const std::size_t id = next_id_++;
        queue_.push_back({id, std::move(latent)});
        return id;
    }
    std::optional<LatentTask> pop() {
        if (queue_.empty()) return std::nullopt;
        const auto best = std::min_element(queue_.begin(), queue_.end(),
            [this](const LatentTask& a, const LatentTask& b) {
                const double da = distance_squared(a.latent, goal_);
                const double db = distance_squared(b.latent, goal_);
                return da < db || (da == db && a.id < b.id);
            });
        LatentTask task = *best;
        queue_.erase(best);
        return task;
    }
    std::size_t size() const { return queue_.size(); }
};

using Dataset = std::vector<std::vector<double>>;
struct AutoencodeData { Dataset train_X, train_Y, valid_X, valid_Y; };

static AutoencodeData make_dataset(const CodeCube& cube, std::size_t cap = 1000) {
    // Each training sample is an actual spatial 2x2x2 voxel neighborhood.
    // Pad out-of-range voxels with zero-valued bytes, normalized to -1.
    // The cube stores the source exactly, while ANN reconstruction is lossy.
    AutoencodeData data;
    const std::size_t tile_edge = (cube.D + 1) / 2;
    const std::size_t total_tiles = tile_edge * tile_edge * tile_edge;
    const std::size_t tiles = std::min(cap, total_tiles);
    if (tiles < 6) throw std::runtime_error("Too little source data to train");
    for (std::size_t t = 0; t < tiles; ++t) {
        // Deterministic stride samples different 3D subvolumes.
        const std::size_t tile = t * total_tiles / tiles;
        const std::size_t base_x = 2 * (tile % tile_edge);
        const std::size_t base_y = 2 * ((tile / tile_edge) % tile_edge);
        const std::size_t base_z = 2 * (tile / (tile_edge * tile_edge));
        std::vector<double> sample(8, -1.0);
        for (std::size_t dz = 0; dz < 2; ++dz)
            for (std::size_t dy = 0; dy < 2; ++dy)
                for (std::size_t dx = 0; dx < 2; ++dx) {
                    const std::size_t k = dx + 2 * (dy + 2 * dz);
                    if (base_x + dx < cube.D && base_y + dy < cube.D &&
                        base_z + dz < cube.D) {
                        const std::size_t index = cube.index(
                            base_x + dx, base_y + dy, base_z + dz);
                        sample[k] = 2.0 * cube.voxels[index] - 1.0;
                    }
                }
        if (t % 5 == 0) {
            data.valid_X.push_back(sample);
            data.valid_Y.push_back(sample);
        } else {
            data.train_X.push_back(sample);
            data.train_Y.push_back(sample);
        }
    }
    if (data.valid_X.empty() || data.train_X.empty())
        throw std::runtime_error("Invalid train/validation partition");
    return data;
}

static std::string app_template(const std::string& kind) {
    if (kind == "hello") {
        return "#include <iostream>\nint main(){std::cout<<\"Hello from ANN OS\\n\";}\n";
    }
    if (kind == "xor") {
        return R"CPP(#include "ann.hpp"
#include <iostream>
#include <vector>
int main() {
    std::vector<std::vector<double>> X={{0,0},{0,1},{1,0},{1,1}};
    std::vector<std::vector<double>> Y={{0},{1},{1},{0}};
    ANN net({2,8,1}, {Activation::Tanh, Activation::Sigmoid}, 42);
    const double mse = net.train(X,Y,20000,4,0.8,1e-4);
    std::cout << "XOR MSE=" << mse << '\n';
    for(std::size_t i=0;i<X.size();++i)
        std::cout << X[i][0] << " XOR " << X[i][1] << " -> " << net.predict(X[i])[0] << '\n';
    return mse < 1e-3 ? 0 : 1;
}
)CPP";
    }
    throw std::invalid_argument("Only tested app templates 'hello' and 'xor' are supported");
}

class ANN_OS {
    CubeFS cube_fs_;
    ANNScheduler scheduler_{{0.0, 0.0, 0.0}};
    ANN autoencoder_{{8, 12, 3, 12, 8},
                     {Activation::Tanh, Activation::Tanh,
                      Activation::Tanh, Activation::Tanh}, 42};
    fs::path self_path_;
    std::string source_;
    AutoencodeData dataset_;
    std::size_t generations_ = 0;
    double boot_ms_ = 0.0;
    std::string generated_kind_;
    bool audio_enabled_ = false;
public:
    explicit ANN_OS(fs::path self_path) : self_path_(std::move(self_path)) {}

    void boot(std::ostream& out = std::cout) {
        auto start = Clock::now();
        source_ = read_text(self_path_);
        const CodeCube kernel = CodeCube::encode(source_);
        if (kernel.decode() != source_)
            throw std::runtime_error("Cube round-trip invariant violated");
        cube_fs_.write("/os/kernel.cpp", kernel);
        cube_fs_.write("/os/README.md", CodeCube::encode(
            "ANN OS: a C++17 user-space demonstrator, not an operating-system kernel.\n"));
        dataset_ = make_dataset(kernel);
        boot_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        out << "ANN OS booted: source=" << source_.size() << " bytes; cube="
            << kernel.D << "^3; train=" << dataset_.train_X.size()
            << "; validation=" << dataset_.valid_X.size()
            << "; boot=" << std::fixed << std::setprecision(3) << boot_ms_ << " ms\n";
    }

    // Verification-aware, transactional optimization: train candidate;
    // accept the entire candidate iff the held-out validation loss improves.
    bool auto_optimize(std::ostream& out = std::cout) {
        if (source_.empty()) throw std::logic_error("boot() first");
        const auto start = Clock::now();
        ++generations_;
        const double before = autoencoder_.evaluate_mse(dataset_.valid_X, dataset_.valid_Y);
        ANN candidate = autoencoder_;
        candidate.train(dataset_.train_X, dataset_.train_Y, 12, 32, 0.06);
        const double after = candidate.evaluate_mse(dataset_.valid_X, dataset_.valid_Y);
        const bool accept = std::isfinite(after) && (after < before - 1e-12);
        if (accept) autoencoder_ = std::move(candidate);
        const double retained = accept ? after : before;
        cube_fs_.set_loss("/os/kernel.cpp", retained);
        const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        out << "[gen " << generations_ << "] validation MSE " << before
            << " -> " << after << (accept ? " [KEPT]" : " [REVERTED]")
            << "; retained=" << retained << "; " << elapsed << " ms\n";
        return accept;
    }

    double validation_loss() const {
        return autoencoder_.evaluate_mse(dataset_.valid_X, dataset_.valid_Y);
    }
    std::size_t generations() const { return generations_; }
    const CubeFS& cubes() const { return cube_fs_; }
    ANNScheduler& scheduler() { return scheduler_; }

    void generate_app(const std::string& kind, std::ostream& out) {
        const std::string code = app_template(kind); // fails before modifying files
        write_text("app/main.cpp", code);
        generated_kind_ = kind;
        cube_fs_.write("/apps/" + kind + ".cpp", CodeCube::encode(code));
        out << "Generated app/main.cpp (deterministic template: " << kind << ")\n";
    }

    void debug_app(std::ostream& out) {
        if (generated_kind_.empty()) throw std::logic_error("Generate an app first");
        // Fixed compiler command, with no shell interpolation of user-provided text.
        const int result = std::system(
            "g++ -O2 -std=c++17 -Wall -Wextra -pedantic -I. app/main.cpp -o app/generated_app");
        if (result != 0) throw std::runtime_error("App compilation failed (g++ required)");
        out << "Compilation successful: app/generated_app\n";
    }

    void documentation(std::ostream& out) const {
        std::ostringstream doc;
        doc << "# ANN OS User-Space Runtime\n\n"
            << "A testable process, not a bootable operating system or code-generating LLM.\n\n"
            << "## Modules\n\n"
            << "- CubeFS: an in-memory map of reversible voxel-encoded byte files.\n"
            << "- ANN: 8 -> 12 -> 3 -> 12 -> 8 lossy autoencoder, backed by ann.hpp.\n"
            << "- Optimization: SGD on 2x2x2 byte tiles; held-out loss acceptance.\n"
            << "- Scheduler: Euclidean latent distance to explicit three-vector goal.\n"
            << "- Generation: deterministic hello and xor templates.\n"
            << "- Replication: source snapshot, not autonomous recompilation.\n\n"
            << "## Metrics\n\nGenerations: " << generations_
            << "\nHeld-out reconstruction MSE: " << validation_loss()
            << "\nSource bytes: " << source_.size() << "\n";
        write_text("os_manual.md", doc.str());
        out << "Wrote os_manual.md (Markdown; no PDF generator installed)\n";
    }

    void image(std::ostream& out) const {
        // A self-contained vector image, not a mislabeled PNG.
        const std::string svg = R"SVG(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 300">
<rect width="900" height="300" fill="#101928"/>
<g fill="#142e48" stroke="#63d8cf" stroke-width="2">
<rect x="25" y="90" width="150" height="100" rx="14"/>
<rect x="242" y="90" width="150" height="100" rx="14"/>
<rect x="459" y="90" width="150" height="100" rx="14"/>
<rect x="676" y="90" width="190" height="100" rx="14"/>
</g>
<g fill="white" font-family="sans-serif" text-anchor="middle" font-size="17">
<text x="100" y="130">Source</text><text x="100" y="158">C++ bytes</text>
<text x="317" y="130">CodeCube</text><text x="317" y="158">3D voxels</text>
<text x="534" y="130">ANN 8-3-8</text><text x="534" y="158">Reconstruct</text>
<text x="771" y="130">CTR verify</text><text x="771" y="158">Keep / revert</text>
</g>
<g stroke="#f6c177" stroke-width="4" fill="none"><path d="M175 140H240M392 140H457M609 140H674"/><path d="M771 192V250H533V192" stroke-dasharray="8 5"/></g>
</svg>
)SVG";
        write_text("os_arch.svg", svg);
        out << "Wrote os_arch.svg (real SVG image)\n";
    }

    void replicate(std::ostream& out) const {
        write_text("app/replica.cpp", source_);
        out << "Copied exact source to app/replica.cpp (no automatic self-rebuild)\n";
    }

    // A narrow, real capability extension: synthesize a 0.5-second PCM waveform.
    void audio(std::ostream& out) const {
        if (!audio_enabled_) throw std::logic_error("Enable via self-evolve first");
        constexpr std::uint32_t rate = 8000;
        constexpr std::uint32_t samples = 4000;
        constexpr std::uint32_t bytes = samples * 2;
        fs::create_directories("app");
        std::ofstream f("app/tone.wav", std::ios::binary | std::ios::trunc);
        if (!f) throw std::runtime_error("Cannot write app/tone.wav");
        auto u16 = [&f](std::uint16_t n) {
            const char b[2] = {static_cast<char>(n & 255), static_cast<char>((n >> 8) & 255)};
            f.write(b, 2);
        };
        auto u32 = [&f](std::uint32_t n) {
            const char b[4] = {static_cast<char>(n & 255), static_cast<char>((n >> 8) & 255),
                               static_cast<char>((n >> 16) & 255), static_cast<char>((n >> 24) & 255)};
            f.write(b, 4);
        };
        f.write("RIFF", 4); u32(36 + bytes); f.write("WAVEfmt ", 8);
        u32(16); u16(1); u16(1); u32(rate); u32(rate * 2); u16(2); u16(16);
        f.write("data", 4); u32(bytes);
        for (std::uint32_t i = 0; i < samples; ++i) {
            constexpr double two_pi = 6.28318530717958647692;
            const double value = 0.25 * std::sin(two_pi * 440.0 * i / rate);
            const auto pcm = static_cast<std::int16_t>(std::lround(value * 32767.0));
            u16(static_cast<std::uint16_t>(pcm));
        }
        if (!f) throw std::runtime_error("WAV write failed");
        out << "Wrote app/tone.wav (8kHz mono PCM; synthesized, not learned)\n";
    }

    // Explicit shell commands only; does NOT rewrite, compile, or execute its own kernel.
    bool dispatch(const std::string& line, std::ostream& out = std::cout) {
        if (line == "exit" || line == "quit") return false;
        if (line == "ls") cube_fs_.ls(out);
        else if (line == "stats") {
            out << "generations=" << generations_ << " queued=" << scheduler_.size()
                << " validation_mse=" << validation_loss() << " boot_ms=" << boot_ms_ << '\n';
        } else if (line == "optimize") {
            for (int i = 0; i < 5; ++i) auto_optimize(out);
        } else if (line == "step") auto_optimize(out);
        else if (line == "doc") documentation(out);
        else if (line == "image") image(out);
        else if (line == "replicate") replicate(out);
        else if (line == "self-evolve") {
            audio_enabled_ = true;
            out << "Registered built-in audio waveform producer (not self-modifying AI)\n";
            auto_optimize(out);
        } else if (line == "audio") audio(out);
        else if (line == "debug") debug_app(out);
        else if (line.rfind("generate app ", 0) == 0)
            generate_app(line.substr(13), out);
        else if (line.rfind("schedule ", 0) == 0) {
            std::istringstream in(line.substr(9));
            double x, y, z;
            std::string extra;
            if (!(in >> x >> y >> z) || (in >> extra))
                throw std::invalid_argument("Usage: schedule <x> <y> <z>");
            const auto id = scheduler_.push({x, y, z});
            out << "Queued task " << id << '\n';
        } else if (line == "run") {
            const auto task = scheduler_.pop();
            if (!task) out << "Queue empty\n";
            else out << "Dispatched task " << task->id << " with latent ["
                     << task->latent[0] << ", " << task->latent[1] << ", "
                     << task->latent[2] << "]\n";
        } else if (line == "save") {
            autoencoder_.save("app/autoencoder.ann17");
            out << "Saved app/autoencoder.ann17\n";
        } else if (line == "help") {
            out << "ls | stats | step | optimize | generate app hello|xor | debug | "
                << "doc | image | replicate | self-evolve | audio | "
                << "schedule x y z | run | save | exit\n";
        } else if (!line.empty()) {
            out << "Unknown command; type help\n";
        }
        return true;
    }

    void shell(std::istream& in = std::cin, std::ostream& out = std::cout) {
        out << "Type help for commands.\n";
        std::string cmd;
        while (out << "ann-os:" << generations_ << "> " && std::getline(in, cmd)) {
            try { if (!dispatch(cmd, out)) break; }
            catch (const std::exception& e) { out << "ERROR: " << e.what() << '\n'; }
        }
    }
};

#ifndef ANN_OS_NO_MAIN
int main(int argc, char** argv) {
    try {
        fs::path self_path = "ann_os.cpp";
        bool batch = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--self" && i + 1 < argc) self_path = argv[++i];
            else if (arg == "--batch") batch = true;
            else if (arg == "--help") {
                std::cout << "Usage: ann_os [--self source.cpp] [--batch]\n";
                return 0;
            } else throw std::invalid_argument("Unexpected argument: " + arg);
        }
        ANN_OS os(self_path);
        os.boot();
        os.auto_optimize();
        if (batch) {
            os.dispatch("stats");
            return 0;
        }
        os.shell();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << '\n';
        return 1;
    }
}
#endif
