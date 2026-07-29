#pragma once

#include "jarvisx/processor.hpp"

namespace jarvisx {
class Rng {
public:
    explicit Rng(std::uint64_t seed) : state_(seed ? seed : 1ULL) {}
    std::uint64_t next() noexcept {
        state_ ^= state_ << 13U;
        state_ ^= state_ >> 7U;
        state_ ^= state_ << 17U;
        return state_;
    }
    int step(int radius) noexcept {
        return static_cast<int>(next() % (2ULL * radius + 1ULL)) - radius;
    }
private:
    std::uint64_t state_;
};

Genome mutate(const Genome& parent, std::uint64_t generation,
              std::size_t candidate, const Evaluation& telemetry) {
    Genome child = parent;
    child.generation = generation;
    child.seed = mix64(parent.seed ^ generation ^
                       (candidate * 0xD1B54A32D192ED03ULL));
    Rng rng(child.seed);

    switch (rng.next() % 9ULL) {
    case 0: child.feature_dim = std::size_t(std::max<long long>(
                32, static_cast<long long>(child.feature_dim) + 16LL * rng.step(2))); break;
    case 1: child.latent_dim = std::size_t(std::max<long long>(
                8, static_cast<long long>(child.latent_dim) + 8LL * rng.step(2))); break;
    case 2: child.iterations = std::uint16_t(int(child.iterations) + rng.step(4)); break;
    case 3: child.diffusion_radius =
                std::uint16_t(int(child.diffusion_radius) + rng.step(2)); break;
    case 4: child.learning_units =
                std::uint16_t(int(child.learning_units) + 10 * rng.step(3)); break;
    case 5: child.omega_units =
                std::uint16_t(int(child.omega_units) + 8 * rng.step(3)); break;
    case 6: child.max_mse_units =
                std::uint16_t(int(child.max_mse_units) + 200 * rng.step(3)); break;
    case 7: child.max_energy_units =
                std::uint16_t(int(child.max_energy_units) + 250 * rng.step(2)); break;
    default: child.min_coherence_units =
                std::uint16_t(int(child.min_coherence_units) + 100 * rng.step(3)); break;
    }

    if (telemetry.valid && telemetry.metrics.mse > 0.05F) {
        child.feature_dim += 16;
        child.learning_units = std::uint16_t(child.learning_units + 4);
    }
    if (telemetry.valid && telemetry.metrics.coherence < 0.35F) {
        child.omega_units = std::uint16_t(child.omega_units + 3);
        child.min_coherence_units = std::uint16_t(
            std::max(0, int(child.min_coherence_units) - 50));
    }
    child.clamp();
    return child;
}

std::string serialize(const Genome& g) {
    std::ostringstream out;
    out << "version=" << g.version << '\n'
        << "generation=" << g.generation << '\n'
        << "feature_dim=" << g.feature_dim << '\n'
        << "latent_dim=" << g.latent_dim << '\n'
        << "iterations=" << g.iterations << '\n'
        << "diffusion_radius=" << g.diffusion_radius << '\n'
        << "learning_units=" << g.learning_units << '\n'
        << "omega_units=" << g.omega_units << '\n'
        << "max_mse_units=" << g.max_mse_units << '\n'
        << "max_energy_units=" << g.max_energy_units << '\n'
        << "min_coherence_units=" << g.min_coherence_units << '\n'
        << "seed=" << g.seed << '\n'
        << "fingerprint=" << g.fingerprint() << '\n';
    return out.str();
}

Genome deserialize(const std::string& text) {
    Genome g;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        const auto split = line.find('=');
        if (split == std::string::npos) continue;
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1);
        if (key == "version") g.version = std::stoul(value);
        else if (key == "generation") g.generation = std::stoull(value);
        else if (key == "feature_dim") g.feature_dim = std::stoull(value);
        else if (key == "latent_dim") g.latent_dim = std::stoull(value);
        else if (key == "iterations") g.iterations = std::stoul(value);
        else if (key == "diffusion_radius") g.diffusion_radius = std::stoul(value);
        else if (key == "learning_units") g.learning_units = std::stoul(value);
        else if (key == "omega_units") g.omega_units = std::stoul(value);
        else if (key == "max_mse_units") g.max_mse_units = std::stoul(value);
        else if (key == "max_energy_units") g.max_energy_units = std::stoul(value);
        else if (key == "min_coherence_units") g.min_coherence_units = std::stoul(value);
        else if (key == "seed") g.seed = std::stoull(value);
    }
    if (g.version != 1) throw std::runtime_error("unsupported genome version");
    g.clamp();
    return g;
}

void atomic_write(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    const fs::path temp = path.string() + ".tmp";
    {
        std::ofstream output(temp, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write " + temp.string());
        output << text;
        output.flush();
        if (!output) throw std::runtime_error("cannot flush " + temp.string());
    }
    std::error_code error;
    fs::rename(temp, path, error);
    if (error) {
        fs::remove(path, error);
        error.clear();
        fs::rename(temp, path, error);
    }
    if (error) throw std::runtime_error("cannot commit " + path.string());
}

std::optional<Genome> load_genome(const fs::path& path) {
    std::ifstream input(path);
    if (!input) return std::nullopt;
    std::ostringstream text;
    text << input.rdbuf();
    return deserialize(text.str());
}

void write_rom(const fs::path& path, const std::vector<Instruction>& rom) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot write bytecode ROM");
    for (const Instruction& instruction : rom) {
        for (int byte = 7; byte >= 0; --byte) {
            output.put(static_cast<char>(
                (instruction.word >> unsigned(byte * 8)) & 0xFFULL));
        }
    }
}

struct Options {
    std::uint64_t generations{4};
    std::size_t population{5};
    fs::path state_dir{".jarvisx-runtime"};
    std::string file;
    std::string text;
    double min_improvement{1.0e-5};
    bool reset{};
    bool quiet{};
};

Options parse_options(int argc, char** argv) {
    Options options;
    auto value = [&](int& i, const std::string& flag) {
        if (i + 1 >= argc) throw std::invalid_argument("missing value after " + flag);
        return std::string(argv[++i]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--generations") options.generations =
            std::max<std::uint64_t>(1, std::stoull(value(i, arg)));
        else if (arg == "--population") options.population =
            std::max<std::size_t>(2, std::min<std::size_t>(
                32, std::stoull(value(i, arg))));
        else if (arg == "--state-dir") options.state_dir = value(i, arg);
        else if (arg == "--file") options.file = value(i, arg);
        else if (arg == "--text") options.text = value(i, arg);
        else if (arg == "--min-improvement") options.min_improvement =
            std::max(0.0, std::stod(value(i, arg)));
        else if (arg == "--reset") options.reset = true;
        else if (arg == "--quiet") options.quiet = true;
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: jarvisx-runtime [options]\n"
                << "  --generations N       evolution generations\n"
                << "  --population N        candidates per generation (2..32)\n"
                << "  --state-dir PATH      checkpoint directory\n"
                << "  --file PATH           external multimodal/binary input\n"
                << "  --text TEXT           text input\n"
                << "  --min-improvement X   minimum fitness delta\n"
                << "  --reset               discard prior checkpoint\n"
                << "  --quiet               hide candidate rows\n"
                << "\nDefault input: this executable's own binary image.\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown option: " + arg);
    }
    return options;
}

enum class MetaOp : std::uint8_t {
    Observe = 1, Spawn = 2, Evaluate = 3, Select = 4,
    Gate = 5, Checkpoint = 6, Loop = 7, Halt = 0xFF
};

class Runtime {
public:
    Runtime(Options options, Packet packet)
        : options_(std::move(options)), packet_(std::move(packet)) {
        fs::create_directories(options_.state_dir);
        genome_path_ = options_.state_dir / "genome.current";
        journal_path_ = options_.state_dir / "evolution.csv";
        rom_path_ = options_.state_dir / "runtime.rom";
        if (!options_.reset) {
            if (const auto loaded = load_genome(genome_path_)) genome_ = *loaded;
        }
        meta_rom_ = {MetaOp::Observe, MetaOp::Spawn, MetaOp::Evaluate,
                     MetaOp::Select, MetaOp::Gate, MetaOp::Checkpoint,
                     MetaOp::Loop, MetaOp::Halt};
    }

    int run() {
        incumbent_ = evaluate(genome_, packet_);
        if (!incumbent_.valid) throw std::runtime_error(
            "incumbent self-evaluation failed: " + incumbent_.error);

        std::cout
            << "Jarvis X Inward Runtime\n"
            << "  lattice: 8192 x 8192 x 8192 virtual cells\n"
            << "  mode: sandboxed bytecode/genome evolution\n"
            << "  starting generation: " << genome_.generation << '\n'
            << "  starting fingerprint: " << genome_.fingerprint() << '\n'
            << "  population: " << options_.population << '\n'
            << "  generations: " << options_.generations << "\n\n";

        while (!halted_) execute(meta_rom_.at(meta_pc_));

        std::cout << "\nFinal genome\n" << serialize(genome_)
                  << "fitness=" << std::setprecision(10) << incumbent_.fitness << '\n'
                  << "mse=" << incumbent_.metrics.mse << '\n'
                  << "coherence=" << incumbent_.metrics.coherence << '\n'
                  << "checkpoint=" << genome_path_.string() << '\n'
                  << "rom=" << rom_path_.string() << '\n'
                  << "journal=" << journal_path_.string() << '\n';
        return 0;
    }

private:
    Options options_;
    Packet packet_;
    Genome genome_;
    Evaluation incumbent_;
    std::vector<Genome> candidates_;
    std::vector<Evaluation> evaluations_;
    Evaluation champion_;
    std::vector<MetaOp> meta_rom_;
    std::size_t meta_pc_{};
    std::uint64_t completed_{};
    bool halted_{};
    bool accepted_{};
    fs::path genome_path_, journal_path_, rom_path_;

    void execute(MetaOp op) {
        switch (op) {
        case MetaOp::Observe:
            std::cout << "[Observe] generation=" << genome_.generation + 1
                      << " fitness=" << std::fixed << std::setprecision(7)
                      << incumbent_.fitness << " mse=" << incumbent_.metrics.mse
                      << " coherence=" << incumbent_.metrics.coherence << '\n';
            ++meta_pc_;
            break;
        case MetaOp::Spawn:
            candidates_.clear();
            {
                Genome anchor = genome_;
                anchor.generation = genome_.generation + 1;
                candidates_.push_back(anchor);
            }
            for (std::size_t i = 1; i < options_.population; ++i) {
                candidates_.push_back(mutate(
                    genome_, genome_.generation + 1, i, incumbent_));
            }
            std::cout << "[Spawn] candidates=" << candidates_.size() << '\n';
            ++meta_pc_;
            break;
        case MetaOp::Evaluate:
            evaluations_.clear();
            for (std::size_t i = 0; i < candidates_.size(); ++i) {
                Evaluation result = evaluate(candidates_[i], packet_);
                if (!options_.quiet) {
                    std::cout << "  candidate=" << i
                              << " fp=" << result.genome.fingerprint()
                              << " valid=" << (result.valid ? "yes" : "no")
                              << " fitness=" << result.fitness
                              << " mse=" << result.metrics.mse
                              << " coherence=" << result.metrics.coherence
                              << " ms=" << result.elapsed_ms << '\n';
                }
                evaluations_.push_back(std::move(result));
            }
            ++meta_pc_;
            break;
        case MetaOp::Select:
            champion_ = *std::max_element(
                evaluations_.begin(), evaluations_.end(),
                [](const Evaluation& left, const Evaluation& right) {
                    if (left.fitness == right.fitness)
                        return left.elapsed_ms > right.elapsed_ms;
                    return left.fitness < right.fitness;
                });
            std::cout << "[Select] champion=" << champion_.genome.fingerprint()
                      << " fitness=" << champion_.fitness << '\n';
            ++meta_pc_;
            break;
        case MetaOp::Gate: {
            const bool coherent = champion_.valid &&
                champion_.metrics.coherence >=
                champion_.genome.min_coherence_units / 10000.0F;
            const bool improved = champion_.fitness >
                incumbent_.fitness + options_.min_improvement;
            accepted_ = coherent && improved;
            if (accepted_) {
                genome_ = champion_.genome;
                incumbent_ = champion_;
                std::cout << "[Lambda] COMMIT\n";
            } else {
                ++genome_.generation;
                incumbent_.genome = genome_;
                std::cout << "[Lambda] ROLLBACK\n";
            }
            ++meta_pc_;
            break;
        }
        case MetaOp::Checkpoint:
            atomic_write(genome_path_, serialize(genome_));
            write_rom(rom_path_, synthesize_rom(genome_));
            append_journal();
            std::cout << "[Checkpoint] generation=" << genome_.generation
                      << " status=" << (accepted_ ? "accepted" : "rolled-back")
                      << '\n';
            ++meta_pc_;
            break;
        case MetaOp::Loop:
            ++completed_;
            meta_pc_ = completed_ < options_.generations ? 0 : meta_pc_ + 1;
            break;
        case MetaOp::Halt:
            halted_ = true;
            break;
        }
    }

    void append_journal() {
        const bool exists = fs::exists(journal_path_);
        std::ofstream output(journal_path_, std::ios::app);
        if (!output) throw std::runtime_error("cannot append journal");
        if (!exists) {
            output << "generation,status,fingerprint,fitness,mse,coherence,"
                      "energy,features,latent,iterations,radius,learning,omega,"
                      "tiles,memory_bytes,elapsed_ms\n";
        }
        output << genome_.generation << ','
               << (accepted_ ? "accepted" : "rolled-back") << ','
               << genome_.fingerprint() << ','
               << std::setprecision(12) << incumbent_.fitness << ','
               << incumbent_.metrics.mse << ','
               << incumbent_.metrics.coherence << ','
               << incumbent_.metrics.energy << ','
               << genome_.feature_dim << ',' << genome_.latent_dim << ','
               << genome_.iterations << ',' << genome_.diffusion_radius << ','
               << genome_.learning_units << ',' << genome_.omega_units << ','
               << incumbent_.tiles << ',' << incumbent_.memory_bytes << ','
               << incumbent_.elapsed_ms << '\n';
    }
};

Packet resolve_packet(int argc, char** argv, const Options& options) {
    if (!options.file.empty()) {
        auto packet = file_packet(options.file);
        if (!packet) throw std::runtime_error("cannot read " + options.file);
        std::cout << "[Input] external packet: " << options.file << '\n';
        return *packet;
    }
    if (!options.text.empty()) {
        std::cout << "[Input] text packet\n";
        return text_packet(options.text);
    }
    if (argc > 0 && argv[0]) {
        auto packet = file_packet(argv[0]);
        if (packet && !packet->bytes.empty()) {
            std::cout << "[Input] inward executable image: " << argv[0]
                      << " (" << packet->bytes.size() << " bytes)\n";
            return *packet;
        }
    }
    std::cout << "[Input] fallback inward identity packet\n";
    return text_packet("Jarvis X reconstructs its own runtime state.");
}

} // namespace jarvisx
