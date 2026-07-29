#pragma once

#include "jarvisx/processor.hpp"

#include <cstdlib>

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
        if (radius <= 0) return 0;
        const auto span = 2ULL * static_cast<std::uint64_t>(radius) + 1ULL;
        return static_cast<int>(next() % span) - radius;
    }

private:
    std::uint64_t state_;
};

std::uint16_t bounded_u16(std::uint16_t value, int delta,
                          int low, int high) noexcept {
    const int adjusted = std::clamp(static_cast<int>(value) + delta, low, high);
    return static_cast<std::uint16_t>(adjusted);
}

std::size_t bounded_size(std::size_t value, long long delta,
                         std::size_t low, std::size_t high) noexcept {
    const auto signed_value = static_cast<long long>(value);
    const auto signed_low = static_cast<long long>(low);
    const auto signed_high = static_cast<long long>(high);
    return static_cast<std::size_t>(
        std::clamp(signed_value + delta, signed_low, signed_high));
}

Genome mutate(const Genome& parent, std::uint64_t generation,
              std::size_t candidate, const Evaluation& telemetry) {
    Genome child = parent;
    child.generation = generation;
    child.seed = mix64(
        parent.seed ^ generation ^
        (static_cast<std::uint64_t>(candidate) * 0xD1B54A32D192ED03ULL));
    Rng rng(child.seed);

    switch (rng.next() % 9ULL) {
    case 0:
        child.feature_dim = bounded_size(
            child.feature_dim, 16LL * rng.step(2), 32U, 512U);
        break;
    case 1:
        child.latent_dim = bounded_size(
            child.latent_dim, 8LL * rng.step(2), 8U, 256U);
        break;
    case 2:
        child.iterations = bounded_u16(
            child.iterations, rng.step(4), 2, 128);
        break;
    case 3:
        child.diffusion_radius = bounded_u16(
            child.diffusion_radius, rng.step(2), 1, 32);
        break;
    case 4:
        child.learning_units = bounded_u16(
            child.learning_units, 10 * rng.step(3), 1, 400);
        break;
    case 5:
        child.omega_units = bounded_u16(
            child.omega_units, 8 * rng.step(3), 1, 300);
        break;
    case 6:
        child.max_mse_units = bounded_u16(
            child.max_mse_units, 200 * rng.step(3), 100, 10000);
        break;
    case 7:
        child.max_energy_units = bounded_u16(
            child.max_energy_units, 250 * rng.step(2), 1000, 10000);
        break;
    default:
        child.min_coherence_units = bounded_u16(
            child.min_coherence_units, 100 * rng.step(3), 0, 9000);
        break;
    }

    if (telemetry.valid && telemetry.metrics.mse > 0.05F) {
        child.feature_dim = bounded_size(child.feature_dim, 16, 32U, 512U);
        child.learning_units = bounded_u16(child.learning_units, 4, 1, 400);
    }
    if (telemetry.valid && telemetry.metrics.coherence < 0.35F) {
        child.omega_units = bounded_u16(child.omega_units, 3, 1, 300);
        child.min_coherence_units = bounded_u16(
            child.min_coherence_units, -50, 0, 9000);
    }
    child.clamp();
    return child;
}

std::string serialize(const Genome& genome) {
    std::ostringstream output;
    output << "version=" << genome.version << '\n'
           << "generation=" << genome.generation << '\n'
           << "feature_dim=" << genome.feature_dim << '\n'
           << "latent_dim=" << genome.latent_dim << '\n'
           << "iterations=" << genome.iterations << '\n'
           << "diffusion_radius=" << genome.diffusion_radius << '\n'
           << "learning_units=" << genome.learning_units << '\n'
           << "omega_units=" << genome.omega_units << '\n'
           << "max_mse_units=" << genome.max_mse_units << '\n'
           << "max_energy_units=" << genome.max_energy_units << '\n'
           << "min_coherence_units=" << genome.min_coherence_units << '\n'
           << "seed=" << genome.seed << '\n'
           << "fingerprint=" << genome.fingerprint() << '\n';
    return output.str();
}

std::uint64_t parse_u64(const std::string& value, const std::string& key) {
    std::size_t consumed = 0;
    const auto parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size()) {
        throw std::runtime_error("invalid numeric genome field: " + key);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::uint16_t parse_u16(const std::string& value, const std::string& key) {
    const std::uint64_t parsed = parse_u64(value, key);
    if (parsed > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("genome field outside uint16 range: " + key);
    }
    return static_cast<std::uint16_t>(parsed);
}

Genome deserialize(const std::string& text) {
    constexpr std::uint32_t kRequiredFields = (1U << 12U) - 1U;
    Genome genome;
    std::uint32_t seen = 0;
    std::optional<std::string> stored_fingerprint;
    std::istringstream input(text);
    std::string line;

    while (std::getline(input, line)) {
        const auto split = line.find('=');
        if (split == std::string::npos) continue;
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1);

        if (key == "version") {
            genome.version = static_cast<std::uint32_t>(parse_u64(value, key));
            seen |= 1U << 0U;
        } else if (key == "generation") {
            genome.generation = parse_u64(value, key);
            seen |= 1U << 1U;
        } else if (key == "feature_dim") {
            const std::uint64_t parsed = parse_u64(value, key);
            if (parsed > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("genome field outside size_t range: " + key);
            }
            genome.feature_dim = static_cast<std::size_t>(parsed);
            seen |= 1U << 2U;
        } else if (key == "latent_dim") {
            const std::uint64_t parsed = parse_u64(value, key);
            if (parsed > std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("genome field outside size_t range: " + key);
            }
            genome.latent_dim = static_cast<std::size_t>(parsed);
            seen |= 1U << 3U;
        } else if (key == "iterations") {
            genome.iterations = parse_u16(value, key);
            seen |= 1U << 4U;
        } else if (key == "diffusion_radius") {
            genome.diffusion_radius = parse_u16(value, key);
            seen |= 1U << 5U;
        } else if (key == "learning_units") {
            genome.learning_units = parse_u16(value, key);
            seen |= 1U << 6U;
        } else if (key == "omega_units") {
            genome.omega_units = parse_u16(value, key);
            seen |= 1U << 7U;
        } else if (key == "max_mse_units") {
            genome.max_mse_units = parse_u16(value, key);
            seen |= 1U << 8U;
        } else if (key == "max_energy_units") {
            genome.max_energy_units = parse_u16(value, key);
            seen |= 1U << 9U;
        } else if (key == "min_coherence_units") {
            genome.min_coherence_units = parse_u16(value, key);
            seen |= 1U << 10U;
        } else if (key == "seed") {
            genome.seed = parse_u64(value, key);
            seen |= 1U << 11U;
        } else if (key == "fingerprint") {
            stored_fingerprint = value;
        }
    }

    if (seen != kRequiredFields) {
        throw std::runtime_error("genome checkpoint is missing required fields");
    }
    if (genome.version != 1U) {
        throw std::runtime_error("unsupported genome version");
    }

    genome.clamp();
    if (!stored_fingerprint || *stored_fingerprint != genome.fingerprint()) {
        throw std::runtime_error("genome checkpoint fingerprint mismatch");
    }
    return genome;
}

void atomic_write(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot write " + temporary.string());
        }
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) {
            throw std::runtime_error("cannot flush " + temporary.string());
        }
    }

    std::error_code error;
    fs::rename(temporary, path, error);
    if (error) {
        fs::remove(path, error);
        error.clear();
        fs::rename(temporary, path, error);
    }
    if (error) {
        throw std::runtime_error("cannot commit " + path.string());
    }
}

std::optional<Genome> load_genome(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    std::ostringstream text;
    text << input.rdbuf();
    if (!input.eof() && input.fail()) {
        throw std::runtime_error("cannot read genome checkpoint");
    }
    return deserialize(text.str());
}

void write_rom(const fs::path& path, const std::vector<Instruction>& rom) {
    std::string bytes;
    bytes.reserve(rom.size() * sizeof(std::uint64_t));
    for (const Instruction& instruction : rom) {
        for (int byte = 7; byte >= 0; --byte) {
            const auto shift = static_cast<unsigned>(byte * 8);
            bytes.push_back(static_cast<char>((instruction.word >> shift) & 0xFFULL));
        }
    }
    atomic_write(path, bytes);
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
    auto value = [&](int& index, const std::string& flag) {
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value after " + flag);
        }
        return std::string(argv[++index]);
    };

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--generations") {
            options.generations = std::max<std::uint64_t>(
                1, parse_u64(value(index, argument), argument));
        } else if (argument == "--population") {
            const auto parsed = parse_u64(value(index, argument), argument);
            options.population = static_cast<std::size_t>(
                std::clamp<std::uint64_t>(parsed, 2, 32));
        } else if (argument == "--state-dir") {
            options.state_dir = value(index, argument);
        } else if (argument == "--file") {
            options.file = value(index, argument);
        } else if (argument == "--text") {
            options.text = value(index, argument);
        } else if (argument == "--min-improvement") {
            std::size_t consumed = 0;
            const std::string raw = value(index, argument);
            const double parsed = std::stod(raw, &consumed);
            if (consumed != raw.size() || !std::isfinite(parsed)) {
                throw std::invalid_argument("invalid minimum improvement");
            }
            options.min_improvement = std::max(0.0, parsed);
        } else if (argument == "--reset") {
            options.reset = true;
        } else if (argument == "--quiet") {
            options.quiet = true;
        } else if (argument == "--help" || argument == "-h") {
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
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

enum class MetaOp : std::uint8_t {
    Observe = 1,
    Spawn = 2,
    Evaluate = 3,
    Select = 4,
    Gate = 5,
    Checkpoint = 6,
    Loop = 7,
    Halt = 0xFF
};

bool evaluation_less(const Evaluation& left, const Evaluation& right) noexcept {
    return left.fitness < right.fitness;
}

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
        meta_rom_ = {
            MetaOp::Observe,
            MetaOp::Spawn,
            MetaOp::Evaluate,
            MetaOp::Select,
            MetaOp::Gate,
            MetaOp::Checkpoint,
            MetaOp::Loop,
            MetaOp::Halt
        };
    }

    int run() {
        incumbent_ = evaluate(genome_, packet_);
        if (!incumbent_.valid) {
            throw std::runtime_error(
                "incumbent self-evaluation failed: " + incumbent_.error);
        }

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
    fs::path genome_path_;
    fs::path journal_path_;
    fs::path rom_path_;

    void execute(MetaOp operation) {
        switch (operation) {
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
            for (std::size_t index = 1; index < options_.population; ++index) {
                candidates_.push_back(mutate(
                    genome_, genome_.generation + 1, index, incumbent_));
            }
            std::cout << "[Spawn] candidates=" << candidates_.size() << '\n';
            ++meta_pc_;
            break;
        case MetaOp::Evaluate:
            evaluations_.clear();
            for (std::size_t index = 0; index < candidates_.size(); ++index) {
                Evaluation result = evaluate(candidates_[index], packet_);
                if (!options_.quiet) {
                    std::cout << "  candidate=" << index
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
            if (evaluations_.empty()) {
                throw std::runtime_error("cannot select from an empty population");
            }
            champion_ = *std::max_element(
                evaluations_.begin(), evaluations_.end(), evaluation_less);
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
        default:
            throw std::runtime_error("invalid meta-runtime opcode");
        }
    }

    void append_journal() {
        std::string existing;
        if (fs::exists(journal_path_)) {
            std::ifstream input(journal_path_, std::ios::binary);
            if (!input) throw std::runtime_error("cannot read evolution journal");
            std::ostringstream buffer;
            buffer << input.rdbuf();
            if (!input.eof() && input.fail()) {
                throw std::runtime_error("cannot read evolution journal");
            }
            existing = buffer.str();
        }

        std::ostringstream row;
        if (existing.empty()) {
            row << "generation,status,fingerprint,fitness,mse,coherence,"
                   "energy,features,latent,iterations,radius,learning,omega,"
                   "tiles,memory_bytes,elapsed_ms\n";
        }
        row << genome_.generation << ','
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

        atomic_write(journal_path_, existing + row.str());
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
